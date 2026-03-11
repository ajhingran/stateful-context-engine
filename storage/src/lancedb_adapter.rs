use std::borrow::Cow;
use std::fmt::Debug;
use std::sync::Arc;
use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, StringArray};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use lancedb;
use lancedb::arrow::arrow_schema::{DataType, Field, Schema, SchemaRef, TimeUnit};
use lancedb::embeddings::{EmbeddingDefinition, EmbeddingFunction};
use lancedb::{Connection, Table, error};

const EMBEDDING_TABLE_NAME: &str = "context_embeddings";
const EMBEDDING_DIM: i32 = 1024;
const EMBEDDER_NAME: &str = "fastembed";

/// Result type for embedding operations returning an Arrow array.
type EmbeddingResult = lancedb::Result<ArrayRef>;

/// Result type for Arrow DataType queries, borrowing from `'a` where possible.
type DataTypeResult<'a> = lancedb::Result<Cow<'a, DataType>>;

/// Local embedding model backed by fastembed's BGE-Large-EN-v1.5.
///
/// Implements [`EmbeddingFunction`] for use with the LanceDB embedding registry.
/// Converts text columns into 1024-dimensional float vectors at insert and query time.
struct Embedder {
    model: TextEmbedding,
}

impl Embedder {
    /// Loads the BGE-Large-EN-v1.5 model locally via fastembed.
    /// Downloads the model on first use if not already cached.
    fn new() -> Result<Self, error::Error> {
        let model: TextEmbedding = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::BGELargeENV15)
        ).map_err(|e| error::Error::Other { message: e.to_string(), source: None })?;
        Ok(Self { model })
    }

    /// Converts a batch of float vectors into a LanceDB-compatible Arrow [`FixedSizeListArray`].
    ///
    /// Flattens `Vec<Vec<f32>>` into a contiguous buffer, then wraps it as a fixed-size list
    /// where each list has [`EMBEDDING_DIM`] elements.
    fn convert_to_arrow(embeddings: Vec<Vec<f32>>) -> EmbeddingResult {
        let floats: Arc<Float32Array> = Arc::new(
            Float32Array::from_iter_values(embeddings.into_iter().flatten())
        );
        let field: Arc<Field> = Arc::new(
            Field::new("item", DataType::Float32, false)
        );

        FixedSizeListArray::try_new(field, EMBEDDING_DIM, floats, None)
            .map(|a| Arc::new(a) as ArrayRef)
            .map_err(|e| error::Error::Other { message: e.to_string(), source: None })
    }
}

impl Debug for Embedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Embedder(BGELargeENV15, dim={})", EMBEDDING_DIM)
    }
}

impl EmbeddingFunction for Embedder {
    fn name(&self) -> &str {
        EMBEDDER_NAME
    }

    /// Source column type — expects UTF-8 text strings.
    fn source_type(&self) -> DataTypeResult<'_> {
        Ok(Cow::Owned(DataType::Utf8))
    }

    /// Output type — a fixed-size list of [`EMBEDDING_DIM`] float32 values per row.
    fn dest_type(&self) -> DataTypeResult<'_> {
        Ok(Cow::Owned(DataType::new_fixed_size_list(DataType::Float32, EMBEDDING_DIM, false)))
    }

    /// Embeds a batch of text values from the `context_text` column at insert time.
    ///
    /// Returns an error if the source array is not UTF-8 or contains null values.
    fn compute_source_embeddings(&self, source: ArrayRef) -> EmbeddingResult {
        match source.data_type() {
            DataType::Utf8 => {
                let string_array = source.as_any().downcast_ref::<StringArray>().unwrap();
                let inputs: Vec<&str> = string_array.iter()
                    .map(|t| t.ok_or_else(|| error::Error::Other {
                        message: "null text value".to_string(), source: None
                    }))
                    .collect::<lancedb::Result<Vec<&str>>>()?;

                let embeddings: Vec<Vec<f32>> = self.model.embed(inputs, None)
                    .map_err(|e| error::Error::Other { message: e.to_string(), source: None })?;

                Embedder::convert_to_arrow(embeddings)
            }
            other => Err(error::Error::Other {
                message: format!("expected Utf8, got {:?}", other),
                source: None,
            }),
        }
    }

    /// Embeds query text at search time using the same model as insertion.
    fn compute_query_embeddings(&self, input: ArrayRef) -> EmbeddingResult {
        self.compute_source_embeddings(input)
    }
}

/// Adapter for storing and retrieving LLM context memory in LanceDB.
///
/// Each row stores a context chunk with:
/// - `id`: unique identifier (UUID string)
/// - `context_text`: detokenized plaintext of the chunk
/// - `tokens`: raw token IDs from llama.cpp for KV cache restoration
/// - `vector`: 1024-dim embedding of `context_text` for similarity search (auto-populated)
/// - `timestamp`: UTC millisecond timestamp for garbage collection
///
/// Embeddings are generated automatically on insert via the [`Embedder`] registered
/// with the LanceDB embedding registry.
pub struct LanceDBAdapter {
    connection: Connection,
    embedding_table_schema: SchemaRef,
}

impl LanceDBAdapter {
    /// Creates a new adapter, connecting to LanceDB at `db_path` and registering
    /// the local fastembed model for automatic embedding on insert.
    pub async fn new(db_path: &str) -> Result<Self, error::Error> {
        let connection = lancedb::connect(db_path).execute().await?;

        connection
            .embedding_registry()
            .register(EMBEDDER_NAME, Arc::new(Embedder::new()?))?;

        Ok(Self {
            connection,
            embedding_table_schema: Schema::new(vec![
                Field::new("id", DataType::Utf8, false),
                Field::new("context_text", DataType::Utf8, false),
                Field::new("tokens", DataType::List(
                    Arc::new(Field::new("item", DataType::Int32, false))
                ), false),
                Field::new("vector", DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, false)),
                    EMBEDDING_DIM,
                ), false),
                Field::new("timestamp", DataType::Timestamp(
                    TimeUnit::Millisecond, Some(Arc::from("UTC"))
                ), false),
            ]).into(),
        })
    }

    /// Opens the context embeddings table, creating it if it does not exist.
    ///
    /// The table is configured with an embedding definition that automatically
    /// populates the `vector` column from `context_text` on insert.
    pub async fn create_embeddings_table_if_not_exists(&self) -> Result<Table, error::Error> {
        match self.connection
            .create_empty_table(EMBEDDING_TABLE_NAME, self.embedding_table_schema.clone())
            .add_embedding(EmbeddingDefinition::new("context_text", EMBEDDER_NAME, Some("vector")))?
            .execute()
            .await
        {
            Ok(table) => Ok(table),
            Err(error::Error::TableAlreadyExists { .. }) => {
                self.connection.open_table(EMBEDDING_TABLE_NAME).execute().await
            }
            Err(e) => Err(e),
        }
    }
}
