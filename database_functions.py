from embeddings_data_models import Base, TextEmbedding, DocumentEmbedding, Document, TokenLevelEmbedding, TokenLevelEmbeddingBundle, TokenLevelEmbeddingBundleCombinedFeatureVector, AudioTranscript
from logger_config import logger
import traceback
import asyncio
import random
from sqlalchemy import select
from sqlalchemy import text as sql_text
from sqlalchemy.exc import SQLAlchemyError, OperationalError, IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from decouple import config

db_writer = None
DATABASE_URL = "sqlite+aiosqlite:///swiss_army_llama.sqlite"
MAX_RETRIES = config("MAX_RETRIES", default=3, cast=int)
DB_WRITE_BATCH_SIZE = config("DB_WRITE_BATCH_SIZE", default=25, cast=int) 
RETRY_DELAY_BASE_SECONDS = config("RETRY_DELAY_BASE_SECONDS", default=1, cast=int)
JITTER_FACTOR = config("JITTER_FACTOR", default=0.1, cast=float)

engine = create_async_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)
class DatabaseWriter:
    def __init__(self, queue):
        self.queue = queue
        self.processing_hashes = set() # Set to store the hashes if everything that is currently being processed in the queue (to avoid duplicates of the same task being added to the queue)

    def _get_hash_from_operation(self, operation):
        attr_name = {
            TextEmbedding: 'text_hash',
            DocumentEmbedding: 'file_hash',
            Document: 'document_hash',
            TokenLevelEmbedding: 'token_hash',
            TokenLevelEmbeddingBundle: 'input_text_hash',
            TokenLevelEmbeddingBundleCombinedFeatureVector: 'combined_feature_vector_hash',
            AudioTranscript: 'audio_file_hash'
        }.get(type(operation))
        hash_value = getattr(operation, attr_name, None)
        llm_model_name = getattr(operation, 'llm_model_name', None)
        return f"{hash_value}_{llm_model_name}" if hash_value and llm_model_name else None

    async def initialize_processing_hashes(self, chunk_size=1000):
        start_time = datetime.utcnow()
        async with AsyncSessionLocal() as session:
            queries = [
                (select(TextEmbedding.text_hash, TextEmbedding.llm_model_name), True),
                (select(DocumentEmbedding.file_hash, DocumentEmbedding.llm_model_name), True),
                (select(Document.document_hash, Document.llm_model_name), True),
                (select(TokenLevelEmbedding.token_hash, TokenLevelEmbedding.llm_model_name), True),
                (select(TokenLevelEmbeddingBundle.input_text_hash, TokenLevelEmbeddingBundle.llm_model_name), True),
                (select(TokenLevelEmbeddingBundleCombinedFeatureVector.combined_feature_vector_hash, TokenLevelEmbeddingBundleCombinedFeatureVector.llm_model_name), True),
                (select(AudioTranscript.audio_file_hash), False)
            ]
            for query, has_llm in queries:
                offset = 0
                while True:
                    result = await session.execute(query.limit(chunk_size).offset(offset))
                    rows = result.fetchall()
                    if not rows:
                        break
                    for row in rows:
                        if has_llm:
                            hash_with_model = f"{row[0]}_{row[1]}"
                        else:
                            hash_with_model = row[0]
                        self.processing_hashes.add(hash_with_model)
                    offset += chunk_size
        end_time = datetime.utcnow()
        total_time = (end_time - start_time).total_seconds()
        if len(self.processing_hashes) > 0:
            logger.info(f"Finished initializing set of input hash/llm_model_name combinations that are either currently being processed or have already been processed. Set size: {len(self.processing_hashes)}; Took {total_time} seconds, for an average of {total_time / len(self.processing_hashes)} seconds per hash.")

    async def _handle_integrity_error(self, e, write_operation, session):
        unique_constraint_msg = {
            TextEmbedding: "token_embeddings.token_hash, token_embeddings.llm_model_name",
            DocumentEmbedding: "document_embeddings.file_hash, document_embeddings.llm_model_name",
            Document: "documents.document_hash, documents.llm_model_name",
            TokenLevelEmbedding: "token_level_embeddings.token_hash, token_level_embeddings.llm_model_name",
            TokenLevelEmbeddingBundle: "token_level_embedding_bundles.input_text_hash, token_level_embedding_bundles.llm_model_name",
            AudioTranscript: "audio_transcripts.audio_file_hash"
        }.get(type(write_operation))
        if unique_constraint_msg and unique_constraint_msg in str(e):
            logger.warning(f"Embedding already exists in the database for given input and llm_model_name: {e}")
            await session.rollback()
        else:
            raise
        
    async def dedicated_db_writer(self):
        while True:
            write_operations_batch = await self.queue.get()
            async with AsyncSessionLocal() as session:
                try:
                    for write_operation in write_operations_batch:
                        session.add(write_operation)
                    await session.flush()  # Flush to get the IDs
                    await session.commit()
                    for write_operation in write_operations_batch:
                        hash_to_remove = self._get_hash_from_operation(write_operation)
                        if hash_to_remove is not None and hash_to_remove in self.processing_hashes:
                            self.processing_hashes.remove(hash_to_remove)
                except IntegrityError as e:
                    await self._handle_integrity_error(e, write_operation, session)
                except SQLAlchemyError as e:
                    logger.error(f"Database error: {e}")
                    await session.rollback()
                except Exception as e:
                    tb = traceback.format_exc()
                    logger.error(f"Unexpected error: {e}\n{tb}")
                    await session.rollback()
                self.queue.task_done()
                    
    async def enqueue_write(self, write_operations):
        write_operations = [op for op in write_operations if self._get_hash_from_operation(op) not in self.processing_hashes]  # Filter out write operations for hashes that are already being processed
        if not write_operations:  # If there are no write operations left after filtering, return early
            return
        for op in write_operations:  # Add the hashes of the write operations to the set
            hash_value = self._get_hash_from_operation(op)
            if hash_value:
                self.processing_hashes.add(hash_value)
        await self.queue.put(write_operations)


async def execute_with_retry(func, *args, **kwargs):
    retries = 0
    while retries < MAX_RETRIES:
        try:
            return await func(*args, **kwargs)
        except OperationalError as e:
            if 'database is locked' in str(e):
                retries += 1
                sleep_time = RETRY_DELAY_BASE_SECONDS * (2 ** retries) + (random.random() * JITTER_FACTOR) # Implementing exponential backoff with jitter
                logger.warning(f"Database is locked. Retrying ({retries}/{MAX_RETRIES})... Waiting for {sleep_time} seconds")
                await asyncio.sleep(sleep_time)
            else:
                raise
    raise OperationalError("Database is locked after multiple retries")

async def initialize_db():
    logger.info("Initializing database, creating tables, and setting SQLite PRAGMAs...")
    list_of_sqlite_pragma_strings = ["PRAGMA journal_mode=WAL;", "PRAGMA synchronous = NORMAL;", "PRAGMA cache_size = -1048576;", "PRAGMA busy_timeout = 2000;", "PRAGMA wal_autocheckpoint = 100;"]
    list_of_sqlite_pragma_justification_strings = ["Set SQLite to use Write-Ahead Logging (WAL) mode (from default DELETE mode) so that reads and writes can occur simultaneously",
                                                "Set synchronous mode to NORMAL (from FULL) so that writes are not blocked by reads",
                                                "Set cache size to 1GB (from default 2MB) so that more data can be cached in memory and not read from disk; to make this 256MB, set it to -262144 instead",
                                                "Increase the busy timeout to 2 seconds so that the database waits",
                                                "Set the WAL autocheckpoint to 100 (from default 1000) so that the WAL file is checkpointed more frequently"]
    assert(len(list_of_sqlite_pragma_strings) == len(list_of_sqlite_pragma_justification_strings))
    async with engine.begin() as conn:
        for pragma_string in list_of_sqlite_pragma_strings:
            await conn.execute(sql_text(pragma_string))
            logger.info(f"Executed SQLite PRAGMA: {pragma_string}")
            logger.info(f"Justification: {list_of_sqlite_pragma_justification_strings[list_of_sqlite_pragma_strings.index(pragma_string)]}")
        try:
            await conn.run_sync(Base.metadata.create_all) # Create tables if they don't exist
        except Exception as e:
            pass
    logger.info("Database initialization completed.")

def get_db_writer() -> DatabaseWriter:
    return db_writer  # Return the existing DatabaseWriter instance
