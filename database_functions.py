from embeddings_data_models import Base, TextEmbedding, DocumentEmbedding, Document, AudioTranscript
from logger_config import setup_logger
import traceback
import asyncio
import random
from sqlalchemy import select, update, UniqueConstraint, exists
from sqlalchemy import text as sql_text
from sqlalchemy.exc import SQLAlchemyError, OperationalError, IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from decouple import config
from datetime import datetime, timedelta

logger = setup_logger()
db_writer = None
DATABASE_URL = "sqlite+aiosqlite:///swiss_army_llama.sqlite"
MAX_RETRIES = config("MAX_RETRIES", default=3, cast=int)
DB_WRITE_BATCH_SIZE = config("DB_WRITE_BATCH_SIZE", default=25, cast=int) 
RETRY_DELAY_BASE_SECONDS = config("RETRY_DELAY_BASE_SECONDS", default=1, cast=int)
JITTER_FACTOR = config("JITTER_FACTOR", default=0.1, cast=float)
TIME_IN_DAYS_BEFORE_RECORDS_ARE_PURGED = config("TIME_IN_DAYS_BEFORE_RECORDS_ARE_PURGED", default=2, cast=int)

engine = create_async_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush=False)

async def consolidate_wal_data():
    consolidate_command = "PRAGMA wal_checkpoint(FULL);"
    try:
        async with engine.begin() as conn:
            result = await conn.execute(sql_text(consolidate_command))
            result_fetch = result.fetchone()
            return result_fetch
    except Exception as e:
        logger.error(f"Error during WAL consolidation: {e}")
        return None

class DatabaseWriter:
    def __init__(self, queue):
        self.queue = queue
        self.processing_hashes = set()

    def _get_hash_from_operation(self, operation):
        if isinstance(operation, TextEmbedding):
            return f"{operation.embedding_hash}"
        elif isinstance(operation, DocumentEmbedding):
            return f"{operation.document_embedding_results_json_compressed_binary}"
        elif isinstance(operation, Document):
            return operation.document_hash
        elif isinstance(operation, AudioTranscript):
            return operation.audio_file_hash
        return None

    async def initialize_processing_hashes(self, chunk_size=1000):
        start_time = datetime.utcnow()
        async with AsyncSessionLocal() as session:
            queries = [
                (select(TextEmbedding.embedding_hash), TextEmbedding),
                (select(DocumentEmbedding.document_embedding_results_json_compressed_binary), DocumentEmbedding),
                (select(Document.document_hash), Document),
                (select(AudioTranscript.audio_file_hash), AudioTranscript)
            ]
            for query, model_class in queries:
                offset = 0
                while True:
                    result = await session.execute(query.limit(chunk_size).offset(offset))
                    rows = result.fetchall()
                    if not rows:
                        break
                    for row in rows:
                        if model_class == TextEmbedding:
                            hash_with_model = row[0]
                        elif model_class == DocumentEmbedding:
                            hash_with_model = row[0]
                        elif model_class == Document:
                            hash_with_model = row[0]
                        elif model_class == AudioTranscript:
                            hash_with_model = row[0]
                        self.processing_hashes.add(hash_with_model)
                    offset += chunk_size
        end_time = datetime.utcnow()
        total_time = (end_time - start_time).total_seconds()
        if len(self.processing_hashes) > 0:
            logger.info(f"Finished initializing set of input hash/llm_model_name combinations that are either currently being processed or have already been processed. Set size: {len(self.processing_hashes)}; Took {total_time} seconds, for an average of {total_time / len(self.processing_hashes)} seconds per hash.")

    async def _record_exists(self, session, operation):
        model_class = type(operation)
        if model_class == TextEmbedding:
            return await session.execute(select(exists().where(TextEmbedding.embedding_hash == operation.embedding_hash)))
        elif model_class == DocumentEmbedding:
            return await session.execute(select(exists().where(DocumentEmbedding.document_embedding_results_json_compressed_binary == operation.document_embedding_results_json_compressed_binary)))
        elif model_class == Document:
            return await session.execute(select(exists().where(Document.document_hash == operation.document_hash)))
        elif model_class == AudioTranscript:
            return await session.execute(select(exists().where(AudioTranscript.audio_file_hash == operation.audio_file_hash)))
        return None

    async def dedicated_db_writer(self):
        while True:
            write_operations_batch = await self.queue.get()
            async with AsyncSessionLocal() as session:
                filtered_operations = []
                try:
                    if write_operations_batch:
                        for write_operation in write_operations_batch:
                            existing_record = await self._record_exists(session, write_operation)
                            if not existing_record.scalar():
                                filtered_operations.append(write_operation)
                                hash_value = self._get_hash_from_operation(write_operation)
                                if hash_value:
                                    self.processing_hashes.add(hash_value)
                            else:
                                await self._update_existing_record(session, write_operation)
                        if filtered_operations:
                            await consolidate_wal_data()  # Consolidate WAL before performing writes
                            session.add_all(filtered_operations)
                            await session.flush()  # Flush to get the IDs
                            await session.commit()
                            for write_operation in filtered_operations:
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

    async def _update_existing_record(self, session, operation):
        model_class = type(operation)
        primary_keys = [key.name for key in model_class.__table__.primary_key]
        unique_constraints = [c for c in model_class.__table__.constraints if isinstance(c, UniqueConstraint)]
        conditions = []
        for constraint in unique_constraints:
            if set(constraint.columns.keys()).issubset(set(operation.__dict__.keys())):
                for col in constraint.columns.keys():
                    conditions.append(getattr(model_class, col) == getattr(operation, col))
                break
        if not conditions:
            for pk in primary_keys:
                conditions.append(getattr(model_class, pk) == getattr(operation, pk))
        values = {col: getattr(operation, col) for col in operation.__dict__.keys() if col in model_class.__table__.columns.keys()}
        stmt = update(model_class).where(*conditions).values(**values)
        await session.execute(stmt)
        await session.commit()

    async def _handle_integrity_error(self, e, write_operation, session):
        unique_constraint_msg = {
            TextEmbedding: "embeddings.embedding_hash",
            DocumentEmbedding: "document_embeddings.document_embedding_results_json_compressed_binary",
            Document: "documents.document_hash",
            AudioTranscript: "audio_transcripts.audio_file_hash"
        }.get(type(write_operation))
        if unique_constraint_msg and unique_constraint_msg in str(e):
            logger.warning(f"Embedding already exists in the database for given input: {e}")
            await self._update_existing_record(session, write_operation)
        else:
            raise        

    async def enqueue_write(self, write_operations):
        write_operations = [op for op in write_operations if self._get_hash_from_operation(op) not in self.processing_hashes]
        if not write_operations:
            return
        for op in write_operations:
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

async def initialize_db(use_verbose = 0):
    logger.info("Initializing database, creating tables, and setting SQLite PRAGMAs...")
    list_of_sqlite_pragma_strings = [
        "PRAGMA journal_mode=WAL;", 
        "PRAGMA synchronous = NORMAL;", 
        "PRAGMA cache_size = -1048576;", 
        "PRAGMA busy_timeout = 2000;", 
        "PRAGMA wal_autocheckpoint = 100;"
    ]
    list_of_sqlite_pragma_justification_strings = [
        "Set SQLite to use Write-Ahead Logging (WAL) mode (from default DELETE mode) so that reads and writes can occur simultaneously",
        "Set synchronous mode to NORMAL (from FULL) so that writes are not blocked by reads",
        "Set cache size to 1GB (from default 2MB) so that more data can be cached in memory and not read from disk; to make this 256MB, set it to -262144 instead",
        "Increase the busy timeout to 2 seconds so that the database waits",
        "Set the WAL autocheckpoint to 100 (from default 1000) so that the WAL file is checkpointed more frequently"
    ]
    assert len(list_of_sqlite_pragma_strings) == len(list_of_sqlite_pragma_justification_strings)
    async with engine.begin() as conn:
        for pragma_string in list_of_sqlite_pragma_strings:
            await conn.execute(sql_text(pragma_string))
            if use_verbose:
                logger.info(f"Executed SQLite PRAGMA: {pragma_string}")
                logger.info(f"Justification: {list_of_sqlite_pragma_justification_strings[list_of_sqlite_pragma_strings.index(pragma_string)]}")
        try:
            await conn.run_sync(Base.metadata.create_all)  # Create tables if they don't exist
        except Exception as e:  # noqa: F841
            pass
    logger.info("Database initialization completed.")

def get_db_writer() -> DatabaseWriter:
    return db_writer  # Return the existing DatabaseWriter instance

def delete_expired_rows(session_factory):
    async def async_delete_expired_rows():
        async with session_factory() as session:
            expiration_time = datetime.utcnow() - timedelta(days=TIME_IN_DAYS_BEFORE_RECORDS_ARE_PURGED)
            models = [TextEmbedding, DocumentEmbedding, Document, AudioTranscript]
            for model in models:
                expired_rows = await session.execute(
                    select(model).where(model.created_at < expiration_time)
                )
                expired_rows = expired_rows.scalars().all()
                for row in expired_rows:
                    await session.delete(row)
            await session.commit()
    return async_delete_expired_rows
