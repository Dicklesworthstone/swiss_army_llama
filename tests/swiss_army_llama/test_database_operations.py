import pytest
import asyncio
from datetime import datetime
from sqlalchemy import select
from swiss_army_llama import DatabaseWriter, execute_with_retry, engine, AsyncSessionLocal
from embeddings_data_models import Base, TextEmbedding, DocumentEmbedding, Document, TokenLevelEmbedding, TokenLevelEmbeddingBundle, TokenLevelEmbeddingBundleCombinedFeatureVector, AudioTranscript
from sqlalchemy.exc import OperationalError

@pytest.fixture(scope='module')
async def setup_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()

@pytest.fixture
def db_writer():
    queue = asyncio.Queue()
    return DatabaseWriter(queue)

@pytest.mark.asyncio
@pytest.mark.usefixtures("setup_db")
async def test_enqueue_text_embedding_write(db_writer):
    async with AsyncSessionLocal() as session:
        text_embedding = TextEmbedding(
            text="text",
            llm_model_name="model",
            embedding_json="{}",
            ip_address="127.0.0.1",
            request_time=datetime.now(),
            response_time=datetime.now(),
            total_time=1.0
        )
        await db_writer.enqueue_write([text_embedding])
        await db_writer.dedicated_db_writer()
        result = await execute_with_retry(session, select(TextEmbedding).where(TextEmbedding.text == "text"), OperationalError)
        assert result.scalar_one().text == "text"

@pytest.mark.asyncio
@pytest.mark.usefixtures("setup_db")
async def test_enqueue_document_embedding_write(db_writer):
    async with AsyncSessionLocal() as session:
        doc_embedding = DocumentEmbedding(
            document_hash="doc_hash",
            filename="file",
            mimetype="text",
            file_hash="file_hash",
            llm_model_name="model",
            file_data=b"data",
            document_embedding_results_json={},
            ip_address="127.0.0.1",
            request_time=datetime.now(),
            response_time=datetime.now(),
            total_time=1.0
        )
        await db_writer.enqueue_write([doc_embedding])
        await db_writer.dedicated_db_writer()
        result = await execute_with_retry(session, select(DocumentEmbedding).where(DocumentEmbedding.filename == "file"), OperationalError)
        assert result.scalar_one().filename == "file"

@pytest.mark.asyncio
@pytest.mark.usefixtures("setup_db")
async def test_enqueue_document_write(db_writer):
    async with AsyncSessionLocal() as session:
        document = Document(
            llm_model_name="model",
            document_hash="doc_hash"
        )
        await db_writer.enqueue_write([document])
        await db_writer.dedicated_db_writer()
        result = await execute_with_retry(session, select(Document).where(Document.document_hash == "doc_hash"), OperationalError)
        assert result.scalar_one().document_hash == "doc_hash"

@pytest.mark.asyncio
@pytest.mark.usefixtures("setup_db")
async def test_enqueue_token_level_embedding_write(db_writer):
    async with AsyncSessionLocal() as session:
        token_embedding = TokenLevelEmbedding(
            token="token",
            llm_model_name="model",
            token_level_embedding_json="{}",
            ip_address="127.0.0.1",
            request_time=datetime.now(),
            response_time=datetime.now(),
            total_time=1.0,
            token_level_embedding_bundle_id=1
        )
        await db_writer.enqueue_write([token_embedding])
        await db_writer.dedicated_db_writer()
        result = await execute_with_retry(session, select(TokenLevelEmbedding).where(TokenLevelEmbedding.token == "token"), OperationalError)
        assert result.scalar_one().token == "token"

@pytest.mark.asyncio
@pytest.mark.usefixtures("setup_db")
async def test_enqueue_token_level_embedding_bundle_write(db_writer):
    async with AsyncSessionLocal() as session:
        token_bundle = TokenLevelEmbeddingBundle(
            input_text="input",
            llm_model_name="model",
            token_level_embeddings_bundle_json="{}",
            ip_address="127.0.0.1",
            request_time=datetime.now(),
            response_time=datetime.now(),
            total_time=1.0
        )
        await db_writer.enqueue_write([token_bundle])
        await db_writer.dedicated_db_writer()
        result = await execute_with_retry(session, select(TokenLevelEmbeddingBundle).where(TokenLevelEmbeddingBundle.input_text == "input"), OperationalError)
        assert result.scalar_one().input_text == "input"

@pytest.mark.asyncio
@pytest.mark.usefixtures("setup_db")
async def test_enqueue_token_level_embedding_bundle_combined_feature_vector_write(db_writer):
    async with AsyncSessionLocal() as session:
        feature_vector = TokenLevelEmbeddingBundleCombinedFeatureVector(
            token_level_embedding_bundle_id=1,
            llm_model_name="model",
            combined_feature_vector_json="{}",
            combined_feature_vector_hash="hash"
        )
        await db_writer.enqueue_write([feature_vector])
        await db_writer.dedicated_db_writer()
        result = await execute_with_retry(session, select(TokenLevelEmbeddingBundleCombinedFeatureVector).where(TokenLevelEmbeddingBundleCombinedFeatureVector.combined_feature_vector_hash == "hash"), OperationalError)
        assert result.scalar_one().combined_feature_vector_hash == "hash"

@pytest.mark.asyncio
@pytest.mark.usefixtures("setup_db")
async def test_enqueue_audio_transcript_write(db_writer):
    async with AsyncSessionLocal() as session:
        audio_transcript = AudioTranscript(
            audio_file_hash="audio_hash",
            audio_file_name="audio_name",
            audio_file_size_mb=1.0,
            segments_json={},
            combined_transcript_text="text",
            combined_transcript_text_list_of_metadata_dicts={},
            info_json={},
            ip_address="127.0.0.1",
            request_time=datetime.now(),
            response_time=datetime.now(),
            total_time=1.0
        )
        await db_writer.enqueue_write([audio_transcript])
        await db_writer.dedicated_db_writer()
        result = await execute_with_retry(session, select(AudioTranscript).where(AudioTranscript.audio_file_hash == "audio_hash"), OperationalError)
        assert result.scalar_one().audio_file_hash == "audio_hash"
