import pytest
import json
import os
import re
import shutil
import tempfile
from hashlib import sha3_256
from datetime import datetime
from fastapi import Request
from fastapi.datastructures import UploadFile
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from swiss_army_llama import (normalize_logprobs, remove_pagination_breaks, sophisticated_sentence_splitter, get_transcript_from_db, save_transcript_to_db, execute_with_retry, db_writer,
                            merge_transcript_segments_into_combined_text, compute_and_store_transcript_embeddings, compute_transcript_with_whisper_from_audio_func, get_or_compute_transcript)
from embeddings_data_models import  AudioTranscript, AudioTranscriptResponse


DATABASE_URL = "sqlite+aiosqlite:///test_swiss_army_llama.sqlite"
engine = create_engine(DATABASE_URL)
async_engine = create_async_engine(DATABASE_URL)

TestingSessionLocal = sessionmaker(
    bind=engine,
    expire_on_commit=False,
)

# Async Session for testing
AsyncTestingSessionLocal = sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

@pytest.mark.asyncio
async def test_get_and_save_transcript():
    audio_file_hash = "test_audio_file_hash"
    audio_file_name = "test_audio_file_name"
    audio_file_size_mb = 1.0
    transcript_segments = json.dumps({"test": "segment"})
    info = json.dumps({"test": "info"})
    ip_address = "127.0.0.1"
    request_time = datetime.now()
    response_time = datetime.now()
    total_time = 1.0
    combined_transcript_text = "test text"
    combined_transcript_text_list_of_metadata_dicts = json.dumps({"test": "metadata"})
    
    # Save transcript to DB
    await save_transcript_to_db(
        audio_file_hash, audio_file_name, audio_file_size_mb, transcript_segments, info,
        ip_address, request_time, response_time, total_time, combined_transcript_text,
        combined_transcript_text_list_of_metadata_dicts
    )
    await db_writer.dedicated_db_writer()

    # Raw SQL query to validate data using sql_text
    async with AsyncTestingSessionLocal() as session:
        query = sql_text("SELECT audio_file_name FROM audio_transcripts WHERE audio_file_hash=:audio_file_hash")
        result = await session.execute(query, {"audio_file_hash": audio_file_hash})
        row = result.fetchone()
        assert row[0] == audio_file_name
    
    # Get transcript from DB using execute_with_retry
    result = await execute_with_retry(get_transcript_from_db, audio_file_hash)
        
    assert isinstance(result, AudioTranscriptResponse)
    assert result.audio_file_name == audio_file_name
    assert result.audio_file_size_mb == audio_file_size_mb
    
    # Raw SQL query to validate data using sql_text
    async with AsyncTestingSessionLocal() as session:
        query = sql_text("SELECT audio_file_name FROM audio_transcripts WHERE audio_file_hash=:audio_file_hash")
        result = await session.execute(query, {"audio_file_hash": audio_file_hash})
        row = result.fetchone()
        assert row[0] == audio_file_name

@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    # Setup: Create tables
    async with AsyncTestingSessionLocal() as session:
        await session.run_sync(AudioTranscript.metadata.create_all)
    yield
    # Teardown: Drop tables
    async with AsyncTestingSessionLocal() as session:
        await session.run_sync(AudioTranscript.metadata.drop_all)

@pytest.mark.asyncio
async def test_text_related_functions():
    # Testing normalize_logprobs
    assert normalize_logprobs(5, 2, 10) == 0.375

    # Testing remove_pagination_breaks (utilizes 're' module)
    assert remove_pagination_breaks("This is a test-\nexample.") == "This is a testexample."
    
    # Using re for an additional test
    assert (re.match(r'^This is', 'This is a test')) is True

    # Testing sophisticated_sentence_splitter
    assert sophisticated_sentence_splitter("This is a test. And another.") == ["This is a test.", "And another."]


@pytest.mark.asyncio
async def test_merge_transcript_segments_into_combined_text():
    segments = [{"start": 0, "end": 2, "text": "Hi", "avg_logprob": -0.5},
                {"start": 2, "end": 5, "text": "there", "avg_logprob": -0.7}]
    combined_text, metadata_dicts, sentences = merge_transcript_segments_into_combined_text(segments)
    assert combined_text == "Hi there "
    assert metadata_dicts[0]['model_confidence_score'] == 1.0


@pytest.mark.asyncio
async def test_get_or_compute_transcript():
    # Preparing a fake audio file
    audio_content = b"fake_audio_data"
    audio_file = UploadFile("fake_audio.wav", file=tempfile.NamedTemporaryFile(delete=False))
    audio_file.file.write(audio_content)
    audio_file.file.seek(0)
    
    # Hashing the audio content using sha3_256
    audio_hash = sha3_256(audio_content).hexdigest()

    # Simulate a request object
    req = Request({"type": "http", "client": ("127.0.0.1", 12345)}, {})
            
    # Actual function call
    result = await get_or_compute_transcript(audio_file, True, "test_model", req)

    # Validate
    assert isinstance(result, AudioTranscriptResponse)
    assert result.audio_file_name == "fake_audio.wav"

    # Compute the hash and validate
    audio_hash = sha3_256(audio_content).hexdigest()
    assert result.audio_file_hash == audio_hash  # Using the hash here for validation

    # Compute and store transcript embeddings (Mocking the function for test)
    # Here, you can replace 'dummy_transcript' and 'dummy_model' with actual data if available
    await compute_and_store_transcript_embeddings('dummy_transcript', [], 'dummy_model', '127.0.0.1', 'dummy_text', req)

    # Cleanup
    audio_file.file.close()
    shutil.rmtree('generated_transcript_embeddings_zip_files', ignore_errors=True)
    os.remove(audio_file.file.name)
    

@pytest.mark.asyncio
async def test_compute_transcript_with_whisper_from_audio_func():
    audio_file_hash = "test_audio_file_hash"
    audio_file_path = "/path/to/audio/file"
    audio_file_name = "test_audio_file_name.wav"
    audio_file_size_mb = 1.0
    ip_address = "127.0.0.1"

    # Simulate a request object
    req = Request({"type": "http", "client": ("127.0.0.1", 12345)}, {})

    # Calling compute_transcript_with_whisper_from_audio_func
    segment_details, info_dict, combined_transcript_text, combined_transcript_text_list_of_metadata_dicts, request_time, response_time, total_time, download_url = await compute_transcript_with_whisper_from_audio_func(
        audio_file_hash, audio_file_path, audio_file_name, audio_file_size_mb, ip_address, req
    )

    # Validate (since this is a test, you may need to adjust these assertions based on what compute_transcript_with_whisper_from_audio_func actually returns)
    assert segment_details is not None
    assert info_dict is not None
    assert combined_transcript_text is not None
    assert combined_transcript_text_list_of_metadata_dicts is not None
    assert request_time is not None
    assert response_time is not None
    assert total_time is not None    