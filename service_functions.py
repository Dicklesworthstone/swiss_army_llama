from logger_config import setup_logger
import shared_resources
from shared_resources import load_model, token_level_embedding_model_cache, text_completion_model_cache, is_gpu_available
from database_functions import AsyncSessionLocal, DatabaseWriter, execute_with_retry
from misc_utility_functions import clean_filename_for_url_func,  FakeUploadFile, sophisticated_sentence_splitter, merge_transcript_segments_into_combined_text, suppress_stdout_stderr
from embeddings_data_models import TextEmbedding, DocumentEmbedding, Document, TokenLevelEmbedding, TokenLevelEmbeddingBundleCombinedFeatureVector, AudioTranscript
from embeddings_data_models import EmbeddingRequest, TextCompletionRequest
from embeddings_data_models import TextCompletionResponse,  AudioTranscriptResponse
import os
import re
import shutil
import psutil
import glob
import json
import asyncio
import zipfile
import tempfile
import traceback
import time
from datetime import datetime
from hashlib import sha3_256
from urllib.parse import quote
import numpy as np
import pandas as pd
import textract
from sqlalchemy import text as sql_text
from sqlalchemy import select
from fastapi import HTTPException, Request, UploadFile, File
from fastapi.concurrency import run_in_threadpool
from typing import List, Optional, Tuple, Dict, Any
from decouple import config
from faster_whisper import WhisperModel
from llama_cpp import Llama, LlamaGrammar
from mutagen import File as MutagenFile
from magika import Magika
import httpx

logger = setup_logger()
magika = Magika()

SWISS_ARMY_LLAMA_SERVER_LISTEN_PORT = config("SWISS_ARMY_LLAMA_SERVER_LISTEN_PORT", default=8089, cast=int)
DEFAULT_MODEL_NAME = config("DEFAULT_MODEL_NAME", default="openchat_v3.2_super", cast=str) 
LLM_CONTEXT_SIZE_IN_TOKENS = config("LLM_CONTEXT_SIZE_IN_TOKENS", default=512, cast=int)
TEXT_COMPLETION_CONTEXT_SIZE_IN_TOKENS = config("TEXT_COMPLETION_CONTEXT_SIZE_IN_TOKENS", default=4000, cast=int)
DEFAULT_MAX_COMPLETION_TOKENS = config("DEFAULT_MAX_COMPLETION_TOKENS", default=100, cast=int)
DEFAULT_NUMBER_OF_COMPLETIONS_TO_GENERATE = config("DEFAULT_NUMBER_OF_COMPLETIONS_TO_GENERATE", default=4, cast=int)
DEFAULT_COMPLETION_TEMPERATURE = config("DEFAULT_COMPLETION_TEMPERATURE", default=0.7, cast=float)
MINIMUM_STRING_LENGTH_FOR_DOCUMENT_EMBEDDING = config("MINIMUM_STRING_LENGTH_FOR_DOCUMENT_EMBEDDING", default=15, cast=int)
USE_PARALLEL_INFERENCE_QUEUE = config("USE_PARALLEL_INFERENCE_QUEUE", default=False, cast=bool)
MAX_CONCURRENT_PARALLEL_INFERENCE_TASKS = config("MAX_CONCURRENT_PARALLEL_INFERENCE_TASKS", default=10, cast=int)
USE_RAMDISK = config("USE_RAMDISK", default=False, cast=bool)
USE_VERBOSE = config("USE_VERBOSE", default=False, cast=bool)
USE_RESOURCE_MONITORING = config("USE_RESOURCE_MONITORING", default=1, cast=bool)
RAMDISK_PATH = config("RAMDISK_PATH", default="/mnt/ramdisk", cast=str)
BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

    
async def get_transcript_from_db(audio_file_hash: str):
    return await execute_with_retry(_get_transcript_from_db, audio_file_hash)

async def _get_transcript_from_db(audio_file_hash: str) -> Optional[dict]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            sql_text("SELECT * FROM audio_transcripts WHERE audio_file_hash=:audio_file_hash"),
            {"audio_file_hash": audio_file_hash},
        )
        row = result.fetchone()
        if row:
            try:
                segments_json = json.loads(row.segments_json)
                combined_transcript_text_list_of_metadata_dicts = json.loads(row.combined_transcript_text_list_of_metadata_dicts)
                info_json = json.loads(row.info_json)
                if hasattr(info_json, '__dict__'):
                    info_json = vars(info_json)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON Decode Error: {e}")
            if not isinstance(segments_json, list) or not isinstance(combined_transcript_text_list_of_metadata_dicts, list) or not isinstance(info_json, dict):
                logger.error(f"Type of segments_json: {type(segments_json)}, Value: {segments_json}")
                logger.error(f"Type of combined_transcript_text_list_of_metadata_dicts: {type(combined_transcript_text_list_of_metadata_dicts)}, Value: {combined_transcript_text_list_of_metadata_dicts}")
                logger.error(f"Type of info_json: {type(info_json)}, Value: {info_json}")
                raise ValueError("Deserialized JSON does not match the expected format.")
            audio_transcript_response = {
                "id": row.id,
                "audio_file_name": row.audio_file_name,
                "audio_file_size_mb": row.audio_file_size_mb,
                "segments_json": segments_json,
                "combined_transcript_text": row.combined_transcript_text,
                "combined_transcript_text_list_of_metadata_dicts": combined_transcript_text_list_of_metadata_dicts,
                "info_json": info_json,
                "ip_address": row.ip_address,
                "request_time": row.request_time,
                "response_time": row.response_time,
                "total_time": row.total_time,
                "url_to_download_zip_file_of_embeddings": ""
            }
            return AudioTranscriptResponse(**audio_transcript_response)
        return None

async def save_transcript_to_db(audio_file_hash, audio_file_name, audio_file_size_mb, transcript_segments, info, ip_address, request_time, response_time, total_time, combined_transcript_text, combined_transcript_text_list_of_metadata_dicts, corpus_identifier_string):
    existing_transcript = await get_transcript_from_db(audio_file_hash)
    if existing_transcript:
        return existing_transcript
    audio_transcript = AudioTranscript(
        audio_file_hash=audio_file_hash,
        audio_file_name=audio_file_name,
        audio_file_size_mb=audio_file_size_mb,
        segments_json=json.dumps(transcript_segments),
        combined_transcript_text=combined_transcript_text,
        combined_transcript_text_list_of_metadata_dicts=json.dumps(combined_transcript_text_list_of_metadata_dicts),
        info_json=json.dumps(info),
        ip_address=ip_address,
        request_time=request_time,
        response_time=response_time,
        total_time=total_time,
        corpus_identifier_string=corpus_identifier_string
    )
    await shared_resources.db_writer.enqueue_write([audio_transcript])

async def compute_and_store_transcript_embeddings(audio_file_name, list_of_transcript_sentences, llm_model_name, ip_address, combined_transcript_text, req: Request):
    logger.info(f"Now computing embeddings for entire transcript of {audio_file_name}...")
    zip_dir = 'generated_transcript_embeddings_zip_files'
    if not os.path.exists(zip_dir):
        os.makedirs(zip_dir)
    sanitized_file_name = clean_filename_for_url_func(audio_file_name)
    document_name = f"automatic_whisper_transcript_of__{sanitized_file_name}"
    file_hash = sha3_256(combined_transcript_text.encode('utf-8')).hexdigest()
    computed_embeddings = await compute_embeddings_for_document(list_of_transcript_sentences, llm_model_name, ip_address, file_hash)
    zip_file_path = f"{zip_dir}/{quote(document_name)}.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        zipf.writestr("embeddings.txt", json.dumps(computed_embeddings))
    download_url = f"download/{quote(document_name)}.zip"
    full_download_url = f"{req.base_url}{download_url}"
    logger.info(f"Generated download URL for transcript embeddings: {full_download_url}")
    fake_upload_file = FakeUploadFile(filename=document_name, content=combined_transcript_text.encode(), content_type='text/plain')
    logger.info(f"Storing transcript embeddings for {audio_file_name} in the database...")
    await store_document_embeddings_in_db(fake_upload_file, file_hash, combined_transcript_text.encode(), json.dumps(computed_embeddings).encode(), computed_embeddings, llm_model_name, ip_address, datetime.utcnow())
    return full_download_url

async def compute_transcript_with_whisper_from_audio_func(audio_file_hash, audio_file_path, audio_file_name, audio_file_size_mb, ip_address, req: Request, corpus_identifier_string: str, compute_embeddings_for_resulting_transcript_document=True, llm_model_name=DEFAULT_MODEL_NAME):
    model_size = "large-v2"
    logger.info(f"Loading Whisper model {model_size}...")
    num_workers = 1 if psutil.virtual_memory().total < 32 * (1024 ** 3) else min(4, max(1, int((psutil.virtual_memory().total - 32 * (1024 ** 3)) / (4 * (1024 ** 3))))) # Only use more than 1 worker if there is at least 32GB of RAM; then use 1 worker per additional 4GB of RAM up to 4 workers max
    model = await run_in_threadpool(WhisperModel, model_size, device="cpu", compute_type="auto", cpu_threads=os.cpu_count(), num_workers=num_workers)
    request_time = datetime.utcnow()
    logger.info(f"Computing transcript for {audio_file_name} which has a {audio_file_size_mb :.2f}MB file size...")
    segments, info = await run_in_threadpool(model.transcribe, audio_file_path, beam_size=20)
    if not segments:
        logger.warning(f"No segments were returned for file {audio_file_name}.")
        return [], {}, "", [], request_time, datetime.utcnow(), 0, ""    
    segment_details = []
    for idx, segment in enumerate(segments):
        details = {
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text,
            "avg_logprob": round(segment.avg_logprob, 2)
        }
        logger.info(f"Details of transcript segment {idx} from file {audio_file_name}: {details}")
        segment_details.append(details)
    combined_transcript_text, combined_transcript_text_list_of_metadata_dicts, list_of_transcript_sentences = merge_transcript_segments_into_combined_text(segment_details)    
    if compute_embeddings_for_resulting_transcript_document:
        download_url = await compute_and_store_transcript_embeddings(audio_file_name, list_of_transcript_sentences, llm_model_name, ip_address, combined_transcript_text, req, corpus_identifier_string)
    else:
        download_url = ''
    response_time = datetime.utcnow()
    total_time = (response_time - request_time).total_seconds()
    logger.info(f"Transcript computed in {total_time} seconds.")
    await save_transcript_to_db(audio_file_hash, audio_file_name, audio_file_size_mb, segment_details, info, ip_address, request_time, response_time, total_time, combined_transcript_text, combined_transcript_text_list_of_metadata_dicts, corpus_identifier_string)
    info_dict = info._asdict()
    return segment_details, info_dict, combined_transcript_text, combined_transcript_text_list_of_metadata_dicts, request_time, response_time, total_time, download_url
    
async def get_or_compute_transcript(file: UploadFile,
                                    compute_embeddings_for_resulting_transcript_document: bool,
                                    llm_model_name: str,
                                    req: Request = None,
                                    corpus_identifier_string: Optional[str] = None) -> dict:
    request_time = datetime.utcnow()
    ip_address = req.client.host if req else "127.0.0.1"
    file_contents = await file.read()
    audio_file_hash = sha3_256(file_contents).hexdigest()
    file.file.seek(0)  # Reset file pointer after read
    unique_id = f"transcript_{audio_file_hash}_{llm_model_name}"
    lock = await shared_resources.lock_manager.lock(unique_id)    
    if lock.valid:
        try:            
            existing_audio_transcript = await get_transcript_from_db(audio_file_hash)
            if existing_audio_transcript:
                return existing_audio_transcript
            current_position = file.file.tell()
            file.file.seek(0, os.SEEK_END)
            audio_file_size_mb = file.file.tell() / (1024 * 1024)
            file.file.seek(current_position)
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                shutil.copyfileobj(file.file, tmp_file)
                audio_file_name = tmp_file.name

            if corpus_identifier_string is None:
                corpus_identifier_string = audio_file_hash

            segment_details, info, combined_transcript_text, combined_transcript_text_list_of_metadata_dicts, request_time, response_time, total_time, download_url = await compute_transcript_with_whisper_from_audio_func(audio_file_hash, audio_file_name, file.filename, audio_file_size_mb, ip_address, req, corpus_identifier_string, compute_embeddings_for_resulting_transcript_document, llm_model_name)
            audio_transcript_response = {
                "audio_file_hash": audio_file_hash,
                "audio_file_name": file.filename,
                "audio_file_size_mb": audio_file_size_mb,
                "segments_json": segment_details,
                "combined_transcript_text": combined_transcript_text,
                "combined_transcript_text_list_of_metadata_dicts": combined_transcript_text_list_of_metadata_dicts,
                "info_json": info,
                "ip_address": ip_address,
                "request_time": request_time,
                "response_time": response_time,
                "total_time": total_time,
                "url_to_download_zip_file_of_embeddings": download_url if compute_embeddings_for_resulting_transcript_document else "",
                "corpus_identifier_string": corpus_identifier_string
            }
            os.remove(audio_file_name)
            return AudioTranscriptResponse(**audio_transcript_response)
        finally:
            await shared_resources.lock_manager.unlock(lock)    
    else:
        return {"status": "already processing"}               
    
    
# Core embedding functions start here:    


def add_model_url(new_url: str) -> str:
    corrected_url = new_url
    if '/blob/main/' in new_url:
        corrected_url = new_url.replace('/blob/main/', '/resolve/main/')
    json_path = os.path.join(BASE_DIRECTORY, "model_urls.json")
    with open(json_path, "r") as f:
        existing_urls = json.load(f)
    if corrected_url not in existing_urls:
        logger.info(f"Model URL not found in database. Adding {new_url} now...")
        existing_urls.append(corrected_url)
        with open(json_path, "w") as f:
            json.dump(existing_urls, f)
        logger.info(f"Model URL added: {new_url}")
    else:
        logger.info("Model URL already exists.")        
    return corrected_url  

async def get_embedding_from_db(text: str, llm_model_name: str):
    text_hash = sha3_256(text.encode('utf-8')).hexdigest() # Compute the hash
    return await execute_with_retry(_get_embedding_from_db, text_hash, llm_model_name)

async def _get_embedding_from_db(text_hash: str, llm_model_name: str) -> Optional[dict]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            sql_text("SELECT embedding_json FROM embeddings WHERE text_hash=:text_hash AND llm_model_name=:llm_model_name"),
            {"text_hash": text_hash, "llm_model_name": llm_model_name},
        )
        row = result.fetchone()
        if row:
            embedding_json = row[0]
            logger.info(f"Embedding found in database for text hash '{text_hash}' using model '{llm_model_name}'")
            return json.loads(embedding_json)
        return None
    
async def get_or_compute_embedding(request: EmbeddingRequest, req: Request = None, client_ip: str = None, document_file_hash: str = None) -> dict:
        request_time = datetime.utcnow()  # Capture request time as datetime object
        ip_address = client_ip or (req.client.host if req else "localhost") # If client_ip is provided, use it; otherwise, try to get from req; if not available, default to "localhost"
        logger.info(f"Received request for embedding for '{request.text}' using model '{request.llm_model_name}' from IP address '{ip_address}'")
        embedding_list = await get_embedding_from_db(request.text, request.llm_model_name) # Check if embedding exists in the database
        if embedding_list is not None:
            response_time = datetime.utcnow()  # Capture response time as datetime object
            total_time = (response_time - request_time).total_seconds()  # Calculate time taken in seconds
            logger.info(f"Embedding found in database for '{request.text}' using model '{request.llm_model_name}'; returning in {total_time:.4f} seconds")
            return {"embedding": embedding_list}
        model = load_model(request.llm_model_name)
        embedding_list = calculate_sentence_embedding(model, request.text) # Compute the embedding if not in the database
        if embedding_list is None:
            logger.error(f"Could not calculate the embedding for the given text: '{request.text}' using model '{request.llm_model_name}!'")
            raise HTTPException(status_code=400, detail="Could not calculate the embedding for the given text")
        embedding_json = json.dumps(embedding_list) # Serialize the numpy array to JSON and save to the database
        response_time = datetime.utcnow()  # Capture response time as datetime object
        total_time = (response_time - request_time).total_seconds() # Calculate total time using datetime objects
        word_length_of_input_text = len(request.text.split())
        if word_length_of_input_text > 0:
            logger.info(f"Embedding calculated for '{request.text}' using model '{request.llm_model_name}' in {total_time} seconds, or an average of {total_time/word_length_of_input_text :.2f} seconds per word. Now saving to database...")
        await save_embedding_to_db(request.text, request.llm_model_name, embedding_json, ip_address, request_time, response_time, total_time, document_file_hash)
        return {"embedding": embedding_list}

async def save_embedding_to_db(text: str, llm_model_name: str, embedding_json: str, ip_address: str, request_time: datetime, response_time: datetime, total_time: float, document_file_hash: str = None):
    existing_embedding = await get_embedding_from_db(text, llm_model_name) # Check if the embedding already exists
    if existing_embedding is not None:
        return existing_embedding
    return await execute_with_retry(_save_embedding_to_db, text, llm_model_name, embedding_json, ip_address, request_time, response_time, total_time, document_file_hash)

async def _save_embedding_to_db(text: str, llm_model_name: str, embedding_json: str, ip_address: str, request_time: datetime, response_time: datetime, total_time: float, document_file_hash: str = None):
    existing_embedding = await get_embedding_from_db(text, llm_model_name)
    if existing_embedding:
        return existing_embedding
    embedding = TextEmbedding(
        text=text,
        llm_model_name=llm_model_name,
        embedding_json=embedding_json,
        ip_address=ip_address,
        request_time=request_time,
        response_time=response_time,
        total_time=total_time,
        document_file_hash=document_file_hash
    )
    await shared_resources.db_writer.enqueue_write([embedding])  # Enqueue the write operation using the db_writer instance
    
def load_token_level_embedding_model(llm_model_name: str, raise_http_exception: bool = True):
    global USE_VERBOSE
    try:
        if llm_model_name in token_level_embedding_model_cache: # Check if the model is already loaded in the cache
            return token_level_embedding_model_cache[llm_model_name]
        models_dir = os.path.join(RAMDISK_PATH, 'models') if USE_RAMDISK else os.path.join(BASE_DIRECTORY, 'models') # Determine the model directory path
        matching_files = glob.glob(os.path.join(models_dir, f"{llm_model_name}*")) # Search for matching model files
        if not matching_files:
            logger.error(f"No model file found matching: {llm_model_name}")
            raise FileNotFoundError
        matching_files.sort(key=os.path.getmtime, reverse=True) # Sort the files based on modification time (recently modified files first)
        model_file_path = matching_files[0]
        with suppress_stdout_stderr():
            gpu_info = is_gpu_available()
            if gpu_info['gpu_found']:
                model_instance = Llama(model_path=model_file_path, embedding=True, n_ctx=LLM_CONTEXT_SIZE_IN_TOKENS, verbose=USE_VERBOSE, n_gpu_layers=-1) # Load the model with GPU acceleration
            else:
                model_instance = Llama(model_path=model_file_path, embedding=True, n_ctx=LLM_CONTEXT_SIZE_IN_TOKENS, verbose=USE_VERBOSE) # Load the model without GPU acceleration
        token_level_embedding_model_cache[llm_model_name] = model_instance # Cache the loaded model
        return model_instance
    except TypeError as e:
        logger.error(f"TypeError occurred while loading the model: {e}")
        raise
    except Exception as e:
        logger.error(f"Exception occurred while loading the model: {e}")
        traceback.print_exc()
        if raise_http_exception:
            raise HTTPException(status_code=404, detail="Model file not found")
        else:
            raise FileNotFoundError(f"No model file found matching: {llm_model_name}")

async def compute_token_level_embedding_bundle_combined_feature_vector(token_level_embeddings) -> List[float]:
    start_time = datetime.utcnow()
    logger.info("Extracting token-level embeddings from the bundle")
    parsed_df = pd.read_json(token_level_embeddings) # Parse the json_content back to a DataFrame
    token_level_embeddings = list(parsed_df['embedding'])
    embeddings = np.array(token_level_embeddings) # Convert the list of embeddings to a NumPy array
    logger.info(f"Computing column-wise means/mins/maxes/std_devs of the embeddings... (shape: {embeddings.shape})")
    assert(len(embeddings) > 0)
    means = np.mean(embeddings, axis=0)
    mins = np.min(embeddings, axis=0)
    maxes = np.max(embeddings, axis=0)
    stds = np.std(embeddings, axis=0)
    logger.info("Concatenating the computed statistics to form the combined feature vector")
    combined_feature_vector = np.concatenate([means, mins, maxes, stds])
    end_time = datetime.utcnow()
    total_time = (end_time - start_time).total_seconds()
    logger.info(f"Computed the token-level embedding bundle's combined feature vector computed in {total_time: .2f} seconds.")
    return combined_feature_vector.tolist()


async def get_or_compute_token_level_embedding_bundle_combined_feature_vector(token_level_embedding_bundle_id, token_level_embeddings, db_writer: DatabaseWriter) -> List[float]:
    request_time = datetime.utcnow()
    request_time = datetime.utcnow()
    logger.info(f"Checking for existing combined feature vector for token-level embedding bundle ID: {token_level_embedding_bundle_id}")
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(TokenLevelEmbeddingBundleCombinedFeatureVector)
            .filter(TokenLevelEmbeddingBundleCombinedFeatureVector.token_level_embedding_bundle_id == token_level_embedding_bundle_id)
        )
        existing_combined_feature_vector = result.scalar_one_or_none()
        if existing_combined_feature_vector:
            response_time = datetime.utcnow()
            total_time = (response_time - request_time).total_seconds()
            logger.info(f"Found existing combined feature vector for token-level embedding bundle ID: {token_level_embedding_bundle_id}. Returning cached result in {total_time:.2f} seconds.")
            return json.loads(existing_combined_feature_vector.combined_feature_vector_json)  # Parse the JSON string into a list
    logger.info(f"No cached combined feature_vector found for token-level embedding bundle ID: {token_level_embedding_bundle_id}. Computing now...")
    combined_feature_vector = await compute_token_level_embedding_bundle_combined_feature_vector(token_level_embeddings)
    combined_feature_vector_db_object = TokenLevelEmbeddingBundleCombinedFeatureVector(
        token_level_embedding_bundle_id=token_level_embedding_bundle_id,
        combined_feature_vector_json=json.dumps(combined_feature_vector)  # Convert the list to a JSON string
    )
    logger.info(f"Writing combined feature vector for database write for token-level embedding bundle ID: {token_level_embedding_bundle_id} to the database...")
    await db_writer.enqueue_write([combined_feature_vector_db_object])
    return combined_feature_vector
        

async def calculate_token_level_embeddings(text: str, llm_model_name: str, client_ip: str, token_level_embedding_bundle_id: int) -> List[np.array]:
    request_time = datetime.utcnow()
    logger.info(f"Starting token-level embedding calculation for text: '{text}' using model: '{llm_model_name}'")
    logger.info(f"Loading model: '{llm_model_name}'")
    llm = load_token_level_embedding_model(llm_model_name)  # Assuming this method returns an instance of the Llama class
    token_embeddings = []
    tokens = text.split()  # Simple whitespace tokenizer; can be replaced with a more advanced one if needed
    logger.info(f"Tokenized text into {len(tokens)} tokens")
    for idx, token in enumerate(tokens, start=1):
        try:  # Check if the embedding is already available in the database
            existing_embedding = await get_token_level_embedding_from_db(token, llm_model_name)
            if existing_embedding is not None:
                token_embeddings.append(np.array(existing_embedding))
                logger.info(f"Embedding retrieved from database for token '{token}'")
                continue
            logger.info(f"Processing token {idx} of {len(tokens)}: '{token}'")
            token_embedding = llm.embed(token)
            token_embedding_array = np.array(token_embedding)
            token_embeddings.append(token_embedding_array)
            response_time = datetime.utcnow()
            token_level_embedding_json = json.dumps(token_embedding_array.tolist())
            await store_token_level_embeddings_in_db(token, llm_model_name, token_level_embedding_json, client_ip, request_time, response_time, token_level_embedding_bundle_id)
        except RuntimeError as e:
            logger.error(f"Failed to calculate embedding for token '{token}': {e}")
    logger.info(f"Completed token embedding calculation for all tokens in text: '{text}'")
    return token_embeddings

async def get_token_level_embedding_from_db(token: str, llm_model_name: str) -> Optional[List[float]]:
    token_hash = sha3_256(token.encode('utf-8')).hexdigest() # Compute the hash
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            sql_text("SELECT token_level_embedding_json FROM token_level_embeddings WHERE token_hash=:token_hash AND llm_model_name=:llm_model_name"),
            {"token_hash": token_hash, "llm_model_name": llm_model_name},
        )
        row = result.fetchone()
        if row:
            embedding_json = row[0]
            logger.info(f"Embedding found in database for token hash '{token_hash}' using model '{llm_model_name}'")
            return json.loads(embedding_json)
        return None

async def store_token_level_embeddings_in_db(token: str, llm_model_name: str, token_level_embedding_json: str, ip_address: str, request_time: datetime, response_time: datetime, token_level_embedding_bundle_id: int):
    total_time = (response_time - request_time).total_seconds()
    embedding = TokenLevelEmbedding(
        token=token,
        llm_model_name=llm_model_name,
        token_level_embedding_json=token_level_embedding_json,
        ip_address=ip_address,
        request_time=request_time,
        response_time=response_time,
        total_time=total_time,
        token_level_embedding_bundle_id=token_level_embedding_bundle_id
    )
    await shared_resources.db_writer.enqueue_write([embedding]) # Enqueue the write operation for the token-level embedding
        
def calculate_sentence_embedding(llama: Llama, text: str) -> np.array:
    sentence_embedding = None
    retry_count = 0
    while sentence_embedding is None and retry_count < 3:
        try:
            if retry_count > 0:
                logger.info(f"Attempting again calculate sentence embedding. Attempt number {retry_count + 1}")
            sentence_embedding = llama.embed_query(text)
        except TypeError as e:
            logger.error(f"TypeError in calculate_sentence_embedding: {e}")
            raise
        except Exception as e:
            logger.error(f"Exception in calculate_sentence_embedding: {e}")
            text = text[:-int(len(text) * 0.1)]
            retry_count += 1
            logger.info(f"Trimming sentence due to too many tokens. New length: {len(text)}")
    if sentence_embedding is None:
        logger.error("Failed to calculate sentence embedding after multiple attempts")
    return sentence_embedding

async def compute_embeddings_for_document(strings: list, llm_model_name: str, client_ip: str, document_file_hash: str) -> List[Tuple[str, np.array]]:
    from swiss_army_llama import get_embedding_vector_for_string
    results = []
    if USE_PARALLEL_INFERENCE_QUEUE:
        logger.info(f"Using parallel inference queue to compute embeddings for {len(strings)} strings")
        start_time = time.perf_counter()  # Record the start time
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_PARALLEL_INFERENCE_TASKS)
        async def compute_embedding(text):  # Define a function to compute the embedding for a given text
            try:
                async with semaphore:  # Acquire a semaphore slot
                    request = EmbeddingRequest(text=text, llm_model_name=llm_model_name)
                    embedding = await get_embedding_vector_for_string(request, client_ip=client_ip, document_file_hash=document_file_hash)
                    return text, embedding["embedding"]
            except Exception as e:
                logger.error(f"Error computing embedding for text '{text}': {e}")
                return text, None
        results = await asyncio.gather(*[compute_embedding(s) for s in strings])  # Use asyncio.gather to run the tasks concurrently
        end_time = time.perf_counter()  # Record the end time
        duration = end_time - start_time
        if len(strings) > 0:
            logger.info(f"Parallel inference task for {len(strings)} strings completed in {duration:.2f} seconds; {duration / len(strings):.2f} seconds per string")
    else:  # Compute embeddings sequentially
        logger.info(f"Using sequential inference to compute embeddings for {len(strings)} strings")
        start_time = time.perf_counter()  # Record the start time
        for s in strings:
            embedding_request = EmbeddingRequest(text=s, llm_model_name=llm_model_name)
            embedding = await get_embedding_vector_for_string(embedding_request, client_ip=client_ip, document_file_hash=document_file_hash)
            results.append((s, embedding["embedding"]))
        end_time = time.perf_counter()  # Record the end time
        duration = end_time - start_time
        if len(strings) > 0:
            logger.info(f"Sequential inference task for {len(strings)} strings completed in {duration:.2f} seconds; {duration / len(strings):.2f} seconds per string")
    filtered_results = [(text, embedding) for text, embedding in results if embedding is not None] # Filter out results with None embeddings (applicable to parallel processing) and return
    return filtered_results

async def read_and_rewrite_file_with_safe_encoding(temp_file_path: str):
    content = None
    try:
        with open(temp_file_path, 'rb') as buffer:
            binary_content = buffer.read()
        try:
            content = binary_content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                content = binary_content.decode('latin1')
            except UnicodeDecodeError:
                content = binary_content.decode('unicode_escape')
    except Exception as e:
        logger.error(f"Error while reading and decoding file: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Error while reading and decoding file: {e}")
    try:
        with open(temp_file_path, 'w', encoding='utf-8') as buffer:
            buffer.write(content)
    except Exception as e:
        logger.error(f"Error while writing file with utf-8 encoding: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Error while writing file with utf-8 encoding: {e}")

async def parse_submitted_document_file_into_sentence_strings_func(temp_file_path: str, mime_type: str):
    if not mime_type.startswith('text/'):
        await read_and_rewrite_file_with_safe_encoding(temp_file_path)
    content = ""
    if mime_type.startswith('text/'):
        try:
            with open(temp_file_path, 'r', encoding='latin1') as buffer:
                content = buffer.read()
        except UnicodeDecodeError:
            with open(temp_file_path, 'r', encoding='unicode_escape') as buffer:
                content = buffer.read()
    else:
        try:
            content = textract.process(temp_file_path, method='pdfminer')
        except Exception as e:
            logger.error(f"Error while processing file: {e}, mime_type: {mime_type}")
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=f"Unsupported file type or error: {e}")
    if isinstance(content, bytes):
        try:
            content = content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                content = content.decode('latin1')
            except UnicodeDecodeError:
                content = content.decode('unicode_escape')        
    sentences = sophisticated_sentence_splitter(content)
    if len(sentences) == 0 and temp_file_path.lower().endswith('.pdf'):
        logger.info("No sentences found, attempting OCR using Tesseract.")
        try:
            content = textract.process(temp_file_path, method='tesseract')
            sentences = sophisticated_sentence_splitter(content)
        except Exception as e:
            logger.error(f"Error while processing file with OCR: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=400, detail="OCR failed: {e}")
    if len(sentences) == 0:
        logger.info("No sentences found in the document")
        raise HTTPException(status_code=400, detail="No sentences found in the document")
    logger.info(f"Extracted {len(sentences)} sentences from the document")
    strings = [s.strip() for s in sentences if len(s.strip()) > MINIMUM_STRING_LENGTH_FOR_DOCUMENT_EMBEDDING]
    return strings

async def _get_document_from_db(file_hash: str):
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Document).filter(Document.document_hash == file_hash))
        return result.scalar_one_or_none()

async def store_document_embeddings_in_db(file: File,
                                        file_hash: str,
                                        original_file_content: bytes,
                                        json_content: bytes,
                                        results: List[Tuple[str, np.array]],
                                        llm_model_name: str,
                                        client_ip: str,
                                        request_time: datetime,
                                        corpus_identifier_string: str):
    document = await _get_document_from_db(file_hash)  # First, check if a Document with the same hash already exists
    if not document:  # If not, create a new Document object
        document = Document(document_hash=file_hash, llm_model_name=llm_model_name, corpus_identifier_string=corpus_identifier_string)
        await shared_resources.db_writer.enqueue_write([document])    
    document_embedding = DocumentEmbedding(
        filename=file.filename,
        mimetype=file.content_type,
        file_hash=file_hash,
        llm_model_name=llm_model_name,
        file_data=original_file_content,
        document_embedding_results_json=json.loads(json_content.decode()),
        ip_address=client_ip,
        request_time=request_time,
        response_time=datetime.utcnow(),
        total_time=(datetime.utcnow() - request_time).total_seconds()
    )
    document.document_embeddings.append(document_embedding)  # Associate it with the Document
    document.update_hash()  # This will trigger the SQLAlchemy event to update the document_hash
    await shared_resources.db_writer.enqueue_write([document, document_embedding])  # Enqueue the write operation for the document embedding
    write_operations = []  # Collect text embeddings to write
    logger.info(f"Storing {len(results)} text embeddings in database")
    for text, embedding in results:
        embedding_entry = await _get_embedding_from_db(text, llm_model_name)
        if not embedding_entry:
            embedding_entry = TextEmbedding(
                text=text,
                llm_model_name=llm_model_name,
                embedding_json=json.dumps(embedding),
                ip_address=client_ip,
                request_time=request_time,
                response_time=datetime.utcnow(),
                total_time=(datetime.utcnow() - request_time).total_seconds(),
                document_file_hash=file_hash  # Link it to the DocumentEmbedding via file_hash
            )
        else:
            write_operations.append(embedding_entry)
    await shared_resources.db_writer.enqueue_write(write_operations)  # Enqueue the write operation for text embeddings


def load_text_completion_model(llm_model_name: str, raise_http_exception: bool = True):
    global USE_VERBOSE
    try:
        if llm_model_name in text_completion_model_cache: # Check if the model is already loaded in the cache
            return text_completion_model_cache[llm_model_name]
        models_dir = os.path.join(RAMDISK_PATH, 'models') if USE_RAMDISK else os.path.join(BASE_DIRECTORY, 'models') # Determine the model directory path
        matching_files = glob.glob(os.path.join(models_dir, f"{llm_model_name}*")) # Search for matching model files
        if not matching_files:
            logger.error(f"No model file found matching: {llm_model_name}")
            raise FileNotFoundError
        matching_files.sort(key=os.path.getmtime, reverse=True) # Sort the files based on modification time (recently modified files first)
        model_file_path = matching_files[0]
        with suppress_stdout_stderr():
            gpu_info = is_gpu_available()
            if gpu_info['gpu_found']:
                model_instance = Llama(model_path=model_file_path, n_ctx=TEXT_COMPLETION_CONTEXT_SIZE_IN_TOKENS, verbose=USE_VERBOSE, n_gpu_layers=-1) # Load the model with GPU acceleration
            else:
                model_instance = Llama(model_path=model_file_path, n_ctx=TEXT_COMPLETION_CONTEXT_SIZE_IN_TOKENS, verbose=USE_VERBOSE) # Load the model without GPU acceleration
        text_completion_model_cache[llm_model_name] = model_instance # Cache the loaded model
        return model_instance
    except TypeError as e:
        logger.error(f"TypeError occurred while loading the model: {e}")
        raise
    except Exception as e:
        logger.error(f"Exception occurred while loading the model: {e}")
        traceback.print_exc()
        if raise_http_exception:
            raise HTTPException(status_code=404, detail="Model file not found")
        else:
            raise FileNotFoundError(f"No model file found matching: {llm_model_name}")
        
async def generate_completion_from_llm(request: TextCompletionRequest, req: Request = None, client_ip: str = None) -> List[TextCompletionResponse]:
    request_time = datetime.utcnow()
    logger.info(f"Starting text completion calculation using model: '{request.llm_model_name}'for input prompt: '{request.input_prompt}'")
    logger.info(f"Loading model: '{request.llm_model_name}'")
    llm = load_text_completion_model(request.llm_model_name)
    logger.info(f"Done loading model: '{request.llm_model_name}'")
    list_of_llm_outputs = []
    grammar_file_string_lower = request.grammar_file_string.lower() if request.grammar_file_string else ""    
    if grammar_file_string_lower:
        list_of_grammar_files = glob.glob("./grammar_files/*.gbnf")
        matching_grammar_files = [x for x in list_of_grammar_files if grammar_file_string_lower in os.path.splitext(os.path.basename(x).lower())[0]]
        if len(matching_grammar_files) == 0:
            logger.error(f"No grammar file found matching: {request.grammar_file_string}")
            raise FileNotFoundError
        matching_grammar_files.sort(key=os.path.getmtime, reverse=True)
        grammar_file_path = matching_grammar_files[0]
        logger.info(f"Loading selected grammar file: '{grammar_file_path}'")
        llama_grammar = LlamaGrammar.from_file(grammar_file_path)
        for ii in range(request.number_of_completions_to_generate):
            logger.info(f"Generating completion {ii+1} of {request.number_of_completions_to_generate} with model {request.llm_model_name} for input prompt: '{request.input_prompt}'")
            output = llm(prompt=request.input_prompt, grammar=llama_grammar, max_tokens=request.number_of_tokens_to_generate, temperature=request.temperature)
            list_of_llm_outputs.append(output)
    else:
        for ii in range(request.number_of_completions_to_generate):
            output = llm(prompt=request.input_prompt, max_tokens=request.number_of_tokens_to_generate, temperature=request.temperature)
            list_of_llm_outputs.append(output)
    response_time = datetime.utcnow()
    total_time_per_completion = ((response_time - request_time).total_seconds()) / request.number_of_completions_to_generate
    list_of_responses = []
    for idx, current_completion_output in enumerate(list_of_llm_outputs):
        generated_text = current_completion_output['choices'][0]['text']
        if request.grammar_file_string == 'json':
            generated_text = generated_text.encode('unicode_escape').decode()
        llm_model_usage_json = json.dumps(current_completion_output['usage'])
        logger.info(f"Completed text completion {idx} in an average of {total_time_per_completion:.2f} seconds for input prompt: '{request.input_prompt}'; Beginning of generated text: \n'{generated_text[:100]}'...")
        response = TextCompletionResponse(input_prompt = request.input_prompt,
                                            llm_model_name = request.llm_model_name,
                                            grammar_file_string = request.grammar_file_string,
                                            number_of_tokens_to_generate = request.number_of_tokens_to_generate,
                                            number_of_completions_to_generate = request.number_of_completions_to_generate,
                                            time_taken_in_seconds = float(total_time_per_completion),
                                            generated_text = generated_text,
                                            llm_model_usage_json = llm_model_usage_json)
        list_of_responses.append(response)
    return list_of_responses

def validate_bnf_grammar_func(grammar):
    defined_rules, used_rules = set(), set()
    for line in grammar.strip().split('\n'):
        if '::=' not in line: 
            continue
        parts = line.split('::=')
        rule = parts[0].strip()
        if rule in defined_rules:
            return False, f"Rule {rule} is defined more than once."
        defined_rules.add(rule)
        expression = parts[-1]
        # Tokenize the expression using regex
        tokens = re.findall(r'\b[\w-]+\b|\[.*?\]|\(.*?\)|".*?"', expression)
        # Additional handling for complex expressions
        complex_tokens = re.findall(r'[\w-]+\[[\w-]+\]', expression)
        tokens.extend(complex_tokens)
        for token in tokens:
            if token.startswith('[') or token.startswith('(') or token.startswith('"'):
                continue  # Skip character classes, optional constructs, and string literals
            if '[' in token and ']' in token:  # Split complex tokens into individual rules
                sub_parts = token.split('[')
                used_rules.add(sub_parts[0])
                used_rules.add(sub_parts[1][:-1])
                continue
            used_rules.add(token)
    for rule in used_rules:
        if rule not in defined_rules:
            return False, f"Used rule {rule} is not defined."
    return True, "Valid BNF Grammar"

async def convert_document_to_sentences_func(file_path: str, mime_type: str) -> Dict[str, Any]:
    sentences = await parse_submitted_document_file_into_sentence_strings_func(file_path, mime_type)
    total_number_of_sentences = len(sentences)
    total_input_file_size_in_bytes = os.path.getsize(file_path)
    total_text_size_in_characters = sum(len(sentence) for sentence in sentences)
    total_words = sum(len(sentence.split()) for sentence in sentences)
    average_words_per_sentence = total_words / total_number_of_sentences if total_number_of_sentences else 0
    result = {
        "individual_sentences": sentences,
        "total_number_of_sentences": total_number_of_sentences,
        "average_words_per_sentence": average_words_per_sentence,
        "total_input_file_size_in_bytes": total_input_file_size_in_bytes,
        "total_text_size_in_characters": total_text_size_in_characters
    }
    return result

async def download_file(url: str, expected_size: int, expected_hash: str) -> str:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file_path = temp_file.name
    hash_obj = sha3_256()
    downloaded_size = 0
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", url) as response:
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to download file")
            async for chunk in response.aiter_bytes():
                downloaded_size += len(chunk)
                if downloaded_size > expected_size:
                    os.remove(temp_file_path)
                    raise HTTPException(status_code=400, detail="Downloaded file size exceeds expected size")
                temp_file.write(chunk)
                hash_obj.update(chunk)
    temp_file.close()
    if downloaded_size != expected_size:
        os.remove(temp_file_path)
        raise HTTPException(status_code=400, detail="Downloaded file size does not match expected size")
    if hash_obj.hexdigest() != expected_hash:
        os.remove(temp_file_path)
        raise HTTPException(status_code=400, detail="File hash mismatch")
    return temp_file_path

def get_audio_duration_seconds(file_path: str) -> float:
    audio = MutagenFile(file_path)
    if audio is None or not hasattr(audio.info, 'length'):
        raise ValueError("Could not determine the length of the audio file.")
    return audio.info.length

def start_resource_monitoring(endpoint_name: str, input_data: Dict[str, Any], client_ip: str) -> Dict[str, Any]:
    if not USE_RESOURCE_MONITORING:
        return {}
    # Capture initial system resource usage
    initial_memory = psutil.virtual_memory().used
    initial_cpu_times = psutil.cpu_times_percent(interval=None)
    start_time = time.time()
    # Placeholder for input-specific details
    request_details = {}
    # Extract endpoint-specific input details
    if endpoint_name == "get_embedding_vector_for_string":
        text = input_data.get("text", "")
        request_details = {
            "num_words": len(text.split()),
            "num_characters": len(text)
        }
    elif endpoint_name == "get_all_embedding_vectors_for_document":
        sentences = input_data.get("sentences", [])
        file_size_mb = input_data.get("file_size_mb", 0)
        mime_type = input_data.get("mime_type", "")
        request_details = {
            "num_sentences": len(sentences),
            "total_words": sum(len(sentence.split()) for sentence in sentences),
            "total_characters": sum(len(sentence) for sentence in sentences),
            "file_size_mb": file_size_mb,
            "mime_type": mime_type
        }
    elif endpoint_name == "compute_transcript_with_whisper_from_audio":
        transcript_details = input_data.get("transcript_details", {})
        file_size_mb = input_data.get("file_size_mb", 0)
        audio_duration_seconds = input_data.get("audio_duration_seconds", 0)
        request_details = {
            "file_size_mb": file_size_mb,
            "audio_duration_seconds": audio_duration_seconds,
            "num_sentences": len(transcript_details.get("sentences", [])),
            "total_words": sum(len(sentence.split()) for sentence in transcript_details.get("sentences", [])),
            "total_characters": sum(len(sentence) for sentence in transcript_details.get("sentences", []))
        }
    elif endpoint_name == "get_text_completions_from_input_prompt":
        input_prompt = input_data.get("input_prompt", "")
        request_details = {
            "num_words": len(input_prompt.split()),
            "num_characters": len(input_prompt),
            "llm_model_name": input_data.get("llm_model_name", ""),
            "temperature": input_data.get("temperature", 0.7),
            "grammar_file_string": input_data.get("grammar_file_string", ""),
            "number_of_completions_to_generate": input_data.get("number_of_completions_to_generate", 1),
            "number_of_tokens_to_generate": input_data.get("number_of_tokens_to_generate", 1000)
        }
    # Store initial state and request details in the context
    context = {
        "endpoint_name": endpoint_name,
        "start_time": start_time,
        "initial_memory": initial_memory,
        "initial_cpu_times": initial_cpu_times,
        "request_details": request_details,
        "client_ip": client_ip,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start_time))
    }
    return context

def end_resource_monitoring(context: Dict[str, Any]):
    if not USE_RESOURCE_MONITORING or not context:
        return
    # Retrieve initial state from context
    endpoint_name = context["endpoint_name"]
    start_time = context["start_time"]
    initial_memory = context["initial_memory"]
    initial_cpu_times = context["initial_cpu_times"]
    request_details = context["request_details"]
    client_ip = context["client_ip"]
    timestamp = context["timestamp"]
    # Capture final system resource usage
    end_time = time.time()
    final_memory = psutil.virtual_memory().used
    final_cpu_times = psutil.cpu_times_percent(interval=None)
    # Calculate the metrics
    memory_used = final_memory - initial_memory
    cpu_used = {
        "user": final_cpu_times.user - initial_cpu_times.user,
        "system": final_cpu_times.system - initial_cpu_times.system,
        "idle": final_cpu_times.idle - initial_cpu_times.idle
    }
    time_taken = end_time - start_time
    # Combine all metrics into a result dictionary
    result = {
        "timestamp": timestamp,
        "client_ip": client_ip,
        "endpoint_name": endpoint_name,
        "request_details": request_details,
        "memory_used": memory_used,
        "cpu_used": cpu_used,
        "time_taken": time_taken
    }
    # Append the result to the log file
    log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resource_monitoring_logs.json")
    try:
        with open(log_file_path, "a") as log_file:
            log_file.write(json.dumps(result) + "\n")
    except Exception as e:
        logger.error(f"Failed to write resource monitoring log: {e}")
    logger.info(f"Request data and system resources used: {result}")