from logger_config import setup_logger
import shared_resources
from shared_resources import load_model, text_completion_model_cache, is_gpu_available
from database_functions import AsyncSessionLocal, execute_with_retry
from misc_utility_functions import clean_filename_for_url_func,  FakeUploadFile, sophisticated_sentence_splitter, merge_transcript_segments_into_combined_text, suppress_stdout_stderr, image_to_base64_data_uri, process_image, find_clip_model_path
from embeddings_data_models import TextEmbedding, DocumentEmbedding, Document, AudioTranscript
from embeddings_data_models import EmbeddingRequest, TextCompletionRequest
from embeddings_data_models import TextCompletionResponse,  AudioTranscriptResponse, ImageQuestionResponse
import os
import re
import unicodedata
import shutil
import psutil
import glob
import json
import io
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
import zstandard as zstd
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from sqlalchemy.inspection import inspect
from fastapi import HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from typing import List, Optional, Dict, Any
from decouple import config
from faster_whisper import WhisperModel
from llama_cpp import Llama, LlamaGrammar
from llama_cpp.llama_chat_format import Llava16ChatHandler
from llama_cpp import llama_types
from mutagen import File as MutagenFile
from magika import Magika
import httpx
from sklearn.decomposition import TruncatedSVD, FastICA, FactorAnalysis
from sklearn.random_projection import GaussianRandomProjection

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
USE_FLASH_ATTENTION = config("USE_FLASH_ATTENTION", default=True, cast=bool)
RAMDISK_PATH = config("RAMDISK_PATH", default="/mnt/ramdisk", cast=str)
BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

    
# Core embedding functions start here:    
    
def prepare_string_for_embedding(text: str) -> str:
    # Normalize Unicode characters to NFKC form
    text = unicodedata.normalize('NFKC', text)
    # Define all possible newline and carriage return characters
    newline_chars = [
        '\r', '\n', '\r\n', '\u2028', '\u2029', '\v', '\f', 
        '\x85', '\u000A', '\u000B', '\u000C', '\u000D', '\u0085',
        '\u000D\u000A'
    ]
    # Replace all newline characters with a space
    for nl in newline_chars:
        text = text.replace(nl, ' ')
    # Replace any sequence of whitespace characters (including non-breaking spaces) with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading and trailing whitespace
    text = text.strip()
    # Remove leading comma followed by whitespace if present
    if text.startswith(', '):
        text = text[2:].strip()
    # Remove all control characters and non-printable characters
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
    # Ensure text is ASCII-encoded to catch any remaining unusual characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Truncate to a maximum length of 5000 characters
    if len(text) > 5000:
        text = text[:5000]
    # Eliminate all blank lines
    text = ' '.join(line for line in text.splitlines() if line.strip() != '')
    #Final trimming
    text = text.strip()
    return text

def compress_data(input_data):
    if isinstance(input_data, str):
        input_data = input_data.encode('utf-8')
    zstd_compression_level = 15 # 22 is the highest compression level; 15 is a good balance between compression and speed
    zstandard_compressor = zstd.ZstdCompressor(level=zstd_compression_level, write_content_size=True, write_checksum=True)
    zstd_compressed_data = zstandard_compressor.compress(input_data)
    return zstd_compressed_data

def decompress_data(compressed_data):
    return zstd.decompress(compressed_data)

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

async def get_embedding_from_db(text: str, llm_model_name: str, embedding_pooling_method: str):
    text_hash = sha3_256(text.encode('utf-8')).hexdigest()
    return await execute_with_retry(_get_embedding_from_db, text_hash, llm_model_name, embedding_pooling_method)

async def _get_embedding_from_db(text_hash: str, llm_model_name: str, embedding_pooling_method: str) -> Optional[TextEmbedding]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(TextEmbedding)
            .filter(TextEmbedding.text_hash == text_hash,
                    TextEmbedding.llm_model_name == llm_model_name,
                    TextEmbedding.embedding_pooling_method == embedding_pooling_method)
        )
        return result.scalars().first()
    
async def get_corpus_identifier_from_embedding_text(text: str, llm_model_name: str, embedding_pooling_method: str):
    text_hash = sha3_256(text.encode('utf-8')).hexdigest()
    return await execute_with_retry(_get_corpus_identifier_from_embedding_text, text_hash, llm_model_name, embedding_pooling_method)

async def _get_corpus_identifier_from_embedding_text(text_hash: str, llm_model_name: str, embedding_pooling_method: str) -> Optional[str]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(TextEmbedding.corpus_identifier_string)
            .filter(TextEmbedding.text_hash == text_hash,
                    TextEmbedding.llm_model_name == llm_model_name,
                    TextEmbedding.embedding_pooling_method == embedding_pooling_method)
        )
        return result.scalar()

async def get_list_of_corpus_identifiers_from_list_of_embedding_texts(list_of_texts: List[str], llm_model_name: str, embedding_pooling_method: str):
    list_of_text_hashes = [sha3_256(text.encode('utf-8')).hexdigest() for text in list_of_texts]
    return await execute_with_retry(_get_list_of_corpus_identifiers_from_list_of_embedding_texts, list_of_text_hashes, llm_model_name, embedding_pooling_method)

async def _get_list_of_corpus_identifiers_from_list_of_embedding_texts(list_of_text_hashes: List[str], llm_model_name: str, embedding_pooling_method: str) -> List[str]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(TextEmbedding.corpus_identifier_string)
            .filter(TextEmbedding.text_hash.in_(list_of_text_hashes),
                    TextEmbedding.llm_model_name == llm_model_name,
                    TextEmbedding.embedding_pooling_method == embedding_pooling_method)
        )
        rows = result.scalars().all()
        return rows
    
async def get_texts_for_corpus_identifier(corpus_identifier_string: str) -> Dict[str, List[str]]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(DocumentEmbedding)
            .options(joinedload(DocumentEmbedding.embeddings))
            .filter(DocumentEmbedding.corpus_identifier_string == corpus_identifier_string)
        )
        document_embeddings = result.unique().scalars().all()
        texts_by_model_and_embedding_pooling_method = {(doc.llm_model_name, doc.embedding_pooling_method): [] for doc in document_embeddings}
        for document_embedding in document_embeddings:
            texts_by_model_and_embedding_pooling_method[(document_embedding.llm_model_name, document_embedding.embedding_pooling_method)].extend(
                [embedding.text for embedding in document_embedding.embeddings]
            )
    return texts_by_model_and_embedding_pooling_method

async def get_texts_for_model_and_embedding_pooling_method(llm_model_name: str, embedding_pooling_method: str) -> Dict[str, List[str]]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(DocumentEmbedding)
            .options(joinedload(DocumentEmbedding.embeddings))
            .filter(DocumentEmbedding.llm_model_name == llm_model_name, DocumentEmbedding.embedding_pooling_method == embedding_pooling_method)
        )
        document_embeddings = result.unique().scalars().all()
        texts_by_model_and_embedding_pooling_method = {(doc.llm_model_name, doc.embedding_pooling_method): [] for doc in document_embeddings}
        for document_embedding in document_embeddings:
            texts_by_model_and_embedding_pooling_method[(document_embedding.llm_model_name, document_embedding.embedding_pooling_method)].extend(
                [embedding.text for embedding in document_embedding.embeddings]
            )
    return texts_by_model_and_embedding_pooling_method

async def get_or_compute_embedding(request: EmbeddingRequest, req: Request = None, client_ip: str = None, document_file_hash: str = None, use_verbose: bool = True) -> dict:
    request_time = datetime.utcnow()  # Capture request time as datetime object
    ip_address = (
        client_ip or (req.client.host if req else "localhost")
    )  # If client_ip is provided, use it; otherwise, try to get from req; if not available, default to "localhost"
    if use_verbose:
        logger.info(f"Received request for embedding for '{request.text}' using model '{request.llm_model_name}' and embedding pooling method '{request.embedding_pooling_method}' from IP address '{ip_address}'")
    text_embedding_instance = await get_embedding_from_db(
        request.text, request.llm_model_name, request.embedding_pooling_method
    )
    if text_embedding_instance is not None: # Check if embedding exists in the database
        response_time = datetime.utcnow()  # Capture response time as datetime object
        total_time = (
            response_time - request_time
        ).total_seconds()  # Calculate time taken in seconds
        if use_verbose:
            logger.info(f"Embedding found in database for '{request.text}' using model '{request.llm_model_name}' and embedding pooling method '{request.embedding_pooling_method}'; returning in {total_time:.4f} seconds")
        return {"text_embedding_dict": text_embedding_instance.as_dict()}
    model = load_model(request.llm_model_name)
    # Compute the embedding if not in the database
    list_of_embedding_entry_dicts = await calculate_sentence_embeddings_list(model, [request.text], request.embedding_pooling_method)
    embedding_entry_dict = list_of_embedding_entry_dicts[0]
    if embedding_entry_dict is None:
        logger.error(
            f"Could not calculate the embedding for the given text: '{request.text}' using model '{request.llm_model_name} and embedding pooling method '{request.embedding_pooling_method}!'"
        )
        raise HTTPException(
            status_code=400,
            detail="Could not calculate the embedding for the given text",
        )
    else:
        embedding = embedding_entry_dict['embedding']
        embedding_hash = embedding_entry_dict['embedding_hash']
        text = request.text
        text_hash = sha3_256(text.encode('utf-8')).hexdigest()
        embedding_json = json.dumps(embedding)
        request_time = datetime.utcnow()
        response_time = datetime.utcnow()
        total_time = (response_time - request_time).total_seconds()
        embedding_instance = TextEmbedding(
            text=text,
            text_hash=text_hash,
            embedding_hash=embedding_hash,
            llm_model_name=request.llm_model_name,
            embedding_pooling_method=request.embedding_pooling_method,
            corpus_identifier_string=request.corpus_identifier_string,
            embedding_json=embedding_json,
            ip_address=client_ip,
            request_time=request_time,
            response_time=response_time,
            total_time=total_time,
            document_file_hash=document_file_hash,
        )
    word_length_of_input_text = len(request.text.split())
    if word_length_of_input_text > 0:
        if use_verbose:
            logger.info(f"Embedding calculated for '{request.text}' using model '{request.llm_model_name}' and embedding pooling method '{request.embedding_pooling_method}' in {total_time:,.2f} seconds, or an average of {total_time/word_length_of_input_text :.2f} seconds per word. Now saving to database...")
    await shared_resources.db_writer.enqueue_write([embedding_instance])  # Enqueue the write operation using the db_writer instance
    return {"text_embedding_dict": embedding_instance.as_dict()}

async def calculate_sentence_embeddings_list(llama, texts: list, embedding_pooling_method: str) -> list:
    start_time = datetime.utcnow()
    total_number_of_sentences = len(texts)
    total_characters = sum(len(s) for s in texts)
    sentence_embeddings_object = llama.create_embedding(texts)
    sentence_embeddings_list = sentence_embeddings_object['data']
    if len(sentence_embeddings_list) != len(texts):
        raise ValueError("Inconsistent number of embeddings found.")
    list_of_embedding_entry_dicts = []
    cnt = 0
    for i, current_text in enumerate(texts):
        current_set_of_embeddings = sentence_embeddings_list[i]['embedding']
        if isinstance(current_set_of_embeddings[0], list):
            number_of_embeddings = len(current_set_of_embeddings)
        else:
            number_of_embeddings = 1
            current_set_of_embeddings = [current_set_of_embeddings]
        logger.info(f"Sentence {i + 1} of {len(texts):,} has {number_of_embeddings:,} embeddings for text '{current_text[:50]}...'")
        embeddings = np.array(current_set_of_embeddings)
        dimension_of_token_embeddings = embeddings.shape[1]
        # Ensure embeddings have enough dimensions for the pooling method
        required_components = {
            "svd": 2,
            "svd_first_four": 4,
            "ica": 2,
            "factor_analysis": 2,
            "gaussian_random_projection": 2
        }
        if number_of_embeddings > 1:
            min_components = required_components.get(embedding_pooling_method, 1)
            if number_of_embeddings < min_components:
                padding = np.zeros((min_components - number_of_embeddings, dimension_of_token_embeddings))
                embeddings = np.vstack([embeddings, padding])
            if embedding_pooling_method == "mean":
                element_wise_mean = np.mean(embeddings, axis=0)
                flattened_vector = element_wise_mean.flatten()
            elif embedding_pooling_method == "mins_maxes":
                element_wise_min = np.min(embeddings, axis=0)
                element_wise_max = np.max(embeddings, axis=0)
                flattened_vector = np.concatenate([element_wise_min, element_wise_max], axis=0)
            elif embedding_pooling_method == "svd":
                svd = TruncatedSVD(n_components=2)
                svd_embeddings = svd.fit_transform(embeddings.T)
                flattened_vector = svd_embeddings.flatten()
            elif embedding_pooling_method == "svd_first_four":
                svd = TruncatedSVD(n_components=4)
                svd_embeddings = svd.fit_transform(embeddings.T)
                flattened_vector = svd_embeddings.flatten()
            elif embedding_pooling_method == "ica":
                ica = FastICA(n_components=2)
                ica_embeddings = ica.fit_transform(embeddings.T)
                flattened_vector = ica_embeddings.flatten()
            elif embedding_pooling_method == "factor_analysis":
                fa = FactorAnalysis(n_components=2)
                fa_embeddings = fa.fit_transform(embeddings.T)
                flattened_vector = fa_embeddings.flatten()           
            elif embedding_pooling_method == "gaussian_random_projection":
                grp = GaussianRandomProjection(n_components=2)
                grp_embeddings = grp.fit_transform(embeddings.T)
                flattened_vector = grp_embeddings.flatten()                 
            else:
                raise ValueError(f"Unknown embedding_pooling_method: {embedding_pooling_method}")
            combined_embedding = flattened_vector.tolist()
        else:
            flattened_vector = embeddings.flatten().tolist()
            combined_embedding = embeddings.flatten().tolist()
        embedding_length = len(combined_embedding)
        cnt += 1
        embedding_json = json.dumps(combined_embedding)
        embedding_hash = sha3_256(embedding_json.encode('utf-8')).hexdigest()
        embedding_entry_dict = {'text_index': i, 'text': current_text, 'embedding_pooling_method': embedding_pooling_method, 'number_of_token_embeddings_used': number_of_embeddings, 'embedding_length': embedding_length, 'embedding_hash': embedding_hash, 'embedding': combined_embedding}
        list_of_embedding_entry_dicts.append(embedding_entry_dict)
    end_time = datetime.utcnow()
    total_time = (end_time - start_time).total_seconds()
    logger.info(f"Calculated {len(flattened_vector):,}-dimensional embeddings (relative to the underlying token embedding dimensions of {dimension_of_token_embeddings:,}) for {total_number_of_sentences:,} sentences in a total of {total_time:,.1f} seconds.")
    logger.info(f"That's an average of {1000*total_time/total_number_of_sentences:,.2f} ms per sentence and {total_number_of_sentences/total_time:,.3f} sentences per second (and {total_characters/(1000*total_time):,.4f} total characters per ms) using pooling method '{embedding_pooling_method}'")
    return list_of_embedding_entry_dicts

async def batch_save_embeddings_to_db(embeddings: List[TextEmbedding]):
    async with AsyncSessionLocal() as session:
        # Extract the unique embedding_hashes from the embeddings list
        embedding_hashes = [embedding.embedding_hash for embedding in embeddings]
        # Query the database for existing embeddings with the same hashes
        existing_embeddings_query = select(TextEmbedding.embedding_hash).where(TextEmbedding.embedding_hash.in_(embedding_hashes))
        result = await session.execute(existing_embeddings_query)
        existing_embedding_hashes = {row.embedding_hash for row in result}
        # Filter out embeddings that already exist in the database
        embeddings_to_insert = [embedding for embedding in embeddings if embedding.embedding_hash not in existing_embedding_hashes]
        # Batch insert the remaining embeddings
        if embeddings_to_insert:
            session.add_all(embeddings_to_insert)
            await session.commit()
            
async def compute_embeddings_for_document(sentences: list, llm_model_name: str, embedding_pooling_method: str, corpus_identifier_string: str, client_ip: str, document_file_hash: str, file: UploadFile, original_file_content: bytes, json_format: str = 'records') -> list:
    request_time = datetime.utcnow()
    sentences = [prepare_string_for_embedding(text) for text in sentences]
    model = load_model(llm_model_name)
    try:
        list_of_embedding_entry_dicts = await calculate_sentence_embeddings_list(model, sentences, embedding_pooling_method)
    except Exception as e:
        logger.error(f"Error computing embeddings for batch: {e}")
        logger.error(traceback.format_exc())
        raise
    embeddings_to_save = []
    list_of_embedding_hashes_added = []
    for embedding_entry_dict in list_of_embedding_entry_dicts:
        embedding = embedding_entry_dict['embedding']
        embedding_hash = embedding_entry_dict['embedding_hash']
        if embedding_hash in list_of_embedding_hashes_added:
            continue
        text_index = embedding_entry_dict['text_index']
        text = sentences[text_index]
        text_hash = sha3_256(text.encode('utf-8')).hexdigest()
        embedding_json = json.dumps(embedding)
        response_time = datetime.utcnow()
        total_time = (response_time - request_time).total_seconds()
        embedding_instance = TextEmbedding(
            text=text,
            text_hash=text_hash,
            embedding_hash=embedding_hash,
            llm_model_name=llm_model_name,
            embedding_pooling_method=embedding_pooling_method,
            corpus_identifier_string=corpus_identifier_string,
            embedding_json=embedding_json,
            ip_address=client_ip,
            request_time=request_time,
            response_time=response_time,
            total_time=total_time,
            document_file_hash=document_file_hash,
        )
        embeddings_to_save.append(embedding_instance)
        list_of_embedding_hashes_added.append(embedding_hash)
    logger.info(f"Storing {len(embeddings_to_save):,} text embeddings in database...")
    await batch_save_embeddings_to_db(embeddings_to_save)
    logger.info(f"Done storing {len(embeddings_to_save):,} text embeddings in database.")
    document_embedding_results_df = pd.DataFrame(list_of_embedding_entry_dicts)
    json_content = document_embedding_results_df.to_json(orient=json_format or 'records').encode()
    if file is not None:
        await store_document_embeddings_in_db(
            file=file,
            document_file_hash=document_file_hash,
            original_file_content=original_file_content,
            sentences=sentences,
            json_content=json_content,
            llm_model_name=llm_model_name,
            embedding_pooling_method=embedding_pooling_method,
            corpus_identifier_string=corpus_identifier_string,
            client_ip=client_ip,
            request_time=request_time,
        )    
    return json_content

async def parse_submitted_document_file_into_sentence_strings_func(temp_file_path: str, mime_type: str):
    content = ""
    try:
        content = textract.process(temp_file_path, method='pdfminer', encoding='utf-8')
        content = content.decode('utf-8')
    except Exception as e:
        logger.error(f"Error while processing file: {e}, mime_type: {mime_type}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Unsupported file type or error: {e}")
    sentences = sophisticated_sentence_splitter(content)
    if len(sentences) == 0 and temp_file_path.lower().endswith('.pdf'):
        logger.info("No sentences found, attempting OCR using Tesseract.")
        try:
            content = textract.process(temp_file_path, method='tesseract', encoding='utf-8')
            content = content.decode('utf-8')
            sentences = sophisticated_sentence_splitter(content)
        except Exception as e:
            logger.error(f"Error while processing file with OCR: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=400, detail="OCR failed: {e}")
    if len(sentences) == 0:
        logger.info("No sentences found in the document")
        raise HTTPException(status_code=400, detail="No sentences found in the document")
    strings = [s.strip() for s in sentences if len(s.strip()) > MINIMUM_STRING_LENGTH_FOR_DOCUMENT_EMBEDDING]
    thousands_of_input_words = round(sum(len(s.split()) for s in strings) / 1000, 2)
    return strings, thousands_of_input_words

async def _get_document_from_db(document_file_hash: str):
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Document).filter(Document.document_hash == document_file_hash))
        return result.scalar_one_or_none()

async def store_document_embeddings_in_db(file, document_file_hash: str, original_file_content: bytes, sentences: List[str], json_content: bytes, llm_model_name: str, embedding_pooling_method:str, corpus_identifier_string: str, client_ip: str, request_time: datetime):
    if file is None:
        logger.error("Received a None file object in store_document_embeddings_in_db")
    else:
        logger.info(f"Received file: {file.filename} with content type: {file.content_type}")
    sentences = json.dumps(sentences)
    document = await _get_document_from_db(document_file_hash)
    if not document:
        document = Document(document_hash=document_file_hash, llm_model_name=llm_model_name, corpus_identifier_string=corpus_identifier_string)
        await shared_resources.db_writer.enqueue_write([document])
    document_embedding_results_json_compressed_binary = compress_data(json_content)
    document_embedding = DocumentEmbedding(
        filename=file.filename,
        mimetype=file.content_type,
        document_file_hash=document_file_hash,
        llm_model_name=llm_model_name,
        embedding_pooling_method=embedding_pooling_method,
        corpus_identifier_string=corpus_identifier_string,
        file_data=original_file_content,
        sentences=sentences,
        document_embedding_results_json_compressed_binary=document_embedding_results_json_compressed_binary,
        ip_address=client_ip,
        request_time=request_time,
        response_time=datetime.utcnow(),
        total_time=(datetime.utcnow() - request_time).total_seconds()
    )
    document.document_embeddings.append(document_embedding)
    document.update_hash()
    await shared_resources.db_writer.enqueue_write([document, document_embedding])
    
def load_text_completion_model(llm_model_name: str, raise_http_exception: bool = True):
    global USE_VERBOSE
    try:
        if llm_model_name in text_completion_model_cache:
            return text_completion_model_cache[llm_model_name]
        models_dir = os.path.join(RAMDISK_PATH, 'models') if USE_RAMDISK else os.path.join(BASE_DIRECTORY, 'models')
        matching_files = glob.glob(os.path.join(models_dir, f"{llm_model_name}*"))
        if not matching_files:
            logger.error(f"No model file found matching: {llm_model_name}")
            raise FileNotFoundError
        matching_files.sort(key=os.path.getmtime, reverse=True)
        model_file_path = matching_files[0]
        is_llava_multimodal_model = 'llava' in llm_model_name and 'mmproj' not in llm_model_name
        chat_handler = None # Determine the appropriate chat handler based on the model name
        if 'llava' in llm_model_name:
            clip_model_path = find_clip_model_path(llm_model_name)
            if clip_model_path is None:
                raise FileNotFoundError
            chat_handler = Llava16ChatHandler(clip_model_path=clip_model_path)
        with suppress_stdout_stderr():
            gpu_info = is_gpu_available()
            if gpu_info:
                num_gpus = gpu_info['num_gpus']
                if num_gpus > 1:
                    llama_split_mode = 2 # 2, // split rows across GPUs | 1, // split layers and KV across GPUs
                else:
                    llama_split_mode = 0
            else:
                num_gpus = 0
            try:                
                model_instance = Llama(
                    model_path=model_file_path,
                    embedding=True if is_llava_multimodal_model else False,
                    n_ctx=TEXT_COMPLETION_CONTEXT_SIZE_IN_TOKENS,
                    flash_attn=USE_FLASH_ATTENTION,
                    verbose=USE_VERBOSE,
                    llama_split_mode=llama_split_mode,
                    n_gpu_layers=-1 if gpu_info['gpu_found'] else 0,
                    clip_model_path=clip_model_path if is_llava_multimodal_model else None,
                    chat_handler=chat_handler
                )
            except Exception as e:  # noqa: F841
                model_instance = Llama(
                    model_path=model_file_path,
                    embedding=True if is_llava_multimodal_model else False,
                    n_ctx=TEXT_COMPLETION_CONTEXT_SIZE_IN_TOKENS,
                    flash_attn=USE_FLASH_ATTENTION,
                    verbose=USE_VERBOSE,
                    clip_model_path=clip_model_path if is_llava_multimodal_model else None,
                    chat_handler=chat_handler
                )                
        text_completion_model_cache[llm_model_name] = model_instance
        return model_instance
    except TypeError as e:
        logger.error(f"TypeError occurred while loading the model: {e}")
        logger.error(traceback.format_exc())        
        raise
    except Exception as e:
        logger.error(f"Exception occurred while loading the model: {e}")
        logger.error(traceback.format_exc())
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
    chat_handler = llm.chat_handler # Use the appropriate chat handler based on the model name
    if chat_handler is None: # Use the default code path if no chat handler is found
        for ii in range(request.number_of_completions_to_generate):
            logger.info(f"Generating completion {ii+1} of {request.number_of_completions_to_generate} with model {request.llm_model_name} for input prompt: '{request.input_prompt}'")
            output = llm(prompt=request.input_prompt, max_tokens=request.number_of_tokens_to_generate, temperature=request.temperature)
            list_of_llm_outputs.append(output)
    else:
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
                output = chat_handler(
                    llama=llm,
                    messages=[llama_types.ChatCompletionRequestUserMessage(content=request.input_prompt)],
                    grammar=llama_grammar,
                    max_tokens=request.number_of_tokens_to_generate,
                    temperature=request.temperature,
                )
                list_of_llm_outputs.append(output)
        else:
            for ii in range(request.number_of_completions_to_generate):
                logger.info(f"Generating completion {ii+1} of {request.number_of_completions_to_generate} with model {request.llm_model_name} for input prompt: '{request.input_prompt}'")
                output = chat_handler(
                    llama=llm,
                    messages=[llama_types.ChatCompletionRequestUserMessage(content=request.input_prompt)],
                    max_tokens=request.number_of_tokens_to_generate,
                    temperature=request.temperature,
                )
                list_of_llm_outputs.append(output)
    response_time = datetime.utcnow()
    total_time_per_completion = ((response_time - request_time).total_seconds()) / request.number_of_completions_to_generate
    list_of_responses = []
    for idx, current_completion_output in enumerate(list_of_llm_outputs):
        model_output = current_completion_output['choices'][0]
        if 'message' in model_output.keys():            
            generated_text = model_output['message']['content']
        else:
            generated_text = model_output['text']
        if request.grammar_file_string == 'json':
            generated_text = generated_text.encode('unicode_escape').decode()
        finish_reason = str(model_output['finish_reason'])                
        llm_model_usage_json = json.dumps(current_completion_output['usage'])
        logger.info(f"Completed text completion {idx:,} in an average of {total_time_per_completion:,.2f} seconds for input prompt: '{request.input_prompt}'; Beginning of generated text: \n'{generated_text[:100]}'...")
        response = TextCompletionResponse(input_prompt = request.input_prompt,
                                            llm_model_name = request.llm_model_name,
                                            grammar_file_string = request.grammar_file_string,
                                            number_of_tokens_to_generate = request.number_of_tokens_to_generate,
                                            number_of_completions_to_generate = request.number_of_completions_to_generate,
                                            time_taken_in_seconds = float(total_time_per_completion),
                                            generated_text = generated_text,
                                            finish_reason = finish_reason,
                                            llm_model_usage_json = llm_model_usage_json)
        list_of_responses.append(response)
    return list_of_responses

async def ask_question_about_image(
    question: str,
    llm_model_name: str,
    temperature: float,
    number_of_tokens_to_generate: int,
    number_of_completions_to_generate: int,
    image: UploadFile,
    req: Request = None,
    client_ip: str = None
) -> List[ImageQuestionResponse]:
    if 'llava' not in llm_model_name:
        logger.error(f"Model '{llm_model_name}' is not a valid LLaVA model.")
        raise HTTPException(status_code=400, detail="Model name must include 'llava'")
    request_time = datetime.utcnow()
    logger.info(f"Starting image question calculation using model: '{llm_model_name}' for question: '{question}'")
    logger.info(f"Loading model: '{llm_model_name}'")
    llm = load_text_completion_model(llm_model_name)
    logger.info(f"Done loading model: '{llm_model_name}'")
    original_image_path = f"/tmp/{image.filename}"
    with open(original_image_path, "wb") as image_file:
        image_file.write(await image.read())
    processed_image_path = process_image(original_image_path)
    image_hash = sha3_256(open(processed_image_path, 'rb').read()).hexdigest()
    data_uri = image_to_base64_data_uri(processed_image_path)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_uri }},
            {"type": "text", "text": question}
        ]},
    ]
    responses = []
    for completion_count in range(number_of_completions_to_generate):
        with suppress_stdout_stderr():
            llm_output = llm.create_chat_completion(
                messages=messages,
                max_tokens=number_of_tokens_to_generate,
                temperature=temperature,
                top_p=0.95,
                stream=False,
            )
        response_time = datetime.utcnow()
        total_time_taken = (response_time - request_time).total_seconds()
        model_output = llm_output['choices'][0]
        generated_text = model_output['message']['content']      
        finish_reason = str(model_output['finish_reason'])
        llm_model_usage_json = json.dumps(llm_output['usage'])
        response = ImageQuestionResponse(
            question=question,
            llm_model_name=llm_model_name,
            image_hash=image_hash,
            time_taken_in_seconds=total_time_taken,
            number_of_tokens_to_generate=number_of_tokens_to_generate,
            number_of_completions_to_generate=number_of_completions_to_generate,
            generated_text=generated_text,
            finish_reason=finish_reason,
            llm_model_usage_json=llm_model_usage_json
        )
        logger.info(f"Completed image question calculation in {total_time_taken:.2f} seconds for question: '{question}'; Beginning of generated text: \n'{generated_text[:100]}'...")
        responses.append(response)
    return responses

def validate_bnf_grammar_func(grammar: str):
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
    sentences, thousands_of_input_words = await parse_submitted_document_file_into_sentence_strings_func(file_path, mime_type)
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
        "total_text_size_in_characters": total_text_size_in_characters,
        "thousands_of_input_words": thousands_of_input_words
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

# Audio Transcript functions start here:

def object_as_dict(obj):
    return {c.key: getattr(obj, c.key) for c in inspect(obj).mapper.column_attrs}

def convert_to_pydantic_response(audio_transcript, compute_embeddings_for_resulting_transcript_document, llm_model_name, embedding_pooling_method, download_url):
    audio_transcript_dict = object_as_dict(audio_transcript)
    # Convert JSON fields from strings to proper lists/dictionaries using json.loads
    audio_transcript_dict['segments_json'] = json.loads(audio_transcript_dict['segments_json'])
    audio_transcript_dict['combined_transcript_text_list_of_metadata_dicts'] = json.loads(audio_transcript_dict['combined_transcript_text_list_of_metadata_dicts'])
    audio_transcript_dict['info_json'] = json.loads(audio_transcript_dict['info_json'])
    # Update fields based on the request
    audio_transcript_dict['url_to_download_zip_file_of_embeddings'] = download_url
    if compute_embeddings_for_resulting_transcript_document:
        audio_transcript_dict['llm_model_name'] = llm_model_name
        audio_transcript_dict['embedding_pooling_method'] = embedding_pooling_method
    else:
        audio_transcript_dict['llm_model_name'] = ""
        audio_transcript_dict['embedding_pooling_method'] = ""
    return audio_transcript_dict

async def get_transcript_from_db(audio_file_hash: str) -> Optional[AudioTranscript]:
    return await execute_with_retry(_get_transcript_from_db, audio_file_hash)

async def _get_transcript_from_db(audio_file_hash: str) -> Optional[AudioTranscript]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(AudioTranscript).filter(AudioTranscript.audio_file_hash == audio_file_hash)
        )
        transcript = result.scalars().first()
        return transcript

async def save_transcript_to_db(audio_file_hash: str, audio_file_name: str, audio_file_size_mb: float, transcript_segments: json.dumps, info: json.dumps, ip_address: str, request_time: datetime, response_time: datetime, total_time: float, combined_transcript_text: str, combined_transcript_text_list_of_metadata_dicts: json.dumps, corpus_identifier_string: str):
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

async def compute_and_store_transcript_embeddings(audio_file_name: str, sentences: list, llm_model_name: str, embedding_pooling_method: str, corpus_identifier_string: str, ip_address: str, combined_transcript_text: str, req: Request):
    request_time=datetime.utcnow()
    logger.info(f"Now computing embeddings for entire transcript of {audio_file_name}...")
    zip_dir = 'generated_transcript_embeddings_zip_files'
    if not os.path.exists(zip_dir):
        os.makedirs(zip_dir)
    sanitized_file_name = clean_filename_for_url_func(audio_file_name)
    document_name = f"automatic_whisper_transcript_of__{sanitized_file_name}"
    document_file_hash = sha3_256(combined_transcript_text.encode('utf-8')).hexdigest()
    sentences = sophisticated_sentence_splitter(combined_transcript_text)
    computed_embeddings = await compute_embeddings_for_document(
        sentences=sentences,
        llm_model_name=llm_model_name,
        embedding_pooling_method=embedding_pooling_method,
        corpus_identifier_string=corpus_identifier_string,
        client_ip=ip_address,
        document_file_hash=document_file_hash,
        file=None,
        original_file_content=combined_transcript_text.encode(),
        json_format="records",
    )
    zip_file_path = f"{zip_dir}/{quote(document_name)}.zip"
    # Ensure computed_embeddings is JSON serializable
    if isinstance(computed_embeddings, bytes):
        computed_embeddings = computed_embeddings.decode('utf-8')    
    zip_file_path = f"{zip_dir}/{quote(document_name)}.zip"
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        zipf.writestr("embeddings.txt", json.dumps(computed_embeddings))
    download_url = f"download/{quote(document_name)}.zip"
    full_download_url = f"{req.base_url}{download_url}"
    logger.info(f"Generated download URL for transcript embeddings: {full_download_url}")
    fake_upload_file = FakeUploadFile(filename=document_name, content=combined_transcript_text.encode(), content_type='text/plain')
    logger.info(f"Storing transcript embeddings for {audio_file_name} in the database...")
    await store_document_embeddings_in_db(
        file=fake_upload_file,
        document_file_hash=document_file_hash,
        original_file_content=combined_transcript_text.encode('utf-8'),
        sentences=sentences,
        json_content=json.dumps(computed_embeddings).encode('utf-8'),
        llm_model_name=llm_model_name,
        embedding_pooling_method=embedding_pooling_method,
        corpus_identifier_string=corpus_identifier_string,
        client_ip=ip_address,
        request_time=request_time,
    )
    return full_download_url

async def compute_transcript_with_whisper_from_audio_func(audio_file_hash, audio_file_path, audio_file_name, audio_file_size_mb, ip_address, req: Request, corpus_identifier_string: str, embedding_pooling_method: str, compute_embeddings_for_resulting_transcript_document=True, llm_model_name=DEFAULT_MODEL_NAME):
    model_size = "large-v3"
    logger.info(f"Loading Whisper model {model_size}...")
    num_workers = 1 if psutil.virtual_memory().total < 32 * (1024 ** 3) else min(4, max(1, int((psutil.virtual_memory().total - 32 * (1024 ** 3)) / (4 * (1024 ** 3))))) # Only use more than 1 worker if there is at least 32GB of RAM; then use 1 worker per additional 4GB of RAM up to 4 workers max
    with suppress_stdout_stderr():
        gpu_info = is_gpu_available()
        if gpu_info['gpu_found']:
            model = await run_in_threadpool(WhisperModel, model_size, device="cuda", compute_type="auto")
        else:
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
        logger.info(f"Details of transcript segment {idx:,} from file {audio_file_name}: {details}")
        segment_details.append(details)
    combined_transcript_text, combined_transcript_text_list_of_metadata_dicts, list_of_transcript_sentences = merge_transcript_segments_into_combined_text(segment_details)    
    if compute_embeddings_for_resulting_transcript_document:
        download_url = await compute_and_store_transcript_embeddings(
            audio_file_name=audio_file_name,
            sentences=list_of_transcript_sentences,
            llm_model_name=llm_model_name,
            embedding_pooling_method=embedding_pooling_method,
            corpus_identifier_string=corpus_identifier_string,
            ip_address=ip_address,
            combined_transcript_text=combined_transcript_text,
            req=req,
        )
    else:
        download_url = ''
    response_time = datetime.utcnow()
    total_time = (response_time - request_time).total_seconds()
    logger.info(f"Transcript computed in {total_time:,.2f} seconds.")
    await save_transcript_to_db(
        audio_file_hash=audio_file_hash,
        audio_file_name=audio_file_name,
        audio_file_size_mb=audio_file_size_mb,
        transcript_segments=segment_details,
        info=info,
        ip_address=ip_address,
        request_time=request_time,
        response_time=response_time,
        total_time=total_time,
        combined_transcript_text=combined_transcript_text,
        combined_transcript_text_list_of_metadata_dicts=combined_transcript_text_list_of_metadata_dicts,
        corpus_identifier_string=corpus_identifier_string
    )
    info_dict = info._asdict()
    return segment_details, info_dict, combined_transcript_text, combined_transcript_text_list_of_metadata_dicts, request_time, response_time, total_time, download_url
    
async def get_or_compute_transcript(file: UploadFile,
                                    compute_embeddings_for_resulting_transcript_document: bool,
                                    llm_model_name: str,
                                    embedding_pooling_method: str,
                                    corpus_identifier_string: str,
                                    req: Request = None
                                    ) -> AudioTranscriptResponse:
    request_time = datetime.utcnow()
    ip_address = req.client.host if req else "127.0.0.1"
    file_contents = await file.read()
    audio_file_hash = sha3_256(file_contents).hexdigest()
    file.file.seek(0)  # Reset file pointer after read
    unique_id = f"transcript_{audio_file_hash}_{llm_model_name}_{embedding_pooling_method}"
    lock = await shared_resources.lock_manager.lock(unique_id)
    if lock.valid:
        try:
            existing_audio_transcript = await get_transcript_from_db(audio_file_hash)
            if existing_audio_transcript:
                existing_audio_transcript_dict = convert_to_pydantic_response(
                    existing_audio_transcript, 
                    compute_embeddings_for_resulting_transcript_document, 
                    llm_model_name, 
                    embedding_pooling_method
                )
                return AudioTranscriptResponse(**existing_audio_transcript_dict)
            current_position = file.file.tell()
            file.file.seek(0, os.SEEK_END)
            audio_file_size_mb = file.file.tell() / (1024 * 1024)
            file.file.seek(current_position)
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                shutil.copyfileobj(file.file, tmp_file)
                audio_file_name = tmp_file.name
            if corpus_identifier_string == "":
                corpus_identifier_string = audio_file_hash
            (
                segment_details,
                info,
                combined_transcript_text,
                combined_transcript_text_list_of_metadata_dicts,
                request_time,
                response_time,
                total_time,
                download_url,
            ) = await compute_transcript_with_whisper_from_audio_func(
                audio_file_hash=audio_file_hash,
                audio_file_path=audio_file_name,
                audio_file_name=file.filename,
                audio_file_size_mb=audio_file_size_mb,
                ip_address=ip_address,
                req=req,
                corpus_identifier_string=corpus_identifier_string,
                embedding_pooling_method=embedding_pooling_method,
                compute_embeddings_for_resulting_transcript_document=compute_embeddings_for_resulting_transcript_document,
                llm_model_name=llm_model_name,
            )
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
                "llm_model_name": llm_model_name if compute_embeddings_for_resulting_transcript_document else "",
                "embedding_pooling_method": embedding_pooling_method if compute_embeddings_for_resulting_transcript_document else "",
                "corpus_identifier_string": corpus_identifier_string if compute_embeddings_for_resulting_transcript_document else "",
            }
            try:
                os.remove(audio_file_name)
            except Exception as e:  # noqa: F841
                pass
            return AudioTranscriptResponse(**audio_transcript_response)
        finally:
            await shared_resources.lock_manager.unlock(lock)
    else:
        return {"status": "already processing"}
    
def get_audio_duration_seconds(audio_input) -> float:
    if isinstance(audio_input, bytes):
        audio_file = io.BytesIO(audio_input)
        audio = MutagenFile(audio_file)
    elif isinstance(audio_input, str):
        audio = MutagenFile(audio_input)
    else:
        raise ValueError("audio_input must be either bytes or a file path string.")
    if audio is None or not hasattr(audio.info, 'length'):
        raise ValueError("Could not determine the length of the audio file.")
    return audio.info.length

def start_resource_monitoring(endpoint_name: str, input_data: Dict[str, Any], client_ip: str) -> Dict[str, Any]:
    if not USE_RESOURCE_MONITORING:
        return {}
    initial_memory = psutil.virtual_memory().used
    initial_cpu_times = psutil.cpu_times_percent(interval=None)
    start_time = time.time()
    request_details = {}
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
    elif endpoint_name == "ask_question_about_image":
        question = input_data.get("question", "")
        request_details = {
            "question": question,
            "num_words_in_question": len(question.split()),
            "num_characters_in_question": len(question),
            "llm_model_name": input_data.get("llm_model_name", ""),
            "temperature": input_data.get("temperature", 0.7),
            "number_of_tokens_to_generate": input_data.get("number_of_tokens_to_generate", 256),
            "number_of_completions_to_generate": input_data.get("number_of_completions_to_generate", 1),
            "image_filename": input_data.get("image").filename if input_data.get("image") else ""
        }
    elif endpoint_name == "advanced_search_stored_embeddings_with_query_string_for_semantic_similarity":
        request_details = {
            "query_text": input_data.get("query_text", ""),
            "llm_model_name": input_data.get("llm_model_name", ""),
            "embedding_pooling_method": input_data.get("embedding_pooling_method", ""),
            "corpus_identifier_string": input_data.get("corpus_identifier_string", ""),
            "similarity_filter_percentage": input_data.get("similarity_filter_percentage", 0.02),
            "number_of_most_similar_strings_to_return": input_data.get("number_of_most_similar_strings_to_return", 10),
            "result_sorting_metric": input_data.get("result_sorting_metric", "hoeffding_d")
        }        
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
        logger.error(traceback.format_exc())
    logger.info(f"Request data and system resources used: {result}")