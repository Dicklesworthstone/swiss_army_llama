import asyncio
import glob
import json
import logging
import os 
import random
import shutil
import subprocess
import tempfile
import time
import traceback
import urllib.request
import zipfile
from logging.handlers import RotatingFileHandler
from hashlib import sha3_256
from typing import List, Optional
from datetime import datetime
from collections import defaultdict
import numpy as np
from decouple import config
import uvicorn
import psutil
import fastapi
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from langchain.embeddings import LlamaCppEmbeddings
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import Column, String, Float, DateTime, Integer, UniqueConstraint, ForeignKey, LargeBinary, select
from sqlalchemy import text as sql_text
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.exc import SQLAlchemyError, OperationalError, IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, validates
import faiss
import pandas as pd
import PyPDF2
from magic import Magic

# Note: the Ramdisk setup and teardown requires sudo; to enable password-less sudo, edit your sudoers file with `sudo visudo`.
# Add the following lines, replacing username with your actual username
# username ALL=(ALL) NOPASSWD: /bin/mount -t tmpfs -o size=*G tmpfs /mnt/ramdisk
# username ALL=(ALL) NOPASSWD: /bin/umount /mnt/ramdisk

# Setup logging
old_logs_dir = 'old_logs' # Ensure the old_logs directory exists
if not os.path.exists(old_logs_dir):
    os.makedirs(old_logs_dir)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_file_path = 'llama2_embeddings_fastapi_service.log'
fh = RotatingFileHandler(log_file_path, maxBytes=10*1024*1024, backupCount=5)
fh.setFormatter(formatter)
logger.addHandler(fh)
def namer(default_log_name): # Move rotated logs to the old_logs directory
    return os.path.join(old_logs_dir, os.path.basename(default_log_name))
def rotator(source, dest):
    shutil.move(source, dest)
fh.namer = namer
fh.rotator = rotator
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)
logger = logging.getLogger(__name__)

# Global variables
use_hardcoded_security_token = 0
if use_hardcoded_security_token:
    SECURITY_TOKEN = "Test123$"
    USE_SECURITY_TOKEN = config("USE_SECURITY_TOKEN", default=False, cast=bool)
else:
    USE_SECURITY_TOKEN = False
DATABASE_URL = "sqlite+aiosqlite:///embeddings.sqlite"
LLAMA_EMBEDDING_SERVER_LISTEN_PORT = config("LLAMA_EMBEDDING_SERVER_LISTEN_PORT", default=8089, cast=int)
DEFAULT_MODEL_NAME = config("DEFAULT_MODEL_NAME", default="llama2_7b_chat_uncensored", cast=str) 
MINIMUM_STRING_LENGTH_FOR_DOCUMENT_EMBEDDING = config("MINIMUM_STRING_LENGTH_FOR_DOCUMENT_EMBEDDING", default=50, cast=int)
USE_PARALLEL_INFERENCE_QUEUE = config("USE_PARALLEL_INFERENCE_QUEUE", default=False, cast=bool)
MAX_CONCURRENT_PARALLEL_INFERENCE_TASKS = config("MAX_CONCURRENT_PARALLEL_INFERENCE_TASKS", default=10, cast=int)
USE_RAMDISK = config("USE_RAMDISK", default=False, cast=bool)
RAMDISK_SIZE_IN_GB = config("RAMDISK_SIZE_IN_GB", default=1, cast=int)
RAMDISK_PATH = "/mnt/ramdisk"
MAX_RETRIES = config("MAX_RETRIES", default=3, cast=int)
RETRY_DELAY_BASE_SECONDS = config("RETRY_DELAY_BASE_SECONDS", default=1, cast=int)
JITTER_FACTOR = config("JITTER_FACTOR", default=0.1, cast=float)
BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
model_cache = {} # Model cache to store loaded models
faiss_indexes = {}  # Dictionary to hold FAISS indexes by model name
associated_texts = []
download_codes = {} # Dictionary to store download codes and their corresponding file paths
logger.info(f"USE_RAMDISK is set to: {USE_RAMDISK}")

app = FastAPI(docs_url="/")  # Set the Swagger UI to root
engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
Base = declarative_base()

def check_that_user_has_required_permissions_to_manage_ramdisks():
    try: # Try to run a harmless command with sudo to test if the user has password-less sudo permissions
        result = subprocess.run(["sudo", "ls"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if "password" in result.stderr.lower():
            raise PermissionError("Password required for sudo")
        logger.info("User has sufficient permissions to manage RAM Disks.")
        return True
    except (PermissionError, subprocess.CalledProcessError) as e:
        logger.info("Sorry, current user does not have sufficient permissions to manage RAM Disks! Disabling RAM Disks for now...")
        logger.debug(f"Permission check error detail: {e}")
        return False
    
def setup_ramdisk():
    cmd_check = f"sudo mount | grep {RAMDISK_PATH}" # Check if RAM disk already exists at the path
    result = subprocess.run(cmd_check, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
    if RAMDISK_PATH in result:
        logger.info(f"RAM Disk already set up at {RAMDISK_PATH}. Skipping setup.")
        return
    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    free_ram_gb = psutil.virtual_memory().free / (1024 ** 3)
    buffer_gb = 2  # buffer to ensure we don't use all the free RAM
    ramdisk_size_gb = max(min(RAMDISK_SIZE_IN_GB, free_ram_gb - buffer_gb), 0.1)
    ramdisk_size_mb = int(ramdisk_size_gb * 1024)
    ramdisk_size_str = f"{ramdisk_size_mb}M"
    logger.info(f"Total RAM: {total_ram_gb}G")
    logger.info(f"Free RAM: {free_ram_gb}G")
    logger.info(f"Calculated RAM Disk Size: {ramdisk_size_gb}G")
    if RAMDISK_SIZE_IN_GB > total_ram_gb:
        raise ValueError(f"Cannot allocate {RAMDISK_SIZE_IN_GB}G for RAM Disk. Total system RAM is {total_ram_gb:.2f}G.")
    logger.info("Setting up RAM Disk...")
    os.makedirs(RAMDISK_PATH, exist_ok=True)
    mount_command = ["sudo", "mount", "-t", "tmpfs", "-o", f"size={ramdisk_size_str}", "tmpfs", RAMDISK_PATH]
    subprocess.run(mount_command, check=True)
    logger.info(f"RAM Disk set up at {RAMDISK_PATH} with size {ramdisk_size_gb}G")

def copy_models_to_ramdisk(models_directory, ramdisk_directory):
    total_size = sum(os.path.getsize(os.path.join(models_directory, model)) for model in os.listdir(models_directory))
    free_ram = psutil.virtual_memory().free
    if total_size > free_ram:
        logger.warning(f"Not enough space on RAM Disk. Required: {total_size}, Available: {free_ram}. Rebuilding RAM Disk.")
        clear_ramdisk()
        free_ram = psutil.virtual_memory().free  # Recompute the available RAM after clearing the RAM disk
        if total_size > free_ram:
            logger.error(f"Still not enough space on RAM Disk even after clearing. Required: {total_size}, Available: {free_ram}.")
            raise ValueError("Not enough RAM space to copy models.")
        setup_ramdisk()
    os.makedirs(ramdisk_directory, exist_ok=True)
    for model in os.listdir(models_directory):
        shutil.copyfile(os.path.join(models_directory, model), os.path.join(ramdisk_directory, model))
        logger.info(f"Copied model {model} to RAM Disk at {os.path.join(ramdisk_directory, model)}")

def clear_ramdisk():
    while True:
        cmd_check = f"sudo mount | grep {RAMDISK_PATH}"
        result = subprocess.run(cmd_check, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
        if RAMDISK_PATH not in result:
            break  # Exit the loop if the RAMDISK_PATH is not in the mount list
        cmd_umount = f"sudo umount -l {RAMDISK_PATH}"
        subprocess.run(cmd_umount, shell=True, check=True)
    logger.info(f"Cleared RAM Disk at {RAMDISK_PATH}")

async def build_faiss_indexes():
    global faiss_indexes, associated_texts_by_model
    faiss_indexes = {}
    associated_texts_by_model = defaultdict(list) # Create a dictionary to store associated texts by model name
    async with AsyncSessionLocal() as session:
        result = await session.execute(sql_text("SELECT model_name, text, embedding_json FROM embeddings"))
        embeddings_by_model = defaultdict(list)
        for row in result.fetchall():
            model_name = row[0]
            associated_texts_by_model[model_name].append(row[1]) # Store the associated text by model name
            embeddings_by_model[model_name].append((row[1], json.loads(row[2])))
        logger.info(f"Loaded embeddings for {len(embeddings_by_model)} models to compute Faiss indexes.")
        for model_name, embeddings in embeddings_by_model.items():
            logger.info(f"Building Faiss index for model {model_name}...")
            embeddings_array = np.array([e[1] for e in embeddings]).astype('float32')
            if embeddings_array.size == 0:
                logger.error(f"No embeddings were loaded from the database for model {model_name}, so nothing to build the Faiss index with!")
                continue
            logger.info(f"Loaded {len(embeddings_array)} embeddings for model {model_name}.")
            logger.info(f"Embedding dimension for model {model_name}: {embeddings_array.shape[1]}")
            logger.info(f"Normalizing {len(embeddings_array)} embeddings for model {model_name}...")
            faiss.normalize_L2(embeddings_array)  # Normalize the vectors for cosine similarity
            faiss_index = faiss.IndexFlatIP(embeddings_array.shape[1])  # Use IndexFlatIP for cosine similarity
            faiss_index.add(embeddings_array)
            logger.info(f"Faiss index built for model {model_name}.")
            faiss_indexes[model_name] = faiss_index  # Store the index by model name
            

class TextEmbedding(Base):
    __tablename__ = "embeddings"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, index=True)
    text_hash = Column(String, index=True)  # Explicitly store the hash
    model_name = Column(String, index=True)
    embedding_json = Column(String)
    ip_address = Column(String)
    request_time = Column(DateTime)
    response_time = Column(DateTime)
    total_time = Column(Float)
    document_id = Column(Integer, ForeignKey('document_embeddings.id'))
    document = relationship("DocumentEmbedding", back_populates="embeddings")
    __table_args__ = (UniqueConstraint('text_hash', 'model_name', name='_text_hash_model_uc'),)
    @validates('text')
    def update_text_hash(self, key, text):
        # Automatically compute the hash when the text is set
        self.text_hash = sha3_256(text.encode('utf-8')).hexdigest()
        return text
        
class DocumentEmbedding(Base):
    __tablename__ = "document_embeddings"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey('documents.id'))
    filename = Column(String)
    mimetype = Column(String)
    file_hash = Column(String, index=True)
    model_name = Column(String, index=True)    
    file_data = Column(LargeBinary) # To store the original file
    results_json = Column(JSON) # To store the embedding results JSON
    document = relationship("Document", back_populates="document_embeddings")
    embeddings = relationship("TextEmbedding", back_populates="document")

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    document_embeddings = relationship("DocumentEmbedding", back_populates="document")
    
class EmbeddingResponse(BaseModel):
    embedding: List[float]

class SimilarityResponse(BaseModel):
    text1: str
    text2: str
    embedding1: List[float]
    embedding2: List[float]
    similarity: float
    
class SimilarStringResponse(BaseModel):
    results: List[dict]  # List of similar strings and their similarity scores

class AllStringsResponse(BaseModel):
    strings: List[str]

class AllDocumentsResponse(BaseModel):
    strings: List[str]

class EmbeddingRequest(BaseModel):
    text: str
    model_name: str = DEFAULT_MODEL_NAME

class SimilarityRequest(BaseModel):
    text1: str
    text2: str
    model_name: Optional[str] = DEFAULT_MODEL_NAME

class SimilarStringRequest(BaseModel):
    text: str
    number_of_most_similar_strings_to_return: int = 10
    model_name: Optional[str] = DEFAULT_MODEL_NAME

async def execute_with_retry(func, *args, **kwargs):
    retries = 0
    while retries < MAX_RETRIES:
        try:
            return await func(*args, **kwargs)
        except OperationalError as e:
            if 'database is locked' in str(e):
                retries += 1
                # Implementing exponential backoff with jitter
                sleep_time = RETRY_DELAY_BASE_SECONDS * (2 ** retries) + (random.random() * JITTER_FACTOR)
                logger.warning(f"Database is locked. Retrying ({retries}/{MAX_RETRIES})... Waiting for {sleep_time} seconds")
                await asyncio.sleep(sleep_time)
            else:
                raise
    raise OperationalError("Database is locked after multiple retries")

async def initialize_db():
    logger.info("Initializing database...")
    async with engine.begin() as conn: # Set SQLite to use Write-Ahead Logging (WAL) mode
        await conn.execute(sql_text("PRAGMA journal_mode=WAL;"))
        await conn.execute(sql_text("PRAGMA synchronous = NORMAL;")) # Set synchronous mode to NORMAL (from FULL)
        await conn.execute(sql_text("PRAGMA cache_size = -1048576;")) # Set cache size to 1GB
        await conn.execute(sql_text("PRAGMA busy_timeout = 5000;")) # Increase the busy timeout to 5 seconds
        await conn.execute(sql_text("PRAGMA wal_autocheckpoint = 1000;")) # Set automatic checkpoint interval (for example, after every 1000 write transactions)
        await conn.execute(sql_text("PRAGMA foreign_keys = ON;")) # Enable foreign key constraints (if needed)
        await conn.run_sync(Base.metadata.create_all) # Create tables if not exists
    logger.info("Database initialization completed.")

def download_models():
    list_of_model_download_urls = [
        'https://huggingface.co/TheBloke/llama2_7b_chat_uncensored-GGML/resolve/main/llama2_7b_chat_uncensored.ggmlv3.q3_K_L.bin',
        'https://huggingface.co/TheBloke/WizardLM-1.0-Uncensored-Llama2-13B-GGML/resolve/main/wizardlm-1.0-uncensored-llama2-13b.ggmlv3.q3_K_L.bin'
    ]
    model_names = [os.path.basename(url) for url in list_of_model_download_urls]
    current_file_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(current_file_path)
    models_dir = os.path.join(base_dir, 'models')
    logger.info("Checking models directory...")
    if USE_RAMDISK:
        ramdisk_models_dir = os.path.join(RAMDISK_PATH, 'models')
        if not os.path.exists(RAMDISK_PATH): # Check if RAM disk exists, and set it up if not
            setup_ramdisk()
        if all(os.path.exists(os.path.join(ramdisk_models_dir, model_name)) for model_name in model_names): # Check if models already exist in RAM disk
            logger.info("Models found in RAM Disk.")
            return model_names
    if not os.path.exists(models_dir): # Check if models directory exists, and create it if not
        os.makedirs(models_dir)
        logger.info(f"Created models directory: {models_dir}")
    else:
        logger.info(f"Models directory exists: {models_dir}")
    for url, model_name_with_extension in zip(list_of_model_download_urls, model_names): # Check if models are in regular disk, download if not
        filename = os.path.join(models_dir, model_name_with_extension)
        if not os.path.exists(filename):
            logger.info(f"Downloading model {model_name_with_extension} from {url}...")
            urllib.request.urlretrieve(url, filename)
            logger.info(f"Downloaded: {filename}")
        else:
            logger.info(f"File already exists: {filename}")
    if USE_RAMDISK: # If RAM disk is enabled, copy models from regular disk to RAM disk
        copy_models_to_ramdisk(models_dir, ramdisk_models_dir)
    logger.info("Model downloads completed.")
    return model_names

async def get_embedding_from_db(text, model_name):
    logger.info(f"Retrieving embedding for '{text}' using model '{model_name}' from database...")
    text_hash = sha3_256(text.encode('utf-8')).hexdigest() # Compute the hash
    return await execute_with_retry(_get_embedding_from_db, text_hash, model_name)

async def _get_embedding_from_db(text_hash, model_name):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            sql_text("SELECT embedding_json FROM embeddings WHERE text_hash=:text_hash AND model_name=:model_name"),
            {"text_hash": text_hash, "model_name": model_name},
        )
        row = result.fetchone()
        if row:
            embedding_json = row[0]
            logger.info(f"Embedding found in database for text hash '{text_hash}' using model '{model_name}'")
            return json.loads(embedding_json)
        return None
    
async def get_or_compute_embedding(request: EmbeddingRequest, req: Request = None, client_ip: str = None) -> dict:
    request_time = datetime.utcnow()  # Capture request time as datetime object
    embedding_list = await get_embedding_from_db(request.text, request.model_name) # Check if embedding exists in the database
    if embedding_list is not None:
        return {"embedding": embedding_list}
    model = load_model(request.model_name)
    embedding_list = calculate_sentence_embedding(model, request.text) # Compute the embedding if not in the database
    if embedding_list is None:
        logger.error("Could not calculate the embedding for the given text")
        raise HTTPException(status_code=400, detail="Could not calculate the embedding for the given text")
    embedding_json = json.dumps(embedding_list) # Serialize the numpy array to JSON and save to the database
    response_time = datetime.utcnow()  # Capture response time as datetime object
    total_time = (response_time - request_time).total_seconds() # Calculate total time using datetime objects
    # If client_ip is provided, use it; otherwise, try to get from req; if not available, default to "localhost"
    ip_address = client_ip or (req.client.host if req else "localhost")
    await save_embedding_to_db(request.text, request.model_name, embedding_json, ip_address, request_time, response_time, total_time)
    return {"embedding": embedding_list}

async def save_embedding_to_db(text, model_name, embedding_json, ip_address, request_time, response_time, total_time):
    existing_embedding = await get_embedding_from_db(text, model_name) # Check if the embedding already exists
    if existing_embedding is not None:
        return existing_embedding
    logger.info(f"Saving embedding for '{text}' using model '{model_name}' to database...")
    return await execute_with_retry(_save_embedding_to_db, text, model_name, embedding_json, ip_address, request_time, response_time, total_time)

async def _save_embedding_to_db(text, model_name, embedding_json, ip_address, request_time, response_time, total_time):
    text_hash = sha3_256(text.encode('utf-8')).hexdigest()
    async with AsyncSessionLocal() as session: # Check if the embedding already exists by using the hash
        existing_embedding = await session.execute(select(TextEmbedding).filter(TextEmbedding.text_hash == text_hash, TextEmbedding.model_name == model_name))
        existing_embedding = existing_embedding.scalar_one_or_none()
        if existing_embedding:
            return existing_embedding
        embedding = TextEmbedding(
            text=text,
            model_name=model_name,
            embedding_json=embedding_json,
            ip_address=ip_address,
            request_time=request_time,
            response_time=response_time,
            total_time=total_time,
        )
        try:
            session.add(embedding)
            await session.commit()
            logger.info(f"Saved embedding for '{text}' using model '{model_name}' to database successfully.")
        except Exception as e:
            logger.error(f"Error saving embedding to database: {e}")
            await session.rollback()
            raise

def load_model(model_name: str, raise_http_exception: bool = True):
    try:
        models_dir = os.path.join(RAMDISK_PATH, 'models') if USE_RAMDISK else os.path.join(BASE_DIRECTORY, 'models')
        if model_name in model_cache:
            return model_cache[model_name]
        matching_files = glob.glob(os.path.join(models_dir, f"{model_name}*"))
        if not matching_files:
            logger.error(f"No model file found matching: {model_name}")
            raise FileNotFoundError
        matching_files.sort(key=os.path.getmtime, reverse=True)
        model_file_path = matching_files[0]
        model_instance = LlamaCppEmbeddings(model_path=model_file_path, use_mlock=True)
        model_instance.client.verbose = False
        model_cache[model_name] = model_instance
        return model_instance
    except TypeError as e:
        logger.error(f"TypeError occurred while loading the model: {e}")
        raise
    except Exception as e:
        logger.error(f"Exception occurred while loading the model: {e}")
        if raise_http_exception:
            raise HTTPException(status_code=404, detail="Model file not found")
        else:
            raise FileNotFoundError(f"No model file found matching: {model_name}")

def calculate_sentence_embedding(llama, text: str) -> np.array:
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

async def compute_embeddings_for_document(strings, model_name, client_ip):
    results = []
    if USE_PARALLEL_INFERENCE_QUEUE:
        logger.info(f"Using parallel inference queue to compute embeddings for {len(strings)} strings")
        start_time = time.perf_counter()  # Record the start time
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_PARALLEL_INFERENCE_TASKS)
        async def compute_embedding(text):  # Define a function to compute the embedding for a given text
            try:
                async with semaphore:  # Acquire a semaphore slot
                    request = EmbeddingRequest(text=text, model_name=model_name)
                    embedding = await get_embedding_vector(request, client_ip=client_ip)
                    return text, embedding["embedding"]
            except Exception as e:
                logger.error(f"Error computing embedding for text '{text}': {e}")
                return text, None
        results = await asyncio.gather(*[compute_embedding(s) for s in strings])  # Use asyncio.gather to run the tasks concurrently
        end_time = time.perf_counter()  # Record the end time
        duration = end_time - start_time
        logger.info(f"Parallel inference task for {len(strings)} strings completed in {duration:.2f} seconds")
    else:  # Compute embeddings sequentially
        logger.info(f"Using sequential inference to compute embeddings for {len(strings)} strings")
        start_time = time.perf_counter()  # Record the start time
        for s in strings:
            embedding_request = EmbeddingRequest(text=s, model_name=model_name)
            embedding = await get_embedding_vector(embedding_request, client_ip=client_ip)
            results.append((s, embedding["embedding"]))
        end_time = time.perf_counter()  # Record the end time
        duration = end_time - start_time
        logger.info(f"Sequential inference task for {len(strings)} strings completed in {duration:.2f} seconds")
    filtered_results = [(text, embedding) for text, embedding in results if embedding is not None] # Filter out results with None embeddings (applicable to parallel processing) and return
    return filtered_results

async def store_document_embeddings_in_db(file, file_hash, original_file_content, json_content, results, model_name):
    results_json_object = json.loads(json_content.decode())
    async with AsyncSessionLocal() as session:
        document = Document()
        session.add(document)
        await session.flush()
        document_embedding = DocumentEmbedding(document_id=document.id, filename=file.filename, mimetype=file.content_type, file_hash=file_hash, file_data=original_file_content, results_json=results_json_object)        
        session.add(document_embedding)
        await session.flush()
        for text, embedding in results:
            embedding_entry = await _get_embedding_from_db(text, model_name)
            if not embedding_entry:
                embedding_entry = TextEmbedding(text=text, model_name=model_name, embedding_json=json.dumps(embedding), document_id=document_embedding.id)
                session.add(embedding_entry)
        await session.commit()
        
@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
    logger.exception(exc)
    return JSONResponse(status_code=500, content={"message": "Database error occurred"})

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception(exc)
    return JSONResponse(status_code=500, content={"message": "An unexpected error occurred"})

@app.get("/", include_in_schema=False)
async def custom_swagger_ui_html():
    return fastapi.templating.get_swagger_ui_html(openapi_url="/openapi.json", title=app.title, swagger_favicon_url=app.swagger_ui_favicon_url)



@app.get("/get_list_of_available_model_names/",
         summary="Retrieve Available Model Names",
         description="""Retrieve the list of available model names for generating embeddings.

### Parameters:
- `token`: Security token (optional).

### Response:
The response will include a JSON object containing the list of available model names. Note that these are all GGML format models designed to work with llama_cpp.

### Example Response:
```json
{
  "model_names": ["llama2_7b_chat_uncensored", "wizardlm-1.0-uncensored-llama2-13b", "my_super_custom_model"]
}
```""",
         response_description="A JSON object containing the list of available model names.")
# @app.get("/get_list_of_available_model_names/")
async def get_list_of_available_model_names(token: str = None):
    if USE_SECURITY_TOKEN and (token is None or token != SECURITY_TOKEN):
        raise HTTPException(status_code=403, detail="Unauthorized")
    models_dir = os.path.join(RAMDISK_PATH, 'models') if USE_RAMDISK else os.path.join(BASE_DIRECTORY, 'models')
    logger.info(f"Looking for models in: {models_dir}") # Add this line for debugging
    logger.info(f"Directory content: {os.listdir(models_dir)}") # Add this line for debugging
    model_files = glob.glob(os.path.join(models_dir, "*.bin")) # Find all files with .ggmlv3.q3_K_L.bin extension
    model_names = [os.path.splitext(os.path.splitext(os.path.basename(model_file))[0])[0] for model_file in model_files] # Remove both extensions
    return {"model_names": model_names}



@app.get("/get_all_stored_strings/",
         summary="Retrieve All Strings",
         description="""Retrieve a list of all stored strings from the database for which embeddings have been computed.

### Parameters:
- `token`: Security token (optional).

### Response:
The response will include a JSON object containing the list of all stored strings with computed embeddings.

### Example Response:
```json
{
  "strings": ["The quick brown fox jumps over the lazy dog", "To be or not to be", "Hello, World!"]
}
```""",
         response_description="A JSON object containing the list of all strings with computed embeddings.")
#@app.get("/get_all_stored_strings/", response_model=AllStringsResponse)
async def get_all_stored_strings(req: Request, token: str = None) -> AllStringsResponse:
    logger.info("Received request to retrieve all stored strings for which embeddings have been computed")
    if USE_SECURITY_TOKEN and use_hardcoded_security_token and (token is None or token != SECURITY_TOKEN):
        raise HTTPException(status_code=403, detail="Unauthorized")
    try:
        logger.info("Retrieving all stored strings with computed embeddings from the database")
        async with AsyncSessionLocal() as session:
            result = await session.execute(sql_text("SELECT DISTINCT text FROM embeddings"))
            all_strings = [row[0] for row in result.fetchall()]
        logger.info(f"Retrieved {len(all_strings)} stored strings with computed embeddings from the database")
        return {"strings": all_strings}
    except Exception as e:
        logger.error(f"An error occurred while processing the request: {e}")
        logger.error(traceback.format_exc())  # Print the traceback
        raise HTTPException(status_code=500, detail="Internal Server Error")



@app.get("/get_all_stored_documents_with_embeddings/",
         summary="Retrieve All Documents with Embeddings",
         description="""Retrieve a list of all stored documents from the database for which embeddings have been computed.

### Parameters:
- `token`: Security token (optional).

### Response:
The response will include a JSON object containing the list of all stored documents with computed embeddings.

### Example Response:
```json
{
  "documents": ["document1.pdf", "document2.txt", "document3.md", "document4.json"]
}
```""",
         response_description="A JSON object containing the list of all documents with computed embeddings.")
#@app.get("/get_all_stored_documents_with_embeddings/", response_model=AllDocumentsResponse)
async def get_all_stored_documents_with_embeddings(req: Request, token: str = None) -> AllDocumentsResponse:
    logger.info("Received request to retrieve all stored documents with computed embeddings")
    if USE_SECURITY_TOKEN and use_hardcoded_security_token and (token is None or token != SECURITY_TOKEN):
        raise HTTPException(status_code=403, detail="Unauthorized")
    try:
        logger.info("Retrieving all stored documents with computed embeddings from the database")
        async with AsyncSessionLocal() as session:
            result = await session.execute(sql_text("SELECT DISTINCT filename FROM document_embeddings"))
            all_documents = [row[0] for row in result.fetchall()]
        logger.info(f"Retrieved {len(all_documents)} stored documents with computed embeddings from the database")
        return {"documents": all_documents}
    except Exception as e:
        logger.error(f"An error occurred while processing the request: {e}")
        logger.error(traceback.format_exc())  # Print the traceback
        raise HTTPException(status_code=500, detail="Internal Server Error")



@app.post("/get_embedding_vector/",
          response_model=EmbeddingResponse,
          summary="Retrieve Embedding Vector for a Given Text",
          description="""Retrieve the embedding vector for a given input text using the specified model.

### Parameters:
- `request`: A JSON object containing the text and the model name.
- `token`: Security token (optional).

### Request JSON Format:
The request must contain the following attributes:
- `text`: The input text for which the embedding vector is to be retrieved.
- `model_name`: The model used to calculate the embedding (optional, will use the default model if not provided).

### Example (note that `model_name` is optional):
```json
{
  "text": "This is a sample text.",
  "model_name": "llama2_7b_chat_uncensored"
}
```

### Response:
The response will include the embedding vector for the input text.

### Example Response:
```json
{
  "embedding": [0.1234, 0.5678, ...]
}
```""", response_description="A JSON object containing the embedding vector for the input text.")
# @app.post("/get_embedding_vector/", response_model=EmbeddingResponse)
async def get_embedding_vector(request: EmbeddingRequest, req: Request = None, token: str = None, client_ip: str = None) -> EmbeddingResponse:
    if USE_SECURITY_TOKEN and use_hardcoded_security_token and (token is None or token != SECURITY_TOKEN):
        raise HTTPException(status_code=403, detail="Unauthorized")
    try:
        return await get_or_compute_embedding(request, req, client_ip)
    except Exception as e:
        logger.error(f"An error occurred while processing the request: {e}")
        logger.error(traceback.format_exc()) # Print the traceback
        raise HTTPException(status_code=500, detail="Internal Server Error")



@app.post("/compute_similarity_between_strings/",
          response_model=SimilarityResponse,
          summary="Compute Similarity Between Two Strings",
          description="""Compute the cosine similarity between two given input strings using specified model embeddings.

### Parameters:
- `request`: A JSON object containing the two strings and the model name.
- `token`: Security token (optional).

### Request JSON Format:
The request must contain the following attributes:
- `text1`: The first input text.
- `text2`: The second input text.
- `model_name`: The model used to calculate embeddings (optional).

### Example (note that `model_name` is optional):
```json
{
  "text1": "This is a sample text.",
  "text2": "This is another sample text.",
  "model_name": "llama2_7b_chat_uncensored"
}
```

### Response:
The response will include the similarity score, as well as the embeddings for both input strings.

### Example Response:
```json
{
  "text1": "This is a sample text.",
  "text2": "This is another sample text.",
  "similarity": 0.9521,
  "embedding1": [0.1234, 0.5678, ...],
  "embedding2": [0.9101, 0.1121, ...]
}
```""", response_description="A JSON object containing the similarity score and embeddings for both input strings.")
# @app.post("/compute_similarity_between_strings/", response_model=SimilarityResponse)
async def compute_similarity_between_strings(request: SimilarityRequest, req: Request, token: str = None) -> SimilarityResponse:
    logger.info(f"Received request: {request}")
    if USE_SECURITY_TOKEN and use_hardcoded_security_token and (token is None or token != SECURITY_TOKEN):
        raise HTTPException(status_code=403, detail="Unauthorized")
    try:
        client_ip = req.client.host if req else "localhost"
        logger.info("Computing similarity between strings")
        embedding_request1 = EmbeddingRequest(text=request.text1, model_name=request.model_name)
        embedding_request2 = EmbeddingRequest(text=request.text2, model_name=request.model_name)
        logger.info(f"Requesting embeddings for: {embedding_request1} and {embedding_request2}")
        embedding1_response = await get_or_compute_embedding(embedding_request1, client_ip=client_ip)
        embedding2_response = await get_or_compute_embedding(embedding_request2, client_ip=client_ip)
        logger.info(f"Received embeddings: {embedding1_response} and {embedding2_response}")
        embedding1 = np.array(embedding1_response["embedding"])
        embedding2 = np.array(embedding2_response["embedding"])
        logger.info(f"Embedding1 size: {embedding1.size}, embedding2 size: {embedding2.size}")
        if embedding1.size == 0 or embedding2.size == 0:
            raise HTTPException(status_code=400, detail="Could not calculate embeddings for the given texts")
        similarity = cosine_similarity([embedding1], [embedding2])
        logger.info(f"Cosine Similarity: {similarity}")
        return {
            "text1": request.text1,
            "text2": request.text2,
            "similarity": similarity[0][0],
            "embedding1": embedding1.tolist(),
            "embedding2": embedding2.tolist()
        }
    except Exception as e:
        logger.error(f"An error occurred while processing the request: {e}")
        traceback.print_exc() # Print the traceback to see where the error occurred
        raise HTTPException(status_code=500, detail="Internal Server Error")



@app.post("/get_most_similar_string_from_database/",
          response_model=SimilarStringResponse,
          summary="Get Most Similar Strings from Database",
          description="""Find the most similar strings in the database to the given input text. This endpoint uses a pre-computed FAISS index to quickly search for the closest matching strings.
          # Additional details and examples here...
          """,
          response_description="A JSON object containing the most similar strings and similarity scores.")



@app.post("/get_most_similar_string_from_database/",
          response_model=SimilarStringResponse,
          summary="Get Most Similar Strings from Database",
          description="""Find the most similar strings in the database to the given input text. This endpoint uses a pre-computed FAISS index to quickly search for the closest matching strings.

### Parameters:
- `request`: A JSON object containing the input text, model name, and an optional number of most similar strings to return.
- `req`: HTTP request object (internal use).
- `token`: Security token (optional).

### Request JSON Format:
The request must contain the following attributes:
- `text`: The input text for which to find the most similar string.
- `model_name`: The model used to calculate embeddings.
- `number_of_most_similar_strings_to_return`: (Optional) The number of most similar strings to return, defaults to 3.

### Example:
```json
{
  "text": "Find me the most similar string!",
  "model_name": "llama2_7b_chat_uncensored",
  "number_of_most_similar_strings_to_return": 5
}
```

### Response:
The response will include the most similar strings found in the database, along with the similarity scores.

### Example Response:
```json
{
  "results": [
    {"text": "This is the most similar string!", "similarity": 0.9823},
    {"text": "Another similar string.", "similarity": 0.9721},
    ...
  ]
}
```""",
          response_description="A JSON object containing the most similar strings and similarity scores.")
async def get_most_similar_string_from_database(request: SimilarStringRequest, req: Request, token: str = None) -> SimilarStringResponse:
    global faiss_indexes, associated_texts_by_model
    model_name = request.model_name
    num_results = request.number_of_most_similar_strings_to_return
    logger.info(f"Received request to find {num_results} most similar strings for: {request.text} using model: {model_name}")
    if USE_SECURITY_TOKEN and use_hardcoded_security_token and (token is None or token != SECURITY_TOKEN):
        raise HTTPException(status_code=403, detail="Unauthorized")
    try:
        logger.info(f"Computing embedding for input text: {request.text}")
        embedding_request = EmbeddingRequest(text=request.text, model_name=request.model_name)
        embedding_response = await get_embedding_vector(embedding_request, req)
        input_embedding = np.array(embedding_response["embedding"]).astype('float32').reshape(1, -1)
        faiss.normalize_L2(input_embedding)  # Normalize the input vector for cosine similarity
        logger.info(f"Computed embedding for input text: {request.text}")
        faiss_index = faiss_indexes.get(model_name)  # Retrieve the correct FAISS index for the model_name
        if faiss_index is None:
            raise HTTPException(status_code=400, detail=f"No FAISS index found for model: {model_name}")
        logger.info("Searching for the most similar string in the FAISS index")
        similarities, indices = faiss_index.search(input_embedding.reshape(1, -1), num_results)  # Search for num_results similar strings
        results = []
        for i in range(num_results):
            similarity = float(similarities[0][i])  # Convert numpy.float32 to native float
            most_similar_text = associated_texts_by_model[model_name][indices[0][i]]
            results.append({"text": most_similar_text, "similarity": similarity})
        logger.info(f"Found most similar strings: {results}")
        return {"results": results}
    except Exception as e:
        logger.error(f"An error occurred while processing the request: {e}")
        logger.error(traceback.format_exc())  # Print the traceback
        raise HTTPException(status_code=500, detail="Internal Server Error")



@app.post("/get_all_embeddings_for_document/",
          summary="Get Embeddings for a Document",
          description="""Extract text embeddings for a document. This endpoint supports both plain text and PDF files. Please note that PDFs requiring OCR are not currently supported.

### Parameters:
- `file`: The uploaded document file (either plain text or PDF).
- `model_name`: The model used to calculate embeddings (optional).
- `json_format`: The format of the JSON response (optional, see details below).
- `token`: Security token (optional).

### JSON Format Options:
The format of the JSON string:

- ‘split’ : dict like {‘index’ -> [index], ‘columns’ -> [columns], ‘data’ -> [values]}
- ‘records’ : list like [{column -> value}, … , {column -> value}]
- ‘index’ : dict like {index -> {column -> value}}
- ‘columns’ : dict like {column -> {index -> value}}
- ‘values’ : just the values array
- ‘table’ : dict like {‘schema’: {schema}, ‘data’: {data}}

### Examples:
- Plain Text: Submit a file containing plain text.
- PDF: Submit a `.pdf` file (OCR not supported).""",
          response_description="A ZIP file containing the embeddings file.")
async def get_all_embeddings_for_document(file: UploadFile = File(...), model_name: str = DEFAULT_MODEL_NAME, json_format: str = 'records', token: str = None, background_tasks: BackgroundTasks = None, req: Request = None):
    client_ip = req.client.host if req else "localhost"
    if USE_SECURITY_TOKEN and use_hardcoded_security_token and (token is None or token != SECURITY_TOKEN): raise HTTPException(status_code=403, detail="Unauthorized")
    temp_file_path = tempfile.mktemp() # Write uploaded file to a temporary file
    with open(temp_file_path, 'wb') as buffer:
        chunk_size = 1024
        chunk = await file.read(chunk_size)
        while chunk:
            buffer.write(chunk)
            chunk = await file.read(chunk_size)
    hash_obj = sha3_256() # Calculate SHA3-256 hash of the file
    with open(temp_file_path, 'rb') as buffer:
        for chunk in iter(lambda: buffer.read(chunk_size), b''):
            hash_obj.update(chunk)
    file_hash = hash_obj.hexdigest()
    logger.info(f"SHA3-256 hash of submitted file: {file_hash}")
    async with AsyncSessionLocal() as session: # Check if the document has been processed before
        result = await session.execute(select(DocumentEmbedding).filter(DocumentEmbedding.file_hash == file_hash, DocumentEmbedding.model_name == model_name))
        existing_document_embedding = result.scalar_one_or_none()
        if existing_document_embedding: # If the document has been processed before, return the existing result
            logger.info(f"Document {file.filename} has been processed before, returning existing result")
            json_content = json.dumps(existing_document_embedding.results_json).encode()
        else: # If the document has not been processed, continue processing
            mime = Magic(mime=True) # Determine file type using magic
            mime_type = mime.from_file(temp_file_path)
            logger.info(f"Received request to extract embeddings for document {file.filename} with MIME type: {mime_type} and size: {os.path.getsize(temp_file_path)} bytes from IP address: {client_ip}")
            strings = []
            if mime_type == 'application/pdf':
                logger.info("Processing PDF file")
                with open(temp_file_path, 'rb') as buffer:
                    pdf_reader = PyPDF2.PdfReader(buffer)
                    content = ""
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        page_content = page.extract_text()
                        content += page_content
                        hash_obj.update(page_content.encode())
                    file_hash = hash_obj.hexdigest()
                    strings = [s.strip() for s in content.replace(". ", ".\n").split('\n') if len(s.strip()) > MINIMUM_STRING_LENGTH_FOR_DOCUMENT_EMBEDDING]
                    logger.info(f"Extracted {len(strings)} strings from PDF file")
            elif mime_type.startswith('text/'):
                logger.info("Processing plain text file")
                with open(temp_file_path, 'r') as buffer:
                    for line in buffer:
                        hash_obj.update(line.encode())
                        line = line.strip()
                        if len(line) > MINIMUM_STRING_LENGTH_FOR_DOCUMENT_EMBEDDING:
                            strings.append(line)
                    logger.info(f"Extracted {len(strings)} strings from plain text file")
                file_hash = hash_obj.hexdigest()
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type")
            results = await compute_embeddings_for_document(strings, model_name, client_ip) # Compute the embeddings and json_content for new documents
            df = pd.DataFrame(results, columns=['text', 'embedding'])
            json_content = df.to_json(orient=json_format or 'records').encode()
            with open(temp_file_path, 'rb') as file_buffer: # Store the results in the database
                original_file_content = file_buffer.read()
            await store_document_embeddings_in_db(file, file_hash, original_file_content, json_content, results, model_name)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json") # Create a temporary JSON file
    temp_file.write(json_content)
    temp_file.close()
    zip_file_path = f"/tmp/{file.filename}.zip"  # Create a ZIP file containing the JSON file
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        zipf.write(temp_file.name, os.path.basename(temp_file.name))
    background_tasks.add_task(os.remove, temp_file.name) # Schedule the temp files for deletion after the response is sent
    background_tasks.add_task(os.remove, temp_file_path) # Return the ZIP file as a download:
    return FileResponse(zip_file_path, headers={"Content-Disposition": f"attachment; filename={file.filename}.zip"})


@app.post("/clear_ramdisk/")
async def clear_ramdisk_endpoint(token: str = None):
    if USE_SECURITY_TOKEN and use_hardcoded_security_token and (token is None or token != SECURITY_TOKEN):
        raise HTTPException(status_code=403, detail="Unauthorized")
    if USE_RAMDISK:
        clear_ramdisk()
        return {"message": "RAM Disk cleared successfully."}
    return {"message": "RAM Disk usage is disabled."}


@app.on_event("startup")
async def startup_event():
    global USE_RAMDISK
    if USE_RAMDISK and not check_that_user_has_required_permissions_to_manage_ramdisks():
        USE_RAMDISK = False
    elif USE_RAMDISK:
        setup_ramdisk()    
    list_of_downloaded_model_names = download_models()
    for model_name in list_of_downloaded_model_names:
        try:
            load_model(model_name, raise_http_exception=False)
        except FileNotFoundError as e:
            logger.error(e)
    await initialize_db()
    await build_faiss_indexes() 
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=LLAMA_EMBEDDING_SERVER_LISTEN_PORT)
