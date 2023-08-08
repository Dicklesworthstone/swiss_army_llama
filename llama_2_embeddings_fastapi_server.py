import json
import logging
import os 
import glob
import urllib.request
import asyncio
import shutil
import subprocess
import psutil
from typing import List
from datetime import datetime
import numpy as np
from decouple import config
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from langchain.embeddings import LlamaCppEmbeddings
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import Column, String, Float, DateTime
from sqlalchemy import Integer, UniqueConstraint
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError, OperationalError
import fastapi
import uvicorn

# Note: the Ramdisk setup and teardown requires sudo; to enable password-less sudo, edit your sudoers file with `sudo visudo`.
# Add the following lines, replacing username with your actual username
# username ALL=(ALL) NOPASSWD: /bin/mount -t tmpfs -o size=*G tmpfs /mnt/ramdisk
# username ALL=(ALL) NOPASSWD: /bin/umount /mnt/ramdisk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
USE_SECURITY_TOKEN = config("USE_SECURITY_TOKEN", default=False, cast=bool)
use_hardcoded_security_token = 1
if use_hardcoded_security_token:
    SECURITY_TOKEN = "Test123$"
DATABASE_URL = "sqlite+aiosqlite:///embeddings.db"
LLAMA_EMBEDDING_SERVER_LISTEN_PORT = config("LLAMA_EMBEDDING_SERVER_LISTEN_PORT", default=8089, cast=int)
USE_RAMDISK = config("USE_RAMDISK", default=False, cast=bool)
RAMDISK_SIZE_IN_GB = config("RAMDISK_SIZE_IN_GB", default=1, cast=int)
RAMDISK_PATH = "/mnt/ramdisk"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1
model_cache = {} # Model cache to store loaded models

def setup_ramdisk():
    # Check if there's enough available RAM
    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)  # Total RAM in GB
    available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)  # Available RAM in GB
    if RAMDISK_SIZE_IN_GB > total_ram_gb:
        raise ValueError(f"Cannot allocate {RAMDISK_SIZE_IN_GB}G for RAM Disk. Total system RAM is {total_ram_gb:.2f}G.")
    elif RAMDISK_SIZE_IN_GB > available_ram_gb:
        raise ValueError(f"Cannot allocate {RAMDISK_SIZE_IN_GB}G for RAM Disk. Only {available_ram_gb:.2f}G is available.")
    logger.info("Setting up RAM Disk...")
    os.makedirs(RAMDISK_PATH, exist_ok=True)
    try:
        subprocess.run(f"sudo mount -t tmpfs -o size={RAMDISK_SIZE_IN_GB}G tmpfs {RAMDISK_PATH}", shell=True, check=True)
        logger.info(f"RAM Disk set up at {RAMDISK_PATH} with size {RAMDISK_SIZE_IN_GB}G")
    except subprocess.CalledProcessError:
        logger.error("Failed to set up RAM Disk.")
        raise

def clear_ramdisk():
    logger.info("Clearing RAM Disk...")
    try:
        subprocess.run(f"sudo umount {RAMDISK_PATH}", shell=True, check=True)
        logger.info("RAM Disk cleared.")
    except subprocess.CalledProcessError:
        logger.error("Failed to clear RAM Disk.")
        raise

class EmbeddingRequest(BaseModel):
    text: str
    model_name: str

app = FastAPI(docs_url="/")  # Set the Swagger UI to root

engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

Base = declarative_base()

class EmbeddingResponse(BaseModel):
    embedding: List[float]

class SimilarityResponse(BaseModel):
    text1: str
    text2: str
    embedding1: List[float]
    embedding2: List[float]
    similarity: float

class TextEmbedding(Base):
    __tablename__ = "embeddings"
    id = Column(Integer, primary_key=True, index=True)  
    text = Column(String, index=True) 
    model_name = Column(String, index=True) 
    embedding_json = Column(String)
    ip_address = Column(String)
    request_time = Column(DateTime)
    response_time = Column(DateTime)
    total_time = Column(Float)
    __table_args__ = (UniqueConstraint('text', 'model_name', name='_text_model_uc'),) # Unique constraint on text and model_name

class SimilarityRequest(BaseModel):
    text1: str
    text2: str
    model_name: str

async def execute_with_retry(func, *args, **kwargs):
    retries = 0
    while retries < MAX_RETRIES:
        try:
            return await func(*args, **kwargs)
        except OperationalError as e:
            if 'database is locked' in str(e):
                retries += 1
                logger.warning(f"Database is locked. Retrying ({retries}/{MAX_RETRIES})...")
                await asyncio.sleep(RETRY_DELAY_SECONDS)
            else:
                raise
    raise OperationalError("Database is locked after multiple retries")

async def initialize_db():
    logger.info("Initializing database...")
    async with engine.begin() as conn:
        await conn.execute("PRAGMA journal_mode=WAL;") # Set SQLite to use Write-Ahead Logging (WAL) mode
        await conn.execute("PRAGMA busy_timeout = 2000;") # Increase the busy timeout (for example, to 2 seconds)
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialization completed.")

def download_models():
    list_of_model_download_urls = [
        'https://huggingface.co/TheBloke/llama2_7b_chat_uncensored-GGML/resolve/main/llama2_7b_chat_uncensored.ggmlv3.q3_K_L.bin',
        'https://huggingface.co/TheBloke/WizardLM-1.0-Uncensored-Llama2-13B-GGML/resolve/main/wizardlm-1.0-uncensored-llama2-13b.ggmlv3.q3_K_L.bin'
    ]
    downloaded_model_names = []
    current_file_path = os.path.abspath(__file__) # Ensuring the directory is consistent regardless of how the module is used
    base_dir = os.path.dirname(current_file_path)
    models_dir = os.path.join(base_dir, 'models')
    logger.info("Checking models directory...")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        logger.info(f"Created models directory: {models_dir}")
    else:
        logger.info(f"Models directory exists: {models_dir}")
    logger.info("Starting model downloads...")
    for url in list_of_model_download_urls:
        filename = os.path.join(models_dir, os.path.basename(url))
        model_name = os.path.splitext(os.path.basename(url))[0]
        if not os.path.exists(filename): # Download the file only if it doesn't exist
            logger.info(f"Downloading model {model_name} from {url}...")
            urllib.request.urlretrieve(url, filename)
            downloaded_model_names.append(model_name)
            logger.info(f"Downloaded: {filename}")
        else:
            logger.info(f"File already exists: {filename}")
    if USE_RAMDISK: # If RAM Disk is enabled, copy models to RAM Disk only if not present
        ramdisk_models_dir = os.path.join(RAMDISK_PATH, 'models')
        os.makedirs(ramdisk_models_dir, exist_ok=True)
        for model_name in downloaded_model_names:
            filename = os.path.join(models_dir, f"{model_name}.bin")
            ramdisk_filename = os.path.join(ramdisk_models_dir, f"{model_name}.bin")
            if not os.path.exists(ramdisk_filename):
                shutil.copyfile(filename, ramdisk_filename)
                logger.info(f"Copied model {model_name} to RAM Disk at {ramdisk_filename}")
            else:
                logger.info(f"Model {model_name} already exists in RAM Disk at {ramdisk_filename}")
    logger.info("Model downloads completed.")
    return downloaded_model_names

async def get_embedding_from_db(text, model_name):
    logger.info(f"Retrieving embedding for '{text}' using model '{model_name}' from database...")
    return await execute_with_retry(_get_embedding_from_db, text, model_name)

async def _get_embedding_from_db(text, model_name):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            "SELECT embedding_json FROM embeddings WHERE text=:text AND model_name=:model_name",
            {"text": text, "model_name": model_name},
        )
        row = result.fetchone()
        if row:
            logger.info(f"Embedding found in database for '{text}' using model '{model_name}'")
            return json.loads(row[0])
        logger.info(f"Embedding not found in database for '{text}' using model '{model_name}'")
        return None
    
async def get_or_compute_embedding(request: EmbeddingRequest, req: Request) -> dict:
    request_time = datetime.utcnow()  # Capture request time as datetime object
    embedding_list = await get_embedding_from_db(request.text, request.model_name) # Check if embedding exists in the database
    if embedding_list is not None:
        return {"embedding": embedding_list}
    model = load_model(request.model_name)
    embedding = calculate_sentence_embedding(model, request.text) # Compute the embedding if not in the database
    if embedding is None:
        logger.error("Could not calculate the embedding for the given text")
        raise HTTPException(status_code=400, detail="Could not calculate the embedding for the given text")
    embedding_list = embedding.tolist() # Serialize the numpy array to JSON and save to the database
    embedding_json = json.dumps(embedding_list)
    response_time = datetime.utcnow()  # Capture response time as datetime object
    total_time = (response_time - request_time).total_seconds() # Calculate total time using datetime objects
    await save_embedding_to_db(request.text, request.model_name, embedding_json, req.client.host, request_time, response_time, total_time)
    return {"embedding": embedding_list}

async def save_embedding_to_db(text, model_name, embedding_json, ip_address, request_time, response_time, total_time):
    logger.info(f"Saving embedding for '{text}' using model '{model_name}' to database...")
    return await execute_with_retry(_save_embedding_to_db, text, model_name, embedding_json, ip_address, request_time, response_time, total_time)

async def _save_embedding_to_db(text, model_name, embedding_json, ip_address, request_time, response_time, total_time):
    async with AsyncSessionLocal() as session:
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
    models_dir = RAMDISK_PATH if USE_RAMDISK else os.path.join(BASE_DIRECTORY, 'models') # Determine the models_dir
    try:
        if model_name in model_cache: # Check the cache first
            logger.info(f"Model {model_name} found in cache")
            return model_cache[model_name]
        matching_files = glob.glob(os.path.join(models_dir, f"{model_name}*.bin")) # Find all files that match the model_name prefix
        logger.info(f"Found {len(matching_files)} model files matching: {model_name}")
        if not matching_files:
            logger.error(f"No model file found matching: {model_name}")
            raise FileNotFoundError
        matching_files.sort(key=os.path.getmtime, reverse=True) # Sort files by modification time (newest first)
        model_file_path = matching_files[0] # Select the newest file
        logger.info(f"Loading model file: {model_file_path}")
        model_instance = LlamaCppEmbeddings(model_path=model_file_path)
        model_cache[model_name] = model_instance
        logger.info(f"Loaded model file: {model_file_path}")
        return model_instance
    except FileNotFoundError:
        logger.error(f"No model file found matching: {model_name}")
        if raise_http_exception:
            raise HTTPException(status_code=404, detail="Model file not found")
        else:
            raise FileNotFoundError(f"No model file found matching: {model_name}")

def calculate_sentence_embedding(llama, text: str) -> np.array:
    sentence_embedding = None
    retry_count = 0
    while sentence_embedding is None and retry_count < 3:
        try:
            sentence_embedding = llama.embed_query(text)
        except Exception as e:
            logger.error(f"Exception in calculate_sentence_embedding: {e}")
            text = text[:-int(len(text) * 0.1)]  # Trim 10% of the text
            retry_count += 1
            logger.info(f"Trimming sentence due to too many tokens. New length: {len(text)}")
    return sentence_embedding

@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
    logger.exception(exc)
    return JSONResponse(status_code=500, content={"message": "Database error occurred"})

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception(exc)
    return JSONResponse(status_code=500, content={"message": "An unexpected error occurred"})

@app.get("/get_list_of_available_model_names/")
async def get_list_of_available_model_names(token: str = None):
    if USE_SECURITY_TOKEN and (token is None or token != SECURITY_TOKEN):
        raise HTTPException(status_code=403, detail="Unauthorized")
    models_dir = RAMDISK_PATH if USE_RAMDISK else os.path.join(BASE_DIRECTORY, 'models') # Determine the models_dir
    model_files = glob.glob(os.path.join(models_dir, "*.bin")) # Find all files with .bin extension
    model_names = [os.path.splitext(os.path.basename(model_file))[0] for model_file in model_files]
    return {"model_names": model_names}

@app.post("/get_embedding_vector/", response_model=EmbeddingResponse)
async def get_embedding_vector(request: EmbeddingRequest, req: Request, token: str = None):
    if USE_SECURITY_TOKEN and use_hardcoded_security_token and (token is None or token != SECURITY_TOKEN):
        raise HTTPException(status_code=403, detail="Unauthorized")
    try:
        return await get_or_compute_embedding(request, req)
    except Exception as e:
        logger.error(f"An error occurred while processing the request: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/compute_similarity_between_strings/", response_model=SimilarityResponse)
async def compute_similarity_between_strings(request: SimilarityRequest, token: str = None):
    if USE_SECURITY_TOKEN and use_hardcoded_security_token and (token is None or token != SECURITY_TOKEN):
        raise HTTPException(status_code=403, detail="Unauthorized")
    try:
        # Compute embeddings for both texts
        embedding1_response = await get_embedding_vector(EmbeddingRequest(text=request.text1, model_name=request.model_name), Request(scope={"client": ("localhost", 0)}))
        embedding2_response = await get_embedding_vector(EmbeddingRequest(text=request.text2, model_name=request.model_name), Request(scope={"client": ("localhost", 0)}))
        # Extract the embedding vectors
        embedding1 = np.array(embedding1_response["embedding"])
        embedding2 = np.array(embedding2_response["embedding"])
        # Ensure the embeddings are valid before computing similarity
        if embedding1.size == 0 or embedding2.size == 0:
            raise HTTPException(status_code=400, detail="Could not calculate embeddings for the given texts")
        # Compute the cosine similarity
        similarity = cosine_similarity([embedding1], [embedding2])
        return {
            "text1": request.text1,
            "text2": request.text2,
            "embedding1": embedding1.tolist(),
            "embedding2": embedding2.tolist(),
            "similarity": similarity[0][0]
        }
    except Exception as e:
        logger.error(f"An error occurred while processing the request: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/", include_in_schema=False)
async def custom_swagger_ui_html():
    return fastapi.templating.get_swagger_ui_html(openapi_url="/openapi.json", title=app.title, swagger_favicon_url=app.swagger_ui_favicon_url)

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
    if USE_RAMDISK:
        setup_ramdisk()    
    list_of_downloaded_model_names = download_models()
    for model_name in list_of_downloaded_model_names:
        try:
            load_model(model_name, raise_http_exception=False)
        except FileNotFoundError as e:
            logger.error(e)
    await initialize_db()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=LLAMA_EMBEDDING_SERVER_LISTEN_PORT)