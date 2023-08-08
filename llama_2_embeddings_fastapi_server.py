import json
import logging
import os 
import glob
import urllib.request
import asyncio
import shutil
import subprocess
import traceback
from typing import List, Optional
from datetime import datetime
import numpy as np
from decouple import config
import uvicorn
import psutil
import fastapi
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from langchain.embeddings import LlamaCppEmbeddings
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import Column, String, Float, DateTime, Integer, UniqueConstraint
from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError, OperationalError
import faiss

# Note: the Ramdisk setup and teardown requires sudo; to enable password-less sudo, edit your sudoers file with `sudo visudo`.
# Add the following lines, replacing username with your actual username
# username ALL=(ALL) NOPASSWD: /bin/mount -t tmpfs -o size=*G tmpfs /mnt/ramdisk
# username ALL=(ALL) NOPASSWD: /bin/umount /mnt/ramdisk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

use_hardcoded_security_token = 0
if use_hardcoded_security_token:
    SECURITY_TOKEN = "Test123$"
    USE_SECURITY_TOKEN = config("USE_SECURITY_TOKEN", default=False, cast=bool)
else:
    USE_SECURITY_TOKEN = False
DATABASE_URL = "sqlite+aiosqlite:///embeddings.db"
LLAMA_EMBEDDING_SERVER_LISTEN_PORT = config("LLAMA_EMBEDDING_SERVER_LISTEN_PORT", default=8089, cast=int)
DEFAULT_MODEL_NAME = "llama2_7b_chat_uncensored"
USE_RAMDISK = config("USE_RAMDISK", default=False, cast=bool)
RAMDISK_SIZE_IN_GB = config("RAMDISK_SIZE_IN_GB", default=1, cast=int)
RAMDISK_PATH = "/mnt/ramdisk"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1
model_cache = {} # Model cache to store loaded models
faiss_index = None
associated_texts = []
logger.info(f"USE_RAMDISK is set to: {USE_RAMDISK}")

app = FastAPI(docs_url="/")  # Set the Swagger UI to root
engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
Base = declarative_base()

def setup_ramdisk():
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

async def build_faiss_index():
    global faiss_index, associated_texts
    embeddings = []
    associated_texts = []
    logger.info("Building Faiss index...")
    async with AsyncSessionLocal() as session:
        result = await session.execute(sql_text("SELECT text, embedding_json FROM embeddings"))
        for row in result.fetchall():
            associated_texts.append(row[0])
            embeddings.append(json.loads(row[1]))
    embeddings = np.array(embeddings).astype('float32')
    faiss.normalize_L2(embeddings)  # Normalize the vectors for cosine similarity
    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])  # Use IndexFlatIP for cosine similarity
    faiss_index.add(embeddings)
    logger.info("Faiss index built.")

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
    
class EmbeddingResponse(BaseModel):
    embedding: List[float]

class SimilarityResponse(BaseModel):
    text1: str
    text2: str
    embedding1: List[float]
    embedding2: List[float]
    similarity: float
    
class SimilarStringResponse(BaseModel):
    text: str
    similarity: float
    message: str = ""

class AllStringsResponse(BaseModel):
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
    model_name: str = DEFAULT_MODEL_NAME

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
        await conn.execute(sql_text("PRAGMA journal_mode=WAL;")) # Set SQLite to use Write-Ahead Logging (WAL) mode
        await conn.execute(sql_text("PRAGMA busy_timeout = 2000;")) # Increase the busy timeout (for example, to 2 seconds)
        await conn.run_sync(Base.metadata.create_all)
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
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        logger.info(f"Created models directory: {models_dir}")
    else:
        logger.info(f"Models directory exists: {models_dir}")
    logger.info("Starting model downloads...")
    for url, model_name_with_extension in zip(list_of_model_download_urls, model_names):
        filename = os.path.join(models_dir, model_name_with_extension)
        if not os.path.exists(filename):
            logger.info(f"Downloading model {model_name_with_extension} from {url}...")
            urllib.request.urlretrieve(url, filename)
            logger.info(f"Downloaded: {filename}")
        else:
            logger.info(f"File already exists: {filename}")
    if USE_RAMDISK:
        logger.info("RAM Disk is enabled. Checking RAM Disk for models...")
        ramdisk_models_dir = os.path.join(RAMDISK_PATH, 'models')
        os.makedirs(ramdisk_models_dir, exist_ok=True)
        copy_models_to_ramdisk(models_dir, ramdisk_models_dir)
    logger.info("Model downloads completed.")
    return model_names

async def get_embedding_from_db(text, model_name):
    logger.info(f"Retrieving embedding for '{text}' using model '{model_name}' from database...")
    return await execute_with_retry(_get_embedding_from_db, text, model_name)

async def _get_embedding_from_db(text, model_name):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            sql_text("SELECT embedding_json FROM embeddings WHERE text=:text AND model_name=:model_name"),
            {"text": text, "model_name": model_name},
        )
        row = result.fetchone()
        if row:
            embedding_json = row[0]
            logger.info(f"Embedding JSON retrieved from database: {embedding_json}")
            logger.info(f"Embedding found in database for '{text}' using model '{model_name}'")
            return json.loads(embedding_json)
        logger.info(f"Embedding not found in database for '{text}' using model '{model_name}'")
        return None
    
async def get_or_compute_embedding(request: EmbeddingRequest, req: Request = None) -> dict:
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
    client_host = req.client.host if req is not None else "localhost" # Handle the case when req is None
    await save_embedding_to_db(request.text, request.model_name, embedding_json, client_host, request_time, response_time, total_time)
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
    try:
        logger.info(f"Attempting to load model: {model_name}")
        models_dir = os.path.join(RAMDISK_PATH, 'models') if USE_RAMDISK else os.path.join(BASE_DIRECTORY, 'models')
        logger.info(f"Searching in models directory: {models_dir}")
        if model_name in model_cache:
            logger.info(f"Model {model_name} found in cache")
            return model_cache[model_name]
        matching_files = glob.glob(os.path.join(models_dir, f"{model_name}*"))
        logger.info(f"Found {len(matching_files)} model files matching: {model_name}")
        if not matching_files:
            logger.error(f"No model file found matching: {model_name}")
            raise FileNotFoundError
        matching_files.sort(key=os.path.getmtime, reverse=True)
        model_file_path = matching_files[0]
        logger.info(f"Loading model file: {model_file_path}")
        model_instance = LlamaCppEmbeddings(model_path=model_file_path)
        model_cache[model_name] = model_instance
        logger.info(f"Loaded model file: {model_file_path}")
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
            logger.info(f"Trying to calculate sentence embedding. Attempt: {retry_count + 1}")
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
    models_dir = os.path.join(RAMDISK_PATH, 'models') if USE_RAMDISK else os.path.join(BASE_DIRECTORY, 'models')
    logger.info(f"Looking for models in: {models_dir}") # Add this line for debugging
    logger.info(f"Directory content: {os.listdir(models_dir)}") # Add this line for debugging
    model_files = glob.glob(os.path.join(models_dir, "*.bin")) # Find all files with .ggmlv3.q3_K_L.bin extension
    model_names = [os.path.splitext(os.path.splitext(os.path.basename(model_file))[0])[0] for model_file in model_files] # Remove both extensions
    return {"model_names": model_names}

@app.get("/get_all_strings_with_embeddings/", response_model=AllStringsResponse)
async def get_all_strings_with_embeddings(req: Request, token: str = None):
    logger.info("Received request to retrieve all strings with computed embeddings")
    if USE_SECURITY_TOKEN and use_hardcoded_security_token and (token is None or token != SECURITY_TOKEN):
        raise HTTPException(status_code=403, detail="Unauthorized")
    try:
        logger.info("Retrieving all strings with computed embeddings from the database")
        async with AsyncSessionLocal() as session:
            result = await session.execute(sql_text("SELECT DISTINCT text FROM embeddings"))
            all_strings = [row[0] for row in result.fetchall()]
        logger.info(f"Retrieved {len(all_strings)} strings with computed embeddings from the database")
        return {"strings": all_strings}
    except Exception as e:
        logger.error(f"An error occurred while processing the request: {e}")
        logger.error(traceback.format_exc())  # Print the traceback
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/get_embedding_vector/", response_model=EmbeddingResponse)
async def get_embedding_vector(request: EmbeddingRequest, req: Request, token: str = None):
    if USE_SECURITY_TOKEN and use_hardcoded_security_token and (token is None or token != SECURITY_TOKEN):
        raise HTTPException(status_code=403, detail="Unauthorized")
    try:
        return await get_or_compute_embedding(request, req)
    except Exception as e:
        logger.error(f"An error occurred while processing the request: {e}")
        logger.error(traceback.format_exc()) # Print the traceback
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/compute_similarity_between_strings/", response_model=SimilarityResponse)
async def compute_similarity_between_strings(request: SimilarityRequest, token: str = None):
    logger.info(f"Received request: {request}")
    if USE_SECURITY_TOKEN and use_hardcoded_security_token and (token is None or token != SECURITY_TOKEN):
        raise HTTPException(status_code=403, detail="Unauthorized")
    try:
        logger.info("Computing similarity between strings")
        embedding_request1 = EmbeddingRequest(text=request.text1, model_name=request.model_name)
        embedding_request2 = EmbeddingRequest(text=request.text2, model_name=request.model_name)
        logger.info(f"Requesting embeddings for: {embedding_request1} and {embedding_request2}")
        embedding1_response = await get_or_compute_embedding(embedding_request1, None)
        embedding2_response = await get_or_compute_embedding(embedding_request2, None)
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

@app.post("/get_most_similar_string_from_database/", response_model=SimilarStringResponse)
async def get_most_similar_string_from_database(request: SimilarStringRequest, req: Request, token: str = None):
    global faiss_index
    logger.info(f"Received request to find most similar string for: {request.text}")
    if USE_SECURITY_TOKEN and use_hardcoded_security_token and (token is None or token != SECURITY_TOKEN):
        raise HTTPException(status_code=403, detail="Unauthorized")
    try:
        logger.info(f"Computing embedding for input text: {request.text}")
        embedding_request = EmbeddingRequest(text=request.text, model_name=request.model_name)
        embedding_response = await get_embedding_vector(embedding_request, req)
        input_embedding = np.array(embedding_response["embedding"]).astype('float32').reshape(1, -1)
        faiss.normalize_L2(input_embedding)  # Normalize the input vector for cosine similarity
        logger.info(f"Computed embedding for input text: {request.text}")
        logger.info("Searching for the most similar string in the FAISS index")
        similarities, indices = faiss_index.search(input_embedding.reshape(1, -1), 1)
        similarity = similarities[0][0]  # Get the similarity value
        most_similar_text = associated_texts[indices[0][0]]  # Retrieve text using the index from FAISS search
        logger.info(f"Found most similar string: {most_similar_text} with similarity: {similarity}")
        return {
            "text": most_similar_text,
            "similarity": similarity
        }
    except Exception as e:
        logger.error(f"An error occurred while processing the request: {e}")
        logger.error(traceback.format_exc())  # Print the traceback
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
    await build_faiss_index() 
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=LLAMA_EMBEDDING_SERVER_LISTEN_PORT)
