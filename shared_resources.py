from misc_utility_functions import  is_redis_running, build_faiss_indexes, suppress_stdout_stderr
from database_functions import DatabaseWriter, initialize_db
from ramdisk_functions import setup_ramdisk, copy_models_to_ramdisk, check_that_user_has_required_permissions_to_manage_ramdisks
from logger_config import setup_logger
from aioredlock import Aioredlock
import aioredis
import asyncio
import subprocess
import urllib.request
import os
import glob
import json
from typing import List, Tuple, Dict
from langchain.embeddings import LlamaCppEmbeddings
from decouple import config
from fastapi import HTTPException

logger = setup_logger()
embedding_model_cache = {} # Model cache to store loaded models
token_level_embedding_model_cache = {} # Model cache to store loaded token-level embedding models
text_completion_model_cache = {} # Model cache to store loaded text completion models

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
RAMDISK_PATH = config("RAMDISK_PATH", default="/mnt/ramdisk", cast=str)
BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

async def initialize_globals():
    global db_writer, faiss_indexes, token_faiss_indexes, associated_texts_by_model, redis, lock_manager
    if not is_redis_running():
        logger.info("Starting Redis server...")
        subprocess.Popen(['redis-server'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        await asyncio.sleep(1)  # Sleep for 1 second to give Redis time to start        
    redis = await aioredis.create_redis_pool('redis://localhost')
    lock_manager = Aioredlock([redis])
    lock_manager.default_lock_timeout = 20000  # Set lock timeout to 20 seconds
    await initialize_db()
    queue = asyncio.Queue()
    db_writer = DatabaseWriter(queue)
    await db_writer.initialize_processing_hashes()
    asyncio.create_task(db_writer.dedicated_db_writer())
    global USE_RAMDISK
    if USE_RAMDISK and not check_that_user_has_required_permissions_to_manage_ramdisks():
        USE_RAMDISK = False
    elif USE_RAMDISK:
        setup_ramdisk()
    list_of_downloaded_model_names, download_status = download_models()
    for llm_model_name in list_of_downloaded_model_names:
        try:
            load_model(llm_model_name, raise_http_exception=False)
        except FileNotFoundError as e:
            logger.error(e)
    faiss_indexes, token_faiss_indexes, associated_texts_by_model = await build_faiss_indexes()

# other shared variables and methods
db_writer = None
faiss_indexes = None
token_faiss_indexes = None
associated_texts_by_model = None
redis = None
lock_manager = None


def download_models() -> Tuple[List[str], List[Dict[str, str]]]:
    download_status = []    
    json_path = os.path.join(BASE_DIRECTORY, "model_urls.json")
    if not os.path.exists(json_path):
        initial_model_urls = [
            'https://huggingface.co/TheBloke/Yarn-Llama-2-7B-128K-GGUF/resolve/main/yarn-llama-2-7b-128k.Q4_K_M.gguf',
            'https://huggingface.co/TheBloke/openchat_v3.2_super-GGUF/resolve/main/openchat_v3.2_super.Q4_K_M.gguf',
            'https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q6_K.gguf'
        ]
        with open(json_path, "w") as f:
            json.dump(initial_model_urls, f)
    with open(json_path, "r") as f:
        list_of_model_download_urls = json.load(f)
    model_names = [os.path.basename(url) for url in list_of_model_download_urls]
    current_file_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(current_file_path)
    models_dir = os.path.join(base_dir, 'models')
    logger.info("Checking models directory...")
    if USE_RAMDISK:
        ramdisk_models_dir = os.path.join(RAMDISK_PATH, 'models')
        if not os.path.exists(RAMDISK_PATH):
            setup_ramdisk()
        if all(os.path.exists(os.path.join(ramdisk_models_dir, llm_model_name)) for llm_model_name in model_names):
            logger.info("Models found in RAM Disk.")
            for url in list_of_model_download_urls:
                download_status.append({"url": url, "status": "success", "message": "Model found in RAM Disk."})
            return model_names, download_status
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        logger.info(f"Created models directory: {models_dir}")
    else:
        logger.info(f"Models directory exists: {models_dir}")
    for url, model_name_with_extension in zip(list_of_model_download_urls, model_names):
        status = {"url": url, "status": "success", "message": "File already exists."}
        filename = os.path.join(models_dir, model_name_with_extension)
        if not os.path.exists(filename):
            logger.info(f"Downloading model {model_name_with_extension} from {url}...")
            urllib.request.urlretrieve(url, filename)
            file_size = os.path.getsize(filename) / (1024 * 1024)  # Convert bytes to MB
            if file_size < 100:
                os.remove(filename)
                status["status"] = "failure"
                status["message"] = "Downloaded file is too small, probably not a valid model file."
            else:
                logger.info(f"Downloaded: {filename}")     
        else:
            logger.info(f"File already exists: {filename}")       
        download_status.append(status)
    if USE_RAMDISK:
        copy_models_to_ramdisk(models_dir, ramdisk_models_dir)
    logger.info("Model downloads completed.")
    return model_names, download_status


def load_model(llm_model_name: str, raise_http_exception: bool = True):
    try:
        models_dir = os.path.join(RAMDISK_PATH, 'models') if USE_RAMDISK else os.path.join(BASE_DIRECTORY, 'models')
        if llm_model_name in embedding_model_cache:
            return embedding_model_cache[llm_model_name]
        matching_files = glob.glob(os.path.join(models_dir, f"{llm_model_name}*"))
        if not matching_files:
            logger.error(f"No model file found matching: {llm_model_name}")
            raise FileNotFoundError
        matching_files.sort(key=os.path.getmtime, reverse=True)
        model_file_path = matching_files[0]
        with suppress_stdout_stderr():
            model_instance = LlamaCppEmbeddings(model_path=model_file_path, use_mlock=True, n_ctx=LLM_CONTEXT_SIZE_IN_TOKENS)
        model_instance.client.verbose = False
        embedding_model_cache[llm_model_name] = model_instance
        return model_instance
    except TypeError as e:
        logger.error(f"TypeError occurred while loading the model: {e}")
        raise
    except Exception as e:
        logger.error(f"Exception occurred while loading the model: {e}")
        if raise_http_exception:
            raise HTTPException(status_code=404, detail="Model file not found")
        else:
            raise FileNotFoundError(f"No model file found matching: {llm_model_name}")

        