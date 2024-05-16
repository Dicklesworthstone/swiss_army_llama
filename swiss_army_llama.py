import shared_resources
from shared_resources import initialize_globals, download_models, is_gpu_available
from logger_config import setup_logger
from database_functions import AsyncSessionLocal, DatabaseWriter, get_db_writer
from ramdisk_functions import clear_ramdisk
from misc_utility_functions import  build_faiss_indexes, safe_path
from embeddings_data_models import DocumentEmbedding, TokenLevelEmbeddingBundle
from embeddings_data_models import EmbeddingRequest, SemanticSearchRequest, AdvancedSemanticSearchRequest, SimilarityRequest, TextCompletionRequest, AddGrammarRequest
from embeddings_data_models import EmbeddingResponse, SemanticSearchResponse, AdvancedSemanticSearchResponse, SimilarityResponse, AllStringsResponse, AllDocumentsResponse, TextCompletionResponse, AddGrammarResponse
from embeddings_data_models import ShowLogsIncrementalModel
from service_functions import get_or_compute_embedding, get_or_compute_transcript, add_model_url, get_or_compute_token_level_embedding_bundle_combined_feature_vector, calculate_token_level_embeddings
from service_functions import parse_submitted_document_file_into_sentence_strings_func, compute_embeddings_for_document, store_document_embeddings_in_db, generate_completion_from_llm, validate_bnf_grammar_func
from grammar_builder import GrammarBuilder
from log_viewer_functions import show_logs_incremental_func, show_logs_func
from uvicorn_config import option
import asyncio
import glob
import json
import os 
import signal
import re
import tempfile
import traceback
import zipfile
from pathlib import Path
from datetime import datetime
from hashlib import sha3_256
from typing import List, Optional, Dict, Any
from urllib.parse import unquote
import numpy as np
from decouple import config
import uvicorn
import fastapi
from fastapi.param_functions import Body
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Depends, Form
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, Response
from contextlib import asynccontextmanager
from sqlalchemy import select
from sqlalchemy import text as sql_text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import joinedload
import faiss
import pandas as pd
from magic import Magic
import fast_vector_similarity as fvs
import uvloop

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
logger = setup_logger()

gpu_check_results = is_gpu_available()
logger.info(f"\nGPU check results:\n {gpu_check_results}\n")

class GracefulExit(BaseException):
    pass

def raise_graceful_exit():
    raise GracefulExit()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    await initialize_globals()
    yield
    # Shutdown code (if any)
    pass


# Note: the Ramdisk setup and teardown requires sudo; to enable password-less sudo, edit your sudoers file with `sudo visudo`.
# Add the following lines, replacing username with your actual username
# username ALL=(ALL) NOPASSWD: /bin/mount -t tmpfs -o size=*G tmpfs /mnt/ramdisk
# username ALL=(ALL) NOPASSWD: /bin/umount /mnt/ramdisk

# Global variables
use_hardcoded_security_token = 0
if use_hardcoded_security_token:
    SECURITY_TOKEN = "Test123$"
    USE_SECURITY_TOKEN = config("USE_SECURITY_TOKEN", default=False, cast=bool)
else:
    USE_SECURITY_TOKEN = False
DEFAULT_MODEL_NAME = config("DEFAULT_MODEL_NAME", default="Meta-Llama-3-8B-Instruct.Q3_K_S", cast=str) 
USE_RAMDISK = config("USE_RAMDISK", default=False, cast=bool)
RAMDISK_PATH = config("RAMDISK_PATH", default="/mnt/ramdisk", cast=str)
BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

logger.info(f"USE_RAMDISK is set to: {USE_RAMDISK}")

description_string = """
🇨🇭🎖️🦙 Swiss Army Llama is your One-Stop-Shop to Quickly and Conveniently Integrate Powerful Local LLM Functionality into your Project via a REST API.
"""
app = FastAPI(title="Swiss Army Llama", description=description_string, docs_url="/", lifespan=lifespan)  # Set the Swagger UI to root

    
@app.exception_handler(SQLAlchemyError) 
async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    logger.exception(exc)
    return JSONResponse(status_code=500, content={"message": "Database error occurred"})

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
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
    "model_names": ["Meta-Llama-3-8B-Instruct", "Hermes-2-Pro-Llama-3-Instruct-Merged-DPO-Q4_K_M", "my_super_custom_model"]
}
```""",
        response_description="A JSON object containing the list of available model names.")
async def get_list_of_available_model_names(token: str = None) -> Dict[str, List[str]]:
    if USE_SECURITY_TOKEN and (token is None or token != SECURITY_TOKEN):
        raise HTTPException(status_code=403, detail="Unauthorized")
    models_dir = os.path.join(RAMDISK_PATH, 'models') if USE_RAMDISK else os.path.join(BASE_DIRECTORY, 'models')
    logger.info(f"Looking for models in: {models_dir}") # Add this line for debugging
    logger.info(f"Directory content: {os.listdir(models_dir)}") # Add this line for debugging
    model_files = glob.glob(os.path.join(models_dir, "*.bin")) +  glob.glob(os.path.join(models_dir, "*.gguf"))# Find all files with .bin or .gguf extension
    model_names = [os.path.splitext(os.path.splitext(os.path.basename(model_file))[0])[0] for model_file in model_files] # Remove both extensions
    return {"model_names": model_names}



@app.get("/get_list_of_available_bnf_grammars",
        response_model=List[str],
        summary="Get Available BNF Grammars",
        description="Returns a list of all the valid .gbnf files in the grammar_files directory.",
        response_description="A list containing the names of all valid .gbnf files.")
async def get_list_of_available_bnf_grammars(token: str = None) -> List[str]:
    if USE_SECURITY_TOKEN and (token is None or token != SECURITY_TOKEN):
        raise HTTPException(status_code=403, detail="Unauthorized")    
    try:
        grammar_files_dir = 'grammar_files'
        if not os.path.exists(grammar_files_dir):
            os.makedirs(grammar_files_dir)
        valid_files = [f for f in os.listdir(grammar_files_dir) if f.endswith('.gbnf')]
        return valid_files
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")



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
async def get_all_stored_strings(req: Request, token: str = None) -> AllStringsResponse:
    logger.info("Received request to retrieve all stored strings for which embeddings have been computed")
    if USE_SECURITY_TOKEN and use_hardcoded_security_token and (token is None or token != SECURITY_TOKEN):
        logger.warning(f"Unauthorized request to retrieve all stored strings for which embeddings have been computed from {req.client.host}")
        raise HTTPException(status_code=403, detail="Unauthorized")
    try:
        logger.info("Retrieving all stored strings with computed embeddings from the database")
        async with AsyncSessionLocal() as session:
            result = await session.execute(sql_text("SELECT DISTINCT text FROM embeddings"))
            all_strings = [row[0] for row in result.fetchall()]
        logger.info(f"Retrieved {len(all_strings)} stored strings with computed embeddings from the database; Last 10 embedded strings: {all_strings[-10:]}")
        return {"strings": all_strings}
    except Exception as e:
        logger.error(f"An error occurred while processing the request: {e}")
        logger.error(traceback.format_exc())  # Print the traceback
        raise HTTPException(status_code=500, detail="Internal Server Error")



@app.get("/get_all_stored_documents/",
        summary="Retrieve All Stored Documents",
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
async def get_all_stored_documents(req: Request, token: str = None) -> AllDocumentsResponse:
    logger.info("Received request to retrieve all stored documents with computed embeddings")
    if USE_SECURITY_TOKEN and use_hardcoded_security_token and (token is None or token != SECURITY_TOKEN):
        logger.warning(f"Unauthorized request to retrieve all stored documents for which all sentence embeddings have been computed from {req.client.host}")
        raise HTTPException(status_code=403, detail="Unauthorized")
    try:
        logger.info("Retrieving all stored documents with computed embeddings from the database")
        async with AsyncSessionLocal() as session:
            result = await session.execute(sql_text("SELECT DISTINCT filename FROM document_embeddings"))
            all_documents = [row[0] for row in result.fetchall()]
        logger.info(f"Retrieved {len(all_documents)} stored documents with computed embeddings from the database; Last 10 processed document filenames: {all_documents[-10:]}")
        return {"documents": all_documents}
    except Exception as e:
        logger.error(f"An error occurred while processing the request: {e}")
        logger.error(traceback.format_exc())  # Print the traceback
        raise HTTPException(status_code=500, detail="Internal Server Error")



@app.post("/add_new_model/",
        summary="Add New Model by URL",
        description="""Submit a new model URL for download and use. The model must satisfy the following criteria:

1. Must be in `.gguf` format.
2. Must be larger than 100 MB to ensure it's a valid model file.

### Parameters:
- `model_url`: The URL of the model weight file, which must end with `.gguf`. For example: `https://huggingface.co/kirp/TinyLlama-1.1B-Chat-v0.2-gguf/blob/main/ggml-model-q5_k_m.gguf`
- `token`: Security token (optional).

### Response:
The response will include a JSON object indicating whether the model was successfully added and downloaded. Possible status values are:

- `success`: Model was added and downloaded successfully.
- `failure`: Model download failed, likely because it's not a valid model file.
- `error`: The URL did not point to a `.gguf` file.
- `unknown`: An unexpected error occurred.

### Example Response:
```json
{
    "status": "success",
    "message": "Model added and downloaded successfully."
}
```
""",
        response_description="A JSON object indicating the status of the model addition and download.")
async def add_new_model(model_url: str, token: str = None) -> Dict[str, Any]:
    if USE_SECURITY_TOKEN and (token is None or token != SECURITY_TOKEN):
        raise HTTPException(status_code=403, detail="Unauthorized")
    unique_id = f"add_model_{hash(model_url)}"     # Generate a unique lock ID based on the model_url
    lock = await shared_resources.lock_manager.lock(unique_id)
    if lock.valid:
        try:
            decoded_model_url = unquote(model_url)
            if not decoded_model_url.endswith('.gguf'):
                return {"status": "error", "message": "Model URL must point to a .gguf file."}
            corrected_model_url = add_model_url(decoded_model_url)
            _, download_status = download_models()
            status_dict = {status["url"]: status for status in download_status}
            if corrected_model_url in status_dict:
                return {"status": status_dict[corrected_model_url]["status"], "message": status_dict[corrected_model_url]["message"]}
            return {"status": "unknown", "message": "Unexpected error."}
        finally:
            await shared_resources.lock_manager.unlock(lock)
    else:
        return {"status": "already processing", "message": "Another worker is already processing this model URL."}



@app.post("/get_embedding_vector_for_string/",
        response_model=EmbeddingResponse,
        summary="Retrieve Embedding Vector for a Given Text String",
        description="""Retrieve the embedding vector for a given input text string using the specified model.

### Parameters:
- `request`: A JSON object containing the input text string (`text`) and the model name.
- `token`: Security token (optional).
- `document_file_hash`: The SHA3-256 hash of the document file, if applicable (optional).

### Request JSON Format:
The request must contain the following attributes:
- `text`: The input text for which the embedding vector is to be retrieved.
- `llm_model_name`: The model used to calculate the embedding (optional, will use the default model if not provided).

### Example (note that `llm_model_name` is optional):
```json
{
    "text": "This is a sample text.",
    "llm_model_name": "bge-m3-q8_0"
}
```

### Response:
The response will include the embedding vector for the input text string.

### Example Response:
```json
{
    "embedding": [0.1234, 0.5678, ...]
}
```""", response_description="A JSON object containing the embedding vector for the input text.")
async def get_embedding_vector_for_string(request: EmbeddingRequest, req: Request = None, token: str = None, client_ip: str = None, document_file_hash: str = None) -> EmbeddingResponse:
    if USE_SECURITY_TOKEN and use_hardcoded_security_token and (token is None or token != SECURITY_TOKEN):
        logger.warning(f"Unauthorized request from client IP {client_ip}")
        raise HTTPException(status_code=403, detail="Unauthorized")
    try:
        unique_id = f"get_embedding_{request.text}_{request.llm_model_name}"
        lock = await shared_resources.lock_manager.lock(unique_id)
        if lock.valid:
            try:
                return await get_or_compute_embedding(request, req, client_ip, document_file_hash)
            finally:
                await shared_resources.lock_manager.unlock(lock)
        else:
            return {"status": "already processing"}
    except Exception as e:
        logger.error(f"An error occurred while processing the request: {e}")
        logger.error(traceback.format_exc()) # Print the traceback
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/get_token_level_embeddings_matrix_and_combined_feature_vector_for_string/",
        summary="Retrieve Token-Level Embeddings and Combined Feature Vector for a Given Input String",
        description="""Retrieve the token-level embeddings and combined feature vector for a given input text using the specified model.

### Parameters:
- `request`: A JSON object containing the text and the model name.
- `db_writer`: Database writer instance for managing write operations.
- `req`: HTTP request object (optional).
- `token`: Security token (optional).
- `client_ip`: Client IP address (optional).
- `json_format`: Format for JSON response of token-level embeddings (optional).
- `send_back_json_or_zip_file`: Whether to return a JSON response or a ZIP file containing the JSON file (optional, defaults to `zip`).

### Request JSON Format:
The request must contain the following attributes:
- `text`: The input text for which the embeddings are to be retrieved.
- `llm_model_name`: The model used to calculate the embeddings (optional).

### Example Request:
```json
{
    "text": "This is a sample text.",
    "llm_model_name": "Meta-Llama-3-8B-Instruct"
}
```

### Response:

The response will include the input text for reference, and token-level embeddings matrix for the input text. The response is organized as a JSON array of objects, each containing a token and its corresponding embedding vector. 
Token level embeddings represent a text by breaking it down into individual tokens (words) and associating an embedding vector with each token. These embeddings capture the semantic and
syntactic meaning of each token within the context of the text. Token level embeddings result in a matrix (number of tokens by embedding size), whereas a single embedding vector results 
in a one-dimensional vector of fixed size.

The response will also include a combined feature vector derived from the the token-level embeddings matrix; this combined feature vector has the great benefit that it is always the same length
for all input texts, regardless of length (whereas the token-level embeddings matrix will have a different number of rows for each input text, depending on the number of tokens in the text).
The combined feature vector is obtained by calculating the column-wise means, mins, maxes, and standard deviations of the token-level embeddings matrix; thus if the token-level embedding vectors
are of length `n`, the combined feature vector will be of length `4n`.

- `input_text`: The original input text.
- `token_level_embedding_bundle`: Either a ZIP file containing the JSON file, or a direct JSON array containing the token-level embeddings and combined feature vector for the input text, depending on the value of `send_back_json_or_zip_file`.
- `combined_feature_vector`: A list containing the combined feature vector, obtained by calculating the column-wise means, mins, maxes, and standard deviations of the token-level embeddings. This vector is always of length `4n`, where `n` is the length of the token-level embedding vectors.

### Example Response:
```json
{
    "input_text": "This is a sample text.",
    "token_level_embedding_bundle": [
        {"token": "This", "embedding": [0.1234, 0.5678, ...]},
        {"token": "is", "embedding": [...]},
        ...
    ],
    "combined_feature_vector": [0.5678, 0.1234, ...]
}
```
""",
        response_description="A JSON object containing the input text, token embeddings, and combined feature vector for the input text.")
async def get_token_level_embeddings_matrix_and_combined_feature_vector_for_string(
    request: EmbeddingRequest, 
    db_writer: DatabaseWriter = Depends(get_db_writer),
    req: Request = None, 
    token: str = None, 
    client_ip: str = None, 
    json_format: str = 'records',
    send_back_json_or_zip_file: str = 'zip'
) -> Response:
    logger.info(f"Received request for token embeddings with text length {len(request.text)} and model: '{request.llm_model_name}' from client IP: {client_ip}; input text: {request.text}")
    request_time = datetime.utcnow()
    if USE_SECURITY_TOKEN and use_hardcoded_security_token and (token is None or token != SECURITY_TOKEN):
        logger.warning(f"Unauthorized request from client IP {client_ip}")
        raise HTTPException(status_code=403, detail="Unauthorized")
    input_text_hash = sha3_256(request.text.encode('utf-8')).hexdigest()
    logger.info(f"Computed input text hash: {input_text_hash}")
    async with AsyncSessionLocal() as session:
        logger.info(f"Querying database for existing token-level embedding bundle for input text string {request.text} and model {request.llm_model_name}")
        result = await session.execute(
                select(TokenLevelEmbeddingBundle)
                .options(joinedload(TokenLevelEmbeddingBundle.token_level_embeddings)) # Eagerly load the relationship
                .filter(TokenLevelEmbeddingBundle.input_text_hash == input_text_hash, TokenLevelEmbeddingBundle.llm_model_name == request.llm_model_name)
            )
        existing_embedding_bundle = result.unique().scalar()
        if existing_embedding_bundle:
            logger.info("Found existing token-level embedding bundle in the database.")
            combined_feature_vector = await get_or_compute_token_level_embedding_bundle_combined_feature_vector(existing_embedding_bundle.id, existing_embedding_bundle.token_level_embeddings, db_writer)
            response_content = {
                'input_text': request.text,
                'token_level_embedding_bundle': json.loads(existing_embedding_bundle.token_level_embeddings_bundle_json),
                'combined_feature_vector': combined_feature_vector
            }
            return JSONResponse(content=response_content)
    unique_id = f"get_token_level_embeddings_{request.text}_{request.llm_model_name}"
    lock = await shared_resources.lock_manager.lock(unique_id)
    if lock.valid:
        try:
            logger.info("No cached result found. Calculating token-level embeddings now...")
            try:
                embedding_bundle = TokenLevelEmbeddingBundle(
                    input_text=request.text,
                    llm_model_name=request.llm_model_name,
                    ip_address=client_ip,
                    request_time=request_time
                )
                token_embeddings = await calculate_token_level_embeddings(request.text, request.llm_model_name, client_ip, embedding_bundle.id)
                tokens = re.findall(r'\b\w+\b', request.text)
                logger.info(f"Tokenized text into {len(tokens)} tokens. Organizing results.")
                df = pd.DataFrame({
                    'token': tokens,
                    'embedding': [embedding.tolist() for embedding in token_embeddings]
                })
                json_content = df.to_json(orient=json_format or 'records')
                response_time=datetime.utcnow()
                total_time = (response_time - request_time).total_seconds()
                embedding_bundle.token_level_embeddings_bundle_json = json_content
                embedding_bundle.response_time = response_time
                embedding_bundle.total_time = total_time
                combined_feature_vector = await get_or_compute_token_level_embedding_bundle_combined_feature_vector(embedding_bundle.id, json_content, db_writer)        
                response_content = {
                    'input_text': request.text,
                    'token_level_embedding_bundle': json.loads(embedding_bundle.token_level_embeddings_bundle_json),
                    'combined_feature_vector': combined_feature_vector
                }
                logger.info(f"Done getting token-level embedding matrix and combined feature vector for input text string {request.text} and model {request.llm_model_name}")
                json_content = embedding_bundle.token_level_embeddings_bundle_json
                json_content_length = len(json.dumps(response_content))
                overall_total_time = (datetime.utcnow() - request_time).total_seconds()
                if len(embedding_bundle.token_level_embeddings_bundle_json) > 0:
                    tokens = re.findall(r'\b\w+\b', request.text)
                    logger.info(f"The response took {overall_total_time} seconds to generate, or {overall_total_time / (float(len(tokens))/1000.0)} seconds per thousand input tokens and {overall_total_time / (float(json_content_length)/1000000.0)} seconds per million output characters.")
                if send_back_json_or_zip_file == 'json': # Assume 'json' response should be sent back
                    logger.info(f"Now sending back JSON response for input text string {request.text} and model {request.llm_model_name}; First 100 characters of JSON response out of {len(json_content)} total characters: {json_content[:100]}")
                    return JSONResponse(content=response_content)
                else: # Assume 'zip' file should be sent back
                    output_file_name_without_extension = f"token_level_embeddings_and_combined_feature_vector_for_input_hash_{input_text_hash}_and_model_name__{request.llm_model_name}"
                    is_safe_json, safe_json_file_path = safe_path('/tmp', f"{output_file_name_without_extension}.json")
                    is_safe_zip, safe_zip_file_path = safe_path('/tmp', f"{output_file_name_without_extension}.zip")
                    if is_safe_json and is_safe_zip:                    
                        with open(safe_json_file_path, 'w') as json_file:
                            json.dump(response_content, json_file)
                        with zipfile.ZipFile(safe_zip_file_path, 'w') as zipf:
                            zipf.write(safe_json_file_path, os.path.basename(safe_json_file_path))
                        logger.info(f"Now sending back ZIP file response for input text string {request.text} and model {request.llm_model_name}; First 100 characters of zipped JSON file out of {len(json_content)} total characters: {json_content[:100]}")                            
                        return FileResponse(safe_zip_file_path, headers={"Content-Disposition": f"attachment; filename={output_file_name_without_extension}.zip"})
                    else:
                        logger.error("Potential path injection attack detected.")
                        raise HTTPException(status_code=500, detail="Internal Server Error")
            except Exception as e:
                logger.error(f"An error occurred while processing the request: {e}")
                logger.error(traceback.format_exc())
                raise HTTPException(status_code=500, detail="Internal Server Error")
        finally:
            await shared_resources.lock_manager.unlock(lock)
    else:
        return {"status": "already processing"}


@app.post("/compute_similarity_between_strings/",
        response_model=SimilarityResponse,
        summary="Compute Similarity Between Two Strings",
        description="""Compute the similarity between two given input strings using specified model embeddings and a selected similarity measure.

### Parameters:
- `request`: A JSON object containing the two strings, the model name, and the similarity measure.
- `token`: Security token (optional).

### Request JSON Format:
The request must contain the following attributes:
- `text1`: The first input text.
- `text2`: The second input text.
- `llm_model_name`: The model used to calculate embeddings (optional).
- `similarity_measure`: The similarity measure to be used. Supported measures include `all`, `spearman_rho`, `kendall_tau`, `approximate_distance_correlation`, `jensen_shannon_similarity`, and `hoeffding_d` (optional, default is `all`).

### Example Request (note that `llm_model_name` and `similarity_measure` are optional):
```json
{
    "text1": "This is a sample text.",
    "text2": "This is another sample text.",
    "llm_model_name": "bge-m3-q8_0",
    "similarity_measure": "all"
}
```""")
async def compute_similarity_between_strings(request: SimilarityRequest, req: Request, token: str = None) -> SimilarityResponse:
    logger.info(f"Received request: {request}")
    request_time = datetime.utcnow()
    similarity_measure = request.similarity_measure.lower()
    if USE_SECURITY_TOKEN and use_hardcoded_security_token and (token is None or token != SECURITY_TOKEN):
        raise HTTPException(status_code=403, detail="Unauthorized")
    unique_id = f"compute_similarity_{request.text1}_{request.text2}_{request.llm_model_name}_{similarity_measure}"
    lock = await shared_resources.lock_manager.lock(unique_id)
    if lock.valid:
        try:
            client_ip = req.client.host if req else "localhost"
            embedding_request1 = EmbeddingRequest(text=request.text1, llm_model_name=request.llm_model_name)
            embedding_request2 = EmbeddingRequest(text=request.text2, llm_model_name=request.llm_model_name)
            embedding1_response = await get_or_compute_embedding(embedding_request1, client_ip=client_ip)
            embedding2_response = await get_or_compute_embedding(embedding_request2, client_ip=client_ip)
            embedding1 = np.array(embedding1_response["embedding"])
            embedding2 = np.array(embedding2_response["embedding"])
            if embedding1.size == 0 or embedding2.size == 0:
                raise HTTPException(status_code=400, detail="Could not calculate embeddings for the given texts")
            params = {
                "vector_1": embedding1.tolist(),
                "vector_2": embedding2.tolist(),
                "similarity_measure": similarity_measure
            }
            similarity_stats_str = fvs.py_compute_vector_similarity_stats(json.dumps(params))
            similarity_stats_json = json.loads(similarity_stats_str)
            if similarity_measure == 'all':
                similarity_score = similarity_stats_json
            else:
                similarity_score = similarity_stats_json.get(similarity_measure, None)
                if similarity_score is None:
                    raise HTTPException(status_code=400, detail="Invalid similarity measure specified")
            response_time = datetime.utcnow()
            total_time = (response_time - request_time).total_seconds()
            logger.info(f"Computed similarity using {similarity_measure} in {total_time} seconds; similarity score: {similarity_score}")
            return {
                "text1": request.text1,
                "text2": request.text2,
                "similarity_measure": similarity_measure,
                "similarity_score": similarity_score,
                "embedding1": embedding1.tolist(),
                "embedding2": embedding2.tolist()
            }
        except Exception as e:
            logger.error(f"An error occurred while processing the request: {e}")
            raise HTTPException(status_code=500, detail="Internal Server Error")
        finally:
            await shared_resources.lock_manager.unlock(lock)
    else:
        return {"status": "already processing"}



@app.post("/search_stored_embeddings_with_query_string_for_semantic_similarity/",
        response_model=SemanticSearchResponse,
        summary="Get Most Similar Strings from Stored Embedddings in Database",
        description="""Find the most similar strings in the database to the given input "query" text. This endpoint uses a pre-computed FAISS index to quickly search for the closest matching strings.

### Parameters:
- `request`: A JSON object containing the query text, model name, and an optional number of most semantically similar strings to return.
- `req`: HTTP request object (internal use).
- `token`: Security token (optional).

### Request JSON Format:
The request must contain the following attributes:
- `query_text`: The input text for which to find the most similar string.
- `llm_model_name`: The model used to calculate embeddings.
- `number_of_most_similar_strings_to_return`: (Optional) The number of most similar strings to return, defaults to 10.

### Example:
```json
{
    "query_text": "Find me the most similar string!",
    "llm_model_name": "bge-m3-q8_0",
    "number_of_most_similar_strings_to_return": 5
}
```

### Response:
The response will include the most similar strings found in the database, along with the similarity scores.

### Example Response:
```json
{
    "query_text": "Find me the most similar string!",  
    "results": [
        {"search_result_text": "This is the most similar string!", "similarity_to_query_text": 0.9823},
        {"search_result_text": "Another similar string.", "similarity_to_query_text": 0.9721},
        ...
    ]
}
```""",
        response_description="A JSON object containing the query text along with the most similar strings and similarity scores.")
async def search_stored_embeddings_with_query_string_for_semantic_similarity(request: SemanticSearchRequest, req: Request, token: str = None) -> SemanticSearchResponse:
    global faiss_indexes, token_faiss_indexes, associated_texts_by_model
    unique_id = f"semantic_search_{request.query_text}_{request.llm_model_name}_{request.number_of_most_similar_strings_to_return}"  # Unique ID for this operation
    lock = await shared_resources.lock_manager.lock(unique_id)        
    if lock.valid:
        try:                
            faiss_indexes, token_faiss_indexes, associated_texts_by_model = await build_faiss_indexes()
            request_time = datetime.utcnow()
            llm_model_name = request.llm_model_name
            num_results = request.number_of_most_similar_strings_to_return
            total_entries = len(associated_texts_by_model[llm_model_name])  # Get the total number of entries for the model
            num_results = min(num_results, total_entries)  # Ensure num_results doesn't exceed the total number of entries
            logger.info(f"Received request to find {num_results} most similar strings for query text: `{request.query_text}` using model: {llm_model_name}")
            if USE_SECURITY_TOKEN and use_hardcoded_security_token and (token is None or token != SECURITY_TOKEN):
                raise HTTPException(status_code=403, detail="Unauthorized")
            try:
                logger.info(f"Computing embedding for input text: {request.query_text}")
                embedding_request = EmbeddingRequest(text=request.query_text, llm_model_name=request.llm_model_name)
                embedding_response = await get_embedding_vector_for_string(embedding_request, req)
                input_embedding = np.array(embedding_response["embedding"]).astype('float32').reshape(1, -1)
                faiss.normalize_L2(input_embedding)  # Normalize the input vector for cosine similarity
                logger.info(f"Computed embedding for input text: {request.query_text}")
                faiss_index = faiss_indexes.get(llm_model_name)  # Retrieve the correct FAISS index for the llm_model_name
                if faiss_index is None:
                    raise HTTPException(status_code=400, detail=f"No FAISS index found for model: {llm_model_name}")
                logger.info("Searching for the most similar string in the FAISS index")
                similarities, indices = faiss_index.search(input_embedding.reshape(1, -1), num_results)  # Search for num_results similar strings
                results = []  # Create an empty list to store the results
                for ii in range(num_results):
                    similarity = float(similarities[0][ii])  # Convert numpy.float32 to native float
                    most_similar_text = associated_texts_by_model[llm_model_name][indices[0][ii]]
                    if most_similar_text != request.query_text:  # Don't return the query text as a result
                        results.append({"search_result_text": most_similar_text, "similarity_to_query_text": similarity})
                response_time = datetime.utcnow()
                total_time = (response_time - request_time).total_seconds()
                logger.info(f"Finished searching for the most similar string in the FAISS index in {total_time} seconds. Found {len(results)} results, returning the top {num_results}.")
                logger.info(f"Found most similar strings for query string {request.query_text}: {results}")
                return {"query_text": request.query_text, "results": results} # Return the response matching the SemanticSearchResponse model
            except Exception as e:
                logger.error(f"An error occurred while processing the request: {e}")
                logger.error(traceback.format_exc())  # Print the traceback
                raise HTTPException(status_code=500, detail="Internal Server Error")
        finally:
            await shared_resources.lock_manager.unlock(lock)
    else:
        return {"status": "already processing"}
        
        
        
@app.post("/advanced_search_stored_embeddings_with_query_string_for_semantic_similarity/",
        response_model=AdvancedSemanticSearchResponse,
        summary="Advanced Semantic Search with Two-Step Similarity Measures",
        description="""Perform an advanced semantic search by first using FAISS and cosine similarity to narrow down the most similar strings in the database, and then applying additional similarity measures for finer comparison.

### Parameters:
- `request`: A JSON object containing the query text, model name, an optional similarity filter percentage, and an optional number of most similar strings to return.
- `req`: HTTP request object (internal use).
- `token`: Security token (optional).

### Request JSON Format:
The request must contain the following attributes:
- `query_text`: The input text for which to find the most similar string.
- `llm_model_name`: The model used to calculate embeddings.
- `similarity_filter_percentage`: (Optional) The percentage of embeddings to filter based on cosine similarity, defaults to 0.02 (i.e., top 2%).
- `number_of_most_similar_strings_to_return`: (Optional) The number of most similar strings to return after applying the second similarity measure, defaults to 10.

### Example:
```json
{
    "query_text": "Find me the most similar string!",
    "llm_model_name": "bge-m3-q8_0",
    "similarity_filter_percentage": 0.02,
    "number_of_most_similar_strings_to_return": 5
}
```

### Response:
The response will include the most similar strings found in the database, along with their similarity scores for multiple measures.

### Example Response:
```json
{
    "query_text": "Find me the most similar string!",
    "results": [
        {"search_result_text": "This is the most similar string!", "similarity_to_query_text": {"cosine_similarity": 0.9823, "spearman_rho": 0.8, ... }},
        {"search_result_text": "Another similar string.", "similarity_to_query_text": {"cosine_similarity": 0.9721, "spearman_rho": 0.75, ... }},
        ...
    ]
}
```""",
        response_description="A JSON object containing the query text and the most similar strings, along with their similarity scores for multiple measures.")
async def advanced_search_stored_embeddings_with_query_string_for_semantic_similarity(request: AdvancedSemanticSearchRequest, req: Request, token: str = None) -> AdvancedSemanticSearchResponse:
    global faiss_indexes, token_faiss_indexes, associated_texts_by_model
    unique_id = f"advanced_semantic_search_{request.query_text}_{request.llm_model_name}_{request.similarity_filter_percentage}_{request.number_of_most_similar_strings_to_return}"
    lock = await shared_resources.lock_manager.lock(unique_id)        
    if lock.valid:
        try:                
            faiss_indexes, token_faiss_indexes, associated_texts_by_model = await build_faiss_indexes()
            request_time = datetime.utcnow()
            llm_model_name = request.llm_model_name
            total_entries = len(associated_texts_by_model[llm_model_name])
            num_results = max([1, int((1 - request.similarity_filter_percentage) * total_entries)])
            logger.info(f"Received request to find {num_results} most similar strings for query text: `{request.query_text}` using model: {llm_model_name}")
            if USE_SECURITY_TOKEN and use_hardcoded_security_token and (token is None or token != SECURITY_TOKEN):
                raise HTTPException(status_code=403, detail="Unauthorized")
            try:
                logger.info(f"Computing embedding for input text: {request.query_text}")
                embedding_request = EmbeddingRequest(text=request.query_text, llm_model_name=llm_model_name)
                embedding_response = await get_embedding_vector_for_string(embedding_request, req)
                input_embedding = np.array(embedding_response["embedding"]).astype('float32').reshape(1, -1)
                faiss.normalize_L2(input_embedding)
                logger.info(f"Computed embedding for input text: {request.query_text}")
                faiss_index = faiss_indexes.get(llm_model_name)
                if faiss_index is None:
                    raise HTTPException(status_code=400, detail=f"No FAISS index found for model: {llm_model_name}")
                _, indices = faiss_index.search(input_embedding, num_results)
                filtered_indices = indices[0]
                similarity_results = []
                for idx in filtered_indices:
                    associated_text = associated_texts_by_model[llm_model_name][idx]
                    embedding_request = EmbeddingRequest(text=associated_text, llm_model_name=llm_model_name)
                    embedding_response = await get_embedding_vector_for_string(embedding_request, req)
                    filtered_embedding = np.array(embedding_response["embedding"])
                    params = {
                        "vector_1": input_embedding.tolist()[0],
                        "vector_2": filtered_embedding.tolist(),
                        "similarity_measure": "all"
                    }
                    similarity_stats_str = fvs.py_compute_vector_similarity_stats(json.dumps(params))
                    similarity_stats_json = json.loads(similarity_stats_str)
                    similarity_results.append({
                        "search_result_text": associated_text,
                        "similarity_to_query_text": similarity_stats_json
                    })
                num_to_return = request.number_of_most_similar_strings_to_return if request.number_of_most_similar_strings_to_return is not None else len(similarity_results)
                results = sorted(similarity_results, key=lambda x: x["similarity_to_query_text"]["hoeffding_d"], reverse=True)[:num_to_return]
                response_time = datetime.utcnow()
                total_time = (response_time - request_time).total_seconds()
                logger.info(f"Finished advanced search in {total_time} seconds. Found {len(results)} results.")
                return {"query_text": request.query_text, "results": results}
            except Exception as e:
                logger.error(f"An error occurred while processing the request: {e}")
                raise HTTPException(status_code=500, detail="Internal Server Error")
        finally:
            await shared_resources.lock_manager.unlock(lock)
    else:
        return {"status": "already processing"}

    

@app.post("/get_all_embedding_vectors_for_document/",
        summary="Get Embeddings for a Document",
        description="""Extract text embeddings for a document. This endpoint supports plain text, .doc/.docx (MS Word), PDF files, images (using Tesseract OCR), and many other file types supported by the textract library.

### Parameters:
- `file`: The uploaded document file (either plain text, .doc/.docx, PDF, etc.).
- `llm_model_name`: The model used to calculate embeddings (optional).
- `json_format`: The format of the JSON response (optional, see details below).
- `send_back_json_or_zip_file`: Whether to return a JSON file or a ZIP file containing the embeddings file (optional, defaults to `zip`).
- `token`: Security token (optional).

### JSON Format Options:
The format of the JSON string returned by the endpoint (default is `records`; these are the options supported by the Pandas `to_json()` function):

- `split` : dict like {`index` -> [index], `columns` -> [columns], `data` -> [values]}
- `records` : list like [{column -> value}, … , {column -> value}]
- `index` : dict like {index -> {column -> value}}
- `columns` : dict like {column -> {index -> value}}
- `values` : just the values array
- `table` : dict like {`schema`: {schema}, `data`: {data}}

### Examples:
- Plain Text: Submit a file containing plain text.
- MS Word: Submit a `.doc` or `.docx` file.
- PDF: Submit a `.pdf` file.""",
        response_description="Either a ZIP file containing the embeddings JSON file or a direct JSON response, depending on the value of `send_back_json_or_zip_file`.")
async def get_all_embedding_vectors_for_document(file: UploadFile = File(...),
                                                llm_model_name: str = "bge-m3-q8_0",
                                                json_format: str = 'records',
                                                token: str = None,
                                                send_back_json_or_zip_file: str = 'zip',
                                                req: Request = None) -> Response:
    client_ip = req.client.host if req else "localhost"
    request_time = datetime.utcnow() 
    if USE_SECURITY_TOKEN and use_hardcoded_security_token and (token is None or token != SECURITY_TOKEN): raise HTTPException(status_code=403, detail="Unauthorized")  # noqa: E701
    _, extension = os.path.splitext(file.filename)
    temp_file = tempfile.NamedTemporaryFile(suffix=extension, delete=False)
    temp_file_path = temp_file.name
    with open(temp_file_path, 'wb') as buffer:
        chunk_size = 1024
        chunk = await file.read(chunk_size)
        while chunk:
            buffer.write(chunk)
            chunk = await file.read(chunk_size)
    hash_obj = sha3_256()
    with open(temp_file_path, 'rb') as buffer:
        for chunk in iter(lambda: buffer.read(chunk_size), b''):
            hash_obj.update(chunk)
    file_hash = hash_obj.hexdigest()
    logger.info(f"SHA3-256 hash of submitted file: {file_hash}")
    unique_id = f"document_embedding_{file_hash}_{llm_model_name}"
    lock = await shared_resources.lock_manager.lock(unique_id)        
    if lock.valid:    
        try:        
            async with AsyncSessionLocal() as session: # Check if the document has been processed before
                result = await session.execute(select(DocumentEmbedding).filter(DocumentEmbedding.file_hash == file_hash, DocumentEmbedding.llm_model_name == llm_model_name))
                existing_document_embedding = result.scalar_one_or_none()
                if existing_document_embedding: # If the document has been processed before, return the existing result
                    logger.info(f"Document {file.filename} has been processed before, returning existing result")
                    json_content = json.dumps(existing_document_embedding.document_embedding_results_json).encode()
                else: # If the document has not been processed, continue processing
                    mime = Magic(mime=True)
                    mime_type = mime.from_file(temp_file_path)            
                    logger.info(f"Received request to extract embeddings for document {file.filename} with MIME type: {mime_type} and size: {os.path.getsize(temp_file_path)} bytes from IP address: {client_ip}")
                    strings = await parse_submitted_document_file_into_sentence_strings_func(temp_file_path, mime_type)
                    results = await compute_embeddings_for_document(strings, llm_model_name, client_ip, file_hash) # Compute the embeddings and json_content for new documents
                    df = pd.DataFrame(results, columns=['text', 'embedding'])
                    json_content = df.to_json(orient=json_format or 'records').encode()
                    with open(temp_file_path, 'rb') as file_buffer: # Store the results in the database
                        original_file_content = file_buffer.read()
                    await store_document_embeddings_in_db(file, file_hash, original_file_content, json_content, results, llm_model_name, client_ip, request_time)
            overall_total_time = (datetime.utcnow() - request_time).total_seconds()
            logger.info(f"Done getting all embeddings for document {file.filename} containing {len(strings)} with model {llm_model_name}")
            json_content_length = len(json_content)
            if len(json_content) > 0:
                logger.info(f"The response took {overall_total_time} seconds to generate, or {overall_total_time / (len(strings)/1000.0)} seconds per thousand input tokens and {overall_total_time / (float(json_content_length)/1000000.0)} seconds per million output characters.")
            if send_back_json_or_zip_file == 'json': # Assume 'json' response should be sent back
                logger.info(f"Returning JSON response for document {file.filename} containing {len(strings)} with model {llm_model_name}; first 100 characters out of {json_content_length} total of JSON response: {json_content[:100]}")
                return JSONResponse(content=json.loads(json_content.decode())) # Decode the content and parse it as JSON
            else: # Assume 'zip' file should be sent back
                original_filename_without_extension, _ = os.path.splitext(file.filename)
                json_file_path = f"/tmp/{original_filename_without_extension}.json"
                with open(json_file_path, 'wb') as json_file: # Write the JSON content as bytes
                    json_file.write(json_content)
                zip_file_path = f"/tmp/{original_filename_without_extension}.zip"
                with zipfile.ZipFile(zip_file_path, 'w') as zipf:
                    zipf.write(json_file_path, os.path.basename(json_file_path))
                logger.info(f"Returning ZIP response for document {file.filename} containing {len(strings)} with model {llm_model_name}; first 100 characters out of {json_content_length} total of JSON response: {json_content[:100]}")
                return FileResponse(zip_file_path, headers={"Content-Disposition": f"attachment; filename={original_filename_without_extension}.zip"})
        finally:
            await shared_resources.lock_manager.unlock(lock)
    else:
        return {"status": "already processing"}



@app.post("/get_text_completions_from_input_prompt/",
        response_model=List[TextCompletionResponse],
        summary="Generate Text Completions for a Given Input Prompt",
        description="""Generate text completions for a given input prompt string using the specified model.
### Parameters:
- `request`: A JSON object containing the input prompt string (`input_prompt`), the model name, an optional grammar file, an optional number of tokens to generate, and an optional number of completions to generate.
- `token`: Security token (optional).

### Request JSON Format:
The request must contain the following attributes:
- `input_prompt`: The input prompt from which to generate a completion with the LLM model.
- `llm_model_name`: The model used to calculate the embedding (optional, will use the default model if not provided).
- `temperature`: The temperature to use for text generation (optional, defaults to 0.7).
- `grammar_file_string`: The grammar file used to restrict text generation (optional; default is to not use any grammar file). Examples: `json`, `list`)
- `number_of_completions_to_generate`: The number of completions to generate (optional, defaults to 1).
- `number_of_tokens_to_generate`: The number of tokens to generate (optional, defaults to 1000).

### Example (note that `llm_model_name` is optional):
```json
{
    "input_prompt": "The Kings of France in the 17th Century:",
    "llm_model_name": "Meta-Llama-3-8B-Instruct",
    "temperature": 0.95,
    "grammar_file_string": "json",
    "number_of_tokens_to_generate": 500,
    "number_of_completions_to_generate": 3
}
```

### Response:
The response will include the generated text completion, the time taken to compute the generation in seconds, and the request details (input prompt, model name, grammar file, and number of tokens to generate).

### Example Response:
```json
[
    {
        "input_prompt": "The Kings of France in the 17th Century:",
        "llm_model_name": "Meta-Llama-3-8B-Instruct",
        "grammar_file_string": "json",
        "number_of_tokens_to_generate": 500,
        "number_of_completions_to_generate": 3,
        "time_taken_in_seconds": 67.17598033333333,
        "generated_text": "{\"kings\":[\\n    {\\n        \"name\": \"Henry IV\",\\n        \"reign_start\": 1589,\\n        \"reign_end\": 1610\\n    },\\n    {\\n        \"name\": \"Louis XIII\",\\n        \"reign_start\": 1610,\\n        \"reign_end\": 1643\\n    },\\n    {\\n        \"name\": \"Louis XIV\",\\n        \"reign_start\": 1643,\\n        \"reign_end\": 1715\\n    },\\n    {\\n        \"name\": \"Louis XV\",\\n        \"reign_start\": 1715,\\n        \"reign_end\": 1774\\n    },\\n    {\\n        \"name\": \"Louis XVI\",\\n        \"reign_start\": 1774,\\n        \"reign_end\": 1792\\n    }\\n]}",
        "llm_model_usage_json": "{\"prompt_tokens\": 13, \"completion_tokens\": 218, \"total_tokens\": 231}"
    },
    {
        "input_prompt": "The Kings of France in the 17th Century:",
        "llm_model_name": "Meta-Llama-3-8B-Instruct",
        "grammar_file_string": "json",
        "number_of_tokens_to_generate": 500,
        "number_of_completions_to_generate": 3,
        "time_taken_in_seconds": 67.17598033333333,
        "generated_text": "{\"kings\":\\n   [ {\"name\": \"Henry IV\",\\n      \"reignStart\": \"1589\",\\n      \"reignEnd\": \"1610\"},\\n     {\"name\": \"Louis XIII\",\\n      \"reignStart\": \"1610\",\\n      \"reignEnd\": \"1643\"},\\n     {\"name\": \"Louis XIV\",\\n      \"reignStart\": \"1643\",\\n      \"reignEnd\": \"1715\"}\\n   ]}",
        "llm_model_usage_json": "{\"prompt_tokens\": 13, \"completion_tokens\": 115, \"total_tokens\": 128}"
    },
    {
        "input_prompt": "The Kings of France in the 17th Century:",
        "llm_model_name": "Meta-Llama-3-8B-Instruct",
        "grammar_file_string": "json",
        "number_of_tokens_to_generate": 500,
        "number_of_completions_to_generate": 3,
        "time_taken_in_seconds": 67.17598033333333,
        "generated_text": "{\\n\"Henri IV\": \"1589-1610\",\\n\"Louis XIII\": \"1610-1643\",\\n\"Louis XIV\": \"1643-1715\",\\n\"Louis XV\": \"1715-1774\",\\n\"Louis XVI\": \"1774-1792\",\\n\"Louis XVIII\": \"1814-1824\",\\n\"Charles X\": \"1824-1830\",\\n\"Louis XIX (previously known as Charles X): \" \\n    : \"1824-1830\",\\n\"Charles X (previously known as Louis XIX)\": \"1824-1830\"}",
        "llm_model_usage_json": "{\"prompt_tokens\": 13, \"completion_tokens\": 168, \"total_tokens\": 181}"
    }
]
```""", response_description="A JSON object containing the the generated text completion of the input prompt and the request details.")
async def get_text_completions_from_input_prompt(request: TextCompletionRequest, req: Request = None, token: str = None, client_ip: str = None) -> List[TextCompletionResponse]:
    if USE_SECURITY_TOKEN and use_hardcoded_security_token and (token is None or token != SECURITY_TOKEN):
        logger.warning(f"Unauthorized request from client IP {client_ip}")
        raise HTTPException(status_code=403, detail="Unauthorized")
    try:
        unique_id = f"text_completion_{hash(request.input_prompt)}_{request.llm_model_name}"
        lock = await shared_resources.lock_manager.lock(unique_id)        
        if lock.valid:
            try:    
                return await generate_completion_from_llm(request, req, client_ip)
            finally:
                await shared_resources.lock_manager.unlock(lock)
        else:
            return {"status": "already processing"}
    except Exception as e:
        logger.error(f"An error occurred while processing the request: {e}")
        logger.error(traceback.format_exc()) # Print the traceback
        raise HTTPException(status_code=500, detail="Internal Server Error")



@app.post("/turn_sample_json_into_bnf_grammar_for_llm/",
        summary="Generate BNF Grammar from Sample JSON",
        description="""Generate BNF grammar based on a sample JSON string or an uploaded JSON file, with optional token-based authentication.
### Parameters:
- `sample_json`: The sample JSON data as a string (optional if file is uploaded).
- `file`: The sample JSON file to upload (optional if JSON data is provided as `sample_json`). File must be JSON type and not exceed 100KB.
- `token`: A security token for authentication (optional).

### Constraints:
- Uploaded files must be of type JSON and not exceed 100KB.

### Validation:
- The generated BNF grammar will be validated. If validation fails, an error message will be returned.

### Example Request with JSON Data:
Use `multipart/form-data` to provide `sample_json` as a string.

### Example Request with File Upload:
Use `multipart/form-data` to upload a JSON file.

### Example Request with Token:
Add a `token` parameter to your request for authentication.

### Response:
The response will be the generated BNF grammar based on the sample JSON provided. If the generated BNF grammar fails validation, an error message will be returned.

### Example Response:
"root ::= '{' ws root_pair_list ws '}' ws ..."
""",
        response_description="A string containing the generated BNF grammar, or an error message if the grammar fails validation.")
async def turn_sample_json_into_bnf_grammar_for_llm(
    sample_json: str = Form(None),
    file: UploadFile = File(None),
    token: str = Form(None)
) -> str:
    if sample_json is None and file is None:
        raise HTTPException(status_code=400, detail="Either sample_json or file must be provided")
    if file:
        if file.content_type != "application/json":
            raise HTTPException(status_code=400, detail="Invalid file type. Only JSON is accepted.")
        file_size = file.file._file.tell()
        if file_size > 102400:
            raise HTTPException(status_code=400, detail="File size exceeds 100KB.")
    gb = GrammarBuilder()
    if sample_json:
        try:
            json_content = json.loads(sample_json)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
        bnf_grammar = gb.json_to_bnf(json.dumps(json_content))
    else:
        file_content = await file.read()
        try:
            json_content = json.loads(file_content.decode('utf-8'))
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
        bnf_grammar = gb.json_to_bnf(json.dumps(json_content))
    is_valid_grammar, validation_message = validate_bnf_grammar_func(bnf_grammar)
    if not is_valid_grammar:
        raise HTTPException(status_code=400, detail=f"Generated BNF grammar could not be validated: {validation_message}")
    return bnf_grammar



@app.post("/turn_pydantic_model_description_into_bnf_grammar_for_llm/",
        summary="Generate BNF Grammar from Pydantic Model Description",
        description="""Generate BNF grammar based on a Pydantic model description string. This endpoint allows you to turn a Pydantic model definition into a corresponding BNF grammar.
        
### Parameters:
- `pydantic_model_description`: The Pydantic model description as a string. Must include the class definition, fields, and their types.

### Validation:
- The generated BNF grammar will be validated. If validation fails, an error message will be returned.

### Authentication:
- `token`: Security token for authorized access (optional if security is disabled).

### Example Request:
Use `multipart/form-data` to provide `pydantic_model_description` as a string.

### Response:
The response will be the generated BNF grammar based on the Pydantic model description provided. If the generated BNF grammar fails validation, an error message will be returned.

### Example Response:
"root ::= '{' ws root_pair_list ws '}' ws ..."
""",
        response_description="A string containing the generated BNF grammar, or an error message if the grammar fails validation.")
async def turn_pydantic_model_description_into_bnf_grammar_for_llm(
    pydantic_model_description: str = Form(...),
    token: str = Form(None)
) -> str:
    if USE_SECURITY_TOKEN and use_hardcoded_security_token and (token is None or token != SECURITY_TOKEN):
        raise HTTPException(status_code=403, detail="Unauthorized")        
    if not pydantic_model_description:
        raise HTTPException(status_code=400, detail="Pydantic model description must be provided")
    gb = GrammarBuilder()
    bnf_grammar = gb.pydantic_to_json_bnf(pydantic_model_description)
    is_valid_grammar, validation_message = validate_bnf_grammar_func(bnf_grammar)
    if not is_valid_grammar:
        raise HTTPException(status_code=400, detail=f"Generated BNF grammar could not be validated: {validation_message}")
    return bnf_grammar



@app.post("/compute_transcript_with_whisper_from_audio/",
        summary="Transcribe and Embed Audio using Whisper and LLM",
        description="""Transcribe an audio file and optionally compute document embeddings. This endpoint uses the Whisper model for transcription and a specified or default language model for embeddings. The transcription and embeddings are then stored, and a ZIP file containing the embeddings can be downloaded.

### Parameters:
- `file`: The uploaded audio file.
- `compute_embeddings_for_resulting_transcript_document`: Boolean to indicate if document embeddings should be computed (optional, defaults to True).
- `llm_model_name`: The language model used for computing embeddings (optional, defaults to the default model name).
- `req`: HTTP Request object for additional request metadata (optional).
- `token`: Security token for API access (optional).
- `client_ip`: Client IP for logging and security (optional).

### Examples:
- Audio File: Submit an audio file for transcription.
- Audio File with Embeddings: Submit an audio file and set `compute_embeddings_for_resulting_transcript_document` to True to also get embeddings.

### Authentication:
- If security tokens are enabled (`USE_SECURITY_TOKEN=True` and `use_hardcoded_security_token=True`), then the `token` parameter must match the hardcoded `SECURITY_TOKEN`.

### Error Handling:
- Unauthorized requests are logged and result in a 403 status.
- All other errors result in a 500 status and are logged with their tracebacks.""",
        response_description="A JSON object containing the complete transcription details, computational times, and an optional URL for downloading a ZIP file of the document embeddings.")
async def compute_transcript_with_whisper_from_audio(
        file: UploadFile, 
        compute_embeddings_for_resulting_transcript_document: Optional[bool] = True, 
        llm_model_name: Optional[str] = DEFAULT_MODEL_NAME, 
        req: Request = None, 
        token: str = None, 
        client_ip: str = None):
    if USE_SECURITY_TOKEN and use_hardcoded_security_token and (token is None or token != SECURITY_TOKEN):
        logger.warning(f"Unauthorized request from client IP {client_ip}")
        raise HTTPException(status_code=403, detail="Unauthorized")
    try:
        audio_transcript = await get_or_compute_transcript(file, compute_embeddings_for_resulting_transcript_document, llm_model_name, req)
        return audio_transcript
    except Exception as e:
        logger.error(f"An error occurred while processing the request: {e}")
        logger.error(traceback.format_exc())  # Print the traceback
        raise HTTPException(status_code=500, detail="Internal Server Error")



@app.post("/add_new_grammar_definition_file/",
        response_model=AddGrammarResponse,
        summary="Add or Update a Grammar Definition File",
        description="""Add a new BNF grammar definition file or update an existing one.
        
### Parameters:
- `bnf_grammar`: The BNF grammar string.
- `grammar_file_name`: The name for the new or existing grammar file.

If a grammar file with the given name already exists, this endpoint will compare the existing content with the new submission. If the content is different, the file will be overwritten.

### Example Request:
```json
{
    "bnf_grammar": "root ::= '{' ws root_pair_list ws '}' ws ...",
    "grammar_file_name": "new_grammar"
}
```

### Response:
The response will include a list of all valid grammar files in the `grammar_files` directory.

### Example Response:
```json
{
    "valid_grammar_files": ["new_grammar.gbnf", "another_grammar.gbnf"]
}
```""",
        response_description="A JSON object containing a list of all valid grammar files.")
async def add_new_grammar_definition_file(request: AddGrammarRequest, token: str = None) -> AddGrammarResponse:
    if USE_SECURITY_TOKEN and use_hardcoded_security_token and (token is None or token != SECURITY_TOKEN):
        raise HTTPException(status_code=403, detail="Unauthorized")    
    is_valid_grammar, validation_message = validate_bnf_grammar_func(request.bnf_grammar)
    if not is_valid_grammar:
        raise HTTPException(status_code=400, detail=f"Invalid BNF grammar: {validation_message}")
    grammar_files_dir = 'grammar_files'
    if not os.path.exists(grammar_files_dir):
        os.makedirs(grammar_files_dir)
    grammar_file_name_with_extension = f"{request.grammar_file_name}.gbnf"
    grammar_file_path = Path(grammar_files_dir) / grammar_file_name_with_extension
    existing_files = await get_list_of_available_bnf_grammars()
    if grammar_file_name_with_extension in existing_files:
        with open(grammar_file_path, "r") as f:
            existing_content = f.read()
        if existing_content != request.bnf_grammar:
            logger.info(f"Grammar file {grammar_file_name_with_extension} already exists, but newly submitted grammar is different-- overwriting!")
            with open(grammar_file_path, "w") as f:
                f.write(request.bnf_grammar)
        else:
            logger.info(f"Grammar file {grammar_file_name_with_extension} already exists and is the same-- not overwriting!")
    else:
        logger.info(f"Grammar file {grammar_file_name_with_extension} does not exist-- creating!")
        with open(grammar_file_path, "w") as f:
            f.write(request.bnf_grammar)
    valid_grammar_files = [f.name for f in Path(grammar_files_dir).glob("*.gbnf")]
    return {"valid_grammar_files": valid_grammar_files}



@app.post("/clear_ramdisk/")
async def clear_ramdisk_endpoint(token: str = None):
    if USE_SECURITY_TOKEN and use_hardcoded_security_token and (token is None or token != SECURITY_TOKEN):
        raise HTTPException(status_code=403, detail="Unauthorized")
    if USE_RAMDISK:
        clear_ramdisk()
        return {"message": "RAM Disk cleared successfully."}
    return {"message": "RAM Disk usage is disabled."}



@app.get("/download/{file_name}",
        summary="Download File by Name",
        description="""Download a file by its name from the 'generated_transcript_embeddings_zip_files' directory, with optional token-based authentication.
### Parameters:
- `file_name`: The name of the file to download.
- `token`: A security token for authentication (optional).

### Example Request with Token:
Add a `token` parameter to your request for authentication.

### Response:
The response will be the requested file in ZIP format if it exists, or a 404 status code if the file is not found.

### Security:
If a security token is required by the application configuration, you must provide a valid `token` to access this endpoint. Unauthorized access will result in a 403 status code.""",
        response_description="The ZIP file that was requested, or a status code indicating an error.")
async def download_file(file_name: str, token: str = None):
    if USE_SECURITY_TOKEN and use_hardcoded_security_token and (token is None or token != SECURITY_TOKEN):
        raise HTTPException(status_code=403, detail="Unauthorized")    
    decoded_file_name = unquote(file_name)
    file_path = os.path.join("generated_transcript_embeddings_zip_files", decoded_file_name)
    absolute_file_path = os.path.abspath(file_path)
    logger.info(f"Trying to fetch file from: {absolute_file_path}")
    if os.path.exists(absolute_file_path):
        with open(absolute_file_path, 'rb') as f:
            logger.info(f"File first 10 bytes: {f.read(10)}")
        return FileResponse(absolute_file_path, media_type="application/zip", filename=decoded_file_name)
    else:
        logger.error(f"File not found at: {absolute_file_path}")
        raise HTTPException(status_code=404, detail="File not found")
    


@app.get("/show_logs_incremental/{minutes}/{last_position}", response_model=ShowLogsIncrementalModel)
def show_logs_incremental(minutes: int, last_position: int):
    return show_logs_incremental_func(minutes, last_position)



@app.get("/show_logs/{minutes}", response_class=HTMLResponse)
def show_logs(minutes: int = 5):
    return show_logs_func(minutes)



        
@app.get("/show_logs",
        response_class=HTMLResponse,
        summary="Show Recent Logs",
        description="""Displays the most recent logs from the 'swiss_army_llama.log' file, highlighting key words and patterns for easier readability. The logs are displayed as HTML content with inline styles and Javascript for dynamic refreshing.
        
### Behavior:
- By default, shows logs from the last 5 minutes.
- Log entries are color-coded based on keywords like 'success', 'error', 'pending', etc.
- Log entries are continuously fetched every 20 seconds.
- Provides options to copy and download the logs.

### Response:
The response will be an HTML page that displays the logs in a human-readable format, highlighting various types of information.

### Additional Features:
- Users can copy or download the logs using buttons provided on the page.""",
        response_description="An HTML page containing the most recent logs with dynamic updating and interactive features.")
def show_logs_default():
    return show_logs_func(5)

if __name__ == "__main__":
    try:
        uvicorn.run("swiss_army_llama:app", **option)
    except GracefulExit:
        logger.info("Received signal to terminate. Shutting down gracefully...")
        sys.exit(0)
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt. Shutting down gracefully...")
        sys.exit(0)
    except Exception:
        logger.exception("Unhandled exception occurred during shutdown.")
        sys.exit(1)
