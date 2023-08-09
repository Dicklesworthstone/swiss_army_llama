# LLama2 Embeddings FastAPI Service

## Introduction

The LLama2 Embedding Server is designed to facilitate and optimize the process of obtaining text embeddings using different LLMs via llama_cpp and langchain. The driving motivation behind this project is to offer a convenient and easy to use API to quickly and easily submit text strings and get back embeddings using Llama2 and similar LLMs. To avoid wasting computation, these embeddings are cached in SQlite and retrieved if they have already been computed before. To speed up the process of loading multiple LLMs, optional RAM Disks can be used, and the process for creating and managing them is handled automatically for you. Finally, some additional useful endpoints are provided, such as computing semantic similarity between submitted text strings, and semantic search across all your cached embeddings using FAISS vector searching. You can also submit a plaintext file or PDF file (not requiring OCR) and get back a zip file containing all of the embeddings for each sentence as JSON, organized in various ways such `records`, `table`, etc. (all the options for the Pandas `to_json()` function).

![Llama2 FastAPI Service Swagger UI](https://github.com/Dicklesworthstone/llama_embeddings_fastapi_service/raw/main/Llama2-FastAPI-Service-%20Swagger%20Screenshot.png)

---

## Features

1. **Text Embedding Computation**: Utilizes pre-trained LLama2 (and other supported models) to generate embeddings for any provided text.
2. **Embedding Caching**: Efficiently stores and retrieves computed embeddings in a SQLite database, minimizing redundant computations.
3. **Similarity Measurements and Retrieval**: Not only computes the cosine similarity between two strings, but also has the capability to find the most similar string in the database compared to a provided input.
4. **Advanced File Processing**: Can submit a text file or PDF and quickly compute embeddings for each sentence, either in parallel or sequentially.
5. **RAM Disk Usage**: Optionally uses RAM Disk to store models, offering significantly faster access and execution.
6. **Robust Exception Handling**: Features comprehensive exception management to ensure system resilience.
7. **Interactive API Documentation**: Integrates with Swagger UI for an interactive and user-friendly API documentation experience.
8. **Scalability and Concurrency**: Built on the FastAPI framework, the system can efficiently handle concurrent requests. It supports parallel inference with configurable concurrency levels.
9. **Flexible Configurations**: Provides configurable settings through environment variables, allowing for easy adjustments based on deployment needs.
10. **Comprehensive Logging**: Maintains detailed logs, streamlined to capture essential information without overwhelming storage or readability.
11. **Support for Multiple Models**: While LLama2 is highlighted, the system is designed to accommodate multiple embedding models.

## Dependencies
- FastAPI
- SQLAlchemy
- numpy
- faiss
- pandas
- pyPDF2
- magic
- decouple
- uvicorn
- psutil
- And other Python standard libraries.

## Running the Application

You can run the application using the following command:

```bash
python llama_2_embeddings_fastapi_server.py
```

The server will start on `0.0.0.0` at the port defined by the `LLAMA_EMBEDDING_SERVER_LISTEN_PORT` variable.

Access the Swagger UI:

```
http://localhost:<LLAMA_EMBEDDING_SERVER_LISTEN_PORT>
```

## Configuration

You can configure the service using environment variables. Here's a list of available configuration options:

- `USE_SECURITY_TOKEN`: Whether to use a hardcoded security token.
- `DATABASE_URL`: SQLite database URL.
- `LLAMA_EMBEDDING_SERVER_LISTEN_PORT`: Port number for the service.
- `DEFAULT_MODEL_NAME`: Default model name to use.
- `MINIMUM_STRING_LENGTH_FOR_DOCUMENT_EMBEDDING`: Minimum string length for document embedding.
- `USE_PARALLEL_INFERENCE_QUEUE`: Use parallel processing.
- `MAX_CONCURRENT_PARALLEL_INFERENCE_TASKS`: Maximum number of parallel inference tasks.
- `USE_RAMDISK`: Use RAM disk.
- `RAMDISK_SIZE_IN_GB`: RAM disk size.
- `MAX_RETRIES`: Maximum retries for locked database.
- `DB_WRITE_BATCH_SIZE`: Database write batch size.
- `RETRY_DELAY_BASE_SECONDS`: Retry delay base in seconds.
- `JITTER_FACTOR`: Jitter factor for retries.

## Contributing

If you'd like to contribute to the project, please submit a pull request.

## License

This project is licensed under the MIT License.

---

This section encompasses all the unique points from both lists while ensuring a cohesive and consistent presentation.

## Requirements

- Python 3.9+
- Libraries: `numpy`, `fastapi`, `uvicorn`, `sqlalchemy`, `faiss`, `psutil`, `sklearn`, `decouple`, `asyncio`, `subprocess`, `traceback`, `glob`, `shutil`, `urllib`, `json`, `logging`, `datetime`

## Setup and Configuration

### Environment Variables

The application can be configured using environment variables or hardcoded values. The following environment variables can be set:

- `USE_SECURITY_TOKEN`: Enables the use of a security token for API access (default: True).
- `USE_PARALLEL_INFERENCE_QUEUE`: Enables the use of a parallel inference queue (default: True).
- `MAX_CONCURRENT_PARALLEL_INFERENCE_TASKS`: Maximum number of concurrent parallel inference tasks (default: 30).
- `DEFAULT_MODEL_NAME`: Specifies the default model name to be used (default: llama2_7b_chat_uncensored).
- `LLAMA_EMBEDDING_SERVER_LISTEN_PORT`: The port on which the server will listen (default: 8089).
- `MINIMUM_STRING_LENGTH_FOR_DOCUMENT_EMBEDDING`: Minimum length for strings to be considered for document embedding (default: 15).
- `MAX_RETRIES`: Maximum number of retries for failed operations (default: 10).
- `DB_WRITE_BATCH_SIZE`: The batch size for database write operations (default: 25).
- `RETRY_DELAY_BASE_SECONDS`: Base delay for retries in seconds (default: 1).
- `JITTER_FACTOR`: Jitter factor for retry delays (default: 0.1).
- `USE_RAMDISK`: Enables the use of RAM Disk (default: True).
- `RAMDISK_SIZE_IN_GB`: The size of the RAM Disk in GB (default: 40).

### RAM Disk Configuration

To enable password-less sudo for RAM Disk setup and teardown, edit the `sudoers` file with `sudo visudo`. Add the following lines, replacing `username` with your actual username:

```plaintext
username ALL=(ALL) NOPASSWD: /bin/mount -t tmpfs -o size=*G tmpfs /mnt/ramdisk
username ALL=(ALL) NOPASSWD: /bin/umount /mnt/ramdisk
```

The application provides functionalities to set up, clear, and manage RAM Disk. RAM Disk is used to store models in memory for faster access. It calculates the available RAM and sets up the RAM Disk accordingly. The functions `setup_ramdisk`, `copy_models_to_ramdisk`, and `clear_ramdisk` manage these tasks.

## API Endpoints

The following endpoints are available:

- **GET `/get_list_of_available_model_names/`**: Retrieves a list of available model names.
- **GET `/get_all_stored_strings/`**: Retrieves all strings that have been stored in the db.
- **POST `/get_embedding_vector/`**: Retrieves or computes the embedding vector for a given text.
- **POST `/compute_similarity_between_strings/`**: Computes the similarity between two strings.
- **POST `/get_most_similar_string_from_database/`**: Finds the most similar string from the database.
- **Post `/get_all_embeddings_for_document/`**: Computes all the embeddings in parallel for every sentence in an uploaded text file or PDF.
- **POST `/clear_ramdisk/`**: Clears the RAM Disk if it is enabled.

For detailed request and response schemas, please refer to the Swagger UI available at the root URL or the section at the end of this `README`.

## Database Structure

The application uses a SQLite database to store computed embeddings. There are two main tables:

### TextEmbedding Table
This table stores individual text embeddings.

- `id`: Primary Key
- `text`: Text for which the embedding was computed
- `text_hash`: Hash of the text, computed using SHA3-256
- `model_name`: Model used to compute the embedding
- `embedding_json`: The computed embedding in JSON format
- `ip_address`: Client IP address
- `request_time`: Timestamp of the request
- `response_time`: Timestamp of the response
- `total_time`: Total time taken to process the request
- `document_id`: Foreign Key referencing the DocumentEmbedding table
- Unique Constraint on `text_hash` and `model_name`

### DocumentEmbedding Table
This table stores embeddings for entire documents.

- `id`: Primary Key
- `document_id`: Foreign Key referencing the Documents table
- `filename`: Name of the document file
- `mimetype`: MIME type of the document file
- `file_hash`: Hash of the file
- `model_name`: Model used to compute the embedding
- `file_data`: Binary data of the original file
- `results_json`: The computed embedding results in JSON format
- Unique Constraint on `file_hash` and `model_name`

## Exception Handling

The application has robust exception handling to deal with various types of errors, including database errors and general exceptions. Custom exception handlers are defined for `SQLAlchemyError` and general `Exception`.

## Logging

Logging is configured at the INFO level to provide detailed logs for debugging and monitoring. The logger provides information about the state of the application, errors, and activities.

The logs are stored in a file named `llama2_embeddings_fastapi_service.log`, and a log rotation mechanism is implemented to handle log file backups. The rotating file handler is configured with a maximum file size of 10 MB, and it keeps up to 5 backup files.

When a log file reaches its maximum size, it is moved to the `old_logs` directory, and a new log file is created. The log entries are also printed to the standard output stream.

Here are some details of the logging configuration:
- Log Level: INFO
- Log Format: `%(asctime)s - %(levelname)s - %(message)s`
- Max Log File Size: 10 MB
- Backup Count: 5
- Old Logs Directory: `old_logs`

Additionally, the log level for SQLAlchemy's engine is set to WARNING to suppress verbose database logs.

## Startup Procedures

During startup, the application performs the following tasks:

1. **Initialize Database Writer Queue**: A dedicated writer queue is initialized to handle database write operations.
2. **RAM Disk Setup**: If enabled and the user has the required permissions, the application sets up a RAM Disk.
3. **Model Downloads**: The application downloads the required models.
4. **Model Loading**: Each downloaded model is loaded into memory. If any model file is not found, an error log is recorded.
5. **Database Initialization**: The application initializes the database, setting up tables and relationships.
6. **FAISS Index Building**: The application builds the FAISS indexes for efficient similarity search.

Note: If RAM Disk is enabled but the user lacks the required permissions, the application will disable the RAM Disk feature and proceed without it.


## Performance Optimizations

This section highlights the major performance enhancements integrated into the provided code to ensure swift responses and optimal resource management.

### 1. **Asynchronous Programming**:
- **Benefit**: Handles multiple tasks concurrently, enhancing efficiency for I/O-bound operations like database transactions and network requests.
- **Implementation**: Utilizes Python's `asyncio` library for asynchronous database operations.

### 2. **Database Optimizations**:
- **Write-Ahead Logging (WAL) Mode**: Enables concurrent reads and writes, optimizing for applications with frequent write demands.
- **Retry Logic with Exponential Backoff**: Manages locked databases by retrying operations with progressive waiting times.
- **Batch Writes**: Aggregates write operations for more efficient database interactions.
- **DB Write Queue**: Uses an asynchronous queue to serialize write operations, ensuring consistent and non-conflicting database writes.

### 3. **RAM Disk Utilization**:
- **Benefit**: Speeds up I/O-bound tasks by prioritizing operations in RAM over disk.
- **Implementation**: Detects and prioritizes a RAM disk (`/mnt/ramdisk`) if available, otherwise defaults to the standard file system.

### 4. **Model Caching**:
- **Benefit**: Reduces overhead by keeping loaded models in memory for subsequent requests.
- **Implementation**: Uses a global `model_cache` dictionary to store and retrieve models.

### 5. **Parallel Inference**:
- **Benefit**: Enhances processing speed for multiple data units, like document sentences.
- **Implementation**: Employs `asyncio.gather` for concurrent inferences, regulated by a semaphore (`MAX_CONCURRENT_PARALLEL_INFERENCE_TASKS`).

### 6. **Embedding Caching**:
- **Benefit**: Once embeddings are computed for a particular text, they are stored in the database, eliminating the need for re-computation during subsequent requests.
- **Implementation**: When a request is made to compute an embedding, the system first checks the database. If the embedding for the given text is found, it is returned immediately, ensuring faster response times.


## Endpoint Functionality and Workflow Overview
Here's a detailed breakdown of the main endpoints provided by the LLama Embedding Server, explaining their functionality, input parameters, and how they interact with underlying models and systems:

### 1. `/get_embedding_vector/` (POST)

#### Purpose
This endpoint computes or retrieves the embedding vector for a given text and model name. If the embedding is already cached in the database, it returns the cached result. Otherwise, it computes the embedding using the specified model and caches the result.

#### Parameters
- `text`: The text for which the embedding is to be calculated.
- `model_name`: The name of the model to be used. Defaults to a pre-configured model.
- `token`: Security token (optional).
- `client_ip`: Client IP address (optional).

#### Workflow
1. **Check Database**: The function first checks if the embedding for the given text and model name is already present in the database.
2. **Compute Embedding**: If not found in the database, the specified model is loaded (or retrieved from the cache if previously loaded), and the embedding is calculated.
3. **Cache Embedding**: The computed embedding is then serialized and stored in the database along with metadata like IP address, request time, and total time taken.
4. **Return Result**: Finally, the embedding is returned in the response.

### 2. `/compute_similarity_between_strings/` (POST)

#### Purpose
This endpoint calculates the cosine similarity between two given strings. It utilizes the embeddings of the strings, either by retrieving them from the cache or computing them on-the-fly.

#### Parameters
- `text`: The query text for which the most similar string is to be found.
- `model_name`: The name of the model to be used for the query text's embedding.
- `number_of_most_similar_strings_to_return`: (Optional) The number of most similar strings to return, defaults to 3.
- `token`: Security token (optional).

#### Workflow
1. **Retrieve Embeddings**: The embeddings for `text1` and `text2` are either retrieved from the database or computed using the specified model.
2. **Compute Similarity**: The cosine similarity between the two embeddings is calculated using the `cosine_similarity` function from `sklearn.metrics.pairwise`.
3. **Return Result**: The similarity score, along with the embeddings and input texts, is returned in the response.

### 3. `/get_most_similar_string_from_database/` (POST)

#### Purpose
This endpoint searches the cached embeddings in the database to find the most similar string to a given query string.

#### Parameters
- `text`: The query text for which the most similar string is to be found.
- `model_name`: The name of the model to be used for the query text's embedding.

#### Workflow
1. **Compute Query Embedding**: The embedding for the query text is computed or retrieved from the database.
2. **Search Faiss Index**: The FAISS (Facebook AI Similarity Search) index, built on the cached embeddings, is searched to find the index of the most similar embedding.
3. **Retrieve Similar Text**: The corresponding text for the most similar embedding is retrieved from the `associated_texts` array.
4. **Return Result**: The most similar text, along with the similarity score, is returned in the response.

### 4. `/get_all_embeddings_for_document/` (POST)

#### Purpose
Extract text embeddings for a document, supporting both plain text and PDF files (OCR not supported).

#### Parameters
- `file`: The uploaded document file (either plain text or PDF).
- `model_name`: (Optional) The model used to calculate embeddings.
- `json_format`: (Optional) The format of the JSON response (see details in API documentation).
- `token`: Security token (optional).

### 5. `/get_list_of_available_model_names/` (GET)

#### Purpose
This endpoint provides the list of available model names that can be used to compute embeddings.

#### Parameters
- `token`: Security token (optional).

### 6. `/clear_ramdisk/` (POST)

#### Purpose
If RAM Disk usage is enabled, this endpoint clears the RAM Disk, freeing up memory.

#### Workflow
1. **Check RAM Disk Usage**: If RAM Disk usage is disabled, a message is returned indicating so.
2. **Clear RAM Disk**: If enabled, the RAM Disk is cleared using the `clear_ramdisk` function.
3. **Return Result**: A success message is returned in the response.

### 7. `/get_all_stored_strings/` (GET)

#### Purpose
Retrieve a list of all stored strings from the database for which embeddings have been computed.

#### Parameters
- `token`: Security token (optional).

### 8. `/get_all_stored_documents_with_embeddings/` (GET)

#### Purpose
Retrieve a list of all stored documents from the database for which embeddings have been computed.

#### Parameters
- `token`: Security token (optional).

These endpoints collectively offer a versatile set of tools and utilities to work with text embeddings, efficiently utilizing cached results, and providing useful functionalities like similarity computation and similarity search. By encapsulating complex operations behind a simple and well-documented API, they make working with LLMs via llama_cpp and langchain accessible and efficient.
