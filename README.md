# Llama2 Embeddings FastAPI Service

## Introduction

The Llama2 Embedding Server is designed to facilitate and optimize the process of obtaining text embeddings using different LLMs via llama_cpp and langchain. To avoid wasting computation, these embeddings are cached in SQlite and retrieved if they have already been computed before. To speed up the process of loading multiple LLMs, optional RAM Disks can be used, and the process for creating and managing them is handled automatically for you. 

Some additional useful endpoints are provided, such as computing semantic similarity between submitted text strings (using various measures of similarity, such as cosine similarity, but also more esoteric measures like [Hoeffding's D](https://blogs.sas.com/content/iml/2021/05/03/examples-hoeffding-d.html) and [HSIC](https://www.sciencedirect.com/science/article/abs/pii/S0950705121008297), and semantic search across all your cached embeddings using FAISS vector searching. 

You can also submit a plaintext file or PDF file (not requiring OCR) and get back a zip file containing all of the embeddings for each sentence as JSON, organized in various ways such `records`, `table`, etc. (i.e., all the export options from the Pandas `to_json()` function). The results of getting the embeddings for all sentences in a document can be returned either as a zip file containing a JSON file (so it won't crash Swagger among other things), or as a direct JSON response if you're using curl or similar.

In addition to fixed-sized embedding vectors, we also expose functionality that allows you to get back token-level embeddings, where each token in the input stream is embedded with its context in the string as a full sized vector, thus producing a matrix that has a number of rows equal to the number of tokens in the input string. This includes far more nuanced information about the contents of the string at the expense of much greater compute and storage requirements. The other drawback is that, instead of having the same sized output for every string, regardless of length (which makes it very easy to compare unequal length strings using cosine similarity and other measures), the token-level embedding matrix obviously differs in dimensions for two different strings if the strings have different numbers of tokens. To deal with this, we introduce combined feature vectors, which compute the column-wise mean, min, max, and std. deviation of the token-level emeddding matrix, and concatenate these together in to a single huge matrix; this allows you to compare strings of different lengths while still capture more nuance. The combined results, including the embedding matrix and associated combined feature vector, can similarly be returned as either a zip file or direct JSON response.


![Llama2 FastAPI Service Swagger UI](https://github.com/Dicklesworthstone/llama_embeddings_fastapi_service/raw/main/Llama2-FastAPI-Service-%20Swagger%20Screenshot.png)

*TLDR:* If you just want to try it very quickly on a fresh Ubuntu 22+ machine (warning, this will install docker using apt):

```bash
git clone https://github.com/Dicklesworthstone/llama_embeddings_fastapi_service
cd llama_embeddings_fastapi_service
chmod +x setup_dockerized_app_on_fresh_machine.sh
sudo ./setup_dockerized_app_on_fresh_machine.sh
```

Then open a browser to `<your_static_ip_address>:8089` if you're using a VPS.

Or to `localhost:8089` if you're using your own machine-- but, really, you should never run untrusted code with sudo on your own machine! Just get a cheap VPS to experiment with for $30/month.

Watch the the automated setup process in action:

[![asciicast](https://asciinema.org/a/601603.svg)](https://asciinema.org/a/601603)

---

## Features

1. **Text Embedding Computation**: Utilizes pre-trained LLama2 and other LLMs via llama_cpp and langchain to generate embeddings for any provided text, including token-level embeddings that capture more nuanced information about the content.
2. **Embedding Caching**: Efficiently stores and retrieves computed embeddings in SQLite, minimizing redundant computations. It supports caching both fixed-sized embedding vectors and token-level embeddings.
3. **Advanced Similarity Measurements and Retrieval**: Offers various measures of similarity like cosine similarity, Hoeffding's D, HSIC, and semantic search across cached embeddings using FAISS vector searching.
4. **File Processing for Documents**: Submit plaintext files or PDFs (not requiring OCR) to get back a ZIP file or JSON response containing embeddings for each sentence, organized in various ways like `records`, `table`, etc., using Pandas `to_json()` function.
5. **Token-Level Embeddings and Combined Feature Vectors**: Provides token-level embeddings to capture the context of each token in the input string. Introduces combined feature vectors by computing the column-wise mean, min, max, and std. deviation of the token-level embedding matrix, allowing comparison of unequal length strings.
6. **RAM Disk Usage**: Optionally uses RAM Disk to store models for faster access and execution. Automatically handles the creation and management of RAM Disks.
7. **Robust Exception Handling**: Features comprehensive exception management to ensure system resilience.
8. **Interactive API Documentation**: Integrates with Swagger UI for an interactive and user-friendly experience, accommodating large result sets without crashing.
9. **Scalability and Concurrency**: Built on the FastAPI framework, handles concurrent requests and supports parallel inference with configurable concurrency levels.
10. **Flexible Configurations**: Offers configurable settings through environment variables and input parameters, including response formats like JSON or ZIP files.
11. **Comprehensive Logging**: Captures essential information with detailed logs, without overwhelming storage or readability.
12. **Support for Multiple Models and Measures**: Accommodates multiple embedding models and similarity measures, allowing flexibility and customization based on user needs.

## Requirements:
```
fastapi
pydantic
uvicorn
sqlalchemy
python-decouple
psutil
aiosqlite
faiss-cpu
pandas
PyPDF2
python-multipart
python-magic
langchain
scikit-learn
llama-cpp-python
httpx
numba
scipy
hyppo
```

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

You can configure the service easily by editing the included `.env` file. Here's a list of available configuration options:

- `USE_SECURITY_TOKEN`: Whether to use a hardcoded security token. (e.g., `True`)
- `USE_PARALLEL_INFERENCE_QUEUE`: Use parallel processing. (e.g., `True`)
- `MAX_CONCURRENT_PARALLEL_INFERENCE_TASKS`: Maximum number of parallel inference tasks. (e.g., `30`)
- `DEFAULT_MODEL_NAME`: Default model name to use. (e.g., `llama2_7b_chat_uncensored`)
- `LLM_CONTEXT_SIZE_IN_TOKENS`: Context size in tokens for LLM. (e.g., `512`)
- `LLAMA_EMBEDDING_SERVER_LISTEN_PORT`: Port number for the service. (e.g., `8089`)
- `MINIMUM_STRING_LENGTH_FOR_DOCUMENT_EMBEDDING`: Minimum string length for document embedding. (e.g., `15`)
- `MAX_RETRIES`: Maximum retries for locked database. (e.g., `10`)
- `DB_WRITE_BATCH_SIZE`: Database write batch size. (e.g., `25`)
- `RETRY_DELAY_BASE_SECONDS`: Retry delay base in seconds. (e.g., `1`)
- `JITTER_FACTOR`: Jitter factor for retries. (e.g., `0.1`)
- `USE_RAMDISK`: Use RAM disk. (e.g., `True`)
- `RAMDISK_PATH`: Path to the RAM disk. (e.g., `"/mnt/ramdisk"`)
- `RAMDISK_SIZE_IN_GB`: RAM disk size in GB. (e.g., `40`)


## Contributing

If you'd like to contribute to the project, please submit a pull request.

## License

This project is licensed under the MIT License.

---

## Setup and Configuration

### RAM Disk Configuration

To enable password-less sudo for RAM Disk setup and teardown, edit the `sudoers` file with `sudo visudo`. Add the following lines, replacing `username` with your actual username:

```plaintext
username ALL=(ALL) NOPASSWD: /bin/mount -t tmpfs -o size=*G tmpfs /mnt/ramdisk
username ALL=(ALL) NOPASSWD: /bin/umount /mnt/ramdisk
```

The application provides functionalities to set up, clear, and manage RAM Disk. RAM Disk is used to store models in memory for faster access. It calculates the available RAM and sets up the RAM Disk accordingly. The functions `setup_ramdisk`, `copy_models_to_ramdisk`, and `clear_ramdisk` manage these tasks.

## API Endpoints

The following endpoints are available:

- **GET `/get_list_of_available_model_names/`**: [Retrieve Available Model Names](#). Retrieves the list of available model names for generating embeddings.
- **GET `/get_all_stored_strings/`**: [Retrieve All Strings](#). Retrieves a list of all stored strings from the database for which embeddings have been computed.
- **GET `/get_all_stored_documents/`**: [Retrieve All Stored Documents](#). Retrieves a list of all stored documents from the database for which embeddings have been computed.
- **POST `/get_embedding_vector_for_string/`**: [Retrieve Embedding Vector for a Given Text String](#). Retrieves the embedding vector for a given input text string using the specified model.
- **POST `/get_token_level_embeddings_matrix_and_combined_feature_vector_for_string/`**: [Retrieve Token-Level Embeddings and Combined Feature Vector for a Given Input String](#). Retrieve the token-level embeddings and combined feature vector for a given input text using the specified model.
- **POST `/compute_similarity_between_strings/`**: [Compute Similarity Between Two Strings](#). Compute the similarity between two given input strings using specified model embeddings and a selected similarity measure.
- **POST `/search_stored_embeddings_with_query_string_for_semantic_similarity/`**: [Get Most Similar Strings from Stored Embeddings in Database](#). Find the most similar strings in the database to the given input "query" text.
- **POST `/get_all_embedding_vectors_for_document/`**: [Get Embeddings for a Document](#). Extract text embeddings for a document, supporting both plain text and PDF files (PDFs requiring OCR are not supported).
- **POST `/clear_ramdisk/`**: [Clear Ramdisk Endpoint](#). Clears the RAM Disk if it is enabled.

For detailed request and response schemas, please refer to the Swagger UI available at the root URL or the section at the end of this `README`.

## Database Structure

The application uses a SQLite database via SQLAlchemy ORM. Here are the data models used, which can be found in the `embeddings_data_models.py` file:

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

## Database Structure

The application uses a SQLite database via SQLAlchemy ORM. Here are the data models used, which can be found in the `embeddings_data_models.py` file:

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
- `document_embedding_results_json`: The computed embedding results in JSON format
- `ip_address`: Client IP address
- `request_time`: Timestamp of the request
- `response_time`: Timestamp of the response
- `total_time`: Total time taken to process the request
- Unique Constraint on `file_hash` and `model_name`

### Document Table
This table represents a document.

- `id`: Primary Key
- `model_name`: Model name associated with the document
- `document_hash`: Hash of the document (concatenation of specific attributes from the `document_embeddings` relationship)

### TokenLevelEmbedding Table
This table stores token-level embeddings.

- `id`: Primary Key
- `token`: Token for which the embedding was computed
- `token_hash`: Hash of the token, computed using SHA3-256
- `model_name`: Model used to compute the embedding
- `token_level_embedding_json`: The computed token-level embedding in JSON format
- `ip_address`: Client IP address
- `request_time`: Timestamp of the request
- `response_time`: Timestamp of the response
- `total_time`: Total time taken to process the request
- `token_level_embedding_bundle_id`: Foreign Key referencing the TokenLevelEmbeddingBundle table
- Unique Constraint on `token_hash` and `model_name`

### TokenLevelEmbeddingBundle Table
This table stores token-level embedding bundles.

- `id`: Primary Key
- `input_text`: Input text associated with the token-level embeddings
- `input_text_hash`: Hash of the input text
- `model_name`: Model used to compute the embeddings
- `token_level_embeddings_bundle_json`: JSON containing the token-level embeddings
- `ip_address`: Client IP address
- `request_time`: Timestamp of the request
- `response_time`: Timestamp of the response
- `total_time`: Total time taken to process the request
- Unique Constraint on `input_text_hash` and `model_name`

### TokenLevelEmbeddingBundleCombinedFeatureVector Table
This table stores combined feature vectors for token-level embedding bundles.

- `id`: Primary Key
- `token_level_embedding_bundle_id`: Foreign Key referencing the TokenLevelEmbeddingBundle table
- `model_name`: Model name associated with the combined feature vector
- `combined_feature_vector_json`: JSON containing the combined feature vector
- `combined_feature_vector_hash`: Hash of the combined feature vector
- Unique Constraint on `combined_feature_vector_hash` and `model_name`

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

Certainly! Here's a more comprehensive `README.md` section for using the Dockerized version of the Llama2 Embeddings API Service app:

---

### Dockerized Llama2 Embeddings API Service App

A bash script is included in this repo, `setup_dockerized_app_on_fresh_machine.sh`, that will automatically do everything for you, including installing docker with apt install. 

To use it, first make the script executable and then run it like this:

```bash
chmod +x setup_dockerized_app_on_fresh_machine.sh
sudo ./setup_dockerized_app_on_fresh_machine.sh
```

If you prefer a manual setup, then read the following instructions:

#### Prerequisites

Ensure that you have Docker installed on your system. If not, follow these steps to install Docker on Ubuntu:

```bash
sudo apt-get update
sudo apt-get install docker.io
sudo systemctl start docker
sudo docker --version
sudo usermod -aG docker $USER
```

You may need to log out and log back in or restart your system to apply the new group permissions, or use sudo in the following steps to build and run the container.

#### Setup and Running the Application

1. **Clone the Repository:**

   Clone the Llama2 Embeddings API Service repository to your local machine:

   ```bash
   git clone https://github.com/Dicklesworthstone/llama_embeddings_fastapi_service
   cd llama_embeddings_fastapi_service
   ```

2. **Build the Docker Image:**

   Build the Docker image using the provided Dockerfile:

   ```bash
   sudo docker build -t llama-embeddings .
   ```

3. **Run the Docker Container:**

   Run the Docker container, mapping the container's port 8089 to the host's port 8089:

   ```bash
   sudo docker run -p 8089:8089 llama-embeddings
   ```

4. **Accessing the Application:**

   The FastAPI application will now be accessible at `http://localhost:8089` or at the static IP address of your VPS instance if you're running on one (You can get a 10-core, 30gb RAM, 1tb SSD with a static IP running Ubuntu 22.04 at Contabo for around $30/month, which is the cheapest I've found so far).

   You can interact then with the API using tools like `curl` or by accessing the FastAPI documentation at `http://localhost:8089/docs`.

6. **Viewing Logs:**

   Logs from the application can be viewed directly in the terminal where you ran the `docker run` command.

#### Stopping and Managing the Container

- To stop the running container, press `Ctrl+C` in the terminal or find the container ID using `docker ps` and run `sudo docker stop <container_id>`.
- To remove the built image, use `sudo docker rmi llama-embeddings`.

---

Based on the provided code, I'll help you update the `Startup Procedures` section of your `readme.md` file. Here's the updated version:

---

## Startup Procedures

During startup, the application performs the following tasks:

1. **Database Initialization**: 
    - The application initializes the SQLite database, setting up tables and executing important PRAGMAs to optimize performance. 
    - Some of the important SQLite PRAGMAs include setting the database to use Write-Ahead Logging (WAL) mode, setting synchronous mode to NORMAL, increasing cache size to 1GB, setting the busy timeout to 2 seconds, and setting the WAL autocheckpoint to 100.
2. **Initialize Database Writer**:
    - A dedicated database writer (`DatabaseWriter`) is initialized with a dedicated asynchronous queue to handle the write operations.
    - A set of hashes is created which represents the operations that are currently being processed or have already been processed. This avoids any duplicate operations in the queue.
3. **RAM Disk Setup**:
    - If the `USE_RAMDISK` variable is enabled and the user has the required permissions, the application sets up a RAM Disk.
    - The application checks if there's already a RAM Disk set up at the specified path, if not, it calculates the optimal size for the RAM Disk and sets it up.
    - If the RAM Disk is enabled but the user lacks the required permissions, the RAM Disk feature is disabled and the application proceeds without it.
4. **Model Downloads**: 
    - The application downloads the required models.
5. **Model Loading**:
    - Each downloaded model is loaded into memory. If any model file is not found, an error log is recorded.
6. **Build FAISS Indexes**: 
    - The application creates FAISS indexes for efficient similarity search using the embeddings from the database.
    - Separate FAISS indexes are built for token-level embeddings.
    - Associated texts are stored by model name for further use.

Note: 
- If the RAM Disk feature is enabled but the user lacks the required permissions, the application will disable the RAM Disk feature and proceed without it.
- For any database operations, if the database is locked, the application will attempt to retry the operation a few times with an exponential backoff and a jitter.

---

## Endpoint Functionality and Workflow Overview
Here's a detailed breakdown of the main endpoints provided by the FastAPI server, explaining their functionality, input parameters, and how they interact with underlying models and systems:

### 1. `/get_embedding_vector_for_string/` (POST)

#### Purpose
Retrieve the embedding vector for a given input text string using the specified model.

#### Parameters
- `text`: The input text for which the embedding vector is to be retrieved.
- `model_name`: The model used to calculate the embedding (optional, will use the default model if not provided).
- `token`: Security token (optional).
- `client_ip`: Client IP address (optional).

#### Workflow
1. **Retrieve Embedding**: The function retrieves or computes the embedding vector for the provided text using the specified or default model.
2. **Return Result**: The embedding vector for the input text string is returned in the response.

### 2. `/compute_similarity_between_strings/` (POST)

#### Purpose
Compute the similarity between two given input strings using specified model embeddings and a selected similarity measure.

#### Parameters
- `text1`: The first input text.
- `text2`: The second input text.
- `model_name`: The model used to calculate embeddings (optional).
- `similarity_measure`: The similarity measure to be used. It can be `cosine_similarity`, `hoeffdings_d`, or `hsic` (optional, default is `cosine_similarity`).
- `token`: Security token (optional).

#### Workflow
1. **Retrieve Embeddings**: The embeddings for `text1` and `text2` are retrieved or computed using the specified or default model.
2. **Compute Similarity**: The similarity between the two embeddings is calculated using the specified similarity measure.
3. **Return Result**: The similarity score, along with the embeddings and input texts, is returned in the response.

### 3. `/search_stored_embeddings_with_query_string_for_semantic_similarity/` (POST)

#### Purpose
Find the most similar strings in the database to the given input "query" text. This endpoint uses a pre-computed FAISS index to quickly search for the closest matching strings.

#### Parameters
- `query_text`: The input text for which to find the most similar string.
- `model_name`: The model used to calculate embeddings.
- `number_of_most_similar_strings_to_return`: (Optional) The number of most similar strings to return, defaults to 10.
- `token`: Security token (optional).

#### Workflow
1. **Search FAISS Index**: The FAISS index, built on stored embeddings, is searched to find the most similar embeddings to the `query_text`.
2. **Return Result**: The most similar strings found in the database, along with the similarity scores, are returned in the response.

### 4. `/get_all_embedding_vectors_for_document/` (POST)

#### Purpose
Extract text embeddings for a document. This endpoint supports both plain text and PDF files. OCR is not supported.

#### Parameters
- `file`: The uploaded document file (either plain text or PDF).
- `model_name`: (Optional) The model used to calculate embeddings.
- `json_format`: (Optional) The format of the JSON response.
- `send_back_json_or_zip_file`: Whether to return a JSON file or a ZIP file containing the embeddings file (optional, defaults to `zip`).
- `token`: Security token (optional).

### 5. `/get_list_of_available_model_names/` (GET)

#### Purpose
Retrieve the list of available model names for generating embeddings.

#### Parameters
- `token`: Security token (optional).

### 6. `/get_all_stored_strings/` (GET)

#### Purpose
Retrieve a list of all stored strings from the database for which embeddings have been computed.

#### Parameters
- `token`: Security token (optional).

### 7. `/get_all_stored_documents/` (GET)

#### Purpose
Retrieve a list of all stored documents from the database for which embeddings have been computed.

#### Parameters
- `token`: Security token (optional).

### 8. `/clear_ramdisk/` (POST)

#### Purpose
Clear the RAM Disk to free up memory.

#### Parameters
- `token`: Security token (optional).

### 9. `/get_token_level_embeddings_matrix_and_combined_feature_vector_for_string/` (POST)

#### Purpose
Retrieve the token-level embeddings and combined feature vector for a given input text using the specified model.

#### Parameters
- `text`: The input text for which the embeddings are to be retrieved.
- `model_name`: The model used to calculate the embeddings (optional).
- `db_writer`: Database writer instance for managing write operations (internal use).
- `req`: HTTP request object (optional).
- `token`: Security token (optional).
- `client_ip`: Client IP address (optional).
- `json_format`: Format for JSON response of token-level embeddings (optional).
- `send_back_json_or_zip_file`: Whether to return a JSON response or a ZIP file containing the JSON file (optional, defaults to `zip`).


