# 🇨🇭🎖️🦙 Swiss Army Llama

<div align="center">
  <img src="https://github.com/Dicklesworthstone/swiss_army_llama/raw/main/swiss_army_llama_logo.webp" width="500">
</div>

## Introduction

The Swiss Army Llama is designed to facilitate and optimize the process of working with local LLMs by using FastAPI to expose convenient REST endpoints for various tasks, including obtaining text embeddings and completions using different LLMs via llama_cpp, as well as automating the process of obtaining all the embeddings for most common document types, including PDFs (even ones that require OCR), Word file, etc; it even allows you to submit an audio file and automatically transcribes it with the Whisper model, cleans up the resulting text, and then computes the embeddings for it. To avoid wasting computation, these embeddings are cached in SQlite and retrieved if they have already been computed before. To speed up the process of loading multiple LLMs, optional RAM Disks can be used, and the process for creating and managing them is handled automatically for you. With a quick and easy setup process, you will immediately get access to a veritable "Swiss Army Knife" of LLM related tools, all accessible via a convenient Swagger UI and ready to be integrated into your own applications with minimal fuss or configuration required.

Some additional useful endpoints are provided, such as computing semantic similarity between submitted text strings. The service leverages a high-performance Rust-based library, `fast_vector_similarity`, to offer a range of similarity measures including `spearman_rho`, `kendall_tau`, `approximate_distance_correlation`, `jensen_shannon_similarity`, and [`hoeffding_d`](https://blogs.sas.com/content/iml/2021/05/03/examples-hoeffding-d.html). Additionally, semantic search across all your cached embeddings is supported using FAISS vector searching. You can either use the built in cosine similarity from FAISS, or supplement this with a second pass that computes the more sophisticated similarity measures for the most relevant subset of the stored vectors found using cosine similarity (see the advanced semantic search endpoint for this functionality).

As mentioned above, you can now submit not only plaintext and fully digital PDFs but also MS Word documents, images, and other file types supported by the textract library. The library can automatically apply OCR using Tesseract for scanned text. The returned embeddings for each sentence in a document can be organized in various formats like records, table, etc., using the Pandas to_json() function. The results can be returned either as a ZIP file containing a JSON file or as a direct JSON response. You can now also submit audio files in MP3 or WAV formats. The library uses OpenAI's Whisper model, as optimized by the Faster Whisper Python library, to transcribe the audio into text. Optionally, this transcript can be treated like any other document, with each sentence's embeddings computed and stored. The results are returned as a URL to a downloadable ZIP file containing a JSON with the embedding vector data.

In addition to fixed-sized embedding vectors, we also expose functionality that allows you to get back token-level embeddings, where each token in the input stream is embedded with its context in the string as a full sized vector, thus producing a matrix that has a number of rows equal to the number of tokens in the input string. This includes far more nuanced information about the contents of the string at the expense of much greater compute and storage requirements. The other drawback is that, instead of having the same sized output for every string, regardless of length (which makes it very easy to compare unequal length strings using cosine similarity and other measures), the token-level embedding matrix obviously differs in dimensions for two different strings if the strings have different numbers of tokens. To deal with this, we introduce combined feature vectors, which compute the column-wise mean, min, max, and std. deviation of the token-level emeddding matrix, and concatenate these together in to a single huge matrix; this allows you to compare strings of different lengths while still capturing more nuance. The combined results, including the embedding matrix and associated combined feature vector, can similarly be returned as either a zip file or direct JSON response.

Finally, we add a new endpoint for generating multiple text completions for a given input prompt, with the ability to specify a grammar file that will enforce a particular form of response, such as JSON. There is also a useful new utility feature: a real-time application log viewer that can be accessed via a web browser, which allows for syntax highlighting and offers options for downloading the logs or copying them to the clipboard. This allows a user to watch the logs without having direct SSH access to the server.

## Screenshots
![Swiss Army Llama Swagger UI](https://github.com/Dicklesworthstone/swiss_army_llama/raw/main/swiss_army_llama__swagger_screenshot.png)
![Swiss Army Llama Runnig](https://github.com/Dicklesworthstone/swiss_army_llama/raw/main/swiss_army_llama__swagger_screenshot_running.png)

*TLDR:* If you just want to try it very quickly on a fresh Ubuntu 22+ machine (warning, this will install docker using apt):

```bash
git clone https://github.com/Dicklesworthstone/swiss_army_llama
cd swiss_army_llama
chmod +x setup_dockerized_app_on_fresh_machine.sh
sudo ./setup_dockerized_app_on_fresh_machine.sh
```

To run it natively (not using Docker) in a Python venv (recommended!), you can use these commands:

```bash
sudo apt-get update
sudo apt-get install libxml2-dev libxslt1-dev antiword unrtf poppler-utils pstotext tesseract-ocr flac ffmpeg lame libmad0 libsox-fmt-mp3 sox libjpeg-dev swig -y
git clone https://github.com/Dicklesworthstone/swiss_army_llama
cd swiss_army_llama
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install wheel
pip install -r requirements.txt
python3 swiss_army_llama.py
```

Then open a browser to `<your_static_ip_address>:8089` if you're using a VPS to get to the FastAPI Swagger page at `http://localhost:8089`.

Or to `localhost:8089` if you're using your own machine-- but, really, you should never run untrusted code with sudo on your own machine! Just get a cheap VPS to experiment with for $30/month.

Watch the the automated setup process in action [here](https://asciinema.org/a/601603).

---

## Features

1. **Text Embedding Computation**: Utilizes pre-trained LLama2 and other LLMs via llama_cpp and langchain to generate embeddings for any provided text, including token-level embeddings that capture more nuanced information about the content.
2. **Embedding Caching**: Efficiently stores and retrieves computed embeddings in SQLite, minimizing redundant computations. It supports caching both fixed-sized embedding vectors and token-level embeddings.
3. **Advanced Similarity Measurements and Retrieval**: Utilizes the author's own `fast_vector_similarity` library written in Rust to offer highly optimized advanced similarity measures such as `spearman_rho`, `kendall_tau`, `approximate_distance_correlation`, `jensen_shannon_similarity`, and `hoeffding_d`. Semantic search across cached embeddings is also supported using FAISS vector searching.
4. **Two-Step Advanced Semantic Search**: The API first leverages FAISS and cosine similarity for rapid filtering, and then applies additional similarity measures like `spearman_rho`, `kendall_tau`, `approximate_distance_correlation`, `jensen_shannon_similarity`, and `hoeffding_d` for a more nuanced comparison.
5. **File Processing for Documents**: The library now accepts a broader range of file types including plaintext, PDFs, MS Word documents, and images. It can also handle OCR automatically. Returned embeddings for each sentence are organized in various formats like records, table, etc., using Pandas to_json() function.
6. **Advanced Text Preprocessing**: The library now employs a more advanced sentence splitter to segment text into meaningful sentences. It handles cases where periods are used in abbreviations, domain names, or numbers and also ensures complete sentences even when quotes are used. It also takes care of pagination issues commonly found in scanned documents, such as awkward newlines and hyphenated line breaks.
7. **Audio Transcription and Embedding**: Upload an audio file in MP3 or WAV format. The library uses OpenAI's Whisper model for transcription. Optionally, sentence embeddings can be computed for the transcript.
8. **Token-Level Embeddings and Combined Feature Vectors**: Provides token-level embeddings to capture the context of each token in the input string. Introduces combined feature vectors by computing the column-wise mean, min, max, and std. deviation of the token-level embedding matrix, allowing comparison of unequal length strings.
9. **RAM Disk Usage**: Optionally uses RAM Disk to store models for faster access and execution. Automatically handles the creation and management of RAM Disks.
10. **Robust Exception Handling**: Features comprehensive exception management to ensure system resilience.
11. **Interactive API Documentation**: Integrates with Swagger UI for an interactive and user-friendly experience, accommodating large result sets without crashing.
12. **Scalability and Concurrency**: Built on the FastAPI framework, handles concurrent requests and supports parallel inference with configurable concurrency levels.
13. **Flexible Configurations**: Offers configurable settings through environment variables and input parameters, including response formats like JSON or ZIP files.
14. **Comprehensive Logging**: Captures essential information with detailed logs, without overwhelming storage or readability.
15. **Support for Multiple Models and Measures**: Accommodates multiple embedding models and similarity measures, allowing flexibility and customization based on user needs.
16. **Ability to Generate Multiple Completions using Specified Grammar**: Get back structured LLM completions for a specified input prompt.
17. **Real-Time Log File Viewer in Browser**: Lets anyone with access to the API server conveniently watch the application logs to gain insight into the execution of their requests.

## Demo Screen Recording in Action
[Here](https://asciinema.org/a/39dZ8vv9nkcNygasUl35wnBPq) is the live console output while I interact with it from the Swagger page to make requests.

---

## Requirements

System requirements for running the application (to support all the file types handled by textract):

```bash
sudo apt-get update
sudo apt-get install libxml2-dev libxslt1-dev antiword unrtf poppler-utils pstotext tesseract-ocr flac ffmpeg lame libmad0 libsox-fmt-mp3 sox libjpeg-dev swig -y
```

Python Requirements:

```bash
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
llama-cpp-python
httpx
filelock
fast_vector_similarity
faster-whisper
textract
pytz
uvloop
```

## Running the Application

You can run the application using the following command:

```bash
python swiss_army_llama.py
```

The server will start on `0.0.0.0` at the port defined by the `SWISS_ARMY_LLAMA_SERVER_LISTEN_PORT` variable.

Access the Swagger UI:

```
http://localhost:<SWISS_ARMY_LLAMA_SERVER_LISTEN_PORT>
```

## Configuration

You can configure the service easily by editing the included `.env` file. Here's a list of available configuration options:

- `USE_SECURITY_TOKEN`: Whether to use a hardcoded security token. (e.g., `True`)
- `USE_PARALLEL_INFERENCE_QUEUE`: Use parallel processing. (e.g., `True`)
- `MAX_CONCURRENT_PARALLEL_INFERENCE_TASKS`: Maximum number of parallel inference tasks. (e.g., `30`)
- `DEFAULT_MODEL_NAME`: Default model name to use. (e.g., `yarn-llama-2-13b-128k`)
- `LLM_CONTEXT_SIZE_IN_TOKENS`: Context size in tokens for LLM. (e.g., `512`)
- `SWISS_ARMY_LLAMA_SERVER_LISTEN_PORT`: Port number for the service. (e.g., `8089`)
- `UVICORN_NUMBER_OF_WORKERS`: Number of workers for Uvicorn. (e.g., `2`)
- `MINIMUM_STRING_LENGTH_FOR_DOCUMENT_EMBEDDING`: Minimum string length for document embedding. (e.g., `15`)
- `MAX_RETRIES`: Maximum retries for locked database. (e.g., `10`)
- `DB_WRITE_BATCH_SIZE`: Database write batch size. (e.g., `25`)
- `RETRY_DELAY_BASE_SECONDS`: Retry delay base in seconds. (e.g., `1`)
- `JITTER_FACTOR`: Jitter factor for retries. (e.g., `0.1`)
- `USE_RAMDISK`: Use RAM disk. (e.g., `True`)
- `RAMDISK_PATH`: Path to the RAM disk. (e.g., `"/mnt/ramdisk"`)
- `RAMDISK_SIZE_IN_GB`: RAM disk size in GB. (e.g., `40`)

## Contributing

If you'd like to contribute to the project, please submit a pull request! Seriously, I'd love to get some more community going so we can make this a standard library!

## License

This project is licensed under the MIT License.

## Some Llama Knife Images I found on Google
<p align="center">
  <img src="https://raw.githubusercontent.com/Dicklesworthstone/swiss_army_llama/main/llama_knife_sticker.webp" width="500" style="margin-right: 10px;">
  <img src="https://raw.githubusercontent.com/Dicklesworthstone/swiss_army_llama/main/llama_knife_sticker2.jpg" width="500">
</p>

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

- **GET `/get_list_of_available_model_names/`**: Retrieve Available Model Names. Retrieves the list of available model names for generating embeddings.
- **GET `/get_all_stored_strings/`**: Retrieve All Strings. Retrieves a list of all stored strings from the database for which embeddings have been computed.
- **GET `/get_all_stored_documents/`**: Retrieve All Stored Documents. Retrieves a list of all stored documents from the database for which embeddings have been computed.
- **GET `/show_logs/`**:  Shows logs for the last 5 minutes by default. Can also provide a parameter like this: `/show_logs/{minutes}` to get the last N minutes of log data.
- **POST `/add_new_model/`**: Add New Model by URL. Submit a new model URL for download and use. The model must be in `.gguf` format and larger than 100 MB to ensure it's a valid model file (you can directly paste in the Huggingface URL)
- **POST `/get_embedding_vector_for_string/`**: Retrieve Embedding Vector for a Given Text String. Retrieves the embedding vector for a given input text string using the specified model.
- **POST `/get_token_level_embeddings_matrix_and_combined_feature_vector_for_string/`**: Retrieve Token-Level Embeddings and Combined Feature Vector for a Given Input String. Retrieve the token-level embeddings and combined feature vector for a given input text using the specified model.
- **POST `/compute_similarity_between_strings/`**: Compute Similarity Between Two Strings. Leverages the `fast_vector_similarity` library to compute the similarity between two given input strings using specified model embeddings and a selected similarity measure.
- **POST `/search_stored_embeddings_with_query_string_for_semantic_similarity/`**: Get Most Similar Strings from Stored Embeddings in Database. Find the most similar strings in the database to the given input "query" text.
- **POST `/advanced_search_stored_embeddings_with_query_string_for_semantic_similarity/`**: Perform a two-step advanced semantic search. First uses FAISS and cosine similarity to narrow down the most similar strings, then applies additional similarity measures for refined comparison.
- **POST `/get_all_embedding_vectors_for_document/`**: Get Embeddings for a Document. Extract text embeddings for a document. This endpoint supports plain text, .doc/.docx (MS Word), PDF files, images (using Tesseract OCR), and many other file types supported by the textract library.
- **POST `/compute_transcript_with_whisper_from_audio/`**: Transcribe and Embed Audio using Whisper and LLM. This endpoint accepts an audio file and optionally computes document embeddings. The transcription and embeddings are stored, and a ZIP file containing the embeddings can be downloaded.
- **POST `/get_text_completions_from_input_prompt/`**: Get back multiple completions from the specified LLM model, with the ability to specify a grammar file which will enforce a particular format of the response, such as JSON. 
- **POST `/clear_ramdisk/`**: Clear Ramdisk Endpoint. Clears the RAM Disk if it is enabled.

For detailed request and response schemas, please refer to the Swagger UI available at the root URL or the section at the end of this `README`.

## Exception Handling

The application has robust exception handling to deal with various types of errors, including database errors and general exceptions. Custom exception handlers are defined for `SQLAlchemyError` and general `Exception`.

## Logging

Logging is configured at the INFO level to provide detailed logs for debugging and monitoring. The logger provides information about the state of the application, errors, and activities.

The logs are stored in a file named `swiss_army_llama.log`, and a log rotation mechanism is implemented to handle log file backups. The rotating file handler is configured with a maximum file size of 10 MB, and it keeps up to 5 backup files.

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

- `id`: Primary Key
- `text`: Text for which the embedding was computed
- `text_hash`: Hash of the text, computed using SHA3-256
- `llm_model_name`: Model used to compute the embedding
- `embedding_json`: The computed embedding in JSON format
- `ip_address`: Client IP address
- `request_time`: Timestamp of the request
- `response_time`: Timestamp of the response
- `total_time`: Total time taken to process the request
- `document_file_hash`: Foreign Key referencing the DocumentEmbedding table

### DocumentEmbedding Table

- `id`: Primary Key
- `document_hash`: Foreign Key referencing the Documents table
- `filename`: Name of the document file
- `mimetype`: MIME type of the document file
- `file_hash`: Hash of the file
- `llm_model_name`: Model used to compute the embedding
- `file_data`: Binary data of the original file
- `document_embedding_results_json`: The computed embedding results in JSON format

### Document Table

- `id`: Primary Key
- `llm_model_name`: Model name associated with the document
- `document_hash`: Computed Hash of the document

### TokenLevelEmbedding Table

- `id`: Primary Key
- `token`: Token for which the embedding was computed
- `token_hash`: Hash of the token, computed using SHA3-256
- `llm_model_name`: Model used to compute the embedding
- `token_level_embedding_json`: The computed token-level embedding in JSON format

### TokenLevelEmbeddingBundle Table

- `id`: Primary Key
- `input_text`: Input text associated with the token-level embeddings
- `input_text_hash`: Hash of the input text
- `llm_model_name`: Model used to compute the embeddings
- `token_level_embeddings_bundle_json`: JSON containing the token-level embeddings

### TokenLevelEmbeddingBundleCombinedFeatureVector Table

- `id`: Primary Key
- `token_level_embedding_bundle_id`: Foreign Key referencing the TokenLevelEmbeddingBundle table
- `llm_model_name`: Model name associated with the combined feature vector
- `combined_feature_vector_json`: JSON containing the combined feature vector
- `combined_feature_vector_hash`: Hash of the combined feature vector

### AudioTranscript Table

- `audio_file_hash`: Primary Key
- `audio_file_name`: Name of the audio file
- `audio_file_size_mb`: File size in MB
- `segments_json`: Transcribed segments as JSON
- `combined_transcript_text`: Combined transcript text
- `info_json`: Transcription info as JSON

### Database Relationships

1. **TextEmbedding - DocumentEmbedding**:
   - `TextEmbedding` has a Foreign Key `document_file_hash` that references `DocumentEmbedding`'s `file_hash`.
   - This means multiple text embeddings can belong to a single document embedding, establishing a one-to-many relationship.
  
2. **DocumentEmbedding - Document**:
   - `DocumentEmbedding` has a Foreign Key `document_hash` that references `Document`'s `document_hash`.
   - This establishes a one-to-many relationship between `Document` and `DocumentEmbedding`.

3. **TokenLevelEmbedding - TokenLevelEmbeddingBundle**:  
   - `TokenLevelEmbedding` has a Foreign Key `token_level_embedding_bundle_id` that references `TokenLevelEmbeddingBundle`'s `id`.
   - This is a one-to-many relationship, meaning multiple token-level embeddings can belong to a single token-level embedding bundle.

4. **TokenLevelEmbeddingBundle - TokenLevelEmbeddingBundleCombinedFeatureVector**:
   - `TokenLevelEmbeddingBundle` has a one-to-one relationship with `TokenLevelEmbeddingBundleCombinedFeatureVector` via `token_level_embedding_bundle_id`.
   - This means each token-level embedding bundle can have exactly one combined feature vector.

5. **AudioTranscript**:  
   - This table doesn't seem to have a direct relationship with other tables based on the given code.

6. **Request/Response Models**:  
   - These are not directly related to the database tables but are used for handling API requests and responses.


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

   Clone the Swiss Army Llama repository to your local machine:

   ```bash
   git clone https://github.com/Dicklesworthstone/swiss_army_llama
   cd swiss_army_llama
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

5. **Viewing Logs:**

   Logs from the application can be viewed directly in the terminal where you ran the `docker run` command.

#### Stopping and Managing the Container

- To stop the running container, press `Ctrl+C` in the terminal or find the container ID using `docker ps` and run `sudo docker stop <container_id>`.
- To remove the built image, use `sudo docker rmi llama-embeddings`.

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
- `llm_model_name`: The model used to calculate embeddings (optional).
- `similarity_measure`: The similarity measure to be used. Supported measures include `all`, `spearman_rho`, `kendall_tau`, `approximate_distance_correlation`, `jensen_shannon_similarity`, and `hoeffding_d` (optional, default is `all`).

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

### 4. `/advanced_search_stored_embeddings_with_query_string_for_semantic_similarity/` (POST)

#### Purpose
Performs a two-step advanced semantic search. Utilizes FAISS and cosine similarity for initial filtering, followed by additional similarity measures for refined comparisons.

#### Parameters
- `query_text`: The input text for which to find the most similar strings.
- `llm_model_name`: The model used to calculate embeddings.
- `similarity_filter_percentage`: (Optional) Percentage of embeddings to filter based on cosine similarity; defaults to 0.02 (i.e., top 2%).
- `number_of_most_similar_strings_to_return`: (Optional) Number of most similar strings to return after second similarity measure; defaults to 10.

#### Workflow
1. **Initial Filtering**: Use FAISS and cosine similarity to find a set of similar strings.
2. **Refined Comparison**: Apply additional similarity measures to the filtered set.
3. **Return Result**: Return the most similar strings along with their multiple similarity scores.

#### Example Request
```json
{
  "query_text": "Find me the most similar string!",
  "llm_model_name": "openchat_v3.2_super",
  "similarity_filter_percentage": 0.02,
  "number_of_most_similar_strings_to_return": 5
}
```
### 5. `/get_all_embedding_vectors_for_document/` (POST)

#### Purpose
Extract text embeddings for a document. The library now supports a wide range of file types including plain text, .doc/.docx, PDF files, images (using Tesseract OCR), and many other types supported by the `textract` library.

#### Parameters
- `file`: The uploaded document file (either plain text, .doc/.docx, PDF, etc.).
- `llm_model_name`: (Optional) The model used to calculate embeddings.
- `json_format`: (Optional) The format of the JSON response.
- `send_back_json_or_zip_file`: Whether to return a JSON file or a ZIP file containing the embeddings file (optional, defaults to `zip`).
- `token`: Security token (optional).


### 6. `/compute_transcript_with_whisper_from_audio/` (POST)

#### Purpose
Transcribe an audio file and optionally compute document embeddings for the resulting transcript. This endpoint uses the Whisper model for transcription and a language model for generating embeddings. The transcription and embeddings can then be stored, and a ZIP file containing the embeddings can be made available for download.

#### Parameters
- `file`: The audio file that you need to upload for transcription.
- `compute_embeddings_for_resulting_transcript_document`: Boolean to indicate whether document embeddings should be computed (optional, defaults to False).
- `llm_model_name`: The language model used for computing embeddings (optional, defaults to the default model name).
- `req`: HTTP request object for additional request metadata (optional).
- `token`: Security token (optional).
- `client_ip`: Client IP address (optional).

#### Request File and Parameters
You will need to use a multipart/form-data request to upload the audio file. The additional parameters like `compute_embeddings_for_resulting_transcript_document` and `llm_model_name` can be sent along as form fields.

#### Example Request
```bash
curl -X 'POST' \
  'http://localhost:8000/compute_transcript_with_whisper_from_audio/' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer YOUR_ACCESS_TOKEN' \
  -F 'file=@your_audio_file.wav' \
  -F 'compute_embeddings_for_resulting_transcript_document=true' \
  -F 'llm_model_name=custom-llm-model'
```

### 7. `/get_token_level_embeddings_matrix_and_combined_feature_vector_for_string/` (POST)

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



#### Response
The response will be a JSON object containing complete transcription details, computational times, and an optional URL for downloading a ZIP file containing the document embeddings.

#### Example Response
```json
{
  "transcript": "This is the transcribed text...",
  "time_taken_for_transcription_in_seconds": 12.345,
  "time_taken_for_embedding_computation_in_seconds": 3.456,
  "embedding_download_url": "http://localhost:8000/download/your_embedding.zip",
  "llm_model_name": "custom-llm-model"
}
```

### 8. `/get_text_completions_from_input_prompt/` (POST)

#### Purpose
Generate text completions for a given input prompt using the specified model.

#### Parameters
- `request`: A JSON object containing various options like `input_prompt`, `llm_model_name`, etc.
- `token`: Security token (optional).
- `req`: HTTP request object (optional).
- `client_ip`: Client IP address (optional).

#### Request JSON Format
The JSON object should have the following keys:
- `input_prompt`
- `llm_model_name`
- `temperature`
- `grammar_file_string`
- `number_of_completions_to_generate`
- `number_of_tokens_to_generate`

#### Example Request
```json
{
  "input_prompt": "The Kings of France in the 17th Century:",
  "llm_model_name": "phind-codellama-34b-python-v1",
  "temperature": 0.95,
  "grammar_file_string": "json",
  "number_of_tokens_to_generate": 500,
  "number_of_completions_to_generate": 3
}
```

### 9. `/get_list_of_available_model_names/` (GET)

#### Purpose
Retrieve the list of available model names for generating embeddings.

#### Parameters
- `token`: Security token (optional).

### 10. `/get_all_stored_strings/` (GET)

#### Purpose
Retrieve a list of all stored strings from the database for which embeddings have been computed.

#### Parameters
- `token`: Security token (optional).

### 11. `/get_all_stored_documents/` (GET)

#### Purpose
Retrieve a list of all stored documents from the database for which embeddings have been computed.

#### Parameters
- `token`: Security token (optional).

### 12. `/clear_ramdisk/` (POST)

#### Purpose
Clear the RAM Disk to free up memory.

#### Parameters
- `token`: Security token (optional).


### 13. `/download/{file_name}` (GET)

#### Purpose
Download a ZIP file containing document embeddings that were generated through the `/compute_transcript_with_whisper_from_audio/` endpoint. The URL for this download will be supplied in the JSON response of the audio file transcription endpoint.

#### Parameters
- `file_name`: The name of the ZIP file that you want to download.

### 14. `/add_new_model/` (POST)

#### Purpose
Submit a new model URL for download and use. The model must be in `.gguf` format and larger than 100 MB to ensure it's a valid model file.

#### Parameters
- `model_url`: The URL of the model weight file, which must end with `.gguf`.
- `token`: Security token (optional).
