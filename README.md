# LLama2 Embeddings FastAPI Service

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Setup and Configuration](#setup-and-configuration)
5. [API Endpoints](#api-endpoints)
6. [RAM Disk Management](#ram-disk-management)
7. [Database Structure](#database-structure)
8. [Exception Handling](#exception-handling)
9. [Logging](#logging)
10. [Startup Procedures](#startup-procedures)
11. [Running the Application](#running-the-application)
12. [License](#license)

## Introduction

The LLama2 Embedding Server is designed to facilitate and optimize the process of obtaining text embeddings using different LLMs via llama_cpp and langchain. The driving motivation behind this project is to offer a convenient and easy to use API to quickly and easily submit text strings and get back embeddings using Llama2 and similar LLMs. To avoid wasting computation, these embeddings are cached in SQlite and retrieved if they have already been computed before. To speed up the process of loading multiple LLMs, optional RAM Disks can be used, and the process for creating and managing them is handled automatically for you. Finally, some additional useful endpoints are provided, such as computing semantic similarity between submitted text strings, and semantic search across all your cached embeddings using FAISS vector searching. 

---

## Features

1. **Text Embedding Computation**: Using pre-trained LLama2 (and other) models to compute embeddings for any given text.
2. **Embedding Caching**: Storing computed embeddings in a SQLite database to eliminate redundant computations.
3. **Similarity Measurements**: Computing the cosine similarity between two strings.
4. **Most Similar String Retrieval**: Finding the most similar string in the database compared to a given input.
5. **RAM Disk Usage**: Utilizing RAM Disk to store models for faster access.
6. **Robust Exception Handling**: Comprehensive handling of different types of exceptions.
7. **Swagger UI Integration**: Interactive API documentation for ease of use.
8. **Scalability**: Built using FastAPI to handle concurrent requests efficiently.

## Requirements

- Python 3.9+
- Libraries: `numpy`, `fastapi`, `uvicorn`, `sqlalchemy`, `faiss`, `psutil`, `sklearn`, `decouple`, `asyncio`, `subprocess`, `traceback`, `glob`, `shutil`, `urllib`, `json`, `logging`, `datetime`

## Setup and Configuration

### Environment Variables

The application can be configured using environment variables or hardcoded values. The following environment variables can be set:

- `USE_SECURITY_TOKEN`: Enables the use of a security token for API access.
- `LLAMA_EMBEDDING_SERVER_LISTEN_PORT`: The port on which the server will listen (default: 8089).
- `USE_RAMDISK`: Enables the use of RAM Disk (default: False).
- `RAMDISK_SIZE_IN_GB`: The size of the RAM Disk in GB (default: 1).

### RAM Disk Configuration

To enable password-less sudo for RAM Disk setup and teardown, edit the `sudoers` file with `sudo visudo`. Add the following lines, replacing `username` with your actual username:

```plaintext
username ALL=(ALL) NOPASSWD: /bin/mount -t tmpfs -o size=*G tmpfs /mnt/ramdisk
username ALL=(ALL) NOPASSWD: /bin/umount /mnt/ramdisk
```

## API Endpoints

The following endpoints are available:

- **GET `/get_list_of_available_model_names/`**: Retrieves a list of available model names.
- **GET `/get_all_strings_with_embeddings/`**: Retrieves all strings with computed embeddings.
- **POST `/get_embedding_vector/`**: Retrieves or computes the embedding vector for a given text.
- **POST `/compute_similarity_between_strings/`**: Computes the similarity between two strings.
- **POST `/get_most_similar_string_from_database/`**: Finds the most similar string from the database.
- **POST `/clear_ramdisk/`**: Clears the RAM Disk if it is enabled.

For detailed request and response schemas, please refer to the Swagger UI available at the root URL or the section at the end of this `README`.

## RAM Disk Management

The application provides functionalities to set up, clear, and manage RAM Disk. RAM Disk is used to store models in memory for faster access. It calculates the available RAM and sets up the RAM Disk accordingly. The functions `setup_ramdisk`, `copy_models_to_ramdisk`, and `clear_ramdisk` manage these tasks.

## Database Structure

The application uses a SQLite database to store computed embeddings. The table structure is as follows:

- `id`: Primary Key
- `text`: Text for which the embedding was computed
- `model_name`: Model used to compute the embedding
- `embedding_json`: The computed embedding in JSON format
- `ip_address`: Client IP address
- `request_time`: Timestamp of the request
- `response_time`: Timestamp of the response
- `total_time`: Total time taken to process the request
- Unique Constraint on `text` and `model_name`

## Exception Handling

The application has robust exception handling to deal with various types of errors, including database errors and general exceptions. Custom exception handlers are defined for `SQLAlchemyError` and general `Exception`.

## Logging

Logging is configured at the INFO level to provide detailed logs for debugging and monitoring. The logger provides information about the state of the application, errors, and activities.

## Startup Procedures

During startup, the application performs the following tasks:

1. Sets up RAM Disk if enabled.
2. Downloads models.
3. Initializes the database.
4. Builds the FAISS index.

## Running the Application

You can run the application using the following command:

```bash
python llama_2_embeddings_fastapi_server.py
```

The server will start on `0.0.0.0` at the port defined by the `LLAMA_EMBEDDING_SERVER_LISTEN_PORT` variable.

## Endpoint Functionality and Workflow Overview
Here's a detailed breakdown of the main endpoints provided by the LLama Embedding Server, explaining their functionality, input parameters, and how they interact with underlying models and systems:

### 1. `/get_embedding_vector/` (POST)

#### Purpose
This endpoint computes or retrieves the embedding vector for a given text and model name. If the embedding is already cached in the database, it returns the cached result. Otherwise, it computes the embedding using the specified model and caches the result.

#### Parameters
- `text`: The text for which the embedding is to be calculated.
- `model_name`: The name of the model to be used. Defaults to a pre-configured model.

#### Workflow
1. **Check Database**: The function first checks if the embedding for the given text and model name is already present in the database.
2. **Compute Embedding**: If not found in the database, the specified model is loaded (or retrieved from the cache if previously loaded), and the embedding is calculated.
3. **Cache Embedding**: The computed embedding is then serialized and stored in the database along with metadata like IP address, request time, and total time taken.
4. **Return Result**: Finally, the embedding is returned in the response.

### 2. `/compute_similarity_between_strings/` (POST)

#### Purpose
This endpoint calculates the cosine similarity between two given strings. It utilizes the embeddings of the strings, either by retrieving them from the cache or computing them on-the-fly.

#### Parameters
- `text1`: The first text string.
- `text2`: The second text string.
- `model_name`: (Optional) The name of the model to be used.

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

### 4. `/get_list_of_available_model_names/` (GET)

#### Purpose
This endpoint provides the list of available model names that can be used to compute embeddings.

#### Workflow
1. **Scan Models Directory**: The server scans the models directory (or RAM Disk if enabled) for files with the appropriate extension.
2. **Extract Model Names**: Model names are extracted from the filenames.
3. **Return Result**: The list of model names is returned in the response.

### 5. `/clear_ramdisk/` (POST)

#### Purpose
If RAM Disk usage is enabled, this endpoint clears the RAM Disk, freeing up memory.

#### Workflow
1. **Check RAM Disk Usage**: If RAM Disk usage is disabled, a message is returned indicating so.
2. **Clear RAM Disk**: If enabled, the RAM Disk is cleared using the `clear_ramdisk` function.
3. **Return Result**: A success message is returned in the response.

### Additional Endpoints

- **`/get_all_strings_with_embeddings/` (GET)**: Retrieves all unique strings for which embeddings have been computed and cached.
- **Custom Swagger UI**: The root endpoint (`/`) serves a customized Swagger UI for interactive API documentation and testing.

These endpoints collectively offer a versatile set of tools and utilities to work with text embeddings, efficiently utilizing cached results and providing useful functionalities like similarity computation and similarity search. By encapsulating complex operations behind a simple and well-documented API, they make working with LLMs via llama_cpp and langchain accessible and efficient.
