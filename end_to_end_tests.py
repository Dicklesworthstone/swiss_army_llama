import asyncio
import httpx
import json
import os
import time
from decouple import config
from typing import List, Dict, Any

LLAMA_EMBEDDING_SERVER_LISTEN_PORT = config("LLAMA_EMBEDDING_SERVER_LISTEN_PORT", default=8089, cast=int)

BASE_URL = f"http://localhost:{LLAMA_EMBEDDING_SERVER_LISTEN_PORT}"
DOCUMENT_PATH = "~/Downloads/tale_two_cities_first_3_chapters.txt"
CORPUS_IDENTIFIER_STRING = "end_to_end_test"
SEARCH_STRING = "equine"

async def get_model_names() -> List[str]:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/get_list_of_available_model_names/")
        model_names = response.json()["model_names"]
        return [name for name in model_names if "llava" not in name]

async def get_embedding_pooling_methods() -> List[str]:
    return ['means', 'means_mins_maxes', 'means_mins_maxes_stds_kurtoses', 'svd', 'svd_first_four', 'gram_matrix',
            'qr_decomposition', 'cholesky_decomposition', 'ica', 'nmf', 'factor_analysis', 'gaussian_random_projection']

async def compute_document_embeddings(model_name: str, embedding_pooling_method: str) -> float:
    with open(os.path.expanduser(DOCUMENT_PATH), "rb") as file:
        start_time = time.time()
        async with httpx.AsyncClient() as client:
            _ = await client.post(
                f"{BASE_URL}/get_all_embedding_vectors_for_document/",
                files={"file": file},
                data={
                    "llm_model_name": model_name,
                    "embedding_pooling_method": embedding_pooling_method,
                    "corpus_identifier_string": CORPUS_IDENTIFIER_STRING,
                }
            )
        end_time = time.time()
        return end_time - start_time

async def perform_semantic_search(model_name: str, embedding_pooling_method: str) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/search_stored_embeddings_with_query_string_for_semantic_similarity/",
            json={
                "query_text": SEARCH_STRING,
                "llm_model_name": model_name,
                "embedding_pooling_method": embedding_pooling_method,
                "corpus_identifier_string": CORPUS_IDENTIFIER_STRING,
            }
        )
        return response.json()

async def main():
    start_time = time.time()
    model_names = await get_model_names()
    embedding_pooling_methods = await get_embedding_pooling_methods()

    results = {}
    for model_name in model_names:
        for embedding_pooling_method in embedding_pooling_methods:
            print(f"Computing embeddings for model {model_name} and pooling method {embedding_pooling_method}...")
            total_time = await compute_document_embeddings(model_name, embedding_pooling_method)
            print(f"Embeddings computed in {total_time:.2f} seconds.")
            results[(model_name, embedding_pooling_method)] = total_time

    for model_name, embedding_pooling_method in results:
        print(f"Performing semantic search for model {model_name} and pooling method {embedding_pooling_method}...")
        search_results = await perform_semantic_search(model_name, embedding_pooling_method)
        saved_outputs_dir = "saved_outputs"
        if not os.path.exists(saved_outputs_dir):
            os.makedirs(saved_outputs_dir)
        filename = f"{model_name}_{embedding_pooling_method}_search_results.json"
        file_path = os.path.join(saved_outputs_dir, filename)
        with open(file_path, "w") as f:
            json.dump(search_results, f, indent=2)
        print(f"Search results saved to {file_path}.")

    end_time = time.time()
    print(f"All tests completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    asyncio.run(main())