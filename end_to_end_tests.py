import asyncio
import httpx
import json
import os
import time
from decouple import config
from typing import List, Dict, Any

LLAMA_EMBEDDING_SERVER_LISTEN_PORT = config("LLAMA_EMBEDDING_SERVER_LISTEN_PORT", default=8089, cast=int)

BASE_URL = f"http://localhost:{LLAMA_EMBEDDING_SERVER_LISTEN_PORT}"
DOCUMENT_PATH = "sample_input_files_for_end_to_end_tests/tale_two_cities_first_3_chapters.txt"
DOCUMENT_PATH_PDF = "sample_input_files_for_end_to_end_tests/bh-us-03-sassaman-conference-slides.pdf"
IMAGE_PATH = "sample_input_files_for_end_to_end_tests/sunset.jpg"
AUDIO_PATH = "sample_input_files_for_end_to_end_tests/Don_King_if_he_lived_in_the_tiny_island_nation_known_as_Japan.mp3"
TEXT_PROMPT = "Make up a poem about Bitcoin in the style of John Donne's 'The Canonization'."
CORPUS_IDENTIFIER_STRING = "end_to_end_test"
SEARCH_STRING = "equine"
SEARCH_STRING_PDF = "Threat model"
HTTPX_TIMEOUT_IN_SECONDS = 600
USE_MANUAL_MODEL_NAME_LIST = 1
MANUAL_MODEL_NAME_LIST = ["Meta-Llama-3-8B-Instruct.Q3_K_S"]

async def get_model_names() -> List[str]:
    print("Requesting list of available model names...")
    async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT_IN_SECONDS) as client:
        response = await client.get(f"{BASE_URL}/get_list_of_available_model_names/")
        model_names = response.json()["model_names"]
        print(f"Received model names: {model_names}")
        return [name for name in model_names if "llava" not in name]

async def get_embedding_pooling_methods() -> List[str]:
    pooling_methods = ['means', 'means_mins_maxes', 'means_mins_maxes_stds_kurtoses', 'svd', 'svd_first_four',
                    'qr_decomposition', 'cholesky_decomposition', 'ica', 'nmf', 'factor_analysis', 'gaussian_random_projection']
    print(f"Using embedding pooling methods: {pooling_methods}")
    return pooling_methods

async def compute_document_embeddings(model_name: str, embedding_pooling_method: str) -> float:
    print(f"Reading document from {DOCUMENT_PATH} for model {model_name} with pooling method {embedding_pooling_method}...")
    with open(os.path.expanduser(DOCUMENT_PATH), "rb") as file:
        start_time = time.time()
        async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT_IN_SECONDS) as client:
            print(f"Sending request to compute document embeddings with model {model_name} and pooling method {embedding_pooling_method}...")
            url = (
                f"{BASE_URL}/get_all_embedding_vectors_for_document/"
                f"?llm_model_name={model_name}"
                f"&embedding_pooling_method={embedding_pooling_method}"
                f"&corpus_identifier_string={CORPUS_IDENTIFIER_STRING}"
                f"&json_format=records"
                f"&send_back_json_or_zip_file=zip"
            )
            response = await client.post(
                url,
                files={"file": ("document.txt", file, "text/plain")},
                data={
                    "url": "",
                    "hash": "",
                    "size": "",
                }
            )
            print(f"Request sent with embedding_pooling_method: {embedding_pooling_method}. Status code: {response.status_code}")
            response_json = response.json()
            print(f"Server response received: {response_json}")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Document embeddings computed in {elapsed_time:.2f} seconds with pooling method {embedding_pooling_method}.")
        return elapsed_time
    
async def perform_semantic_search(model_name: str, embedding_pooling_method: str) -> Dict[str, Any]:
    print(f"Performing semantic search for model {model_name} with pooling method {embedding_pooling_method}...")
    try:
        async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT_IN_SECONDS) as client:
            response = await client.post(
                f"{BASE_URL}/search_stored_embeddings_with_query_string_for_semantic_similarity/",
                json={
                    "query_text": SEARCH_STRING,
                    "number_of_most_similar_strings_to_return": 10,
                    "llm_model_name": model_name,
                    "embedding_pooling_method": embedding_pooling_method,
                    "corpus_identifier_string": CORPUS_IDENTIFIER_STRING,
                }
            )
            response.raise_for_status()  # Raise an exception for HTTP errors
            search_results = response.json()
            print(f"Semantic search completed. Results: {search_results}")
            return search_results
    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
        return {"error": f"HTTP error occurred: {e.response.status_code}"}
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return {"error": str(e)}
    
async def perform_advanced_semantic_search(model_name: str, embedding_pooling_method: str) -> Dict[str, Any]:
    print(f"Performing advanced semantic search for model {model_name} with pooling method {embedding_pooling_method}...")
    try:
        async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT_IN_SECONDS) as client:
            response = await client.post(
                f"{BASE_URL}/advanced_search_stored_embeddings_with_query_string_for_semantic_similarity/",
                json={
                    "query_text": SEARCH_STRING,
                    "llm_model_name": model_name,
                    "embedding_pooling_method": embedding_pooling_method,
                    "corpus_identifier_string": CORPUS_IDENTIFIER_STRING,
                    "similarity_filter_percentage": 0.01,
                    "number_of_most_similar_strings_to_return": 10,
                    "result_sorting_metric": "hoeffding_d"
                }
            )
            response.raise_for_status()  # Raise an exception for HTTP errors
            advanced_search_results = response.json()
            print(f"Advanced semantic search completed. Results: {advanced_search_results}")
            return advanced_search_results
    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
        return {"error": f"HTTP error occurred: {e.response.status_code}"}
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return {"error": str(e)}

async def generate_text_completion(input_prompt: str, model_name: str) -> Dict[str, Any]:
    print(f"Generating text completion for model {model_name} with prompt '{input_prompt}'...")
    async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT_IN_SECONDS) as client:
        response = await client.post(
            f"{BASE_URL}/get_text_completions_from_input_prompt/",
            json={
                "input_prompt": input_prompt,
                "llm_model_name": model_name,
                "temperature": 0.7,
                "number_of_completions_to_generate": 1,
                "number_of_tokens_to_generate": 150
            }
        )
        completion_results = response.json()
        print(f"Text completion generated. Results: {completion_results}")
        return completion_results

async def ask_question_about_image(image_path: str, question: str, model_name: str) -> Dict[str, Any]:
    print(f"Asking question '{question}' about image at {image_path} with model {model_name}...")
    with open(os.path.expanduser(image_path), "rb") as file:
        async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT_IN_SECONDS) as client:
            response = await client.post(
                f"{BASE_URL}/ask_question_about_image/",
                files={"image": file},
                data={
                    "question": question,
                    "llm_model_name": model_name,
                    "temperature": 0.7,
                    "number_of_tokens_to_generate": 256,
                    "number_of_completions_to_generate": 1
                }
            )
        image_question_results = response.json()
        print(f"Question about image answered. Results: {image_question_results}")
        return image_question_results

async def compute_transcript_with_whisper(audio_path: str) -> Dict[str, Any]:
    print(f"Computing transcript for audio file at {audio_path}...")
    with open(os.path.expanduser(audio_path), "rb") as file:
        async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT_IN_SECONDS) as client:
            response = await client.post(
                f"{BASE_URL}/compute_transcript_with_whisper_from_audio/",
                files={"file": file},
                data={
                    "compute_embeddings_for_resulting_transcript_document": True,
                    "llm_model_name": config("DEFAULT_MODEL_NAME", default="Meta-Llama-3-8B-Instruct.Q3_K_S"),
                    "embedding_pooling_method": "svd",
                    "corpus_identifier_string": CORPUS_IDENTIFIER_STRING
                }
            )
        transcript_results = response.json()
        print(f"Transcript computed. Results: {transcript_results}")
        return transcript_results

async def main():
    start_time = time.time()
    print("Starting the main async process...")

    if USE_MANUAL_MODEL_NAME_LIST:
        model_names = MANUAL_MODEL_NAME_LIST
    else:
        model_names = await get_model_names()
    embedding_pooling_methods = await get_embedding_pooling_methods()

    results = {}
    for model_name in model_names:
        for embedding_pooling_method in embedding_pooling_methods:
            print(f"\n{'_'*100}\n")
            print(f"Computing embeddings for model {model_name} and pooling method {embedding_pooling_method}...")
            total_time = await compute_document_embeddings(model_name, embedding_pooling_method)
            print(f"Embeddings computed in {total_time:.2f} seconds.")
            results[(model_name, embedding_pooling_method)] = total_time

    for model_name, embedding_pooling_method in results:
        print(f"\n{'_'*100}\n")
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

        print(f"Performing advanced semantic search for model {model_name} and pooling method {embedding_pooling_method}...")
        advanced_search_results = await perform_advanced_semantic_search(model_name, embedding_pooling_method)
        advanced_filename = f"{model_name}_{embedding_pooling_method}_advanced_search_results.json"
        advanced_file_path = os.path.join(saved_outputs_dir, advanced_filename)
        with open(advanced_file_path, "w") as f:
            json.dump(advanced_search_results, f, indent=2)
        print(f"Advanced search results saved to {advanced_file_path}.")

    # Test text completion
    for model_name in model_names:
        print(f"\n{'_'*100}\n")
        print(f"Generating text completion for model {model_name}...")
        completion_results = await generate_text_completion(TEXT_PROMPT, model_name)
        completion_file = f"{model_name}_text_completion.json"
        completion_file_path = os.path.join(saved_outputs_dir, completion_file)
        with open(completion_file_path, "w") as f:
            json.dump(completion_results, f, indent=2)
        print(f"Text completion results saved to {completion_file_path}.")

    # Test image question
    print(f"\n{'_'*100}\n")
    image_question_model_name = config("DEFAULT_MULTI_MODAL_MODEL_NAME", default="llava-llama-3-8b-v1_1-int4")
    print(f"Asking question about image with model {image_question_model_name}...")
    image_question_results = await ask_question_about_image(IMAGE_PATH, "What is happening in this image?", image_question_model_name)
    image_question_file = f"{image_question_model_name}_image_question.json"
    image_question_file_path = os.path.join(saved_outputs_dir, image_question_file)
    with open(image_question_file_path, "w") as f:
        json.dump(image_question_results, f, indent=2)
    print(f"Image question results saved to {image_question_file_path}.")

    # Test Whisper transcript
    print(f"\n{'_'*100}\n")
    print(f"Computing transcript with Whisper for audio file {AUDIO_PATH}...")
    transcript_results = await compute_transcript_with_whisper(AUDIO_PATH)
    transcript_file = "whisper_transcript.json"
    transcript_file_path = os.path.join(saved_outputs_dir, transcript_file)
    with open(transcript_file_path, "w") as f:
        json.dump(transcript_results, f, indent=2)
    print(f"Whisper transcript results saved to {transcript_file_path}.")

    end_time = time.time()
    print(f"\n{'_'*100}\n")
    print(f"All tests completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    asyncio.run(main())
