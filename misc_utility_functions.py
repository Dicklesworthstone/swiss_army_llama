from logger_config import logger
from database_functions import AsyncSessionLocal
import socket
import os
import re
import json
import io
import sys
import numpy as np
import faiss
from typing import Any
from collections import defaultdict
from sqlalchemy import text as sql_text

class suppress_stdout_stderr(object):
    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')
        self.old_stdout_fileno_undup    = sys.stdout.fileno()
        self.old_stderr_fileno_undup    = sys.stderr.fileno()
        self.old_stdout_fileno = os.dup ( sys.stdout.fileno() )
        self.old_stderr_fileno = os.dup ( sys.stderr.fileno() )
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        os.dup2 ( self.outnull_file.fileno(), self.old_stdout_fileno_undup )
        os.dup2 ( self.errnull_file.fileno(), self.old_stderr_fileno_undup )
        sys.stdout = self.outnull_file        
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):        
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        os.dup2 ( self.old_stdout_fileno, self.old_stdout_fileno_undup )
        os.dup2 ( self.old_stderr_fileno, self.old_stderr_fileno_undup )
        os.close ( self.old_stdout_fileno )
        os.close ( self.old_stderr_fileno )
        self.outnull_file.close()
        self.errnull_file.close()
    
def safe_path(base_path, file_name):
    abs_base_path = os.path.abspath(base_path)
    abs_user_path = os.path.abspath(os.path.join(base_path, file_name))
    return abs_user_path.startswith(abs_base_path), abs_user_path

def clean_filename_for_url_func(dirty_filename: str) -> str:
    clean_filename = re.sub(r'[^\w\s]', '', dirty_filename) # Remove special characters and replace spaces with underscores
    clean_filename = clean_filename.replace(' ', '_')
    return clean_filename

def is_redis_running(host='localhost', port=6379):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((host, port))
        return True
    except ConnectionRefusedError:
        return False
    finally:
        s.close()

async def build_faiss_indexes():
    global faiss_indexes, token_faiss_indexes, associated_texts_by_model
    if os.environ.get("FAISS_SETUP_DONE") == "1":
        logger.info("Faiss indexes already built by another worker. Skipping.")
        return faiss_indexes, token_faiss_indexes, associated_texts_by_model
    faiss_indexes = {}
    token_faiss_indexes = {} # Separate FAISS indexes for token-level embeddings
    associated_texts_by_model = defaultdict(list)  # Create a dictionary to store associated texts by model name
    async with AsyncSessionLocal() as session:
        result = await session.execute(sql_text("SELECT llm_model_name, text, embedding_json FROM embeddings")) # Query regular embeddings
        token_result = await session.execute(sql_text("SELECT llm_model_name, token, token_level_embedding_json FROM token_level_embeddings")) # Query token-level embeddings
        embeddings_by_model = defaultdict(list)
        token_embeddings_by_model = defaultdict(list)
        for row in result.fetchall(): # Process regular embeddings
            llm_model_name = row[0]
            associated_texts_by_model[llm_model_name].append(row[1])  # Store the associated text by model name
            embeddings_by_model[llm_model_name].append((row[1], json.loads(row[2])))
        for row in token_result.fetchall(): # Process token-level embeddings
            llm_model_name = row[0]
            token_embeddings_by_model[llm_model_name].append(json.loads(row[2]))
        for llm_model_name, embeddings in embeddings_by_model.items():
            logger.info(f"Building Faiss index over embeddings for model {llm_model_name}...")
            embeddings_array = np.array([e[1] for e in embeddings]).astype('float32')
            if embeddings_array.size == 0:
                logger.error(f"No embeddings were loaded from the database for model {llm_model_name}, so nothing to build the Faiss index with!")
                continue
            logger.info(f"Loaded {len(embeddings_array)} embeddings for model {llm_model_name}.")
            logger.info(f"Embedding dimension for model {llm_model_name}: {embeddings_array.shape[1]}")
            logger.info(f"Normalizing {len(embeddings_array)} embeddings for model {llm_model_name}...")
            faiss.normalize_L2(embeddings_array)  # Normalize the vectors for cosine similarity
            faiss_index = faiss.IndexFlatIP(embeddings_array.shape[1])  # Use IndexFlatIP for cosine similarity
            faiss_index.add(embeddings_array)
            logger.info(f"Faiss index built for model {llm_model_name}.")
            faiss_indexes[llm_model_name] = faiss_index  # Store the index by model name
        for llm_model_name, token_embeddings in token_embeddings_by_model.items():
            token_embeddings_array = np.array(token_embeddings).astype('float32')
            if token_embeddings_array.size == 0:
                logger.error(f"No token-level embeddings were loaded from the database for model {llm_model_name}, so nothing to build the Faiss index with!")
                continue
            logger.info(f"Normalizing {len(token_embeddings_array)} token-level embeddings for model {llm_model_name}...")
            faiss.normalize_L2(token_embeddings_array)  # Normalize the vectors for cosine similarity
            token_faiss_index = faiss.IndexFlatIP(token_embeddings_array.shape[1])  # Use IndexFlatIP for cosine similarity
            token_faiss_index.add(token_embeddings_array)
            logger.info(f"Token-level Faiss index built for model {llm_model_name}.")
            token_faiss_indexes[llm_model_name] = token_faiss_index  # Store the token-level index by model name
    os.environ["FAISS_SETUP_DONE"] = "1"
    logger.info("Faiss indexes built.")
    return faiss_indexes, token_faiss_indexes, associated_texts_by_model


def normalize_logprobs(avg_logprob, min_logprob, max_logprob):
    range_logprob = max_logprob - min_logprob
    return (avg_logprob - min_logprob) / range_logprob if range_logprob != 0 else 0.5

def remove_pagination_breaks(text: str) -> str:
    text = re.sub(r'-(\n)(?=[a-z])', '', text) # Remove hyphens at the end of lines when the word continues on the next line
    text = re.sub(r'(?<=\w)(?<![.?!-]|\d)\n(?![\nA-Z])', ' ', text) # Replace line breaks that are not preceded by punctuation or list markers and not followed by an uppercase letter or another line break   
    return text

def sophisticated_sentence_splitter(text):
    text = remove_pagination_breaks(text)
    pattern = r'\.(?!\s*(com|net|org|io)\s)(?![0-9])'  # Split on periods that are not followed by a space and a top-level domain or a number
    pattern += r'|[.!?]\s+'  # Split on whitespace that follows a period, question mark, or exclamation point
    pattern += r'|\.\.\.(?=\s)'  # Split on ellipses that are followed by a space
    sentences = re.split(pattern, text)
    refined_sentences = []
    temp_sentence = ""
    for sentence in sentences:
        if sentence is not None:
            temp_sentence += sentence
            if temp_sentence.count('"') % 2 == 0:  # If the number of quotes is even, then we have a complete sentence
                refined_sentences.append(temp_sentence.strip())
                temp_sentence = ""
    if temp_sentence:
        refined_sentences[-1] += temp_sentence
    return [s.strip() for s in refined_sentences if s.strip()]

def merge_transcript_segments_into_combined_text(segments):
    if not segments:
        return "", [], []
    min_logprob = min(segment['avg_logprob'] for segment in segments)
    max_logprob = max(segment['avg_logprob'] for segment in segments)
    combined_text = ""
    sentence_buffer = ""
    list_of_metadata_dicts = []
    list_of_sentences = []
    char_count = 0
    time_start = None
    time_end = None
    total_logprob = 0.0
    segment_count = 0
    for segment in segments:
        if time_start is None:
            time_start = segment['start']
        time_end = segment['end']
        total_logprob += segment['avg_logprob']
        segment_count += 1
        sentence_buffer += segment['text'] + " "
        sentences = sophisticated_sentence_splitter(sentence_buffer)
        for sentence in sentences:
            combined_text += sentence.strip() + " "
            list_of_sentences.append(sentence.strip())
            char_count += len(sentence.strip()) + 1  # +1 for the space
            avg_logprob = total_logprob / segment_count
            model_confidence_score = normalize_logprobs(avg_logprob, min_logprob, max_logprob)
            metadata = {
                'start_char_count': char_count - len(sentence.strip()) - 1,
                'end_char_count': char_count - 2,
                'time_start': time_start,
                'time_end': time_end,
                'model_confidence_score': model_confidence_score
            }
            list_of_metadata_dicts.append(metadata)
        sentence_buffer = sentences[-1] if len(sentences) % 2 != 0 else ""
    return combined_text, list_of_metadata_dicts, list_of_sentences
    
class JSONAggregator:
    def __init__(self):
        self.completions = []
        self.aggregate_result = None

    @staticmethod
    def weighted_vote(values, weights):
        tally = defaultdict(float)
        for v, w in zip(values, weights):
            tally[v] += w
        return max(tally, key=tally.get)

    @staticmethod
    def flatten_json(json_obj, parent_key='', sep='->'):
        items = {}
        for k, v in json_obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(JSONAggregator.flatten_json(v, new_key, sep=sep))
            else:
                items[new_key] = v
        return items

    @staticmethod
    def get_value_by_path(json_obj, path, sep='->'):
        keys = path.split(sep)
        item = json_obj
        for k in keys:
            item = item[k]
        return item

    @staticmethod
    def set_value_by_path(json_obj, path, value, sep='->'):
        keys = path.split(sep)
        item = json_obj
        for k in keys[:-1]:
            item = item.setdefault(k, {})
        item[keys[-1]] = value

    def calculate_path_weights(self):
        all_paths = []
        for j in self.completions:
            all_paths += list(self.flatten_json(j).keys())
        path_weights = defaultdict(float)
        for path in all_paths:
            path_weights[path] += 1.0
        return path_weights

    def aggregate(self):
        path_weights = self.calculate_path_weights()
        aggregate = {}
        for path, weight in path_weights.items():
            values = [self.get_value_by_path(j, path) for j in self.completions if path in self.flatten_json(j)]
            weights = [weight] * len(values)
            aggregate_value = self.weighted_vote(values, weights)
            self.set_value_by_path(aggregate, path, aggregate_value)
        self.aggregate_result = aggregate

class FakeUploadFile:
    def __init__(self, filename: str, content: Any, content_type: str = 'text/plain'):
        self.filename = filename
        self.content_type = content_type
        self.file = io.BytesIO(content)
    def read(self, size: int = -1) -> bytes:
        return self.file.read(size)
    def seek(self, offset: int, whence: int = 0) -> int:
        return self.file.seek(offset, whence)
    def tell(self) -> int:
        return self.file.tell()