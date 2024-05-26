from sqlalchemy import Column, String, Float, DateTime, Integer, UniqueConstraint, ForeignKey, LargeBinary
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.ext.declarative import declared_attr
from hashlib import sha3_256
from pydantic import BaseModel, field_validator
from typing import List, Optional, Union, Dict
from decouple import config
from sqlalchemy import event
from datetime import datetime

Base = declarative_base()
DEFAULT_MODEL_NAME = config("DEFAULT_MODEL_NAME", default="Meta-Llama-3-8B-Instruct.Q3_K_S", cast=str) 
DEFAULT_MULTI_MODAL_MODEL_NAME = config("DEFAULT_MULTI_MODAL_MODEL_NAME", default="llava-llama-3-8b-v1_1-int4", cast=str)
DEFAULT_MAX_COMPLETION_TOKENS = config("DEFAULT_MAX_COMPLETION_TOKENS", default=100, cast=int)
DEFAULT_NUMBER_OF_COMPLETIONS_TO_GENERATE = config("DEFAULT_NUMBER_OF_COMPLETIONS_TO_GENERATE", default=4, cast=int)
DEFAULT_COMPLETION_TEMPERATURE = config("DEFAULT_COMPLETION_TEMPERATURE", default=0.7, cast=float)
DEFAULT_EMBEDDING_POOLING_METHOD = config("DEFAULT_EMBEDDING_POOLING_METHOD", default="svd", cast=str)

class SerializerMixin:
    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()

    def as_dict(self):
        return {c.key: getattr(self, c.key) for c in self.__table__.columns}
    
class TextEmbedding(Base, SerializerMixin):
    __tablename__ = "embeddings"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, index=True)
    text_hash = Column(String, index=True)
    embedding_pooling_method = Column(String, index=True)
    embedding_hash = Column(String, index=True)
    llm_model_name = Column(String, index=True)
    corpus_identifier_string = Column(String, index=True)    
    embedding_json = Column(String)
    ip_address = Column(String)
    request_time = Column(DateTime)
    response_time = Column(DateTime)
    total_time = Column(Float)
    document_file_hash = Column(String, ForeignKey('document_embeddings.document_file_hash'))
    document = relationship("DocumentEmbedding", back_populates="embeddings", foreign_keys=[document_file_hash, corpus_identifier_string])
    __table_args__ = (UniqueConstraint('embedding_hash', name='_embedding_hash_uc'),)

class DocumentEmbedding(Base):
    __tablename__ = "document_embeddings"
    id = Column(Integer, primary_key=True, index=True)
    document_hash = Column(String, ForeignKey('documents.document_hash'))
    filename = Column(String)
    mimetype = Column(String)
    document_file_hash = Column(String, index=True)
    embedding_pooling_method = Column(String, index=True)
    llm_model_name = Column(String, index=True)
    corpus_identifier_string = Column(String, index=True)
    file_data = Column(LargeBinary)  # To store the original file
    sentences = Column(String)
    document_embedding_results_json_compressed_binary = Column(LargeBinary)  # To store the embedding results JSON
    ip_address = Column(String)
    request_time = Column(DateTime)
    response_time = Column(DateTime)
    total_time = Column(Float)
    embeddings = relationship("TextEmbedding", back_populates="document", foreign_keys=[TextEmbedding.document_file_hash])
    __table_args__ = (UniqueConstraint('document_embedding_results_json_compressed_binary', name='_document_embedding_results_json_compressed_binary_uc'),)
    document = relationship("Document", back_populates="document_embeddings", foreign_keys=[document_hash])

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    llm_model_name = Column(String, index=True)
    corpus_identifier_string = Column(String, index=True)    
    document_hash = Column(String, index=True)
    document_embeddings = relationship("DocumentEmbedding", back_populates="document", foreign_keys=[DocumentEmbedding.document_hash])
    def update_hash(self):  # Concatenate specific attributes from the document_embeddings relationship
        hash_data = "".join([emb.filename + emb.mimetype for emb in self.document_embeddings])
        self.document_hash = sha3_256(hash_data.encode('utf-8')).hexdigest()
@event.listens_for(Document.document_embeddings, 'append')
def update_document_hash_on_append(target, value, initiator):
    target.update_hash()
@event.listens_for(Document.document_embeddings, 'remove')
def update_document_hash_on_remove(target, value, initiator):
    target.update_hash()

# Request/Response models start here:

class EmbeddingRequest(BaseModel):
    text: str
    llm_model_name: str
    embedding_pooling_method: str
    corpus_identifier_string: str

class SimilarityRequest(BaseModel):
    text1: str
    text2: str
    llm_model_name: str
    embedding_pooling_method: str
    similarity_measure: str
    @field_validator('similarity_measure')
    def validate_similarity_measure(cls, value):
        valid_measures = ["all", "spearman_rho", "kendall_tau", "approximate_distance_correlation", "jensen_shannon_similarity", "hoeffding_d"]
        if value.lower() not in valid_measures:
            raise ValueError(f"Invalid similarity measure. Supported measures are: {', '.join(valid_measures)}")
        return value.lower()
    
class SemanticSearchRequest(BaseModel):
    query_text: str
    number_of_most_similar_strings_to_return: int
    llm_model_name: str
    embedding_pooling_method: str
    corpus_identifier_string: str
        
class SemanticSearchResponse(BaseModel):
    query_text: str
    corpus_identifier_string: str
    embedding_pooling_method: str
    results: List[dict]  # List of similar strings and their similarity scores using cosine similarity with Faiss (in descending order)

class AdvancedSemanticSearchRequest(BaseModel):
    query_text: str
    llm_model_name: str
    embedding_pooling_method: str
    corpus_identifier_string: str
    similarity_filter_percentage: float
    number_of_most_similar_strings_to_return: int
    result_sorting_metric: str
    @field_validator('result_sorting_metric')
    def validate_similarity_measure(cls, value):
        valid_measures = ["all", "spearman_rho", "kendall_tau", "approximate_distance_correlation", "jensen_shannon_similarity", "hoeffding_d"]
        if value.lower() not in valid_measures:
            raise ValueError(f"Invalid similarity measure. Supported measures are: {', '.join(valid_measures)}")
        return value.lower()
    
class AdvancedSemanticSearchResponse(BaseModel):
    query_text: str
    corpus_identifier_string: str
    embedding_pooling_method: str
    results: List[Dict[str, Union[str, float, Dict[str, float]]]]

class EmbeddingResponse(BaseModel):
    id: int
    text: str
    text_hash: str
    embedding_pooling_method: str
    embedding_hash: str
    llm_model_name: str
    corpus_identifier_string: str
    embedding_json: str
    ip_address: Optional[str]
    request_time: datetime
    response_time: datetime
    total_time: float
    document_file_hash: Optional[str]
    embedding: List[float]

class SimilarityResponse(BaseModel):
    text1: str
    text2: str
    similarity_measure: str
    embedding_pooling_method: str
    similarity_score: Union[float, Dict[str, float]]  # Now can be either a float or a dictionary
    embedding1: List[float]
    embedding2: List[float]
        
class AllStringsResponse(BaseModel):
    strings: List[str]

class AllDocumentsResponse(BaseModel):
    documents: List[str]

class TextCompletionRequest(BaseModel):
    input_prompt: str
    llm_model_name: str
    temperature: float
    grammar_file_string: str
    number_of_tokens_to_generate: int
    number_of_completions_to_generate: int
    
class TextCompletionResponse(BaseModel):
    input_prompt: str
    llm_model_name: str
    grammar_file_string: str
    number_of_tokens_to_generate: int
    number_of_completions_to_generate: int
    time_taken_in_seconds: float
    generated_text: str
    finish_reason: str
    llm_model_usage_json: str

class ImageQuestionResponse(BaseModel):
    question: str
    llm_model_name: str
    image_hash: str
    time_taken_in_seconds: float
    number_of_tokens_to_generate: int
    number_of_completions_to_generate: int
    time_taken_in_seconds: float
    generated_text: str
    finish_reason: str
    llm_model_usage_json: str
    
class AudioTranscript(Base):
    __tablename__ = "audio_transcripts"
    audio_file_hash = Column(String, primary_key=True, index=True)
    audio_file_name = Column(String, index=True)
    audio_file_size_mb = Column(Float)  # File size in MB
    segments_json = Column(JSON)  # Transcribed segments as JSON
    combined_transcript_text = Column(String)
    combined_transcript_text_list_of_metadata_dicts = Column(JSON)
    info_json = Column(JSON)  # Transcription info as JSON
    ip_address = Column(String)
    request_time = Column(DateTime)
    response_time = Column(DateTime)
    total_time = Column(Float)
    corpus_identifier_string = Column(String, index=True)

class AudioTranscriptResponse(BaseModel):
    audio_file_hash: str
    audio_file_name: str
    audio_file_size_mb: float
    segments_json: List[dict]
    combined_transcript_text: str
    combined_transcript_text_list_of_metadata_dicts: List[dict]
    info_json: dict
    url_to_download_zip_file_of_embeddings: str
    ip_address: str
    request_time: datetime
    response_time: datetime
    total_time: float
    url_to_download_zip_file_of_embeddings: str
    llm_model_name: str
    embedding_pooling_method: str
    corpus_identifier_string: str
    
class ShowLogsIncrementalModel(BaseModel):
    logs: str
    last_position: int

class AddGrammarRequest(BaseModel):
    bnf_grammar: str
    grammar_file_name: str

class AddGrammarResponse(BaseModel):
    valid_grammar_files: List[str]

def fill_default_values_in_request(request):
    if isinstance(request, EmbeddingRequest):
        if request.llm_model_name is None:
            request.llm_model_name = DEFAULT_MODEL_NAME
        if request.embedding_pooling_method is None:
            request.embedding_pooling_method = DEFAULT_EMBEDDING_POOLING_METHOD
        if request.corpus_identifier_string is None:
            request.corpus_identifier_string = ""
    elif isinstance(request, SimilarityRequest):
        if request.llm_model_name is None:
            request.llm_model_name = DEFAULT_MODEL_NAME
        if request.embedding_pooling_method is None:
            request.embedding_pooling_method = DEFAULT_EMBEDDING_POOLING_METHOD
        if request.similarity_measure is None:
            request.similarity_measure = "all"
    elif isinstance(request, SemanticSearchRequest):
        if request.llm_model_name is None:
            request.llm_model_name = DEFAULT_MODEL_NAME
        if request.embedding_pooling_method is None:
            request.embedding_pooling_method = DEFAULT_EMBEDDING_POOLING_METHOD
        if request.corpus_identifier_string is None:
            request.corpus_identifier_string = ""
    elif isinstance(request, AdvancedSemanticSearchRequest):
        if request.llm_model_name is None:
            request.llm_model_name = DEFAULT_MODEL_NAME
        if request.embedding_pooling_method is None:
            request.embedding_pooling_method = DEFAULT_EMBEDDING_POOLING_METHOD
        if request.corpus_identifier_string is None:
            request.corpus_identifier_string = ""
        if request.similarity_filter_percentage is None:
            request.similarity_filter_percentage = 0.01
        if request.number_of_most_similar_strings_to_return is None:
            request.number_of_most_similar_strings_to_return = 10
        if request.result_sorting_metric is None:
            request.result_sorting_metric = "hoeffding_d"
    elif isinstance(request, TextCompletionRequest):
        if request.llm_model_name is None:
            request.llm_model_name = DEFAULT_MODEL_NAME
        if request.temperature is None:
            request.temperature = DEFAULT_COMPLETION_TEMPERATURE
        if request.grammar_file_string is None:
            request.grammar_file_string = ""
        if request.number_of_tokens_to_generate is None:
            request.number_of_tokens_to_generate = DEFAULT_MAX_COMPLETION_TOKENS
        if request.number_of_completions_to_generate is None:
            request.number_of_completions_to_generate = DEFAULT_NUMBER_OF_COMPLETIONS_TO_GENERATE
