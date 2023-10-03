from sqlalchemy import Column, String, Float, DateTime, Integer, UniqueConstraint, ForeignKey, LargeBinary
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import declarative_base, relationship, validates
from hashlib import sha3_256
from pydantic import BaseModel, field_validator
from typing import List, Optional, Union, Dict
from decouple import config
from sqlalchemy import event
from datetime import datetime

Base = declarative_base()
DEFAULT_MODEL_NAME = config("DEFAULT_MODEL_NAME", default="llama2_7b_chat_uncensored", cast=str) 
DEFAULT_MAX_COMPLETION_TOKENS = config("DEFAULT_MAX_COMPLETION_TOKENS", default=100, cast=int)
DEFAULT_NUMBER_OF_COMPLETIONS_TO_GENERATE = config("DEFAULT_NUMBER_OF_COMPLETIONS_TO_GENERATE", default=4, cast=int)
DEFAULT_COMPLETION_TEMPERATURE = config("DEFAULT_COMPLETION_TEMPERATURE", default=0.7, cast=float)

class TextEmbedding(Base):
    __tablename__ = "embeddings"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, index=True)
    text_hash = Column(String, index=True)
    llm_model_name = Column(String, index=True)
    embedding_json = Column(String)
    ip_address = Column(String)
    request_time = Column(DateTime)
    response_time = Column(DateTime)
    total_time = Column(Float)
    document_file_hash = Column(String, ForeignKey('document_embeddings.file_hash'))
    document = relationship("DocumentEmbedding", back_populates="embeddings")
    __table_args__ = (UniqueConstraint('text_hash', 'llm_model_name', name='_text_hash_model_uc'),)
    @validates('text')
    def update_text_hash(self, key, text):
        self.text_hash = sha3_256(text.encode('utf-8')).hexdigest()
        return text
        
        
class DocumentEmbedding(Base):
    __tablename__ = "document_embeddings"
    id = Column(Integer, primary_key=True, index=True)
    document_hash = Column(String, ForeignKey('documents.document_hash'))
    filename = Column(String)
    mimetype = Column(String)
    file_hash = Column(String, index=True)
    llm_model_name = Column(String, index=True)    
    file_data = Column(LargeBinary) # To store the original file
    document_embedding_results_json = Column(JSON) # To store the embedding results JSON
    ip_address = Column(String)
    request_time = Column(DateTime)
    response_time = Column(DateTime)
    total_time = Column(Float)    
    document = relationship("Document", back_populates="document_embeddings")
    embeddings = relationship("TextEmbedding", back_populates="document")
    __table_args__ = (UniqueConstraint('file_hash', 'llm_model_name', name='_file_hash_model_uc'),)
    

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    llm_model_name = Column(String, index=True)
    document_hash = Column(String, index=True)
    document_embeddings = relationship("DocumentEmbedding", back_populates="document")
    def update_hash(self): # Concatenate specific attributes from the document_embeddings relationship
        hash_data = "".join([emb.filename + emb.mimetype for emb in self.document_embeddings])
        self.document_hash = sha3_256(hash_data.encode('utf-8')).hexdigest()

@event.listens_for(Document.document_embeddings, 'append')
def update_document_hash_on_append(target, value, initiator):
    target.update_hash()

@event.listens_for(Document.document_embeddings, 'remove')
def update_document_hash_on_remove(target, value, initiator):
    target.update_hash()

class TokenLevelEmbedding(Base):
    __tablename__ = "token_level_embeddings"
    id = Column(Integer, primary_key=True, index=True)
    token = Column(String, index=True)
    token_hash = Column(String, index=True)
    llm_model_name = Column(String, index=True)
    token_level_embedding_json = Column(String)
    ip_address = Column(String)
    request_time = Column(DateTime)
    response_time = Column(DateTime)
    total_time = Column(Float)
    token_level_embedding_bundle_id = Column(Integer, ForeignKey('token_level_embedding_bundles.id'))
    token_level_embedding_bundle = relationship("TokenLevelEmbeddingBundle", back_populates="token_level_embeddings")
    __table_args__ = (UniqueConstraint('token_hash', 'llm_model_name', name='_token_hash_model_uc'),)
    @validates('token')
    def update_token_hash(self, key, token):
        self.token_hash = sha3_256(token.encode('utf-8')).hexdigest()
        return token
    
    
class TokenLevelEmbeddingBundle(Base):
    __tablename__ = "token_level_embedding_bundles"
    id = Column(Integer, primary_key=True, index=True)
    input_text = Column(String, index=True)
    input_text_hash = Column(String, index=True)  # Hash of the input text
    llm_model_name = Column(String, index=True)
    token_level_embeddings_bundle_json = Column(String)  # JSON containing the token-level embeddings
    ip_address = Column(String)
    request_time = Column(DateTime)
    response_time = Column(DateTime)
    total_time = Column(Float)
    token_level_embeddings = relationship("TokenLevelEmbedding", back_populates="token_level_embedding_bundle")
    combined_feature_vector = relationship("TokenLevelEmbeddingBundleCombinedFeatureVector", uselist=False, back_populates="token_level_embedding_bundle")
    __table_args__ = (UniqueConstraint('input_text_hash', 'llm_model_name', name='_input_text_hash_model_uc'),)
    @validates('input_text')
    def update_input_text_hash(self, key, input_text):
        self.input_text_hash = sha3_256(input_text.encode('utf-8')).hexdigest()
        return input_text


class TokenLevelEmbeddingBundleCombinedFeatureVector(Base):
    __tablename__ = "token_level_embedding_bundle_combined_feature_vectors"
    id = Column(Integer, primary_key=True, index=True)
    token_level_embedding_bundle_id = Column(Integer, ForeignKey('token_level_embedding_bundles.id'))
    llm_model_name = Column(String, index=True)
    combined_feature_vector_json = Column(JSON)  # JSON containing the combined feature vector
    combined_feature_vector_hash = Column(String, index=True)  # Hash of the combined feature vector
    token_level_embedding_bundle = relationship("TokenLevelEmbeddingBundle", back_populates="combined_feature_vector")        
    __table_args__ = (UniqueConstraint('combined_feature_vector_hash', 'llm_model_name', name='_combined_feature_vector_hash_model_uc'),)
    @validates('combined_feature_vector_json')
    def update_text_hash(self, key, combined_feature_vector_json):
        self.combined_feature_vector_hash = sha3_256(combined_feature_vector_json.encode('utf-8')).hexdigest()
        return combined_feature_vector_json


# Request/Response models start here:

class EmbeddingRequest(BaseModel):
    text: str
    llm_model_name: Optional[str] = DEFAULT_MODEL_NAME

class SimilarityRequest(BaseModel):
    text1: str
    text2: str
    llm_model_name: Optional[str] = DEFAULT_MODEL_NAME
    similarity_measure: Optional[str] = "all"
    @field_validator('similarity_measure')
    def validate_similarity_measure(cls, value):
        valid_measures = ["all", "spearman_rho", "kendall_tau", "approximate_distance_correlation", "jensen_shannon_similarity", "hoeffding_d"]
        if value.lower() not in valid_measures:
            raise ValueError(f"Invalid similarity measure. Supported measures are: {', '.join(valid_measures)}")
        return value.lower()
    
class SemanticSearchRequest(BaseModel):
    query_text: str
    number_of_most_similar_strings_to_return: Optional[int] = 10
    llm_model_name: Optional[str] = DEFAULT_MODEL_NAME
        
class SemanticSearchResponse(BaseModel):
    query_text: str
    results: List[dict]  # List of similar strings and their similarity scores using cosine similarity with Faiss (in descending order)

class AdvancedSemanticSearchRequest(BaseModel):
    query_text: str
    llm_model_name: str = DEFAULT_MODEL_NAME
    similarity_filter_percentage: float = 0.98
    number_of_most_similar_strings_to_return: Optional[int] = None

class AdvancedSemanticSearchResponse(BaseModel):
    query_text: str
    results: List[Dict[str, Union[str, float, Dict[str, float]]]]


class EmbeddingResponse(BaseModel):
    embedding: List[float]

class SimilarityResponse(BaseModel):
    text1: str
    text2: str
    similarity_measure: str
    similarity_score: Union[float, Dict[str, float]]  # Now can be either a float or a dictionary
    embedding1: List[float]
    embedding2: List[float]
        
class AllStringsResponse(BaseModel):
    strings: List[str]

class AllDocumentsResponse(BaseModel):
    documents: List[str]

class TextCompletionRequest(BaseModel):
    input_prompt: str
    llm_model_name: Optional[str] = DEFAULT_MODEL_NAME
    temperature: Optional[float] = DEFAULT_COMPLETION_TEMPERATURE
    grammar_file_string: Optional[str] = ""
    number_of_tokens_to_generate: Optional[int] = DEFAULT_MAX_COMPLETION_TOKENS
    number_of_completions_to_generate: Optional[int] = DEFAULT_NUMBER_OF_COMPLETIONS_TO_GENERATE
    
class TextCompletionResponse(BaseModel):
    input_prompt: str
    llm_model_name: str
    grammar_file_string: str
    number_of_tokens_to_generate: int
    number_of_completions_to_generate: int
    time_taken_in_seconds: float
    generated_text: str
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

class ShowLogsIncrementalModel(BaseModel):
    logs: str
    last_position: int

class AddGrammarRequest(BaseModel):
    bnf_grammar: str
    grammar_file_name: str

class AddGrammarResponse(BaseModel):
    valid_grammar_files: List[str]
