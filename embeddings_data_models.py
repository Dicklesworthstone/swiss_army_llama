from sqlalchemy import Column, String, Float, DateTime, Integer, UniqueConstraint, ForeignKey, LargeBinary
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import declarative_base, relationship, validates
from hashlib import sha3_256
from pydantic import BaseModel
from typing import List, Optional
from decouple import config
from sqlalchemy import event
from hashlib import sha256

Base = declarative_base()
DEFAULT_MODEL_NAME = config("DEFAULT_MODEL_NAME", default="llama2_7b_chat_uncensored", cast=str) 


class TextEmbedding(Base):
    __tablename__ = "embeddings"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, index=True)
    text_hash = Column(String, index=True)
    model_name = Column(String, index=True)
    embedding_json = Column(String)
    ip_address = Column(String)
    request_time = Column(DateTime)
    response_time = Column(DateTime)
    total_time = Column(Float)
    document_id = Column(Integer, ForeignKey('document_embeddings.id'))
    document = relationship("DocumentEmbedding", back_populates="embeddings")
    __table_args__ = (UniqueConstraint('text_hash', 'model_name', name='_text_hash_model_uc'),)
    @validates('text')
    def update_text_hash(self, key, text):
        self.text_hash = sha3_256(text.encode('utf-8')).hexdigest()
        return text
        
        
class DocumentEmbedding(Base):
    __tablename__ = "document_embeddings"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey('documents.id'))
    filename = Column(String)
    mimetype = Column(String)
    file_hash = Column(String, index=True)
    model_name = Column(String, index=True)    
    file_data = Column(LargeBinary) # To store the original file
    document_embedding_results_json = Column(JSON) # To store the embedding results JSON
    ip_address = Column(String)
    request_time = Column(DateTime)
    response_time = Column(DateTime)
    total_time = Column(Float)    
    document = relationship("Document", back_populates="document_embeddings")
    embeddings = relationship("TextEmbedding", back_populates="document")
    __table_args__ = (UniqueConstraint('file_hash', 'model_name', name='_file_hash_model_uc'),)
    

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, index=True)
    document_hash = Column(String, index=True)
    document_embeddings = relationship("DocumentEmbedding", back_populates="document")
    def update_hash(self): # Concatenate specific attributes from the document_embeddings relationship
        hash_data = "".join([emb.filename + emb.mimetype for emb in self.document_embeddings])
        self.document_hash = sha256(hash_data.encode('utf-8')).hexdigest()

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
    model_name = Column(String, index=True)
    token_level_embedding_json = Column(String)
    ip_address = Column(String)
    request_time = Column(DateTime)
    response_time = Column(DateTime)
    total_time = Column(Float)
    token_level_embedding_bundle_id = Column(Integer, ForeignKey('token_level_embedding_bundles.id'))
    token_level_embedding_bundle = relationship("TokenLevelEmbeddingBundle", back_populates="token_level_embeddings")
    __table_args__ = (UniqueConstraint('token_hash', 'model_name', name='_token_hash_model_uc'),)
    @validates('token')
    def update_token_hash(self, key, token):
        self.token_hash = sha3_256(token.encode('utf-8')).hexdigest()
        return token
    
    
class TokenLevelEmbeddingBundle(Base):
    __tablename__ = "token_level_embedding_bundles"
    id = Column(Integer, primary_key=True, index=True)
    input_text = Column(String, index=True)
    input_text_hash = Column(String, index=True)  # Hash of the input text
    model_name = Column(String, index=True)
    token_level_embeddings_bundle_json = Column(String)  # JSON containing the token-level embeddings
    ip_address = Column(String)
    request_time = Column(DateTime)
    response_time = Column(DateTime)
    total_time = Column(Float)
    token_level_embeddings = relationship("TokenLevelEmbedding", back_populates="token_level_embedding_bundle")
    combined_feature_vector = relationship("TokenLevelEmbeddingBundleCombinedFeatureVector", uselist=False, back_populates="token_level_embedding_bundle")
    __table_args__ = (UniqueConstraint('input_text_hash', 'model_name', name='_input_text_hash_model_uc'),)
    @validates('input_text')
    def update_input_text_hash(self, key, input_text):
        self.input_text_hash = sha3_256(input_text.encode('utf-8')).hexdigest()
        return input_text


class TokenLevelEmbeddingBundleCombinedFeatureVector(Base):
    __tablename__ = "token_level_embedding_bundle_combined_feature_vectors"
    id = Column(Integer, primary_key=True, index=True)
    token_level_embedding_bundle_id = Column(Integer, ForeignKey('token_level_embedding_bundles.id'))
    model_name = Column(String, index=True)
    combined_feature_vector_json = Column(JSON)  # JSON containing the combined feature vector
    combined_feature_vector_hash = Column(String, index=True)  # Hash of the combined feature vector
    token_level_embedding_bundle = relationship("TokenLevelEmbeddingBundle", back_populates="combined_feature_vector")        
    __table_args__ = (UniqueConstraint('combined_feature_vector_hash', 'model_name', name='_combined_feature_vector_hash_model_uc'),)
    @validates('combined_feature_vector_json')
    def update_text_hash(self, key, combined_feature_vector_json):
        self.combined_feature_vector_hash = sha3_256(combined_feature_vector_json.encode('utf-8')).hexdigest()
        return combined_feature_vector_json


# Request/Response models start here:

class EmbeddingRequest(BaseModel):
    text: str
    model_name: Optional[str] = DEFAULT_MODEL_NAME

class SimilarityRequest(BaseModel):
    text1: str
    text2: str
    model_name: Optional[str] = DEFAULT_MODEL_NAME
    similarity_measure: Optional[str] = "hoeffdings_d" # Default to hoeffdings_d; can also be "cosine_similarity" or "hsic"
    
class SemanticSearchRequest(BaseModel):
    query_text: str
    number_of_most_similar_strings_to_return: Optional[int] = 10
    model_name: Optional[str] = DEFAULT_MODEL_NAME
        
class SemanticSearchResponse(BaseModel):
    query_text: str
    results: List[dict]  # List of similar strings and their similarity scores using cosine similarity with Faiss (in descending order)

class EmbeddingResponse(BaseModel):
    embedding: List[float]

class SimilarityResponse(BaseModel):
    text1: str
    text2: str
    similarity_measure: str
    similarity_score: float
    embedding1: List[float]
    embedding2: List[float]
        
class AllStringsResponse(BaseModel):
    strings: List[str]

class AllDocumentsResponse(BaseModel):
    documents: List[str]
