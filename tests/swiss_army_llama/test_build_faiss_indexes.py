import pytest
import json
from unittest.mock import AsyncMock
import numpy as np
from swiss_army_llama import build_faiss_indexes, AsyncSessionLocal
import faiss

@pytest.mark.asyncio
async def test_build_faiss_indexes(monkeypatch):
    # Mocking data returned from the database for embeddings
    mock_embedding_data = [("model1", "text1", json.dumps([1.0, 1.0])), ("model1", "text2", json.dumps([1.0, 1.0]))]
    
    # Mocking SQLAlchemy execute method to return our mock data
    async def mock_execute(*args, **kwargs):
        if "SELECT llm_model_name, text, embedding_json FROM embeddings" in args[0]:
            return AsyncMock(fetchall=AsyncMock(return_value=mock_embedding_data))()

    # Mocking the database session
    monkeypatch.setattr(AsyncSessionLocal, "execute", mock_execute)
    
    # Run the function to test
    faiss_indexes, associated_texts_by_model_and_pooling_method = await build_faiss_indexes()
    
    # Verify that FAISS indexes have been built for the mock data
    assert "model1" in faiss_indexes
    
    # Verify that associated texts have been correctly identified
    assert associated_texts_by_model_and_pooling_method["model1"] == ["text1", "text2"]
    
    # Verify that the FAISS index is valid
    embedding_array = np.array([[1.0, 1.0], [1.0, 1.0]]).astype('float32')
    faiss.normalize_L2(embedding_array)
    assert faiss_indexes["model1"].ntotal == len(embedding_array)
