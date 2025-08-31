import pytest
from unittest.mock import Mock, patch, MagicMock
from src.retriever import Retriever
from langchain.schema import Document


class TestRetriever:
    """Test cases for the Retriever class."""
    
    @patch('src.retriever.FAISS')
    @patch('src.retriever.GoogleGenerativeAIEmbeddings')
    @patch('os.path.exists')
    def test_retriever_initialization_success(self, mock_exists, mock_embeddings, mock_faiss):
        """Test successful initialization of Retriever."""
        # Mock file existence
        mock_exists.return_value = True
        
        # Mock FAISS load_local
        mock_vector_store = Mock()
        mock_faiss.load_local.return_value = mock_vector_store
        
        # Initialize retriever
        retriever = Retriever(api_key="test-api-key")
        
        # Assertions
        assert retriever.api_key == "test-api-key"
        assert retriever.vector_store == mock_vector_store
        mock_embeddings.assert_called_once()
        mock_faiss.load_local.assert_called_once_with(
            "data/vector_store_20250831_150902",
            retriever.embeddings,
            allow_dangerous_deserialization=True
        )
    
    @patch('os.path.exists')
    def test_retriever_initialization_no_api_key(self, mock_exists):
        """Test initialization fails without API key."""
        mock_exists.return_value = True
        
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="Google API key must be provided"):
                Retriever()
    
    @patch('src.retriever.GoogleGenerativeAIEmbeddings')
    @patch('os.getenv')
    def test_retriever_initialization_vector_store_not_found(self, mock_getenv, mock_embeddings):
        """Test initialization fails when vector store doesn't exist."""
        mock_getenv.return_value = "test-api-key"
        
        with patch('os.path.exists', return_value=False):
            with pytest.raises(FileNotFoundError, match="Vector store not found"):
                Retriever()
    
    @patch('src.retriever.FAISS')
    @patch('src.retriever.GoogleGenerativeAIEmbeddings')
    @patch('os.path.exists')
    def test_retrieve_success(self, mock_exists, mock_embeddings, mock_faiss):
        """Test successful document retrieval."""
        # Setup mocks
        mock_exists.return_value = True
        mock_vector_store = Mock()
        mock_faiss.load_local.return_value = mock_vector_store
        
        # Mock search results
        mock_results = [
            (Document(page_content="Control valve sizing", metadata={"chunk_index": 0}), 0.85),
            (Document(page_content="Actuator types", metadata={"chunk_index": 1}), 0.75),
            (Document(page_content="Valve characteristics", metadata={"chunk_index": 2}), 0.65)
        ]
        # Mock both methods used in retrieval
        mock_vector_store.max_marginal_relevance_search.return_value = [doc for doc, _ in mock_results]
        mock_vector_store.similarity_search_with_score.return_value = mock_results
        
        # Initialize retriever and retrieve
        retriever = Retriever(api_key="test-api-key")
        results = retriever.retrieve("valve sizing")
        
        # Assertions
        assert len(results) == 3
        assert results[0][1] == 0.85
        assert "Control valve sizing" in results[0][0].page_content
        mock_vector_store.max_marginal_relevance_search.assert_called_once_with(
            "valve sizing",
            k=10,
            fetch_k=20,
            lambda_mult=0.5
        )
        mock_vector_store.similarity_search_with_score.assert_called_once_with("valve sizing", k=20)
    
    @patch('src.retriever.FAISS')
    @patch('src.retriever.GoogleGenerativeAIEmbeddings')
    @patch('os.path.exists')
    def test_retrieve_empty_results(self, mock_exists, mock_embeddings, mock_faiss):
        """Test retrieval with no results."""
        # Setup mocks
        mock_exists.return_value = True
        mock_vector_store = Mock()
        mock_faiss.load_local.return_value = mock_vector_store
        mock_vector_store.max_marginal_relevance_search.return_value = []
        mock_vector_store.similarity_search_with_score.return_value = []
        
        # Initialize retriever and retrieve
        retriever = Retriever(api_key="test-api-key")
        results = retriever.retrieve("non-existent query")
        
        # Assertions
        assert results == []
    
    @patch('src.retriever.FAISS')
    @patch('src.retriever.GoogleGenerativeAIEmbeddings')
    @patch('os.path.exists')
    def test_retrieve_error_handling(self, mock_exists, mock_embeddings, mock_faiss):
        """Test error handling during retrieval."""
        # Setup mocks
        mock_exists.return_value = True
        mock_vector_store = Mock()
        mock_faiss.load_local.return_value = mock_vector_store
        mock_vector_store.max_marginal_relevance_search.side_effect = Exception("Search error")
        
        # Initialize retriever and retrieve
        retriever = Retriever(api_key="test-api-key")
        results = retriever.retrieve("test query")
        
        # Should return empty list on error
        assert results == []
    
    @patch('src.retriever.FAISS')
    @patch('src.retriever.GoogleGenerativeAIEmbeddings')
    @patch('os.path.exists')
    def test_retrieve_custom_k(self, mock_exists, mock_embeddings, mock_faiss):
        """Test retrieval with custom k parameter."""
        # Setup mocks
        mock_exists.return_value = True
        mock_vector_store = Mock()
        mock_faiss.load_local.return_value = mock_vector_store
        mock_vector_store.max_marginal_relevance_search.return_value = []
        mock_vector_store.similarity_search_with_score.return_value = []
        
        # Initialize retriever and retrieve with custom k
        retriever = Retriever(api_key="test-api-key")
        retriever.retrieve("test query", k=5)
        
        # Check that custom k was used
        mock_vector_store.max_marginal_relevance_search.assert_called_once_with(
            "test query",
            k=5,
            fetch_k=20,
            lambda_mult=0.5
        )