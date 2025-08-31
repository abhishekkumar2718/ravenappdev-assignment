import pytest
from unittest.mock import Mock, patch, MagicMock
from src.presenter import Presenter
from langchain.schema import Document


class TestPresenter:
    """Test cases for the Presenter class."""
    
    @patch('src.presenter.genai')
    @patch('os.getenv')
    def test_presenter_initialization_success(self, mock_getenv, mock_genai):
        """Test successful initialization of Presenter."""
        mock_getenv.return_value = "test-api-key"
        
        presenter = Presenter()
        
        assert presenter.api_key == "test-api-key"
        mock_genai.configure.assert_called_once_with(api_key="test-api-key")
        mock_genai.GenerativeModel.assert_called_once_with('gemini-2.5-flash')
    
    def test_presenter_initialization_no_api_key(self):
        """Test initialization fails without API key."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="Google API key must be provided"):
                Presenter()
    
    @patch('src.presenter.genai')
    def test_present_empty_results(self, mock_genai):
        """Test presentation with no results."""
        presenter = Presenter(api_key="test-api-key")
        
        result = presenter.present("test query", [])
        
        assert result == "No relevant information found for your query."
    
    @patch('src.presenter.genai')
    def test_present_success(self, mock_genai):
        """Test successful content presentation."""
        # Mock the Gemini model
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "The control valve sizing is based on [flow coefficient]([1]) and [pressure drop]([2])."
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        presenter = Presenter(api_key="test-api-key")
        
        # Create test results
        results = [
            (Document(page_content="Flow coefficient Cv is the key parameter", metadata={}), 0.85),
            (Document(page_content="Pressure drop calculations are essential", metadata={}), 0.75),
            (Document(page_content="Valve characteristics affect performance", metadata={}), 0.65)
        ]
        
        result = presenter.present("valve sizing factors", results)
        
        # Check that response was generated
        assert "control valve sizing" in result
        assert "[1]" in result
        assert "[2]" in result
        mock_model.generate_content.assert_called_once()
        
        # Verify prompt contains the query and context
        prompt_arg = mock_model.generate_content.call_args[0][0]
        assert "valve sizing factors" in prompt_arg
        assert "Flow coefficient Cv" in prompt_arg
        assert "Pressure drop calculations" in prompt_arg
    
    @patch('src.presenter.genai')
    def test_present_error_handling(self, mock_genai):
        """Test error handling during presentation."""
        # Mock the Gemini model to raise an error
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("API error")
        mock_genai.GenerativeModel.return_value = mock_model
        
        presenter = Presenter(api_key="test-api-key")
        
        results = [
            (Document(page_content="Test content", metadata={}), 0.85)
        ]
        
        result = presenter.present("test query", results)
        
        assert result == "An error occurred while generating the response."
    
    @patch('src.presenter.genai')
    def test_present_uses_top_5_results(self, mock_genai):
        """Test that presenter only uses top 5 results."""
        # Mock the Gemini model
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Response based on top 5 results."
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        presenter = Presenter(api_key="test-api-key")
        
        # Create more than 5 results
        results = [
            (Document(page_content=f"Content {i}", metadata={}), 0.9 - i*0.1)
            for i in range(10)
        ]
        
        presenter.present("test query", results)
        
        # Check that only top 5 are in the prompt
        prompt_arg = mock_model.generate_content.call_args[0][0]
        assert "Content 0" in prompt_arg
        assert "Content 4" in prompt_arg
        assert "Content 5" not in prompt_arg  # 6th result should not be included