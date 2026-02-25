"""
Test suite for InvoiceProcessor class
Tests for LLM-based invoice processing with OCR integration
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import sys
import os
from PIL import Image, ImageDraw

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from invoice_processor import InvoiceProcessor
from invoice_schema import Invoice, Seller, Client, LineItem, Summary, Address


@pytest.mark.phase2
class TestInvoiceProcessorInitialization:
    """Test suite for InvoiceProcessor initialization"""
    
    def test_init_default_parameters(self):
        """Test initialization with default parameters"""
        processor = InvoiceProcessor()
        assert processor.model_path == Path("./models/Phi-3.5-mini-instruct-Q4_K_M.gguf")
        assert processor.schema_path == Path("./schema.json")
        assert processor.n_ctx == 4096
        assert processor.n_threads == 8
        assert processor.max_retries == 3
        assert processor.llm is None
        assert processor.ocr_text is None
        assert processor.structured_data is None
        assert processor.validation_errors == []
    
    def test_init_custom_parameters(self):
        """Test initialization with custom parameters"""
        processor = InvoiceProcessor(
            model_path="./custom/model.gguf",
            n_ctx=2048,
            n_threads=4,
            schema_path="./custom_schema.json",
            max_retries=5
        )
        assert processor.model_path == Path("./custom/model.gguf")
        assert processor.schema_path == Path("./custom_schema.json")
        assert processor.n_ctx == 2048
        assert processor.n_threads == 4
        assert processor.max_retries == 5
    
    def test_init_loads_schema(self, tmp_path):
        """Test that initialization loads schema if available"""
        schema_file = tmp_path / "test_schema.json"
        test_schema = {"invoice_number": "", "issue_date": ""}
        schema_file.write_text(json.dumps(test_schema))
        
        processor = InvoiceProcessor(schema_path=str(schema_file))
        assert processor.schema == test_schema
    
    def test_init_missing_schema_file(self):
        """Test initialization with missing schema file"""
        processor = InvoiceProcessor(schema_path="./nonexistent_schema.json")
        assert processor.schema is None


@pytest.mark.phase2
class TestLoadSchema:
    """Test suite for _load_schema method"""
    
    def test_load_schema_success(self, tmp_path):
        """Test successful schema loading"""
        schema_file = tmp_path / "schema.json"
        test_schema = {
            "invoice_number": "",
            "issue_date": "",
            "seller": {"name": ""}
        }
        schema_file.write_text(json.dumps(test_schema))
        
        processor = InvoiceProcessor(schema_path=str(schema_file))
        assert processor.schema == test_schema
    
    def test_load_schema_invalid_json(self, tmp_path):
        """Test loading invalid JSON schema"""
        schema_file = tmp_path / "invalid.json"
        schema_file.write_text("{ invalid json }")
        
        processor = InvoiceProcessor(schema_path=str(schema_file))
        assert processor.schema is None
    
    def test_load_schema_empty_file(self, tmp_path):
        """Test loading empty schema file"""
        schema_file = tmp_path / "empty.json"
        schema_file.write_text("")
        
        processor = InvoiceProcessor(schema_path=str(schema_file))
        assert processor.schema is None
    
    def test_load_schema_nonexistent_file(self):
        """Test loading nonexistent schema file"""
        processor = InvoiceProcessor(schema_path="./nonexistent.json")
        assert processor.schema is None


@pytest.mark.phase2
class TestLoadModel:
    """Test suite for load_model method"""
    
    def test_load_model_file_not_exists(self):
        """Test loading model when file doesn't exist"""
        processor = InvoiceProcessor(model_path="./nonexistent_model.gguf")
        result = processor.load_model()
        assert result is False
        assert processor.llm is None
    
    @patch('invoice_processor.Llama')
    def test_load_model_success(self, mock_llama, tmp_path):
        """Test successful model loading"""
        # Create a dummy model file
        model_file = tmp_path / "test_model.gguf"
        model_file.write_bytes(b"dummy model data")
        
        mock_llm_instance = MagicMock()
        mock_llama.return_value = mock_llm_instance
        
        processor = InvoiceProcessor(model_path=str(model_file))
        result = processor.load_model()
        
        assert result is True
        assert processor.llm == mock_llm_instance
        mock_llama.assert_called_once_with(
            model_path=str(model_file),
            n_ctx=4096,
            n_threads=8,
            verbose=False
        )
    
    @patch('invoice_processor.Llama')
    def test_load_model_exception(self, mock_llama, tmp_path):
        """Test model loading with exception"""
        model_file = tmp_path / "test_model.gguf"
        model_file.write_bytes(b"dummy model data")
        
        mock_llama.side_effect = Exception("Model loading error")
        
        processor = InvoiceProcessor(model_path=str(model_file))
        result = processor.load_model()
        
        assert result is False
        assert processor.llm is None
    
    @patch('invoice_processor.Llama')
    def test_load_model_custom_parameters(self, mock_llama, tmp_path):
        """Test model loading with custom parameters"""
        model_file = tmp_path / "test_model.gguf"
        model_file.write_bytes(b"dummy model data")
        
        mock_llm_instance = MagicMock()
        mock_llama.return_value = mock_llm_instance
        
        processor = InvoiceProcessor(
            model_path=str(model_file),
            n_ctx=2048,
            n_threads=4
        )
        processor.load_model()
        
        mock_llama.assert_called_once_with(
            model_path=str(model_file),
            n_ctx=2048,
            n_threads=4,
            verbose=False
        )


@pytest.mark.phase2
class TestExtractTextFromImage:
    """Test suite for extract_text_from_image method"""
    
    @pytest.fixture
    def temp_image(self, tmp_path):
        """Create a temporary test image"""
        img_path = tmp_path / "test_invoice.png"
        img = Image.new('RGB', (400, 200), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), "INVOICE #12345", fill='black')
        draw.text((20, 60), "Amount: $100.00", fill='black')
        img.save(img_path)
        return str(img_path)
    
    @patch('invoice_processor.OCRExtractor')
    def test_extract_text_success(self, mock_ocr_class, temp_image):
        """Test successful text extraction"""
        mock_extractor = MagicMock()
        mock_extractor.process.return_value = "INVOICE #12345\nAmount: $100.00"
        mock_ocr_class.return_value = mock_extractor
        
        processor = InvoiceProcessor()
        result = processor.extract_text_from_image(temp_image)
        
        assert result == "INVOICE #12345\nAmount: $100.00"
        assert processor.ocr_text == "INVOICE #12345\nAmount: $100.00"
        mock_ocr_class.assert_called_once_with(temp_image)
        mock_extractor.process.assert_called_once()
    
    @patch('invoice_processor.OCRExtractor')
    def test_extract_text_failure(self, mock_ocr_class):
        """Test text extraction failure"""
        mock_extractor = MagicMock()
        mock_extractor.process.return_value = None
        mock_ocr_class.return_value = mock_extractor
        
        processor = InvoiceProcessor()
        result = processor.extract_text_from_image("test.jpg")
        
        assert result is None
        assert processor.ocr_text is None
    
    @patch('invoice_processor.OCRExtractor')
    def test_extract_text_empty_result(self, mock_ocr_class):
        """Test extraction with empty result"""
        mock_extractor = MagicMock()
        mock_extractor.process.return_value = ""
        mock_ocr_class.return_value = mock_extractor
        
        processor = InvoiceProcessor()
        result = processor.extract_text_from_image("test.jpg")
        
        assert result == ""
        assert processor.ocr_text == ""


@pytest.mark.phase2
class TestCreateExtractionPrompt:
    """Test suite for create_extraction_prompt method"""
    
    def test_create_prompt_basic(self, tmp_path):
        """Test basic prompt creation"""
        schema_file = tmp_path / "schema.json"
        test_schema = {"invoice_number": "", "issue_date": ""}
        schema_file.write_text(json.dumps(test_schema))
        
        processor = InvoiceProcessor(schema_path=str(schema_file))
        ocr_text = "INVOICE #12345\nDate: 2024-01-01"
        
        prompt = processor.create_extraction_prompt(ocr_text)
        
        assert "INVOICE #12345" in prompt
        assert "Date: 2024-01-01" in prompt
        assert "JSON Schema:" in prompt
        assert "invoice_number" in prompt
        assert "JSON Output:" in prompt
    
    def test_create_prompt_with_validation_error(self, tmp_path):
        """Test prompt creation with validation error"""
        schema_file = tmp_path / "schema.json"
        test_schema = {"invoice_number": ""}
        schema_file.write_text(json.dumps(test_schema))
        
        processor = InvoiceProcessor(schema_path=str(schema_file))
        ocr_text = "INVOICE #12345"
        validation_error = "Field 'invoice_number' is required"
        
        prompt = processor.create_extraction_prompt(ocr_text, validation_error)
        
        assert "validation errors:" in prompt
        assert "Field 'invoice_number' is required" in prompt
        assert "fix these errors" in prompt
    
    def test_create_prompt_no_schema(self):
        """Test prompt creation without schema"""
        processor = InvoiceProcessor(schema_path="./nonexistent.json")
        ocr_text = "INVOICE #12345"
        
        prompt = processor.create_extraction_prompt(ocr_text)
        
        assert "INVOICE #12345" in prompt
        assert "{}" in prompt  # Empty schema
    
    def test_create_prompt_formatting_rules(self, tmp_path):
        """Test that prompt includes formatting rules"""
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps({"test": ""}))
        
        processor = InvoiceProcessor(schema_path=str(schema_file))
        prompt = processor.create_extraction_prompt("test text")
        
        assert "Return ONLY valid JSON" in prompt
        assert "Do not include explanations" in prompt
        assert "If a field cannot be found, return null" in prompt
        assert "All numeric values must be numbers" in prompt
        assert "Convert dates to ISO format" in prompt


@pytest.mark.phase2
class TestValidateInvoiceData:
    """Test suite for validate_invoice_data method"""
    
    def test_validate_valid_data(self):
        """Test validation with valid invoice data"""
        processor = InvoiceProcessor()
        valid_data = {
            "invoice_number": "INV-12345",
            "issue_date": "2024-01-01",
            "currency": "USD",
            "seller": {"name": "Test Seller"},
            "client": {"name": "Test Client"},
            "items": [],
            "summary": {"net_total": 100.0}
        }
        
        is_valid, error_msg, validated_invoice = processor.validate_invoice_data(valid_data)
        
        assert is_valid is True
        assert error_msg is None
        assert validated_invoice is not None
        assert isinstance(validated_invoice, Invoice)
        assert validated_invoice.invoice_number == "INV-12345"
    
    def test_validate_minimal_data(self):
        """Test validation with minimal data (all optional fields)"""
        processor = InvoiceProcessor()
        minimal_data = {}
        
        is_valid, error_msg, validated_invoice = processor.validate_invoice_data(minimal_data)
        
        assert is_valid is True
        assert error_msg is None
        assert validated_invoice is not None
    
    def test_validate_invalid_data_type(self):
        """Test validation with invalid data types"""
        processor = InvoiceProcessor()
        invalid_data = {
            "invoice_number": 12345,  # Should be string
            "items": "not a list"  # Should be list
        }
        
        is_valid, error_msg, validated_invoice = processor.validate_invoice_data(invalid_data)
        
        assert is_valid is False
        assert error_msg is not None
        assert validated_invoice is None
    
    def test_validate_nested_structures(self):
        """Test validation with nested structures"""
        processor = InvoiceProcessor()
        data_with_nested = {
            "seller": {
                "name": "Test Seller",
                "address": {
                    "street": "123 Main St",
                    "city": "Test City",
                    "postal_code": "12345"
                }
            },
            "items": [
                {
                    "number": 1,
                    "description": "Test Item",
                    "quantity": 2.0,
                    "net_amount": 100.0
                }
            ]
        }
        
        is_valid, error_msg, validated_invoice = processor.validate_invoice_data(data_with_nested)
        
        assert is_valid is True
        assert validated_invoice.seller.name == "Test Seller"
        assert validated_invoice.seller.address.city == "Test City"
        assert len(validated_invoice.items) == 1
    
    def test_validate_exception_handling(self):
        """Test validation with unexpected exception"""
        processor = InvoiceProcessor()
        # Pass non-dict data to trigger unexpected error
        invalid_data = "not a dictionary"
        
        is_valid, error_msg, validated_invoice = processor.validate_invoice_data(invalid_data)
        
        assert is_valid is False
        assert "Unexpected validation error" in error_msg
        assert validated_invoice is None


@pytest.mark.phase2
class TestExtractJsonFromResponse:
    """Test suite for _extract_json_from_response method"""
    
    def test_extract_plain_json(self):
        """Test extraction of plain JSON"""
        processor = InvoiceProcessor()
        response = '{"invoice_number": "12345", "amount": 100.0}'
        
        result = processor._extract_json_from_response(response)
        
        assert result == {"invoice_number": "12345", "amount": 100.0}
    
    def test_extract_json_with_markdown_json_block(self):
        """Test extraction from markdown JSON code block"""
        processor = InvoiceProcessor()
        response = '```json\n{"invoice_number": "12345"}\n```'
        
        result = processor._extract_json_from_response(response)
        
        assert result == {"invoice_number": "12345"}
    
    def test_extract_json_with_generic_markdown_block(self):
        """Test extraction from generic markdown code block"""
        processor = InvoiceProcessor()
        response = '```\n{"invoice_number": "12345"}\n```'
        
        result = processor._extract_json_from_response(response)
        
        assert result == {"invoice_number": "12345"}
    
    def test_extract_json_with_surrounding_text(self):
        """Test extraction with surrounding text in markdown"""
        processor = InvoiceProcessor()
        response = 'Here is the result:\n```json\n{"invoice_number": "12345"}\n```\nDone!'
        
        result = processor._extract_json_from_response(response)
        
        assert result == {"invoice_number": "12345"}
    
    def test_extract_invalid_json(self):
        """Test extraction with invalid JSON"""
        processor = InvoiceProcessor()
        response = '{"invalid": json}'
        
        result = processor._extract_json_from_response(response)
        
        assert result is None
    
    def test_extract_empty_string(self):
        """Test extraction from empty string"""
        processor = InvoiceProcessor()
        response = ''
        
        result = processor._extract_json_from_response(response)
        
        assert result is None
    
    def test_extract_complex_nested_json(self):
        """Test extraction of complex nested JSON"""
        processor = InvoiceProcessor()
        complex_json = {
            "invoice_number": "12345",
            "seller": {"name": "Test", "address": {"city": "NYC"}},
            "items": [{"id": 1}, {"id": 2}]
        }
        response = json.dumps(complex_json)
        
        result = processor._extract_json_from_response(response)
        
        assert result == complex_json


@pytest.mark.phase2
class TestExtractStructuredData:
    """Test suite for extract_structured_data method"""
    
    @patch('invoice_processor.Llama')
    def test_extract_without_loaded_model(self, mock_llama):
        """Test extraction without loading model first"""
        processor = InvoiceProcessor()
        result = processor.extract_structured_data("test text")
        
        assert result is None
    
    def test_extract_without_ocr_text(self):
        """Test extraction without OCR text"""
        processor = InvoiceProcessor()
        processor.llm = MagicMock()
        
        result = processor.extract_structured_data()
        
        assert result is None
    
    @patch('invoice_processor.Llama')
    def test_extract_success_first_attempt(self, mock_llama, tmp_path):
        """Test successful extraction on first attempt"""
        # Setup schema
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps({"invoice_number": ""}))
        
        # Setup mock LLM
        mock_llm_instance = MagicMock()
        valid_response = {"invoice_number": "12345"}
        mock_llm_instance.return_value = {
            "choices": [{"text": json.dumps(valid_response)}]
        }
        
        processor = InvoiceProcessor(schema_path=str(schema_file))
        processor.llm = mock_llm_instance
        processor.ocr_text = "INVOICE #12345"
        
        result = processor.extract_structured_data()
        
        assert result is not None
        assert result["invoice_number"] == "12345"
        assert processor.structured_data == result
        assert len(processor.validation_errors) == 0
    
    @patch('invoice_processor.Llama')
    def test_extract_with_retry_on_invalid_json(self, mock_llama, tmp_path):
        """Test retry logic when LLM returns invalid JSON"""
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps({"invoice_number": ""}))
        
        mock_llm_instance = MagicMock()
        # First attempt: invalid JSON, second attempt: valid JSON
        mock_llm_instance.side_effect = [
            {"choices": [{"text": "invalid json {"}]},
            {"choices": [{"text": '{"invoice_number": "12345"}'}]}
        ]
        
        processor = InvoiceProcessor(schema_path=str(schema_file), max_retries=2)
        processor.llm = mock_llm_instance
        processor.ocr_text = "INVOICE #12345"
        
        result = processor.extract_structured_data()
        
        assert result is not None
        assert result["invoice_number"] == "12345"
        assert mock_llm_instance.call_count == 2
    
    @patch('invoice_processor.Llama')
    def test_extract_max_retries_exceeded(self, mock_llama, tmp_path):
        """Test behavior when max retries exceeded"""
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps({"invoice_number": ""}))
        
        mock_llm_instance = MagicMock()
        # Always return invalid JSON
        mock_llm_instance.return_value = {"choices": [{"text": "invalid json"}]}
        
        processor = InvoiceProcessor(schema_path=str(schema_file), max_retries=2)
        processor.llm = mock_llm_instance
        processor.ocr_text = "INVOICE #12345"
        
        result = processor.extract_structured_data()
        
        assert result is None
        assert mock_llm_instance.call_count == 2
    
    @patch('invoice_processor.Llama')
    def test_extract_with_validation_retry(self, mock_llama, tmp_path):
        """Test retry with validation feedback"""
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps({"invoice_number": ""}))
        
        mock_llm_instance = MagicMock()
        # First: invalid data type, second: valid
        mock_llm_instance.side_effect = [
            {"choices": [{"text": '{"invoice_number": 12345}'}]},  # Number instead of string
            {"choices": [{"text": '{"invoice_number": "12345"}'}]}
        ]
        
        processor = InvoiceProcessor(schema_path=str(schema_file), max_retries=2)
        processor.llm = mock_llm_instance
        processor.ocr_text = "INVOICE #12345"
        
        result = processor.extract_structured_data()
        
        assert result is not None
        assert result["invoice_number"] == "12345"
        assert len(processor.validation_errors) == 1
    
    @patch('invoice_processor.Llama')
    def test_extract_returns_data_on_last_retry_despite_validation_error(self, mock_llama, tmp_path):
        """Test that data is returned on last retry even with validation errors"""
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps({"invoice_number": ""}))
        
        mock_llm_instance = MagicMock()
        # Always return invalid type
        mock_llm_instance.return_value = {"choices": [{"text": '{"invoice_number": 12345}'}]}
        
        processor = InvoiceProcessor(schema_path=str(schema_file), max_retries=2)
        processor.llm = mock_llm_instance
        processor.ocr_text = "INVOICE #12345"
        
        result = processor.extract_structured_data()
        
        # Should return data despite validation failure
        assert result is not None
        assert result["invoice_number"] == 12345
        assert len(processor.validation_errors) == 2
    
    @patch('invoice_processor.Llama')
    def test_extract_with_custom_temperature(self, mock_llama, tmp_path):
        """Test extraction with custom temperature"""
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps({"invoice_number": ""}))
        
        mock_llm_instance = MagicMock()
        mock_llm_instance.return_value = {
            "choices": [{"text": '{"invoice_number": "12345"}'}]
        }
        
        processor = InvoiceProcessor(schema_path=str(schema_file))
        processor.llm = mock_llm_instance
        processor.ocr_text = "INVOICE #12345"
        
        processor.extract_structured_data(temperature=0.5)
        
        # Check that temperature was passed to LLM
        call_args = mock_llm_instance.call_args
        assert call_args[1]['temperature'] == 0.5
    
    @patch('invoice_processor.Llama')
    def test_extract_with_exception_handling(self, mock_llama, tmp_path):
        """Test exception handling during extraction"""
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps({"invoice_number": ""}))
        
        mock_llm_instance = MagicMock()
        mock_llm_instance.side_effect = Exception("LLM error")
        
        processor = InvoiceProcessor(schema_path=str(schema_file), max_retries=2)
        processor.llm = mock_llm_instance
        processor.ocr_text = "INVOICE #12345"
        
        result = processor.extract_structured_data()
        
        assert result is None
        assert mock_llm_instance.call_count == 2


@pytest.mark.phase2
class TestProcessInvoice:
    """Test suite for process_invoice method (complete pipeline)"""
    
    @patch('invoice_processor.OCRExtractor')
    @patch('invoice_processor.Llama')
    def test_process_invoice_complete_success(self, mock_llama, mock_ocr_class, tmp_path):
        """Test complete invoice processing pipeline"""
        # Setup
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps({"invoice_number": ""}))
        
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"dummy model")
        
        img_path = tmp_path / "invoice.jpg"
        Image.new('RGB', (100, 100), color='white').save(img_path)
        
        # Mock OCR
        mock_extractor = MagicMock()
        mock_extractor.process.return_value = "INVOICE #12345"
        mock_ocr_class.return_value = mock_extractor
        
        # Mock LLM
        mock_llm_instance = MagicMock()
        mock_llm_instance.return_value = {
            "choices": [{"text": '{"invoice_number": "12345"}'}]
        }
        mock_llama.return_value = mock_llm_instance
        
        processor = InvoiceProcessor(
            model_path=str(model_file),
            schema_path=str(schema_file)
        )
        
        result = processor.process_invoice(str(img_path))
        
        assert result is not None
        assert result["invoice_number"] == "12345"
        assert processor.ocr_text == "INVOICE #12345"
        assert processor.structured_data == result
    
    @patch('invoice_processor.OCRExtractor')
    def test_process_invoice_ocr_failure(self, mock_ocr_class):
        """Test processing when OCR fails"""
        mock_extractor = MagicMock()
        mock_extractor.process.return_value = None
        mock_ocr_class.return_value = mock_extractor
        
        processor = InvoiceProcessor()
        result = processor.process_invoice("test.jpg")
        
        assert result is None
    
    @patch('invoice_processor.OCRExtractor')
    @patch('invoice_processor.Llama')
    def test_process_invoice_loads_model_if_needed(self, mock_llama, mock_ocr_class, tmp_path):
        """Test that process_invoice loads model if not already loaded"""
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps({"invoice_number": ""}))
        
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"dummy model")
        
        # Mock OCR
        mock_extractor = MagicMock()
        mock_extractor.process.return_value = "INVOICE #12345"
        mock_ocr_class.return_value = mock_extractor
        
        # Mock LLM
        mock_llm_instance = MagicMock()
        mock_llm_instance.return_value = {
            "choices": [{"text": '{"invoice_number": "12345"}'}]
        }
        mock_llama.return_value = mock_llm_instance
        
        processor = InvoiceProcessor(
            model_path=str(model_file),
            schema_path=str(schema_file)
        )
        
        # Model should not be loaded yet
        assert processor.llm is None
        
        result = processor.process_invoice("test.jpg")
        
        # Model should be loaded now
        assert processor.llm is not None
        assert result is not None
    
    @patch('invoice_processor.OCRExtractor')
    def test_process_invoice_model_load_failure(self, mock_ocr_class):
        """Test processing when model loading fails"""
        mock_extractor = MagicMock()
        mock_extractor.process.return_value = "INVOICE #12345"
        mock_ocr_class.return_value = mock_extractor
        
        processor = InvoiceProcessor(model_path="./nonexistent_model.gguf")
        result = processor.process_invoice("test.jpg")
        
        assert result is None
    
    @patch('invoice_processor.OCRExtractor')
    @patch('invoice_processor.Llama')
    def test_process_invoice_with_custom_temperature(self, mock_llama, mock_ocr_class, tmp_path):
        """Test processing with custom temperature"""
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps({"invoice_number": ""}))
        
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"dummy model")
        
        mock_extractor = MagicMock()
        mock_extractor.process.return_value = "INVOICE #12345"
        mock_ocr_class.return_value = mock_extractor
        
        mock_llm_instance = MagicMock()
        mock_llm_instance.return_value = {
            "choices": [{"text": '{"invoice_number": "12345"}'}]
        }
        mock_llama.return_value = mock_llm_instance
        
        processor = InvoiceProcessor(
            model_path=str(model_file),
            schema_path=str(schema_file)
        )
        
        result = processor.process_invoice("test.jpg", temperature=0.7)
        
        # Verify temperature was used
        call_args = mock_llm_instance.call_args
        assert call_args[1]['temperature'] == 0.7


@pytest.mark.phase2
class TestSaveResults:
    """Test suite for save_results method"""
    
    def test_save_results_success(self, tmp_path):
        """Test successful results saving"""
        output_file = tmp_path / "output.json"
        
        processor = InvoiceProcessor()
        processor.structured_data = {"invoice_number": "12345"}
        processor.ocr_text = "INVOICE #12345"
        
        result = processor.save_results(str(output_file))
        
        assert result is True
        assert output_file.exists()
        
        with open(output_file, 'r') as f:
            saved_data = json.load(f)
        
        assert "structured_data" in saved_data
        assert saved_data["structured_data"]["invoice_number"] == "12345"
        assert "raw_ocr_text" in saved_data
        assert saved_data["raw_ocr_text"] == "INVOICE #12345"
    
    def test_save_results_without_ocr_text(self, tmp_path):
        """Test saving without OCR text"""
        output_file = tmp_path / "output.json"
        
        processor = InvoiceProcessor()
        processor.structured_data = {"invoice_number": "12345"}
        
        result = processor.save_results(str(output_file), include_ocr_text=False)
        
        assert result is True
        
        with open(output_file, 'r') as f:
            saved_data = json.load(f)
        
        assert "structured_data" in saved_data
        assert "raw_ocr_text" not in saved_data
    
    def test_save_results_no_structured_data(self, tmp_path):
        """Test saving when no structured data exists"""
        output_file = tmp_path / "output.json"
        
        processor = InvoiceProcessor()
        result = processor.save_results(str(output_file))
        
        assert result is False
        assert not output_file.exists()
    
    def test_save_results_invalid_path(self):
        """Test saving to invalid path"""
        processor = InvoiceProcessor()
        processor.structured_data = {"invoice_number": "12345"}
        
        result = processor.save_results("/invalid/path/output.json")
        
        assert result is False
    
    def test_save_results_complex_data(self, tmp_path):
        """Test saving complex nested data"""
        output_file = tmp_path / "output.json"
        
        processor = InvoiceProcessor()
        processor.structured_data = {
            "invoice_number": "12345",
            "seller": {
                "name": "Test Seller",
                "address": {"city": "NYC"}
            },
            "items": [
                {"id": 1, "amount": 100.0},
                {"id": 2, "amount": 200.0}
            ]
        }
        
        result = processor.save_results(str(output_file), include_ocr_text=False)
        
        assert result is True
        
        with open(output_file, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data["structured_data"]["seller"]["name"] == "Test Seller"
        assert len(saved_data["structured_data"]["items"]) == 2
    
    def test_save_results_unicode_content(self, tmp_path):
        """Test saving with unicode characters"""
        output_file = tmp_path / "output.json"
        
        processor = InvoiceProcessor()
        processor.structured_data = {
            "invoice_number": "12345",
            "seller": {"name": "Société Française"}
        }
        processor.ocr_text = "Société Française\n€100.00"
        
        result = processor.save_results(str(output_file))
        
        assert result is True
        
        with open(output_file, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        assert "Société Française" in saved_data["structured_data"]["seller"]["name"]
        assert "€" in saved_data["raw_ocr_text"]


@pytest.mark.phase2
class TestInvoiceProcessorStateManagement:
    """Test suite for state management across operations"""
    
    def test_initial_state(self):
        """Test initial state of processor"""
        processor = InvoiceProcessor()
        
        assert processor.llm is None
        assert processor.ocr_text is None
        assert processor.structured_data is None
        assert processor.validation_errors == []
    
    @patch('invoice_processor.OCRExtractor')
    def test_state_after_ocr(self, mock_ocr_class):
        """Test state after OCR extraction"""
        mock_extractor = MagicMock()
        mock_extractor.process.return_value = "INVOICE TEXT"
        mock_ocr_class.return_value = mock_extractor
        
        processor = InvoiceProcessor()
        processor.extract_text_from_image("test.jpg")
        
        assert processor.ocr_text == "INVOICE TEXT"
        assert processor.structured_data is None
    
    @patch('invoice_processor.Llama')
    def test_state_after_model_load(self, mock_llama, tmp_path):
        """Test state after model loading"""
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"dummy")
        
        mock_llm_instance = MagicMock()
        mock_llama.return_value = mock_llm_instance
        
        processor = InvoiceProcessor(model_path=str(model_file))
        processor.load_model()
        
        assert processor.llm is not None
        assert processor.ocr_text is None
        assert processor.structured_data is None


@pytest.mark.phase2
@pytest.mark.parametrize("max_retries,expected_attempts", [
    (1, 1),
    (3, 3),
    (5, 5),
])
def test_retry_configuration(max_retries, expected_attempts, tmp_path):
    """Test that max_retries configuration works correctly"""
    schema_file = tmp_path / "schema.json"
    schema_file.write_text(json.dumps({"invoice_number": ""}))
    
    mock_llm_instance = MagicMock()
    # Always return invalid JSON to trigger retries
    mock_llm_instance.return_value = {"choices": [{"text": "invalid json"}]}
    
    processor = InvoiceProcessor(
        schema_path=str(schema_file),
        max_retries=max_retries
    )
    processor.llm = mock_llm_instance
    processor.ocr_text = "INVOICE #12345"
    
    result = processor.extract_structured_data()
    
    assert result is None
    assert mock_llm_instance.call_count == expected_attempts


@pytest.mark.phase2
class TestEdgeCases:
    """Test suite for edge cases and error conditions"""
    
    def test_empty_ocr_text(self):
        """Test processing with empty OCR text"""
        processor = InvoiceProcessor()
        processor.llm = MagicMock()
        processor.ocr_text = ""
        
        result = processor.extract_structured_data()
        
        assert result is None
    
    def test_special_characters_in_data(self, tmp_path):
        """Test handling of special characters"""
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps({"invoice_number": ""}))
        
        mock_llm_instance = MagicMock()
        mock_llm_instance.return_value = {
            "choices": [{"text": '{"invoice_number": "INV-2024/01\\n\\t@#$"}'}]
        }
        
        processor = InvoiceProcessor(schema_path=str(schema_file))
        processor.llm = mock_llm_instance
        processor.ocr_text = "Special chars: @#$%"
        
        result = processor.extract_structured_data()
        
        assert result is not None
        assert "INV-2024/01" in result["invoice_number"]
    
    def test_null_values_in_response(self, tmp_path):
        """Test handling of null values in LLM response"""
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps({
            "invoice_number": "",
            "issue_date": "",
            "seller": {"name": ""}
        }))
        
        mock_llm_instance = MagicMock()
        mock_llm_instance.return_value = {
            "choices": [{"text": json.dumps({
                "invoice_number": "12345",
                "issue_date": None,
                "seller": None
            })}]
        }
        
        processor = InvoiceProcessor(schema_path=str(schema_file))
        processor.llm = mock_llm_instance
        processor.ocr_text = "INVOICE #12345"
        
        result = processor.extract_structured_data()
        
        assert result is not None
        assert result["invoice_number"] == "12345"
        assert result["issue_date"] is None