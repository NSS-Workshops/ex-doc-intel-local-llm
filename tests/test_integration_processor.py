"""
Integration tests for Invoice Processor
Tests with complete end-to-end workflows
"""

import pytest
import json
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ocr_extractor_class import OCRExtractor
from invoice_processor import InvoiceProcessor


@pytest.mark.integration
class TestInvoiceProcessorIntegration:
    """Integration tests for complete invoice processing"""
    
    @patch('invoice_processor.Llama')
    @patch('invoice_processor.OCRExtractor')
    def test_end_to_end_processing(self, mock_ocr_class, mock_llama, tmp_path):
        """Test complete end-to-end invoice processing"""
        # Setup files
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps({
            "invoice_number": "",
            "issue_date": "",
            "seller": {"name": ""},
            "items": []
        }))
        
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"dummy model")
        
        output_file = tmp_path / "output.json"
        
        # Mock OCR
        mock_extractor = MagicMock()
        mock_extractor.process.return_value = "INVOICE #12345\nDate: 2024-01-01\nSeller: Test Corp"
        mock_ocr_class.return_value = mock_extractor
        
        # Mock LLM
        mock_llm_instance = MagicMock()
        mock_llm_instance.return_value = {
            "choices": [{"text": json.dumps({
                "invoice_number": "12345",
                "issue_date": "2024-01-01",
                "seller": {"name": "Test Corp"},
                "items": []
            })}]
        }
        mock_llama.return_value = mock_llm_instance
        
        # Process
        processor = InvoiceProcessor(
            model_path=str(model_file),
            schema_path=str(schema_file)
        )
        
        result = processor.process_invoice("test.jpg")
        assert result is not None
        
        save_result = processor.save_results(str(output_file))
        assert save_result is True
        
        # Verify saved file
        with open(output_file, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data["structured_data"]["invoice_number"] == "12345"
        assert saved_data["structured_data"]["seller"]["name"] == "Test Corp"
    
    @patch('invoice_processor.OCRExtractor')
    def test_processing_with_real_invoice_images(self, mock_ocr_class):
        """Test with real invoice images if available"""
        invoice_path = "data/batch_1/batch_1/batch1_3/batch1-1044.jpg"
        
        if not Path(invoice_path).exists():
            pytest.skip(f"Invoice file not found: {invoice_path}")
        
        # Use real OCR, don't mock it
        from ocr_extractor_class import OCRExtractor
        processor = InvoiceProcessor()
        
        # Test OCR extraction only (no LLM needed for this test)
        extractor = OCRExtractor(invoice_path)
        ocr_result = extractor.process()
        
        assert ocr_result is not None
        assert isinstance(ocr_result, str)
        assert len(ocr_result) > 0
