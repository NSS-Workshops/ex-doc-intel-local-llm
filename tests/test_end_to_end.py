"""
End-to-end tests for complete invoice processing pipeline
No mocking - tests the full system with real invoices and LLM
"""

import pytest
import json
import sys
import os
from pathlib import Path

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from invoice_processor import InvoiceProcessor


@pytest.mark.e2e
class TestEndToEndProcessing:
    """End-to-end tests with real invoices and LLM processing"""
    
    @pytest.fixture
    def processor(self):
        """Create processor instance with default model"""
        model_path = "./models/Phi-3.5-mini-instruct-Q4_K_M.gguf"
        if not Path(model_path).exists():
            pytest.skip(f"Model file not found: {model_path}")
        
        processor = InvoiceProcessor(
            model_path=model_path,
            schema_path="./schema.json",
            max_retries=3
        )
        
        # Load model once for all tests
        if not processor.load_model():
            pytest.skip("Failed to load model")
        
        return processor
    
    def test_batch1_invoice_complete_pipeline(self, processor, tmp_path):
        """Test complete pipeline with batch 1 invoice"""
        invoice_path = "data/batch_1/batch_1/batch1_3/batch1-1044.jpg"
        
        if not Path(invoice_path).exists():
            pytest.skip(f"Invoice file not found: {invoice_path}")
        
        # Process invoice
        result = processor.process_invoice(invoice_path)
        
        # Assert result structure
        assert result is not None, "Processing should return structured data"
        assert isinstance(result, dict), "Result should be a dictionary"
        
        # Assert expected fields exist
        assert "invoice_number" in result or result.get("invoice_number") is not None
        assert "issue_date" in result or result.get("issue_date") is not None
        assert "currency" in result or result.get("currency") is not None
        
        # Assert seller information
        if result.get("seller"):
            assert isinstance(result["seller"], dict)
            assert "name" in result["seller"] or result["seller"].get("name") is not None
        
        # Assert client information
        if result.get("client"):
            assert isinstance(result["client"], dict)
            assert "name" in result["client"] or result["client"].get("name") is not None
        
        # Assert items structure
        if result.get("items"):
            assert isinstance(result["items"], list)
            if len(result["items"]) > 0:
                item = result["items"][0]
                assert isinstance(item, dict)
                # Check for common item fields
                assert any(key in item for key in ["description", "quantity", "net_amount"])
        
        # Assert summary information
        if result.get("summary"):
            assert isinstance(result["summary"], dict)
            # Check for numeric fields
            if result["summary"].get("gross_total") is not None:
                assert isinstance(result["summary"]["gross_total"], (int, float))
        
        # Save and verify output
        output_file = tmp_path / "batch1_output.json"
        save_success = processor.save_results(str(output_file))
        assert save_success is True, "Should successfully save results"
        assert output_file.exists(), "Output file should exist"
        
        # Verify saved file structure
        with open(output_file, 'r') as f:
            saved_data = json.load(f)
        
        assert "structured_data" in saved_data
        assert "raw_ocr_text" in saved_data
        assert len(saved_data["raw_ocr_text"]) > 0, "OCR text should not be empty"
    
    def test_batch2_invoice_complete_pipeline(self, processor, tmp_path):
        """Test complete pipeline with batch 2 invoice"""
        invoice_path = "data/batch_2/batch_2/batch2_2/batch2-0644.jpg"
        
        if not Path(invoice_path).exists():
            pytest.skip(f"Invoice file not found: {invoice_path}")
        
        # Process invoice
        result = processor.process_invoice(invoice_path)
        
        # Assert result structure
        assert result is not None, "Processing should return structured data"
        assert isinstance(result, dict), "Result should be a dictionary"
        
        # Assert expected fields exist
        assert "invoice_number" in result or result.get("invoice_number") is not None
        assert "issue_date" in result or result.get("issue_date") is not None
        
        # Assert seller information exists
        if result.get("seller"):
            assert isinstance(result["seller"], dict)
        
        # Assert client information exists
        if result.get("client"):
            assert isinstance(result["client"], dict)
        
        # Assert items is a list
        if result.get("items"):
            assert isinstance(result["items"], list)
        
        # Save and verify output
        output_file = tmp_path / "batch2_output.json"
        save_success = processor.save_results(str(output_file))
        assert save_success is True
        assert output_file.exists()
        
        # Verify saved file has required structure
        with open(output_file, 'r') as f:
            saved_data = json.load(f)
        
        assert "structured_data" in saved_data
        assert "raw_ocr_text" in saved_data
    
    def test_batch3_invoice_complete_pipeline(self, processor, tmp_path):
        """Test complete pipeline with batch 3 invoice"""
        invoice_path = "data/batch_3/batch_3/batch3_1/batch3-0008.jpg"
        
        if not Path(invoice_path).exists():
            pytest.skip(f"Invoice file not found: {invoice_path}")
        
        # Process invoice
        result = processor.process_invoice(invoice_path)
        
        # Assert result structure
        assert result is not None, "Processing should return structured data"
        assert isinstance(result, dict), "Result should be a dictionary"
        
        # Assert key fields are present (even if None)
        expected_top_level_fields = ["invoice_number", "issue_date", "currency", "seller", "client", "items", "summary"]
        present_fields = [field for field in expected_top_level_fields if field in result]
        assert len(present_fields) > 0, "At least some expected fields should be present"
        
        # If seller exists, verify structure
        if result.get("seller") and result["seller"] is not None:
            assert isinstance(result["seller"], dict)
            if result["seller"].get("address"):
                assert isinstance(result["seller"]["address"], dict)
        
        # If client exists, verify structure
        if result.get("client") and result["client"] is not None:
            assert isinstance(result["client"], dict)
            if result["client"].get("address"):
                assert isinstance(result["client"]["address"], dict)
        
        # If items exist, verify structure
        if result.get("items") and result["items"] is not None:
            assert isinstance(result["items"], list)
            for item in result["items"]:
                assert isinstance(item, dict)
        
        # If summary exists, verify structure
        if result.get("summary") and result["summary"] is not None:
            assert isinstance(result["summary"], dict)
            # Verify numeric fields are numbers if present
            for field in ["net_total", "vat_total", "gross_total", "vat_percent"]:
                if result["summary"].get(field) is not None:
                    assert isinstance(result["summary"][field], (int, float)), f"{field} should be numeric"
        
        # Save and verify output
        output_file = tmp_path / "batch3_output.json"
        save_success = processor.save_results(str(output_file))
        assert save_success is True
        assert output_file.exists()
        
        # Verify saved file structure
        with open(output_file, 'r') as f:
            saved_data = json.load(f)
        
        assert "structured_data" in saved_data
        assert "raw_ocr_text" in saved_data
        assert isinstance(saved_data["structured_data"], dict)
        assert isinstance(saved_data["raw_ocr_text"], str)
    
    def test_ocr_text_extraction_quality(self, processor):
        """Test that OCR extracts meaningful text from invoices"""
        invoice_paths = [
            "data/batch_1/batch_1/batch1_3/batch1-1044.jpg",
            "data/batch_2/batch_2/batch2_2/batch2-0644.jpg",
            "data/batch_3/batch_3/batch3_1/batch3-0008.jpg",
        ]
        
        for invoice_path in invoice_paths:
            if Path(invoice_path).exists():
                ocr_text = processor.extract_text_from_image(invoice_path)
                
                assert ocr_text is not None, f"OCR should extract text from {invoice_path}"
                assert len(ocr_text) > 50, f"OCR text should be substantial for {invoice_path}"
                assert isinstance(ocr_text, str), "OCR text should be a string"
                
                # Check for common invoice keywords
                text_lower = ocr_text.lower()
                has_invoice_keywords = any(keyword in text_lower for keyword in 
                    ["invoice", "total", "date", "amount", "tax", "vat", "seller", "buyer", "client"])
                
                assert has_invoice_keywords, f"OCR text should contain invoice-related keywords for {invoice_path}"
    
    def test_structured_data_consistency(self, processor):
        """Test that structured data is consistent across multiple runs"""
        invoice_path = "data/batch_1/batch_1/batch1_3/batch1-1044.jpg"
        
        if not Path(invoice_path).exists():
            pytest.skip(f"Invoice file not found: {invoice_path}")
        
        # Process same invoice twice
        result1 = processor.process_invoice(invoice_path, temperature=0.0)
        result2 = processor.process_invoice(invoice_path, temperature=0.0)
        
        # With temperature=0, results should be identical or very similar
        assert result1 is not None and result2 is not None
        
        # Check that key fields match
        if result1.get("invoice_number") and result2.get("invoice_number"):
            assert result1["invoice_number"] == result2["invoice_number"], \
                "Invoice number should be consistent across runs"
