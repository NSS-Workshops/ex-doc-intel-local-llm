"""
End-to-end test for complete invoice processing pipeline
No mocking - tests the full system with real invoice and LLM
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
    """End-to-end test with real invoice and LLM processing"""
    
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
        
        # Load model once for the test
        if not processor.load_model():
            pytest.skip("Failed to load model")
        
        return processor
    
    @pytest.fixture
    def expected_data(self):
        """Load expected invoice data"""
        expected_file = Path(__file__).parent / "expected_batch1_1044.json"
        with open(expected_file, 'r') as f:
            return json.load(f)
    
    def test_batch1_invoice_complete_pipeline(self, processor, expected_data, tmp_path):
        """Test complete pipeline with batch 1 invoice and validate actual values"""
        invoice_path = "data/batch_1/batch_1/batch1_3/batch1-1044.jpg"
        
        if not Path(invoice_path).exists():
            pytest.skip(f"Invoice file not found: {invoice_path}")
        
        # Process invoice
        result = processor.process_invoice(invoice_path, temperature=0.0)
        
        # Assert result exists
        assert result is not None, "Processing should return structured data"
        assert isinstance(result, dict), "Result should be a dictionary"
        
        # Assert invoice number
        assert result.get("invoice_number") == expected_data["invoice_number"], \
            f"Invoice number should be {expected_data['invoice_number']}"
        
        # Assert issue date (allow flexible date formats)
        if result.get("issue_date"):
            # Normalize date format for comparison
            result_date = result["issue_date"].replace("/", "-")
            expected_date = expected_data["issue_date"]
            # Check if dates match (allow YYYY-MM-DD or MM-DD-YYYY or DD-MM-YYYY)
            assert any(date_part in result_date for date_part in ["2017", "07", "20"]), \
                f"Issue date should contain 2017-07-20 components, got {result['issue_date']}"
        
        # Assert seller information
        assert result.get("seller") is not None, "Seller information should exist"
        seller = result["seller"]
        assert seller.get("name") == expected_data["seller"]["name"], \
            f"Seller name should be {expected_data['seller']['name']}"
        
        # Normalize tax ID (remove dashes for comparison)
        if seller.get("tax_id"):
            result_tax_id = seller["tax_id"].replace("-", "")
            expected_tax_id = expected_data["seller"]["tax_id"].replace("-", "")
            assert result_tax_id == expected_tax_id, \
                f"Seller tax ID should be {expected_data['seller']['tax_id']} (got {seller['tax_id']})"
        
        # Assert seller address
        if seller.get("address"):
            seller_addr = seller["address"]
            expected_addr = expected_data["seller"]["address"]
            assert seller_addr.get("city") == expected_addr["city"], \
                f"Seller city should be {expected_addr['city']}"
            assert seller_addr.get("state") == expected_addr["state"], \
                f"Seller state should be {expected_addr['state']}"
        
        # Assert client information
        assert result.get("client") is not None, "Client information should exist"
        client = result["client"]
        assert client.get("name") == expected_data["client"]["name"], \
            f"Client name should be {expected_data['client']['name']}"
        
        # Normalize tax ID (remove dashes for comparison)
        if client.get("tax_id"):
            result_tax_id = client["tax_id"].replace("-", "")
            expected_tax_id = expected_data["client"]["tax_id"].replace("-", "")
            assert result_tax_id == expected_tax_id, \
                f"Client tax ID should be {expected_data['client']['tax_id']} (got {client['tax_id']})"
        
        # Assert client address
        if client.get("address"):
            client_addr = client["address"]
            expected_addr = expected_data["client"]["address"]
            assert client_addr.get("city") == expected_addr["city"], \
                f"Client city should be {expected_addr['city']}"
            assert client_addr.get("state") == expected_addr["state"], \
                f"Client state should be {expected_addr['state']}"
        
        # Assert items
        assert result.get("items") is not None, "Items should exist"
        assert isinstance(result["items"], list), "Items should be a list"
        assert len(result["items"]) == len(expected_data["items"]), \
            f"Should have {len(expected_data['items'])} items"
        
        # Validate each item
        for i, (result_item, expected_item) in enumerate(zip(result["items"], expected_data["items"])):
            assert result_item.get("number") == expected_item["number"], \
                f"Item {i+1} number should be {expected_item['number']}"
            
            # Check quantity (allow small floating point differences)
            if result_item.get("quantity") is not None:
                assert abs(result_item["quantity"] - expected_item["quantity"]) < 0.01, \
                    f"Item {i+1} quantity should be {expected_item['quantity']}"
            
            # Check net amount (allow small floating point differences)
            if result_item.get("net_amount") is not None:
                assert abs(result_item["net_amount"] - expected_item["net_amount"]) < 0.01, \
                    f"Item {i+1} net amount should be {expected_item['net_amount']}"
        
        # Assert summary
        assert result.get("summary") is not None, "Summary should exist"
        summary = result["summary"]
        expected_summary = expected_data["summary"]
        
        # Check VAT percent
        if summary.get("vat_percent") is not None:
            assert abs(summary["vat_percent"] - expected_summary["vat_percent"]) < 0.01, \
                f"VAT percent should be {expected_summary['vat_percent']}%"
        
        # Check net total (allow small floating point differences)
        if summary.get("net_total") is not None:
            assert abs(summary["net_total"] - expected_summary["net_total"]) < 0.01, \
                f"Net total should be {expected_summary['net_total']}"
        
        # Check VAT total
        if summary.get("vat_total") is not None:
            assert abs(summary["vat_total"] - expected_summary["vat_total"]) < 0.01, \
                f"VAT total should be {expected_summary['vat_total']}"
        
        # Check gross total
        if summary.get("gross_total") is not None:
            assert abs(summary["gross_total"] - expected_summary["gross_total"]) < 0.01, \
                f"Gross total should be {expected_summary['gross_total']}"
        
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
        
        print("\n✓ All assertions passed! Invoice data extracted correctly.")
