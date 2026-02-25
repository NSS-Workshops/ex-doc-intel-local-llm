"""
Integration tests for OCR Extractor
Tests with real invoice images
"""

import pytest
import sys
import os
from pathlib import Path
from PIL import Image

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ocr_extractor_class import OCRExtractor


@pytest.mark.integration
class TestOCRExtractorIntegration:
    """Integration tests with real invoice images"""
    
    def test_process_with_real_invoice_batch1(self):
        """Test with actual invoice from batch_1"""
        invoice_path = "data/batch_1/batch_1/batch1_3/batch1-1044.jpg"
        if Path(invoice_path).exists():
            extractor = OCRExtractor(invoice_path)
            result = extractor.process()
            assert result is not None
            assert isinstance(result, str)
            assert len(result) > 0
        else:
            pytest.skip(f"Invoice file not found: {invoice_path}")
    
    def test_process_with_real_invoice_batch2(self):
        """Test with actual invoice from batch_2"""
        invoice_path = "data/batch_2/batch_2/batch2_2/batch2-0644.jpg"
        if Path(invoice_path).exists():
            extractor = OCRExtractor(invoice_path)
            result = extractor.process()
            assert result is not None
            assert isinstance(result, str)
            assert len(result) > 0
        else:
            pytest.skip(f"Invoice file not found: {invoice_path}")
    
    def test_process_with_real_invoice_batch3(self):
        """Test with actual invoice from batch_3"""
        invoice_path = "data/batch_3/batch_3/batch3_1/batch3-0008.jpg"
        if Path(invoice_path).exists():
            extractor = OCRExtractor(invoice_path)
            result = extractor.process()
            assert result is not None
            assert isinstance(result, str)
            assert len(result) > 0
        else:
            pytest.skip(f"Invoice file not found: {invoice_path}")
    
    def test_multiple_invoices_batch_processing(self):
        """Test processing multiple invoices"""
        invoice_paths = [
            "data/batch_1/batch_1/batch1_3/batch1-1044.jpg",
            "data/batch_2/batch_2/batch2_2/batch2-0644.jpg",
            "data/batch_3/batch_3/batch3_1/batch3-0008.jpg",
        ]
        
        results = []
        for path in invoice_paths:
            if Path(path).exists():
                extractor = OCRExtractor(path)
                result = extractor.process()
                results.append(result)
        
        if results:
            assert all(r is not None for r in results)
            assert all(isinstance(r, str) for r in results)
            assert all(len(r) > 0 for r in results)
        else:
            pytest.skip("No invoice files found for batch processing test")


@pytest.mark.integration
class TestOCRPerformance:
    """Performance tests for OCR extraction"""
    
    def test_ocr_performance_standard_image(self, tmp_path):
        """Test OCR processing time for standard size image"""
        import time
        
        img_path = tmp_path / "perf_test.png"
        img = Image.new('RGB', (800, 600), color='white')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), "Performance Test Image", fill='black')
        img.save(img_path)
        
        start = time.time()
        extractor = OCRExtractor(str(img_path))
        extractor.process()
        duration = time.time() - start
        
        # Should complete within reasonable time (5 seconds)
        assert duration < 5.0
    
    def test_ocr_performance_large_image(self, tmp_path):
        """Test OCR processing time for large image"""
        import time
        
        img_path = tmp_path / "large_perf_test.png"
        img = Image.new('RGB', (2000, 1500), color='white')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.text((100, 100), "Large Performance Test", fill='black')
        img.save(img_path)
        
        start = time.time()
        extractor = OCRExtractor(str(img_path))
        extractor.process()
        duration = time.time() - start
        
        # Large images may take longer but should still be reasonable
        assert duration < 10.0
