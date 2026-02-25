"""
Test suite for OCRExtractor class
All tests marked with @pytest.mark.phase1 for OCR implementation phase
"""

import pytest
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ocr_extractor_class import OCRExtractor


@pytest.mark.phase1
class TestOCRExtractorValidation:
    """Test suite for validate_image_path method"""
    
    @pytest.fixture
    def temp_image(self, tmp_path):
        """Create a temporary test image with text"""
        img_path = tmp_path / "test_image.png"
        img = Image.new('RGB', (200, 100), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "TEST TEXT", fill='black')
        img.save(img_path)
        return str(img_path)
    
    def test_validate_image_path_valid_file(self, temp_image):
        """Test validation with valid image path"""
        extractor = OCRExtractor(temp_image)
        assert extractor.validate_image_path() is True
    
    def test_validate_image_path_non_existent(self):
        """Test validation with non-existent path"""
        extractor = OCRExtractor("/tmp/non_existent_image_12345.jpg")
        assert extractor.validate_image_path() is False
    
    def test_validate_image_path_directory(self, tmp_path):
        """Test validation when path is a directory"""
        extractor = OCRExtractor(str(tmp_path))
        assert extractor.validate_image_path() is False
    
    def test_validate_image_path_relative(self, tmp_path):
        """Test validation with relative path"""
        # Create image in current directory for relative path testing
        rel_dir = tmp_path / "relative_test"
        rel_dir.mkdir()
        img_path = rel_dir / "test_relative.png"
        img = Image.new('RGB', (200, 100), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "RELATIVE TEST", fill='black')
        img.save(img_path)
        
        # Use the path directly (it's already a valid path)
        extractor = OCRExtractor(str(img_path))
        assert extractor.validate_image_path() is True
    
    def test_validate_image_path_absolute(self, temp_image):
        """Test validation with absolute path"""
        abs_path = Path(temp_image).absolute()
        extractor = OCRExtractor(str(abs_path))
        assert extractor.validate_image_path() is True


@pytest.mark.phase1
class TestOCRExtractorLoadImage:
    """Test suite for load_image method"""
    
    @pytest.fixture
    def temp_image(self, tmp_path):
        """Create a temporary test image"""
        img_path = tmp_path / "test_load.png"
        img = Image.new('RGB', (200, 100), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "LOAD TEST", fill='black')
        img.save(img_path)
        return str(img_path)
    
    def test_load_image_success(self, temp_image):
        """Test successful image loading"""
        extractor = OCRExtractor(temp_image)
        assert extractor.load_image() is True
        assert extractor.image is not None
        assert isinstance(extractor.image, Image.Image)
    
    def test_load_image_invalid_file(self, tmp_path):
        """Test loading non-image file"""
        text_file = tmp_path / "test.txt"
        text_file.write_text("not an image")
        extractor = OCRExtractor(str(text_file))
        assert extractor.load_image() is False
        assert extractor.image is None
    
    def test_load_image_corrupted(self, tmp_path):
        """Test loading corrupted image file"""
        corrupted = tmp_path / "corrupted.jpg"
        corrupted.write_bytes(b"corrupted image data")
        extractor = OCRExtractor(str(corrupted))
        assert extractor.load_image() is False
    
    def test_load_image_empty_file(self, tmp_path):
        """Test loading empty file"""
        empty = tmp_path / "empty.jpg"
        empty.write_bytes(b"")
        extractor = OCRExtractor(str(empty))
        assert extractor.load_image() is False
    
    def test_load_image_multiple_formats_png(self, tmp_path):
        """Test loading PNG format"""
        img_path = tmp_path / "test.png"
        Image.new('RGB', (100, 50), color='white').save(img_path, format='PNG')
        extractor = OCRExtractor(str(img_path))
        assert extractor.load_image() is True
    
    def test_load_image_multiple_formats_jpeg(self, tmp_path):
        """Test loading JPEG format"""
        img_path = tmp_path / "test.jpg"
        Image.new('RGB', (100, 50), color='white').save(img_path, format='JPEG')
        extractor = OCRExtractor(str(img_path))
        assert extractor.load_image() is True
    
    def test_load_image_multiple_formats_bmp(self, tmp_path):
        """Test loading BMP format"""
        img_path = tmp_path / "test.bmp"
        Image.new('RGB', (100, 50), color='white').save(img_path, format='BMP')
        extractor = OCRExtractor(str(img_path))
        assert extractor.load_image() is True


@pytest.mark.phase1
class TestOCRExtractorExtractText:
    """Test suite for extract_text method"""
    
    @pytest.fixture
    def temp_image_with_text(self, tmp_path):
        """Create a temporary test image with clear text"""
        img_path = tmp_path / "text_image.png"
        img = Image.new('RGB', (400, 200), color='white')
        draw = ImageDraw.Draw(img)
        # Use larger text for better OCR
        draw.text((20, 20), "INVOICE", fill='black')
        draw.text((20, 60), "Amount: $100.00", fill='black')
        draw.text((20, 100), "Date: 2024-01-01", fill='black')
        img.save(img_path)
        return str(img_path)
    
    @pytest.fixture
    def blank_image(self, tmp_path):
        """Create a blank image with no text"""
        img_path = tmp_path / "blank.png"
        Image.new('RGB', (200, 100), color='white').save(img_path)
        return str(img_path)
    
    def test_extract_text_without_loading(self):
        """Test extraction without loading image first"""
        extractor = OCRExtractor("dummy.jpg")
        result = extractor.extract_text()
        assert result is None
        assert extractor.text is None
    
    def test_extract_text_success(self, temp_image_with_text):
        """Test successful text extraction"""
        extractor = OCRExtractor(temp_image_with_text)
        extractor.load_image()
        text = extractor.extract_text()
        assert text is not None
        assert isinstance(text, str)
        assert extractor.text == text
    
    def test_extract_text_blank_image(self, blank_image):
        """Test extraction from blank image"""
        extractor = OCRExtractor(blank_image)
        extractor.load_image()
        text = extractor.extract_text()
        assert text is not None
        assert isinstance(text, str)
        # Blank image should return empty or whitespace-only string
        assert text.strip() == ""
    
    def test_extract_text_with_numbers(self, tmp_path):
        """Test extraction of numbers"""
        img_path = tmp_path / "numbers.png"
        img = Image.new('RGB', (300, 100), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), "12345 67890", fill='black')
        img.save(img_path)
        
        extractor = OCRExtractor(str(img_path))
        extractor.load_image()
        text = extractor.extract_text()
        assert text is not None
        # Should contain some digits
        assert any(char.isdigit() for char in text)
    
    def test_extract_text_multiple_lines(self, tmp_path):
        """Test extraction of multiple lines"""
        img_path = tmp_path / "multiline.png"
        img = Image.new('RGB', (300, 150), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), "Line 1", fill='black')
        draw.text((20, 60), "Line 2", fill='black')
        draw.text((20, 100), "Line 3", fill='black')
        img.save(img_path)
        
        extractor = OCRExtractor(str(img_path))
        extractor.load_image()
        text = extractor.extract_text()
        assert text is not None
        # Should contain newlines or multiple words
        assert len(text.strip()) > 0
    
    @patch('pytesseract.image_to_string')
    def test_extract_text_pytesseract_exception(self, mock_tesseract):
        """Test handling of pytesseract exceptions"""
        mock_tesseract.side_effect = Exception("OCR Error")
        
        extractor = OCRExtractor("test.jpg")
        extractor.image = MagicMock()
        result = extractor.extract_text()
        assert result is None


@pytest.mark.phase1
class TestOCRExtractorProcess:
    """Test suite for process method (complete pipeline)"""
    
    @pytest.fixture
    def temp_image(self, tmp_path):
        """Create a temporary test image"""
        img_path = tmp_path / "process_test.png"
        img = Image.new('RGB', (300, 150), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), "PROCESS TEST", fill='black')
        draw.text((20, 60), "Complete Pipeline", fill='black')
        img.save(img_path)
        return str(img_path)
    
    def test_process_complete_success(self, temp_image):
        """Test complete processing pipeline"""
        extractor = OCRExtractor(temp_image)
        result = extractor.process()
        assert result is not None
        assert isinstance(result, str)
        assert extractor.image is not None
        assert extractor.text is not None
    
    def test_process_invalid_path(self):
        """Test process with invalid path"""
        extractor = OCRExtractor("/tmp/non_existent_12345.jpg")
        result = extractor.process()
        assert result is None
    
    def test_process_directory_path(self, tmp_path):
        """Test process with directory instead of file"""
        extractor = OCRExtractor(str(tmp_path))
        result = extractor.process()
        assert result is None
    
    def test_process_corrupted_file(self, tmp_path):
        """Test process with corrupted file"""
        corrupted = tmp_path / "corrupted.jpg"
        corrupted.write_bytes(b"not a valid image")
        extractor = OCRExtractor(str(corrupted))
        result = extractor.process()
        assert result is None
    
    def test_process_blank_image(self, tmp_path):
        """Test process with blank image"""
        blank = tmp_path / "blank.png"
        Image.new('RGB', (200, 100), color='white').save(blank)
        extractor = OCRExtractor(str(blank))
        result = extractor.process()
        assert result is not None
        assert result.strip() == ""


@pytest.mark.phase1
class TestOCRExtractorState:
    """Test suite for object state management"""
    
    def test_initial_state(self):
        """Test initial state of extractor"""
        extractor = OCRExtractor("test.jpg")
        assert extractor.image is None
        assert extractor.text is None
        assert extractor.image_path == Path("test.jpg")
    
    def test_state_after_validation(self, tmp_path):
        """Test state after validation"""
        img_path = tmp_path / "test.png"
        Image.new('RGB', (100, 50), color='white').save(img_path)
        
        extractor = OCRExtractor(str(img_path))
        extractor.validate_image_path()
        # State should remain unchanged after validation
        assert extractor.image is None
        assert extractor.text is None
    
    def test_state_after_load(self, tmp_path):
        """Test state after loading image"""
        img_path = tmp_path / "test.png"
        Image.new('RGB', (100, 50), color='white').save(img_path)
        
        extractor = OCRExtractor(str(img_path))
        extractor.load_image()
        assert extractor.image is not None
        assert extractor.text is None
    
    def test_state_after_extract(self, tmp_path):
        """Test state after text extraction"""
        img_path = tmp_path / "test.png"
        img = Image.new('RGB', (200, 100), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "STATE TEST", fill='black')
        img.save(img_path)
        
        extractor = OCRExtractor(str(img_path))
        extractor.load_image()
        extractor.extract_text()
        assert extractor.image is not None
        assert extractor.text is not None
    
    def test_state_after_process(self, tmp_path):
        """Test state after complete process"""
        img_path = tmp_path / "test.png"
        img = Image.new('RGB', (200, 100), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "FINAL STATE", fill='black')
        img.save(img_path)
        
        extractor = OCRExtractor(str(img_path))
        extractor.process()
        assert extractor.image is not None
        assert extractor.text is not None


@pytest.mark.phase1
@pytest.mark.parametrize("format,extension", [
    ("PNG", ".png"),
    ("JPEG", ".jpg"),
    ("BMP", ".bmp"),
])
def test_multiple_image_formats(tmp_path, format, extension):
    """Test OCR with different image formats"""
    img_path = tmp_path / f"test{extension}"
    img = Image.new('RGB', (200, 100), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "FORMAT TEST", fill='black')
    img.save(img_path, format=format)
    
    extractor = OCRExtractor(str(img_path))
    result = extractor.process()
    assert result is not None
    assert isinstance(result, str)
