"""
OCR Text Extractor Class
Extracts text from images using pytesseract
"""

import sys
from pathlib import Path
from PIL import Image
import pytesseract


class OCRExtractor:
    """Class to handle OCR text extraction from images"""
    
    def __init__(self, image_path):
        """
        Initialize the OCR extractor with an image path
        
        Args:
            image_path (str): Path to the image file
        """
        self.image_path = Path(image_path)
        self.image = None
        self.text = None
    
    def validate_image_path(self):
        """
        Validate that the image path exists and is a file
        
        Returns:
            bool: True if valid, False otherwise
        """
        if not self.image_path.exists():
            print(f"Error: Image file '{self.image_path}' does not exist.", file=sys.stderr)
            return False
        
        if not self.image_path.is_file():
            print(f"Error: '{self.image_path}' is not a file.", file=sys.stderr)
            return False
        
        return True
    
    def load_image(self):
        """
        Load the image using PIL
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.image = Image.open(self.image_path)
            return True
        except Exception as e:
            print(f"Error loading image: {e}", file=sys.stderr)
            return False
    
    def extract_text(self):
        """
        Extract text from the loaded image using pytesseract
        
        Returns:
            str: Extracted text or None if extraction fails
        """
        if self.image is None:
            print("Error: No image loaded. Call load_image() first.", file=sys.stderr)
            return None
        
        try:
            self.text = pytesseract.image_to_string(self.image)
            return self.text
        except Exception as e:
            print(f"Error extracting text: {e}", file=sys.stderr)
            return None
    
    def process(self):
        """
        Complete processing pipeline: validate, load, and extract text
        
        Returns:
            str: Extracted text or None if any step fails
        """
        if not self.validate_image_path():
            return None
        
        if not self.load_image():
            return None
        
        return self.extract_text()
