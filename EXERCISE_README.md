# OCR Text Extractor - Coding Exercise

## 🎯 Objective

Complete the implementation of the `extract_text()` method in the [`OCRExtractor`](ocr_extractor_class.py) class to extract text from images using pytesseract.

## 📋 Background

This project uses Optical Character Recognition (OCR) to extract text from images. The [`OCRExtractor`](ocr_extractor_class.py) class handles the complete pipeline:
1. Validating image paths
2. Loading images using PIL (Python Imaging Library)
3. **Extracting text using pytesseract** ← Your task!

## 📝 Your Task

### Location
File: [`ocr_extractor_class.py`](ocr_extractor_class.py)  
Method: `extract_text()` (lines 57-68)

### Requirements

Implement the [`extract_text()`](ocr_extractor_class.py:57) method to:

1. **Check if image is loaded**: 
   - If `self.image` is `None`, print an error message to stderr and return `None`
   - Error message format: `"Error: No image loaded. Call load_image() first."`

2. **Extract text using pytesseract**:
   - Use `pytesseract.image_to_string()` to extract text from `self.image`
   - Store the extracted text in `self.text`
   - Return the extracted text

3. **Handle exceptions**:
   - Wrap the pytesseract call in a try-except block
   - If an exception occurs, print an error message to stderr and return `None`
   - Error message format: `f"Error extracting text: {e}"`

### Method Signature
```python
def extract_text(self):
    """
    Extract text from the loaded image using pytesseract
    
    Returns:
        str: Extracted text or None if extraction fails
    """
    # Your implementation here
```

### Expected Behavior

**Success case:**
```python
extractor = OCRExtractor("path/to/image.jpg")
extractor.load_image()
text = extractor.extract_text()
# text contains the extracted text as a string
# extractor.text also contains the same text
```

**Error case (no image loaded):**
```python
extractor = OCRExtractor("path/to/image.jpg")
text = extractor.extract_text()  # No load_image() called
# Prints: "Error: No image loaded. Call load_image() first."
# Returns: None
```

**Error case (pytesseract exception):**
```python
extractor = OCRExtractor("path/to/image.jpg")
extractor.load_image()
# If pytesseract fails for any reason
# Prints: "Error extracting text: <exception message>"
# Returns: None
```

## 🧪 Testing

### Run Phase 1 Tests

The project includes comprehensive tests marked with `@pytest.mark.phase1`. Run these to verify your implementation:

```bash
pytest tests/test_ocr_extractor.py -m phase1 -v
```

### Expected Test Results

**Before implementation** (tests should FAIL):
```
FAILED tests/test_ocr_extractor.py::TestOCRExtractorExtractText::test_extract_text_without_loading
FAILED tests/test_ocr_extractor.py::TestOCRExtractorExtractText::test_extract_text_success
... (more failures)
```

**After correct implementation** (tests should PASS):
```
PASSED tests/test_ocr_extractor.py::TestOCRExtractorValidation::test_validate_image_path_valid_file
PASSED tests/test_ocr_extractor.py::TestOCRExtractorLoadImage::test_load_image_success
PASSED tests/test_ocr_extractor.py::TestOCRExtractorExtractText::test_extract_text_success
... (all tests pass)
```

## 💡 Hints

<details>
<summary><strong>Hint 1: Basic Structure</strong> (Click to reveal)</summary>

Your implementation should follow this structure:

```python
def extract_text(self):
    # Step 1: Check if image is loaded
    if self.image is None:
        # Print error to stderr
        # Return None
    
    # Step 2: Try to extract text
    try:
        # Use pytesseract.image_to_string()
        # Store result in self.text
        # Return the text
    except Exception as e:
        # Print error to stderr
        # Return None
```

**Key points:**
- Use `print(..., file=sys.stderr)` to print to stderr
- `pytesseract.image_to_string()` takes an image object as parameter
- Store the result in both `self.text` AND return it

</details>

<details>
<summary><strong>Hint 2: Complete Implementation Template</strong> (Click to reveal)</summary>

Here's a more detailed template:

```python
def extract_text(self):
    """
    Extract text from the loaded image using pytesseract
    
    Returns:
        str: Extracted text or None if extraction fails
    """
    # Check if image is loaded
    if self.image is None:
        print("Error: No image loaded. Call load_image() first.", file=sys.stderr)
        return None
    
    # Try to extract text
    try:
        # Call pytesseract to extract text from the image
        self.text = pytesseract.image_to_string(self.image)
        # Return the extracted text
        return self.text
    except Exception as e:
        # Handle any exceptions
        print(f"Error extracting text: {e}", file=sys.stderr)
        return None
```

1. **Line 1-2**: Check if `self.image` is `None` (no image loaded)
2. **Line 3**: Print error message to stderr using `sys.stderr`
3. **Line 4**: Return `None` to indicate failure
4. **Line 7-8**: Use `pytesseract.image_to_string()` to extract text and store in `self.text`
5. **Line 9**: Return the extracted text
6. **Line 10-12**: Catch any exceptions, print error, and return `None`

</details>

## 📚 Resources

- [pytesseract Documentation](https://pypi.org/project/pytesseract/)
- [Pillow (PIL) Documentation](https://pillow.readthedocs.io/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [pytest Documentation](https://docs.pytest.org/)

## ✅ Success Criteria

Your implementation is complete when:
1. ✅ All phase1 tests pass: `pytest tests/test_ocr_extractor.py -m phase1 -v`
3. ✅ The method extracts text from valid images
4. ✅ The method handles exceptions gracefully
5. ✅ Both `self.text` and the return value contain the extracted text

## 🚀 Getting Started

1. Open [`ocr_extractor_class.py`](ocr_extractor_class.py)
2. Find the [`extract_text()`](ocr_extractor_class.py:57) method (around line 57)
3. Replace the `pass` statement with your implementation
4. Run the tests: `pytest tests/test_ocr_extractor.py -m phase1 -v`
5. Iterate until all tests pass!

Good luck! 🎉
