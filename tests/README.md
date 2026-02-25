# Invoice Processing Test Suite

This directory contains comprehensive tests for the invoice processing system, including OCR extraction and LLM-based structured data extraction.

## Test Structure

Tests are organized into four categories using pytest markers:

- **`phase1`**: OCR implementation tests (unit tests)
- **`phase2`**: Invoice processor tests (unit tests with mocking)
- **`integration`**: Integration tests with real data
- **`e2e`**: End-to-end tests with complete pipeline (no mocking, requires model)

## Test Files

### 1. [`test_ocr_extractor.py`](test_ocr_extractor.py) - Phase 1 Tests

Unit tests for the `OCRExtractor` class:

- **TestOCRExtractorValidation** - Tests for `validate_image_path()` method
  - Valid file paths (relative and absolute)
  - Non-existent paths
  - Directory paths
  
- **TestOCRExtractorLoadImage** - Tests for `load_image()` method
  - Multiple image formats (PNG, JPEG, BMP)
  - Corrupted files
  - Non-image files
  - Empty files

- **TestOCRExtractorExtractText** - Tests for `extract_text()` method
  - Text extraction from images
  - Blank images
  - Multiple lines
  - Numbers and special characters
  - Error handling

- **TestOCRExtractorProcess** - Tests for `process()` method (complete pipeline)
  - Successful processing
  - Failure at each pipeline stage
  
- **TestOCRExtractorState** - Tests for object state management
  - Initial state
  - State after each operation

### 2. [`test_invoice_processor.py`](test_invoice_processor.py) - Phase 2 Tests

Unit tests for the `InvoiceProcessor` class (60+ tests):

- **TestInvoiceProcessorInitialization** - Initialization with default/custom parameters
- **TestLoadSchema** - Schema loading and error handling
- **TestLoadModel** - LLM model loading with mocking
- **TestExtractTextFromImage** - OCR integration testing
- **TestCreateExtractionPrompt** - Prompt generation with/without validation errors
- **TestValidateInvoiceData** - Pydantic validation testing
- **TestExtractJsonFromResponse** - JSON parsing from various formats
- **TestExtractStructuredData** - LLM extraction with retry logic
- **TestProcessInvoice** - Complete pipeline testing
- **TestSaveResults** - File saving with various options
- **TestInvoiceProcessorStateManagement** - State tracking across operations
- **TestEdgeCases** - Edge cases and error conditions

### 3. [`test_integration_ocr.py`](test_integration_ocr.py) - OCR Integration Tests

Integration tests for OCR with real invoice images:

- **TestOCRExtractorIntegration** - Tests with real invoice images from batches 1, 2, 3
- **TestOCRPerformance** - Performance tests for standard and large images

### 4. [`test_integration_processor.py`](test_integration_processor.py) - Processor Integration Tests

Integration tests for complete invoice processing:

- **TestInvoiceProcessorIntegration** - End-to-end processing and real image tests

### 5. [`test_end_to_end.py`](test_end_to_end.py) - End-to-End Tests

Complete pipeline tests with real invoices and LLM (no mocking):

- **TestEndToEndProcessing** - Full pipeline tests with invoices from each batch
  - Tests complete OCR → LLM → JSON pipeline
  - Validates JSON structure and field types
  - Tests with real model (requires model file)
  - Verifies data consistency across runs
  - Tests OCR text extraction quality

**Note:** These tests require the LLM model file to be present and will be skipped if not found.

## Installation

Install all dependencies (including test dependencies):

```bash
pip install -r requirements.txt
```

## Running Tests

### Run all tests:
```bash
pytest tests/
```

### Run by phase:
```bash
# Phase 1: OCR tests only
pytest -m phase1

# Phase 2: Invoice processor tests only
pytest -m phase2

# Integration tests only
pytest -m integration

# End-to-end tests only (requires model)
pytest -m e2e
```

### Run specific test file:
```bash
pytest tests/test_ocr_extractor.py -v
pytest tests/test_invoice_processor.py -v
pytest tests/test_integration_ocr.py -v
pytest tests/test_integration_processor.py -v
pytest tests/test_end_to_end.py -v
```

### Run specific test class:
```bash
pytest tests/test_ocr_extractor.py::TestOCRExtractorValidation -v
pytest tests/test_invoice_processor.py::TestLoadModel -v
```

### Run specific test:
```bash
pytest tests/test_ocr_extractor.py::TestOCRExtractorValidation::test_validate_image_path_valid_file -v
```

### Run excluding integration and e2e tests:
```bash
pytest -m "not integration and not e2e"
```

### Run only unit tests (phase1 and phase2):
```bash
pytest -m "phase1 or phase2"
```

### Run all tests except e2e (faster, no model needed):
```bash
pytest -m "not e2e"
```

## Test Markers

The test suite uses the following pytest markers (defined in [`pytest.ini`](../pytest.ini)):

- **`phase1`**: OCR implementation tests (unit tests)
- **`phase2`**: Invoice processor tests (unit tests with mocking)
- **`integration`**: Integration tests with real data
- **`e2e`**: End-to-end tests with complete pipeline (no mocking, requires model)