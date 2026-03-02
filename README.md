@DEPRECATED - too many prerequisites that some tracks are not familiar with. OOP/classes, mocking , software testing etc... 

# Invoice Processor with Local LLM

A Python system that combines OCR text extraction with local LLM processing to extract structured data from invoice images.

## Features

- **OCR Text Extraction**: Uses pytesseract to extract text from invoice images
- **Local LLM Processing**: Uses Phi-3.5-mini model via llama-cpp-python for structured data extraction
- **Structured JSON Output**: Converts unstructured invoice text into structured JSON
- **Flexible Schema**: Customizable extraction schema
- **Command-line Interface**: Easy to use CLI tool
- **No API Costs**: Runs completely locally with no external API calls

## Architecture

```
Invoice Image → OCR Extractor → Raw Text → LLM Processor → Structured JSON
```

## Quick Start

### Setup

Run the setup script to automatically install all dependencies:

```bash
./setup.sh
```

This script will:
- Install Tesseract OCR (system dependency)
- Install Python packages from requirements.txt
- Create necessary directories (models, output)
- Download the LLM model (~2.4GB)
- Verify all installations

**Note:** The setup script is required and handles everything automatically. It supports macOS, Linux (Ubuntu/Debian), and Windows (Git Bash/WSL).

**For Windows users:** If Tesseract is not installed, the script will provide download instructions. After installing Tesseract, re-run the setup script.

## Usage

### Basic Usage

Process an invoice image and print structured JSON:

```bash
python main.py path/to/invoice.jpg
```

### Save Output to File

```bash
python main.py path/to/invoice.jpg -o output.json
```

### Specify Custom Model Path

```bash
python main.py path/to/invoice.jpg -m ./models/custom-model.gguf
```

### Advanced Options

```bash
python main.py path/to/invoice.jpg \
    -o ./output/output.json \
    --temperature 0.1 \
    --ctx 4096 \
    --no-ocr-text
```

python main.py data/batch_1/batch_1/batch1_1/batch1-0001.jpg \
    -o ./output/output.json

### Using as a Python Class

```python
from invoice_processor import InvoiceProcessor

# Create processor instance
processor = InvoiceProcessor(
    model_path="./models/Phi-3.5-mini-instruct-Q4_K_M.gguf",
    n_ctx=4096,
    n_threads=8
)

# Process an invoice
structured_data = processor.process_invoice("invoice.jpg")

if structured_data:
    print(structured_data)
    
    # Save results
    processor.save_results("output.json")
```

### Step-by-Step Processing

```python
from invoice_processor import InvoiceProcessor

processor = InvoiceProcessor()

# Step 1: Load the model
processor.load_model()

# Step 2: Extract text from image
ocr_text = processor.extract_text_from_image("invoice.jpg")

# Step 3: Extract structured data
structured_data = processor.extract_structured_data(ocr_text)

# Step 4: Save results
processor.save_results("output.json")
```

## Extracted Fields

The system extracts the following fields by default:

- **invoice_number**: Invoice/document number
- **date**: Invoice date
- **seller_name**: Vendor/seller company name
- **seller_address**: Seller's address
- **seller_tax_id**: Seller's tax ID
- **client_name**: Client/buyer company name
- **client_address**: Client's address
- **client_tax_id**: Client's tax ID
- **items**: List of line items with:
  - description
  - quantity
  - unit_price
  - total
- **subtotal**: Net worth/subtotal amount
- **vat_rate**: VAT/tax rate percentage
- **vat_amount**: VAT/tax amount
- **total**: Final total amount

## Example Output

```json
{
  "structured_data": {
    "invoice_number": "37060863",
    "date": "01/30/2021",
    "seller_name": "Floyd-Figueroa",
    "seller_address": "8060 Bass Walks, Coleshire, OK 97999",
    "seller_tax_id": "964-80-9962",
    "client_name": "Torres and Sons",
    "client_address": "2696 Trevino Ridges, South Sara, MT 86708",
    "client_tax_id": "953-87-7430",
    "items": [
      {
        "description": "VTech Game Console Bundle",
        "quantity": 2,
        "unit_price": 219.99,
        "total": 439.98
      }
    ],
    "subtotal": 1838.93,
    "vat_rate": "10%",
    "vat_amount": 183.89,
    "total": 2022.82
  },
  "raw_ocr_text": "Invoice no: 37060863\n..."
}
```

