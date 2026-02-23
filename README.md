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

## Prerequisites

### 1. Install Tesseract OCR

**macOS:**
```bash
brew install tesseract
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install tesseract-ocr
```

**Windows:**
Download from: https://github.com/UB-Mannheim/tesseract/wiki

### 2. Download the LLM Model

Download the Phi-3.5-mini-instruct model:

```bash
# Create models directory
mkdir -p models

# Download the model (example using wget)
cd models
wget https://huggingface.co/microsoft/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf

# Or download manually from:
# https://huggingface.co/microsoft/Phi-3.5-mini-instruct-GGUF
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Note:** Installing `llama-cpp-python` may take a few minutes as it compiles C++ code.

## Installation

```bash
# Clone or download the project
cd doc-intel-local-llm

# Install dependencies
pip install -r requirements.txt

# Download the model (see Prerequisites above)
```

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

## Class Methods

### `InvoiceProcessor(model_path, n_ctx, n_threads)`
Constructor to initialize the processor with model configuration.

### `load_model()`
Load the LLM model into memory.

### `extract_text_from_image(image_path)`
Extract text from an invoice image using OCR.

### `extract_structured_data(ocr_text, temperature)`
Use LLM to extract structured data from OCR text.

### `process_invoice(image_path, temperature)`
Complete pipeline: OCR + LLM extraction in one call.

### `save_results(output_path, include_ocr_text)`
Save structured data to a JSON file.

## Performance Tips

1. **CPU Threads**: Adjust `--threads` to match your CPU cores for faster processing
2. **Context Window**: Reduce `--ctx` if you have memory constraints
3. **Temperature**: Use 0 or 0.1 for deterministic, consistent output
4. **Model Size**: Q4_K_M provides good balance of speed and accuracy

## Troubleshooting

### Model Not Found
```
Error: Model file './models/Phi-3.5-mini-instruct-Q4_K_M.gguf' does not exist.
```
**Solution**: Download the model (see Prerequisites section)

### OCR Extraction Failed
```
Error: Image file 'invoice.jpg' does not exist.
```
**Solution**: Check the image path is correct

### JSON Parsing Error
```
Error: Failed to parse JSON from LLM output
```
**Solution**: Try adjusting temperature or providing a schema hint

### Out of Memory
**Solution**: Reduce `--ctx` parameter or use a smaller model

## File Structure

```
project/
├── ocr_extractor_class.py      # OCR extraction class
├── invoice_processor.py        # LLM processing class
├── main.py          # CLI script
├── requirements.txt            # Python dependencies
├── models/                     # Model files directory
│   └── Phi-3.5-mini-instruct-Q4_K_M.gguf
└── data/                       # Invoice images
```

## Next Steps

1. **Customize Schema**: Modify the extraction prompt in `create_extraction_prompt()` method
2. **Add Validation**: Implement field validation for extracted data
3. **Batch Processing**: Process multiple invoices in parallel
4. **Fine-tune Model**: Train on your specific invoice format for better accuracy

## Requirements

- Python 3.8+
- Pillow (PIL)
- pytesseract
- llama-cpp-python
- Tesseract OCR (system installation)
- ~2GB RAM for model
- ~4GB disk space for model file
