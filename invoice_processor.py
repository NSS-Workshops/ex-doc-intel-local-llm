"""
Invoice Processor Class
Combines OCR extraction with local LLM for structured data extraction
"""

import json
import sys
from pathlib import Path
from typing import Optional, Tuple
from llama_cpp import Llama
from ocr_extractor_class import OCRExtractor
from pydantic import ValidationError
from invoice_schema import Invoice


class InvoiceProcessor:
    """Class to process invoices: OCR extraction + LLM structured data extraction"""
    
    def __init__(self, model_path="./models/Phi-3.5-mini-instruct-Q4_K_M.gguf",
                 n_ctx=4096, n_threads=8, schema_path="./schema.json", max_retries=3):
        """
        Initialize the invoice processor with LLM model
        
        Args:
            model_path (str): Path to the GGUF model file
            n_ctx (int): Context window size
            n_threads (int): Number of CPU threads to use
            schema_path (str): Path to the JSON schema file
            max_retries (int): Maximum number of retry attempts for validation failures
        """
        self.model_path = Path(model_path)
        self.schema_path = Path(schema_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.max_retries = max_retries
        self.llm = None
        self.ocr_text = None
        self.structured_data = None
        self.schema = self._load_schema()
        self.validation_errors = []
    
    def _load_schema(self):
        """
        Load the JSON schema from file
        
        Returns:
            dict: The schema as a dictionary, or None if loading fails
        """
        try:
            with open(self.schema_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
            return schema
        except Exception as e:
            print(f"Warning: Could not load schema from {self.schema_path}: {e}", file=sys.stderr)
            return None
    
    def load_model(self):
        """
        Load the LLM model
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.model_path.exists():
            print(f"Error: Model file '{self.model_path}' does not exist.", file=sys.stderr)
            print(f"Please download the model and place it at: {self.model_path}", file=sys.stderr)
            return False
        
        try:
            print(f"Loading model from {self.model_path}...", file=sys.stderr)
            self.llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False
            )
            print("Model loaded successfully.", file=sys.stderr)
            return True
        except Exception as e:
            print(f"Error loading model: {e}", file=sys.stderr)
            return False
    
    def extract_text_from_image(self, image_path):
        """
        Extract text from image using OCR
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Extracted text or None if extraction fails
        """
        print(f"Extracting text from {image_path}...", file=sys.stderr)
        extractor = OCRExtractor(image_path)
        self.ocr_text = extractor.process()
        
        if self.ocr_text:
            print(f"OCR extraction successful. Extracted {len(self.ocr_text)} characters.", 
                  file=sys.stderr)
        else:
            print("OCR extraction failed.", file=sys.stderr)
        
        return self.ocr_text
    
    def create_extraction_prompt(self, ocr_text, validation_error=None):
        """
        Create a prompt for the LLM to extract structured data
        
        Args:
            ocr_text (str): The OCR extracted text
            validation_error (str): Optional validation error from previous attempt
            
        Returns:
            str: The formatted prompt
        """
        # Convert schema to formatted JSON string
        schema_text = json.dumps(self.schema, indent=2) if self.schema else "{}"
        
        base_prompt = f"""You are a data extraction assistant. Extract structured information from the following invoice text.

Use the JSON schema provided below.
Populate all fields exactly as defined in that schema.
Do not add extra fields.
Do not rename fields.
Do not omit required fields.

Formatting rules:
- Return ONLY valid JSON.
- Do not include explanations, markdown, comments, or code blocks.
- If a field cannot be found, return null.
- All numeric values must be numbers, not strings.
- Convert dates to ISO format YYYY-MM-DD.

Invoice Text:
{ocr_text}

JSON Schema:
{schema_text}"""

        if validation_error:
            base_prompt += f"""

IMPORTANT: The previous attempt had validation errors:
{validation_error}

Please fix these errors and ensure the output strictly follows the schema."""

        base_prompt += "\n\nJSON Output:"
        
        return base_prompt
    
    def validate_invoice_data(self, data: dict) -> Tuple[bool, Optional[str], Optional[Invoice]]:
        """
        Validate invoice data against Pydantic schema
        
        Args:
            data (dict): The data to validate
            
        Returns:
            Tuple[bool, Optional[str], Optional[Invoice]]:
                (is_valid, error_message, validated_invoice)
        """
        try:
            validated_invoice = Invoice(**data)
            return True, None, validated_invoice
        except ValidationError as e:
            error_msg = str(e)
            return False, error_msg, None
        except Exception as e:
            error_msg = f"Unexpected validation error: {str(e)}"
            return False, error_msg, None
    
    def _extract_json_from_response(self, generated_text: str) -> Optional[dict]:
        """
        Extract and parse JSON from LLM response
        
        Args:
            generated_text (str): Raw LLM output
            
        Returns:
            Optional[dict]: Parsed JSON or None if parsing fails
        """
        # Remove markdown code blocks if present
        if "```json" in generated_text:
            generated_text = generated_text.split("```json")[1].split("```")[0].strip()
        elif "```" in generated_text:
            generated_text = generated_text.split("```")[1].split("```")[0].strip()
        
        try:
            return json.loads(generated_text)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}", file=sys.stderr)
            print(f"Raw output (first 500 chars): {generated_text[:500]}...", file=sys.stderr)
            return None
    
    def extract_structured_data(self, ocr_text=None, temperature=0.1):
        """
        Use LLM to extract structured data from OCR text with validation and retry logic
        
        Args:
            ocr_text (str): The OCR text to process (uses self.ocr_text if None)
            temperature (float): LLM temperature (0 = deterministic)
            
        Returns:
            dict: Structured data as dictionary, or None if extraction fails
        """
        if self.llm is None:
            print("Error: Model not loaded. Call load_model() first.", file=sys.stderr)
            return None
        
        text_to_process = ocr_text if ocr_text else self.ocr_text
        
        if not text_to_process:
            print("Error: No OCR text available to process.", file=sys.stderr)
            return None
        
        self.validation_errors = []
        validation_error = None
        
        for attempt in range(self.max_retries):
            try:
                print(f"Extracting structured data with LLM (attempt {attempt + 1}/{self.max_retries})...",
                      file=sys.stderr)
                
                prompt = self.create_extraction_prompt(text_to_process, validation_error)
                
                response = self.llm(
                    prompt,
                    temperature=temperature,
                    max_tokens=2048,
                    stop=["</s>", "\n\n\n"]
                )
                
                # Extract the generated text
                generated_text = response["choices"][0]["text"].strip()
                
                # Parse JSON
                parsed_data = self._extract_json_from_response(generated_text)
                if parsed_data is None:
                    if attempt < self.max_retries - 1:
                        print(f"Retrying due to JSON parsing error...", file=sys.stderr)
                        validation_error = "Previous output was not valid JSON. Ensure output is properly formatted JSON."
                        continue
                    else:
                        return None
                
                # Validate against Pydantic schema
                is_valid, error_msg, validated_invoice = self.validate_invoice_data(parsed_data)
                
                if is_valid:
                    print("✓ Structured data extraction and validation successful.", file=sys.stderr)
                    # Convert back to dict for compatibility
                    self.structured_data = validated_invoice.model_dump()
                    return self.structured_data
                else:
                    print(f"✗ Validation failed: {error_msg}", file=sys.stderr)
                    self.validation_errors.append({
                        'attempt': attempt + 1,
                        'error': error_msg,
                        'data': parsed_data
                    })
                    
                    if attempt < self.max_retries - 1:
                        validation_error = error_msg
                        print(f"Retrying with validation feedback...", file=sys.stderr)
                    else:
                        print(f"Max retries ({self.max_retries}) reached. Returning last parsed data despite validation errors.",
                              file=sys.stderr)
                        # Return the data even if validation failed on last attempt
                        self.structured_data = parsed_data
                        return self.structured_data
                
            except Exception as e:
                print(f"Error during structured data extraction: {e}", file=sys.stderr)
                if attempt < self.max_retries - 1:
                    print(f"Retrying...", file=sys.stderr)
                    validation_error = f"Previous attempt failed with error: {str(e)}"
                else:
                    return None
        
        return None
    
    def process_invoice(self, image_path, temperature=0.1):
        """
        Complete pipeline: OCR extraction + LLM structured data extraction
        
        Args:
            image_path (str): Path to the invoice image
            schema_hint (str): Optional hint about desired JSON schema
            temperature (float): LLM temperature
            
        Returns:
            dict: Structured invoice data or None if processing fails
        """
        # Extract text from image
        ocr_text = self.extract_text_from_image(image_path)
        if not ocr_text:
            return None
        
        # Load model if not already loaded
        if self.llm is None:
            if not self.load_model():
                return None
        
        # Extract structured data
        return self.extract_structured_data(ocr_text, temperature)
    
    def save_results(self, output_path, include_ocr_text=True):
        """
        Save the structured data to a JSON file
        
        Args:
            output_path (str): Path to save the JSON file
            include_ocr_text (bool): Whether to include raw OCR text in output
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.structured_data is None:
            print("Error: No structured data to save.", file=sys.stderr)
            return False
        
        try:
            output_data = {
                "structured_data": self.structured_data
            }
            
            if include_ocr_text and self.ocr_text:
                output_data["raw_ocr_text"] = self.ocr_text
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"Results saved to: {output_path}", file=sys.stderr)
            return True
            
        except Exception as e:
            print(f"Error saving results: {e}", file=sys.stderr)
            return False
