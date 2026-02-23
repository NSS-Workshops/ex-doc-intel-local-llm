#!/usr/bin/env python3
"""
Main script to process invoices with OCR + Local LLM
"""

import argparse
import json
import sys
from invoice_processor import InvoiceProcessor


def main():
    """Main function to handle command line arguments and process invoices"""
    parser = argparse.ArgumentParser(
        description='Process invoice images: OCR extraction + LLM structured data extraction'
    )
    parser.add_argument(
        'image_path',
        type=str,
        help='Path to the invoice image file to process'
    )
    parser.add_argument(
        '-m', '--model',
        type=str,
        default='./models/Phi-3.5-mini-instruct-Q4_K_M.gguf',
        help='Path to the GGUF model file (default: ./models/Phi-3.5-mini-instruct-Q4_K_M.gguf)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Optional output JSON file path to save structured data',
        default=None
    )
    parser.add_argument(
        '--no-ocr-text',
        action='store_true',
        help='Do not include raw OCR text in output file'
    )
    parser.add_argument(
        '-t', '--temperature',
        type=float,
        default=0.1,
        help='LLM temperature (0 = deterministic, default: 0.1)'
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=8,
        help='Number of CPU threads to use (default: 8)'
    )
    parser.add_argument(
        '--ctx',
        type=int,
        default=4096,
        help='Context window size (default: 4096)'
    )
    
    args = parser.parse_args()
    
    # Create invoice processor instance
    processor = InvoiceProcessor(
        model_path=args.model,
        n_ctx=args.ctx,
        n_threads=args.threads
    )
    
    # Process the invoice
    structured_data = processor.process_invoice(
        args.image_path,
        temperature=args.temperature
    )
    
    if structured_data is not None:
        # Print structured data to console
        print(json.dumps(structured_data, indent=2, ensure_ascii=False))
        
        # Optionally save to file
        if args.output:
            success = processor.save_results(
                args.output,
                include_ocr_text=not args.no_ocr_text
            )
            if not success:
                sys.exit(1)
        
        sys.exit(0)
    else:
        print("Error: Failed to process invoice.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
