# src/main.py
import sys
from extract_outline import process_all_pdfs
from persona_intelligence import process_collection

if __name__ == '__main__':
    input_dir = 'input/'
    output_dir = 'output/'
    
    if len(sys.argv) > 1 and sys.argv[1] == '1b':
        process_collection(input_dir, output_dir)
    else:
        process_all_pdfs(input_dir, output_dir)