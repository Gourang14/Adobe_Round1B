# src/extract_outline.py
import json
import os
from pdf_utils import load_pdf, extract_text_blocks, calculate_document_stats, extract_title, build_outline

def process_all_pdfs(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        if filename.endswith('.pdf'):
            try:
                pdf_path = os.path.join(input_dir, filename)
                doc = load_pdf(pdf_path)
                blocks = extract_text_blocks(doc)
                stats = calculate_document_stats(blocks)
                title = extract_title(blocks)
                outline = build_outline(blocks, stats)
                output = {"title": title, "outline": outline}
                json_filename = filename.replace('.pdf', '.json')
                with open(os.path.join(output_dir, json_filename), 'w', encoding='utf-8') as f:  # UTF-8 for multilingual
                    json.dump(output, f, indent=4, ensure_ascii=False)  # Non-ASCII support
                doc.close()
                print(f"Processed {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == '__main__':
    process_all_pdfs('input/', 'output/')