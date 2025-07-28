# src/extract_outline.py (full updated version)
import json
import os
import traceback  # For better error logging
from pdf_utils import load_pdf, extract_text_blocks, calculate_document_stats, extract_title, build_outline
from multiprocess import Pool  # Import here

def process_single_pdf(args):
    filename, input_dir, output_dir = args
    try:
        pdf_path = os.path.join(input_dir, filename)
        doc = load_pdf(pdf_path)
        blocks = extract_text_blocks(doc)  # Now single-threaded
        stats = calculate_document_stats(blocks)
        title = extract_title(blocks)
        outline = build_outline(blocks, stats)
        output = {"title": title, "outline": outline}
        json_filename = filename.replace('.pdf', '.json')
        with open(os.path.join(output_dir, json_filename), 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=4, ensure_ascii=False)
        doc.close()
        return f"Processed {filename}"
    except Exception as e:
        return f"Error processing {filename}: {str(e)} - Full traceback: {traceback.format_exc()}"

def process_all_pdfs(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]
    with Pool(processes=4) as pool:  # Parallel here
        results = pool.map(process_single_pdf, [(f, input_dir, output_dir) for f in pdf_files])
    for res in results:
        print(res)

if __name__ == '__main__':
    process_all_pdfs('input/', 'output/')