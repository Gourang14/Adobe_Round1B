import json
import os
import datetime
from rank_bm25 import BM25Okapi
import re
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from pdf_utils import load_pdf, extract_text_blocks, calculate_document_stats, build_outline, extract_section_text
from multiprocess import Pool
import traceback
import time

nltk.data.path.append('models/nltk_data')
local_model_path = 'models/ms-marco-MiniLM-L-12-v2'

if not os.path.exists(local_model_path):
    raise FileNotFoundError(f"Local model path '{local_model_path}' does not exist. Run test_model.py to download it first.")

tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForSequenceClassification.from_pretrained(local_model_path)

def fallback_tokenize(text):
    return re.findall(r'\w+', text.lower())

def compute_relevance(section_texts, job_task, persona_role):
    try:
        query = f"{persona_role} needs to: {job_task}"
        filtered_texts = [text for text in section_texts if len(text.split()) >= 20]
        if not filtered_texts:
            return [0.0] * len(section_texts)

        try:
            from nltk.tokenize import word_tokenize
            tokenized_texts = [word_tokenize(t.lower()) for t in filtered_texts]
            query_tokens = word_tokenize(query.lower())
        except LookupError:
            print("NLTK 'punkt_tab' not found - using fallback tokenization.")
            tokenized_texts = [fallback_tokenize(t) for t in filtered_texts]
            query_tokens = fallback_tokenize(query)
        
        bm25 = BM25Okapi(tokenized_texts)
        bm25_scores = bm25.get_scores(query_tokens)

        top_indices = np.argsort(bm25_scores)[-50:][::-1]
        top_texts = [filtered_texts[i] for i in top_indices]

        pairs = [[query, text[:500]] for text in top_texts] 
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits.squeeze().cpu().numpy()
        rerank_scores = 1 / (1 + np.exp(-logits)) 

        full_scores = [0.0] * len(filtered_texts)
        for idx, score in zip(top_indices, rerank_scores):
            full_scores[idx] = score
        original_scores = [0.0] * len(section_texts)
        filtered_idx = 0
        for i, text in enumerate(section_texts):
            if len(text.split()) >= 20:
                original_scores[i] = full_scores[filtered_idx]
                filtered_idx += 1

        hybrid_scores = []
        for score, text in zip(original_scores, section_texts):
            length = len(text.split())
            length_penalty = 1.0 if 50 <= length <= 500 else 0.8
            hybrid_scores.append(score * length_penalty)

        max_score = max(hybrid_scores) if hybrid_scores and max(hybrid_scores) > 0 else 1.0
        normalized_scores = [s / max_score for s in hybrid_scores]

        avg_score = np.mean(normalized_scores)
        print(f"Avg relevance score (BM25 + Rerank): {avg_score:.2f}")
        
        return normalized_scores
    except Exception as e:
        print(f"Relevance error: {e}")
        return [0.0] * len(section_texts)

def process_single_doc(args):
    doc_info, input_dir, job_task, persona_role = args
    doc_name = doc_info['filename']
    try:
        pdf_path = os.path.join(input_dir, doc_name)
        doc = load_pdf(pdf_path)
        blocks = extract_text_blocks(doc)
        stats = calculate_document_stats(blocks)
        outline = build_outline(blocks, stats)
        sections = extract_section_text(doc, outline)
        section_texts = [s['refined_text'][:500] for s in sections]
        scores = compute_relevance(section_texts, job_task, persona_role)
        for i, section in enumerate(sections):
            section['doc'] = doc_name
            section['importance_rank'] = scores[i]
            section['page'] = section['page']
        doc.close()
        return sections
    except Exception as e:
        print(f"Error processing {doc_name}: {str(e)} - {traceback.format_exc()}")
        return []

def process_collection(input_dir, output_dir):
    start_time = time.time()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    config_path = os.path.join(input_dir, 'config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    challenge_id = config['challenge_info']['challenge_id']
    documents = config['documents']
    persona_role = config['persona']['role']
    job_task = config['job_to_be_done']['task']
    timestamp = datetime.datetime.now().isoformat().replace(':', '-') 
    with Pool(processes=os.cpu_count() // 2) as pool:
        results = pool.map(process_single_doc, [(d, input_dir, job_task, persona_role) for d in documents])
    all_sections = [item for sublist in results for item in sublist if item['importance_rank'] >= 0.45]
    
    all_sections.sort(key=lambda x: (-x['importance_rank'], x['page']))
    for i, sec in enumerate(all_sections):
        sec['importance_rank'] = i + 1
    
    output_data = {
        "metadata": {
            "input_documents": [d['filename'] for d in documents],
            "persona": persona_role,
            "job_to_be_done": job_task,
            "processing_timestamp": timestamp
        },
        "extracted_sections": [
            {
                "document": sec['doc'],
                "page_number": sec['page'],
                "section_title": sec['title'],
                "importance_rank": sec['importance_rank']
            }
            for sec in all_sections
        ],
        "subsection_analysis": [
            {
                "document": sec['doc'],
                "refined_text": sec['refined_text'],
                "page_number": sec['page']
            }
            for sec in all_sections
        ]
    }
    
    output_filename = os.path.join(output_dir, f"results_{challenge_id}_{timestamp}.json")
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    print(f"Output saved to {output_filename}")
    print(f"Processing time: {time.time() - start_time} seconds")

if __name__ == '__main__':
    process_collection('input/', 'output/')
