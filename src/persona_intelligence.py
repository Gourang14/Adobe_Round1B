# src/persona_intelligence.py
import json
import os
import datetime
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from pdf_utils import load_pdf, build_outline, extract_text_blocks, calculate_document_stats
from multiprocess import Pool

model = SentenceTransformer('all-MiniLM-L6-v2')  # Offline model

def extract_section_text(doc, outline):
    sections = []
    for i, entry in enumerate(outline):
        start_page = entry['page'] - 1
        end_page = outline[i+1]['page'] - 1 if i+1 < len(outline) else len(doc) - 1
        text = ''
        for p in range(start_page, end_page + 1):
            text += doc[p].get_text()
        refined_text = text.strip().replace('\n', ' ')[:500]  # Refined (shortened/cleaned)
        sections.append({'doc': '', 'title': entry['text'], 'refined_text': refined_text, 'page': entry['page'], 'level': entry['level']})
    return sections

def compute_relevance(section_texts, job):
    # Combined TF-IDF + Embeddings (upgraded for better scoring)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([job] + section_texts)
    tfidf_scores = tfidf_matrix[0, 1:].toarray().flatten()
    
    job_emb = model.encode(job)
    section_embs = model.encode(section_texts)
    emb_scores = [util.cos_sim(job_emb, emb)[0][0].item() for emb in section_embs]
    
    return [(t + e) / 2 for t, e in zip(tfidf_scores, emb_scores)]  # Average for hybrid

def process_doc(args):
    input_dir, doc_name, job = args
    try:
        pdf_path = os.path.join(input_dir, doc_name)
        doc = load_pdf(pdf_path)
        blocks = extract_text_blocks(doc)
        stats = calculate_document_stats(blocks)
        outline = build_outline(blocks, stats)
        sections = extract_section_text(doc, outline)
        section_texts = [s['refined_text'] for s in sections]
        scores = compute_relevance(section_texts, job)
        for i, section in enumerate(sections):
            section['doc'] = doc_name
            section['importance_rank'] = scores[i]
        doc.close()
        return sections
    except Exception as e:
        print(f"Error processing {doc_name}: {e}")
        return []

def process_collection(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(input_dir, 'config.json'), 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    documents = config['documents']
    persona = config['persona']
    job = config['job']
    timestamp = datetime.datetime.now().isoformat()
    
    with Pool(processes=4) as pool:  # Parallel docs
        results = pool.map(process_doc, [(input_dir, doc_name, job) for doc_name in documents])
    all_sections = [item for sublist in results for item in sublist]
    
    # Rank overall
    all_sections.sort(key=lambda x: x['importance_rank'], reverse=True)
    for i, sec in enumerate(all_sections):
        sec['importance_rank'] = i + 1  # Final rank
    
    output_data = {
        "metadata": {
            "input_documents": documents,
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": timestamp
        },
        "extracted_sections": [
            {"document": sec['doc'], "page_number": sec['page'], "section_title": sec['title'], "importance_rank": sec['importance_rank']}
            for sec in all_sections
        ],
        "sub_section_analysis": [
            {"document": sec['doc'], "refined_text": sec['refined_text'], "page_number": sec['page']}
            for sec in all_sections
        ]
    }
    
    output_filename = os.path.join(output_dir, f"results_{timestamp.replace(':', '-')}.json")
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    process_collection('input/', 'output/')