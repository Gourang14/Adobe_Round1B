# Round 1B – Persona‑Driven Document Intelligence  
*Theme:* Connect What Matters — For the User Who Matters

---

## Directory Structure


.
├── input/
│   └── config.json         # persona + job definition  
├── models/  
│   ├── ms-marco-MiniLM-L-12-v2 
├── output/  
│   ├── results_round_1b_002_1.json  
│   ├── results_round_1b_002_2.json  
│   └── results_round_1b_002_3.json  
├── samples/1b/             # for local testing  
│   ├── collection1/  
│   ├── collection2/  
│   └── collection3/  
├── python_packages/        # pip‑installed dependencies  
├── src/  
│   ├── __init__.py  
│   ├── evaluate_1b.py  
│   ├── extract_outline.py  
│   ├── main.py  
│   ├── pdf_utils.py  
│   ├── persona_intelligence.py  
│   ├── process_pdfs.py  
│   └── test_imports.py  
├── venv/                   # local virtualenv (git‑ignored)  
├── .gitignore  
├── approach_explanation.md  
├── Dockerfile  
├── README.md  
└── requirements.txt  


---

## Problem Restatement  
Given a *collection of PDFs, a **persona definition, and a **job‑to‑be‑done, extract and rank the **most relevant sections* (and their subsections) in JSON. Your system must generalize across domains (research papers, financial reports, textbooks, etc.) and personas (researcher, student, analyst).

---

## Approach  

1. *Document Ingestion & Chunking*  
   - *Load* each PDF via PyMuPDF, extracting text by page.  
   - *Detect section boundaries* using font‑size heuristics (from Round 1A) and/or simple regex on common heading markers (e.g. “Chapter”, “Section”, “##”).  
   - *Split* each section further into 200–300 word sub‑chunks for fine‑grain analysis.

2. *Embedding & Semantic Indexing*  
   - Use a lightweight, CPU‑friendly sentence embedding model (e.g. all‑MiniLM‑L6‑v2, ~100 MB) from Sentence‑Transformers.  
   - *Embed* each section header and sub‑chunk.  
   - *Store* embeddings in memory (or a small FAISS index if needed).

3. *Persona + Job‑to‑be‑Done Query Vector*  
   - Concatenate persona description and job prompt.  
   - *Embed* this combined query to obtain a persona‑job vector.

4. *Relevance Scoring & Ranking*  
   - Compute cosine similarity between the persona‑job vector and each section embedding.  
   - *Rank* sections by similarity (1 = most relevant).  
   - Within each top section, rank sub‑chunks for detailed analysis.

5. *Assemble Output JSON*  
   - *Metadata*: list of input documents, persona, job, timestamp.  
   - *Extracted Sections*: document name, page, section title, importance_rank.  
   - *Sub‑section Analysis*: top sub‑chunks with refined text and page ranges.

---

## Tools & Libraries  

- PyMuPDF (fitz) for PDF parsing  
- Sentence-Transformers (ms-marco-MiniLM-L-12-v2) for embeddings  
- NumPy for vector math  
- multiprocess for parallelism  

All dependencies are offline‑capable, CPU‑only, and keep the image <1 GB.

---

## Performance & Constraints  

- *≤ 60 s* for 3–5 documents on 8 CPUs  
- *≤ 1 GB* total model + dependencies  
- *No internet* during execution  

---

## Pipeline  

text
PDFs → Text + Sections → Section & Sub‑chunk Embedding
   Persona+Job Query Embedding → Cosine Similarity & Ranking → JSON Output
