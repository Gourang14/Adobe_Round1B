# src/pdf_utils.py
import fitz
import re
import statistics
from collections import namedtuple
from multiprocess import Pool  # CPU parallel
import unicodedata  # For multilingual normalization

TextBlock = namedtuple('TextBlock', ['text', 'font_size', 'flags', 'bbox', 'page'])

def load_pdf(pdf_path):
    return fitz.open(pdf_path)

def extract_page_blocks(page_args):
    page, page_num = page_args
    blocks = []
    page_blocks = page.get_text("dict")["blocks"]
    for b in page_blocks:
        if 'lines' in b:
            for line in b['lines']:
                for span in line['spans']:
                    text = unicodedata.normalize('NFKC', span['text'].strip())  # Multilingual normalize
                    if text:
                        blocks.append(TextBlock(
                            text=text,
                            font_size=span['size'],
                            flags=span['flags'],
                            bbox=span['bbox'],
                            page=page_num + 1
                        ))
    return blocks

def extract_text_blocks(doc):
    with Pool(processes=4) as pool:  # Use 4 CPUs (adjust to 8 for runtime)
        results = pool.map(extract_page_blocks, [(doc[i], i) for i in range(len(doc))])
    return [item for sublist in results for item in sublist]  # Flatten

def calculate_document_stats(blocks):
    font_sizes = [b.font_size for b in blocks if b.font_size > 0]
    avg_font_size = statistics.mean(font_sizes) if font_sizes else 10
    font_size_std = statistics.stdev(font_sizes) if len(font_sizes) > 1 else 0
    return {'avg_font_size': avg_font_size, 'font_size_std': font_size_std}

def is_heading(block, stats, prev_block=None):
    score = 0
    # Font features (upgraded with std dev for robustness)
    if block.font_size > stats['avg_font_size'] + stats['font_size_std']:
        score += 3
    if block.flags & (1 << 4):  # Bold
        score += 3
    if block.flags & (1 << 1):  # Italic (sometimes used in headings)
        score += 1
    
    # Content features (upgraded with more patterns, multilingual keywords)
    words = len(block.text.split())
    if 1 <= words <= 12:  # Slightly wider range for accuracy
        score += 1
    patterns = [
        r'^\d+\.?\d*\s',  # 1. or 1.1
        r'^[A-Z]\.\s',    # A.
        r'^(Chapter|Section|部|章)',  # English/Japanese keywords (bonus multilingual)
        r'^[IVX]+\.\s',   # Roman numerals
        r'^(Abstract|Introduction|結論|概要)'  # Common sections in EN/JP
    ]
    if any(re.match(p, block.text, re.IGNORECASE | re.UNICODE) for p in patterns):
        score += 4
    cap_ratio = sum(1 for c in block.text if c.isupper()) / len(block.text) if len(block.text) > 0 else 0
    if cap_ratio > 0.5:
        score += 2
    
    # Spatial features (upgraded with better spacing/isolation check)
    if block.bbox[1] < 150:  # Near top (adjusted threshold)
        score += 2
    if prev_block and (block.bbox[1] - prev_block.bbox[3] > 30):  # Larger spacing
        score += 3
    if block.bbox[2] - block.bbox[0] < 300:  # Not too wide (isolated line)
        score += 1
    
    return score >= 6  # Adjusted threshold for better precision/recall

def determine_level(block, stats):
    ratio = block.font_size / stats['avg_font_size']
    if ratio > 1.6:
        return "H1"
    elif ratio > 1.3:
        return "H2"
    else:
        return "H3"

def extract_title(blocks):
    for b in blocks:
        if b.page == 1 and b.font_size > 14 and (b.flags & (1 << 4)):
            return b.text
    return "Untitled"

def build_outline(blocks, stats):
    outline = []
    prev_block = None
    for block in blocks:
        if is_heading(block, stats, prev_block):
            level = determine_level(block, stats)
            outline.append({"level": level, "text": block.text, "page": block.page})
        prev_block = block
    return outline