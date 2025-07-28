import fitz
import re
import statistics
from collections import namedtuple
import unicodedata

TextBlock = namedtuple('TextBlock', ['text', 'font_size', 'flags', 'bbox', 'page'])

def load_pdf(pdf_path):
    return fitz.open(pdf_path)

def extract_text_blocks(doc):
    blocks = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_blocks = page.get_text("dict")["blocks"]
        for b in page_blocks:
            if 'lines' in b:
                for line in b['lines']:
                    for span in line['spans']:
                        text = unicodedata.normalize('NFKC', span['text'].strip()).replace('  ', ' ')
                        if text:
                            blocks.append(TextBlock(
                                text=text,
                                font_size=span['size'],
                                flags=span['flags'],
                                bbox=span['bbox'],
                                page=page_num + 1
                            ))
    return blocks

def calculate_document_stats(blocks):
    font_sizes = [b.font_size for b in blocks if b.font_size > 0]
    avg_font_size = statistics.mean(font_sizes) if font_sizes else 10
    font_size_std = statistics.stdev(font_sizes) if len(font_sizes) > 1 else 0
    return {'avg_font_size': avg_font_size, 'font_size_std': font_size_std}

def is_heading(block, stats, prev_block=None):
    score = 0
    if block.font_size > stats['avg_font_size'] + (stats['font_size_std'] * 0.5):
        score += 3
    if block.flags & (1 << 4):  # Bold
        score += 3
    if block.flags & (1 << 1):  # Italic
        score += 1
    words = len(block.text.split())
    if 1 <= words <= 12:
        score += 1
    patterns = [
        r'^\d+\.\s', r'^\d+\.\d+\s', r'^[A-Z]\.\s', r'^(Chapter|Section|部|章)',
        r'^[IVX]+\.\s', r'^(Abstract|Introduction|結論|概要)', r'^\d+\s', r'^S\.No\s'
    ]
    if any(re.match(p, block.text, re.IGNORECASE | re.UNICODE) for p in patterns):
        score += 4
    cap_ratio = sum(1 for c in block.text if c.isupper()) / len(block.text) if len(block.text) > 0 else 0
    if cap_ratio > 0.5:
        score += 2
    if block.bbox[1] < 200:
        score += 2
    if prev_block and (block.bbox[1] - prev_block.bbox[3] > 20):
        score += 3
    if block.bbox[2] - block.bbox[0] < 300:
        score += 1
    return score >= 4  # Tuned threshold

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
    seen_texts = set()
    prev_block = None
    for block in blocks:
        if is_heading(block, stats, prev_block) and block.text not in seen_texts:
            seen_texts.add(block.text)
            level = determine_level(block, stats)
            outline.append({"level": level, "text": block.text, "page": block.page})
        prev_block = block
    return outline

def extract_section_text(doc, outline):
    sections = []
    for i, entry in enumerate(outline):
        start_page = entry['page'] - 1
        end_page = outline[i+1]['page'] - 1 if i+1 < len(outline) else len(doc) - 1
        text = ''
        for p in range(start_page, end_page + 1):
            text += doc[p].get_text('text') + ' '
        refined_text = ' '.join(text.strip().split())[:1000]  # Clean and truncate
        sections.append({
            'title': entry['text'],
            'refined_text': refined_text,
            'page': entry['page'],
            'level': entry['level']
        })
    return sections