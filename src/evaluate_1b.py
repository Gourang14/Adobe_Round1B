# import json
# import os
# import sys
# import numpy as np
# from sklearn.metrics import ndcg_score  # pip install scikit-learn if needed

# def load_json(file_path):
#     # Normalize path to handle Windows issues
#     normalized_path = os.path.normpath(file_path).replace('\\', '/')
#     if not os.path.exists(normalized_path):
#         raise FileNotFoundError(f"File not found: {normalized_path}. Tip: Use forward slashes (/) or double backslashes (\\) in paths. Check if the file exists in the project folder.")
#     with open(normalized_path, 'r', encoding='utf-8') as f:
#         return json.load(f)

# def compute_precision_recall(your_sections, gt_sections):
#     # Simple set-based precision/recall for section titles (adapt as needed)
#     your_titles = set(sec['section_title'] for sec in your_sections)
#     gt_titles = set(sec['section_title'] for sec in gt_sections)
    
#     true_positives = len(your_titles.intersection(gt_titles))
#     precision = true_positives / len(your_titles) if your_titles else 0
#     recall = true_positives / len(gt_titles) if gt_titles else 0
#     return precision, recall

# def compute_ndcg(your_sections, gt_sections):
#     # NDCG for ranking (assumes importance_rank in both; lower rank = better)
#     # Map to scores: inverse rank for relevance
#     your_ranks = {sec['section_title']: 1 / sec['importance_rank'] for sec in your_sections}
#     gt_ranks = {sec['section_title']: 1 / sec['importance_rank'] for sec in gt_sections}
    
#     # Align titles
#     common_titles = list(set(your_ranks.keys()) & set(gt_ranks.keys()))
#     if not common_titles:
#         return 0.0
#     y_true = np.array([[gt_ranks.get(title, 0) for title in common_titles]])
#     y_score = np.array([[your_ranks.get(title, 0) for title in common_titles]])
#     return ndcg_score(y_true, y_score)

# def evaluate_output(your_json_path, gt_json_path):
#     try:
#         your_data = load_json(your_json_path)
#         gt_data = load_json(gt_json_path)
        
#         # Extract sections
#         your_sections = your_data.get('extracted_sections', [])
#         gt_sections = gt_data.get('extracted_sections', [])
        
#         # Compute metrics
#         precision, recall = compute_precision_recall(your_sections, gt_sections)
#         ndcg = compute_ndcg(your_sections, gt_sections)
        
#         print(f"Evaluation Results:")
#         print(f"- Precision: {precision:.2f} (fraction of your sections that match ground truth)")
#         print(f"- Recall: {recall:.2f} (fraction of ground truth sections captured by you)")
#         print(f"- NDCG@K (ranking quality): {ndcg:.2f} (1.0 = perfect ranking match)")
        
#         # Optional: Check subsection_analysis similarly (add if needed)
#     except Exception as e:
#         print(f"Error during evaluation: {e}")

# if __name__ == '__main__':
#     if len(sys.argv) != 3:
#         print("Usage: python src/evaluate_1b.py <your_output.json> <ground_truth.json>")
#         print("Example: python src/evaluate_1b.py output/results_XXX.json samples/1b/collection1/challenge1b_output.json")
#         sys.exit(1)
    
#     your_json = sys.argv[1]
#     gt_json = sys.argv[2]
#     evaluate_output(your_json, gt_json)


import json
import os
import sys
import numpy as np
from sklearn.metrics import ndcg_score  # pip install scikit-learn

def load_json(file_path):
    # Normalize path to handle Windows issues (backslashes)
    normalized_path = os.path.normpath(file_path).replace('\\', '/')
    if not os.path.exists(normalized_path):
        raise FileNotFoundError(f"File not found: {normalized_path}. Tip: Use forward slashes (/) or double backslashes (\\) in paths. Check if the file exists in the project folder.")
    with open(normalized_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compute_precision_recall_f1(your_sections, gt_sections):
    # Simple set-based metrics for section titles (adapt as needed)
    your_titles = set(sec['section_title'] for sec in your_sections)
    gt_titles = set(sec['section_title'] for sec in gt_sections)
    
    true_positives = len(your_titles.intersection(gt_titles))
    precision = true_positives / len(your_titles) if your_titles else 0
    recall = true_positives / len(gt_titles) if gt_titles else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def compute_ndcg(your_sections, gt_sections):
    # NDCG for ranking (assumes importance_rank in both; lower rank = better)
    # Map to scores: inverse rank for relevance
    your_ranks = {sec['section_title']: 1 / sec['importance_rank'] for sec in your_sections}
    gt_ranks = {sec['section_title']: 1 / sec['importance_rank'] for sec in gt_sections}
    
    # Align titles
    common_titles = list(set(your_ranks.keys()) & set(gt_ranks.keys()))
    if not common_titles:
        return 0.0
    y_true = np.array([[gt_ranks.get(title, 0) for title in common_titles]])
    y_score = np.array([[your_ranks.get(title, 0) for title in common_titles]])
    return ndcg_score(y_true, y_score)

def evaluate_output(your_json_path, gt_json_path):
    try:
        your_data = load_json(your_json_path)
        gt_data = load_json(gt_json_path)
        
        # Extract sections
        your_sections = your_data.get('extracted_sections', [])
        gt_sections = gt_data.get('extracted_sections', [])
        
        # Compute metrics
        precision, recall, f1 = compute_precision_recall_f1(your_sections, gt_sections)
        ndcg = compute_ndcg(your_sections, gt_sections)
        
        print("Evaluation Results:")
        print(f"- Precision: {precision:.2f} (fraction of your sections that match ground truth)")
        print(f"- Recall: {recall:.2f} (fraction of ground truth sections captured by you)")
        print(f"- F1-Score: {f1:.2f} (balanced measure of precision and recall)")
        print(f"- NDCG@K (ranking quality): {ndcg:.2f} (1.0 = perfect ranking match)")
        
        # Optional: Add subsection_analysis comparison here if needed
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python src/evaluate_1b.py <your_output.json> <ground_truth.json>")
        print("Example: python src/evaluate_1b.py output/results_XXX.json samples/1b/collection1/challenge1b_output.json")
        sys.exit(1)
    
    your_json = sys.argv[1]
    gt_json = sys.argv[2]
    evaluate_output(your_json, gt_json)
