# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('all-MiniLM-L6-v2')  # Downloads ~80MB model
# model.save('models/all-MiniLM-L6-v2')  # Save for offline use
# print("Model saved offline.")

#### APPROACH 1
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import os

#    # Download and save tokenizer + model locally (run this once with internet)
# model_name = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
# local_path = 'models/ms-marco-MiniLM-L-12-v2'  # Save here (adjust if needed)

# if not os.path.exists(local_path):
#        print(f"Downloading {model_name} to {local_path}...")
#        tokenizer = AutoTokenizer.from_pretrained(model_name)
#        model = AutoModelForSequenceClassification.from_pretrained(model_name)
#        tokenizer.save_pretrained(local_path)
#        model.save_pretrained(local_path)
#        print(f"Model saved to {local_path}")
# else:
#        print(f"Model already exists at {local_path}")

   
### APPROACH 3

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Download and save cross-encoder locally (run once with internet; ~130MB)
model_name = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
local_path = 'models/ms-marco-MiniLM-L-12-v2'  # Save here

if not os.path.exists(local_path):
    print(f"Downloading {model_name} to {local_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer.save_pretrained(local_path)
    model.save_pretrained(local_path)
    print(f"Model saved to {local_path}")
else:
    print(f"Model already exists at {local_path}")

# Add blocks for other models (e.g., distilgpt2 from Approach 2) if needed...
