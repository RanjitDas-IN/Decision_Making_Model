import os
import pandas as pd
import torch
from tqdm import tqdm
from transformers import RobertaTokenizer

# Set paths
csv_path = os.path.expanduser(r"/home/ranjit/Desktop/Decision_Making_Model/1_Pipeline_Intent_Data_Acquisition/Day2_cleaned_dataset.csv")

# Load dataset
df = pd.read_csv(csv_path, sep='|')
assert 'utterance' in df.columns, "CSV must contain an 'utterance' column."

# Initialize RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

encoded = tokenizer(
        df['utterance'].tolist(),
        padding= True,
        truncation=True,
        max_length=80,
        return_tensors='pt'
    )


# Prepare token tensors
tokens = {
    'input_ids': encoded['input_ids'],
    'attention_mask': encoded['attention_mask']
}

# Save tokenized data
save_path = os.path.expanduser(r"/home/ranjit/Desktop/Decision_Making_Model/roberta_tokens.pt")
torch.save(tokens, save_path)
print(f"Tokenization complete. Saved tensors to {save_path}")