import os
import torch
import pandas as pd
from transformers import RobertaTokenizer

# === Load tokenizer and tokenized tensor file ===
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
tokens = torch.load("/home/ranjit/Desktop/Decision_Making_Model/roberta_tokens.pt")

# === Convert tensors to lists ===
input_id_list = tokens['input_ids'].tolist()
attention_mask_list = tokens['attention_mask'].tolist()

# === Decode tokens to strings ===
token_list = [tokenizer.convert_ids_to_tokens(ids) for ids in input_id_list]
token_string_list = [' '.join(tokens) for tokens in token_list]

# === Create DataFrame ===
df = pd.DataFrame({
    'input_ids': input_id_list,
    'attention_mask': attention_mask_list,
    'tokens': token_list,
    'token_string': token_string_list
})

# === Print a preview ===
print(df.head(5).to_markdown())

# === Optional: Save to readable formats ===
df.to_csv("readable_tokenized_dataset.csv", index=False)
# df.to_excel("readable_tokenized_dataset.xlsx", index=False)

print("Saved readable tokenized dataset.")
