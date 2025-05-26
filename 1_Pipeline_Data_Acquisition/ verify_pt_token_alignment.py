import torch
from tqdm import tqdm
import pandas as pd

TOKENS_PATH = r"/home/ranjit/Desktop/Decision_Making_Model/roberta_tokens.pt"

if __name__ == "__main__":
    data = torch.load(TOKENS_PATH, map_location="cpu")

    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]

    print(f"Type of input_ids: {type(input_ids)}")
    print(f"Shape of input_ids: {getattr(input_ids, 'shape', 'N/A')}")

    print(f"Type of attention_mask: {type(attention_mask)}")
    print(f"Shape of attention_mask: {getattr(attention_mask, 'shape', 'N/A')}")

    print("\nSample input_ids[0]:", input_ids[0])
    print("Sample attention_mask[0]:", attention_mask[0])
# Detect skipped index
from transformers import RobertaTokenizer
df = pd.read_csv(r"/home/ranjit/Desktop/Decision_Making_Model/1_Pipeline_Data_Acquisition/Day2_cleaned_dataset.csv", sep='|')
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

tokenized_lengths = []
for idx, text in tqdm(enumerate(df['utterance']), total=len(df), desc="Appending"):
# for idx, text in enumerate(df['utterance']): #add a tqmd here
    try:
        enc = tokenizer(text, return_tensors='pt')
        tokenized_lengths.append((idx, enc['input_ids'].shape))
    except Exception as e:
        print(f"Failed at row {idx}: {text}")
print("Total length of the Dateframe: ",print(len(df)),"\nDone the appending process, With no error (if none printed above).")