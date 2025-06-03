import os
import pandas as pd
import tensorflow as tf
from transformers import RobertaTokenizer

# Define file paths
csv_path = r"/home/ranjit/Desktop/Decision_Making_Model/1_Pipeline_Data_Acquisition/Day2_cleaned_dataset.csv"
save_tfrecord_path = r"/home/ranjit/Desktop/Decision_Making_Model/roberta_tokens.tfrecord"

# Load dataset
print("Loading dataset...")
df = pd.read_csv(csv_path, sep='|')
assert 'utterance' in df.columns, "CSV must contain an 'utterance' column."

# Initialize RoBERTa tokenizer
print("Initializing tokenizer...")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Tokenize all utterances
print("Tokenizing utterances...")
encoded = tokenizer(
    df["utterance"].tolist(),
    padding=True,
    truncation=True,
    max_length=80,
    return_tensors="tf"  # Return TensorFlow tensors
)

input_ids = encoded["input_ids"]      # tf.Tensor of shape (N, 80)
attention_mask = encoded["attention_mask"]  # tf.Tensor of shape (N, 80)

# Create a tf.data.Dataset
print("Creating tf.data.Dataset...")
dataset = tf.data.Dataset.from_tensor_slices({
    "input_ids": input_ids,
    "attention_mask": attention_mask,
})

# Helper function to convert array of ints to tf.train.Feature

def _int_feature(values):
    """Returns a tf.train.Feature int64_list from a list or 1D tensor of ints."""
    if isinstance(values, tf.Tensor):
        values = values.numpy().tolist()
    elif hasattr(values, 'tolist'):
        values = values.tolist()
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

# Serialize a single example

def serialize_example(example):
    """Creates a tf.train.Example message ready to be written to a file."""
    feature = {
        'input_ids': _int_feature(example['input_ids']),
        'attention_mask': _int_feature(example['attention_mask']),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

# Write TFRecord file
print("Writing TFRecord file...")
total_written = 0
with tf.io.TFRecordWriter(save_tfrecord_path) as writer:
    for record in dataset:
        serialized = serialize_example(record)
        writer.write(serialized)
        total_written += 1

print(f"Tokenization complete. TFRecord saved to {save_tfrecord_path}")

# Optional: Save dataset in TF-Shard format
# save_dir = "/home/ranjit/Desktop/Decision_Making_Model/roberta_tokens_dataset"
# dataset.save(save_dir)
# print(f"Dataset also saved in TF-Shard format at {save_dir}")
