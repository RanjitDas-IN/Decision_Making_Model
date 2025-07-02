import pandas as pd
import matplotlib.pyplot as plt

# def view_outliers(file_path,
#                   output_image="outliers.png",
#                   max_word_threshold=50):
#     # Load dataset
#     df = pd.read_csv(file_path, sep="|")
#     df = df.dropna(subset=['utterance'])

#     # Count words
#     df['word_count'] = df['utterance'].apply(lambda x: len(str(x).split()))

#     # Filter: only include utterances with word count <= threshold
#     filtered = df[df['word_count'] <= max_word_threshold]

#     # Group
#     count_distribution = filtered['word_count'].value_counts().sort_index()

#     # Plot
#     plt.figure(figsize=(12, 6))
#     bars = plt.bar(count_distribution.index, count_distribution.values, color='skyblue', edgecolor='black')

#     # Optional: color low/high extremes
#     for bar in bars:
#         wc = int(bar.get_x() + bar.get_width() / 2)
#         if wc <= 1 or wc >= max_word_threshold - 5:
#             bar.set_color('red')

#     # Annotate
#     for idx, value in enumerate(count_distribution.values):
#         plt.text(count_distribution.index[idx], value + 2, str(value), ha='center', fontsize=8)

#     # Labels
#     plt.xlabel("Number of Words in Utterance")
#     plt.ylabel("Number of Utterances")
#     plt.title(f"Word Count Distribution (≤ {max_word_threshold} words)")
#     plt.xticks(count_distribution.index, rotation=90)
#     plt.tight_layout()

#     # Save
#     plt.savefig(output_image)
#     plt.close()
#     print(f"[✓] Cleaned chart saved to: {output_image}")


# def format_movie_schedule_data(input_path="Temporary/find_movie_schedule.txt",
#                                 output_path="Temporary/Movie_Schedule_Cleaned.csv",
#                                 intent_name="find_movie_schedule"):
#     """
#     Reads raw movie schedule utterances from a text file, appends a pipe separator and intent label,
#     and writes the result into a CSV file with 'utterance' and 'intent' columns.

#     Parameters:
#     - input_path (str): Path to the input .txt file containing one utterance per line.
#     - output_path (str): Path where the cleaned .csv will be saved.
#     - intent_name (str): Intent label to be appended to each utterance.

#     Output:
#     - A CSV file with two columns: 'utterance' and 'intent', each row formatted as 'utterance|intent'.
#     """

#     # Step 1: Read all lines from the input text file
#     with open(input_path, 'r', encoding='utf-8') as file:
#         lines = [line.strip() for line in file if line.strip()]  # Remove empty lines

#     # Step 2: Create a DataFrame with utterance and intent
#     df = pd.DataFrame({
#         'utterance': lines,
#         'intent': intent_name
#     })

#     # Step 3: Save to CSV using pipe '|' as delimiter
#     df.to_csv(output_path, sep='|', index=False)

#     print(f"✅ Cleaned data saved to: {output_path} with {len(df)} utterances.")


# # Example usage
# format_movie_schedule_data()


# -------------------------------------------------------Finding and Remove (dependent)-------------------------------------------------------

# def find_duplicates():
#     """Find and print duplicated rows."""
#     duplicates = df.duplicated()
#     duplicate_rows = df[duplicates]
#     print("\nNo of duplicate values:",duplicate_rows.shape[0])
#     print("Duplicate rows:\n", duplicate_rows)


# def drop_duplicates():
#     """Drop duplicates in-place and save the updated CSV."""
#     print("Before removing the duplicates shape:",df.shape)
#     df.drop_duplicates(inplace=True)
#     print("Shape after dropping duplicates:", df.shape)
#     df.to_csv(r"Temporary/system.txt", sep="|", index=False)
#     print("Cleaned data saved to CSV.")


# ------------------------------------------------------Collect text and convert to my dataset format-----------------------------------------

# input_path = "Temporary/system.txt"
# output_path = "/home/ranjit/Desktop/Decision_Making_Model/Temporary/TEMP.txt"

# # Read the dataset
# df = pd.read_csv(input_path, sep='|')
# # print(df['intent'].unique())


# # Filter intent
# df = df[df["intent"] == "volume control"]
# print("Before dropping case-insensitive duplicates:", df.shape)

# # Remove case-insensitive duplicates but preserve original case
# df["text_lower"] = df["text"].str.lower()
# dff = df.drop_duplicates(subset="text_lower").drop(columns=["text_lower"])
# print("After dropping:", dff.shape)

# # Write output
# with open(output_path, "w", encoding="utf-8") as f:
#     for _, row in dff.iterrows():
#         f.write(f"{row['text']}|system\n")

# print(f"{len(dff)} entries written to {output_path}")


# --------------------------------------------------------Adding double cote in the utterance------------------------------------------------

# Wrap the 'text' column with double quotes
# df["Phrase"] = df["Phrase"].apply(lambda x: f'"{x}"')


# --------------------------------------------------------Dataset Link for system commands---------------------------------------------------

# https://github.com/lbasyal/Intent_classification/blob/main/datasets.csv
# https://www.kaggle.com/datasets/lochanbasyal/home-automation-intent-classification    
# https://www.kaggle.com/datasets/bouweceunen/smart-home-commands-dataset


# ----------------------------------------------------------Read data in JSON format and convert to my dataset format----------------------- 

# import json
# # Paths
# input_file = r"reminder.txt"
# output_file = r"1_Pipeline_Data_Acquisition/Clean_data.txt"

# # Read the original JSON-like data
# with open(input_file, 'r', encoding='utf-8') as f:
#     data = json.load(f)

# # Extract utterances
# utterances = [entry[0] for entry in data.get("oos_val", [])]

# # Write utterances with trailing '|' to the clean data file
# with open(output_file, 'w', encoding='utf-8') as f:
#     for utt in utterances:
#         f.write(f"{utt}|")
#         f.write("\n")

# print(f"Extracted {len(utterances)} utterances and saved to {output_file}")

# -----------------------------------------------------------Read the txt file as CSV and proceessing the data cleaning------------------------


# df = pd.read_csv("Temporary/system.txt",sep='|')
# print(df["intent"].unique())

# find_duplicates()
# drop_duplicates()

# --------------------------------------------------------------------------------------------------------------------------------------------

input_path = "/home/ranjit/Desktop/Decision_Making_Model/1_Pipeline_Data_Acquisition/Day2_cleaned_dataset.csv"
output_path = "outliers.csv"

# df = pd.read_csv(input_path, sep='|')

# dff = df[df['word_count'] == 26]

# print(sorted(map(int, df["word_count"].unique())))
# print(dff["intent"].unique())

# # print(dff.shape)
# # print(dff)  
# ## utterance|intent|word_count

# --------------------------------------------------------------------------------------------------------------------------------------------

# import pandas as pd
# import re

# df['word_count'] = df['utterance'].apply(lambda x: len(re.findall(r'\b\w+\b', str(x))))

# # Identify rows to remove (word count > 17)
# removed_df = df[df['word_count'] > 17]

# # Display what got removed
# print("Utterances removed (more than 17 words):")
# print(removed_df[['utterance', 'word_count']])
# removed_df[['utterance', 'intent', 'word_count']].to_csv("Temporary/Removed_Utterances.csv", sep="|", index=False)

# # Keep only utterances with 17 or fewer words
# cleaned_df = df[df['word_count'] <= 17].copy()

# # Drop the helper column
# cleaned_df.drop(columns=['word_count'], inplace=True)

# # Save the cleaned dataset
# cleaned_df.to_csv("demo.csv", sep="|", index=False)

# print(f"\nCleaned CSV saved as 'demo.csv'. {len(removed_df)} utterances removed.")


# --------------------------------------------------------------------------------------------------------------------------------------------
# def check_word_count(padding_or_length):
#     # assume df is already in scope
#     df['word_count'] = df['utterance'].apply(lambda x: len(str(x).split()))

#     # filter rows with exactly `padding_or_length` words
#     matching_rows = df[df['word_count'] == padding_or_length]

#     if not matching_rows.empty:
#         print(f"Found {len(matching_rows)} utterance(s) with exactly {padding_or_length} words:\n")
#         for idx, row in matching_rows.iterrows():
#             print(f"- Row {idx + 1}: {row['word_count']} words, intent: {row['intent']}")
#             print(f"  Utterance: \"{row['utterance']}\"\n")
#     else:
#         print(f"No utterances found with exactly {padding_or_length} words.")

# check_word_count(4)

# # --------------------------------------------------------------------------------------------------------------------------------------------
# df = pd.read_csv("/home/ranjit/Desktop/Decision_Making_Model/Removed_Utterances.csv", sep='|')
# # print(df.shape)
# # print(df["intent"].unique()) #reminder' 'content'
# reminder_df = df[df["intent"]== "content"]
# print(reminder_df.shape)

# # with open("Temporary/content.txt", "w", encoding="utf-8") as f:
# #     for _, row in reminder_df.iterrows():
# #         f.write(f"{row['utterance']}|{row['intent']}\n")

# # --------------------------------------------------------------------------------------------------------------------------------------------

df = pd.read_csv("Temporary/calculator.txt", sep='|')
print(df.shape)
print(df["intent"].unique())  # 'reminder'
# check for duplicates
duplicates = df.duplicated(subset=['utterance'])
if duplicates.any():
    print(f"Found {duplicates.sum()} duplicate utterances.")
    # df = df[~duplicates]  # Remove duplicates
else:
    print("No duplicate utterances found.")
