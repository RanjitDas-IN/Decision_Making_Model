import pandas as pd
# ------------------------------------------------------Appending the pipe & intent-------------------------------------------------------------------- 
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
#     df.to_csv("Temporary/TEMP.txt", sep="|", index=False)
#     print("Cleaned data saved to CSV.")


# ------------------------------------------------------Collect text and convert to my dataset format-----------------------------------------


# # Input and output paths
# input_path = "/home/ranjit/Desktop/Decision_Making_Model/Dataset-train-pipe.csv"
# output_path = "/home/ranjit/Desktop/Decision_Making_Model/Temporary/TEMP.txt"

# ## Read the dataset using pipe delimiter
# df = pd.read_csv(input_path,sep='|')
# print(df["intent"].unique())
# # print(df)

# df = df[df["intent"] == "yes"]
# print(df.shape)
# # Write to output file in the desired format
# with open(output_path, "w", encoding="utf-8") as f:
#     for _, row in df.iterrows():
#         f.write(f"{row['utterance']}|general\n")
# print(f"{len(df)} entries written to {output_path}")


# --------------------------------------------------------Adding double cote in the utterance------------------------------------------------

# Wrap the 'text' column with double quotes
# df["Phrase"] = df["Phrase"].apply(lambda x: f'"{x}"')


# --------------------------------------------------------Dataset Link for 'system' commands---------------------------------------------------

# https://github.com/lbasyal/Intent_classification/blob/main/datasets.csv
# https://www.kaggle.com/datasets/lochanbasyal/home-automation-intent-classification    
# https://www.kaggle.com/datasets/bouweceunen/smart-home-commands-dataset


# ----------------------------------------------------------Read data in JSON format and convert to my dataset format----------------------- 

# import json
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

# ----------------------------------------------------------Various Cheaks in Dataset------------------------------------------------------------

# def check_word_count(padding_or_length):

#     df['word_count'] = df['utterance'].apply(lambda x: len(str(x).split()))

#     target_word_count = padding_or_length

#     matching_rows = df[df['word_count'] == target_word_count]

#     if not matching_rows.empty:
#         first_match = matching_rows.iloc[0]
#         original_index = first_match.name

#         print(f"yes, in {first_match['word_count']} word in {original_index + 1} row, intent is {first_match['intent']}")
#     else:
#         print(f"{padding_or_length} word not present")

# # no of counts of a word in a particular utterance

# def appearing_a_word_in_utterance():
#     count=0
#     intent_name = "google search"
#     word = "tutorial"

#     gdf=df[df["intent"]== intent_name]
#     print("\n\n")

#     for i in gdf["utterance"]:
#         if  (word) in i:
#             print(i)
#             count += 1
#     print(f"\n\nNo of appearing the word '{word}' in '{intent_name}' is {count}")


# -----------------------------------------------------------Read the txt file as CSV and proceessing the data cleaning------------------------

#calculator, content, find_movie_schedule, RateBook, Reminder, system
# df = pd.read_csv("Temporary/system.txt", sep="|")
# print(df.shape)
# print(df["intent"].unique())

# find_duplicates()
# drop_duplicates()