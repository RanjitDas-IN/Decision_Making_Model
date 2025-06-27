import pandas as pd

file_path = r"/home/ranjit/Desktop/Decision_Making_Model/1_Pipeline_Data_Acquisition/Day2_cleaned_dataset.csv"
df= pd.read_csv(file_path, sep="|")

print("Shape of the CSV:",df.shape)
# print(df.describe())






def find_duplicates():
    """Find and print duplicated rows."""
    duplicates = df.duplicated()
    duplicate_rows = df[duplicates]
    print("\nNo of duplicate values:",duplicate_rows.shape[0])
    print("Duplicate rows:\n", duplicate_rows)

def drop_duplicates():
    """Drop duplicates in-place and save the updated CSV."""
    print("Before removing the duplicates shape:",df.shape)
    df.drop_duplicates(inplace=True)
    print("Shape after dropping duplicates:", df.shape)
    df.to_csv(file_path, sep="|", index=False)
    print("Cleaned data saved to CSV.")







def intent_count():
    intent_counts = df['intent'].value_counts()
    # Print the result
    print(intent_counts)
    print("The unique intents:",df["intent"].unique())







def check_word_count(padding_or_length):

    df['word_count'] = df['utterance'].apply(lambda x: len(str(x).split()))

    target_word_count = padding_or_length

    matching_rows = df[df['word_count'] == target_word_count]

    if not matching_rows.empty:
        first_match = matching_rows.iloc[0]
        original_index = first_match.name

        print(f"yes, in {first_match['word_count']} word in {original_index + 1} row, intent is {first_match['intent']}")
    else:
        print(f"{padding_or_length} word not present")







# no of counts of a word in a particular utterance

def appearing_a_word_in_utterance(df, word="tell me about"):
    total = 0
    print(f"\n🔍 Searching for the word '{word}' across all intents...\n")

    for intent in df["intent"].unique():
        gdf = df[df["intent"] == intent]
        count = 0

        for utterance in gdf["utterance"]:
            if word in utterance.lower():
                if count == 0:
                    print(f"👉 Intent: {intent}")
                print("   ", utterance)
                count += 1
                total += 1
        
        if count > 0:
            print(f"   ⟶ {count} matches in intent '{intent}'\n")

    print(f"\n✅ Total '{word}' occurrences across all intents: {total}")



    
def convert_weather_containing_utterance_to_google_search_intent_if_needed(csv_path: str, delimiter: str = '|'):
        
            df = pd.read_csv(csv_path, sep=delimiter)

            mask = df['utterance'].str.contains('weather', case=False) & (df['intent'] != 'google search')
            changes = df[mask]


            if not changes.empty:
                print("Changes to be made (intent changed to 'google search'):\n")
                print(changes)
            else:
                print("✅ No changes needed. No 'weather' entries outside of 'google search' intent.")


            df.loc[mask, 'intent'] = 'google search'


            # df.to_csv(csv_path, sep=delimiter, index=False)

            print(f"\n✔ Total entries modified: {len(changes)}")
import pandas as pd




# intent_count()
# find_duplicates()
# drop_duplicates()
# appearing_a_word_in_utterance(df)
# check_word_count(padding_or_length=2)
# convert_weather_containing_utterance_to_google_search_intent_if_needed(file_path)