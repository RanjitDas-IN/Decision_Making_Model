import pandas as pd
import matplotlib.pyplot as plt

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
    # assume df is already in scope
    df['word_count'] = df['utterance'].apply(lambda x: len(str(x).split()))

    # filter rows with exactly `padding_or_length` words
    matching_rows = df[df['word_count'] == padding_or_length]

    if not matching_rows.empty:
        print(f"Found {len(matching_rows)} utterance(s) with exactly {padding_or_length} words:\n")
        for idx, row in matching_rows.iterrows():
            print(f"- Row {idx + 1}: {row['word_count']} words, intent: {row['intent']}")
            print(f"  Utterance: \"{row['utterance']}\"\n")
    else:
        print(f"No utterances found with exactly {padding_or_length} words.")








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





def view_outliers(file_path,
                  output_image="outliers.png",
                  max_word_threshold=50):
    # Load dataset
    df = pd.read_csv(file_path, sep="|")
    df = df.dropna(subset=['utterance'])

    # Count words
    df['word_count'] = df['utterance'].apply(lambda x: len(str(x).split()))

    # Filter: only include utterances with word count <= threshold
    filtered = df[df['word_count'] <= max_word_threshold]

    # Group
    count_distribution = filtered['word_count'].value_counts().sort_index()

    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(count_distribution.index, count_distribution.values, color='skyblue', edgecolor='black')

    # Optional: color low/high extremes
    for bar in bars:
        wc = int(bar.get_x() + bar.get_width() / 2)
        if wc <= 1 or wc >= max_word_threshold - 5:
            bar.set_color('red')

    # Annotate
    for idx, value in enumerate(count_distribution.values):
        plt.text(count_distribution.index[idx], value + 2, str(value), ha='center', fontsize=8)

    # Labels
    plt.xlabel("Number of Words in Utterance")
    plt.ylabel("Number of Utterances")
    plt.title(f"Word Count Distribution (≤ {max_word_threshold} words)")
    plt.xticks(count_distribution.index, rotation=90)
    plt.tight_layout()

    # Save
    plt.savefig(output_image)
    plt.close()
    print(f"[✓] Cleaned chart saved to: {output_image}")






intent_count()
# view_outliers(file_path)
# find_duplicates()
# drop_duplicates()
# appearing_a_word_in_utterance(df)
# check_word_count(padding_or_length=21)
# convert_weather_containing_utterance_to_google_search_intent_if_needed(file_path)