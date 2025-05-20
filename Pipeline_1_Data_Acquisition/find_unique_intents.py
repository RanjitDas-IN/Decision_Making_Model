import pandas as pd

df= pd.read_csv(r"/home/ranjit/Desktop/Decision_Making_Model/Pipeline_1_Data_Acquisition/Day2_cleaned_dataset.csv", sep='|')


def intent_count():
    intent_counts = df['intent'].value_counts()
    # Print the result
    print(intent_counts)

def check_word_count(padding_or_length):

    df['word_count'] = df['utterance'].apply(lambda x: len(str(x).split()))

    target_word_count = padding_or_length

    matching_rows = df[df['word_count'] == target_word_count]

    if not matching_rows.empty:
        first_match = matching_rows.iloc[0]
        original_index = first_match.name

        print(f"yes, in {first_match['word_count']} word in {original_index + 1} row, intent is {first_match['intent']}")
    else:
        print("no")

# no of counts of a word in a particular utterance

def appearing_a_word_in_utterance():
    count=0
    intent_name = "reminder"
    word = "reminder"

    gdf=df[df["intent"]== intent_name]

    for i in gdf["utterance"]:
        if  (word and "remind") in i:
            # print(i)
            count += 1
    print(f"\nNo of appearing the word '{word}' in '{intent_name}' is {count}")




# intent_count()
# appearing_a_word_in_utterance()
check_word_count(padding_or_length=40)