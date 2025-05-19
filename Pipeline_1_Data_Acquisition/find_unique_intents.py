import pandas as pd

df= pd.read_csv(r"/home/ranjit/Desktop/Decision_Making_Model/1_Pipeline_Intent_Data_Acquisition/Day2_cleaned_dataset.csv", sep='|')



intent_counts = df['intent'].value_counts()
# Print the result
print(intent_counts)




# no of counts of a word in a particular utterance

count=0
intent_name = "reminder"
word = "reminder"

gdf=df[df["intent"]== intent_name]
# print(gdf)qw


for i in gdf["utterance"]:
    if  (word and "remind") in i:
        # print(i)
        count += 1


# print(f"\nNo of appearing the word '{word}' in '{intent_name}' is {count}")