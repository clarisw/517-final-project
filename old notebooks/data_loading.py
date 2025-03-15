import pandas as pd 
import numpy as np


system_prompt = "You are an avid novel reader and a code generator. Please output in JSON format. No preambles."
prompt = "Your task is to read a conversation between two people and infer the type of relationship between the two people from the given list of relationship types. Input: Following is the conversation between {char_a} and {char_b}. {context} What is the type of the relationship between {char_a} and {char_b} according to the below list of type of relationships: [ChildParent, Child-Other Family Elder, Siblings, Spouse, Lovers, Courtship, Friends, Neighbors, Roommates, Workplace Superior - Subordinate, Colleague/Partners, Opponents, Professional Contact] Constraint: Please answer in JSON format with the type of relationship and explanation for the inferred relationship. Type of relationship can only be from the provided list. Output in JSON format:"


# load in data
df = pd.read_csv('annotated_dialogues_release.csv')

# filtering based on what authors did
filtered_df = df[df['GenderA'] != df['GenderB']] # filtering different gendering in origianl dataset
filtered_df = filtered_df[filtered_df['Remarks'].isna() | (filtered_df['Remarks'].str.strip() == '')] # filtering empty remarks
filtered_df = filtered_df[:150] # only using first 150 for time

X = filtered_df[['context', 'charA', 'charB']].to_numpy()
y = np.where(filtered_df['relation'].isin(['Spouse', 'Lovers', 'Courtship']), 1, 0)

print(f"Number of samples: {len(X)}")