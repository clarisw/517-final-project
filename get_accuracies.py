import os
import json
import pandas as pd
import re

def is_romantic(relationship_type: str) -> bool:
    if relationship_type in ["Spouse", "Lovers", "Courtship"]:
        return 1
    else:
        return 0
    

names_map = {}
names_df = pd.read_csv("names.csv")
for index, row in names_df.iterrows():
    race = row.iloc[1]
    name = row.iloc[2]
    percent = row.iloc[3]
    names_map[name] = race

contexts_df = pd.read_csv("contexts.csv")

# Dictionary to store accuracy per name pair
# accuracy_per_pair = {}
# totals_per_pair = {}
tp_per_pair = {}
fp_per_pair = {}
fn_per_pair = {}
tn_per_pair = {}
# correct_per_pair = {}
all_name_pairs = []
results_folder = "results"
for filename in os.listdir(results_folder):
    # 1. parse the file name format and get the names and idx from there
    match = re.match(r"(.+?)_(.+?)_\((\d+):(\d+)\)\.json", filename)
    if not match:
        print(f"Skipping non-matching file: {filename}")
        continue

    new_name1, new_name2, start_idx, end_idx = match.groups()
    start_idx, end_idx = int(start_idx), int(end_idx)

    # open json file...
    with open(os.path.join(results_folder, filename), "r") as f:
        predictions = json.load(f)

    # 2. filter to the indexes
    subset = contexts_df.iloc[start_idx:end_idx + 1]

    # 3. compute accuracy
    correct = 0
    total = 0
    # loop through each prediction in the subset of data and count whether prediction is correct
    for pred, (_, row) in zip(predictions, subset.iterrows()):
        predicted = is_romantic(pred["type"]) # boolean whether prediction is correct
        actual = bool(row["y"])  
        if predicted and actual:
            tp_per_pair[(new_name1, new_name2)] = tp_per_pair.get((new_name1, new_name2), 0) + 1
        if predicted and not actual:
            fp_per_pair[(new_name1, new_name2)] = fp_per_pair.get((new_name1, new_name2), 0) + 1
        if not predicted and actual:
            fn_per_pair[(new_name1, new_name2)] = fn_per_pair.get((new_name1, new_name2), 0) + 1
        if not predicted and not actual:
            tn_per_pair[(new_name1, new_name2)] = tn_per_pair.get((new_name1, new_name2), 0) + 1 
        all_name_pairs.append((new_name1, new_name2))

counts = {}
for pair in all_name_pairs:
    # print(f"Pair {pair}: {total}")
    counts[total] = counts.get(total, 0) + 1
    
print(counts)

# Output to csv file
final_results = []
for pair in all_name_pairs:
    name1, name2 = pair
    final_results.append(
        {"name1": name1, 
         "name2": name2, 
         "race": names_map[name1], 
         "tp": tp_per_pair.get(pair, 0),
         "fp": fp_per_pair.get(pair, 0),
         "tn": tn_per_pair.get(pair, 0),
         "fn": fn_per_pair.get(pair, 0),
         "total": tp_per_pair.get(pair, 0) + fp_per_pair.get(pair, 0) + tn_per_pair.get(pair, 0) + fn_per_pair.get(pair, 0)
         }
    )
results_df = pd.DataFrame(final_results)
results_df.to_csv("accuracy_results.csv", index=False)

print("Saved results to accuracy_results.csv")
