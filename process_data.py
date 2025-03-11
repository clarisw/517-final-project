import argparse
import gc
import json
import os
import re

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def is_romantic(relationship_type: str) -> bool:
    if relationship_type in ["Spouse", "Lovers", "Courtship"]:
        return 1
    else:
        return 0

def get_opts():
    parser = argparse.ArgumentParser(description="Fine-tuning for survival prediction")

    parser.add_argument("--start-idx", type=int, help="Start Index")
    parser.add_argument("--end-idx", type=int, help="Ending Index")
    parser.add_argument("--model-name", type=str, help="Model Name from HF")
    parser.add_argument("--race", type=str, help="Race Category of Data")

    return parser.parse_args()


if __name__ == "__main__":
    opts = get_opts()

    df = pd.read_csv("annotated_dialogues_release.csv")

    system_prompt = "You are an avid novel reader and a code generator. Please output in JSON format. No preambles."

    filtered_df = df[df["GenderA"] != df["GenderB"]]
    filtered_df = filtered_df[filtered_df["Remarks"].isna() | (filtered_df["Remarks"].str.strip() == "")].iloc[opts.start_idx:opts.end_idx]

    X = filtered_df[["context", "charA", "charB"]]
    y = np.where(filtered_df["relation"].isin(["Spouse", "Lovers", "Courtship"]), 1, 0)

    name_df = pd.read_csv("names.csv")
    name_df = name_df[name_df["Race"] == opts.race]

    tokenizer = AutoTokenizer.from_pretrained(opts.model_name, token=True)
    model = AutoModelForCausalLM.from_pretrained(
        opts.model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        token=True,
    )
    print(f"Model device: {model.device}")

    results = {}

    total_combs = len(name_df) ** 2 - len(name_df)
    comb_counter = 1
    for name_row1 in name_df.values:
        new_name1 = name_row1[2]
        results[name_row1[2]] = {}

        for name_row2 in name_df.values:
            new_name2 = name_row2[2]

            if new_name1 == new_name2:
                continue

            filename = f"results/{new_name1}_{new_name2}_({opts.start_idx}:{opts.end_idx}).json"

            if os.path.exists(filename):
                continue

            print(f"Processing {comb_counter}/{total_combs} combination")

            comb_results_list = []
            for context_row, label in tqdm(zip(X.values, y), total=len(X)):
                old_name1, old_name2 = context_row[1], context_row[2]
                replaced_context = context_row[0].replace(old_name1, new_name1).replace(old_name2, new_name2)

                user_prompt = f"Your task is to read a conversation between two people and infer the type of relationship between the two people from the given list of relationship types. \n\nInput: Following is the conversation between {new_name1} and {new_name2}. \n\n{replaced_context} \n\nWhat is the type of the relationship between {new_name1} and {new_name2} according to the below list of type of relationships: [ChildParent, Child-Other Family Elder, Siblings, Spouse, Lovers, Courtship, Friends, Neighbors, Roommates, Workplace Superior - Subordinate, Colleague/Partners, Opponents, Professional Contact] \n\nConstraint: Please answer in a JSON item format with the type of relationship and explanation for the inferred relationship. Type of relationship can only be from the provided list. \n\nOutput in JSON format:"

                combined_prompt = f"{system_prompt}\n\nUser: {user_prompt}"
                inputs = tokenizer(combined_prompt, return_tensors="pt").to(model.device)

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,  # adjust as needed
                    do_sample=False,
                )

                input_length = inputs["input_ids"].shape[1]
                generated_tokens = outputs[0][input_length:]

                response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

                match = re.search(r"\{[\s\S]*?\}", response)
                json_match = re.search(r'\{.*?\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    try:
                        parsed_json = json.loads(json_str)
                        comb_results_list.append(parsed_json)
                    except json.JSONDecodeError as e:
                        print("JSON decoding failed:", e)
                        print("Original Response:", response)
                else:
                    print("No JSON block found in response:", response)

            with open(filename, "w") as f:
                json.dump(comb_results_list, f, indent=2)
            comb_counter += 1

            gc.collect()
            torch.cuda.empty_cache()

