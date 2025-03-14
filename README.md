# 517-final-project
Here, we aim to reproduce parts of results from [On the Influence of Gender and Race in Romantic Relationship Prediction from Large Language Models](https://aclanthology.org/2024.emnlp-main.29/) (Sancheti et al., EMNLP 2024).

## Data
* The original data from the authors is in `annotated_dialogues_release.csv`. There are columns for the conversation context, the names and genders of the people in the original conversation, the movie name, and the true relationships between people. 
* The filtered contexts we use is in `contexts.csv` and is created in `data_loading.ipynb`. 
* The names we used to experiment with is in `names.csv`. This file was generated from `names_list_to_csv.ipynb` and is based on data in the appendix of the original paper.

## Running the Model
The model is run with the `process_data.sh`script. 

## Model Outputs
Model outputs are stored in the results folder with the naming convention: `"results/{new_name1}_{new_name2}_({opts.start_idx}:{opts.end_idx}).json"`. 

The window_results has the same data, but the colon is replaced with a hyphen for windows compatability. 

## Evalutating
1. `get_accuracies.ipynb` generates `accuracy_results.csv` that has the distinct pairs of names and race, and the accuracy results for each.
2. `generate_heatmaps.ipynb` creates the resulting heatmaps based on the accuracy results. 
