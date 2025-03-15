import seaborn as sns
import pandas as pd

names_df = pd.read_csv("names.csv")

# race -> name -> percent female at birth
names_map = {}
for index, row in names_df.iterrows():
    race = row.iloc[1]
    name = row.iloc[2]
    percent = row.iloc[3]
    if race not in names_map:
        names_map[race] = {}
    names_map[race][name] = percent

results_df = pd.read_csv("accuracy_results.csv")

results_df["percent 1"] = results_df.apply(lambda x: names_map[x["race"]][x["name1"]], axis=1)
results_df["percent 2"] = results_df.apply(lambda x: names_map[x["race"]][x["name2"]], axis=1)

results_df["bucket 1"] = pd.cut(x=results_df['percent 1'], bins=[-1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 100], labels=["0-2", "2-5", "5-10", "10-25", "25-50", "50-75", "75-90", "90-95", "95-98", "98-100"])
results_df["bucket 2"] = pd.cut(x=results_df['percent 2'], bins=[-1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 100], labels=["0-2", "2-5", "5-10", "10-25", "25-50", "50-75", "75-90", "90-95", "95-98", "98-100"])

# grouped_df = results_df.groupby(by=["bucket 1", "bucket 2"], as_index=False)["accuracy"].mean()
grouped_df = results_df.groupby(by=["race", "bucket 1", "bucket 2"], as_index=False)[["tp","tn","fp","fn","total"]].sum()

grouped_df["precision"] = (grouped_df["tp"] / (grouped_df["tp"] + grouped_df["fp"])).round(2)
grouped_df["accuracy"] = ((grouped_df["tp"] + grouped_df["tn"]) / grouped_df["total"]).round(2)
grouped_df["recall"] = (grouped_df["tp"] / (grouped_df["tp"] + grouped_df["fn"])).round(2)
grouped_df["f1"] = (2 * (grouped_df["precision"] * grouped_df["recall"]) / (grouped_df["precision"] + grouped_df["recall"])).round(2)

import matplotlib.pyplot as plt
def generate_heatmap(df, value, race):
    heatmap_df = pd.pivot_table(grouped_df[grouped_df["race"] == race], values=value, index='bucket 1',
                       columns='bucket 2')
    cmap = sns.light_palette("seagreen", as_cmap=True)
    plt.figure()
    ax = sns.heatmap(data=heatmap_df, cmap="Greens", annot=True)
    ax.set_xlabel("% Female")
    ax.set_ylabel("% Female")
    ax.set_title(f"{race} {value}")
    plt.savefig(f"{race}-{value}.png")
    plt.plot()

races = ["Asian", "Black", "Hispanic","White"]
metric_types = ["recall", "accuracy", "precision","f1"]
for race in races:
    for metric in metric_types:
        generate_heatmap(grouped_df, metric, race)
