#%%

import os

os.chdir(os.path.dirname(__file__))

# %% CONFIGURATION

brats_dataset = "/home/VoxLogicA/datasets/MICCAI_BraTS2020_TrainingData"
voxlogica_path = (
    "/home/VoxLogicA/binaries/VoxLogicA_1.3.3-experimental_linux-x64/VoxLogicA"
)
output_path = "output"
stats_output_path = "output"  # for debug, should be equal to output_path for production

hIRange = range(88, 96)

# NOTE: if VoxLogicA runs out of memory, reduce the following range, and repeat the experiment. Overlapping ranges do not harm, the results will be merged without repeating computations
vIRange = range(70, 97)

do_run_voxlogica = (
    False  # set to False to just compute the statistics out of the json files
)

# %% Imports

from voxlogica import VoxLogicA
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %% VoxLogicA specification as a Python function


def functions():
    return f"""
let dice(f,g) = (2 .*. volume(f & g)) ./. (volume(f) .+. volume(g))
let sensitivity(f,g) = volume(f & g) ./. (volume(f & g) .+. volume((!f) & (g)))
let specificity(f,g) = volume((!f) & (!g)) ./. (volume((!f) & (!g)) .+. volume((f) & (!g)))

let grow(f,g) = (f | touch(g,f)) 
let smoothen(r,a) = distleq(r,distgeq(r,! a))
let smoothen(r,f) = distleq(r,distgeq(r,! f))
"""


def gbm_tacas_part1(dataset_path, case_name, hI):
    return f"""
load imgflair = "{dataset_path}/{case_name}/{case_name}_flair.nii.gz"
let flair = intensity(imgflair)

load imgGrndTruth = "{dataset_path}/{case_name}/{case_name}_seg.nii.gz"
let grndTruthGTV = intensity(imgGrndTruth) >. 0

let background = touch(flair <. 0.1,border)
let brain = !background
let pflair = percentiles(flair,brain,0)
let hI = pflair >. 0.{hI}
let hyperIntense = smoothen(5.0,hI)
save "{output_path}/{case_name}/hyperIntense-{hI}.nii.gz" hyperIntense
"""


def gbm_tacas_part2(case_name, hI, vI):
    return f"""
let vI = pflair >. 0.{vI}
let veryIntense =  smoothen(2.0,vI)
let growTum = grow(hyperIntense,veryIntense)
let gtv = growTum
let ctv = distleq(25,gtv) & brain
let grndTruthCTV = distleq(25,grndTruthGTV) & brain

print "dice-{case_name}-{hI}-{vI}" dice(gtv,grndTruthGTV)
print "sensitivity-{case_name}-{hI}-{vI}" sensitivity(gtv,grndTruthGTV)
print "specificity-{case_name}-{hI}-{vI}" specificity(gtv,grndTruthGTV)

print "dicectv-{case_name}-{hI}-{vI}" dice(ctv,grndTruthCTV)
print "sensitivityctv-{case_name}-{hI}-{vI}" sensitivity(ctv,grndTruthCTV)
print "specificityctv-{case_name}-{hI}-{vI}" specificity(ctv,grndTruthCTV)

save "{output_path}/{case_name}/veryIntense-{hI}-{vI}.nii.gz" veryIntense
save "{output_path}/{case_name}/gtv-{hI}-{vI}.nii.gz" gtv

"""


def gbm_tacas(dataset_path, case_name, hI, vIs):
    return (
        functions()
        + gbm_tacas_part1(dataset_path, case_name, hI)
        + "\n".join([gbm_tacas_part2(case_name, hI, vI) for vI in vIs])
    )


# %%
vl = VoxLogicA(voxlogica_path)

# %%

##%% Run the specification
result = {}

if os.path.isdir(brats_dataset):
    cases = [
        f
        for f in os.listdir(brats_dataset)
        if os.path.isdir(os.path.join(brats_dataset, f))
    ]
else:
    if do_run_voxlogica:
        raise f"Cannot run voxlogica because the dataset directory {brats_dataset} is not present"
    else:
        cases = pd.read_csv("BRATS2020_name_mapping.csv")[
            "BraTS_2020_subject_ID"
        ].to_list()

# %%

if do_run_voxlogica:
    os.makedirs(output_path, exist_ok=True)

    for case in cases:
        for hIThr in hIRange:
            label = f"{case}-hI-{hIThr}"
            success_path = os.path.join(output_path, f"{label}-results.json")
            vIs = set(range(vIRange.start, min(vIRange.stop, hIThr + 1)))
            existing_results = {}

            if os.path.exists(success_path):
                to_remove = set()
                json_data = {}
                with open(success_path) as f:
                    existing_results = json.load(f)
                for key in existing_results.keys():
                    to_remove.add(int(key.split("-")[3]))
                vIs = vIs.difference(to_remove)

            vIList = sorted(list(vIs))
            print(f"Running on case: {label} hI: {hIThr} vIs: {vIList}")

            if len(vIList) > 0:
                specification = gbm_tacas(brats_dataset, case, hIThr, vIList)
                result = vl.run(specification)
                print(result["log"])
                if result["exitcode"] == 0:
                    existing_results.update(result["values"])
                    with open(success_path, "w") as file:
                        json.dump(existing_results, file)
                else:
                    with open(f"{output_path}/error_log", "a") as file:
                        file.write(f"\n\nError on case {label}\n\n")
                        file.write(result["log"])


# %%  Transform all the json files in a single PANDAS table


def load_json_files(directory):
    data = []

    # loop through each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            # open the file and load the json data
            with open(os.path.join(directory, filename)) as f:
                json_data = json.load(f)
            # loop through each key,value pair in the json data
            for key, value in json_data.items():
                # split the key into its four fields
                fields = key.split("-")
                # create a dictionary with the fields and value
                row = {
                    "Metric": fields[0],
                    "Case": fields[1],
                    "HIThr": int(fields[2]),
                    "VIThr": int(fields[3]),
                    "Value": value,
                }
                # append the dictionary to the data list
                data.append(row)

    raw_df = pd.DataFrame(data)

    # create a pivot table with Case, HIThr, and VIThr as the index and Metric as the columns
    df = pd.pivot_table(
        raw_df, index=["Case", "HIThr", "VIThr"], columns="Metric", values="Value"
    )

    # reset the index to make Case, HIThr, and VIThr columns
    df = df.reset_index()

    # rename the columns
    df.columns.name = None
    return df


df = load_json_files(stats_output_path)

# %% Discarding the well-known bad cases

# First of all, find the BRATS2020 name of the cases, which now have a BRATS2019 name.

discarded_tacas19 = pd.read_csv("discarded_tacas19.csv")
BRATS2020_name_mapping = pd.read_csv("BRATS2020_name_mapping.csv")

#%%

discarded_BRATS2020 = pd.merge(
    BRATS2020_name_mapping, discarded_tacas19, on="BraTS_2017_subject_ID"
)["BraTS_2020_subject_ID"].rename("Case")

#%%

# Mieke: discarding cases from both csv lists (2025/03/14)
# discarded_20BraTS17 = pd.read_csv("discarded_first20BraTS17.csv")
# discarded_BRATS2020_minFirst20 = pd.merge(
#    BRATS2020_name_mapping, [discarded_tacas19,discarded_20BraTS17], on=["BraTS_2019_subject_ID","BraTS_2017_subject_ID"]
# )["BraTS_2020_subject_ID"].rename("Case")


# Next, remove the corresponding rows from the dataframe, obtaining a smaller dataframe
df_smaller = df[~df["Case"].isin(discarded_BRATS2020)]

# Mieke: Remove also first 20 cases (2025/03/14)
# df_smaller_min20 = df[~df["Case"].isin(discarded_BRATS2020_minFirst20)]

# %%

HGG = BRATS2020_name_mapping[BRATS2020_name_mapping["Grade"] == "HGG"]

LGG = BRATS2020_name_mapping[BRATS2020_name_mapping["Grade"] == "LGG"]

df_HGG = df[df["Case"].isin(HGG["BraTS_2020_subject_ID"])]
df_LGG = df[df["Case"].isin(LGG["BraTS_2020_subject_ID"])]

df_smaller_HGG = df_smaller[df_smaller["Case"].isin(HGG["BraTS_2020_subject_ID"])]
df_smaller_LGG = df_smaller[df_smaller["Case"].isin(LGG["BraTS_2020_subject_ID"])]

df_BRATS2019 = BRATS2020_name_mapping[
    BRATS2020_name_mapping["BraTS_2019_subject_ID"] >= "BraTS"
]
df_BRATS2017 = BRATS2020_name_mapping[
    BRATS2020_name_mapping["BraTS_2017_subject_ID"] >= "BraTS"
]

# df_HGG_BraTS19 = df_BRATS2019[df_BRATS2019["Case"].isin(BRATS2020_name_mapping["BraTS_2020_subject_ID"])]
# df_LGG_BraTS19 = df[df["Case"].isin(LGG["BraTS_2019_subject_ID"])]

# df_smaller_HGG_BraTS19 =df_smaller[df_smaller["Case"].isin(HGG["BraTS_2019_subject_ID"])]
# df_smaller_LGG_BraTS19 =df_smaller[df_smaller["Case"].isin(LGG["BraTS_2019_subject_ID"])]

# df_HGG_BraTS17 = df[df["Case"].isin(HGG["BraTS_2017_subject_ID"])]
# df_LGG_BraTS17 = df[df["Case"].isin(LGG["BraTS_2017_subject_ID"])]

# df_smaller_HGG_BraTS17 =df_smaller[df_smaller["Case"].isin(HGG["BraTS_2017_subject_ID"])]
# df_smaller_LGG_BraTS17 =df_smaller[df_smaller["Case"].isin(LGG["BraTS_2017_subject_ID"])]


# %% Compute the pair (HIThr,VIThr) that maximises the overall dice


def find_best_average(df, metric):
    # group the table by HiThr and ViThr, and calculate the average dice for each group
    grouped = df.groupby(["HIThr", "VIThr"])

    mean_dice = grouped[metric].mean()
    stdev_dice = grouped[metric].std()

    # find the optimal pair of HIThr and VIThr
    optimal = mean_dice.idxmax()
    return {
        "HIThr": int(optimal[0]),
        "VIThr": int(optimal[1]),
        metric: float(mean_dice.loc[optimal]),
        "stddev": float(stdev_dice.loc[optimal]),
    }


# %% Now find the optimal individual thresholds


def find_best_individual(df, metric):
    max_dice_rows_intermediate = df.groupby("Case").apply(
        lambda group: group.loc[group[metric].idxmax()]
    )
    max_dice_rows = max_dice_rows_intermediate[["Case", "HIThr", "VIThr", metric]]
    max_dice_rows.reset_index(drop=True, inplace=True)
    return {
        "average": max_dice_rows[metric].mean(),
        "stDev": max_dice_rows[metric].std(),
        "minimum": max_dice_rows[metric].min(),
        "maximum": max_dice_rows[metric].max(),
        "table": max_dice_rows,
    }


# %% Histogram of a metric


def plot_hist(df, metric, bins=20):
    plt.hist(df[metric], bins=bins, color="blue", alpha=0.7, range=(0, 1))
    plt.xlabel(metric)
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {metric} distribution")
    plt.grid(True)


def plotHeatmap(
    df,
    metric,
    cmap="RdBu",
    bins=(hIRange.stop - hIRange.start, vIRange.stop - vIRange.start),
    vmax=30,
):
    H, xedges, yedges = np.histogram2d(df["HIThr"], df["VIThr"], bins=bins)

    # Create the heatmap
    # plt.imshow(H.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],vmin=0, vmax=vmax, cmap=cmap, aspect='auto', origin='lower')
    plt.imshow(
        H.T,
        vmin=0,
        extent=(
            hIRange.start - 0.5,
            hIRange.stop - 0.5,
            vIRange.start - 1.5,
            vIRange.stop - 1.5,
        ),
        vmax=vmax,
        cmap=cmap,
        aspect="auto",
        origin="lower",
    )
    plt.colorbar(label="Count")
    plt.xlabel("HIThr")
    plt.ylabel("VIThr")
    plt.title(f"2D Histogram (Heatmap) of HIThr vs. VIThr for {metric}")
    plt.grid(True)

    # Show the heatmap
    plt.show()


# %%
