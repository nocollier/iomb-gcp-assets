from pathlib import Path

import ilamb3
import pandas as pd
from ilamb3.run import run_study
from ilamb3.meta import generate_dashboard_page

root = Path("/albedo/work/user/srmoha001/ILAMB_sample/DATA/")
ref_datasets = []
for dirpath, _, files in root.walk():
    for fname in files:
        if not fname.endswith(".nc"):
            continue
        path = str((dirpath / fname).absolute())
        ref_datasets.append({"key": "/".join(path.split("/")[-2:]), "path": path})
ref_datasets = pd.DataFrame(ref_datasets).set_index("key")
print(ref_datasets)


root = Path("./GCB_CLIM/")
df_datasets = []
for dirpath, _, files in root.walk(follow_symlinks=True):
    for fname in files:
        if not fname.endswith(".nc"):
            continue
        path = str((dirpath / fname).absolute())
        df_datasets.append(
            {
                "mip_era": "",
                "activity_id": path.split("/")[-4],
                "institution_id": "",
                "source_id": path.split("/")[-3],
                "experiment_id": "historical",
                "member_id": "",
                "table_id": "Omon",
                "variable_id": path.split("/")[-2],
                "grid_label": "gn",
                "path": path,
            }
        )
df_datasets = pd.DataFrame(df_datasets)

ilamb3.conf.set(
    model_name_facets=["source_id", "activity_id"],
    comparison_groupby=["activity_id", "source_id"],
    use_cached_results=True,
    debug_mode=True,
)
print(df_datasets)
output = Path("_build_GCB")

run_study("gcb.yaml", df_datasets, ref_datasets, output_path=output)
generate_dashboard_page(output, "GCB2023 GCB2024")
