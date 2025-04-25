from pathlib import Path

import ilamb3
import pandas as pd
from ilamb3.run import run_study

root = Path("./GCB2023")
df_datasets = []
for dirpath, _, files in root.walk():
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

ilamb3.conf.set(model_name_facets=["source_id", "activity_id"])

run_study("gcb.yaml", df_datasets)
