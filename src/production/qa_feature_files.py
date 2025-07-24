from pathlib import Path
import pandas as pd

BASE = Path("/Volumes/education/DMS_744_S_Mirkialangaroodi/Databases")
for name in ("Week7_features.parquet", "Week7_train.parquet", "Week7_test.parquet"):
    df = pd.read_parquet(BASE / name)
    print(f"\n{name}: shape={df.shape}")

    # 1. Are the dtypes as expected?
    print(df.dtypes.head(10))

    # 2. Check target prevalence in train vs. test
    if "LD_PHQ9depr_event" in df.columns:
        print("event rate:", df["LD_PHQ9depr_event"].mean().round(3))

    # 3. Make sure standstep_q4 is categorical
    if "standstep_q4" in df.columns:
        print("standstep_q4 dtype:", df["standstep_q4"].dtype)

#check The number of cases too
for df_name in ("Week7_train.parquet", "Week7_test.parquet"):
    df = pd.read_parquet(BASE / df_name)
    counts = df["LD_PHQ9depr_event"].value_counts()
    print(df_name, counts.to_dict())