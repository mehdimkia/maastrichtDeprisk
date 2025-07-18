import joblib
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

BASE = Path("/Volumes/education/DMS_744_S_Mirkialangaroodi/Databases")
train = pd.read_parquet(BASE / "Week7_train.parquet")
test  = pd.read_parquet(BASE / "Week7_test.parquet")

y_train = train.pop("LD_PHQ9depr_event")
y_test  = test.pop("LD_PHQ9depr_event")

cat_cols = train.select_dtypes("category").columns
num_cols = train.select_dtypes("number").columns

pre = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
    remainder="passthrough"
)

clf = LogisticRegression(max_iter=1000, class_weight="balanced")

pipe = Pipeline([("pre", pre), ("clf", clf)])
pipe.fit(train, y_train)

probs = pipe.predict_proba(test)[:, 1]
print("AUROC:", round(roc_auc_score(y_test, probs), 3))
print("AUPRC:", round(average_precision_score(y_test, probs), 3))


joblib.dump(pipe, BASE / "logreg_model.joblib")
