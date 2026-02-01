from __future__ import annotations
import os, json

# --- BASE DIRECTORY FIX ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils import clean_feature_names, build_feature_groups

SEED = 42
np.random.seed(SEED)

DATA_DIR = os.path.join(ROOT_DIR, "UCI HAR Dataset")
OUT_DIR = os.path.join(ROOT_DIR, "outputs", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# --- Yardımcı fonksiyonlar ---
def _read_lines(path: str):
    return pd.read_csv(path, header=None, delim_whitespace=True)

def load_ucihar(data_dir: str) -> pd.DataFrame:
    features = _read_lines(os.path.join(data_dir, "features.txt"))
    feature_names = clean_feature_names(features[1].tolist())
    act_labels = _read_lines(os.path.join(data_dir, "activity_labels.txt"))
    act_map = dict(zip(act_labels[0], act_labels[1]))

    # train
    X_train = _read_lines(os.path.join(data_dir, "train", "X_train.txt"))
    y_train = _read_lines(os.path.join(data_dir, "train", "y_train.txt"))[0]
    subj_train = _read_lines(os.path.join(data_dir, "train", "subject_train.txt"))[0]
    X_train.columns = feature_names
    df_train = X_train.copy()
    df_train["activity_id"] = y_train
    df_train["activity"] = df_train["activity_id"].map(act_map)
    df_train["subject"] = subj_train
    df_train["split"] = "train"

    # test
    X_test = _read_lines(os.path.join(data_dir, "test", "X_test.txt"))
    y_test = _read_lines(os.path.join(data_dir, "test", "y_test.txt"))[0]
    subj_test = _read_lines(os.path.join(data_dir, "test", "subject_test.txt"))[0]
    X_test.columns = feature_names
    df_test = X_test.copy()
    df_test["activity_id"] = y_test
    df_test["activity"] = df_test["activity_id"].map(act_map)
    df_test["subject"] = subj_test
    df_test["split"] = "test"

    return pd.concat([df_train, df_test], ignore_index=True)

def stratified_split(df: pd.DataFrame, ratio=(70,10,20), random_state=SEED):
    assert sum(ratio) == 100
    tr, va, te = ratio
    X = df.drop(columns=["activity_id", "activity", "subject", "split"])
    y = df["activity"]

    # önce train ve temp
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=(100-tr)/100, stratify=y, random_state=random_state
    )
    val_ratio = va / (va + te)
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=(1 - val_ratio), stratify=y_tmp, random_state=random_state
    )
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)

def scale_standard(Xtr, Xva, Xte):
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte)
    return scaler, Xtr_s, Xva_s, Xte_s

def save_npz(prefix, Xtr, ytr, Xva, yva, Xte, yte, cols):
    path = os.path.join(OUT_DIR, f"{prefix}.npz")
    np.savez_compressed(
        path,
        X_train=Xtr, y_train=ytr.values,
        X_val=Xva,  y_val=yva.values,
        X_test=Xte, y_test=yte.values,
        feature_names=np.array(cols)
    )
    print(f"[OK] Saved {path}")
    return path

def main():
    df = load_ucihar(DATA_DIR)
    feature_cols = [c for c in df.columns if c not in ("activity_id","activity","subject","split")]
    groups = build_feature_groups(feature_cols)

    # kaydet
    with open(os.path.join(OUT_DIR, "feature_groups.json"), "w") as f:
        json.dump(groups, f, indent=2)

    # --- varyant A: 70/10/20 ---
    (Xa_tr, ya_tr), (Xa_va, ya_va), (Xa_te, ya_te) = stratified_split(df, (70,10,20))
    _, Xa_trs, Xa_vas, Xa_tes = scale_standard(Xa_tr, Xa_va, Xa_te)
    save_npz("ucihar_70_10_20_std", Xa_trs, ya_tr, Xa_vas, ya_va, Xa_tes, ya_te, Xa_tr.columns)

    # --- varyant B: 80/10/10 ---
    (Xb_tr, yb_tr), (Xb_va, yb_va), (Xb_te, yb_te) = stratified_split(df, (80,10,10))
    _, Xb_trs, Xb_vas, Xb_tes = scale_standard(Xb_tr, Xb_va, Xb_te)
    save_npz("ucihar_80_10_10_std", Xb_trs, yb_tr, Xb_vas, yb_va, Xb_tes, yb_te, Xb_tr.columns)

if __name__ == "__main__":
    main()
