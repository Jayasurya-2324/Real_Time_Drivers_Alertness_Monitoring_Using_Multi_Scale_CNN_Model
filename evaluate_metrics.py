# auto_label_and_evaluate.py
"""
Auto-label + evaluate for your project.

Place this file in the same directory as output_data.csv and run:
    python auto_label_and_evaluate.py

What it does:
 - tries to infer true_drowsy / true_distracted from 'status' column or heuristics
 - if insufficient automatic labels found, falls back to interactive labeling
 - computes accuracy, precision, recall, f1 for both components
 - saves updated CSV and metrics files
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

CSV_IN = "output_data.csv"
CSV_OUT = "updated_output_data.csv"
METRICS_OUT = "metrics_report.csv"
CM_OUT = "confusion_matrices.csv"

# column names in your CSV (change if different)
COL_STATUS = "status"
COL_PRED_DROWSY = "drowsy"
COL_PRED_DISTRACT = "distracted"
COL_EAR = "ear"
FRAME_COL_CANDIDATES = ["frame_index","frame","frame_no","frameIdx"]

TRUE_DROWSY = "true_drowsy"
TRUE_DISTRACT = "true_distracted"

AUTO_EAR_THRESHOLD = 0.20   # heuristic EAR threshold for closed eyes (tune if needed)

def load_csv(path):
    if not os.path.exists(path):
        print(f"[ERROR] {path} not found in current folder: {os.getcwd()}")
        sys.exit(1)
    df = pd.read_csv(path)
    return df

def try_auto_label(df):
    df = df.copy()
    created = 0
    # ensure true cols exist as NaN
    if TRUE_DROWSY not in df.columns:
        df[TRUE_DROWSY] = np.nan
    if TRUE_DISTRACT not in df.columns:
        df[TRUE_DISTRACT] = np.nan

    # 1) Use status text if available
    if COL_STATUS in df.columns:
        s = df[COL_STATUS].astype(str).str.lower()
        # map obvious keywords
        drowsy_mask = s.str.contains("drowsy|sleep|sleepy|yawn|microsleep|eyes closed|eyelid", na=False)
        distract_mask = s.str.contains("distract|phone|looking|lookaway|text|mobile|hand|talking", na=False)
        if drowsy_mask.any():
            df.loc[drowsy_mask, TRUE_DROWSY] = 1
            created += drowsy_mask.sum()
        if distract_mask.any():
            df.loc[distract_mask, TRUE_DISTRACT] = 1
            created += distract_mask.sum()

    # 2) If no status, try predicted columns when they are binary: copy predicted -> true (best-effort)
    # But only do this if predicted column looks binary and true_* are still empty
    if pd.isna(df[TRUE_DROWSY]).all() and COL_PRED_DROWSY in df.columns:
        unique_vals = df[COL_PRED_DROWSY].dropna().unique()
        # if it's clearly binary (0/1 or True/False)
        if set(map(lambda x: str(x), unique_vals)).issubset(set(["0","1","0.0","1.0","true","false","True","False"])):
            df[TRUE_DROWSY] = df[COL_PRED_DROWSY].astype(int)
            created += df[TRUE_DROWSY].notna().sum()

    if pd.isna(df[TRUE_DISTRACT]).all() and COL_PRED_DISTRACT in df.columns:
        unique_vals = df[COL_PRED_DISTRACT].dropna().unique()
        if set(map(lambda x: str(x), unique_vals)).issubset(set(["0","1","0.0","1.0","true","false","True","False"])):
            df[TRUE_DISTRACT] = df[COL_PRED_DISTRACT].astype(int)
            created += df[TRUE_DISTRACT].notna().sum()

    # 3) Heuristic using EAR if present and still no true_drowsy labels:
    if pd.isna(df[TRUE_DROWSY]).all() and COL_EAR in df.columns:
        try:
            ear_vals = pd.to_numeric(df[COL_EAR], errors='coerce')
            closed_mask = ear_vals < AUTO_EAR_THRESHOLD
            if closed_mask.any():
                # mark only a subset so we don't overwrite everything: mark only when predicted was drowsy or status none
                if COL_PRED_DROWSY in df.columns:
                    selected = closed_mask & (pd.to_numeric(df[COL_PRED_DROWSY], errors='coerce').fillna(0) > 0)
                else:
                    selected = closed_mask
                df.loc[selected, TRUE_DROWSY] = 1
                created += selected.sum()
        except Exception:
            pass

    return df, created

def interactive_labeling(df):
    # interactive labeling same as previous script (simple)
    if TRUE_DROWSY not in df.columns:
        df[TRUE_DROWSY] = np.nan
    if TRUE_DISTRACT not in df.columns:
        df[TRUE_DISTRACT] = np.nan

    # detect frame column for reference
    frame_col = None
    for c in FRAME_COL_CANDIDATES:
        if c in df.columns:
            frame_col = c
            break
    print("Detected frame column:", frame_col if frame_col else "None; will use row index for ranges")

    def parse_spec(spec_str):
        specs = []
        for part in spec_str.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                raise ValueError("Spec missing ':' -> " + part)
            rng, lab = part.split(":")
            lab = int(lab)
            if "-" in rng:
                s,e = rng.split("-")
                specs.append((int(s), int(e), lab))
            else:
                i = int(rng)
                specs.append((i, i, lab))
        return specs

    def apply_specs(specs, target_col):
        for s,e,label in specs:
            if frame_col:
                mask = (df[frame_col] >= s) & (df[frame_col] <= e)
            else:
                mask = (df.index >= s) & (df.index <= e)
            df.loc[mask, target_col] = label
            print(f"Applied {label} to {mask.sum()} rows for {target_col}")

    while True:
        print("\nActions: 1) Label drowsy  2) Label distracted  3) Show unlabeled  4) Save & eval  5) Quit")
        ch = input("Choice (1-5): ").strip()
        if ch == "1":
            spec = input("Enter drowsy specs (e.g. 10-50:1,100:0): ").strip()
            specs = parse_spec(spec)
            apply_specs(specs, TRUE_DROWSY)
        elif ch == "2":
            spec = input("Enter distracted specs (e.g. 30-70:1): ").strip()
            specs = parse_spec(spec)
            apply_specs(specs, TRUE_DISTRACT)
        elif ch == "3":
            print("Unlabeled counts:", df[TRUE_DROWSY].isna().sum(), "drowsy unlabeled,", df[TRUE_DISTRACT].isna().sum(), "distracted unlabeled")
            print("First 10 rows:")
            print(df.head(10).to_string(index=False))
        elif ch == "4":
            print("Saving and computing metrics...")
            df.to_csv(CSV_OUT, index=False)
            compute_metrics_and_save(df)
            print("Saved outputs. Exiting.")
            break
        elif ch == "5":
            print("Exiting without saving.")
            break
        else:
            print("Invalid choice.")

def compute_metrics_and_save(df):
    rows = []
    cms = {}
    for comp, pred_col, true_col in [
        ("drowsy", COL_PRED_DROWSY, TRUE_DROWSY),
        ("distracted", COL_PRED_DISTRACT, TRUE_DISTRACT)
    ]:
        if true_col not in df.columns or df[true_col].dropna().empty:
            print(f"[WARN] No true labels for {comp} -> skipping")
            continue
        sub = df[[pred_col, true_col]].dropna()
        # ensure integers
        y_true = sub[true_col].astype(int).values
        y_pred = pd.to_numeric(sub[pred_col], errors='coerce').fillna(0).astype(int).values
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='binary', zero_division=0)
        rec = recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        rows.append({"component":comp, "n":len(y_true), "accuracy":acc, "precision":prec, "recall":rec, "f1":f1})
        cms[comp] = cm
        print(f"\nMetrics for {comp}: n={len(y_true)} acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")
        print("Confusion matrix (true rows [0,1], pred cols [0,1]):")
        print(cm)
        print(classification_report(y_true, y_pred, zero_division=0))

    if rows:
        pd.DataFrame(rows).to_csv(METRICS_OUT, index=False)
        print(f"Saved metrics summary to {METRICS_OUT}")
    if cms:
        cms_rows = []
        for comp, cm in cms.items():
            cms_rows.append({"component":comp, "tn":int(cm[0,0]), "fp":int(cm[0,1]), "fn":int(cm[1,0]), "tp":int(cm[1,1])})
        pd.DataFrame(cms_rows).to_csv(CM_OUT, index=False)
        print(f"Saved confusion matrices to {CM_OUT}")

def main():
    df = load_csv(CSV_IN)
    print("Loaded", len(df), "rows. Columns:", df.columns.tolist())
    # try automatic labeling
    df_auto, created = try_auto_label(df)
    print(f"[AUTO] Created/filled {created} true-label cells via heuristics.")
    # show head
    print("\nSample rows (with predicted + auto-true):")
    cols_show = [c for c in [FRAME_COL_CANDIDATES[0]] + [COL_STATUS, COL_PRED_DROWSY, COL_PRED_DISTRACT, COL_EAR, TRUE_DROWSY, TRUE_DISTRACT] if c in df_auto.columns]
    print(df_auto.head(10)[cols_show].to_string(index=False))
    # if no labels created, go interactive
    if created == 0:
        print("\nNo automatic labels created. Launching interactive labeling helper.")
        interactive_labeling(df_auto)
    else:
        # if some labels exist, ask user whether to proceed or annotate more
        while True:
            print("\nActions: 1) Compute metrics now  2) Annotate more  3) Save and exit without eval  4) Quit")
            ch = input("Choice (1-4): ").strip()
            if ch == "1":
                df_auto.to_csv(CSV_OUT, index=False)
                compute_metrics_and_save(df_auto)
                break
            elif ch == "2":
                interactive_labeling(df_auto)
                break
            elif ch == "3":
                df_auto.to_csv(CSV_OUT, index=False)
                print(f"Saved file {CSV_OUT}. Exiting.")
                break
            elif ch == "4":
                print("Exiting without save.")
                break
            else:
                print("Invalid choice.")

if __name__ == "__main__":
    main()