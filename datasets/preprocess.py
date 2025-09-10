#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-processing for MO-GRPO-Med
Generates:
  <PREFIX>_train_sft.csv
  <PREFIX>_train_rl.csv
  <PREFIX>_dev.csv
  <PREFIX>_test.csv
Splits are subject-level isolated.
"""
import argparse, os, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split

# ---------- Column sets ----------
BCH_INPUT = [
    "note_id","subject_id","hadm_id","age_at_charttime","sex","race",
    "chief_complaint","history_present_illness","past_medical_history",
    "social_family_history","allergies","medications_admission",
    "admission_type","admission_location","arrival_transport","service","insurance",
    "physical_examination","major_procedures","acute_issues","chronic_issues","pertinent_results"
]
BCH_TARGET = ["brief_hospital_course"]

DI_INPUT  = BCH_INPUT + [
    "brief_hospital_course","discharge_diagnosis",
    "discharge_medications","discharge_condition","discharge_disposition"
]
DI_TARGET = ["discharge_instructions"]

RAW_CSV = "raw/MIMIC-IV-Ext_iDS_with_PR.csv"
OUT_DIR = "processed"


# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser("Dataset splitter (patient-level + row-level caps)")
    ap.add_argument("--mode", choices=["BCH","DI"], required=True,
                    help="Task type: BCH or DI")
    ap.add_argument("--split", default="9,0.5,0.5",
                    help="train,dev,test patient ratios (default 90/5/5)")
    ap.add_argument("--rl_ratio", type=float, default=0.2,
                    help="Fraction of train_sft reused as train_rl (default 0.2)")
    # row-level caps
    ap.add_argument("--max_dev",  type=int, default=1500,
                    help="Max rows in dev set   (0 = unlimited)")
    ap.add_argument("--max_test", type=int, default=1500,
                    help="Max rows in test set  (0 = unlimited)")
    # patient-level caps (optional)
    ap.add_argument("--max_dev_patient",  type=int, default=0,
                    help="Max patients in dev   (0 = unlimited)")
    ap.add_argument("--max_test_patient", type=int, default=0,
                    help="Max patients in test  (0 = unlimited)")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


# ---------- helpers ----------
def sample_patients(df: pd.DataFrame, max_patients: int, seed: int):
    """limit by unique subject_id first"""
    if max_patients and df["subject_id"].nunique() > max_patients:
        keep_ids = np.random.RandomState(seed).choice(
            df["subject_id"].unique(), size=max_patients, replace=False
        )
        df = df[df["subject_id"].isin(keep_ids)]
    return df


def sample_rows(df: pd.DataFrame, max_rows: int, seed: int):
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed)
    return df.reset_index(drop=True)


def subset(df: pd.DataFrame, ids):
    return df[df["subject_id"].isin(ids)].reset_index(drop=True)


# ---------- main ----------
def main():
    args = parse_args()

    # ratios
    tr_p, dv_p, te_p = map(float, args.split.split(","))
    s = tr_p + dv_p + te_p
    tr_ratio, dv_ratio, te_ratio = tr_p/s, dv_p/s, te_p/s

    # choose columns
    if args.mode == "BCH":
        in_cols, tgt_cols, prefix = BCH_INPUT, BCH_TARGET, "BCH"
    else:
        in_cols, tgt_cols, prefix = DI_INPUT, DI_TARGET, "DI"

    print(f"Loading {RAW_CSV} ...")
    df = pd.read_csv(RAW_CSV, usecols=in_cols+tgt_cols, dtype=str)

    # patient-level split
    patients = df["subject_id"].unique()
    train_ids, tmp_ids = train_test_split(
        patients, test_size=1-tr_ratio, random_state=args.seed
    )
    dev_size = dv_ratio / (dv_ratio + te_ratio)
    dev_ids, test_ids = train_test_split(
        tmp_ids, test_size=1-dev_size, random_state=args.seed
    )

    train_sft = subset(df, train_ids)
    dev_df    = subset(df, dev_ids)
    test_df   = subset(df, test_ids)

    # optional caps
    dev_df  = sample_patients(dev_df,  args.max_dev_patient,  args.seed)
    test_df = sample_patients(test_df, args.max_test_patient, args.seed)
    dev_df  = sample_rows(dev_df,     args.max_dev,  args.seed)
    test_df = sample_rows(test_df,    args.max_test, args.seed)

    # train_rl
    train_rl = (train_sft.sample(frac=args.rl_ratio, random_state=args.seed)
                if args.rl_ratio > 0 else
                pd.DataFrame(columns=train_sft.columns))

    # save
    os.makedirs(OUT_DIR, exist_ok=True)
    def save(df_, name):
        path = os.path.join(OUT_DIR, f"{prefix}_{name}.csv")
        df_.to_csv(path, index=False, na_rep="")
        print(f"{name:<9}: {len(df_):>6} rows  -> {path}")

    save(train_sft, "train_sft")
    save(train_rl , "train_rl")
    save(dev_df   , "dev")
    save(test_df  , "test")


if __name__ == "__main__":
    main()
