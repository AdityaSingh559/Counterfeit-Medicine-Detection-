"""Does drug matching help? Compare CNN alone vs. similarity-to-genuine-references vs. both fused.

    python src/evaluate_drugmatch.py [--pred-tag final]

Pills only (boxes have no drug labels). The fusion (logistic regression on logit(P_cnn), similarity,
drug confidence) is fitted on VALIDATION pills and scored once on TEST pills.
Operating points are chosen on validation: threshold at which <=10% of genuine pills are flagged.
Writes outputs/metrics/drugmatch_report.json and models/fusion.json (coefficients for decide.py).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from drug_match import DrugMatcher  # noqa: E402

METRICS = config.OUTPUTS_DIR / "metrics"
FPR_TARGET = 0.10


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def thr_at_fpr(score, is_fake, fpr=FPR_TARGET):
    """Smallest threshold with <= fpr of genuine items scoring above it."""
    g = np.sort(score[is_fake == 0])
    return float(g[int(np.ceil((1 - fpr) * len(g))) - 1])


def recall_at(score, is_fake, thr):
    return float((score[is_fake == 1] > thr).mean())


def load_split(split, tag, matcher):
    suffix = f"_{tag}" if tag else ""
    pred = pd.read_csv(METRICS / f"{split}{suffix}_predictions.csv")
    lab = pd.read_csv(config.DATA_DIR / "labels.csv", keep_default_na=False).set_index("filename")
    d = pred.join(lab[["drug_id", "drug_name"]], on="filename")
    d = d[d.kind == "medi"].reset_index(drop=True)
    imgs = [Image.open(config.DATA_DIR / f).convert("RGB") for f in d.filename]
    a = pd.DataFrame(matcher.analyse(imgs))
    d["pred_drug"], d["drug_conf"], d["sim"] = a.drug_id, a.drug_conf, a.similarity
    d["is_fake"] = (d.label == "counterfeit").astype(int)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-tag", default="")
    args = ap.parse_args()

    m = DrugMatcher()
    val, test = load_split("val", args.pred_tag, m), load_split("test", args.pred_tag, m)
    known = set(m.classes)
    rep = {"n_val_pills": len(val), "n_test_pills": len(test)}

    # 1. Drug identification (genuine photos, then the fakes made from them).
    for name, df in (("val", val), ("test", test)):
        k = df[df.drug_id.isin(known)]
        rep[f"drug_id_top1_{name}"] = {
            "genuine": float((k[k.is_fake == 0].pred_drug == k[k.is_fake == 0].drug_id).mean()),
            "counterfeit": float((k[k.is_fake == 1].pred_drug == k[k.is_fake == 1].drug_id).mean()),
            "n_known_drug_rows": int(len(k)), "n_unknown_drug_rows": int(len(df) - len(k))}

    # 2. Detection power of each signal (test AUC; higher score = more likely fake).
    def feats(df):
        return np.c_[logit(df.p_cf.to_numpy()), df["sim"].to_numpy(), df.drug_conf.to_numpy()]

    Xv, Xt = feats(val), feats(test)
    sc = StandardScaler().fit(Xv)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(Xv), val.is_fake)
    scores = {"cnn_only": (val.p_cf.to_numpy(), test.p_cf.to_numpy()),
              "drug_similarity_only": (-val["sim"].to_numpy(), -test["sim"].to_numpy()),
              "fused_cnn+drug": (lr.decision_function(sc.transform(Xv)), lr.decision_function(sc.transform(Xt)))}
    rep["signals"] = {}
    for name, (sv, st) in scores.items():
        thr = thr_at_fpr(sv, val.is_fake.to_numpy())
        rep["signals"][name] = {
            "test_auc": float(roc_auc_score(test.is_fake, st)),
            "threshold_from_val_at_10pct_false_alarm": thr,
            "test_false_alarm_rate": float((st[test.is_fake.to_numpy() == 0] > thr).mean()),
            "test_counterfeit_recall": recall_at(st, test.is_fake.to_numpy(), thr)}
    rep["fusion_coef_[logit_cnn,similarity,drug_conf]"] = lr.coef_[0].tolist()

    out = METRICS / "drugmatch_report.json"
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    (config.ROOT / "models" / "fusion.json").write_text(json.dumps({
        "features": ["logit_p_cnn", "similarity", "drug_conf"], "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
        "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
        "threshold_10pct_false_alarm": rep["signals"]["fused_cnn+drug"]["threshold_from_val_at_10pct_false_alarm"]},
        indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
