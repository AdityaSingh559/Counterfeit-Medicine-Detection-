"""Streamlit demo: Explainable Counterfeit Medicine Detection (prototype).

    streamlit run app.py

Prototype scope: classifier + Grad-CAM heatmap + approve/flag/escalate + drug matching + results.
The blockchain audit trail (Phases 7-8) is NOT built yet; this page shows the two SHA-256 fingerprints
that will be recorded there, and says clearly that nothing is logged.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import config  # noqa: E402

METRICS = config.OUTPUTS_DIR / "metrics"
st.set_page_config(page_title="Counterfeit Medicine Inspector", page_icon="💊", layout="wide")


# ----------------------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner="Loading classifier ...")
def get_classifier():
    from explain import load_classifier
    return load_classifier()


@st.cache_resource(show_spinner="Loading drug matcher ...")
def get_drug_matcher():
    from drug_match import DrugMatcher
    return DrugMatcher()


@st.cache_data
def get_labels():
    return pd.read_csv(config.DATA_DIR / "labels.csv", keep_default_na=False)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def decide(p, t_low, t_high):
    if p >= t_high:
        return "FLAG", "counterfeit"
    if p <= t_low:
        return "APPROVE", "genuine"
    return "ESCALATE", "uncertain"


# ----------------------------------------------------------------------------- sidebar
if not config.MODEL_PATH.exists():
    st.error(f"Model file not found: {config.MODEL_PATH}. Train first: `python src/train.py`.")
    st.stop()

model, ck, device = get_classifier()
with st.sidebar:
    st.header("System")
    st.write(f"Model: MobileNetV2 ({ck['img_size']}px, trained epoch {ck['epoch']})")
    st.write(f"Running on: `{device}`")
    st.markdown("Blockchain audit trail: :orange[**not built yet**] (Phases 7-8)")
    st.subheader("Decision thresholds")
    t_low, t_high = st.slider("APPROVE below / FLAG above  (P counterfeit)", 0.0, 1.0,
                              (config.T_LOW, config.T_HIGH), 0.01)
    st.caption("Between the two thresholds the case is ESCALATED to a human. "
               "Missing a fake is worse than a false alarm, so the APPROVE bound should be strict.")
    st.divider()
    st.caption("Prototype trained on SIMULATED counterfeits. It is a screening aid, not a lab test.")

st.title("💊 Counterfeit Medicine Inspector")
st.caption("Explainable detection prototype: verdict + heatmap + drug check. Results are for demonstration only.")
tab_check, tab_results, tab_status = st.tabs(["Check medicine", "Model results", "Project status"])


# ----------------------------------------------------------------------------- tab 1: check
with tab_check:
    left, right = st.columns([3, 2])
    with left:
        up = st.file_uploader("Upload a photo of a pill or a medicine box (JPG/PNG)", type=["jpg", "jpeg", "png"])
    with right:
        st.write("Or try an image from the held-out test pool (never seen in training):")
        if st.button("🎲 Pick a random test image"):
            test = get_labels().query("split == 'test'")
            st.session_state["sample"] = test.sample(1).iloc[0].to_dict()
            st.session_state["reveal"] = False

    image_bytes, source_label, truth = None, None, None
    if up is not None:
        image_bytes, source_label = up.getvalue(), f"uploaded: {up.name}"
    elif "sample" in st.session_state:
        row = st.session_state["sample"]
        image_bytes = (config.DATA_DIR / row["filename"]).read_bytes()
        source_label, truth = f"test pool: {row['filename']}", row

    if image_bytes is None:
        st.info("Upload an image or pick a random test image to begin.")
    else:
        try:
            import io
            pil = Image.open(io.BytesIO(image_bytes))
            pil.load()
        except (UnidentifiedImageError, OSError):
            st.error("Invalid image: this file could not be read as a JPG/PNG. Nothing was analysed.")
            st.stop()

        from explain import png_bytes, predict_with_heatmap
        t0 = time.time()
        with st.spinner("Analysing ..."):
            p_cf, overlay, cam = predict_with_heatmap(pil, model, ck, device)
        elapsed = time.time() - t0
        action, verdict = decide(p_cf, t_low, t_high)
        heat_png = png_bytes(overlay)

        c1, c2, c3 = st.columns([2, 2, 2])
        c1.image(pil.convert("RGB"), caption=f"Original ({source_label})", width="stretch")
        c2.image(overlay, caption="Heatmap: where the model saw evidence of a fake (warm = strong). A hint, not proof.", width="stretch")
        with c3:
            msg = {"FLAG": ("error", "🚩 FLAG: looks counterfeit"),
                   "APPROVE": ("success", "✅ APPROVE: looks genuine"),
                   "ESCALATE": ("warning", "⚠️ ESCALATE: not sure, send to a human / lab")}[action]
            getattr(st, msg[0])(msg[1])
            st.metric("Counterfeit probability", f"{p_cf:.1%}")
            st.progress(min(1.0, max(0.0, p_cf)))
            st.caption(f"Analysis time: {elapsed:.2f}s on {device}")
            if action == "FLAG":
                st.write("The model is confident this differs from genuine examples. Warm (red/yellow) areas on the "
                         "heatmap are where it found the strongest evidence.")
            elif action == "APPROVE":
                st.write("Nothing in this image strongly resembles the fake examples the model learned from. "
                         "This does **not** prove the medicine is real.")
            else:
                st.write("The evidence is mixed, so the system does not decide on its own. A person should inspect it.")

        st.subheader("Fingerprints that the blockchain will record")
        h1, h2 = st.columns(2)
        h1.code(f"image   SHA-256: {sha256(image_bytes)}", language="text")
        h2.code(f"heatmap SHA-256: {sha256(heat_png)}", language="text")
        st.warning("Not recorded on a blockchain yet: the smart contract and chain link (Phases 7-8) are still to be built. "
                   "Today these fingerprints are only displayed.")

        # Drug matching (pills only)
        st.subheader("Drug matching (loose pills only)")
        is_pill = st.checkbox("This is a loose pill (not a box)", value=bool(truth and truth.get("kind") == "medi"))
        if is_pill:
            if not (ROOT / "models" / "drug_model.pt").exists():
                st.info("Drug model not found. Run `python src/train_drugid.py`.")
            else:
                dm = get_drug_matcher().analyse([pil])[0]
                a, b, c = st.columns(3)
                a.metric("Most likely drug", dm["drug_name"])
                b.metric("Identification confidence", f"{dm['drug_conf']:.0%}")
                c.metric("Similarity to genuine references", f"{dm['similarity']:.2f}")
                st.caption("Only 23 drugs from the MEDISEG data can be recognised. Low similarity means the pill does not "
                           "look like genuine examples of that drug (weak extra evidence; see Model results).")
                if dm["drug_conf"] < 0.5:
                    st.info("Low confidence: this may be a drug the system was never trained on.")

        if truth is not None:
            st.divider()
            if st.checkbox("Reveal the true label from the answer key (for the demo)", key="reveal"):
                ok = (verdict == truth["label"]) or (action == "ESCALATE")
                st.write(f"True label: **{truth['label']}** ({truth['kind'] == 'medi' and 'pill' or 'box'}). "
                         + ("Handled correctly." if ok else "The model got this one wrong."))
                if truth["label"] == "counterfeit":
                    man = pd.read_csv(config.MANIFEST_PATH, keep_default_na=False).set_index("fake_filename")
                    if truth["filename"] in man.index:
                        m = man.loc[truth["filename"]]
                        st.caption(f"Simulated edits applied: {m['alterations'].replace(';', ', ')} "
                                   f"(strength: {m['strength']}).")


# ----------------------------------------------------------------------------- tab 2: results
with tab_results:
    st.subheader("How well does it work?")
    st.caption("Scored once on 298 held-out test images (149 genuine + 149 simulated counterfeit). "
               "Counterfeits are simulated, so this is not real-world accuracy.")
    t = read_json(METRICS / "test_final_metrics.json")
    if t is None:
        st.info("No metrics yet. Run `python src/train.py`.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Accuracy", f"{t['accuracy']:.1%}")
        b.metric("Counterfeit recall (fakes caught)", f"{t['counterfeit']['recall']:.1%}", help="Plan target: 90%")
        c.metric("Counterfeit precision", f"{t['counterfeit']['precision']:.1%}")
        d.metric("ROC-AUC", f"{t['roc_auc']:.3f}")
        st.warning("Counterfeit recall is below the 90% target in the plan. This is reported as is.")

        col1, col2 = st.columns([1, 1])
        with col1:
            cm = METRICS / "confusion_matrix_test_final.png"
            if cm.exists():
                st.image(str(cm), caption="Confusion matrix (test)", width="stretch")
        with col2:
            st.write("**Fakes caught, by alteration type**")
            alt = {k: v["recall"] for k, v in t["cf_recall_by_alteration"].items() if v}
            st.bar_chart(pd.Series(alt, name="recall"))
            st.write("**By strength:** " + ", ".join(f"{k} {v['recall']:.0%}" for k, v in t["cf_recall_by_strength"].items() if v))
            st.write("**By type:** " + ", ".join(f"{k}: fakes caught {v['cf_recall']['recall']:.0%}, genuine passed "
                                                 f"{v['genuine_recall']:.0%}" for k, v in t["by_kind"].items()))

    s = read_json(METRICS / "sanity_checks.json")
    if s:
        st.subheader("Sanity checks (is the model learning something real?)")
        st.write(f"- Trivial colour-histogram baseline AUC: **{s['baseline_colour_histogram']['auc']:.2f}** "
                 f"(CNN: {t['roc_auc']:.2f})" if t else "")
        st.write(f"- File size / dimensions alone, AUC: **{s['shortcut_file_features']['auc']:.2f}** (0.5 = no shortcut)")
        st.write(f"- Overfit-16-images test accuracy: **{s['overfit_16']['train_acc_on_16']:.0%}** (training loop works)")

    dmr = read_json(METRICS / "drugmatch_report.json")
    if dmr:
        st.subheader("Does drug matching help? (pills only)")
        sig = dmr["signals"]
        st.dataframe(pd.DataFrame({k: {"test AUC": round(v["test_auc"], 3),
                                       "fakes caught": f"{v['test_counterfeit_recall']:.0%}",
                                       "false alarms": f"{v['test_false_alarm_rate']:.0%}"} for k, v in sig.items()}).T)
        st.caption("Drug similarity adds weak evidence; fusing it with the CNN improves AUC slightly but raises false alarms.")

    ev = read_json(METRICS / "explain_validation_test.json")
    if ev:
        st.subheader("Do the heatmaps point at the real edit?")
        e = ev["explainer"]
        st.caption(f"Checked on {ev['n_localised_fakes']} test fakes where the edit location is known (method: {e['method']}, "
                   f"layer {e['layer']}, chosen on validation data only).")
        a, b, c = st.columns(3)
        a.metric("Hottest point inside the edit", f"{ev['fake']['hit']:.0%}",
                 help="Share of fakes where the single hottest pixel falls inside the edited box.")
        b.metric("Attention on the edit vs chance", f"{ev['fake']['lift']:.1f}x")
        c.metric("Same spot on the genuine twin", f"{ev['genuine_twin_control']['hit']:.0%}", help="Control: should be low.")
        gc = config.OUTPUTS_DIR / "gradcam" / "gradcam_cases.png"
        if gc.exists():
            with st.expander("Curated examples: fake caught, genuine passed, false alarm, fake missed (green box = real edit)"):
                st.image(str(gc), width="stretch")

    sorted_csv = config.OUTPUTS_DIR / "sorted" / "sorted_report.csv"
    if sorted_csv.exists():
        r = pd.read_csv(sorted_csv)
        if "truth" in r:
            st.subheader("Sorting the mixed test pool")
            auto = r[r.sorted_as != "escalate"]
            st.write(f"Of {len(r)} anonymous images: auto-decided **{len(auto)}** "
                     f"({len(auto) / len(r):.0%}), of which **{(auto.sorted_as == auto.truth).mean():.1%}** were right; "
                     f"{(r.sorted_as == 'escalate').sum()} escalated to a human.")
            st.dataframe(pd.crosstab(r.truth, r.sorted_as))


# ----------------------------------------------------------------------------- tab 3: status
with tab_status:
    st.subheader("Where the project stands")
    status = pd.DataFrame([
        ("1", "Planning, repo", "✅ Done"),
        ("2", "Environment (Python, Node, Hardhat, GPU)", "✅ Done"),
        ("3", "Dataset (MEDISEG pills + Mendeley drug boxes)", "✅ Done"),
        ("4", "Counterfeit simulation + leak-free splits", "✅ Done"),
        ("5", "CNN training and evaluation", "✅ Done (fakes caught 85%, target 90%)"),
        ("+", "Drug identification and matching", "✅ Done (small gain)"),
        ("6", "Explainability", "✅ Heatmap + validation + figures done; SHAP/LIME comparison to do"),
        ("7", "Smart contract + tests", "⬜ Not started"),
        ("8", "Python-blockchain link + fallback", "⬜ Not started"),
        ("9", "Decision pipeline + threshold tuning", "🟡 Basic thresholds only"),
        ("10", "Full Streamlit app", "🟡 This prototype"),
        ("11", "Full testing + robustness", "🟡 Sanity checks done"),
        ("12", "Report, logbook, viva", "⬜ Not started"),
    ], columns=["Phase", "What", "Status"])
    st.dataframe(status, hide_index=True, width="stretch")
    st.subheader("Honest limitations")
    st.markdown(
        "- Counterfeits are **simulated** (colour shift, imprint blur/erase, wrong text, shape warp, noise/JPEG). "
        "No public labelled counterfeit dataset exists.\n"
        "- Small dataset (1,000 genuine images: 600 pills + 400 boxes).\n"
        "- Photos from other cameras/backgrounds or unseen products may be misjudged; uncertain cases should be escalated.\n"
        "- A looks-genuine verdict is **not** proof; only lab chemistry can confirm a medicine.\n"
        "- Drug matching covers only 23 drugs and only loose pills.")
