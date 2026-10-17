# Explainable Counterfeit Medicine Detection with Blockchain Audit Trail

CNN (MobileNetV2) + Grad-CAM explanations + autonomous APPROVE / FLAG / ESCALATE decisions + Ethereum smart-contract audit trail (local Hardhat) + Streamlit app.

College project (team of 2). Counterfeits are **simulated**; results do not claim real-world performance.

See [implementation_plan.md](implementation_plan.md) for the full PRD, TRD and 12-phase plan.

## Run order (to be completed as phases land)

```
# Terminal 1: local chain
cd blockchain && npx hardhat node

# Terminal 2: project root (venv active)
python src/deploy.py          # after every node restart
streamlit run app.py
```

## Roles
- Person A: data, model, explainability (branch `data-model`)
- Person B: contract, web3, app (branch `chain-app`)

Fallback rule: if blockchain integration is not working after 2 days of effort, set `AUDIT_BACKEND="hashchain"`.

## Dataset

Genuine class: a 1,000-image sample of **MEDISEG (32-pills subset)**, City, University of London, licensed **CC BY 4.0**.
https://doi.org/10.25383/city.28574786.v2

Reproduce the data (data/ is not committed):

```
# 1. download MEDISEG.tar.gz from the figshare link above and extract it
python src/import_mediseg.py --src <extracted>/MEDISEG/32pills   # samples 1100 distinct photos with a visible pill
python src/clean_dataset.py                                      # cleans, standardises JPEG q95, caps at 1000
python src/generate_fakes.py                                     # 70/15/15 split, simulated counterfeits, manifest
```

Counterfeits are **simulated** (colour shift, imprint blur/erase, wrong text, shape warp, noise/JPEG), and localised edits are
placed on the annotated pill. Findings do not claim real-world counterfeit-detection performance.
