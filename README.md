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

Genuine class (1,000 images = 600 pills + 400 medicine boxes), both CC BY 4.0:

- **Pills:** MEDISEG (32-pills subset), City, University of London. https://doi.org/10.25383/city.28574786.v2
- **Boxes:** Mobile-Captured Drug Packs, Mendeley Data (phone photos of drug packs). https://doi.org/10.17632/bsmy5jjysy.3

Reproduce the data (`data/` is not committed):

```
# 1. download MEDISEG.tar.gz from the figshare link above and extract it
python src/import_mediseg.py --src <extracted>/MEDISEG/32pills --n 600   # distinct photos with a visible pill
python src/import_drugpacks.py --n 400 --per-group 5                     # downloads box photos (checksum-verified)
python src/clean_dataset.py                                              # clean, standardise JPEG q95, cap 1000
python src/generate_fakes.py                                             # 70/15/15 split, simulated counterfeits, manifest
```

Counterfeits (pills **and** boxes) are **simulated** (colour shift, imprint/label blur or erase, wrong text, shape warp,
noise/JPEG). Localised edits are placed on the annotated pill; for boxes they are placed inside the photo (no pack
annotations exist, so some land on background). Box photos are split by pack group (same pack never in two splits;
groups are inferred from capture time / file order, so this is approximate). Findings do not claim real-world
counterfeit-detection performance.
