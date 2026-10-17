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
