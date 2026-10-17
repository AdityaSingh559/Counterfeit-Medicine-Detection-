"""Environment smoke test: imports, torch device, Hardhat node connection."""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

MODULES = [
    "torch", "torchvision", "pytorch_grad_cam", "shap", "lime", "streamlit",
    "sklearn", "matplotlib", "PIL", "cv2", "pandas", "web3", "solcx",
]

GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"


def ok(msg):
    print(f"{GREEN}[ OK ]{RESET} {msg}")


def fail(msg):
    print(f"{RED}[FAIL]{RESET} {msg}")


def main() -> int:
    failures = 0

    for name in MODULES:
        try:
            importlib.import_module(name)
            ok(f"import {name}")
        except Exception as e:  # noqa: BLE001
            fail(f"import {name}: {e}")
            failures += 1

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        label = torch.cuda.get_device_name(0) if device == "cuda" else "CPU only"
        ok(f"torch {torch.__version__} device: {device} ({label})")
    except Exception as e:  # noqa: BLE001
        fail(f"torch device check: {e}")
        failures += 1

    try:
        from web3 import Web3
        w3 = Web3(Web3.HTTPProvider(config.RPC_URL))
        if not w3.is_connected():
            raise ConnectionError(f"cannot reach {config.RPC_URL} (is `npx hardhat node` running?)")
        ok(f"Hardhat node connected at {config.RPC_URL}, chain id {w3.eth.chain_id}")
        n = len(w3.eth.accounts)
        if n == 20:
            ok(f"{n} funded accounts")
        else:
            fail(f"expected 20 accounts, found {n}")
            failures += 1
    except Exception as e:  # noqa: BLE001
        fail(f"Hardhat node: {e}")
        failures += 1

    try:
        from solcx import get_installed_solc_versions
        versions = [str(v) for v in get_installed_solc_versions()]
        if config.SOLC_VERSION in versions:
            ok(f"solc {config.SOLC_VERSION} installed")
        else:
            fail(f"solc {config.SOLC_VERSION} not installed (installed: {versions})")
            failures += 1
    except Exception as e:  # noqa: BLE001
        fail(f"solc check: {e}")
        failures += 1

    print()
    print("All checks passed." if failures == 0 else f"{failures} check(s) failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
