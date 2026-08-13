import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Where local checkpoint dirs live. Checkpoints for the SR comparison are large
# and sit on the data volume, so allow an env override instead of hard-coding a
# `weights/` dir inside the repo.
POLICY_DIR = Path(os.environ.get("VLA_POLICY_DIR", ROOT / "weights"))
POLICY_DIR.mkdir(parents=True, exist_ok=True)

# Only claim the hub cache when the caller has not already placed it somewhere.
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(POLICY_DIR))
