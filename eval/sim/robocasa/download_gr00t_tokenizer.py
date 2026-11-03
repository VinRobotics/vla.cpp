#!/usr/bin/env python3
# Copyright 2026 VinRobotics - Apache-2.0
"""Download the Eagle tokenizer bundled with NVIDIA's N1.6 release; no weights."""

import argparse
from pathlib import Path
from urllib.request import urlopen

REVISION = "ead52833afbbf4243f8cd5e7664f48a94de03b19"
BASE_URL = (f"https://raw.githubusercontent.com/NVIDIA/Isaac-GR00T/{REVISION}/"
            "gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2")
FILES = ("tokenizer_config.json", "vocab.json", "merges.txt", "added_tokens.json",
         "special_tokens_map.json", "chat_template.json")


def download_tokenizer(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        with urlopen(f"{BASE_URL}/{name}", timeout=120) as response:
            body = response.read()
        temporary = output_dir / f"{name}.tmp"
        temporary.write_bytes(body)
        temporary.replace(output_dir / name)
        print(f"[tokenizer] {output_dir / name}", flush=True)
    (output_dir / "upstream_revision.txt").write_text(REVISION + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[3] / "weights/GR00T-N1.6/tokenizer")
    download_tokenizer(parser.parse_args().output_dir)
