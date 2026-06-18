"""Push the processed ball data to a public Hugging Face dataset for the Space.

The deployed Space downloads this at startup so every match in the catalog is replayable
without baking ~300MB into the image. This is factual Cricsheet ball data, hence a public
dataset. Idempotent: re-run to update.

  HF_TOKEN=hf_... uv run python -m scripts.push_ball_data
"""

from __future__ import annotations

import argparse
from pathlib import Path

from configs.settings import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="push processed ball data to the Hub")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--repo", default="kattymandy/cricket-commentary-ball-data")
    args = parser.parse_args()

    token = get_settings().hf_token
    if token is None:
        raise SystemExit("set HF_TOKEN in the environment / .env")

    from huggingface_hub import HfApi

    api = HfApi(token=token.get_secret_value())
    api.create_repo(args.repo, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(
        folder_path=str(args.processed_dir),
        repo_id=args.repo,
        repo_type="dataset",
        allow_patterns=["*.jsonl"],
    )
    count = len(list(args.processed_dir.glob("*.jsonl")))
    print(f"pushed {count} match files -> https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
