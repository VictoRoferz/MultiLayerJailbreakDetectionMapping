#!/usr/bin/env python3
"""
save_to_hf.py

Push artifacts and data from Lambda AI to HuggingFace Hub.

Usage:
    python save_to_hf.py --token YOUR_HF_TOKEN
    python save_to_hf.py --token YOUR_HF_TOKEN --run-name run1_eps015
    python save_to_hf.py --token YOUR_HF_TOKEN --folders artifacts data
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import login, create_repo, upload_folder


REPO_ID = "victorroferz/jailbreak-activation-mapping"


def main():
    parser = argparse.ArgumentParser(description="Push results to HuggingFace Hub")
    parser.add_argument("--token", type=str, required=True, help="HuggingFace write token")
    parser.add_argument("--repo-id", type=str, default=REPO_ID, help="HuggingFace repo ID")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Optional run name (e.g. run1_eps015). Uploads under runs/<run-name>/")
    parser.add_argument("--folders", nargs="+", default=["artifacts", "data"],
                        help="Folders to upload (default: artifacts data)")
    args = parser.parse_args()

    login(token=args.token)
    create_repo(args.repo_id, private=True, exist_ok=True, repo_type="model")

    for folder in args.folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"[SKIP] {folder}/ does not exist")
            continue

        if args.run_name:
            remote_path = f"runs/{args.run_name}/{folder}"
        else:
            remote_path = folder

        print(f"[UPLOAD] {folder}/ -> {args.repo_id}/{remote_path}/")
        upload_folder(
            folder_path=str(folder_path),
            repo_id=args.repo_id,
            path_in_repo=remote_path,
        )
        print(f"[DONE] {folder}/ uploaded")

    print(f"\nAll done! View at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
