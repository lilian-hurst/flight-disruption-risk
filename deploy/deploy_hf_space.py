"""One-shot deploy to a free Hugging Face Space (Docker SDK).

Usage:
    HF_TOKEN=hf_xxx python3 deploy/deploy_hf_space.py [--space-name flight-disruption-risk]

Requires a Hugging Face account (free, no credit card) and an access token
with "write" permission, created at https://huggingface.co/settings/tokens

What this script does:
1. Creates a public Space under your account (type=space, sdk=docker) if it
   doesn't already exist.
2. Uploads the code (src/, static/), the trained model (models/*.json),
   requirements.txt and the Dockerfile -- NOT the raw training data (data/),
   which is regenerable and unnecessary for serving predictions.
3. Uploads a README.md with the YAML front-matter Hugging Face Spaces needs
   to know how to build the Space (sdk: docker, app_port, etc.), followed by
   the project's normal documentation.
4. Prints the public URL once the Space starts building.

The Space builds and serves the exact same Docker image that was already
built and tested locally in this project (see README.md "Ce qui a été
vérifié").
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent

SPACE_README_HEADER = """---
title: Flight Disruption Risk
emoji: ✈️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space-name", default="flight-disruption-risk")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    if not args.token:
        print("Erreur : fournis un token via --token ou la variable d'env HF_TOKEN")
        print("Crée un token (write) sur https://huggingface.co/settings/tokens")
        sys.exit(1)

    api = HfApi(token=args.token)
    username = api.whoami()["name"]
    repo_id = f"{username}/{args.space_name}"

    print(f"-> Création/mise à jour du Space {repo_id}...")
    api.create_repo(
        repo_id=repo_id, repo_type="space", space_sdk="docker",
        token=args.token, exist_ok=True, private=False,
    )

    print("-> Upload du code, du modèle et du Dockerfile...")
    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=str(ROOT),
        token=args.token,
        allow_patterns=[
            "src/**", "static/**", "models/*.json", "requirements.txt",
            "Dockerfile",
        ],
        ignore_patterns=["**/__pycache__/**", "**/*.pyc"],
        commit_message="Deploy: code + modèle entraîné + interface",
    )

    print("-> Upload du README (avec métadonnées Hugging Face Spaces)...")
    project_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    space_readme = SPACE_README_HEADER + project_readme
    api.upload_file(
        path_or_fileobj=space_readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="space",
        token=args.token,
        commit_message="Deploy: README avec métadonnées Space",
    )

    url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"\nSpace créé/mis à jour : {url}")
    print("Le build Docker démarre automatiquement (2-4 minutes en général).")
    print(f"Une fois prêt, l'app sera accessible sur : https://{username}-{args.space_name}.hf.space")


if __name__ == "__main__":
    main()
