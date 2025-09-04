#!/usr/bin/env python3

import os
from huggingface_hub import snapshot_download


def download_conlid():
    os.makedirs("models/conlid-model", exist_ok=True)
    snapshot_download(repo_id="epfl-nlp/ConLID", local_dir="models/conlid-model")


if __name__ == "__main__":
    download_conlid()
