"""Download files from Google Drive share links (one link per file).

Used by ``train.py`` via the ``gdrive_files`` field of the training config:

    gdrive_files:
      ./w2v2-384:
        config.json: "https://drive.google.com/file/d/<ID>/view?usp=sharing"
        model.safetensors: "https://drive.google.com/file/d/<ID>/view?usp=sharing"

Usage (pre-download / debugging):
    python gdrive_download.py --config configs/train/offline/train_config_w2v2_384_kaggle.yml
"""

from pathlib import Path
import argparse
import re

import yaml


def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def _drive_file_id(url: str) -> str | None:
    """Extract the file id from `.../file/d/<id>/view?...` or `...?id=<id>` links."""
    match = re.search(r"/file/d/([-\w]+)", url) or re.search(r"[?&]id=([-\w]+)", url)
    return match.group(1) if match else None


def _download(url: str, out_path: Path):
    import gdown

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # pass the file id (supported by all gdown versions; `fuzzy` was removed in gdown 6)
    # gdown handles the large-file confirmation page
    file_id = _drive_file_id(url)
    if file_id is not None:
        result = gdown.download(id=file_id, output=str(out_path), quiet=False)
    else:
        result = gdown.download(url, str(out_path), quiet=False)
    if result is None or not out_path.exists():
        raise RuntimeError(
            f"Failed to download `{url}` -> `{out_path}`. "
            "Make sure the file is shared as 'Anyone with the link'."
        )


def download_gdrive_files(spec: dict[str, dict[str, str]] | None):
    """Download every `{dir: {filename: url}}` entry, skipping existing files."""
    if not spec:
        return
    for dir_path, files in spec.items():
        for filename, url in files.items():
            out_path = Path(dir_path) / filename
            if out_path.exists():
                print(f"[gdrive] Found `{out_path}`, skipping download")
                continue
            if "<ID>" in url:
                raise ValueError(
                    f"Placeholder link for `{out_path}`; put the real Google Drive link in the config"
                )
            print(f"[gdrive] Downloading `{out_path}`")
            _download(url, out_path)


def maybe_download_config(
    path_or_url: str,
    dest: str = "./configs/train/offline/_downloaded_config.yml",
) -> str:
    """If `path_or_url` is a URL download it to `dest` and return the local path."""
    if not _is_url(path_or_url):
        return path_or_url
    dest_path = Path(dest)
    if not dest_path.exists():
        print(f"[gdrive] Downloading training config to `{dest_path}`")
        _download(path_or_url, dest_path)
    return str(dest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path or Google Drive link to the training config YAML",
    )
    args = parser.parse_args()

    config_path = maybe_download_config(args.config)
    with open(config_path, encoding="utf-8") as f:
        config_dict = yaml.safe_load(f) or {}
    download_gdrive_files(config_dict.get("gdrive_files"))
