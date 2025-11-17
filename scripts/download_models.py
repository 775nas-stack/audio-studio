import os
import sys
import requests
from pathlib import Path

SESSION = requests.Session()
headers = {"Accept": "application/vnd.github+json"}
token = os.environ.get("GITHUB_TOKEN")
if token:
    headers["Authorization"] = f"Bearer {token}"
SESSION.headers.update(headers)

MODELS = [
    {
        "repo": "marl/crepe",
        "path": "crepe/assets",
        "filename": "model-full.h5",
        "destination": "backend/vendor/crepe/model-full.h5",
    },
    {
        "repo": "maxrmorrison/torchcrepe",
        "path": "torchcrepe/assets",
        "filename": "full.pth",
        "destination": "backend/vendor/torchcrepe/full.pth",
    },
]


def fetch_download_url(repo: str, content_path: str, target_filename: str) -> str:
    url = f"https://api.github.com/repos/{repo}/contents/{content_path}"
    print(f"Fetching contents from {url}")
    response = SESSION.get(url)
    response.raise_for_status()
    entries = response.json()
    if not isinstance(entries, list):
        raise RuntimeError(f"Unexpected response when listing {repo}:{content_path}")
    for entry in entries:
        if entry.get("name") == target_filename:
            download_url = entry.get("download_url")
            if not download_url:
                raise RuntimeError(f"download_url missing for {target_filename} in {repo}")
            print(f"Found download URL for {target_filename}")
            return download_url
    raise FileNotFoundError(f"{target_filename} not found in {repo}/{content_path}")


def download_file(download_url: str, destination: str) -> None:
    dest_path = Path(destination)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {download_url} -> {destination}")
    with SESSION.get(download_url, stream=True) as response:
        response.raise_for_status()
        with open(dest_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)
    size = dest_path.stat().st_size
    if size <= 0:
        raise RuntimeError(f"Downloaded file at {destination} is empty")
    print(f"Saved {destination} ({size} bytes)")


def main() -> None:
    for model in MODELS:
        try:
            url = fetch_download_url(model["repo"], model["path"], model["filename"])
            download_file(url, model["destination"])
        except Exception as exc:
            print(f"Error processing {model['filename']}: {exc}", file=sys.stderr)
            raise
    print("Model download completed successfully.")


if __name__ == "__main__":
    main()
