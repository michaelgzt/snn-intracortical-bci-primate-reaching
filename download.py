"""
Dataset Download Utilities

Download primate reaching dataset from Zenodo with MD5 verification.
Based on neurobench/datasets/utils.py (BSD 3-Clause License, PyTorch Vision).
"""

import os
import sys
import hashlib
import urllib.request
from urllib.error import URLError

try:
    from torch.utils.model_zoo import tqdm
except ImportError:
    from tqdm import tqdm

# Zenodo URL for primate reaching dataset
ZENODO_URL = "https://zenodo.org/record/583331/files/"

# MD5 checksums for integrity verification
MD5_CHECKSUMS = {
    "indy_20170131_02.mat": "2790b1c869564afaa7772dbf9e42d784",
    "indy_20160630_01.mat": "197413a5339630ea926cbd22b8b43338",
    "indy_20160622_01.mat": "c33d5fff31320d709d23fe445561fb6e",
    "loco_20170301_05.mat": "47342da09f9c950050c9213c3df38ea3",
    "loco_20170215_02.mat": "739b70762d838f3a1f358733c426bb02",
    "loco_20170210_03.mat": "4cae63b58c4cb9c8abd44929216c703b",
}

USER_AGENT = "snn-training"


def calculate_md5(fpath: str, chunk_size: int = 1024 * 1024) -> str:
    """Calculate MD5 checksum of a file."""
    if sys.version_info >= (3, 9):
        md5 = hashlib.md5(usedforsecurity=False)
    else:
        md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def check_integrity(fpath: str, md5: str = None) -> bool:
    """Check if file exists and has correct MD5 checksum."""
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return md5 == calculate_md5(fpath)


def _save_response_content(content, destination, length):
    """Save response content to file with progress bar."""
    with open(destination, "wb") as fh, tqdm(total=length) as pbar:
        for chunk in content:
            if not chunk:
                continue
            fh.write(chunk)
            pbar.update(len(chunk))


def _urlretrieve(url: str, filename: str, chunk_size: int = 1024 * 32):
    """Download URL to file."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response:
        _save_response_content(
            iter(lambda: response.read(chunk_size), b""),
            filename,
            length=response.length,
        )


def _get_redirect_url(url: str, max_hops: int = 3) -> str:
    """Follow redirects and return final URL."""
    initial_url = url
    headers = {"Method": "HEAD", "User-Agent": USER_AGENT}

    for _ in range(max_hops + 1):
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request) as response:
            if response.url == url or response.url is None:
                return url
            url = response.url

    raise RecursionError(
        f"Request to {initial_url} exceeded {max_hops} redirects."
    )


def download_url(url: str, file_path: str, md5: str = None) -> None:
    """
    Download a file from URL with optional MD5 verification.

    Args:
        url: URL to download from
        file_path: Full path to save the file
        md5: Expected MD5 checksum (optional)
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Check if file already exists and is valid
    if check_integrity(file_path, md5):
        print(f"Using downloaded and verified file: {file_path}")
        return

    # Follow redirects
    url = _get_redirect_url(url)

    # Download
    try:
        print(f"Downloading {url} to {file_path}")
        _urlretrieve(url, file_path)
    except (URLError, OSError) as e:
        # Try HTTP fallback if HTTPS fails
        if url.startswith("https"):
            url = url.replace("https:", "http:")
            print(f"HTTPS failed, trying HTTP: {url}")
            _urlretrieve(url, file_path)
        else:
            raise e

    # Verify download
    if md5 and not check_integrity(file_path, md5):
        raise RuntimeError(f"Downloaded file {file_path} failed MD5 verification")


def ensure_dataset(data_dir: str, filename: str) -> str:
    """
    Ensure dataset file exists, downloading if necessary.

    Args:
        data_dir: Directory to store/find the dataset
        filename: Dataset filename (e.g., 'indy_20160622_01.mat')

    Returns:
        Full path to the dataset file
    """
    file_path = os.path.join(data_dir, filename)

    if os.path.isfile(file_path):
        # Check integrity if we have a checksum
        md5 = MD5_CHECKSUMS.get(filename)
        if md5 and not check_integrity(file_path, md5):
            print(f"File {file_path} exists but failed MD5 check, re-downloading...")
        else:
            return file_path

    # Download
    if filename not in MD5_CHECKSUMS:
        raise ValueError(
            f"Unknown dataset file: {filename}. "
            f"Available files: {list(MD5_CHECKSUMS.keys())}"
        )

    url = ZENODO_URL + filename
    md5 = MD5_CHECKSUMS[filename]
    download_url(url, file_path, md5)

    return file_path


def list_available_datasets() -> list:
    """Return list of available dataset filenames."""
    return list(MD5_CHECKSUMS.keys())
