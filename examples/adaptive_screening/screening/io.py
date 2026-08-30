"""Atomic result files and content hashes for frozen scientific artifacts."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from .config import digest


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_table(path, rows, metadata):
    frame = pd.DataFrame(rows)
    for column in ("environment_hash", "code_commit", "source_hash"):
        frame[column] = metadata[column]
    frame.to_csv(path, index=False, float_format="%.17g")
    return frame


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_manifest(directory, filenames, metadata):
    files = {name: file_hash(Path(directory) / name) for name in sorted(filenames)}
    payload = {"environment_hash": metadata["environment_hash"], "files": files}
    return {**payload, "pilot_result_hash": digest(payload)}


def verify_manifest(directory):
    directory = Path(directory)
    manifest = read_json(directory / "result_manifest.json")
    payload = {
        "environment_hash": manifest["environment_hash"],
        "files": manifest["files"],
    }
    if digest(payload) != manifest["pilot_result_hash"]:
        raise ValueError("Invalid pilot result manifest")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or file_hash(directory / name) != expected:
            raise ValueError(f"Pilot result hash mismatch: {name}")
    return manifest
