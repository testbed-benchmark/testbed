r"""Moves a W&B sweep's run directories between disk and a Hugging Face dataset.

``upload`` resolves a sweep id to its runs and maps each one back to its run
directory the same way ``scripts/sweep_reeval.py`` does, then pushes each
directory to the dataset repo under
``<project>/<sweep_name>/<sweep_id>/<run_name>``. A ``runs.jsonl`` manifest
lands beside them holding each run's id, state, W&B url, config, summary and,
crucially, the ``save_dir`` the run was trained under.

``download`` puts the directories back exactly where they came from:
``<save_dir>/<run_name>``, read per run from the manifest, which is what
``sweep_reeval.py`` and ``eval_run.py`` expect to find. Since ``save_dir``
defaults to the relative ``runs``, download from the repo root to land in
``runs/<run_name>``. Files are staged first and then moved into place, so a
half-finished transfer never leaves a partial run directory behind.

``huggingface_hub`` is not a testbed dependency: install it in whatever
environment you run this from. Authentication comes from ``huggingface-cli
login`` or ``$HF_TOKEN``.

Usage::

    python scripts/sweep_hf_upload.py upload 1dpl3rg2 --project location_finding_2src
    python scripts/sweep_hf_upload.py download 1dpl3rg2 --project location_finding_2src
    python scripts/sweep_hf_upload.py download 1dpl3rg2 --project location_finding_2src \
        --list
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys

from sweep_reeval import resolve_sweep_path, run_directory

DEFAULT_REPO_ID = "RainML/testbed"
MANIFEST_NAME = "runs.jsonl"


def hf_api(token: str | None):
    """Returns an HfApi client, or exits if huggingface_hub is missing."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit(
            "huggingface_hub is not installed; pip install it into the "
            "environment you run this script from"
        )
    return HfApi(token=token)


def path_segment(name: str) -> str:
    """Makes a sweep name safe to use as one path component."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "unnamed"


def sweep_object(sweep_path: str):
    """Returns the W&B sweep, whose ``.name`` and ``.runs`` this script needs."""
    import wandb

    return wandb.Api().sweep(sweep_path)


def sorted_runs(sweep):
    """Returns the sweep's runs, ordered by name to match sweep_reeval."""
    return sorted(sweep.runs, key=lambda run: run.name)


def upload_prefix(sweep, project: str, sweep_id: str) -> str:
    """Builds ``<project>/<sweep_name>/<sweep_id>`` for one sweep."""
    return f"{project}/{path_segment(sweep.name or sweep_id)}/{sweep_id}"


def download_prefix(api, args, project: str, sweep_id: str) -> str:
    """Finds the stored ``<project>/<sweep_name>/<sweep_id>`` prefix in the repo."""
    prefixes = set()
    for path in api.list_repo_files(args.repo_id, repo_type="dataset"):
        parts = path.split("/")
        if len(parts) > 3 and parts[0] == project and parts[2] == sweep_id:
            prefixes.add("/".join(parts[:3]))
    if not prefixes:
        raise SystemExit(f"{args.repo_id} holds no {project}/*/{sweep_id}")
    if len(prefixes) > 1:
        listed = ", ".join(sorted(prefixes))
        raise SystemExit(f"{sweep_id} sits under several prefixes: {listed}")
    return prefixes.pop()


def run_metadata(run, run_dir: str, path_in_repo: str) -> dict:
    """Summarises one W&B run for the manifest.

    ``save_dir`` is what download needs: it is the root the run was trained
    under, so the directory can go back to the path every other script derives.
    """
    return {
        "name": run.name,
        "id": run.id,
        "state": run.state,
        "url": run.url,
        "path": path_in_repo,
        "save_dir": os.path.dirname(os.path.normpath(run_dir)),
        "config": dict(run.config),
        "summary": dict(run.summary),
    }


def upload_run(api, run_dir: str, path_in_repo: str, args) -> None:
    """Pushes one run directory to the dataset repo."""
    api.upload_folder(
        folder_path=run_dir,
        path_in_repo=path_in_repo,
        repo_id=args.repo_id,
        repo_type="dataset",
        ignore_patterns=args.ignore or None,
        commit_message=f"Upload {path_in_repo}",
    )


def upload_manifest(api, rows: list[dict], path_in_repo: str, args) -> None:
    """Pushes the runs.jsonl manifest for the sweep."""
    body = "".join(json.dumps(row, default=str) + "\n" for row in rows)
    api.upload_file(
        path_or_fileobj=body.encode(),
        path_in_repo=path_in_repo,
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=f"Upload {path_in_repo}",
    )


def upload(args) -> None:
    """Uploads every finished run of one sweep to the dataset repo."""
    sweep_path = resolve_sweep_path(args.sweep, args.entity, args.project)
    _, project, sweep_id = sweep_path.split("/")

    sweep = sweep_object(sweep_path)
    runs = sorted_runs(sweep)
    if not runs:
        raise SystemExit(f"{sweep_path} has no runs")
    prefix = args.path_in_repo or upload_prefix(sweep, project, sweep_id)

    if args.list:
        print(f"{sweep_path}: {len(runs)} runs -> {args.repo_id}/{prefix}")
        for index, run in enumerate(runs):
            run_dir = run_directory(run, args.save_dir)
            present = "ok" if os.path.isdir(run_dir) else "MISSING"
            print(f"  [{index}] {run.id} {run.state:9s} {present:7s} {run_dir}")
        return

    api = None if args.dry_run else hf_api(args.token)
    if api is not None:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )

    uploaded = []
    failures = []
    for run in runs:
        run_dir = run_directory(run, args.save_dir)
        if run.state != "finished" and not args.all_states:
            print(f"skipping {run.name}: state is {run.state}", file=sys.stderr)
            continue
        if not os.path.isdir(run_dir):
            print(f"skipping {run.name}: no run dir at {run_dir}", file=sys.stderr)
            continue

        destination = f"{prefix}/{run.name}"
        print(f"→ {run_dir} -> {args.repo_id}/{destination}")
        if args.dry_run:
            continue
        try:
            upload_run(api, run_dir, destination, args)
        except Exception as e:  # noqa: BLE001 - report and keep going
            print(f"upload failed for {run.name}: {e!r}", file=sys.stderr)
            failures.append(run.name)
            continue
        uploaded.append(run_metadata(run, run_dir, destination))

    if uploaded:
        upload_manifest(api, uploaded, f"{prefix}/{MANIFEST_NAME}", args)
        print(f"uploaded {len(uploaded)} runs to {args.repo_id}/{prefix}")

    if failures:
        print(f"upload failed for: {', '.join(failures)}", file=sys.stderr)
        sys.exit(1)


def fetch_manifest(args, prefix: str) -> dict[str, dict]:
    """Reads the sweep's manifest, keyed by run name. Empty if it is absent."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    try:
        path = hf_hub_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            filename=f"{prefix}/{MANIFEST_NAME}",
            token=args.token,
        )
    except EntryNotFoundError:
        return {}
    with open(path) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {row["name"]: row for row in rows}


def restore_target(run_name: str, manifest: dict[str, dict], args) -> str:
    """Returns the on-disk path a downloaded run belongs at."""
    if args.save_dir is not None:
        return os.path.join(args.save_dir, run_name)
    save_dir = manifest.get(run_name, {}).get("save_dir")
    if not save_dir:
        raise SystemExit(
            f"{run_name} has no save_dir in the manifest; pass --save-dir to say "
            f"where the run directories belong"
        )
    return os.path.join(save_dir, run_name)


def move_into_place(staged: str, target: str, overwrite: bool) -> bool:
    """Moves one staged run directory to its on-disk home."""
    if os.path.exists(target):
        if not overwrite:
            print(f"skipping {target}: already on disk", file=sys.stderr)
            return False
        shutil.rmtree(target)
    parent = os.path.dirname(os.path.normpath(target))
    if parent:
        os.makedirs(parent, exist_ok=True)
    shutil.move(staged, target)
    return True


def download(args) -> None:
    """Pulls one sweep out of the dataset repo, back to its training paths."""
    from huggingface_hub import snapshot_download

    _, project, sweep_id = resolve_sweep_path(
        args.sweep, args.entity, args.project
    ).split("/")
    api = hf_api(args.token)
    prefix = args.path_in_repo or download_prefix(api, args, project, sweep_id)

    files = [
        path
        for path in api.list_repo_files(args.repo_id, repo_type="dataset")
        if path.startswith(f"{prefix}/")
    ]
    if not files:
        raise SystemExit(f"{args.repo_id} holds nothing under {prefix}")
    depth = len(prefix.split("/"))
    run_names = sorted(
        {path.split("/")[depth] for path in files if len(path.split("/")) > depth + 1}
    )

    manifest = fetch_manifest(args, prefix)
    targets = {name: restore_target(name, manifest, args) for name in run_names}

    if args.list or args.dry_run:
        print(f"{args.repo_id}/{prefix}: {len(run_names)} runs")
        for name in run_names:
            state = "EXISTS" if os.path.exists(targets[name]) else "new"
            print(f"  {state:7s} {targets[name]}")
        return

    staging = os.path.abspath(args.staging_dir)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        allow_patterns=[f"{prefix}/*"],
        local_dir=staging,
        token=args.token,
    )

    restored = 0
    for name in run_names:
        staged = os.path.join(staging, prefix, name)
        if not os.path.isdir(staged):
            print(f"skipping {name}: not staged at {staged}", file=sys.stderr)
            continue
        print(f"→ {args.repo_id}/{prefix}/{name} -> {targets[name]}")
        restored += move_into_place(staged, targets[name], args.overwrite)

    if not args.keep_staging:
        shutil.rmtree(staging, ignore_errors=True)
    print(f"restored {restored} of {len(run_names)} runs")


def main() -> None:
    """Runs the upload or download mode."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["upload", "download"])
    parser.add_argument(
        "sweep", help="Sweep id, or the full <entity>/<project>/<sweep_id> path."
    )
    parser.add_argument(
        "--entity",
        default=os.environ.get("WANDB_ENTITY", "rainml-testbed"),
        help="Default: $WANDB_ENTITY, else rainml-testbed.",
    )
    parser.add_argument(
        "--project", default=None, help="Required when the sweep id is bare."
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Root holding the run directories. Default on upload: each run's "
        "own save_dir; on download: the save_dir recorded in the manifest.",
    )
    parser.add_argument(
        "--repo-id", default=DEFAULT_REPO_ID, help=f"Default: {DEFAULT_REPO_ID}."
    )
    parser.add_argument(
        "--path-in-repo",
        default=None,
        help="Prefix in the dataset repo. Default: <project>/<sweep_name>/<sweep_id>.",
    )
    parser.add_argument(
        "--ignore",
        action="append",
        default=None,
        help="Upload: glob to leave out, repeatable. Pass '*.pt' to skip checkpoints.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Default: $HF_TOKEN, else the cached huggingface-cli login.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Upload: create the repo private when it does not exist yet.",
    )
    parser.add_argument(
        "--all-states",
        action="store_true",
        help="Upload: include unfinished runs. Default: finished runs only.",
    )
    parser.add_argument(
        "--staging-dir",
        default=".hf_download",
        help="Download: where files land before being moved into place. Keep it "
        "on the same filesystem as save_dir. Default: .hf_download.",
    )
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="Download: leave the staging directory in place afterwards.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Download: replace run directories that are already on disk.",
    )
    parser.add_argument(
        "--list", action="store_true", help="Print what would move and exit."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the transfers without running."
    )
    args = parser.parse_args()

    if args.mode == "upload":
        upload(args)
    else:
        download(args)


if __name__ == "__main__":
    main()
