#!/usr/bin/env python3
"""
b2_cli.py — CLI wrapper around B2Uploader class (from our b2_uploader package).
"""
import argparse
import os
import sys
import getpass
from typing import Tuple
from b2_uploader import B2Uploader


def _normalize_remote(remote: str) -> Tuple[str, str, str]:
    """
    Accepts either "b2://bucket/prefix" or "bucket/prefix" (or just "bucket").
    Returns (normalized_remote, bucket, prefix)
    where normalized_remote is "b2://bucket/prefix" with prefix possibly empty.
    """
    r = remote.strip()
    if r.startswith("b2://"):
        r = r[5:]  # strip scheme
    # Now r is "bucket[/prefix...]"
    parts = r.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) == 2 else ""
    # Assemble normalized
    if prefix:
        norm = f"b2://{bucket}/{prefix}"
    else:
        norm = f"b2://{bucket}"
    return norm, bucket, prefix


def _prompt_if_missing(args):
    if not args.key_id:
        args.key_id = input("Backblaze keyID: ").strip()
    if not args.app_key:
        args.app_key = getpass.getpass("Backblaze application key (hidden): ").strip()
    if not args.remote:
        args.remote = input("Remote B2 path (e.g., b2://bucket/prefix or bucket/prefix): ").strip()
    if not args.local:
        args.local = input("Local path to folder: ").strip()
    if args.direction not in ("upload", "download"):
        d = ""
        while d not in ("upload", "download"):
            d = input("Direction ('upload' local->B2 or 'download' B2->local): ").strip().lower()
        args.direction = d
    return args


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Upload/download folders with B2Uploader.")
    p.add_argument("--key-id", help="Backblaze keyID (or set B2_KEY_ID env var).")
    p.add_argument("--app-key", help="Backblaze application key (or set B2_APP_KEY env var).")
    p.add_argument("--remote", help="Remote path: 'b2://bucket/prefix' or 'bucket/prefix'.")
    p.add_argument("--local", help="Local folder path.")
    p.add_argument("--direction", choices=["upload", "download"], help="Operation direction.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Pull from env if not provided
    ## remove pulling items from environment for now
    args.key_id = args.key_id # or os.getenv("B2_KEY_ID")
    args.app_key = args.app_key # or os.getenv("B2_APP_KEY")
    args.remote = args.remote # or os.getenv("B2_REMOTE")
    args.local = args.local # or os.getenv("B2_LOCAL")
    args.direction = args.direction # or (os.getenv("B2_DIRECTION") or "").lower() or None

    args = _prompt_if_missing(args)

    # Normalize remote and extract bucket
    normalized_remote, bucket, prefix = _normalize_remote(args.remote)

    # Ensure local path exists for download; for upload, we validate it's a dir
    if args.direction == "upload":
        if not os.path.isdir(args.local):
            raise FileNotFoundError(f"Local directory not found: {args.local}")
    else:
        os.makedirs(args.local, exist_ok=True)

    # B2Uploader reads these at __init__, so set them first
    os.environ["B2_APPLICATION_KEY_ID"] = args.key_id
    os.environ["B2_APPLICATION_KEY"] = args.app_key
    os.environ["B2_BUCKET_NAME"] = bucket

    # Initialize user's uploader and authorize
    uploader = B2Uploader(b2_bucket_name=bucket)

    # Execute
    if args.direction == "upload":
    # local -> B2
        uploader.upload_file(args.local, prefix)
    else:
    # B2 -> local
        uploader.download_file(prefix, args.local)

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
