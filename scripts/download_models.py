#!/usr/bin/env python
from dotenv import load_dotenv
import os
import time
import sys
from b2sdk.v2 import (
    B2Api, InMemoryAccountInfo, AuthInfoCache, ScanPoliciesManager, 
    parse_folder, Synchronizer, SyncReport, CompareVersionMode, 
    NewerFileSyncMode
)

# Load API Key 
load_dotenv()
b2_api_key_id = os.getenv("B2_APPLICATION_KEY_ID")
b2_api_key = os.getenv("B2_APPLICATION_KEY")
b2_bucket_name = os.getenv("B2_BUCKET_NAME")

# Validate environment variables
if not all([b2_api_key_id, b2_api_key, b2_bucket_name]):
    raise ValueError("Missing required environment variables: B2_APPLICATION_KEY_ID, B2_APPLICATION_KEY, or B2_BUCKET_NAME")

# Assert that b2_bucket_name is a string
assert isinstance(b2_bucket_name, str)
b2_bucket_name_path = 'b2://' + b2_bucket_name

# Parameters
source_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
if not os.path.exists(source_path):
    raise ValueError(f"Source path does not exist: {source_path}")

try:
    # Initialize the B2 API
    info = InMemoryAccountInfo()
    b2_api = B2Api(info, cache=AuthInfoCache(info)) 
    b2_api.authorize_account("production", b2_api_key_id, b2_api_key)
    
    # Parse folders
    local_folder = parse_folder(source_path, b2_api)
    bucket_folder = parse_folder(b2_bucket_name_path, b2_api)
    
    policies_manager = ScanPoliciesManager(exclude_all_symlinks=True)
    
    synchronizer = Synchronizer(
        max_workers=10,
        policies_manager=policies_manager,
        dry_run=False,
        allow_empty_source=True,
        compare_version_mode=CompareVersionMode.MODTIME,
        newer_file_mode=NewerFileSyncMode.SKIP,  # Skip if destination is newer
        compare_threshold=1  # 1 second threshold for modification time comparison
    )
    
    with SyncReport(sys.stdout, no_progress=False) as reporter: # type: ignore
        print("\nSyncing bucket files to local...")
        synchronizer.sync_folders(
            source_folder=bucket_folder,
            dest_folder=local_folder,
            now_millis=int(round(time.time() * 1000)),
            reporter=reporter,
        )
    print("\nSynchronization completed successfully!")

except Exception as e:
    print(f"Synchronization failed: {e}")
    sys.exit(1)