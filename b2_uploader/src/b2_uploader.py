import os
import sys
from dotenv import load_dotenv
from b2sdk.v2 import (
    B2Api, InMemoryAccountInfo, AuthInfoCache, ScanPoliciesManager,
    parse_folder, Synchronizer, SyncReport, CompareVersionMode,
    NewerFileSyncMode
)
import time


class SynchronizerBase:
    def __init__(self):
        pass

    def upload_file(self, file_path, destination_path):
        pass

    def download_file(self, bucket_path, dest_directory):
        pass


class B2Synchronizer(SynchronizerBase):
    def __init__(self, b2_bucket_name=None):
        super().__init__()
        self.b2_bucket_name = b2_bucket_name
        
        # Load API Key and bucket name
        load_dotenv()
        self.b2_api_key_id = os.getenv("B2_APPLICATION_KEY_ID")
        self.b2_api_key = os.getenv("B2_APPLICATION_KEY")

        # Set b2_bucket_name to value from env if not provided
        if b2_bucket_name is None:
            b2_bucket_name = os.getenv("B2_BUCKET_NAME")

        # Validate environment variables
        if not all([self.b2_api_key_id, self.b2_api_key, self.b2_bucket_name]):
            raise ValueError(
                "Missing required environment variables: B2_APPLICATION_KEY_ID, B2_APPLICATION_KEY, or B2_BUCKET_NAME")

        # Check that b2_bucket_name is a string, set path to bucket (required for some API functions)
        if not isinstance(self.b2_bucket_name, str):
            raise TypeError("B2 bucket name must be a string")
        self.b2_bucket_name_path = 'b2://' + self.b2_bucket_name

        try:
            # Initialize the B2 API
            info = InMemoryAccountInfo()
            self.b2_api = B2Api(info, cache=AuthInfoCache(info)) # type: ignore
            self.b2_api.authorize_account("production", self.b2_api_key_id, self.b2_api_key)
        
        except Exception as e:
            print(f"B2 API initialization failed: {e}")
            raise(e)


    def upload_file(self, file_path, dest_directory):
        """
        Upload a file to B2.
        :param file_path: Path to the file to upload.
        :param dest_directory: Destination directory in B2 bucket.
        """
        source_path = os.path.abspath(file_path)
        if not os.path.exists(source_path):
            raise ValueError(f"Source path does not exist: {source_path}")
        
        try:
            # Parse folders
            local_folder = parse_folder(source_path, self.b2_api)
            bucket_folder = parse_folder(f"{self.b2_bucket_name_path}/{dest_directory}", self.b2_api)

        except Exception as e:
            raise e
        
        self._push(local_folder, bucket_folder)


    def download_file(self, bucket_path, dest_directory):
        """
        Download a file or directory from B2.
        :param bucket_path: Path in bucket to download.
        :param dest_directory: Destination directory local.
        """
        dest_path = os.path.abspath(dest_directory)
        if not os.path.exists(dest_path):
            raise ValueError(f"Destination path does not exist: {dest_path}")

        try:
            # Parse folders
            local_folder = parse_folder(dest_path, self.b2_api)
            bucket_folder = parse_folder(f"{self.b2_bucket_name_path}/{bucket_path}", self.b2_api)
        except Exception as e:
                raise e
        
        self._push(bucket_folder, local_folder)


    def _push(self, parsed_source_path, parsed_dest_path):
        """
        Sync files from source to destination.
        :param parsed_source_path: Parsed source path.
        :param parsed_dest_path: Parsed destination path.
        """
        try:
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
            with SyncReport(sys.stdout, no_progress=False) as reporter:  # type: ignore
                synchronizer.sync_folders(
                    source_folder=parsed_source_path,
                    dest_folder=parsed_dest_path,
                    now_millis=int(round(time.time() * 1000)),
                    reporter=reporter,
                )

        except Exception as e:
            raise e