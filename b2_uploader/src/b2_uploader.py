import os
from b2sdk.v2 import (
    B2Api, InMemoryAccountInfo, AuthInfoCache, ScanPoliciesManager,
    parse_folder, Synchronizer, SyncReport, CompareVersionMode,
    NewerFileSyncMode
)
import time

class UploaderBase:
    def __init__(self):
        pass

    def upload_file(self, file_path, destination_path):
        pass

class B2Uploader(UploaderBase):
    def __init__(self):
        super().__init__()
        self.b2_api_key_id = os.getenv("B2_APPLICATION_KEY_ID")
        self.b2_api_key = os.getenv("B2_APPLICATION_KEY")
        self.b2_bucket_name = os.getenv("B2_BUCKET_NAME")

        # Validate environment variables
        if not all([self.b2_api_key_id, self.b2_api_key, self.b2_bucket_name]):
            raise ValueError(
                "Missing required environment variables: B2_APPLICATION_KEY_ID, B2_APPLICATION_KEY, or B2_BUCKET_NAME")

        # Assert that b2_bucket_name is a string
        assert isinstance(self.b2_bucket_name, str)
        self.b2_bucket_name_path = 'b2://' + self.b2_bucket_name



    def upload_file(self, file_path, dest_directory):
        """
        Upload a file to B2.
        :param file_path: Path to the file to upload.
        :param dest_directory: Destination directory.
        """
        self.source_path = os.path.abspath(file_path)
        if not os.path.exists(self.source_path):
            raise ValueError(f"Source path does not exist: {self.source_path}")
        try:
            # Initialize the B2 API
            info = InMemoryAccountInfo()
            b2_api = B2Api(info, cache=AuthInfoCache(info))
            b2_api.authorize_account("production", self.b2_api_key_id, self.b2_api_key)

            # Parse folders
            local_folder = parse_folder(self.source_path, b2_api)
            bucket_folder = parse_folder(f"{self.b2_bucket_name_path}/{dest_directory}", b2_api)

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
                    # source_folder=bucket_folder,
                    # dest_folder=local_folder,
                    source_folder=local_folder,
                    dest_folder=bucket_folder,
                    now_millis=int(round(time.time() * 1000)),
                    reporter=reporter,
                )

        except Exception as e:
            raise e


