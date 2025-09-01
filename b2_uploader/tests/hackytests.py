import tempfile
import os
from datetime import datetime
from dotenv import load_dotenv
from b2_uploader import B2Uploader

load_dotenv()

mytest = B2Uploader("Johannesh-jodan-llm-project-25-bucket")

with tempfile.TemporaryDirectory() as temp_dir:
    temp_file_path = os.path.join(temp_dir, f"test_file_{datetime.now().isoformat()}.txt")
    with open(temp_file_path, 'w') as f:
        f.write(f"Upload test content: {datetime.now().isoformat()}")
    mytest.upload_file(temp_dir, "tests")

mytest.download_file('downloadtest', 'downloadtest')