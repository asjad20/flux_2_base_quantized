import os
from huggingface_hub import snapshot_download, login
from dotenv import load_dotenv

load_dotenv()

login(token=os.getenv("HF_TOKEN"))
snapshot_download('black-forest-labs/FLUX.2-klein-base-9B')
snapshot_download('Asjad1020/flux2klein-transformer-qint8')
print('Done!')