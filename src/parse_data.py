import os
from dotenv import load_dotenv

load_dotenv()

DATASET_BASE_PATH = os.getenv('DATASET_BASE_PATH')
DATASET_CACHE_PATH = os.getenv('DATASET_CACHE_PATH')
TMP_PATH = os.getenv('TMP_DIR')

os.environ["TMPDIR"] = "/data/jmacambi/tmp"
os.environ["HF_HOME"] = DATASET_CACHE_PATH
os.environ["HF_DATASETS_CACHE"] = DATASET_CACHE_PATH

import pandas as pd
import numpy as np
from datasets import Dataset

if __name__ == '__main__':
    ds = Dataset.from_parquet(f"{DATASET_BASE_PATH}/spectrogram.parquet")
    n = np.array(ds[0]['spectrogram'])
    print(n.shape)
    print(n.min())
    print(n.max())