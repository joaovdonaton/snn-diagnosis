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


def get_file(filename): 
    return f'{DATASET_BASE_PATH}/{filename}'


def get_participant_phenotype(participant_id):
    data = pd.read_csv(get_file('phenotype.tsv'), sep='\t')
    
    return data[data['participant_id'] == participant_id].iloc[0]


def get_session_features(session_id, task_name):
    """
    Note: each recording is identified by a session_id and a task name
    """
    data = pd.read_csv(get_file('static_features.tsv'), sep='\t')

    return data[(data['session_id'] == session_id) & (data['task_name'] == task_name)].iloc[0]


if __name__ == '__main__':
    ds = Dataset.from_parquet(get_file('spectrogram.parquet'))
    n = np.array(ds[0]['spectrogram'])
    print(ds[0].keys())
    print(n.shape)
    print(n.min())
    print(n.max())

    #print(ds[0]['session_id'])


    print(get_session_features('07031CC3', 'Animal-fluency'))