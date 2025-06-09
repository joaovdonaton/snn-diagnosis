import os
from dotenv import load_dotenv

load_dotenv()

DATASET_BASE_PATH = os.getenv('DATASET_BASE_PATH')
DATASET_CACHE_PATH = os.getenv('DATASET_CACHE_PATH')
TMP_PATH = os.getenv('TMP_DIR')

os.environ["TMPDIR"] = TMP_PATH
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


def get_classifying_data():
    """
    Reads phenotype.json for all participants in study to extract relevant information for our classification models
    """
    data = pd.read_csv(get_file('phenotype.tsv'), sep='\t')

    data = data[['participant_id', 'age', 'is_control_participant', 'eligible_studies___1', 'eligible_studies___2', 'eligible_studies___3',
    'eligible_studies___4']]

    disorder_categories = {'eligible_studies___1': 'voice', 'eligible_studies___2': 'neurological_and_neurodegenerative', 
    'eligible_studies___3': 'mood_and_psychiatric', 'eligible_studies___4': 'respiratory'}

    # type corrections    
    data['age'] = data['age'].replace('90 and above', '90.0').astype(np.float32)

    data['is_control_participant'] = data['is_control_participant'].replace({'Yes': '1', 'No': '0'}).astype(np.float32)

    # change each category to boolean and rename
    for i in range(1, 5):
        col = f'eligible_studies___{i}'
        data.loc[data[col].notna(), col] = '1'
        data[col] = data[col].fillna('0')
        data[col] = data[col].astype(np.float32)

    data = data.rename(columns=disorder_categories)
    
    # debug
    # print(data.dtypes)
    # print(data)

    return data


if __name__ == '__main__':
    ds = Dataset.from_parquet(get_file('spectrogram.parquet'))
    n = np.array(ds[0]['spectrogram'])
    print(ds[0].keys())
    print(n.shape)
    print(n.min())
    print(n.max())

    #print(ds[0]['session_id'])


    #print(get_session_features('07031CC3', 'Animal-fluency'))

    d = get_classifying_data()
    print(d)