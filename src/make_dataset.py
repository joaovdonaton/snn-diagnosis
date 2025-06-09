import os
from dotenv import load_dotenv

load_dotenv()

DATASET_BASE_PATH = os.getenv('DATASET_BASE_PATH')
DATASET_CACHE_PATH = os.getenv('DATASET_CACHE_PATH')
TMP_PATH = os.getenv('TMP_DIR')

os.environ["TMPDIR"] = "/data/jmacambi/tmp"
os.environ["HF_HOME"] = DATASET_CACHE_PATH
os.environ["HF_DATASETS_CACHE"] = DATASET_CACHE_PATH

import parse_data
import numpy as np
from datasets import Dataset
import pandas as pd
import argparse


def make_labels_and_origin(dataset_rows, phenotype_data):
    """
    Extract labels for the participant involved in each session+task. Also returns a information related to what patient, session, task (aka the origin) that the
    spectrogram corresponds to, so we can trace back if needed for further analysis.
    """
    labels, origins = [], []
    for _, row in dataset_rows.iterrows():
        label_row = phenotype_data[phenotype_data['participant_id'] == row['participant_id']][['is_control_participant', 'voice', 
        'neurological_and_neurodegenerative', 'mood_and_psychiatric', 'respiratory']]
        labels.append(label_row.to_numpy())
        origins.append(np.array([row['participant_id'], row['session_id'], row['task_name']]))

    labels = np.stack(labels).squeeze()
    origins = np.stack(origins).squeeze()

    return (labels, origins)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', default='./', required=True)
    parser.add_argument('-s', '--size', help='size out of total samples in dataset, will be split 80/10/10 for train/val/test')
    parser.add_argument('-t', '--type', choices=['spectrogram', 'mfcc'], required=True,)

    args = parser.parse_args()
    DATASET_TYPE = args.type

    print('Preparing dataset...')
    ds = Dataset.from_parquet(parse_data.get_file('spectrogram.parquet' if DATASET_TYPE == 'spectrogram' else 'mfcc.parquet'))
    phenotype_data = parse_data.get_classifying_data()

    MAX_SAMPLES = len(ds) if args.size is None else int(args.size) 

    ds = ds.to_pandas()
    ds = ds.sample(frac=1).reset_index(drop=True) # shuffle rows
    ds = ds.iloc[:MAX_SAMPLES]

    # make each spectrogram matrix a proper ndarray
    new_data_col = []
    for entry in ds[DATASET_TYPE]:
        new_data_col.append(np.stack(entry))
    ds[DATASET_TYPE] = pd.Series(new_data_col)

    # build train/test/val split
    train_size = int(MAX_SAMPLES*0.8)
    val_size = int(MAX_SAMPLES*0.1) # same for test size

    train_rows = ds.iloc[:train_size]
    train_set = train_rows[DATASET_TYPE].to_numpy()

    train_labels, train_origins = make_labels_and_origin(train_rows, phenotype_data)

    print(train_labels)
    print(train_origins)