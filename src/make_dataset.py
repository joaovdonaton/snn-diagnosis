import os
from dotenv import load_dotenv

load_dotenv()

DATASET_BASE_PATH = os.getenv('DATASET_BASE_PATH')
DATASET_CACHE_PATH = os.getenv('DATASET_CACHE_PATH')
TMP_PATH = os.getenv('TMP_DIR')
DEFAULT_DATA_PATH = os.getenv('DEFAULT_DATA_PATH')

os.environ["TMPDIR"] = TMP_PATH
os.environ["HF_HOME"] = DATASET_CACHE_PATH
os.environ["HF_DATASETS_CACHE"] = DATASET_CACHE_PATH

import parse_data
import numpy as np
from datasets import Dataset
import pandas as pd
import argparse
import torch.utils.data as torchdata
 
# TODO: I think we might be able to remove the is_control_group label from our multi-label labels for each sample

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


def load_dataset(path):
    d = np.load(path, allow_pickle=True) # need to allow deserializing because of how we stored some of the data
    
    return (d['metadata'], d['train_set'], d['train_labels'], d['train_origins'], d['validation_set'], d['validation_labels'], d['validation_origins'],
        d['test_set'], d['test_labels'], d['test_origins'])


class MatrixDataset(torchdata.Dataset):
    """
    Class for storing and retrieving 2d spectrogram-like data
    """
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i], self.labels[i]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', default=f'{DEFAULT_DATA_PATH}/dataset.npz', required=True)
    parser.add_argument('-s', '--size', help='size out of total samples in dataset, will be split 80/10/10 for train/val/test', required=True)
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
    print('Creating split...')
    train_size = int(MAX_SAMPLES*0.8)
    val_size = int(MAX_SAMPLES*0.1) # same for test size

    train_rows = ds.iloc[:train_size]
    train_set = train_rows[DATASET_TYPE].to_numpy()
    train_labels, train_origins = make_labels_and_origin(train_rows, phenotype_data)
    
    validation_rows = ds.iloc[train_size:train_size+val_size]
    validation_set = validation_rows[DATASET_TYPE].to_numpy()
    validation_labels, validation_origins = make_labels_and_origin(validation_rows, phenotype_data)
    
    test_rows = ds.iloc[train_size+val_size:]
    test_set = test_rows[DATASET_TYPE].to_numpy()
    test_labels, test_origins = make_labels_and_origin(test_rows, phenotype_data)
    
    print(f'Writing npz to "{args.output}"...')
    np.savez(args.output,
        metadata=[f'Total size: {MAX_SAMPLES}', f'Type: {DATASET_TYPE}'],
        train_set=train_set,
        train_labels=train_labels,
        train_origins=train_origins,
        validation_set=validation_set,
        validation_labels=validation_labels,
        validation_origins=validation_origins,
        test_set=test_set,
        test_labels=test_labels,
        test_origins=test_origins,
    )