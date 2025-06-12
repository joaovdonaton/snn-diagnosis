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
from tqdm import tqdm
 
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
    parser.add_argument('-s', '--size', help='size out of total samples in dataset, will be split 80/10/10 for train/val/test. Defaults to full dataset', required=False)
    parser.add_argument('-t', '--type', choices=['spectrogram', 'mfcc'], required=True,)
    parser.add_argument('-w', '--window_size', default=512, help='Size of windows for chunking individual samples, if T < window_size, then zero pad')
    parser.add_argument('-l', '--hop_length', default=256, help='Size of hop of windows for chunking')

    args = parser.parse_args()
    DATASET_TYPE = args.type

    print('Preparing dataset...')
    ds = Dataset.from_parquet(parse_data.get_file('spectrogram.parquet' if DATASET_TYPE == 'spectrogram' else 'mfcc.parquet'))
    phenotype_data = parse_data.get_classifying_data()

    MAX_SAMPLES = len(ds) if args.size is None else int(args.size) 
    WINDOW_SIZE = args.window_size
    HOP_LENGTH = args.hop_length

    ds = ds.to_pandas()
    ds = ds.sample(frac=1).reset_index(drop=True) # shuffle rows
    ds = ds.iloc[:MAX_SAMPLES]

    # matrix data preprocessing, windowing + padding
    # turns ds into a new dataframe, where the spectrogram column is now a window, and each row's data is repeated for windows of same source sample audio
    new_ds = pd.DataFrame(columns=ds.columns)

    print('Windowing...')
    for i in tqdm(range(len(ds)), ncols=120):
        raw_matrix = np.stack(ds.iloc[i][DATASET_TYPE]) # convert data into proper 2d array
        timesteps = raw_matrix.shape[1]

        # per sample normalize
        if DATASET_TYPE == 'spectrogram':
            mean, std = raw_matrix.mean(), raw_matrix.std()
            raw_matrix = (raw_matrix - mean) / std
        
        # will be used to add CMNV or channel wise later
        elif DATASET_TYPE == 'mfcc':
            mean, std = raw_matrix.mean(), raw_matrix.std()
            raw_matrix = (raw_matrix - mean) / std

        # edge case timesteps == WINDOW_SIZE????
        end = False # flag for when we reach end to avoid creating additional window
        for j in range(0, timesteps, HOP_LENGTH):
            w = raw_matrix[:, j:j+WINDOW_SIZE]

            if w.shape[1] < WINDOW_SIZE: # padding 
                w = np.concatenate((w, np.zeros((w.shape[0], WINDOW_SIZE-w.shape[1]))), axis=1)
                end = True

            if not np.all(w == 0): # case where we jump passed end
                old_row = ds.iloc[i].copy()
                old_row[DATASET_TYPE] = w
                new_ds.loc[len(new_ds)] = old_row
            
            if end: break

    ds = new_ds

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
        metadata=[f'total_size: {MAX_SAMPLES}', f'type: {DATASET_TYPE}', f'window_size:{WINDOW_SIZE}', f'hop_length:{HOP_LENGTH}'],
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