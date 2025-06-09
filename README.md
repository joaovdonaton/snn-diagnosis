# snn-diagnosis

### Setting up project
<b>Creating Virtual Environment</b>
- Setup environment using dependencies under `conda-env.yml` using `conda env create -f conda-env.yml`
- Activate environment using `conda activate snndiag`

<b>Environment Variables</b>
- create a `.env` file in th root directory
- Add the following:
```python
DATASET_BASE_PATH=PATH_HERE # path to dataset source files
DATASET_CACHE_PATH=PATH_HERE # path to cache for huggingface loader
TMP_DIR=PATH_HERE # tmp file path
DEFAULT_DATA_PATH=PATH_HERE # default path to write datasets and other files
```