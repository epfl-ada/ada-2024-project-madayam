from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent.parent

CONFIG = {
    'data_path': ROOT_PATH / 'ADAData',
    'cmu_path': ROOT_PATH / 'ADAData/CMU',
    'tmdb_path': ROOT_PATH / 'ADAData/TMDB',
}