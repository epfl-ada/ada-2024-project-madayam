from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent.parent

CONFIG = {
    'data_path': ROOT_PATH / 'ADA Data',
    'cmu_path': ROOT_PATH / 'ADA Data/CMU',
    'tmdb_path': ROOT_PATH / 'ADA Data/TMDB',
}