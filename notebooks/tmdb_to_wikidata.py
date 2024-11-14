import pandas as pd

data_path = '../data'
tmdb_data = pd.read_csv(f'{data_path}/TMDB_movie_dataset_v11.csv')


import requests
from tqdm import tqdm
tqdm.pandas()
def get_wikidata_id(tmdb_id):
    url = "https://query.wikidata.org/sparql"
    query = """
    SELECT ?item WHERE {
      ?item wdt:P4947 "%s" .
    }
    """ % tmdb_id
    headers = {
        "User-Agent": "YourAppName/1.0 (your_email@example.com)"
    }
    response = requests.get(url, params={'query': query, 'format': 'json'}, headers=headers)
    data = response.json()
    if data['results']['bindings']:
        return data['results']['bindings'][0]['item']['value'].split("/")[-1]
    else:
        return None

tmdb_data['wikidata_id'] = tmdb_data['id'].progress_apply(get_wikidata_id)
tmdb_data.to_csv(f'{data_path}/TMDB_movie_dataset_v12.csv', index=False)