import pandas as pd

data_path = '../data'
tmdb_data = pd.read_csv(f'{data_path}/TMDB_movie_dataset_v11.csv')


import requests
from tqdm import tqdm
tqdm.pandas()

# def get_wikidata_id(tmdb_id):
#     url = "https://query.wikidata.org/sparql"
#     query = """
#     SELECT ?item WHERE {
#       ?item wdt:P4947 "%s" .
#     }
#     """ % tmdb_id
#     headers = {
#         "User-Agent": "YourAppName/1.0 (your_email@example.com)"
#     }
#     response = requests.get(url, params={'query': query, 'format': 'json'}, headers=headers)
#     data = response.json()
#     if data['results']['bindings']:
#         return data['results']['bindings'][0]['item']['value'].split("/")[-1]
#     else:
#         return None
#
# tmdb_data['wikidata_id'] = tmdb_data['id'].progress_apply(get_wikidata_id)
# tmdb_data.to_csv(f'{data_path}/TMDB_movie_dataset_v12.csv', index=False)




import aiohttp
import asyncio
import pandas as pd

async def fetch_wikidata_id(session, tmdb_ids):
    query = f"""
    SELECT ?item ?tmdb_id WHERE {{
      VALUES ?tmdb_id {{"{" ".join(f'"{tmdb_id}"' for tmdb_id in tmdb_ids)}}}
      ?item wdt:P4947 ?tmdb_id .
    }}
    """
    url = "https://query.wikidata.org/sparql"
    headers = {"User-Agent": "YourAppName/1.0 (your_email@example.com)"}
    async with session.get(url, params={'query': query, 'format': 'json'}, headers=headers) as response:
        return await response.json()

async def batch_request(tmdb_ids, batch_size=100):
    results = []
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(tmdb_ids), batch_size):
            batch = tmdb_ids[i:i + batch_size]
            data = await fetch_wikidata_id(session, batch)
            results.extend(data['results']['bindings'])
    return results

# List of TMDb IDs
# tmdb_ids = ["ID1", "ID2", "ID3", ...]  # Replace with actual TMDb IDs
# results = asyncio.run(batch_request(tmdb_ids))

tmdb_ids = tmdb_data['id'].tolist()
results = asyncio.run(batch_request(tmdb_ids))

wikidata_ids = {result['tmdb_id']['value']: result['item']['value'].split("/")[-1] for result in results}
tmdb_data['wikidata_id'] = tmdb_data['id'].map(wikidata_ids)
tmdb_data.to_csv(f'{data_path}/TMDB_movie_dataset_v12.csv', index=False)