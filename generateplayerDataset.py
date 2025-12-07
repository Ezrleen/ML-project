import requests
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_key")
r1=['europe','americas','asia','sea']
ROUTING=input("give server ").lower()
while ROUTING not in r1:
    ROUTING=input("please enter one of these servers europe americas asia sea ").lower()
username = input("give user: ")
tag = input("give tag: ")


a = f"https://{ROUTING}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{username}/{tag}?api_key={API_KEY}"
response1 = requests.get(a)
data = response1.json()
PUUID = data["puuid"]


ids_MatchUrl = (
    f"https://{ROUTING}.api.riotgames.com/lol/match/v5/matches/by-puuid/{PUUID}/ids?queue=420&start=0&count=30&api_key={API_KEY}"
) 

response = requests.get(ids_MatchUrl)
match_ids = response.json()


matches_data = []

for SingleMatch_id in match_ids:
    match_url = f"https://{ROUTING}.api.riotgames.com/lol/match/v5/matches/{SingleMatch_id}?api_key={API_KEY}"
    match_response = requests.get(match_url).json()
    
    for participant in match_response['info']['participants']:
        if participant['puuid'] == PUUID:
            matches_data.append({
    'match_id': SingleMatch_id,
    'kills': participant['kills'],
    'deaths': participant['deaths'],
    'assists': participant['assists'],
    'totalGold': participant['goldEarned'],
    'ChampionName': participant['championName'],
    'lane': participant['lane'],
    'visionScore': participant['visionScore'],
    'gameDuration': match_response['info']['gameDuration'],
    'win': participant['win']
})

dataset = pd.DataFrame(matches_data)

dataset['KDA'] = (dataset['kills'] + dataset['assists']) / dataset['deaths'].replace(0, 1)
dataset['GoldPerMin'] = dataset['totalGold'] / dataset['gameDuration'].replace(0, 1)
dataset['VisionPerMin'] = dataset['visionScore'] / dataset['gameDuration'].replace(0, 1)
dataset['win'] = dataset['win'].astype(int)
print(dataset)
dataset.to_csv("PlayerDataSetFetch.csv", index=False)

