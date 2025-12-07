import requests
import pandas as pd

# -----------------------------
# Configuration
# -----------------------------
API_KEY = "RGAPI-d8c9fed9-a702-450e-a008-6e1d2b771f1f"
ROUTING = "europe"

username = input("give user: ")
tag = input("give tag: ")

# 1. Get PUUID
a = f"https://europe.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{username}/{tag}?api_key={API_KEY}"
response1 = requests.get(a)
data = response1.json()
PUUID = data["puuid"]

# 2. Fetch ONLY RANKED SOLO/DUO match IDs (queue=420)
match_ids_url = (
    f"https://{ROUTING}.api.riotgames.com/lol/match/v5/matches/by-puuid/{PUUID}/ids?queue=420&start=0&count=10&api_key={API_KEY}"
) 

response = requests.get(match_ids_url)
match_ids = response.json()

# -----------------------------
# Step 2: Fetch match details
# -----------------------------
matches_data = []

for match_id in match_ids:
    match_url = f"https://{ROUTING}.api.riotgames.com/lol/match/v5/matches/{match_id}?api_key={API_KEY}"
    match_response = requests.get(match_url).json()
    
    for participant in match_response['info']['participants']:
        if participant['puuid'] == PUUID:
            matches_data.append({
    'match_id': match_id,
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
print(matches_data)
# -----------------------------
# Step 3: Convert to DataFrame
# -----------------------------
dataset = pd.DataFrame(matches_data)

# Optional: derived stats
dataset['KDA'] = (dataset['kills'] + dataset['assists']) / dataset['deaths'].replace(0, 1)
dataset['GoldPerMin'] = dataset['totalGold'] / dataset['gameDuration'].replace(0, 1)
dataset['VisionPerMin'] = dataset['visionScore'] / dataset['gameDuration'].replace(0, 1)
dataset['win'] = dataset['win'].astype(int)
print(dataset)

dataset.to_csv("tt.csv", index=False)
print("Saved as match_history.csv")
