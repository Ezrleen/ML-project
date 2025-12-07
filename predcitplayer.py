import pandas as pd
import joblib
import numpy as np

# -----------------------------
# Step 1: Load saved model
# -----------------------------
rf_clf = joblib.load('Data/rf_model.pkl')
champ_le = joblib.load('Data/champ_le.pkl')
print("Model and encoder loaded!")

# -----------------------------
# Step 2: Load player's match data
# -----------------------------
player_data = pd.read_csv('tt.csv')


# -----------------------------
# Step 3: Encode categorical features
# -----------------------------
# Lane mapping
lane_map = {'TOP':0, 'JUNGLE':1, 'MID':2, 'BOTTOM':3, 'UTILITY':4}
player_data['lane_encoded'] = player_data['lane'].map(lane_map).fillna(2).astype(int)

# Champion encoding
player_data['champ_encoded'] = champ_le.transform(player_data['ChampionName'].fillna('Unknown'))

# -----------------------------
# Step 4: Feature engineering
# -----------------------------
player_data['KDA'] = (player_data['kills'] + player_data['assists']) / player_data['deaths'].replace(0, 1)
player_data['GoldPerMin'] = player_data['totalGold'] / player_data['gameDuration'].replace(0, 1)
player_data['VisionPerMin'] = player_data['visionScore'] / player_data['gameDuration'].replace(0, 1)

# -----------------------------
# Step 5: Prepare features
# -----------------------------
feature_cols = ['lane_encoded', 'champ_encoded', 'kills', 'deaths', 'assists',
                'KDA', 'GoldPerMin', 'VisionPerMin']
X_player = player_data[feature_cols]

# -----------------------------
# Step 6: Predict ranks
# -----------------------------
player_data['PredictedRank'] = rf_clf.predict(X_player)

# -----------------------------
# Step 7: Most relevant rank
# -----------------------------
most_relevant_rank = player_data['PredictedRank'].mode()[0]
print("Most relevant rank for this player:", most_relevant_rank)

# -----------------------------
# Step 8: Calculate top champions by win rate
# -----------------------------
champ_stats = player_data.groupby('ChampionName').agg(games_played=('match_id', 'count'),wins=('win', 'sum'))# Win rate as percentage
champ_stats['win_rate'] = (champ_stats['wins'] / champ_stats['games_played']) * 100
champ_stats['win_rate'] = champ_stats['win_rate'].round(2)
champ_stats = champ_stats[champ_stats['games_played'] >= 3]
top_champs = champ_stats.sort_values(by='win_rate', ascending=False).head(3)
print("\nTop 3 champions by win rate (%):")
print(top_champs)


# -----------------------------
# Step 9: Save predictions
# -----------------------------
player_data.to_csv('tt_predicted.csv', index=False)
print("Predictions saved to tt_predicted.csv")
