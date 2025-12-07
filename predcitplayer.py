import pandas as pd
import joblib
import numpy as np

ModelRf = joblib.load('Data/ModelRf.pkl')
champ_le = joblib.load('Data/champ_le.pkl')
print("Model and encoder loaded!")


player_data = pd.read_csv('PlayerDataSetFetch.csv')



lane_map = {'TOP':0, 'JUNGLE':1, 'MID':2, 'BOTTOM':3, 'UTILITY':4}
player_data['Lane'] = player_data['lane'].map(lane_map).fillna(2).astype(int)


player_data['champ'] = champ_le.transform(player_data['ChampionName'].fillna('Unknown'))


player_data['KDA'] = (player_data['kills'] + player_data['assists']) / player_data['deaths'].replace(0, 1)
player_data['GoldPerMin'] = player_data['totalGold'] / player_data['gameDuration'].replace(0, 1)
player_data['VisionPerMin'] = player_data['visionScore'] / player_data['gameDuration'].replace(0, 1)


feature_cols = ['Lane', 'champ', 'kills', 'deaths', 'assists',
                'KDA', 'GoldPerMin', 'VisionPerMin']
X_player = player_data[feature_cols]


player_data['PredictedRank'] = ModelRf.predict(X_player)


most_relevant_rank = player_data['PredictedRank'].mode()[0]
print("Most relevant rank for this player:", most_relevant_rank)


champ_stats = player_data.groupby('ChampionName').agg(games_played=('match_id', 'count'),wins=('win', 'sum'))# Win rate as percentage
champ_stats['win_rate'] = (champ_stats['wins'] / champ_stats['games_played']) * 100
champ_stats['win_rate'] = champ_stats['win_rate'].round(2)
champ_stats = champ_stats[champ_stats['games_played'] >= 3]
top_champs = champ_stats.sort_values(by='win_rate', ascending=False).head(3)
print("\nTop 3 champions by win rate (%):")
print(top_champs)


player_data.to_csv('predicted.csv', index=False)
print("Predictions saved to tt_predicted.csv")
