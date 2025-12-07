import pandas as pd

# --- Load CSVs from the Data folder ---
match_stats = pd.read_csv('Data/MatchStatsTbl.csv')
summoner_match = pd.read_csv('Data/SummonerMatchTbl.csv')
match_tbl = pd.read_csv('Data/MatchTbl.csv')
rank_tbl = pd.read_csv('Data/RankTbl.csv')
champion_tbl = pd.read_csv('Data/ChampionTbl.csv')
team_stats = pd.read_csv('Data/TeamMatchTbl.csv')

# --- Merge player-level tables ---
player_data = match_stats.merge(
    summoner_match,
    left_on='SummonerMatchFk',
    right_on='SummonerMatchId',
    how='inner'
)
player_data = player_data.merge(
    match_tbl,
    left_on='MatchFk',
    right_on='MatchId',
    how='inner'
)
player_data = player_data.merge(
    rank_tbl,
    left_on='RankFk',
    right_on='RankId',
    how='inner'
)
player_data = player_data.merge(
    champion_tbl,
    left_on='ChampionFk',
    right_on='ChampionId',
    how='left'
)

# --- Drop unnecessary columns ---
drop_cols = [
    'MatchStatsId', 'SummonerMatchFk', 'SummonerMatchId', 'SummonerFk',
    'MatchId', 'RankFk', 'ChampionFk', 'ChampionId', 'Patch',
    'PrimaryKeyStone', 'PrimarySlot1', 'PrimarySlot2', 'PrimarySlot3',
    'SecondarySlot1', 'SecondarySlot2', 'SummonerSpell1', 'SummonerSpell2',
    'item1','item2','item3','item4','item5','item6'
]
player_data = player_data.drop(columns=drop_cols)

# --- Save dataset ready for upsampling & modeling ---
player_data.to_csv('Data/player_rank_dataset_noitems.csv', index=False)
print("Player rank dataset (no items) ready!")
