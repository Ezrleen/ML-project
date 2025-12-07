import pandas as pd


matchstats = pd.read_csv('Data/MatchStatsTbl.csv')
summonermatch = pd.read_csv('Data/SummonerMatchTbl.csv')

matchtbl = pd.read_csv('Data/MatchTbl.csv')
matchtbl = matchtbl[ matchtbl['QueueType'] == "CLASSIC" ]

ranktbl = pd.read_csv('Data/RankTbl.csv')
championtbl = pd.read_csv('Data/ChampionTbl.csv')
teamstats = pd.read_csv('Data/TeamMatchTbl.csv')

playerdata = matchstats.merge(
    summonermatch,
    left_on='SummonerMatchFk',
    right_on='SummonerMatchId',

)

playerdata = playerdata.merge(
    matchtbl,
    left_on='MatchFk',
    right_on='MatchId',
    
)

playerdata = playerdata.merge(
    ranktbl,
    left_on='RankFk',
    right_on='RankId',
)

playerdata = playerdata.merge(
    championtbl,
    left_on='ChampionFk',
    right_on='ChampionId',
    how='left'
)

drop_cols = [
    'MatchStatsId', 'SummonerMatchFk', 'SummonerMatchId', 'SummonerFk',
    'MatchId', 'RankFk', 'ChampionFk', 'ChampionId', 'Patch',
    'PrimaryKeyStone', 'PrimarySlot1', 'PrimarySlot2', 'PrimarySlot3',
    'SecondarySlot1', 'SecondarySlot2', 'SummonerSpell1', 'SummonerSpell2',
    'item1','item2','item3','item4','item5','item6'
]
playerdata = playerdata.drop(columns=drop_cols)

playerdata.to_csv('Data/player_rank_dataset.csv', index=False)
print("Player rank dataset saved")
