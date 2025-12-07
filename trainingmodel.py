import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.utils import resample
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

player_data = pd.read_csv('Data/player_rank_dataset.csv')

rank_counts = player_data['RankName'].value_counts()
max_count = rank_counts.max()

upsampled_frames = []
for rank in rank_counts.index:
    rank_df = player_data[player_data['RankName'] == rank]
    if len(rank_df) < max_count:
        rank_df = resample(rank_df, replace=True, n_samples=max_count, random_state=42)
    upsampled_frames.append(rank_df)

player_balanced = pd.concat(upsampled_frames).reset_index(drop=True)


plt.figure(figsize=(10,6))
player_balanced['RankName'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.xlabel("Rank")
plt.ylabel("Number of players")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("rank_distribution.png")  
plt.close()  

lane_le = LabelEncoder()
player_balanced['Lane'] = lane_le.fit_transform(player_balanced['Lane'])

champ_le = LabelEncoder()
player_balanced['champ'] = champ_le.fit_transform(player_balanced['ChampionName'])


player_balanced['KDA'] = (player_balanced['kills'] + player_balanced['assists']) / player_balanced['deaths'].replace(0, 1)
player_balanced['GoldPerMin'] = player_balanced['TotalGold'] / player_balanced['GameDuration'].replace(0, 1)
player_balanced['VisionPerMin'] = player_balanced['visionScore'] / player_balanced['GameDuration'].replace(0, 1)


y = player_balanced['RankName']
feature_cols = ['Lane', 'champ', 'kills', 'deaths', 'assists', 'KDA', 'GoldPerMin', 'VisionPerMin']
X = player_balanced[feature_cols]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


ModelRf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    class_weight='balanced',
    random_state=42
)
ModelRf.fit(X_train, y_train)


y_pred = ModelRf.predict(X_test)

print("acuracy on test set:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred, average='macro'))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=ModelRf.classes_,
            yticklabels=ModelRf.classes_)
plt.xlabel("predicted Rank")
plt.ylabel("actual Rank")
plt.title("confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()  

joblib.dump(ModelRf, 'Data/ModelRf.pkl')
joblib.dump(champ_le, 'Data/champ_le.pkl')
joblib.dump(lane_le, 'Data/lane_le.pkl')

