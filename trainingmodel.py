import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.utils import resample

# -----------------------------
# Step 1: Load training dataset
# -----------------------------
train_file = 'Data/player_rank_dataset_noitems.csv'
player_data = pd.read_csv(train_file)

# -----------------------------
# Step 2: Upsample minority ranks
# -----------------------------
# Separate each rank
rank_counts = player_data['RankName'].value_counts()
max_count = rank_counts.max()  # target number of samples per rank

upsampled_frames = []
for rank in rank_counts.index:
    rank_df = player_data[player_data['RankName'] == rank]
    if len(rank_df) < max_count:
        rank_df = resample(rank_df,
                           replace=True,  # upsample with replacement
                           n_samples=max_count,
                           random_state=42)
    upsampled_frames.append(rank_df)

player_data_balanced = pd.concat(upsampled_frames).reset_index(drop=True)
print("Balanced dataset counts:\n", player_data_balanced['RankName'].value_counts())

# -----------------------------
# Step 3: Encode categorical features
# -----------------------------
# Lane
lane_le = LabelEncoder()
player_data_balanced['lane_encoded'] = lane_le.fit_transform(player_data_balanced['Lane'])

# Champion
champ_le = LabelEncoder()
player_data_balanced['champ_encoded'] = champ_le.fit_transform(player_data_balanced['ChampionName'])

# -----------------------------
# Step 4: Feature engineering
# -----------------------------
player_data_balanced['KDA'] = (player_data_balanced['kills'] + player_data_balanced['assists']) / player_data_balanced['deaths'].replace(0, 1)
player_data_balanced['GoldPerMin'] = player_data_balanced['TotalGold'] / player_data_balanced['GameDuration'].replace(0, 1)
player_data_balanced['VisionPerMin'] = player_data_balanced['visionScore'] / player_data_balanced['GameDuration'].replace(0, 1)

# -----------------------------
# Step 5: Define target and features
# -----------------------------
y = player_data_balanced['RankName']
feature_cols = ['lane_encoded', 'champ_encoded', 'kills', 'deaths', 'assists', 'KDA', 'GoldPerMin', 'VisionPerMin']
X = player_data_balanced[feature_cols]

# -----------------------------
# Step 6: Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# -----------------------------
# Step 7: Train Random Forest with class balancing
# -----------------------------
rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    class_weight='balanced',
    random_state=42
)
rf_clf.fit(X_train, y_train)

# -----------------------------
# Step 8: Evaluate model
# -----------------------------
y_pred = rf_clf.predict(X_test)
print("Accuracy on test set:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(rf_clf, 'Data/rf_model.pkl')
print("Model saved as rf_model.pkl")
joblib.dump(champ_le, 'Data/champ_le.pkl')
print("Champion LabelEncoder saved!")