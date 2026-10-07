import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

IN_PATH = Path("Android_Templated.csv")
df = pd.read_csv(IN_PATH, dtype=str, keep_default_na=False)

print(f"loaded {len(df)} rows")
print(f"unique PIDs: {df['pid'].nunique()}")

# ---- Basic per-session summary features ----
df['event_id'] = df['event_id'].astype(int)

# Aggregations per PID
session_summary = df.groupby('pid').agg(
    length=('event_id', 'size'),
    n_unique=('event_id', 'nunique'),
    tag_diversity=('tag', 'nunique'),
).reset_index()

# Level counts per session
level_counts = (
    df.assign(is_E=(df['level'] == 'E').astype(int),
              is_W=(df['level'] == 'W').astype(int))
      .groupby('pid')[['is_E', 'is_W']]
      .sum()
      .reset_index()
      .rename(columns={'is_E': 'count_E', 'is_W': 'count_W'})
)

session_summary = session_summary.merge(level_counts, on='pid', how='left')

# Fraction of E-level events per session
session_summary['frac_E'] = session_summary['count_E'] / session_summary['length']
session_summary['frac_W'] = session_summary['count_W'] / session_summary['length']

print(f"sessions: {len(session_summary)}")
print(session_summary.head())

# ---- Bag-of-events per session ----
bag = (
    df.groupby(['pid', 'event_id'])
      .size()
      .unstack(fill_value=0)
      .rename_axis(columns=None)
      .reset_index()
)
bag.columns = ['pid'] + [f'event_{int(c)}' for c in bag.columns[1:]]
print(f"bag shape: {bag.shape}")

# ---- Merge ----
features = session_summary.merge(bag, on='pid', how='left')
print(f"feature matrix: {features.shape}")

# ---- Build feature matrix (size-independent) ----
SUMMARY_FEATURES = ['n_unique', 'tag_diversity', 'frac_E', 'frac_W']
EVENT_FEATURES = [c for c in bag.columns if c.startswith('event_')]

features = session_summary.merge(bag, on='pid', how='left')
print(f"feature matrix: {features.shape}")

X = features[SUMMARY_FEATURES + EVENT_FEATURES]
print(f"X shape (model input): {X.shape}")

# ---- Scale + fit ----
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

iso = IsolationForest(n_estimators=200, contamination=0.05,
                      random_state=42, n_jobs=-1)
iso.fit(X_scaled)

features['anomaly_score'] = iso.decision_function(X_scaled)
features['is_anomaly']    = (iso.predict(X_scaled) == -1).astype(int)

print(f"\nflagged anomalies: {features['is_anomaly'].sum()} / {len(features)}")
# ---- Save ----
features.to_csv('Android_Sessions_Scored.csv', index=False)
print("wrote Android_Sessions_Scored.csv")

# ---- Figure 1: Anomaly score distribution ----
plt.figure(figsize=(10, 4))
plt.hist(features['anomaly_score'], bins=40, edgecolor='black')
plt.axvline(features.loc[features['is_anomaly'] == 1, 'anomaly_score'].max(),
            color='red', linestyle='--', label='threshold')
plt.xlabel('IsolationForest anomaly score')
plt.ylabel('Number of sessions')
plt.title('Anomaly score distribution across Android sessions')
plt.legend()
plt.savefig('android_anomaly_scores.png', dpi=140, bbox_inches='tight')
plt.close()
print("saved android_anomaly_scores.png")

# ---- Figure 2: Feature profile of anomalies vs. normal ----
summary_feats = ['n_unique', 'tag_diversity', 'frac_E', 'frac_W']
profile = features.groupby('is_anomaly')[summary_feats].mean().T
profile.columns = ['Normal', 'Anomaly']
profile.plot.barh(figsize=(8, 5))
plt.title('Feature averages: Normal vs. Flagged sessions')
plt.xlabel('Mean value')
plt.tight_layout()
plt.savefig('android_profile.png', dpi=140, bbox_inches='tight')
plt.close()
print("saved android_profile.png")

# ---- Figure 3: Feature differences (most discriminating) ----
overall = features[summary_feats].mean()
anom = features[features['is_anomaly'] == 1][summary_feats].mean()
diff = ((anom - overall) / overall.replace(0, np.nan)).sort_values()

print("\nFeature differences (anomalies vs. overall, as %):")
print((diff * 100).round(1).to_string())

diff.plot.barh(figsize=(8, 5))
plt.title('Feature deviations in flagged sessions vs. overall mean')
plt.xlabel('% difference from overall mean')
plt.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig('android_deviations.png', dpi=140, bbox_inches='tight')
plt.close()
print("saved android_deviations.png")

# Re-fit at three contamination levels, keep the flagged sets
results = {}
for c in [0.01, 0.05, 0.10]:
    iso = IsolationForest(n_estimators=200, contamination=c,
                          random_state=42, n_jobs=-1)
    iso.fit(X_scaled)
    flags = (iso.predict(X_scaled) == -1).astype(int)
    results[c] = set(features.loc[flags == 1, 'pid'].tolist())

# Are the small sets nested inside the bigger ones?
print("1% flagged PIDs:", sorted(results[0.01]))
print("5% flagged PIDs:", sorted(results[0.05]))
print("10% flagged PIDs:", sorted(results[0.10]))
print()
print(f"1% ⊂ 5%: {results[0.01].issubset(results[0.05])}")
print(f"5% ⊂ 10%: {results[0.05].issubset(results[0.10])}")

flagged_pids = features[features['is_anomaly'] == 1]['pid'].tolist()
print(f"\nflagged PIDs ({len(flagged_pids)}): {sorted(flagged_pids)}")

for pid in flagged_pids[:5]:
    print(f"\n=== PID {pid} ===")
    row = features[features['pid'] == pid].iloc[0]
    print(f"  length={row['length']}  n_unique={row['n_unique']}  "
          f"tag_diversity={row['tag_diversity']}  "
          f"frac_E={row['frac_E']:.3f}  frac_W={row['frac_W']:.3f}  "
          f"score={row['anomaly_score']:.4f}")
    sample = df[df['pid'] == pid].head(8)[['level', 'tag', 'message']]
    print(sample.to_string())

joblib.dump(iso, 'android_isoforest.joblib')
joblib.dump(scaler, 'android_scaler.joblib')       # scaler travels with the model