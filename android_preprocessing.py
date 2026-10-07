import re
import pandas as pd
from pathlib import Path

# keep_default_na=False: empty fields stay as '' not NaN
DATA_PATH = 'Android_Parsed.csv'
df = pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False)
print(f"loaded {len(df)} rows")

# ---- Build timestamp ----
df['timestamp'] = pd.to_datetime(
    "2020-" + df['date'] + " " + df['time'],
    format="%Y-%m-%d %H:%M:%S.%f",
    errors="coerce",
    utc=True,
)
print(f"bad timestamps: {df['timestamp'].isna().sum()}")
df = df.dropna(subset=['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# ---- Drop rows with empty messages (safety net) ----
df = df[df['message'].astype(bool)].reset_index(drop=True)
print(f"after dropping empty messages: {len(df)}")

# ---- Session grouping key = PID ----
print(f"unique PIDs: {df['pid'].nunique()}")

# ---- Templating ----
blk    = re.compile(r'blk_-?\d+')
portIP = re.compile(r'\d+\.\d+\.\d+\.\d+:\d+')
ip     = re.compile(r'\d+\.\d+\.\d+\.\d+')
hexnum = re.compile(r'0x[0-9a-fA-F]+')
uuid   = re.compile(r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-'
                    r'[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b')
num    = re.compile(r'\d+')

def template(msg):
    if not isinstance(msg, str):
        return ''
    msg = blk.sub('blk_<*>', msg)
    msg = portIP.sub('<IP:PORT>', msg)
    msg = ip.sub('<IP>', msg)
    msg = hexnum.sub('<HEX>', msg)
    msg = uuid.sub('<UUID>', msg)
    msg = num.sub('<NUM>', msg)
    return msg

df['template'] = df['message'].apply(template)

# Include ONLY the tag, not the level
df['full_template'] = df['tag'] + ': ' + df['template']

print(f"distinct templates (before filtering): {df['full_template'].nunique()}")

# ---- Drop rare templates (stricter) ----
df['full_template'] = df['template']       # drop the tag from template
counts = df['full_template'].value_counts()
keep = counts[counts >= 500].index
print(f"keeping {len(keep)} templates with >=500 occurrences")
df = df[df['full_template'].isin(keep)].copy()
print(f"rows after filter: {len(df)}")

# ---- Assign event IDs ----
vocab = {t: i for i, t in enumerate(sorted(df['full_template'].unique()))}
df['event_id'] = df['full_template'].map(vocab)
print(f"vocab size: {len(vocab)}")

df.to_csv("Android_Templated.csv", index=False)