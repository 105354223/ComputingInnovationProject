import re
import pandas as pd
from pathlib import Path

DATA_PATH = 'Android_Parsed.csv'
df = pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False)
print(f"loaded {len(df)} rows")

# Format time to be UTC
df['timestamp'] = pd.to_datetime(
    "2020-" + df['date'] + " " + df['time'],
    format="%Y-%m-%d %H:%M:%S.%f",
    errors="coerce",
    utc=True,
)
# print(f"bad timestamps: {df['timestamp'].isna().sum()}")
df = df.dropna(subset=['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

df = df[df['message'].astype(bool)].reset_index(drop=True)
# print(f"after dropping empty messages: {len(df)}")

# print(f"unique PIDs: {df['pid'].nunique()}")

blk    = re.compile(r'blk_-?\d+')
portIP = re.compile(r'\d+\.\d+\.\d+\.\d+:\d+')
ip     = re.compile(r'\d+\.\d+\.\d+\.\d+')
hexnum = re.compile(r'0x[0-9a-fA-F]+')
uuid   = re.compile(r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-'
                    r'[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b')
num    = re.compile(r'\d+')

# template function to convert log messages into variables 
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

df['full_template'] = df['tag'] + ': ' + df['template']

print(f"distinct templates (before filtering): {df['full_template'].nunique()}")

# Android logs have many unique events thus this filter allows for the most meaningful data
df['full_template'] = df['template']
counts = df['full_template'].value_counts()
keep = counts[counts >= 500].index
print(f"keeping {len(keep)} templates with >=500 occurrences")
df = df[df['full_template'].isin(keep)].copy()
print(f"rows after filter: {len(df)}")

vocab = {t: i for i, t in enumerate(sorted(df['full_template'].unique()))}
df['event_id'] = df['full_template'].map(vocab)
print(f"vocab size: {len(vocab)}")

df.to_csv("Android_Templated.csv", index=False)