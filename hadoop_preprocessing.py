import re
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# Read parsed csv file
DATA_PATH = 'HDFS_Parsed.csv'
df = pd.read_csv(DATA_PATH, dtype = str)
# convert date and time fields to single timestamp field
df['timestamp'] = pd.to_datetime(
    df['date'] + df['time'], format = '%y%m%d%H%M%S', errors = 'coerce', utc = True
)
df = df.sort_values('timestamp')
df = df.drop(columns = ["date", "time"])

df = df[['timestamp', 'pid', 'level', 'component', 'message', 'blockId', 'sourceIP', 'destinationIP']]

# drop NA blockId fields
df = df.dropna(subset=["blockId"])
print(f"unique blocks: {df['blockId'].nunique()}")

# log parsing to compile into events
blk = re.compile(r'blk_-?\d+')
portIP = re.compile(r'\d+\.\d+\.\d+\.\d+:\d+')
ip = re.compile(r'\d+\.\d+\.\d+\.\d+')
dir = re.compile(r'(/user/root/)\w+')
num = re.compile(r'\d+')

def template(msg):
    msg = blk.sub('blk_<*>', msg)
    msg = portIP.sub('<IP:PORT>', msg)
    msg = ip.sub('<IP>', msg)
    msg = dir.sub(r'\1<JOB>', msg)
    msg = num.sub('<NUM>', msg)
    return msg

df['template'] = df['message'].apply(template)
print(df['template'].value_counts().head(30).to_string())

eventID = {t: i for i, t in enumerate(sorted(df['template'].unique()))}
df['event_id'] = df['template'].map(eventID)

print(f"eventID size: {len(eventID)}")
print(list(eventID.items())[:5])

# df.to_csv('HDFS_Templated.csv', index=False)

traces = (
    df.sort_values('timestamp')
    .groupby('blockId')['event_id']
    .apply(list)
)

print(f"number of traces: {len(traces)}")
print(f"Frist 5 traces:")
print(traces.head(5))
print()
print("trace length distribution:")
print(traces.apply(len).describe())

labels = pd.read_csv('basicDatasets/Log Anomaly Detection/anomaly_label.csv')
print(labels.columns.tolist())
print(labels.head())

labels = labels.rename(columns={'BlockId': 'blockId', 'Label': 'label'})
labels = labels.set_index('blockId')['label']

traces_df = traces.reset_index()
traces_df.columns = ['blockId', 'sequence']
traces_df['label'] = traces_df['blockId'].map(labels)

print(traces_df['label'].value_counts())
print(traces_df['label'].value_counts(normalize=True))

print(df.head(5))

traces_df.to_pickle('hdfs_traces_labeled.pkl')

# LOG_PATH = "basicDatasets/Log Anomaly Detection/HDFS.log"
# MAX_LINES = 200_000  # small slice while testing; set to None for everything

# # ---- 1. Parse: split each line into fields, find the block id(s) ----
# HEADER = re.compile(r"(\d{6}) (\d{6}) (\d+) (\S+) ([^\s:]+): (.*)")

# rows = []
# skipped = 0
# with open(LOG_PATH, "r") as file:
#     for i, line in enumerate(file):
#         if MAX_LINES and i >= MAX_LINES:
#             break
#         m = HEADER.match(line.rstrip("\r\n"))
#         blocks = re.findall(r"blk_-?\d+", line)
#         if not m or not blocks:
#             skipped += 1
#             continue
#         date, time, pid, level, component, message = m.groups()
#         for block in set(blocks):
#             rows.append({"blockId": block, "level": level,
#                          "component": component, "message": message})

# df = pd.DataFrame(rows)
# print(f"parsed {len(df)} rows, skipped {skipped}")

# # ---- 2. Make an "event" by blanking out the numbers in each message ----
# df["event"] = df["message"].str.replace(r"blk_-?\d+|[\d.:/]+", "<*>", regex=True)
# print(f"{df['event'].nunique()} unique events")

# # ---- 3. Count how often each event happens in each block ----
# counts = pd.crosstab(df["blockId"], df["event"])

# # ---- 4. Flag the blocks that look unusual ----
# model = IsolationForest(contamination=0.03, random_state=42)
# counts["anomaly"] = model.fit_predict(counts) == -1

# print(counts["anomaly"].sum(), "blocks flagged")
# counts[counts["anomaly"]].to_csv("HDFS_Flagged.csv")
# df.to_csv("HDFS_Parsed.csv", index=False)