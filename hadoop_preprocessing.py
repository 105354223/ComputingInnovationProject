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

df.to_csv('HDFS_Templated.csv', index=False)

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