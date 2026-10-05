import re
import pandas as pd
from sklearn.ensemble import IsolationForest

LOG_PATH = "basicDatasets/Log Anomaly Detection/HDFS.log"
MAX_LINES = 200_000  # small slice while testing; set to None for everything

# ---- 1. Parse: split each line into fields, find the block id(s) ----
HEADER = re.compile(r"(\d{6}) (\d{6}) (\d+) (\S+) ([^\s:]+): (.*)")

rows = []
skipped = 0
with open(LOG_PATH, "r") as file:
    for i, line in enumerate(file):
        if MAX_LINES and i >= MAX_LINES:
            break
        m = HEADER.match(line.rstrip("\r\n"))
        blocks = re.findall(r"blk_-?\d+", line)
        if not m or not blocks:
            skipped += 1
            continue
        date, time, pid, level, component, message = m.groups()
        for block in set(blocks):
            rows.append({"blockId": block, "level": level,
                         "component": component, "message": message})

df = pd.DataFrame(rows)
print(f"parsed {len(df)} rows, skipped {skipped}")

# ---- 2. Make an "event" by blanking out the numbers in each message ----
df["event"] = df["message"].str.replace(r"blk_-?\d+|[\d.:/]+", "<*>", regex=True)
print(f"{df['event'].nunique()} unique events")

# ---- 3. Count how often each event happens in each block ----
counts = pd.crosstab(df["blockId"], df["event"])

# ---- 4. Flag the blocks that look unusual ----
model = IsolationForest(contamination=0.03, random_state=42)
counts["anomaly"] = model.fit_predict(counts) == -1

print(counts["anomaly"].sum(), "blocks flagged")
counts[counts["anomaly"]].to_csv("HDFS_Flagged.csv")
df.to_csv("HDFS_Parsed.csv", index=False)