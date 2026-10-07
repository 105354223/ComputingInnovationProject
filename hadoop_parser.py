import re
import pandas as pd

# Regex pattern for HDFS logs
HDFS_pattern = r'(\d{6}) (\d{6}) (\d+) (\S+) (\S+) (.*)'

# Sub regex patterns for blk and IP extraction
blk_pattern = r'(blk_-?\d+)'
srcIP_pattern = r'src:\s*(\S+)'
dstIP_pattern = r'dest:\s*(\S+)'

# Matching groups from regex
def parse_HDFS(lines):
    m = re.match(HDFS_pattern, lines)
    if not m:
        return None
    date, time, pid, level, component, message = m.groups()

    def sub_pattern(pattern, text):
        n = re.search(pattern, text)
        return n.group(1) if n else None

    return {
        "date": date,
        "time": time,
        "pid": pid,
        "level": level,
        "component": component,
        "message": message,
        "blockId": sub_pattern(blk_pattern, message),
        "sourceIP": sub_pattern(srcIP_pattern, message),
        "destinationIP": sub_pattern(dstIP_pattern, message),
    }

# Event and skipped line counter
events = []
skipped = 0

# Loop for event and skipped counter
with open("basicDatasets/Log Anomaly Detection/HDFS.log", "r") as file:
    for lines in file:
        lines = lines.rstrip("\r\n")
        if not lines:
            continue
        parsed = parse_HDFS(lines)
        if parsed is None:
            skipped += 1
            continue
        events.append(parsed)

# Print skipped and events passed
print(f"parsed {len(events)} lines, skipped {skipped}")
print(events[0])

# Log to csv converter
df = pd.DataFrame(events)
df.to_csv("HDFS_Parsed.csv", index = False)