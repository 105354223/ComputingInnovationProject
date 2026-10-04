import re

HDFS_Pattern = r'(\d{6}) (\d{6}) (\d+) (\S+) (\S+) (.*) (blk_-?\d+)'

def parse_HDFS(lines):
    m = re.match(HDFS_Pattern, lines)
    if not m:
        return None
    date, time, pid, level, component, message, blockId = m.groups()
    return {
        "date": date,
        "time": time,
        "pid": pid,
        "level": level,
        "component": component,
        "message": message,
        "blockId": blockId,
    }

events = []
skipped = 0

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

print(f"parsed {len(events)} lines, skipped {skipped}")
print(events[0])
