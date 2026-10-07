import re
import pandas as pd

# Regex pattern for Android Logs
android_pattern = (
    r'^(\d{2}-\d{2})\s+'
    r'(\d{2}:\d{2}:\d{2}\.\d{3})\s+'
    r'(\d+)\s+'
    r'(\d+)\s+'
    r'([VDIWEF])\s+'
    r'([^:]+):\s*'
    r'(.*)$'
)

# Matches regex pattern to groups to be distinguished
def parse_android(line):
    m = re.match(android_pattern, line)
    if not m:
        return None
    date, time, pid, tid, level, tag, message = m.groups()
    return {
        "date": date,
        "time": time,
        "pid": pid,
        "tid": tid,
        "level": level,
        "tag": tag.strip(),
        "message": message.strip(),
    }

events = []
skipped = 0
skipped_samples = []

# Loop for event and skipped counters
with open("basicDatasets/Android_v1/Android.log", "r",
          encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.rstrip("\r\n")
        if not line:
            continue
        parsed = parse_android(line)
        if parsed is None:
            skipped += 1
            if len(skipped_samples) < 5:
                skipped_samples.append(line)
            continue
        events.append(parsed)

# Print skipped and events passed
# print(f"parsed {len(events)} lines, skipped {skipped}")
# if events:
#     print("first event:", events[0])
# if skipped_samples:
#     print("sample skipped:")
#     for s in skipped_samples:
#         print(" ", repr(s))

# Log to csv converter
df = pd.DataFrame(events)
df.to_csv("Android_Parsed.csv", index=False)
# print(f"wrote Android_Parsed.csv with {len(df)} rows")