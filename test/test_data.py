import json

with open('/home/next/cr_repo/bf/test/data_sl.json', 'r') as f:
    for line in f:
        line = json.loads(line)
        tag = line['tags']
