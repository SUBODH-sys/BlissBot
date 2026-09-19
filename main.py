import json
with open('/workspaces/BlissBot/PyschoEducationalDataset.json', 'r') as f:
    json_data = json.load(f)
count = len(json_data) 
print(f"Total number of entries in the dataset: {count}")