import json
with open("players.json") as f:
    players = json.load(f)
no_id = [p for p in players if not p.get("mlbID")]
print(f"Missing mlbID: {len(no_id)}")
for p in no_id:
    print(p["id"], p["name"])