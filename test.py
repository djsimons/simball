import json

with open("players.json") as f:
    players = json.load(f)

no_war = [p for p in players if p["WAR"] is None]
print(f"Missing WAR: {len(no_war)}")
for p in no_war:
    print(p["id"], p["name"])