import json
import time

print("Dummy training started")
time.sleep(20)

with open("adapter_output.json", "w") as f:
    json.dump({"final_loss": 0.42, "note": "dummy run, no real training"}, f)

print("Dummy training complete")