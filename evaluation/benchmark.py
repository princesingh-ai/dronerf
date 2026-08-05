from pathlib import Path
import json

from predict import predict

dataset = Path("external_val/BOTH")
files = sorted(dataset.rglob("*.dat"))
print(f"Found {len(files)} recordings")

results = []

for index, file in enumerate(files, start=1):
    print(f"\n[{index}/{len(files)}] {file.name}")

    result = predict(str(file))

    drone, flight_mode = file.parent.name.split("_")

    result["file"] = file.name
    result["drone"] = drone
    result["flight_mode"] = flight_mode
    result["interference"] = file.parents[1].name

    results.append(result)

correct = sum(
    r["prediction"] == "Drone"
    for r in results
)

summary = {
    "total_files": len(results),
    "correct": correct,
    "accuracy": correct / len(results),
}

output = {
    "summary": summary,
    "results": results,
}

Path("docs/external_val_results").mkdir(
    parents=True,
    exist_ok=True,
)

with open("docs/external_val_results/both.json", "w") as f:
    json.dump(output, f, indent=4)