import sys
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from morphing_udp import morphing_rover_UDP, Rover

RESULTS = ROOT / "solution" / "results"


def scenario_scores(udp, chromosome):
    rover = Rover(chromosome)
    scores = []

    for map_id in range(6):
        for scenario_id in range(5):
            result = udp.run_single_scenario(
                rover, map_id, scenario_id
            )
            score = float((1 + result[0]) * result[1])

            scores.append({
                "map": map_id + 1,
                "scenario": scenario_id + 1,
                "fitness": score
            })

    return scores


def main():
    udp = morphing_rover_UDP()

    example = udp.example()
    optimized = np.load(RESULTS / "tamro_best.npy")

    print("Evaluating ESA example...", flush=True)
    before = scenario_scores(udp, example)

    print("Evaluating TAMRO v1...", flush=True)
    after = scenario_scores(udp, optimized)

    comparisons = []

    for old, new in zip(before, after):
        comparisons.append({
            "map": old["map"],
            "scenario": old["scenario"],
            "esa_fitness": old["fitness"],
            "tamro_fitness": new["fitness"],
            "improvement": old["fitness"] - new["fitness"]
        })

    output = RESULTS / "scenario_comparison.csv"

    with open(output, "w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=comparisons[0].keys()
        )
        writer.writeheader()
        writer.writerows(comparisons)

    print("\n=== FITNESS COMPARISON ===")
    print(f"{'Map':<6}{'Scenario':<12}{'ESA':<12}{'TAMRO':<12}{'Change':<12}")

    for row in comparisons:
        print(
            f"{row['map']:<6}"
            f"{row['scenario']:<12}"
            f"{row['esa_fitness']:<12.4f}"
            f"{row['tamro_fitness']:<12.4f}"
            f"{row['improvement']:+.4f}"
        )

    print("\n=== SUMMARY ===")

    improved = sum(r["improvement"] > 0 for r in comparisons)
    worsened = sum(r["improvement"] < 0 for r in comparisons)

    print(f"Improved scenarios: {improved}/30")
    print(f"Worsened scenarios: {worsened}/30")
    print(f"ESA average: {np.mean([r['esa_fitness'] for r in comparisons]):.6f}")
    print(f"TAMRO average: {np.mean([r['tamro_fitness'] for r in comparisons]):.6f}")

    print("\n=== 5 HARDEST TAMRO SCENARIOS ===")
    hardest = sorted(
        comparisons,
        key=lambda r: r["tamro_fitness"],
        reverse=True
    )

    for row in hardest[:5]:
        print(
            f"Map {row['map']}, Scenario {row['scenario']}: "
            f"{row['tamro_fitness']:.6f}"
        )

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
