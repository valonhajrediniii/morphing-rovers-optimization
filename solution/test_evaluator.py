
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from morphing_udp import morphing_rover_UDP
from evaluator import RoverEvaluator


def main():
    udp = morphing_rover_UDP()
    evaluator = RoverEvaluator(udp)
    chromosome = udp.example()

    start = time.perf_counter()
    quick = evaluator.quick_fitness(chromosome)
    quick_time = time.perf_counter() - start

    start = time.perf_counter()
    full = evaluator.full_fitness(chromosome)
    full_time = time.perf_counter() - start

    print("=== TAMRO v2 Evaluation Benchmark ===")
    print(f"Quick fitness: {quick:.6f}")
    print(f"Quick time: {quick_time:.2f}s")
    print(f"Full fitness: {full:.6f}")
    print(f"Full time: {full_time:.2f}s")


if __name__ == "__main__":
    main()
