import numpy as np


def optimize(udp, initial_chromosome, generations=5, mutation_rate=0.01):
    rng = np.random.default_rng(42)

    best = np.array(initial_chromosome, dtype=float).copy()
    best_fitness = udp.fitness(best.tolist())[0]

    print(f"Initial fitness: {best_fitness:.6f}")

    for generation in range(1, generations + 1):
        candidate = best.copy()

        # Ndryshojmë vetëm disa parametra
        mask = rng.random(len(candidate)) < mutation_rate
        candidate[mask] += rng.normal(0, 0.1, mask.sum())

        # Kufizimet për parametrat realë
        candidate = np.clip(candidate, -100, 100)

        # 7 parametrat e fundit janë hiperparametra diskretë
        candidate[-7:-5] = np.clip(
            np.rint(candidate[-7:-5]), 0, 1
        )
        candidate[-5:-1] = np.clip(
            np.rint(candidate[-5:-1]), 0, 6
        )
        candidate[-1] = np.clip(
            np.rint(candidate[-1]), 0, 4
        )

        # Qendrat e katër maskave duhet të jenë pozitive
        candidate[60] = max(candidate[60], 1e-16)
        candidate[181] = max(candidate[181], 1e-16)
        candidate[302] = max(candidate[302], 1e-16)
        candidate[423] = max(candidate[423], 1e-16)

        fitness = udp.fitness(candidate.tolist())[0]

        if fitness < best_fitness:
            best = candidate
            best_fitness = fitness

        print(
            f"Generation {generation}: "
            f"Fitness = {fitness:.6f}, "
            f"Best = {best_fitness:.6f}"
        )

    return best, best_fitness