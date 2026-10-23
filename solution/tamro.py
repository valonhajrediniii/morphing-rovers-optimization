
import numpy as np

from morphing_udp import (
    MASK_SIZE,
    MASK_CENTRES,
    NUM_MODE_PARAMETERS,
    NUM_BIASES,
    NUM_WEIGHTS,
)


def optimize(udp, initial_chromosome, iterations=6, seed=2026):
    rng = np.random.default_rng(seed)

    best = np.asarray(initial_chromosome, dtype=np.float64).copy()
    best_score = float(udp.fitness(best.tolist())[0])

    step_size = 0.10
    history = [best_score]

    # Pozicionet e parametrave të daljes së rrjetit neural
    output_bias_start = NUM_MODE_PARAMETERS + NUM_BIASES - 2
    output_weight_start = (
        NUM_MODE_PARAMETERS + NUM_BIASES + NUM_WEIGHTS - 80
    )

    print(f"TAMRO initial fitness: {best_score:.6f}")

    for iteration in range(1, iterations + 1):
        candidate = best.copy()

        if iteration % 2 == 1:
            # Optimizimi lokal i njërës prej katër formave
            mode = int(rng.integers(0, 4))
            row = int(rng.integers(0, MASK_SIZE - 2))
            col = int(rng.integers(0, MASK_SIZE - 2))

            start = mode * MASK_SIZE * MASK_SIZE

            rover_mask = candidate[
                start:start + MASK_SIZE**2
            ].reshape(MASK_SIZE, MASK_SIZE)

            patch = rng.normal(0, step_size, (3, 3))
            rover_mask[row:row + 3, col:col + 3] += patch

            operation = "TERRAIN MASK"

        else:
            # Optimizimi i kontrollit neural
            indices = np.arange(
                output_weight_start,
                output_weight_start + 80
            )

            selected = rng.choice(
                indices,
                size=8,
                replace=False
            )

            candidate[selected] += rng.normal(
                0, step_size, 8
            )

            candidate[
                output_bias_start:output_bias_start + 2
            ] += rng.normal(0, step_size, 2)

            operation = "NEURAL CONTROL"

        # Kufijtë e parametrave realë
        candidate = np.clip(candidate, -100.0, 100.0)

        # Elementet qendrore të maskave duhet të jenë pozitive
        for centre in MASK_CENTRES:
            candidate[centre] = max(candidate[centre], 1e-16)

        score = float(udp.fitness(candidate.tolist())[0])

        if np.isfinite(score) and score < best_score:
            best = candidate
            best_score = score
            step_size = min(step_size * 1.15, 2.0)
            status = "IMPROVED"
        else:
            step_size = max(step_size * 0.85, 0.001)
            status = "REJECTED"

        history.append(best_score)

        print(
            f"Iteration {iteration} | "
            f"{operation} | "
            f"Fitness: {score:.6f} | "
            f"Best: {best_score:.6f} | "
            f"{status}"
        )

    return best, best_score, history
