
import numpy as np

from morphing_udp import Rover


class RoverEvaluator:
    def __init__(self, udp):
        self.udp = udp

        # Nga një skenar për secilën prej 6 hartave.
        self.quick_scenarios = [
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 0),
            (5, 0),
        ]

    def quick_fitness(self, chromosome):
        rover = Rover(chromosome)
        total = 0.0

        for map_id, scenario_id in self.quick_scenarios:
            result = self.udp.run_single_scenario(
                rover,
                map_id,
                scenario_id
            )

            total += (1 + result[0]) * result[1]

        return float(total / len(self.quick_scenarios))

    def full_fitness(self, chromosome):
        return float(self.udp.fitness(chromosome)[0])
