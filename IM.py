import numpy as np
import random
from random import uniform, choice, randint, sample
from operator import attrgetter
# from Individual import Individual

# selection
class IM:
    # prob not this one
    def __init__(self, l, w, target_image):
        self.l = l
        self.w = w
        self.target_image = target_image


    def selection_p (self, population):
        """Fitness proportionate selection implementation.

        Args:
            population (Population): The population we want to select from.

        Returns:
            Individual: selected individual.
        """
        if population.optim == "max":
            total_fitness = sum([i.fitness for i in population])
            r = uniform(0, total_fitness)
            position = 0
            for individual in population:
                position += individual.fitness
                if position > r:
                    return individual
        elif population.optim == "min":
            max_fitness = max([i.fitness for i in population])
            inverted_fitness = [max_fitness - i.fitness for i in population]
            total_fitness = sum(inverted_fitness)
            r = random.uniform(0, total_fitness)
            position = 0
            for individual, inv_fitness in zip(population, inverted_fitness):
                position += inv_fitness
                if position > r:
                    return individual
        else:
            raise Exception(f"Optimization not specified (max/min)")

    # this one
    def tournament_sel(population, tour_size=6):
        tournament = [random.choice(population) for _ in range(tour_size)]
        if population.optim == "max":
            return max(tournament, key=attrgetter('fitness'))
        elif population.optim == "min":
            return min(tournament, key=attrgetter('fitness'))
        else:
            raise Exception("Optimization not specified or incorrect (must be 'max' or 'min')")


    # crossover
    def single_point_xo(p1, p2):
        """Implementation of single point crossover.

        Args:
            parent1 (Individual): First parent for crossover.
            parent2 (Individual): Second parent for crossover.

        Returns:
            Individuals: Two offspring, resulting from the crossover.
        """
        xo_point = random.randint(1, len(p1.colors) - 1)
        offspring1 = np.concatenate((p1.colors[:xo_point], p2.colors[xo_point:]))
        offspring2 = np.concatenate((p2.colors[:xo_point], p1.colors[xo_point:]))
        return individual(len(p1.colors), offspring1), individual(len(p2.colors), offspring2)

    # chat xo
    def uniform_crossover(p1, p2):
        child1_colors = np.zeros_like(p1.colors)
        child2_colors = np.zeros_like(p2.colors)
        for i in range(len(p1.colors)):
            if random.random() < 0.5:
                child1_colors[i] = p1.colors[i]
                child2_colors[i] = p2.colors[i]
            else:
                child1_colors[i] = p2.colors[i]
                child2_colors[i] = p1.colors[i]
        return individual(len(p1.colors), child1_colors), individual(len(p2.colors), child2_colors)

    def blend_crossover(p1, p2, alpha=0.5):
        child1_colors = alpha * p1.colors + (1 - alpha) * p2.colors
        child2_colors = alpha * p2.colors + (1 - alpha) * p1.colors
        return Individual(len(p1.colors), child1_colors), Individual(len(p2.colors), child2_colors)



    # mutation

    def inversion_mutation(individual):
        """Inversion mutation for a GA individual. Reverts a portion of the representation.

        Args:
            individual (Individual): A GA individual from charles.py

        Returns:
            Individual: Mutated Individual
        """
        # Flatten the 3D array into a 1D array for easy indexing
        flat_image = individual.array.flatten()
        # Select two random indices
        mut_indexes = random.sample(range(len(flat_image)), 2)
        mut_indexes.sort()

        # Invert the pixel values in the selected range
        flat_image[mut_indexes[0]:mut_indexes[1]] = flat_image[mut_indexes[0]:mut_indexes[1]][::-1]

        # Reshape back to the original 3D shape
        individual.array = flat_image.reshape(individual.l, individual.w, 3)

        # Update the image
        individual.to_image()

        return individual

    def swap_mutation(individual):
        """Swap mutation for a GA individual. Swaps the bits.

        Args:
            individual (Individual): A GA individual from charles.py

        Returns:
            Individual: Mutated Individual
        """
        # Flatten the 3D array into a 1D array for easy indexing
        flat_image = individual.array.flatten()
        # Select two random indices
        mut_indexes = random.sample(range(len(flat_image)), 2)

        # Invert the pixel values in the selected range
        flat_image[mut_indexes[0]], flat_image[mut_indexes[1]] = flat_image[mut_indexes[1]], flat_image[mut_indexes[0]]

        # Reshape back to the original 3D shape
        individual.array = flat_image.reshape(individual.l, individual.w, 3)

        # Update the image
        individual.to_image()
        return individual

    # Gaussian mutation or geometric mutation





    # evolve function

    def evolve(self, population, gens, xo_prob, mut_prob, select, xo, mutate, elitism):
        for gen in range(gens):
            new_pop = []

            # Elitism: Retain the best individual(s)
            if elitism:
                if population.optim == "max":
                    elite = copy.copy(max(population.individuals, key=attrgetter('fitness')))
                elif population.optim == "min":
                    elite = copy.copy(min(population.individuals, key=attrgetter('fitness')))
                new_pop.append(elite)

            while len(new_pop) < population.size:
                # Selection
                parent1, parent2 = select(population.individuals), select(population.individuals)

                # Crossover
                if random.random() < xo_prob:
                    offspring1, offspring2 = xo(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1, parent2

                # Mutation
                if random.random() < mut_prob:
                    offspring1 = mutate(offspring1)
                if random.random() < mut_prob:
                    offspring2 = mutate(offspring2)

                new_pop.append(offspring1)
                if len(new_pop) < population.size:
                    new_pop.append(offspring2)

            # Replace the worst individual with the elite if elitism is used
            if elitism:
                if population.optim == "max":
                    worst = min(new_pop, key=attrgetter('fitness'))
                    if elite.fitness > worst.fitness:
                        new_pop.pop(new_pop.index(worst))
                        new_pop.append(elite)
                elif population.optim == "min":
                    worst = max(new_pop, key=attrgetter('fitness'))
                    if elite.fitness < worst.fitness:
                        new_pop.pop(new_pop.index(worst))
                        new_pop.append(elite)

            population.individuals = new_pop

            # Logging best individual of the generation
            if population.optim == "max":
                print(f"Best individual of gen #{gen + 1}: {max(population.individuals, key=attrgetter('fitness'))}")
            elif population.optim == "min":
                print(f"Best individual of gen #{gen + 1}: {min(population.individuals, key=attrgetter('fitness'))}")


