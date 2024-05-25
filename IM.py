import copy
from PIL import Image
import numpy as np
import random
from random import uniform, choice, randint, sample
from operator import attrgetter
from individual import Individual
import colour
# from Individual import Individual

# selection
class IM:

    def __init__(self, target_image, optim, size, **kwargs):
        self.target_image = Image.open(target_image).convert('RGB')
        self.optim= optim
        self.size= size
        self.w ,self.l = self.target_image.size

        #Create the population
        self.individuals = []

        for _ in range(size):
            self.individuals.append(
                Individual(
                    l=self.l,
                    w=self.w,
                    valid_set=kwargs["valid_set"],
                    repetition=kwargs["repetition"]
                )
            )
        # Compute the fitness
        for i in range(size):
            self.individuals[i].get_fitness_mean(self.target_image)



    # evolve function

    def evolve(self, gens, xo_prob, mut_prob, select, xo, mutate, elitism):
        for gen in range(gens):
            new_pop = []

            # Elitism: Retain the best individual(s)
            if elitism:
                if self.optim == "max":
                    elite = copy.copy(max(self.individuals, key=attrgetter('fitness')))
                elif self.optim == "min":
                    elite = copy.copy(min(self.individuals, key=attrgetter('fitness')))
                new_pop.append(elite)

            while len(new_pop) < self.size:
                # Selection
                parent1, parent2 = select(self.individuals), select(self.individuals)

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
                offspring1.get_fitness_mean(self.target_image)
                if len(new_pop) < self.size:
                    new_pop.append(offspring2)
                    offspring2.get_fitness_mean(self.target_image)

            # Replace the worst individual with the elite if elitism is used
            if elitism:
                if self.optim == "max":
                    worst = min(new_pop, key=attrgetter('fitness'))
                    if elite.fitness > worst.fitness:
                        new_pop.pop(new_pop.index(worst))
                        new_pop.append(elite)
                elif self.optim == "min":
                    worst = max(new_pop, key=attrgetter('fitness'))
                    if elite.fitness < worst.fitness:
                        new_pop.pop(new_pop.index(worst))
                        new_pop.append(elite)

            self.individuals = new_pop

            # Logging best individual of the generation
            if self.optim == "max":
                print(f"Best individual of gen #{gen + 1}: {max(self.individuals, key=attrgetter('fitness'))}")
            elif self.optim == "min":
                print(f"Best individual of gen #{gen + 1}: {min(self.individuals, key=attrgetter('fitness'))}")

    def selection_p (self):
        """Fitness proportionate selection implementation.

        Args:
            population (Population): The population we want to select from.

        Returns:
            Individual: selected individual.
        """
        if self.optim == "max":
            total_fitness = sum([i.fitness for i in self.size])
            r = uniform(0, total_fitness)
            position = 0
            for individual in self.size:
                position += individual.fitness
                if position > r:
                    return individual
        elif self.optim == "min":
            max_fitness = max([i.fitness for i in self.size])
            inverted_fitness = [max_fitness - i.fitness for i in self.size]
            total_fitness = sum(inverted_fitness)
            r = random.uniform(0, total_fitness)
            position = 0
            for individual, inv_fitness in zip(self.size, inverted_fitness):
                position += inv_fitness
                if position > r:
                    return individual
        else:
            raise Exception(f"Optimization not specified (max/min)")

    # this one
    def tournament_sel(self, tour_size= 6):
        tournament = [random.choice(self.individuals) for _ in range(6)]
        if self.optim == "max":
            return max(tournament, key=attrgetter('fitness'))
        elif self.optim == "min":
            return min(tournament, key=attrgetter('fitness'))
        else:
            raise Exception("Optimization not specified or incorrect (must be 'max' or 'min')")


    # crossover
    def single_point_xo(self, p1, p2):
        """Implementation of single point crossover.

        Args:
            parent1 (Individual): First parent for crossover.
            parent2 (Individual): Second parent for crossover.

        Returns:
            Individuals: Two offspring, resulting from the crossover.
        """
        xo_point = random.randint(1, len(p1.representation) - 1)
        offspring1 = np.concatenate((p1.representation[:xo_point], p2.representation[xo_point:]))
        offspring2 = np.concatenate((p2.representation[:xo_point], p1.representation[xo_point:]))
        return Individual(p1.l,p1.w, offspring1), Individual(p1.l,p1.w, offspring2)

    # chat xo
    def uniform_crossover(self, p1, p2):
        offspring1 = np.zeros_like(p1.representation)
        offspring2 = np.zeros_like(p2.representation)
        for i in range(len(p1.colors)):
            if random.random() < 0.5:
                offspring1[i] = p1.representation[i]
                offspring2[i] = p2.representation[i]
            else:
                offspring1[i] = p2.representation[i]
                offspring2[i] = p1.representation[i]
        return Individual(p1.l,p1.w, offspring1), Individual(p1.l,p1.w, offspring2)

    def blend_crossover(self,p1, p2, alpha=0.5):
        offspring1 = alpha * p1.representation + (1 - alpha) * p2.representation
        offspring2 = alpha * p2.representation + (1 - alpha) * p1.representation
        return Individual(len(p1.colors), offspring1), Individual(len(p2.colors), offspring2)



    # mutation
    #ASK ABOUT SELF
    def inversion_mutation(self, individual):
        """Inversion mutation for a GA individual. Reverts a portion of the representation.

        Args:
            individual (Individual): A GA individual from charles.py

        Returns:
            Individual: Mutated Individual
        """
        # Flatten the 3D array into a 1D array for easy indexing
        flat_image = individual.representation.flatten()
        # Select two random indices
        mut_indexes = random.sample(range(len(flat_image)), 2)
        mut_indexes.sort()

        # Invert the pixel values in the selected range
        flat_image[mut_indexes[0]:mut_indexes[1]] = flat_image[mut_indexes[0]:mut_indexes[1]][::-1]

        # Reshape back to the original 3D shape
        individual.representation = flat_image.reshape(individual.l, individual.w, 3)

        # Update the image
        individual.get_image()

        return individual

    def swap_mutation(self,individual):
        """Swap mutation for a GA individual. Swaps the bits.

        Args:
            individual (Individual): A GA individual from charles.py

        Returns:
            Individual: Mutated Individual
        """
        # Flatten the 3D array into a 1D array for easy indexing
        flat_image = individual.representation.flatten()
        # Select two random indices
        mut_indexes = random.sample(range(len(flat_image)), 2)

        # Invert the pixel values in the selected range
        flat_image[mut_indexes[0]], flat_image[mut_indexes[1]] = flat_image[mut_indexes[1]], flat_image[mut_indexes[0]]

        # Reshape back to the original 3D shape
        individual.representation = flat_image.reshape(individual.l, individual.w, 3)

        # Update the image
        individual.get_image()
        return individual

    # Gaussian mutation or geometric mutation





    def visualize_population(self):
        """
        Visualize the images of the population.

        Args:
            population (list): List of Individual objects.
        """
        num_individuals = len(self.individuals)
        cols = 5  # Number of columns for subplots
        rows = (num_individuals // cols) + 1  # Number of rows for subplots
        fig, axs = plt.subplots(rows, cols, figsize=(15, 15))

        for i, individual in enumerate(self.individuals):
            ax = axs[i // cols, i % cols] if num_individuals > 1 else axs
            ax.imshow(individual.get_image())
            ax.set_title(f"Fitness: {individual.fitness:.2f}")
            ax.axis('off')

        # Hide empty subplots
        for i in range(num_individuals, rows * cols):
            ax = axs[i // cols, i % cols] if num_individuals > 1 else axs
            ax.axis('off')

        plt.tight_layout()
        plt.show()







p = IM('IMG_0744.jpg', size= 50, optim='min',
       valid_set=[0,256], repetition = True )

p.evolve(gens=3, xo_prob=0.9, mut_prob=0.15,
         select=p.tournament_sel, xo=p.single_point_xo, mutate=p.inversion_mutation, elitism=True)

p.visualize_population()
