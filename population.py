import copy
from PIL import Image
import numpy as np
import random
from random import uniform, choice, randint, sample
from operator import attrgetter

from matplotlib import pyplot as plt

from individual import Individual
import colour
# from Individual import Individual

# selection
class Population:

    def __init__(self, target_image, optim, size, **kwargs):
        original = Image.open(target_image).convert('RGB')

        #IMG_0744
        self.target_image= original.resize((151,202))

        #Circle
        #self.target_image = original.resize((125,137))

        #Geometric
        #self.target_image = original.resize((180, 90))

        self.optim= optim
        self.size= size
        self.target_array= np.array(self.target_image)
        self.w ,self.l = self.target_image.size

        #Create the population
        self.individuals = []

        for _ in range(self.size):
            self.individuals.append(
                Individual(
                    l=self.l,
                    w=self.w,
                    valid_set=kwargs["valid_set"],
                    repetition=kwargs["repetition"]
                )
            )
        # Compute the fitness
        for i in range(self.size):
            self.individuals[i].hvs_fitness(self.target_array)



    # evolve function

    def evolve(self, gens, xo_prob, mut_prob, select, xo, mutate, elitism):
        for gen in range(gens):
            new_pop = []
            fittest = None

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
                offspring1.hvs_fitness(self.target_array)

                if len(new_pop) < self.size:
                    new_pop.append(offspring2)
                    offspring2.hvs_fitness(self.target_array)


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
        offspring1_repr = np.concatenate((p1.representation[:xo_point], p2.representation[xo_point:]))
        offspring2_repr = np.concatenate((p2.representation[:xo_point], p1.representation[xo_point:]))

        return  Individual(p1.l,p1.w, offspring1_repr), Individual(p1.l,p1.w, offspring2_repr)

    # chat xo
    def uniform_crossover(self, p1, p2):
        offspring1_repr = np.zeros_like(p1.representation)
        offspring2_repr = np.zeros_like(p2.representation)
        for i in range(len(p1.representation)):
            if random.random() < 0.5:
                offspring1_repr[i] = p1.representation[i]
                offspring2_repr[i] = p2.representation[i]
            else:
                offspring1_repr[i] = p2.representation[i]
                offspring2_repr[i] = p1.representation[i]

        return Individual(p1.l,p1.w, offspring1_repr),  Individual(p1.l,p1.w, offspring2_repr)

    def blend_crossover(self,p1, p2, alpha=0.5):
        offspring1_repr = alpha * p1.representation + (1 - alpha) * p2.representation
        offspring2_repr= alpha * p2.representation + (1 - alpha) * p1.representation
        return Individual(p1.l,p1.w, offspring1_repr),Individual(p1.l,p1.w, offspring2_repr)


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

        return individual


    def visualize_population(self):
        """
        Visualize the image of the fittest individual at the last generation.

        Args:
            population (list): List of Individual objects.
        """
        fittest_individual = max(self.individuals, key=lambda x: x.fitness)
        fig, ax = plt.subplots(1, 1)
        ax.imshow(fittest_individual.get_image())
        ax.set_title(f"Fittest Individual (Fitness: {fittest_individual.fitness:.2f})")
        ax.axis('off')
        plt.show()

p = Population('IMG_0744.jpg', size= 100, optim='min',
       valid_set=[0,256], repetition = True )

p.evolve(gens=20000, xo_prob=0.9, mut_prob=0.15,
         select=p.tournament_sel, xo=p.uniform_crossover, mutate=p.swap_mutation, elitism=True)

p.visualize_population()
