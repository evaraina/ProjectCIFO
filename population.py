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
        constant_fitness_generations = 0
        previous_best_fitness = None
        current_best_fitness=None
        best=[]

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
                parent1= select(self.individuals)
                parent2 = select(self.individuals)

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

            # Logging the best individual of the generation
            if self.optim == "max":
                current_best_fitness = max(self.individuals, key=attrgetter('fitness')).fitness
                print(f"Best individual of gen #{gen + 1}: {max(self.individuals, key=attrgetter('fitness'))}")
            elif self.optim == "min":
                current_best_fitness = min(self.individuals, key=attrgetter('fitness')).fitness
                print(f"Best individual of gen #{gen + 1}: {min(self.individuals, key=attrgetter('fitness'))}")

            # Check if the fitness has remained constant
            if previous_best_fitness is not None and current_best_fitness == previous_best_fitness:
                constant_fitness_generations += 1
            else:
                constant_fitness_generations = 0

            if constant_fitness_generations > 20:
                print("Fitness has remained constant for more than 20 generations. Stopping evolution.")
                break

            previous_best_fitness = current_best_fitness
            best.append(current_best_fitness)

        return best

    # Selection

    def selection_p(self, population):
        population= self.individuals
        """
        Fitness proportionate selection implementation.
        """
        if self.optim == "max":
            total_fitness = sum([i.fitness for i in population])
            r = uniform(0, total_fitness)
            position = 0
            for individual in population:
                position += individual.fitness
                if position > r:
                    return individual
        elif self.optim == "min":
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

    def tournament_sel(self, population, tour_size=6):
        tournament = [random.choice(population) for _ in range(tour_size)]
        if self.optim == "max":
            return max(tournament, key=attrgetter('fitness'))
        elif self.optim == "min":
            return min(tournament, key=attrgetter('fitness'))
        else:
            raise Exception("Optimization not specified or incorrect (must be 'max' or 'min')")


    # Crossover
    def single_point_xo(self, p1, p2):

        xo_point = random.randint(1, len(p1.representation) - 1)
        offspring1_repr = np.concatenate((p1.representation[:xo_point], p2.representation[xo_point:]))
        offspring2_repr = np.concatenate((p2.representation[:xo_point], p1.representation[xo_point:]))

        return  Individual(p1.l,p1.w, offspring1_repr), Individual(p1.l,p1.w, offspring2_repr)


    def uniform_crossover(self, p1, p2):
        """
            Each pixel is assign as equal to parent1 or parent2 two with a probability 0.5
        """
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


    def row_wise_crossover(self,p1, p2):
        """
        The row_wise_crossover randomly choose if each child row belongs to parent 1 or parent2.
        """
        # Get the dimensions of the image
        rows = p1.l

        offspring1_repr = p1.representation
        offspring2_repr = p2.representation

        # Randomly decide which rows to swap
        for row in range(rows):
            if np.random.rand() < 0.5:
                # Swap the row between the two children
                offspring1_repr[row, :], offspring2_repr[row, :] = p2[row, :], p1[row, :]
        return Individual(p1.l,p1.w, offspring1_repr),  Individual(p1.l,p1.w, offspring2_repr)


    # Mutation
    def inversion_mutation(self, individual):
        """
            Inversion mutation for a GA individual. Reverts a portion of the representation.
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
        """
            Swap mutation for a GA individual. Swaps the bits.
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
        """
        fittest_individual = max(self.individuals, key=lambda x: x.fitness)
        fig, ax = plt.subplots(1, 1)
        ax.imshow(fittest_individual.get_image())
        ax.set_title(f"Fittest Individual (Fitness: {fittest_individual.fitness:.2f})")
        ax.axis('off')
        plt.show()


gen=20000

save_run = []

for i in range(5):

    p = Population('IMG_0744.jpg', size=50, optim='min',
                   valid_set=[0, 256], repetition=True)

    result = p.evolve(gens=gen, xo_prob=0.9, mut_prob=0.15,
                      select=p.tournament_sel, xo=p.uniform_crossover,
                      mutate=p.swap_mutation, elitism=True)

    save_run.append(result)

# Compute the mean of the different runs
sum_elements1 = [0] * gen
for lst in save_run:
    for i in range(gen):
        sum_elements1[i] += lst[i]
mean_elements1 = [sum_elem / len(save_run) for sum_elem in sum_elements1]



# Plot the mean elements of the two example
plt.figure(figsize=(10, 6))
plt.plot(mean_elements1, marker='o', linestyle='-', color='b', label='Uniform crossover')
plt.legend()
plt.title('Mean Fitness Plot')
plt.xlabel('Generation')
plt.ylabel('Mean Fitness')
plt.grid(True)
plt.show()

# Visualize the best image recreation
p.visualize_population()
