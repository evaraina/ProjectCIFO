import random
from colour import Color
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


class Individual:
    def __init__(self, l, w, representation = None, valid_set=None ,repetition=True):
        self.l=l
        self.w=w
        self.fitness=float('inf')
        self.image=None
        self.size = l*w*3
        if representation is None:
            self.representation = np.random.randint(0,256, (l,w, 3),dtype=np.uint8)
        # if we pass an argument like Individual(my_path)
        else:
            self.representation = representation

    def index(self, value):
        return self.representation.index(value)

    def __len__(self):
        return len(self.representation)

    def __getitem__(self, position):
         return self.representation[position]

    def __setitem__(self, position, value):
        self.representation[position] = value

    def __repr__(self):
         return f" Fitness: {self.fitness}"

    # Create function to link the representation array and the corrispondent image and viceversa

    def get_image(self):
        """
            Create the image starting from the array of RGB.
            Returns:
            - image (PIL.Image.Image): created image.
            """
        image = Image.fromarray(np.uint8(self.representation))
        return image

    def get_representation(self, image):
        """
        From a given image compute the corrispondent representation array with the RGB values
            Args:
                - image (PIL.Image)
            Return:
                 -representation array
            """
        representation= np.array(image)
        return representation


    # Create function to compute the fitness
    def get_fitness_delta(self, target_repr):
        """
         Compute the fitness of the individual compare to the target image using the ΔE CIE 1976
            Args:
                - target_array (numpy array): array with the RGB vales of the target image
            Returns:
             - fitness (float)
                """
        # Compute the difference ΔE_CIE1976 for each pixel
        delta_e = Color.delta_e.delta_E_CIE1976(target_repr, self.representation)

        # Compute the mean of ΔE and set it as the fitness
        self.fitness = np.mean(delta_e)
        return self.fitness

    def hvs_fitness(self, target_repr):
        """
        Calcola la fitness utilizzando un modello del sistema visivo umano (HVS).

        Args:
        - image1 (numpy.ndarray): Prima immagine (valori RGB).
        - image2 (numpy.ndarray): Seconda immagine (valori RGB).

        Returns:
        - fitness (float): Valutazione della similarità tra le immagini basata sul modello HVS.
        """
        # Conversione delle immagini in scala di grigi
        gray1 = np.dot(self.representation[..., :3], [0.2989, 0.5870, 0.1140])
        gray2 = np.dot(target_repr[..., :3], [0.2989, 0.5870, 0.1140])

        # Calcolo della differenza assoluta
        diff = np.abs(gray1 - gray2)

        # Applicazione di un filtro di attenuazione del contrasto
        attenuation_filter = np.exp(-diff / 100.0)

        # Calculate the weighted mean to get the fitness
        new_fitness = np.mean(attenuation_filter)

        # Check if fitness remains constant
        if new_fitness == self.fitness:
            self.constant_fitness_generations += 1
            if self.constant_fitness_generations > 10:
                raise Exception("Fitness has remained constant for too many generations. Interrupting...")
        else:
            self.constant_fitness_generations = 0

        self.fitness = new_fitness

        return self.fitness

        def get_fitness_mean(self, target_repr):
        """
         Compute the fitness of the individual compare to the target image using the Mean Square Error (MSE).
               Args:
               - target_repr(numpy array): array with the RGB vales of the target image.
               Returns:
                   -fitness (float)
               """
        # Compute Mean Square Error (MSE)
        new_fitness = np.mean((self.representation - target_repr) ** 2)

        # Check if fitness remains constant
        if new_fitness == self.fitness:
            self.constant_fitness_generations += 1
            if self.constant_fitness_generations > 10:
                raise Exception("Fitness has remained constant for too many generations. Interrupting...")

        # If fitness changes, reset the counter
        else:
            self.constant_fitness_generations = 0

        self.fitness = new_fitness
        return self.fitness


if __name__ == '__main__':
    target_image = Image.open(r"IMG_0744.jpg").resize((756,1008))
    tar_l = target_image.height
    tar_w =target_image.width
    target_array= np.array(target_image)
    indiv = Individual(l=tar_l, w = tar_w, representation=None,valid_set= [0,255], repetition =True)
    print(f'The fitness is:  {indiv.hvs_fitness(target_array)}')
    fig, ax = plt.subplots(1, 1)
    ax.imshow(indiv.get_image())
    ax.axis('off')
    plt.show()

