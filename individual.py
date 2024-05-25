from random import randint
from colour import Color
from PIL import Image
import numpy as np



class Individual:
    def __init__(self, l, w, representation = None, valid_set=None ,repetition=True):
        self.l=l
        self.w=w
        self.fitness=None
        self.image=None
        self.size = l*w*3
        if representation is None:
            self.representation = np.random.randint(0, 256, (l, w, 3), dtype=np.uint8)
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
        image = Image.fromarray(self.representation)

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
    def get_fitness_mean(self, target_repr):
        """
         Compute the fitness of the individual compare to the target image using the Mean Square Error (MSE).
               Args:
               - target_repr(numpy array): array with the RGB vales of the target image.
               Returns:
                   -fitness (float)
               """

        # Compute Mean Square Error (MSE)
        self.fitness = np.mean((self.representation - target_repr) ** 2)
        return self.fitness


if __name__ == '__main__':
    target_image = Image.open(r"IMG_0744.jpg")
    tar_l = target_image.height
    tar_w =target_image.width
    target_array= np.array(target_image)
    indiv = Individual(l=tar_l, w = tar_w, representation=None,valid_set= [0,255], repetition =True)
    print(f'The fitness is:  {indiv.get_fitness_mean(target_array)}')

