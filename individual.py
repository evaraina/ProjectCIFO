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

    def hvs_fitness(self, target_repr):
        """
        Compute the fitness using human visual system (HVS) and a scale of gray.
        """
        # Conversion on a gray scale
        gray1 = np.dot(self.representation[..., :3], [0.2989, 0.5870, 0.1140])
        gray2 = np.dot(target_repr[..., :3], [0.2989, 0.5870, 0.1140])

        diff = np.abs(gray1 - gray2)

        self.fitness  = np.sum(diff)

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

