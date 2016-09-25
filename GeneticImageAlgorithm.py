import random
from PIL import Image
from deap import base, creator, tools, algorithms
import numpy as np


class GeneticImageAlgorithm:
    def __init__(self, image_name):
        self.target_image = Image.open(image_name)
        self.pix = self.target_image.load()

        self.current_pixel = [0, 0]

        # Instantiate creator variables
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # Create toolbox
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_rgb", self.__create_value)
        self.toolbox.register("individual", tools.initRepeat,
                              creator.Individual, self.toolbox.attr_rgb, n=3)

        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Register toolbox functions
        self.toolbox.register("evaluate", self.__get_fitness)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self.__mutate, indpb=1 / 3)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def run(self):
        width, height = self.target_image.size
        result_pixels = []

        pixel_count = 1
        for y in range(height):
            for x in range(width):
                print("Evolving pixel %s of %s" % (pixel_count, width * height))
                pixel_count += 1

                self.current_pixel = [x, y]

                pop = self.toolbox.population(n=10)
                hof = tools.HallOfFame(maxsize=1)
                stats = tools.Statistics(lambda ind: ind.fitness.values)
                stats.register("min", np.min)

                algorithms.eaSimple(population=pop, toolbox=self.toolbox,
                                    cxpb=0.5, mutpb=0.2, ngen=50, stats=stats,
                                    halloffame=hof, verbose=False)

                result_pixels.append(hof[0])

        return result_pixels

    # Create value
    def __create_value(self):
        return random.randint(0, 255)

    # Score individual fitness
    def __get_fitness(self, individual):
        current_x, current_y = self.current_pixel

        tr, tg, tb = self.pix[current_x, current_y]
        ir, ig, ib = tuple(individual)

        fitness = abs(tr - ir) + abs(tg - ig) + abs(tb - ib)
        return fitness,

    # Mutate genes
    def __mutate(self, individual, indpb):
        for i in range(len(individual)):
            individual[i] = self.__create_value() if random.random() <= indpb else individual[i]
        return individual,


###################################################################

# Run main application
gia = GeneticImageAlgorithm(image_name="mona_lisa_xs.png")
output_pixels = gia.run()

final_pixels = []
print("Converting to image format")
for pix in output_pixels:
    final_pixels.append(tuple(pix))

result = Image.new(mode=gia.target_image.mode, size=gia.target_image.size)
result.putdata(data=final_pixels)
result.save("result_test.png")
result.show()

###################################################################
