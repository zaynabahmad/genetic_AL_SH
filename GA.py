from random import choices, randint, randrange, random, shuffle
from typing import List, Optional, Callable, Tuple

Genome = List[int]
Population = List[Genome]
PopulateFunc = Callable[[], Population]
FitnessFunc = Callable[[Genome], int]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]
PrinterFunc = Callable[[Population, int, FitnessFunc], None]


#points coordinates 
points_coordinates = [
    (0, 0),
    (105, 150),
    (201, 100),
    (515, 412),
    (200, 310),
    (603, 702),
    (802, 901),
    (115, 110),
    (120, 135),
    (149, 159),
    (560, 170),
    (180, 190)
]

#help in printing
city_names = ["City1", "City2", "City3", "City4","City5","City6","City7","City8","City9","City10","City11","City12"]

#function to determine the whole distance of takes specific path 

#this is to calculate the distance between two points 
def calculate_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

#and this to calculate the total distance of the whole path 
def total_distance(genome: Genome) -> int:
    distance = 0
    for i in range(len(genome) - 1):
        distance += calculate_distance(points_coordinates[genome[i]], points_coordinates[genome[i + 1]])
    # Add distance from the last point back to the starting point
    distance += calculate_distance(points_coordinates[genome[-1]], points_coordinates[genome[0]])   # to be modifying 
    return distance


def generate_genome(length: int) -> Genome:
    genome = list(range(length))
    shuffle(genome)
    return genome


def generate_population(size: int, genome_length: int) -> Population:
    return [generate_genome(genome_length) for _ in range(size)]


def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    if len(a) != len(b):
        raise ValueError("Genomes a and b must be of same length")

    length = len(a)
    if length < 2:
        return a, b

    p = randint(1, length - 1)
    return a[0:p] + b[p:], b[0:p] + a[p:]

# this is will be used in tsp to insure that 
def ordered_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    if len(a) != len(b):
        # Ensure both genomes are of the same length
        min_length = min(len(a), len(b))
        a = a[:min_length]
        b = b[:min_length]

    length = len(a)
    if length < 2:
        return a, b

    p1, p2 = sorted([randint(0, length - 1) for _ in range(2)])
    segment_a = a[p1:p2]
    segment_b = b[p1:p2]
 
    child_a = [city for city in b if city not in segment_a]   # g1 : 0,1,2,3,4,5,6,7,8,9,10,11   p1 = 2 , p2 = 5  , seg_a = 2,3,4,5
                                                              # g2 : 11,10,9,8,7,6,5,4,3,2,1,0                       seg_b = 9,8,7,6
                                                              # child_a = 11,10,9,8,7,6,1,0    , child_b = 0,1,2,3,4,5,10,11
                                                              # child_a = 11,10,2,3,4,5,9,8,7,6,1,0
                                                              # child_b = 0,1,9,8,7,6,2,3,4,5,10,11

    child_b = [city for city in a if city not in segment_b]

    # Fill in the segments at their original positions
    child_a[p1:p2] = segment_a
    child_b[p1:p2] = segment_b

    return child_a, child_b

def ordered_crossover_1(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    if len(a) != len(b):
        raise ValueError("Genomes a and b must be of the same length")

    length = len(a)
    if length < 2:
        return a, b

    p1, p2 = sorted([randint(0, length - 1) for _ in range(2)])
    child_a = [-1] * length
    child_b = [-1] * length

    # Copy the segment from parent a to child a
    child_a[p1:p2] = a[p1:p2]
    # Copy the segment from parent b to child b
    child_b[p1:p2] = b[p1:p2]

    # Fill in the remaining positions in child a with values from parent b
    idx = p2
    for value in b[p2:] + b[:p2]:
        if value not in child_a:
            child_a[idx % length] = value
            idx += 1

    # Fill in the remaining positions in child b with values from parent a
    idx = p2
    for value in a[p2:] + a[:p2]:
        if value not in child_b:
            child_b[idx % length] = value
            idx += 1

    return child_a, child_b


def mutation(genome: Genome, num: int = 1, probability: float = 0.5) -> Genome:
    for _ in range(num):
        index = randrange(len(genome))
        genome[index] = genome[index] if random() > probability else abs(genome[index] - 1)
    return genome


# swap mutation will use in the tsp to ensure not to visit the same point twice 
def swap_mutation(genome: Genome, num: int = 1) -> Genome:
    for _ in range(num):
        index_a, index_b = randrange(len(genome)), randrange(len(genome))
        genome[index_a], genome[index_b] = genome[index_b], genome[index_a]
    return genome



def population_fitness(population: Population, fitness_func: FitnessFunc) -> int:
    return sum([fitness_func(genome) for genome in population])


def selection_pair(population: Population, fitness_func: FitnessFunc) -> Population:
    return choices(
        population=population,
        weights=[fitness_func(gene) for gene in population],
        k=2
    )


def sort_population(population: Population, fitness_func: FitnessFunc) -> Population:
    return sorted(population, key=fitness_func, reverse=True)


def genome_to_string(genome: Genome) -> str:
    return "".join(map(str, genome))


def print_stats(population: Population, generation_id: int, fitness_func: FitnessFunc):
    print("GENERATION %02d" % generation_id)
    print("=============")
    print("Population: [%s]" % ", ".join([genome_to_string(gene) for gene in population]))
    print("Avg. Fitness: %f" % (population_fitness(population, fitness_func) / len(population)))
    sorted_population = sort_population(population, fitness_func)
    best_genome = sorted_population[0]
    print("Best Route: %s (Total Distance: %f)" % (genome_to_cities(best_genome), fitness_func(best_genome)))
    print("")
    return best_genome




def run_evolution(
        populate_func: PopulateFunc,
        fitness_func: FitnessFunc,
        fitness_limit: int,
        selection_func: SelectionFunc = selection_pair,
        crossover_func: CrossoverFunc = ordered_crossover,
        mutation_func: MutationFunc = swap_mutation,
        generation_limit: int = 100,
        printer: Optional[PrinterFunc] = None) \
        -> Tuple[Genome, int]:
    population = populate_func()

    best_genome = None
    best_fitness = float('-inf')

    for i in range(generation_limit):
        population = sorted(population, key=lambda genome: fitness_func(genome), reverse=True)

        if printer is not None:
            current_best_genome = printer(population, i, fitness_func)
        else:
            current_best_genome = population[0]

        current_best_fitness = fitness_func(current_best_genome)

        if current_best_fitness > best_fitness:
            best_genome = current_best_genome
            best_fitness = current_best_fitness

        if best_fitness >= fitness_limit:
            return best_genome, best_fitness

        next_generation = population[0:2]

        for j in range(int(len(population) / 2) - 1):
            parents = selection_func(population, fitness_func)
            offspring_a, offspring_b = crossover_func(parents[0], parents[1])
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
            next_generation += [offspring_a, offspring_b]

        population = next_generation

    # Return the best genome and its fitness value if the limit is not reached
    return best_genome, best_fitness



def genome_to_cities(genome):
    return [city_names[index] for index in genome]



if __name__ == "__main__":
    def tsp_population() -> Population:
        return generate_population(size=50, genome_length=12)

    tsp_fitness_limit = 10000 # Set an appropriate fitness limit
    tsp_generation_limit = 100
    tsp_printer: Optional[PrinterFunc] = print_stats

    best_route, distance = run_evolution(
    populate_func=tsp_population,
    fitness_func=total_distance,
    fitness_limit=tsp_fitness_limit,
    selection_func=selection_pair,
    crossover_func=ordered_crossover_1,
    mutation_func=swap_mutation,
    generation_limit=tsp_generation_limit,
    printer=tsp_printer
)

print("Best Route:",genome_to_cities( best_route))
print("Total Distance:", distance)

