import pandas as pd
import numpy as np
from random import shuffle
from bitstring import BitArray
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import statistics
import math
from linreg import LinReg

file = pd.read_csv(r'/Users/beanie/Desktop/genetic_algorithm_folder/files/MSFT.csv')

x_file = file[['Open', 'High', 'Low', 'Adj Close', 'Volume']]
y_file = file['Close']
# x_file = file[['Open', 'High', 'Low', 'Adj Close', 'Volume']].drop(labels=range(1, 6), axis=0)
# y_file = file['Close'].drop(labels=range(0, 6), axis=0)
# adding new columns
# previous opening

# cropped in advance

prev_open = file['Open'][6:-1]
# opening 2 days before
prev_open2 = file['Open'][5:-2]
prev_open3 = file['Open'][4:-3]

# average for the past trading week
avg_week = [statistics.mean(file['Open'][i - 7: i ])  for i, _ in enumerate(file['Open']) if i >= 7]
# median for the past trading week
median_week = [statistics.median(file['Open'][i -7 : i ]) for i, _ in enumerate(file['Open']) if i >= 7]

# x_file['Prev_open'] = prev_open
# x_file['Prev_open2'] = prev_open2
# x_file['Prev_open3'] = prev_open3
# x_file['avg_week'] = avg_week
# x_file['median_week'] = median_week


# BOUNDARIES = [0, 32]

X_MIN = 0
X_MAX = 2 ** (int(file.shape[1]) - 1) - 1

MUT_PROB = 0.2
CROSS_TIMES = 1

def fitness(individual):
    data = LinReg().get_columns(individual, x_file)
    result = LinReg().get_fitness(data, y_file)
    return result
    # TODO
    # result = 0

def entropy_calc(population):
    p = [list(tup[0]) for tup in population]
    p = np.asarray(p).transpose()
    # list_of_probs = [0 for i in range(len(population))]
    summ = 0
    for ind in range(len(p)):
        counter = np.count_nonzero(p[ind] == '1')
        if counter != 0:
            summ += counter * math.log(counter, 2)
    return - summ

def normalization(binary_number):
    # num = BitArray(bin=binary_number).int
    num = int(binary_number, 2)
    # frac = num / (2 ** (len(binary_number)) - 1)
    # num_norm = ((BOUNDARIES[1] - BOUNDARIES[0]) * frac) + BOUNDARIES[0]
    # return num_norm
    return num


def fitness_sine(individual):
    return math.sin(normalization(individual))

def pop_gen(n_bits, n_pop, fitness_function=fitness):
    res = list()
    for _ in range(n_pop):
        string = ''.join([str(i) for i in np.random.randint(0, 2, n_bits)])
        # tuple = [string, fitness_function(string)]
        if string != '00000':
            res.append((string, fitness_function(string)))
    return res

def median_fitness(population):
    return statistics.median([el[1] for el in population])

# another ways for parent selection - round with probabilities (roulette wheel selection) is below

def parent_selection(population, sign):
    selected = list()
    for tup in population:
        ind, fit = tup[0], tup[1]
        if fit >= median_fitness(population) and sign == '>':
            selected.append(tup)
        if fit <= median_fitness(population) and sign == '<':
            selected.append(tup)
    selected1 = selected
    shuffle(selected)
    res = selected + selected1
    shuffle(res)
    return res

def roulette_wheel_selection(population, sign):
    # Computes the totality of the population fitness
    # it works as since we divide by 1, the proportion is the same -> 1/8 < 1/5, though 5 < 8
    # the same will work with 0.132 > 0.120 , 1/ 0.132 < 1 / 0.120
    # hence it will be a maximization problem for inverted values.
    if sign == '<':
        population_fitness = sum([1/chromosome[1] for chromosome in population])

        # Computes for each chromosome the probability
        chromosome_probabilities = [(1 / (chromosome[1]))/population_fitness for chromosome in population]
    elif sign == '>':
        population_fitness = sum([chromosome[1] for chromosome in population])
        chromosome_probabilities = [chromosome[1]/population_fitness for chromosome in population]
    else:
        print('If you want a maximazing funtion, enter > as a parameter, otherwise <')
    # Selects one chromosome based on the computed probabilities
    r = []
    indexes = [i for i in range(len(population))]
    for _ in range(len(population)):
        chosen = np.random.choice(indexes, p=chromosome_probabilities)
        r.append(population[chosen])
    return r



def mutation(mut_prob, bitstring):
    for i in range(len(bitstring)):
        # check for a mutation
        if np.random.rand() < mut_prob:
            # flip the bit
            if bitstring[i] == '0':
                bitstring = bitstring[:i] + '1' + bitstring[i + 1:]
            if bitstring[i] == '1':
                bitstring = bitstring[:i] + '0' + bitstring[i + 1:]
    return bitstring



def crossover(p1, p2, r_cross):
    pt = np.random.randint(1, len(p1[0])-2)
        # perform crossover
    c1 = p1[0][:pt] + p2[0][pt:]
    c2 = p2[0][:pt] + p1[0][pt:]
    return [c1, c2]


def offspring_pop(parents, fitness_function=fitness):
    n = len(parents)
    children = list()
    for i in range(0, n, 2):
        # get selected parents in pairs
        p1, p2 = parents[i], parents[i + 1]
        # crossover and mutation
        for c in crossover(p1, p2, CROSS_TIMES):
            # mutation
            mutation(MUT_PROB, c)
            # store for next generation
            children.append((c, fitness_function(c)))
    return children


def best_fitness(population, n_high, bool):
    pop2 = population
    pop2.sort(key=lambda x: x[1], reverse=bool)
    return [[normalization(tup[0]), tup[1]]for tup in pop2[-n_high:]]


NUM_GENERATIONS = 5
SIGN = '<'
columns = 5
n_top_individuals = 6
# plt.plot(x, y, color='black')
pop = pop_gen(columns, 30)
pair = best_fitness(pop, n_top_individuals, False)
x = [i[0] for i in pair]
y = [int(i[1]*1000)/1000 for i in pair]
print(x, y)
# plt.scatter([1, 2, 3], [3, 4, 5])
# plt.scatter(x, y, color='black', label = f'0-th generation')
# print(median_fitness(pop))

for i in range(NUM_GENERATIONS):
    pop_selected = parent_selection(pop, SIGN)
    pop = offspring_pop(pop_selected)
    pair = best_fitness(pop, n_top_individuals, False)
    x = [i[0] for i in pair]
    y = [i[1] for i in pair]
    print(x, y)
    if i == NUM_GENERATIONS - 1:
        print(x, 'this is the encoding for the subset of columns best for prediction')
    # plt.scatter(x, y, label = f'{i + 1}-th generation')
# plt.legend()
# plt.show()

