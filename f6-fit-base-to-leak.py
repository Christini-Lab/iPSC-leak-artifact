import random
from math import log10
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.signal import find_peaks
import pandas as pd
import pickle

from multiprocessing import Pool
from deap import base, creator, tools # pip install deap
import myokit

random.seed(1)

#from supercell_ga import Ga_Config, plot_generation
def start_ga(pop_size=20, max_generations=10):
    # 1. Initializing GA hyperparameters
    target_t, target_v = get_target_ap()

    global GA_CONFIG
    GA_CONFIG = Ga_Config(population_size=pop_size,
                          max_generations=max_generations,
                          params_lower_bound=0.1,
                          params_upper_bound=10,
                          tunable_parameters=['ibna.g_b_Na',
                                              'ibca.g_b_Ca',
                                              'inak.g_scale'],
                          mate_probability=0.9,
                          mutate_probability=0.9,
                          gene_swap_probability=0.2,
                          gene_mutation_probability=0.2,
                          tournament_size=2,
                          target=[target_t, target_v])
    
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))

    creator.create('Individual', list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register('init_param',
                     _initialize_individual)
    toolbox.register('individual',
                     tools.initRepeat,
                     creator.Individual,
                     toolbox.init_param,
                     n=1)
    toolbox.register('population',
                     tools.initRepeat,
                     list,
                     toolbox.individual)

    toolbox.register('evaluate', _evaluate_fitness)
    toolbox.register('select',
                     tools.selTournament,
                     tournsize=GA_CONFIG.tournament_size)
    toolbox.register('mate', _mate)
    toolbox.register('mutate', _mutate)

    # To speed things up with multi-threading
    p = Pool()
    toolbox.register("map", p.map)

    # Use this if you don't want multi-threading
    #toolbox.register("map", map)

    # 2. Calling the GA to run
    final_population = run_ga(toolbox)

    return final_population


def run_ga(toolbox):
    """
    Runs an instance of the genetic algorithm.

    Returns
    -------
        final_population : List[Individuals]
    """
    print('Evaluating initial population.')

    # 3. Calls _initialize_individuals GA_CONFIG.population size number of times and returns the initial population
    population = toolbox.population(GA_CONFIG.population_size)


    # 4. Calls _evaluate_fitness on every individual in the population
    fitnesses = toolbox.map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = (fit,)
        
    # Note: visualize individual fitnesses with: population[0].fitness
    gen_fitnesses = [ind.fitness.values[0] for ind in population]

    print(f'\tAvg fitness is: {np.mean(gen_fitnesses)}')
    print(f'\tBest fitness is {np.min(gen_fitnesses)}')

    # Store initial population details for result processing.
    final_population = [population]

    for generation in range(1, GA_CONFIG.max_generations):
        print('Generation {}'.format(generation))
        # Offspring are chosen through tournament selection. They are then
        # cloned, because they will be modified in-place later on.

        # 5. DEAP selects the individuals 
        selected_offspring = toolbox.select(population, len(population))

        offspring = [toolbox.clone(i) for i in selected_offspring]

        # 6. Mate the individualse by calling _mate()
        for i_one, i_two in zip(offspring[::2], offspring[1::2]):
            if random.random() < GA_CONFIG.mate_probability:
                toolbox.mate(i_one, i_two)
                del i_one.fitness.values
                del i_two.fitness.values

        # 7. Mutate the individualse by calling _mutate()
        for i in offspring:
            if random.random() < GA_CONFIG.mutate_probability:
                toolbox.mutate(i)
                del i.fitness.values

        # All individuals who were updated, either through crossover or
        # mutation, will be re-evaluated.
        
        # 8. Evaluating the offspring of the current generation
        updated_individuals = [i for i in offspring if not i.fitness.values]
        fitnesses = toolbox.map(toolbox.evaluate, updated_individuals)
        for ind, fit in zip(updated_individuals, fitnesses):
            ind.fitness.values = (fit,)

        population = offspring

        gen_fitnesses = [ind.fitness.values[0] for ind in population]

        print(f'\tAvg fitness is: {np.mean(gen_fitnesses)}')
        print(f'\tBest fitness is {np.min(gen_fitnesses)}')

        final_population.append(population)

    return final_population


def _initialize_individual():
    # Builds a list of parameters using random upper and lower bounds.
    lower_exp = log10(GA_CONFIG.params_lower_bound)
    upper_exp = log10(GA_CONFIG.params_upper_bound)
    initial_params = [10**random.uniform(lower_exp, upper_exp)
                      for i in range(0, len(
                          GA_CONFIG.tunable_parameters))]

    keys = [val for val in GA_CONFIG.tunable_parameters]
    return dict(zip(keys, initial_params))


def _evaluate_fitness(ind):
    sim_t, sim_v = get_kernik_ap(ind[0])

    try:
        v_diff_rmse = (np.array(GA_CONFIG.target[1]) - np.array(sim_v))**2
    except:
        return 10000000
        print(sim_v)
    #v_diff_abs = np.abs(np.array(GA_CONFIG.target[1]) - np.array(sim_v))
    err = np.sum(v_diff_rmse)

    print(err)

    return err 

    # Returns
    if feature_error == 500000:
        return feature_error

    #ead_fitness = get_ead_error(ind)
    fitness = feature_error #+ ead_fitness

    return fitness


def _mate(i_one, i_two):
    """Performs crossover between two individuals.

    There may be a possibility no parameters are swapped. This probability
    is controlled by `GA_CONFIG.gene_swap_probability`. Modifies
    both individuals in-place.

    Args:
        i_one: An individual in a population.
        i_two: Another individual in the population.
    """
    for key, val in i_one[0].items():
        if random.random() < GA_CONFIG.gene_swap_probability:
            i_one[0][key],\
                i_two[0][key] = (
                    i_two[0][key],
                    i_one[0][key])


def _mutate(individual):
    """Performs a mutation on an individual in the population.

    Chooses random parameter values from the normal distribution centered
    around each of the original parameter values. Modifies individual
    in-place.

    Args:
        individual: An individual to be mutated.
    """
    keys = [k for k, v in individual[0].items()]

    for key in keys:
        if random.random() < GA_CONFIG.gene_mutation_probability:
            new_param = -1

            while ((new_param < GA_CONFIG.params_lower_bound) or
                   (new_param > GA_CONFIG.params_upper_bound)):
                new_param = np.random.normal(
                        individual[0][key],
                        individual[0][key] * .1)

            individual[0][key] = new_param


class Ga_Config():
    def __init__(self,
             population_size,
             max_generations,
             params_lower_bound,
             params_upper_bound,
             tunable_parameters,
             mate_probability,
             mutate_probability,
             gene_swap_probability,
             gene_mutation_probability,
             tournament_size,
             target):
        self.population_size = population_size
        self.max_generations = max_generations
        self.params_lower_bound = params_lower_bound
        self.params_upper_bound = params_upper_bound
        self.tunable_parameters = tunable_parameters
        self.mate_probability = mate_probability
        self.mutate_probability = mutate_probability
        self.gene_swap_probability = gene_swap_probability
        self.gene_mutation_probability = gene_mutation_probability
        self.tournament_size = tournament_size
        self.target=target


def get_kernik_ap(param_updates={}):
    mfile = './mmt/kernik_2019_mc.mmt'
    k_mod, p, x = myokit.load(mfile)

    for param, val in param_updates.items():
        group, key = param.split('.')
        k_mod[group][key].set_rhs(val*k_mod[group][key].value())

    s_base = myokit.Simulation(k_mod)

    t_max = 100000

    times = np.arange(0, t_max, .5)

    res_base = s_base.run(t_max, log_times=times)

    t = res_base.time()
    v = res_base['membrane.V']

    single_t, single_v = get_single_ap(t, v)

    return (single_t, single_v)


def get_target_ap(param_updates={'membrane.gLeak': .2}):
    mfile = './mmt/kernik_leak.mmt'
    k_mod, p, x = myokit.load(mfile)
    
    for param, val in param_updates.items():
        group, key = param.split('.')
        k_mod[group][key].set_rhs(val*k_mod[group][key].value())

    s_base = myokit.Simulation(k_mod)

    t_max = 100000

    times = np.arange(0, t_max, .5)

    res_base = s_base.run(t_max, log_times=times)

    t = res_base.time()
    v = res_base['membrane.V']

    single_t, single_v = get_single_ap(t, v)

    return (single_t, single_v)


def get_single_ap(t, v):
    t = np.array(t)

    interval = t[1] - t[0]

    min_v, max_v = np.min(v), np.max(v)

    if (max_v - min_v) < 10:
        return None, None

    dvdt_peaks = find_peaks(np.diff(v)/np.diff(t), distance=20/interval, width=1/interval, prominence=.3)[0]

    if dvdt_peaks.size  < 4:
        return None, None

    start_idx = int(dvdt_peaks[-3] - 200 / interval)
    end_idx = int(dvdt_peaks[-3] + 800 / interval)

    t = t - t[start_idx]

    return t[start_idx:end_idx], v[start_idx:end_idx] 


def plot_generation(inds,
                    gen=None,
                    is_top_ten=True,
                    lower_bound=.1,
                    upper_bound=10):
    if gen is None:
        gen = len(inds) - 1

    pop = inds[gen]

    pop.sort(key=lambda x: x.fitness.values[0])
    best_ind = pop[0]

    if is_top_ten:
        pop = pop[0:10]

    keys = [k for k in pop[0][0].keys()]
    empty_arrs = [[] for i in range(len(keys))]
    all_ind_dict = dict(zip(keys, empty_arrs))

    fitnesses = []

    for ind in pop:
        for k, v in ind[0].items():
            all_ind_dict[k].append(v)

        fitnesses.append(ind.fitness.values[0])

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(.07, .10, .95, .98)


    for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    curr_x = 0

    for k, conds in all_ind_dict.items():
        for i, g in enumerate(conds):
            g = log10(g)
            x = curr_x + np.random.normal(0, .05)
            g_val = 1 - fitnesses[i] / (max(fitnesses)+.5)
            axs[0].scatter(x, g, color=(g_val, g_val, g_val), alpha=.4)
            #if i < 10:
            #    axs[0].scatter(x-.1, g, color='r')


        curr_x += 1


    best_vals = [log10(v) for v in best_ind[0].values()]
    axs[0].scatter([0, 1, 2], best_vals, c='tomato', marker='s', s=45)


    curr_x = 0

    axs[0].hlines(0, -.5, (len(keys)-.5), colors='grey', linestyle='--')
    axs[0].set_xticks([i for i in range(0, len(keys))])
    axs[0].set_xticklabels(['GbNa', 'GbCa', 'GNaK'], fontsize=10)
    axs[0].set_ylim(log10(lower_bound),
                    log10(upper_bound))
    axs[0].set_ylabel('Log10 Conductance', fontsize=14)

    t, v = get_target_ap()
    axs[1].plot(t, v, 'k', label='Target w Leak')

    #t, v, cai, i_ion = get_normal_sim_dat(best_ind)
    t, v = get_kernik_ap(best_ind[0])
    axs[1].plot(t, v, 'tomato', linestyle='--', label='Best Fit')

    print('Best individual')
    print(best_ind[0])

    t, v = get_kernik_ap()
    axs[1].plot(t, v, c='grey', linestyle='--', alpha=.5,
                                            label='Baseline Kernik')

    #t, v, cai, i_ion = get_normal_sim_dat(None)

    axs[1].set_ylabel('Voltage (mV)', fontsize=14)
    axs[1].set_xlabel('Time (ms)', fontsize=14)

    axs[1].legend()

    #fig.suptitle(f'Generation {gen+1}', fontsize=14)

    matplotlib.rcParams['pdf.fonttype'] = 42
    plt.savefig('./figure-pdfs/f6-fit-bg-currs.pdf', transparent=True)


    plt.show()



creator.create('FitnessMin', base.Fitness, weights=(-1.0,))

creator.create('Individual', list, fitness=creator.FitnessMin)


def main():
    #all_individuals = start_ga(pop_size=150, max_generations=20)

    #pickle.dump(all_individuals, open('./data/ga_results/inds.pkl', 'wb'))

    all_individuals = pickle.load(open('./data/ga_results/inds.pkl', 'rb'))

    plot_generation(all_individuals,
                    gen=None,
                    is_top_ten=False,
                    lower_bound=.1,
                    upper_bound=10)


if __name__ == '__main__':
    main()

