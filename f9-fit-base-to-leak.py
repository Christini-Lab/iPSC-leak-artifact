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

plt.rcParams['lines.linewidth'] = .9
plt.rcParams['lines.markersize'] = 4
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.labelsize'] = 10 
plt.rcParams['axes.labelsize'] = 10 
plt.rc('legend', fontsize = 8)


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
                                              'ibca.g_b_Ca'
                                              ],
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

    # Use this if you don't want multi-threading or get an error
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
    mfile = './mmt/kernik_2019_mc_fixed.mmt'
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
    mfile = './mmt/kernik_leak_fixed.mmt'
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
                    upper_bound=10,
                    axs=None):

    if gen is None:
        gen = len(inds) - 1

    pop = inds[gen]

    pop.sort(key=lambda x: x.fitness.values[0])
    best_ind = pop[0]

    if is_top_ten:
        pop = pop[0:10]

    keys = [k for k in pop[0][0].keys()]

    #SETUP Axes because need to rasterize
    axs[0].set_xticks([i for i in range(0, len(keys))])
    axs[0].set_xticklabels([r'$g_{bNa}$', '$g_{bCa}$'])
    axs[0].set_ylim(log10(lower_bound),
                    log10(upper_bound))
    axs[0].set_ylabel(r'$Log_{10}$ $g_{scale}$')

    axs[1].set_ylabel('Voltage (mV)')
    axs[1].set_xlabel('Time (ms)')

    empty_arrs = [[] for i in range(len(keys))]
    all_ind_dict = dict(zip(keys, empty_arrs))

    fitnesses = []

    for ind in pop:
        for k, v in ind[0].items():
            all_ind_dict[k].append(v)

        fitnesses.append(ind.fitness.values[0])

    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.75))
        fig.subplots_adjust(.11, .15, .96, .98)


    for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    curr_x = 0

    for k, conds in all_ind_dict.items():
        for i, g in enumerate(conds):
            g = log10(g)
            x = curr_x + np.random.normal(0, .05)
            g_val = 1 - fitnesses[i] / (max(fitnesses)+.5)
            #axs[0].scatter(x, g, color=(g_val, g_val, g_val), alpha=.4)
            axs[0].scatter(x, g, color=(.7, .7, .7), alpha=.2)
            #if i < 10:
            #    axs[0].scatter(x-.1, g, color='r')


        curr_x += 1

    best_vals = [log10(v) for v in best_ind[0].values()]
    x_vals = [i for i in range(0, len(best_vals))]
    axs[0].scatter(x_vals, best_vals, c='tomato', marker='s', s=20)


    curr_x = 0

    axs[0].hlines(0, -.5, (len(keys)-.5), colors='grey', linestyle='--')
    #axs[0].set_xticks([i for i in range(0, len(keys))])
    ##axs[0].set_xticklabels([r'$G_{bNa}$', '$G_{bCa}$', '$G_{NaK}$'])
    #axs[0].set_xticklabels([r'$g_{bNa}$', '$g_{bCa}$'])
    #axs[0].set_ylim(log10(lower_bound),
    #                log10(upper_bound))
    #axs[0].set_ylabel(r'$Log_{10}$ $g_{scale}$')

    t, v = get_target_ap()
    axs[1].plot(t, v, 'k', label='Original w Leak', rasterized=True)

    #t, v, cai, i_ion = get_normal_sim_dat(best_ind)
    t, v = get_kernik_ap(best_ind[0])
    axs[1].plot(t, v, 'tomato', linestyle='--', label='Fit', rasterized=True)

    print('Best individual')
    print(best_ind[0])

    t, v = get_kernik_ap()
    axs[1].plot(t, v, c='grey', linestyle='dotted', alpha=.5,
                                            label='Original', rasterized=True)

    #t, v, cai, i_ion = get_normal_sim_dat(None)

    #axs[1].set_ylabel('Voltage (mV)')
    #axs[1].set_xlabel('Time (ms)')

    axs[1].legend(loc=1, framealpha=1, bbox_to_anchor=(1.1, 1.02))
    axs[1].set_ylim(-80, 48)
    axs[1].set_xlim(-50, 1100)

    matplotlib.rcParams['pdf.fonttype'] = 42


creator.create('FitnessMin', base.Fitness, weights=(-1.0,))

creator.create('Individual', list, fitness=creator.FitnessMin)


def plot_background_currs(axs):
    #1. Baseline+leak model response
    #2. Fit model response

    scales = [{'membrane.gLeak': .2,
              'ibna.g_b_Na': 1,
              'ibca.g_b_Ca': 1,
              'inak.g_scale': 1},
              {'ibna.g_b_Na': 6.6189158206846255,
               'ibca.g_b_Ca': 0.24933241003255358,
               'inak.g_scale': 0.8430817227634139}, 
              {'ibna.g_b_Na': 1,
               'ibca.g_b_Ca': 1}
               ]

    scales = [{'membrane.gLeak': .2,
              'ibna.g_b_Na': 1,
              'ibca.g_b_Ca': 1
              },
              {'ibna.g_b_Na': 7.081649113309206,
               'ibca.g_b_Ca': 0.3127089921965326}, 
              {'ibna.g_b_Na': 1,
               'ibca.g_b_Ca': 1}
               ]

    iv_curves = []
    voltages = np.arange(-90, 60, 10)

    for i, mod in enumerate(['mmt/kernik_leak_fixed.mmt',
                             'mmt/kernik_2019_mc_fixed.mmt']):
        mod = myokit.load_model(mod)

        for name, scale in scales[i].items():
            group, param = name.split('.')
            val = mod[group][param].value()
            mod[group][param].set_rhs(val*scale)

        sim = myokit.Simulation(mod)

        t_max = 100000
        times = np.arange(0, t_max, .5)

        res_base = sim.run(t_max, log_times=times)

        g_b_Na = .00029 * 1.5
        g_b_Ca = .000592 * .62
        Cm = 60

        iv_curves.append({'i_na': np.array([]),
                          'i_ca': np.array([]),
                          'i_leak': np.array([]),
                          'i_nak': np.array([])})

        for v in voltages:
            i_na = (scales[i]['ibna.g_b_Na'] *
                        g_b_Na * (v - res_base['erev.E_Na'][-1]))
            i_ca = (scales[i]['ibca.g_b_Ca'] *
                    g_b_Ca * (v - np.min(res_base['erev.E_Ca'][-2000:]))) 

            iv_curves[i]['i_na'] = np.append(iv_curves[i]['i_na'], i_na)
            iv_curves[i]['i_ca'] = np.append(iv_curves[i]['i_ca'], i_ca)

            if i == 0:
                i_leak = scales[i]['membrane.gLeak'] * v / Cm 
                iv_curves[i]['i_leak'] = np.append(iv_curves[i]['i_leak'], i_leak)

            #INaK
            #g_scale = scales[i]['inak.g_scale']
            #PNaK = 1.362 * 1.818
            #Ko = 5.4
            #Nai = res_base['nai.Nai'][-1]
            #Km_K = 1
            #Km_Na = 40
            #FRT = 96.4853415 / (8.314472 * 310)
            #i_nak = g_scale * PNaK * Ko * Nai / ((Ko + Km_K) * (Nai + Km_Na) * (1 + 0.1245 * np.exp(-0.1 * v * FRT) + 0.0353 * np.exp(-v * FRT)))
            #iv_curves[i]['i_nak'] = np.append(iv_curves[i]['i_nak'], i_nak)

    linestyles = ['-', '-', '-', '--', '--']
    markers = ['^', 's', 'o', '^', 's']
    colors = ['k', 'k', 'k', 'tomato', 'tomato']

    axs[0].plot(voltages, iv_curves[0]['i_na'], c='k', marker='^',
            linestyle='-', label=r'Original: $I_{bNa}$')
    axs[0].plot(voltages, iv_curves[0]['i_ca'], c='k', marker='s',
            linestyle='-', label=r'Original: $I_{bCa}$')
    axs[0].plot(voltages, iv_curves[0]['i_leak'], c='k', marker='o',
            linestyle='-', label=r'$I_{leak}$ w 5$G\Omega$')
    axs[0].plot(voltages, iv_curves[1]['i_na'], c='tomato', marker='^',
            linestyle='--', label=r'Fit: $I_{bNa}$')
    axs[0].plot(voltages, iv_curves[1]['i_ca'], c='tomato', marker='s',
            linestyle='--', label=r'Fit: $I_{bCa}$')

    i_base_leak = (iv_curves[0]['i_na'] +
                        iv_curves[0]['i_ca'] + iv_curves[0]['i_leak']
                        )
    i_fit = (iv_curves[1]['i_na'] + iv_curves[1]['i_ca'])
    i_base = (iv_curves[0]['i_na'] + iv_curves[0]['i_ca'])
    
    curr_names = [r'Original w leak: $I_{bNa}+I_{bCa}+I_{leak}$', r'Fit: $I_{bNa} + I_{bCa}$',
            r'Original: $I_{bNa}+I_{bCa}$']

    axs[1].plot(voltages, i_base_leak, c='k', marker='o', linestyle='-',
            label=curr_names[0])
    axs[1].plot(voltages, i_fit, c='tomato', marker='o', linestyle='--',
            label=curr_names[1])
    axs[1].plot(voltages, i_base, c='grey', marker='o', linestyle='dotted',
            label=curr_names[2])

    axs[0].set_ylim(-.79, .21)
    axs[0].legend(framealpha=1)
    axs[1].set_ylim(-.6, .46)
    axs[1].legend(loc=2, framealpha=1)

    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Voltage (mV)')
        ax.set_ylabel('Current (A/F)')


def plot_figure():
    fig, axs = plt.subplots(2, 2, figsize=(6.5, 5.5))

    fig.subplots_adjust(.11, .09, .96, .95, wspace=.25, hspace=.2)

    all_individuals = pickle.load(
            open('./data/ga_results/inds_bCa_bNa_fixed.pkl', 'rb'))

    plot_generation(all_individuals,
                    gen=None,
                    is_top_ten=False,
                    lower_bound=.1,
                    upper_bound=10,
                    axs=axs[0])

    plot_background_currs(axs[1])

    letters = ['A', 'B', 'C', 'D']

    axs = axs.flatten()

    for i, ax in enumerate(axs):
        ax.set_title(letters[i], y=.94, x=-.2)

    matplotlib.rcParams['pdf.fonttype'] = 42
    plt.savefig('./figure-pdfs/f9.pdf', transparent=True, dpi=1000)

    plt.show()


def test_plot_bg_sodium():
    fig, axs = plt.subplots(2, 1,sharex=True, figsize=(12, 8))

    inds = pickle.load(open('./data/ga_results/inds.pkl', 'rb'))
    pop = inds[-1]

    pop.sort(key=lambda x: x.fitness.values[0])
    best_ind = pop[0]

    labs = ['original', 'best ind']

    for i, param_updates in enumerate([{}, best_ind[0]]):
        mfile = './mmt/kernik_2019_mc_fixed.mmt'
        k_mod, p, x = myokit.load(mfile)

        for param, val in param_updates.items():
            group, key = param.split('.')
            k_mod[group][key].set_rhs(val*k_mod[group][key].value())

        s_base = myokit.Simulation(k_mod)

        t_max = 500000

        times = np.arange(0, t_max, .5)

        res_base = s_base.run(t_max, log_times=times)

        t = res_base.time()
        v = res_base['membrane.V']

        axs[0].plot(t, v)
        axs[1].plot(t, res_base['nai.Nai'], label=labs[i])

    
    mfile = './mmt/kernik_leak_fixed.mmt'
    k_mod, p, x = myokit.load(mfile)

    k_mod['membrane']['gLeak'].set_rhs(.2)

    s_base = myokit.Simulation(k_mod)

    t_max = 500000

    times = np.arange(0, t_max, .5)

    res_base = s_base.run(t_max, log_times=times)

    t = res_base.time()
    v = res_base['membrane.V']

    axs[0].plot(t, v)
    axs[1].plot(t, res_base['nai.Nai'], label='Baseline+Leak')

    axs[0].set_ylabel('Voltage (mV)')
    axs[1].set_ylabel('Nai (mM)')
    axs[1].set_xlabel('Time (ms)')

    axs[1].legend()
    
    plt.show()


def fit_model():
    all_individuals = start_ga(pop_size=150, max_generations=20)

    pickle.dump(all_individuals, open('./data/ga_results/inds_bCa_bNa_fixed.pkl', 'wb'))


def main():
    # Uncomment this if you want to run the GA
    #fit_model() # Fit the model

    # Uncomment this to plot figure 9
    plot_figure()


if __name__ == '__main__':
    main()
