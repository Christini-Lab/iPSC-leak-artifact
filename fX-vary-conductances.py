import myokit
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import matplotlib
from scipy.signal import find_peaks
from multiprocessing import Pool

from utility_classes import get_single_ap


plt.rcParams['lines.linewidth'] = .9
plt.rcParams['lines.markersize'] = 4
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rc('legend', fontsize = 8)


def plot_lin_leak():
    fig = plt.figure(figsize=(6.5, 5))

    fig.subplots_adjust(.1, .1, .95, .95)

    grid = fig.add_gridspec(2, 1, hspace=.15, wspace=0.1)

    all_leaks = np.linspace(0, 1, 10)
    all_leaks = 1/10**all_leaks

    subgrid = grid[0, 0].subgridspec(2, 4, wspace=0.8, hspace=.3)
    plot_mod_leak(subgrid, fig, all_leaks, './mmt/kernik_2019_mc_fixed.mmt', './mmt/kernik_leak_fixed.mmt')

    subgrid = grid[1, 0].subgridspec(2, 4, wspace=0.8, hspace=.3)
    plot_mod_leak(subgrid, fig, all_leaks, './mmt/paci-2013-ventricular-fixed.mmt', './mmt/paci-2013-ventricular-leak-fixed.mmt')

    matplotlib.rcParams['pdf.fonttype'] = 42
    plt.savefig('./figure-pdfs/f-leak-model-effects.pdf', transparent=True)

    plt.show()


global LIMIT_EDGE
LIMIT_EDGE = .05


def plot_models():
    p = Pool()
    dat = p.map(run_simulation, range(0, 70))
    #dat = [run_simulation(i) for i in range(0, 5)]

    fig = plt.figure(figsize=(6.5, 4))
    fig.subplots_adjust(.1, .1, .95, .95)

    grid = fig.add_gridspec(1, 2, hspace=.15, wspace=0.3)

    subgrid = grid[0].subgridspec(1, 1, wspace=.9, hspace=.1)
    ax_aps = fig.add_subplot(subgrid[0]) 

    subgrid = grid[1].subgridspec(2, 2, wspace=.6, hspace=.1)
    axs_biom = [fig.add_subplot(subgrid[0, 0]),
                fig.add_subplot(subgrid[0, 1]),
                fig.add_subplot(subgrid[1, 0]),
                fig.add_subplot(subgrid[1, 1])]

    for sim in dat:
        t, v, biomarkers, leak = sim
        if biomarkers is None:
            continue
            
        ax_aps.plot(t, v)

        for i, val in enumerate(biomarkers):
            axs_biom[i].scatter(1/leak, val, c='k')

    ax_aps.set_xlabel('Time (ms)')
    ax_aps.set_ylabel('Voltage (mV)')
    ax_aps.set_xlim(-100, 600)
    ax_aps.set_title(f'+/- {LIMIT_EDGE*100}%', y=.9)

    all_axs = [ax_aps] + axs_biom

    for ax in all_axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    biom_names = ['MDP (mV)', 'CL (ms)', r'$dV/dt_{max}$ (V/s)', r'$APD_{90}$ (ms)']

    mdp_range = [-80, -35]
    cl_range = [200, 1900]
    dvdt_range = [-2, 30]
    apd90_range = [200, 500]

    biomarker_ranges = [mdp_range, cl_range, dvdt_range, apd90_range]
    baseline_vals = [-75.5, 982.0, 26.3, 413.3]

    for i, ax in enumerate(axs_biom):
        ax.set_xscale('log')
        labs = [1]
        labs += [None for i in range(2, 10)]
        labs += [10]
        ax.set_xticks(list(range(1,11)))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xticklabels(labs)
        ax.set_ylabel(biom_names[i])
        ax.set_ylim(biomarker_ranges[i][0], biomarker_ranges[i][1])
        ax.axhline(baseline_vals[i], color=(1, .2, .2), linestyle='--')


    axs_biom[2].set_xlabel(r'$R_{leak} (G\Omega)$')
    axs_biom[3].set_xlabel(r'$R_{leak} (G\Omega)$')

    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel(r'$APD_{90}$')
    ax.set_ylabel(r'MDP')

    for sim in dat:
        t, v, biomarkers, leak = sim
        if biomarkers is None:
            continue

        ax.scatter(biomarkers[3], biomarkers[0], c='k')

    ax.set_xlim(0, 600)
    ax.set_ylim(-100, -35)

    plt.show()



def run_simulation(_):
    mod, p, x = myokit.load('./mmt/kernik_leak_fixed.mmt')
    limit_edge = LIMIT_EDGE 

    #1. Change parameter values
    #cond_names = ['ik1.g_K1',
    #                     'ikr.g_Kr',
    #                     'iks.g_Ks',
    #                     'ito.g_to',
    #                     'ical.p_CaL',
    #                     'icat.g_CaT',
    #                     'inak.PNaK',
    #                     'ina.g_Na',
    #                     'inaca.kNaCa',
    #                     'ipca.g_PCa',
    #                     'ifunny.g_f',
    #                     'ibna.g_b_Na',
    #                     'ibca.g_b_Ca']
    np.random.seed()
    #cond_scales = [np.random.uniform(1-limit_edge, 1+limit_edge) for i in
    #                                range(0, len(cond_names))]
    #new_conductances = dict(zip(cond_names, cond_scales))
    new_conductances = {}
    gLeak = 1/(10**np.random.uniform(0, 1))
    new_conductances['membrane.gLeak'] = gLeak
    new_conductances['geom.Cm'] = np.random.uniform(15, 90)
    #print(gLeak)

    for param, val in new_conductances.items():
        group, key = param.split('.')
        #mod[group][key].set_rhs(val*mod[group][key].value())
        mod[group][key].set_rhs(val)

    #2. Simulate model
    s_base = myokit.Simulation(mod)

    t_max = 100000

    times = np.arange(0, t_max, .5)

    res_base = s_base.run(t_max, log_times=times)

    t = res_base.time()
    v = res_base['membrane.V']

    single_t, single_v = get_single_ap(t, v)

    #3. Calculate biomarkers
    try:
        biomarkers = get_biomarkers(t[-20000:], v[-20000:])
    except:
        biomarkers = None

    print(biomarkers)

    #4. Return t, v, and biomarkers
    return (single_t, single_v, biomarkers, gLeak)



def get_biomarkers(t, v):
    t, v = np.asarray(t), np.asarray(v)
    mdp = np.min(v)

    pks = find_peaks(v, distance=200, height=-40)[0]
    dt = t[pks[3]] - t[pks[2]]

    dvdt_info = find_peaks(np.diff(v)/np.diff(t), height=.1, distance=150)
    dvdt = np.average(dvdt_info[1]['peak_heights'])

    dvdt_idxs = dvdt_info[0]
    min_v_idxs = find_peaks(-v[dvdt_idxs[0]:], height=30, distance=100)[0]+dvdt_idxs[0]
    apd90 = []
    for i, dvdt_idx in enumerate(dvdt_idxs[:-1]):
        v_range = v[dvdt_idx: min_v_idxs[i]]
        t_range = t[dvdt_idx: min_v_idxs[i]]
        amp = np.max(v_range) - np.min(v_range)
        apd90_v = np.max(v_range) - amp*.9
        pk_idx = np.argmax(v_range)

        apd90_idx = np.argmin(np.abs(v_range[pk_idx:] - apd90_v)) + pk_idx
        apd90.append(t_range[apd90_idx] - t_range[0])

    return mdp, dt, dvdt, np.average(apd90)


def main():
    plot_models()


if __name__ == '__main__':
    main()
