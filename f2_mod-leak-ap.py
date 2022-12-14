import myokit
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from scipy.signal import find_peaks

from utility import get_single_ap


plt.rcParams['lines.linewidth'] = .9
plt.rcParams['lines.markersize'] = 4
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.labelsize'] = 10 
plt.rcParams['axes.labelsize'] = 10 
plt.rc('legend', fontsize = 8)


def plot_lin_leak():
    # Array with leak values
    all_leaks = np.linspace(0, 1, 10)
    all_leaks = 1/10**all_leaks
    
    # Make figure grid
    fig = plt.figure(figsize=(6.5, 5))
    fig.subplots_adjust(.1, .1, .95, .95)
    grid = fig.add_gridspec(1, 2, wspace=0.3)

    # Create subgrids for left column (APs) and right columns (biomarkers)
    subgrid_aps = grid[0, 0].subgridspec(2, 1, wspace=.3, hspace=.2)
    subgrid_biom = grid[0, 1].subgridspec(4, 2, wspace=.4, hspace=.3)

    # Plot APs and biomarkers for Kernik model (Panels A and C)
    ax_ap = fig.add_subplot(subgrid_aps[0])
    ax_ap.text(-300, 40, 'A', fontsize=12)
    axs_biom = [fig.add_subplot(subgrid_biom[0, 0]),
                fig.add_subplot(subgrid_biom[1, 0]),
                fig.add_subplot(subgrid_biom[2, 0]),
                fig.add_subplot(subgrid_biom[3, 0])]
    axs_biom[0].text(.05, -35, 'C', fontsize=12)
    plot_mod_leak(ax_ap, axs_biom, fig, all_leaks, './mmt/kernik_2019_mc_fixed.mmt', './mmt/kernik_leak_fixed.mmt')

    # Plot APs and biomarkers for Paci model (Panels B and D)
    ax_ap = fig.add_subplot(subgrid_aps[1])
    ax_ap.text(-300, 30, 'B', fontsize=12)
    axs_biom = [fig.add_subplot(subgrid_biom[0, 1]),
                fig.add_subplot(subgrid_biom[1, 1]),
                fig.add_subplot(subgrid_biom[2, 1]),
                fig.add_subplot(subgrid_biom[3, 1])]
    axs_biom[0].text(.05, -35, 'D', fontsize=12)
    plot_mod_leak(ax_ap, axs_biom, fig, all_leaks, './mmt/paci-2013-ventricular-fixed.mmt', './mmt/paci-2013-ventricular-leak-fixed.mmt')

    matplotlib.rcParams['pdf.fonttype'] = 42
    #plt.savefig('./figure-pdfs/f-leak-model-effects.pdf', transparent=True)
    # Save the figure
    plt.savefig('./figure-pdfs/f2.pdf', transparent=True)

    plt.show()


def plot_mod_leak(ax_ap, axs_biom, fig, all_leaks, base_model, leak_model):
    if 'kernik' in base_model:
        ax_ap.set_title('Kernik', y=.94)
        axs_biom[0].set_title('Kernik', y=.94)
    else:
        ax_ap.set_title('Paci', y=.94)
        axs_biom[0].set_title('Paci', y=.94)


    base, p, x = myokit.load(base_model)
    s_base = myokit.Simulation(base)
    prepace = 500000
    s_base.pre(prepace)
    res_base = s_base.run(10000, log_times=np.arange(0, 10000, 1))

    t = res_base.time()
    v = res_base['membrane.V']

    single_t, single_v = get_single_ap(t, v)

    ax_ap.plot(single_t, single_v, c=(1, .2, .2), linestyle='--', label='Baseline')

    #GET BIOMARKERS
    baseline_biomarkers = get_biomarkers(t, v)

    cmap = matplotlib.cm.get_cmap('viridis')
    all_cols = [cmap(i*22) for i in range(0, len(all_leaks))]

    all_biomarkers = []

    for num, leak in enumerate(all_leaks):
        print(leak)
        mod, p, x = myokit.load(leak_model)
        mod['membrane']['gLeak'].set_rhs(leak)
        if 'kernik' in leak_model:
            mod['geom']['Cm'].set_rhs(98.7)

        s_base = myokit.Simulation(mod)
        s_base.pre(15000)

        res_base = s_base.run(10000, log_times=np.arange(0, 10000, 1))

        t = res_base.time()
        v = res_base['membrane.V']

        t = np.asarray(t) - 6000
        arg_t = np.argmin(np.abs(t))

        single_t, single_v = get_single_ap(np.array(res_base.time()),
                                           np.array(res_base['membrane.V']))

        ax_ap.plot(single_t, single_v, c=all_cols[num])

        trace = [np.asarray(res_base.time()), np.asarray(res_base['membrane.V'])]
        biomarkers = get_biomarkers(t, v)

        all_biomarkers.append(biomarkers)

    ax_ap.set_xlim(-100, 1800)

    ax_ap.spines['top'].set_visible(False)
    ax_ap.spines['right'].set_visible(False)
    ax_ap.set_ylabel('Voltage (mV)')
    if 'paci' in base_model:
        ax_ap.set_xlabel('Time (ms)')

    biom_names = ['MP (mV)', r'$dV/dt_{max}$ (V/s)', r'$APD_{90}$ (ms)', 'CL (ms)']
    all_biomarkers = np.array(all_biomarkers)
    lks = 1 / all_leaks

    mdp_range = [-80, -35]
    cl_range = [200, 1900]
    dvdt_range = [-2, 30]
    apd90_range = [200, 540]

    biomarker_ranges = [mdp_range, dvdt_range, apd90_range, cl_range]

    for i, biom_name in enumerate(biom_names):
        curr_biom = all_biomarkers[:, i]
        axs_biom[i].plot(1/lks, curr_biom, c='k', alpha=.8)
        axs_biom[i].axhline(baseline_biomarkers[i],
                color=(1, .2, .2), linestyle='--')
        [axs_biom[i].scatter(1/lks[j], curr_biom[j], color=all_cols[j])
                                            for j in range(0, len(lks))]

        axs_biom[i].spines['top'].set_visible(False)
        axs_biom[i].spines['right'].set_visible(False)
        axs_biom[i].set_xscale('log')
        if 'kernik' in base_model:
            axs_biom[i].scatter(1/lks[0:2], curr_biom[0:2], facecolor='none', edgecolor='grey', s=40, marker='s')
            axs_biom[i].set_ylabel(biom_name)
        if i == 3:
            axs_biom[i].set_xlabel(r'$g_{seal}\ (nS)$')

        #axs_biom[i].xticks(np.linspace(.1, 1, .1))
        labs = [.1]
        labs += [None for i in range(2, 10)]
        labs += [1]
        axs_biom[i].set_xticks(np.linspace(.1, 1, 10))
        axs_biom[i].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axs_biom[i].set_xticklabels(labs)
        axs_biom[i].set_ylim(biomarker_ranges[i][0], biomarker_ranges[i][1])
        axs_biom[i].set_xlim(0.08, 1.2)


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

    return mdp, dvdt, np.average(apd90), dt


def main():
    plot_lin_leak()


if __name__ == '__main__':
    main()
