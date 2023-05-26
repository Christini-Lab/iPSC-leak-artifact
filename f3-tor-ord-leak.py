import myokit
import matplotlib.pyplot as plt
import numpy as np
from utility import get_single_ap
from scipy.signal import find_peaks
import matplotlib



#often 1uF/cm^2: https://www.sas.upenn.edu/LabManuals/BBB251/NIA/NEUROLAB/APPENDIX/MEMBRANE/capcurr.htm

def plot_lin_leak():
    # Array with leak values
    all_leaks = np.linspace(0, 1, 10)
    all_leaks = 1/10**all_leaks
    
    # Make figure grid
    fig = plt.figure(figsize=(6.5, 5))
    fig.subplots_adjust(.1, .1, .95, .95)
    grid = fig.add_gridspec(1, 2, wspace=0.3)#, width_ratios=[1, 3])

    subgrid_aps = grid[0, 0].subgridspec(2, 1, wspace=.3, hspace=.2)
    subgrid_biom = grid[0, 1].subgridspec(3, 2, wspace=.43, hspace=.3)

    # Plot APs and biomarkers for Kernik model (Panels A and C)
    ax_ap = fig.add_subplot(subgrid_aps[0])
    ax_ap.text(-300, 44, 'A', fontsize=12)
    axs_biom = [fig.add_subplot(subgrid_biom[0, 0]),
                fig.add_subplot(subgrid_biom[1, 0]),
                fig.add_subplot(subgrid_biom[2, 0]),]
                #fig.add_subplot(subgrid_biom[3, 0])]
    axs_biom[0].text(.7, -15, 'C', fontsize=12)
    plot_mod_leak(ax_ap, axs_biom, fig, all_leaks, 50, './mmt/tor_ord_endo.mmt', './mmt/tor_ord_endo_fixed_leak.mmt')

    # Plot APs and biomarkers for Paci model (Panels B and D)
    ax_ap = fig.add_subplot(subgrid_aps[1])
    ax_ap.text(-300, 30, 'B', fontsize=12)
    axs_biom = [fig.add_subplot(subgrid_biom[0, 1]),
                fig.add_subplot(subgrid_biom[1, 1]),
                fig.add_subplot(subgrid_biom[2, 1]),]
                #fig.add_subplot(subgrid_biom[3, 1])]
    axs_biom[0].text(.7, -15, 'D', fontsize=12)
    plot_mod_leak(ax_ap, axs_biom, fig, all_leaks, 153, './mmt/tor_ord_endo.mmt', './mmt/tor_ord_endo_fixed_leak.mmt')#'./mmt/paci-2013-ventricular-fixed.mmt', './mmt/paci-2013-ventricular-leak-fixed.mmt')
    ax_ap.set_xlabel('Time (ms)')

    matplotlib.rcParams['pdf.fonttype'] = 42
    #plt.savefig('./figure-pdfs/f-leak-model-effects.pdf', transparent=True)
    # Save the figure
    plt.savefig('./figure-pdfs/f-tor-leak.pdf', transparent=True)

    plt.show()


def plot_mod_leak(ax_ap, axs_biom, fig, all_leaks, Cm, base_model, leak_model):
    ax_ap.set_title(f'ToR-ORd, ' + r'$C_m=$' + f'{Cm}pF', y=.91)
    axs_biom[0].set_title(r'$C_m=$' + f'{Cm}pF', y=.94)

    t_step = .02

    #Cm = 194
    #Cm = 50 

    base, p, x = myokit.load(base_model)
    s_base = myokit.Simulation(base, p)
    #prepace = 500000
    prepace = 5000
    s_base.pre(prepace)
    res_base = s_base.run(10000, log_times=np.arange(0, 10000, t_step))

    t = res_base['engine.time'] 
    v = res_base['membrane.v']

    single_t, single_v = get_single_ap(t, v)

    ax_ap.plot(single_t, single_v, c=(1, .2, .2), linestyle='--', label='Baseline')
    #ax_ap.plot(t, v, c=(1, .2, .2), linestyle='--', label='Baseline')

    #GET BIOMARKERS
    baseline_biomarkers = get_biomarkers(t, v)

    cmap = matplotlib.cm.get_cmap('viridis')
    all_cols = [cmap(i*22) for i in range(0, len(all_leaks))]

    all_biomarkers = []

    for num, leak in enumerate(all_leaks):
        print(leak)
        mod, p, x = myokit.load(leak_model)
        mod['membrane']['gLeak'].set_rhs(leak)
        mod['membrane']['Cm'].set_rhs(Cm)
        
        s_base = myokit.Simulation(mod, p)
        s_base.pre(prepace)

        t_max = 10000
        res_base = s_base.run(t_max, log_times=np.arange(0, t_max, t_step))

        t = res_base.time()
        v = res_base['membrane.v']

        t = np.asarray(t) - 6000
        arg_t = np.argmin(np.abs(t))

        if (np.max(v) - np.min(v)) < 10:
            single_t, single_v = t, v
        else:
            single_t, single_v = get_single_ap(np.array(res_base.time()),
                                               np.array(res_base['membrane.v']))

        ax_ap.plot(single_t, single_v, c=all_cols[num])
        #ax_ap.plot(t, v, c=all_cols[num])

        trace = [np.asarray(res_base.time()), np.asarray(res_base['membrane.v'])]
        biomarkers = get_biomarkers(t, v)

        all_biomarkers.append(biomarkers)

    ax_ap.set_xlim(-100, 800)

    ax_ap.spines['top'].set_visible(False)
    ax_ap.spines['right'].set_visible(False)
    ax_ap.set_ylabel('Voltage (mV)')
    if 'paci' in base_model:
        ax_ap.set_xlabel('Time (ms)')

    biom_names = ['MP (mV)', r'$dV/dt_{max}$ (V/s)', r'$APD_{90}$ (ms)', 'CL (s)']
    all_biomarkers = np.array(all_biomarkers)
    lks = 1 / all_leaks

    mdp_range = [-92, -60]
    cl_range = [.9, 1.1]
    dvdt_range = [0, 325]
    apd90_range = [260, 290]

    biomarker_ranges = [mdp_range, dvdt_range, apd90_range, cl_range]

    for i, biom_name in enumerate(biom_names[:-1]):
        curr_biom = all_biomarkers[:, i]
        if ('kernik' in base_model) and biom_name != 'MP (mV)':
            axs_biom[i].plot(lks[3:], curr_biom[3:], c='k', alpha=.8)
            [axs_biom[i].scatter(lks[j], curr_biom[j], color=all_cols[j])
                                                for j in range(3, len(lks))]
            #axs_biom[i].scatter(lks[0:3], curr_biom[0:3], facecolor='none', edgecolor='grey', s=40, marker='s')
        else:
            axs_biom[i].plot(lks, curr_biom, c='k', alpha=.8)
            [axs_biom[i].scatter(lks[j], curr_biom[j], color=all_cols[j])
                                                for j in range(0, len(lks))]

        axs_biom[i].axhline(baseline_biomarkers[i],
                color=(1, .2, .2), linestyle='--')

        axs_biom[i].spines['top'].set_visible(False)
        axs_biom[i].spines['right'].set_visible(False)
        axs_biom[i].set_xscale('log')

        if Cm == 50:
            axs_biom[i].set_ylabel(biom_name)

        if 'CL' in biom_name:
            axs_biom[i].set_xlabel(r'$R_{seal}\ (G\Omega)$')

        #axs_biom[i].xticks(np.linspace(.1, 1, .1))
        labs = [1]
        labs += [None for i in range(2, 10)]
        labs += [10]
        axs_biom[i].set_xticks(np.linspace(1, 10, 10))
        axs_biom[i].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axs_biom[i].set_xticklabels(labs)
        axs_biom[i].set_ylim(biomarker_ranges[i][0], biomarker_ranges[i][1])
        #axs_biom[i].set_xlim(0.08, 1.2)


def get_biomarkers(t, v):
    t, v = np.asarray(t), np.asarray(v)
    step_size = t[1] - t[0]

    mdp = np.min(v[int(-2000/step_size):])

    start_pts = np.argwhere(np.mod(t-10, 1000)==0).flatten() + int(1/step_size)

    pks = find_peaks(v, distance=200/step_size, height=-40)[0]
    if len(pks) < 2:
        return np.min(v), 0, 0, 0 

    if  (np.max(v) - np.min(v)) < 10:
        return np.min(v), 0, 0, 0

    dt = t[pks[3]] - t[pks[2]]

    dvdts = []
    for pt in start_pts:
        temp_v_range = v[pt:int(pt+10/step_size)]
        dvdts.append(np.max(np.diff(temp_v_range)/step_size))
        
    dvdt = np.average(dvdts)

    #min_v_idxs = find_peaks(-v[start_pts[0]:], height=40, distance=100/step_size)[0]+start_pts[0]
    
    apd90 = []
    for i, dvdt_idx in enumerate(start_pts[:-1]):
        min_idx = dvdt_idx+np.argmin(v[dvdt_idx:start_pts[i+1]])
        v_range = v[dvdt_idx: min_idx]
        t_range = t[dvdt_idx: min_idx]
        amp = np.max(v_range) - np.min(v_range)
        apd90_v = np.max(v_range) - amp*.9
        pk_idx = np.argmax(v_range)

        apd90_idx = np.argmin(np.abs(v_range[pk_idx:] - apd90_v)) + pk_idx
        apd90.append(t_range[apd90_idx] - t_range[0])

    return mdp, dvdt, np.average(apd90), 1



def plot_tor_ord():
    fig, ax = plt.subplots(1, 1, True, figsize=(12, 8))

    for i, f in enumerate(['mmt/tor_ord_endo.mmt', 'mmt/tor_ord_endo_fixed_leak.mmt']):
        mod, proto, x = myokit.load(f)
        sim = myokit.Simulation(mod, proto)
        dat_normal = sim.run(20000)

        t = dat_normal['engine.time']
        v = dat_normal['membrane.v']

        ax.plot(t, v, label=i)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Time (ms)', fontsize=14)
    ax.set_ylabel('Voltage (mV)', fontsize=14)
    plt.legend()
    plt.show()


#plot_tor_ord()
plot_lin_leak()
