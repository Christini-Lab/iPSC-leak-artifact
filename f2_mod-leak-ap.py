import myokit
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

from methods import post, plots
import methods.data
import methods.plots
import methods.post

from utility_classes import get_single_ap



def lin_leak_mod_biomarkers():
    fig = plt.figure(figsize=(10, 8))
    fig.subplots_adjust(.07, .10, .95, .98)

    grid = fig.add_gridspec(2, 1, hspace=.15, wspace=0.2)

    indices = [0, 3, 4, 7, 10, 11, 14, 15, 17]
    norm = np.array([500, 40, 110, 155, 35, 15, 5, 30, 10, -50, -55, -40, -30, 25, 10])

    all_axs = []
    all_subgrids = []

    fs = 16

    all_leaks = np.linspace(.1, 1.5, 15)
    #all_leaks = np.linspace(.1, 1.5, 2)

    all_biomarkers = {0: [], 1:[]}

    for i, mfile in enumerate(['./mmt/kernik_2019_mc.mmt', './mmt/paci-2013-ventricular.mmt']):
        kernik_base, p, x = myokit.load(mfile)
        s_base = myokit.Simulation(kernik_base)
        s_base.pre(35000)

        res_base = s_base.run(10000)

        subgrid = grid[i, 0].subgridspec(3, 6, wspace=0.7, hspace=.4)
        all_subgrids.append(subgrid)

        ax = fig.add_subplot(subgrid[0:3, 0:3])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if i == 1:
            ax.set_xlabel('Time (ms)', fontsize=fs)

        ax.set_ylabel('Voltage (mV)', fontsize=fs)
        all_axs.append(ax)
        t = res_base.time()
        v = res_base['membrane.V']

        single_t, single_v = get_single_ap(t, v)

        ax.plot(single_t, single_v, c=(1, .2, .2), linestyle='--', label='Baseline')

        #GET BIOMARKERS
        trace = [np.asarray(res_base.time()), np.asarray(res_base['membrane.V'])]

        biomarkers = post.biomarkers(*trace, denoise=False)

        biomarkers = biomarkers.mean(1)
        biomarkers = np.append(biomarkers, 0)

        all_biomarkers[i].append(biomarkers)


    cmap = matplotlib.cm.get_cmap('viridis')
    #cmap = matplotlib.cm.get_cmap('cividis')

    all_cols = [(1, .2, .2)]


    for num, leak in enumerate(all_leaks):
        print(leak)
        for i, modname in enumerate(['./mmt/kernik_leak.mmt', './mmt/paci-2013-ventricular-leak.mmt']):
            mod, p, x = myokit.load(modname)
            mod['membrane']['gLeak'].set_rhs(leak)
            if 'kernik' in modname:
                mod['geom']['Cm'].set_rhs(98.7)
                #mod['ifunny']['g_f'].set_rhs(mod['ifunny']['g_f'].value()*2)
                #mod['ik1']['g_K1'].set_rhs(mod['ik1']['g_K1'].value()*.1)
                #mod['ibna']['g_b_Na'].set_rhs(mod['ibna']['g_b_Na'].value()*.1)
                #mod['ibca']['g_b_Ca'].set_rhs(mod['ibca']['g_b_Ca'].value()*.1)
                #mod['ical']['g_scale'].set_rhs(mod['ical']['g_scale'].value()*.3)
                #mod['ina']['g_scale'].set_rhs(mod['ina']['g_scale'].value()*4)

            s_base = myokit.Simulation(mod)
            s_base.pre(15000)

            cols = cmap(num*14)
            if i == 0:
                if ((leak >.96) and (leak < 1.02)):
                    all_cols.append('k')
                else:
                    all_cols.append(cols)

            res_base = s_base.run(10000)

            t = res_base.time()
            v = res_base['membrane.V']

            t = np.asarray(t) - 6000
            arg_t = np.argmin(np.abs(t))

            ax = all_axs[i]

            single_t, single_v = get_single_ap(np.array(res_base.time()),
                                               np.array(res_base['membrane.V']))

            if ((leak >.96) and (leak < 1.02)):
                ax.plot(single_t, single_v, c='k', linestyle='--', label=r'1 G$\Omega$')
                ax.legend()
            else:
                ax.plot(single_t, single_v, c=cols)

            trace = [np.asarray(res_base.time()), np.asarray(res_base['membrane.V'])]

            biomarkers = post.biomarkers(*trace, denoise=False)

            biomarkers = biomarkers.mean(1)
            biomarkers = np.append(biomarkers, 1/leak)

            all_biomarkers[i].append(biomarkers)


    all_axs[0].set_xlim(-100, 500)
    all_axs[1].set_xlim(-100, 800)

    biomarker_names = np.array(methods.plots.BIOM_NAMES)
    biomarker_names = np.append(biomarker_names, r'$R_{leak}$')[indices]
    biom_units = ['ms', 'ms', 'ms', 'mV', 'mV', 'mV', 'V/s', 'ms', r'G$\Omega$']

    biomarker_names = [f'{n} ({biom_units[i]})' for i, n in enumerate(biomarker_names)]


    patterns = ['-' for i in range(0, (len(all_cols)-1))]
    patterns = [''] + patterns

    for k, v in all_biomarkers.items():
        subgrid = all_subgrids[k]
        mod_biomarkers = np.array(v)
        for i in range(0, 9):
            biom_name = biomarker_names[i]
            ax = fig.add_subplot(subgrid[int(i/3), 3+i%3])
            curr_bioms = mod_biomarkers[:, indices[i]]

            bars = ax.bar(list(range(0, len(curr_bioms))), curr_bioms, color=all_cols)
            bars[0].set(hatch='-', edgecolor='white')
            bars[10].set(hatch='-', edgecolor='white')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xticklabels([])
            ax.set_xlabel(biom_name, labelpad=-5)
            ax.tick_params(bottom=False)

    matplotlib.rcParams['pdf.fonttype'] = 42
    plt.savefig('./figure-pdfs/f2-leak-model-effects.pdf', transparent=True)

    plt.show()


def main():
    lin_leak_mod_biomarkers()


if __name__ == '__main__':
    main()
