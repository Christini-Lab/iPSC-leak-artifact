import myokit
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['lines.linewidth'] = .9
plt.rcParams['lines.markersize'] = 4
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rc('legend', fontsize = 8)


def plot_figure():
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 3))

    fig.subplots_adjust(.12, .15, .99, .99, wspace=.2)

    mk_proto = myokit.Protocol()
    mk_proto.add_step(-90, 10000)

    all_voltages = [v for v in range(-90, 70, 10)]

    for i in all_voltages:
        mk_proto.add_step(i+.1, 10000)

    starts = np.array([int(19.9*10000 + i*100000) for i in
                                    range(0, len(all_voltages))])
    ends = starts + 50 

    conds = {'membrane.gLeak': .2}
    t, dat_base_leak = get_mod_response(
                            conds, mk_proto, f_name='./mmt/kernik_leak.mmt')

    conds = {'ibna.g_b_Na': 6.6189158206846255,
             'ibca.g_b_Ca': 0.24933241003255358,
             'inak.g_scale': 0.8430817227634139}
    t, dat_fit = get_mod_response(
                        conds, mk_proto, f_name='./mmt/kernik_2019_mc.mmt')

    # ax[0]
    #INa
    i_base_leak_na = np.asarray(dat_base_leak['ibna.i_b_Na'])
    i_fit_na = np.asarray(dat_fit['ibna.i_b_Na'])

    #ICaL
    i_base_leak_ca = np.asarray(dat_base_leak['ibca.i_b_Ca'])
    i_fit_ca = np.asarray(dat_fit['ibca.i_b_Ca'])

    #ILeak
    i_base_leak_lk = np.asarray(dat_base_leak['membrane.ILeak'])

    curr_names = ['i_base_leak_na', 'i_base_leak_ca', 'i_base_leak_lk',
                                                    'i_fit_na', 'i_fit_ca']
    
    all_currs = [i_base_leak_na, i_base_leak_ca, i_base_leak_lk,
                                                i_fit_na, i_fit_ca]

    iv_dat = {} 
    for i, curr in enumerate(all_currs):
        curr_steps = [np.average(curr[starts[i]:ends[i]])
                                for i in range(0, len(all_voltages))]
        iv_dat[curr_names[i]] = np.array(curr_steps)
    
    i = 0

    curr_names = [r'Base: $I_{bNa}$', r'Base: $I_{bCa}$', r'Base: $I_{leak}$',
                                        r'Fit: $I_{bNa}$', r'Fit: $I_{bCa}$']

    linestyles = ['-', '-', '-', '--', '--']
    markers = ['^', 's', 'o', '^', 's']
    colors = ['k', 'k', 'k', 'tomato', 'tomato']

    for curr_name, curr_vals in iv_dat.items():
        axs[0].plot(all_voltages, curr_vals,
                    marker=markers[i], label=curr_names[i],
                    color=colors[i], linestyle=linestyles[i])
        i += 1

    axs[0].legend()
    # ax[1]
    i_base_bg = iv_dat['i_base_leak_na'] + iv_dat['i_base_leak_ca']
    i_base_leak_tot = i_base_bg + iv_dat['i_base_leak_lk']
    i_fit_bg = iv_dat['i_fit_na'] + iv_dat['i_fit_ca']

    curr_names = [r'Base: $I_{bNa}+I_{bCa}+I_{lk}$', r'Fit: $I_{bNa} + I_{bCa}$',
            r'Base: $I_{bNa}+I_{bCa}$']

    styles = ['-', '--', 'dotted']
    colors = ['k', 'tomato', 'grey']

    for i, curr_vals in enumerate([i_base_leak_tot, i_fit_bg, i_base_bg]):
        axs[1].plot(all_voltages, curr_vals, marker='o', label=curr_names[i], 
                linestyle=styles[i], color=colors[i])

    axs[1].legend()

    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Voltage (mV)')
        ax.set_ylabel('Current (A/F)')

    plt.show()
    import pdb
    pdb.set_trace()


def plot_bg_iv_leak(ax):
    mk_proto = myokit.Protocol()
    mk_proto.add_step(-90, 10000)

    all_voltages = [v for v in range(-90, 70, 10)]

    for i in all_voltages:
        mk_proto.add_step(i+.1, 10000)

    conds = {'membrane.gLeak': .2}
    t, dat = get_mod_response(conds, mk_proto, f_name='./mmt/kernik_leak.mmt')

    starts = np.array([int(19.9*10000 + i*100000) for i in
                                    range(0, len(all_voltages))])
    ends = starts + 50 

    i_b_Na = np.asarray(dat['ibna.i_b_Na'])
    i_b_Ca = np.asarray(dat['ibca.i_b_Ca'])
    bg_tot = i_b_Na + i_b_Ca
    i_lk = np.asarray(dat['membrane.ILeak'])
    i_all = i_lk + bg_tot

    #curr_names = [r'$I_{bNa}$', r'$I_{bCa}$', r'$I_{tot(bg)}$', r'$I_{leak}$', r'$I_{leak}+I_{bg}$']
    curr_names = [r'$I_{bNa}$', r'$I_{bCa}$', r'$I_{tot(bg)}$', r'$I_{leak}$', r'$I_{leak}+I_{bg}$']

    for j, curr in enumerate([i_b_Na, i_b_Ca, bg_tot, i_lk, i_all]):
        all_currs = [np.average(curr[starts[i]:ends[i]])
                                for i in range(0, len(all_voltages))]
        ax.plot(all_voltages, all_currs, marker='o', label=curr_names[j])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel('Voltage (mV)')
    ax.set_ylabel('Current (A/F)')

    ax.legend()


def plot_bg_iv_fit(ax):
    mk_proto = myokit.Protocol()
    mk_proto.add_step(-90, 10000)

    all_voltages = [v for v in range(-90, 70, 10)]

    for i in all_voltages:
        mk_proto.add_step(i+.1, 10000)

    conds = {'ibna.g_b_Na': 6.6189158206846255,
             'ibca.g_b_Ca': 0.24933241003255358,
             'inak.g_scale': 0.8430817227634139}
    t, dat = get_mod_response(conds, mk_proto, f_name='./mmt/kernik_2019_mc.mmt')

    starts = np.array([int(19.9*10000 + i*100000) for i in
                                    range(0, len(all_voltages))])
    ends = starts + 50 

    i_b_Na = np.asarray(dat['ibna.i_b_Na'])
    i_b_Ca = np.asarray(dat['ibca.i_b_Ca'])
    bg_tot = i_b_Na + i_b_Ca

    curr_names = [r'$I_{bNa}$', r'$I_{bCa}$', r'$I_{tot(bg)}$']

    for j, curr in enumerate([i_b_Na, i_b_Ca, bg_tot]):
        all_currs = [np.average(curr[starts[i]:ends[i]])
                                for i in range(0, len(all_voltages))]
        ax.plot(all_voltages, all_currs, marker='o', label=curr_names[j])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel('Voltage (mV)')
    ax.set_ylabel('Current (A/F)')

    ax.legend()


def get_mod_response(conductances, vc_proto, f_name):
    
    mod = myokit.load_model(f_name)

    for cond, g in conductances.items():
        group, param = cond.split('.')
        val = mod[group][param].value()
        mod[group][param].set_rhs(val*g)

    p = mod.get('engine.pace')
    p.set_binding(None)

    v = mod.get('membrane.V')
    v.demote()
    v.set_rhs(0)
    v.set_binding('pace')

    t_max = vc_proto.characteristic_time()

    sim = myokit.Simulation(mod, vc_proto)

    times = np.arange(0, t_max, 0.1)

    #sim.set_tolerance(1E-8, 1E-8)

    dat = sim.run(t_max, log_times=times)

    return times, dat



plot_figure()
