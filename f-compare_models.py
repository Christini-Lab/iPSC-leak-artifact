import myokit
import matplotlib.pyplot as plt
import time
import numpy as np
import matplotlib

plt.rcParams['lines.linewidth'] = .9
plt.rcParams['lines.linewidth'] = 1.3 
plt.rcParams['lines.markersize'] = 4
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rc('legend', fontsize = 8)



def plot_figure_all_currs():
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(4, 6.5), gridspec_kw={'height_ratios':[3,1,1,1]})

    fig.subplots_adjust(.17, .10, .95, .95)

    y_labs = ['Voltage (mV)', r'$I_{ion}$', r'$I_{K1}$', r'$I_{f}$']#, r'$I_{NaCa}$', r'$I_{Nab}$', r'$I_{Cab}$']
    #ylims = [[-95, 40], [-.1, 1.5], [-.1, .1]]

    for i, ax in enumerate(axs):
        ax.set_ylabel(y_labs[i]+'(A/F)')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #ax.set_ylim(ylims[i][0], ylims[i][1])
        ax.set_xlim(-5, 8000)
        ax.axhline(0, color='grey', alpha=.2)

    plot_tor_ord_currs(axs)
    plot_kernik_currs(axs)
    #plot_kernik_currs(axs, {'ik1.g_K1': 5}, with_stim=True)
    plot_paci_currs(axs)

    axs[-1].set_xlabel('Time (ms)')

    axs[0].legend()

    matplotlib.rcParams['pdf.fonttype'] = 42
    plt.savefig('./figure-pdfs/f-adult_vs_ipsc_currs.pdf', transparent=True)


    plt.show()


def plot_tor_ord(ax):
    mod, proto, x = myokit.load('mmt/tor_ord_endo.mmt')
    sim = myokit.Simulation(mod, proto)
    t = 1600 
    times = np.arange(0, t, .1)
    dat = sim.run(t, log_times=times)

    t = np.array([t for t in dat['engine.time']])
    v = dat['membrane.v']

    ax.plot(t-10, v, 'k')


def plot_kernik(ax):
    mod, proto, x = myokit.load('mmt/kernik_2019_mc_fixed.mmt')
    
    sim = myokit.Simulation(mod, proto)
    t = 2200 
    times = np.arange(0, t, .1)
    dat = sim.run(t, log_times=times)

    t = np.array([t for t in dat['engine.time']])
    v = dat['membrane.V']

    ax.plot(t[4950:]-505, v[4950:], 'k')


def plot_kernik_cc(ax):
    mod, proto, x = myokit.load('mmt/kernik_2019_mc_fixed.mmt')
    sim = myokit.Simulation(mod, proto)
    t = 10000 
    times = np.arange(0, t, .1)
    dat = sim.run(t, log_times=times)

    t = np.array([t for t in dat['engine.time']])
    v = dat['membrane.V']

    ax.plot(t[4950:]-505, v[4950:], 'k', label='iPSC-CM Model')


def plot_tor_ord_currs(axs):
    mod, proto, x = myokit.load('mmt/tor_ord_endo.mmt')
    sim = myokit.Simulation(mod, proto)
    t = 2200 
    times = np.arange(0, t, .1)
    dat = sim.run(t, log_times=times)

    t = np.array([t for t in dat['engine.time']])
    v = dat['membrane.v']

    axs[0].plot(t-10, v, 'k', label='Tor-ORd Adult')


    for i, curr_i in enumerate(['membrane.i_ion', 'IK1.IK1', 'if']):#, 'INaCa.INaCa_i', 'INab.INab', 'ICab.ICab']):
        if curr_i == 'if':
            continue
        if 'IKb' in curr_i:
            continue
        axs[i+1].plot(t, dat[curr_i], 'k')
        #axs[i+1].fill_between(t, dat[curr_i], color='k', alpha=.5)


def plot_kernik_currs(axs, new_conds={}, with_stim=False):
    mod, proto, x = myokit.load('mmt/kernik_2019_mc_fixed.mmt')

    if with_stim:
        proto.events()[0]._level = 12 
        proto.events()[0]._duration = 1
        proto.events()[0]._period = 1000
        
    #curr_state = mod.state()
    #curr_state[4] = 150
    #mod.set_state(curr_state)

    for k, v in new_conds.items():
        group, param = k.split('.')
        mod[group][param].set_rhs(mod[group][param].value()*v)

    sim = myokit.Simulation(mod, proto)
    t = 10000 
    times = np.arange(0, t, .1)
    dat = sim.run(t, log_times=times)

    t = np.array([t for t in dat['engine.time']])
    v = dat['membrane.V']

    if with_stim:
        offset = 0
    else:
        offset = 501

    axs[0].plot(t[4950:]-offset, v[4950:], 'k--', label='Kernik iPSC-CM')

    for i, curr_i in enumerate(['membrane.i_ion', 'ik1.i_K1', 'ifunny.i_f']):#/, 'inaca.i_NaCa', 'ibna.i_b_Na', 'ibca.i_b_Ca']):
        if curr_i == 'ikb':
            continue
        axs[i+1].plot(t[4950:]-offset, dat[curr_i][4950:], 'k--')
        #axs[i+1].fill_between(t[4950:]-505, dat[curr_i][4950:], color='k', alpha=.5)


def plot_paci_currs(axs):
    mod, proto, x = myokit.load('mmt/paci-2013-ventricular-fixed.mmt')
    sim = myokit.Simulation(mod, proto)
    t = 2200 
    times = np.arange(0, t, .1)
    dat = sim.run(t, log_times=times)

    t = np.array([t for t in dat['engine.time']])
    v = dat['membrane.V']

    offset = 542

    axs[0].plot(t[4950:]-offset, v[4950:], 'k', linestyle='dotted')

    for i, curr_i in enumerate(['membrane.i_ion', 'ik1.IK1', 'if.If']):#, 'inaca.INaCa', 'ibna.IbNa', 'ibca.IbCa', ]):
        if curr_i == 'ikb':
            continue
        axs[i+1].plot(t[4950:]-offset, dat[curr_i][4950:], 'k', linestyle='dotted')
        #axs[i+1].fill_between(t[4950:]-505, dat[curr_i][4950:], color='k', alpha=.5)


#plot_figure()
plot_figure_all_currs()
#plot_figure_all_currs2()

