#
# Plotting and plotting-related methods
#
import matplotlib.patches
import matplotlib.path
import numpy as np


#
# Normalization for biomarkers
#
# Biomarker     Mean    Median  Chosen
# Period         543       418     500
# Peak-to-AP1     44        37      40
# Peak-to-AP2    110       109     110
# Peak-to-MDP    161       153     155
# D-to-peak       38        30      35
# UP1-to-peak     16        16      15
# UP2-to-peak      6         6       5
# V peak          29        28      30
# V AP1           12        12      10
# V AP2          -49       -49     -50
# MDP            -56       -55     -55
# V D            -40       -40     -40
# V UP1          -28       -28     -30
# V UP2           24        24      25
# dVdt            10         8      10
#
BIOM_NORM = np.array([
    500, 40, 110, 155, 35, 15, 5, 30, 10, -50, -55, -40, -30, 25, 10])

# Biomarker names
BIOM_NAMES = [
    r'$T_{CL}$',
    r'$T_{AP1}$',
    r'$T_{AP2}$',
    r'$T_{PtoMDP}$',
    r'$T_{TOtoPk}$',
    r'$T_{UP1}$',
    r'$T_{UP2}$',
    r'$V_{P}$',
    r'$V_{AP1}$',
    r'$V_{AP2}$',
    r'$V_{MDP}$',
    r'$V_{TO}$',
    r'$V_{UP1}$',
    r'$V_{UP2}$',
    r'$\dot{V}_{max}$',
    r'$APD_{90}$',
    r'$V_{AMP}$'
]

CONTRIB_ORDER = [
    'I_NaCa',
    'I_K1',
    'I_Kr',
    'I_Ks',
    'I_to',
    'I_Kb',
    'I_f',
    'I_Kur',
    'I_NaK',
    'I_Na',
    'I_NaL',
    'I_CaL',
    'I_CaP',
    'I_NaB',
    'I_CaB',
    'I_ClCa',
    'I_CaT',
]


def axletter(axes, letter, offset=-0.05, tweak=0,
             weight='bold', fontsize=14, ha='center'):
    """
    Draw a letter (e.g. "A") near the top left of an axes system.

    Arguments:

    ``axes``
        The axes to label.
    ``letter``
        The letter (or text) to label the axes with.
    ``offset``
        An x offset, specified in figure coordinates. Using figure coordinates
        lets you align labels for different axes.
    ``tweak``
        An optional y coordinate tweak (in figure coordinates).
    ``weight``
        The font weight (default: bold)
    ``fontsize``
        The font size (default: 14)
    ``ha``
        Horizontal alignment (default: center)

    """
    # Get top of axes, in figure coordinates
    trans = axes.transAxes
    x, y = trans.transform((0, 1))
    trans = axes.get_figure().transFigure
    x, y = trans.inverted().transform((x, y))

    font = dict(weight=weight, fontsize=fontsize)
    axes.text(x + offset, y + tweak, letter, font, ha=ha, va='top',
              transform=trans)


def biomarker_polar(biomarkers, norm=BIOM_NORM):
    """
    Takes a 2D array of (unnormalised) biomarkers and returns data ``(t, rs)``
    for a polar plot, where ``t`` is an array of angles, and each row in ``rs``
    is a set of radii representing normalised biomarkers.

    A biomarker array for normalisation can be passed in as ``norm``.
    """
    # Normalise biomarkers
    biomarkers = (np.asarray(biomarkers).T / norm).T

    nb = len(norm)
    r = np.arange(nb) * 2 * np.pi / nb
    r = np.concatenate((r, r[:1]))
    b = np.concatenate((biomarkers, biomarkers[:1]))
    return r, b


def contribution(model, parameters=None):
    """ Return ``(d, variables, labels)`` to make contribution plots. """

    # Run a simulation
    d = model.simulate_full(parameters).npview()

    # Get currents
    currents, i_total = model.current_names()

    todo = list(currents)
    ordered = ['remainder']
    labels = ['Remainder']
    for label in CONTRIB_ORDER:
        for current in todo:
            if current.label() == label:
                ordered.append(current.qname())
                labels.append(label.replace('_', ''))
                todo.remove(current)
                continue

    if todo:
        raise ValueError(
            'Not all currents added: ' + ', '.join([x.qname() for x in todo]))

    # Add remainder current
    ctotal = d[i_total]
    for c in currents:
        ctotal -= d[c]
    if np.sum(np.abs(ctotal) < 1):
        ordered = ordered[1:]
        labels = labels[1:]
    else:
        d['remainder'] = ctotal

    return d, ordered, labels


def fft(signal, dt):
    """
    Returns a discrete fourier transform (DFT) of the given ``signal``, which
    must have been sampled with points equidistantly spaced ``dt`` time units
    apart.

    Real-valued input is assumed, only the right half of the (symmetric) fft is
    returned.

    Returns an tuple ``(x, y)`` where ``x`` is a frequency (in Hz if ``dt`` is
    in seconds) and ``y`` is a (complex-valued) DFT.
    """
    import scipy.fft
    y = scipy.fft.rfft(signal)
    x = scipy.fft.rfftfreq(len(signal), dt)
    return x, y


def protocol_bands(ax, protocol, color='#f0f0f0f0', zorder=0):
    """ Show background bands corresponding to protocol steps. """
    for e in list(protocol)[::2]:
        ax.axvspan(
            e.start(), e.start() + e.duration(), color=color, zorder=zorder)


def robustness(info, parameters):
    """ Returns a tuple ``(e, d, xe, xp)`` for a robustness plot. """

    # info = (run, error, time, iterations, evaluations)
    # Extract errors
    e = info[:, 1]

    # Extract max distance in parameters
    d = np.max(parameters / parameters[0] - 1, axis=1)

    # Count how many scores were within 1% of best
    ib = e / e[0] - 1 < 0.01

    # Keep only errors near best
    eb = e[ib]

    # Select ones that were also close in parameter space
    db = d[ib]
    db = db[db < 0.01]

    # Get percentage with good score
    xe = 100 * len(eb) / len(e)

    # Get percentage with good score and same parameters
    xp = 100 * len(db) / len(d)

    return d, e, xe, xp


def swarmx(y, nbins=None):
    """
    Returns x coordinates for the points in ``y``, so that plotting ``x`` and
    ``y`` results in a bee swarm plot.
    """
    nbins = len(y) // 6

    # Get upper bounds of bins
    y = np.asarray(y)
    x = np.zeros(len(y))
    ylo = np.min(y)
    yhi = np.max(y)
    dy = (yhi - ylo) / nbins
    ybins = np.linspace(ylo + dy, yhi - dy, nbins - 1)

    # Divide indices into bins
    i = np.arange(len(y))
    ibs = [0] * nbins
    ybs = [0] * nbins
    nmax = 0
    for j, ybin in enumerate(ybins):
        f = y <= ybin
        ibs[j], ybs[j] = i[f], y[f]
        nmax = max(nmax, len(ibs[j]))
        f = ~f
        i, y = i[f], y[f]
    ibs[-1], ybs[-1] = i, y
    nmax = max(nmax, len(ibs[-1]))

    # Assign x indices
    dx = 1 / (nmax // 2)
    for i, y in zip(ibs, ybs):
        if len(i) > 1:
            j = len(i) % 2
            i = i[np.argsort(y)]
            a = i[j::2]
            b = i[j + 1::2]
            x[a] = (0.5 + j / 3 + np.arange(len(a))) * dx
            x[b] = (0.5 + j / 3 + np.arange(len(b))) * -dx

    return x


def vconnect(axhi, axlo, xhi, xlo=None,
             facecolor='#f0f0f0', edgecolor='#e0e0e0',
             shadehi=False, shadelo=False):
    """
    Draw region connecting the same times on axes ``axhi`` and ``axlo``, where
    ``axhi`` is situatated above ``axlo``.

    Arguments:

    ``axhi``
        Axes, must be situated above ``axlo``.
    ``axlo``
        Axes, must be situated below ``axhi``.
    ``xhi``
        A tuple ``(x1, x2)`` indicating the region (in data coordinates) to
        connect. By default, the region is drawn from ``(x1, x2)`` on ``axhi``
        to ``(x1, x2)`` on ``axlo``.
    ``xlo``
        An optional tuple indicating alternative points on ``axlo`` to draw the
        region to.
    ``facecolor``, ``edgecolor``
        Face and edge color for the regions
    ``shadehi=False``
        Set to true to shade the relevant area on ``axhi``.
    ``shadelo=False``
        Set to true to shade the relevant area on ``axlo``.

    """
    if xlo is None:
        xlo = xhi

    # Get transform and inverse transform for lower axes
    trans = axlo.transAxes
    itrans = trans.inverted()

    # Get points, in "display" coordinates.
    ahi, _ = axhi.transData.transform((xhi[0], 0))
    bhi, _ = axhi.transData.transform((xhi[1], 0))
    _, yhi = axhi.transAxes.transform((0, 0))
    alo, _ = axlo.transData.transform((xlo[0], 0))
    blo, _ = axlo.transData.transform((xlo[1], 0))

    # Display coordinates can change if the figure size changes, so get them
    # all in axlo.transAxes coordinates
    ahi, yhi = itrans.transform((ahi, yhi))
    bhi, _ = itrans.transform((bhi, 0))
    alo, _ = itrans.transform((alo, 0))
    blo, _ = itrans.transform((blo, 0))
    ylo = 1

    # Create path linking axes
    Path = matplotlib.path.Path
    path_data = [
        (Path.MOVETO, (ahi, yhi)),
        (Path.LINETO, (bhi, yhi)),
        (Path.LINETO, (blo, ylo)),
        (Path.LINETO, (alo, ylo)),
        (Path.CLOSEPOLY, (1, 1)),   # (0, 0) is ignored!
    ]
    codes, verts = zip(*path_data)
    path = matplotlib.path.Path(verts, codes)

    # Draw patch
    patch = matplotlib.patches.PathPatch(
        path, transform=trans, clip_on=False,
        edgecolor=edgecolor, facecolor=facecolor)
    axhi.add_patch(patch)

    # Shade in high graph
    if shadehi:
        axhi.axvspan(xhi[0], xhi[1], color=facecolor)
    if shadelo:
        axlo.axvspan(xhi[0], xhi[1], color=facecolor)
