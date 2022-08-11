#
# Data loading methods
#
import myokit
import numpy as np

_data_dir = 'data'

# Cell names
_cnames = {
    'beat1': '6_033121_2_alex_control',
    'beat2': '6_040821_2_alex_quinidine',
    'beat3': '6_040921_1_alex_quinidine',
    'beat4': '7_042121_1_alex_quinine',
    'beat5': '7_042621_6_alex_quinine',
    'beat6': '7_042721_4_alex_quinine',
    'beat7': '7_042921_1_alex_quinine',
    'beat8': '7_042921_2_alex_quinine',
    'beat9': '7_042021_2_alex_quinine',
    'flat1': '6_033121_1_alex_control',
    'flat2': '6_033121_3_alex_control',
    'flat3': '7_042121_3_alex_quinine',
}

# Cell parameters
# Cm (pF), Ra, Rm (MOhm)
cell_parameters = {
    'beat1': (42, 42, 34000),
    'beat2': (92, 20, 453),
    'beat3': (43, 29, 740),
    'beat4': (48, 19, 513),
    'beat5': (50, 37, 1900),
    'beat6': (65, 25, 2800),
    'beat7': (44, 30, 636),
    'beat8': (55, 37, 730),
    'beat9': (70, 16, 232),
    'flat1': (13, 29, 979),
    'flat2': (29, 37, 549),
    'flat3': (13, 39, 970),
}


# Protocol names
_pnames = {
    'B': 'proto_B',
    'O': 'proto_O',
    'vcp_opt': 'vcp_opt',
}


def data_sets(include_synth=True):
    """ Returns a list of available data sets. """
    names = list(_cnames.keys())
    if include_synth:
        names = ['synth1'] + names
    return names


def load_named(dname, pname=None, model=None, parameters=None, voltage=False,
               normalised=True):
    """
    Generates or loads a named data set.

    A valid name is either:

    - a data set name ``dname`` and a protocol name ``pname``,
    - a data set name ``dname`` and a ``model`` with a set of ``parameters``,

    where ``dname`` is one of ``['synth1', 'beat1', 'beat2', ...]`` (for a full
    list see :meth:`dnames()`.
    """
    if dname == 'synth1':
        return fake(model, parameters, sigma=0.1, seed=1)

    try:
        cname = _cnames[dname]
    except KeyError:
        raise ValueError(f'Unknown data set {dname}')
    try:
        pname = _pnames[pname]
    except KeyError:
        raise ValueError(f'Unknown protocol {pname}')

    data = load(f'{_data_dir}/{cname}/Pre-drug_{pname}.zip', voltage=voltage)
    if not normalised:
        cell_params = cell_parameters[dname]
        data = list(data)
        data[1] *= cell_params[0]
        data = tuple(data)

    return data


def load(path, voltage=False):
    """
    Loads a zipped DataLog and returns a tuple ``(t, c)`` where t ``t`` is time
    in ms and ``c`` is current in A/F.

    If ``voltage=True`` the method will try to return a third entry ``v`` with
    voltage in mV.
    """
    data = myokit.DataLog.load(path).npview()
    if voltage:
        return data.time(), data['current_pApF'], data['voltage_mV']
    return data.time(), data['current_pApF']


def load_spont(dname):
    """
    Loads a zipped DataLog and returns a tuple ``(t, v)`` where t ``t`` is time
    in ms and ``v`` is voltage in mV.
    """
    try:
        cname = _cnames[dname]
    except KeyError:
        raise ValueError(f'Unknown data set {dname}')

    path = f'{_data_dir}/{cname}/Pre-drug_spont.zip'
    data = myokit.DataLog.load(path).npview()
    return data.time(), data['voltage_mV']


def fake(model, parameters, sigma, seed=None):
    """
    Generates synthetic data by running simulations with the given ``model``
    and ``parameters`` and adding noise with ``sigma``.

    If a ``seed`` is passed in a new random generator will be created with this
    seed, and used to generate the added noise.
    """
    t = model.times()
    #v = model.voltage()
    c = model.simulate(parameters)

    sigma = 0.1
    if seed is not None:
        # Create new random generator, leave the shared one unaltered.
        r = np.random.default_rng(seed=seed)
        c += r.normal(scale=sigma, size=c.shape)
    else:
        c += np.random.normal(scale=sigma, size=c.shape)

    return t, c


def cm():
    """ Returns a vector of Cm values for all cells. """
    return np.array([x[0] for x in cell_parameters.values()])


def ra():
    """ Returns a vector of Ra values for all cells. """
    return np.array([x[1] for x in cell_parameters.values()])


def rm():
    """ Returns a vector of Rm values for all cells. """
    return np.array([x[2] for x in cell_parameters.values()])

