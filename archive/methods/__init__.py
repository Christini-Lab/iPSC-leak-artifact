#
# Methods module
#

# Get methods directory
import os
import inspect
frame = inspect.currentframe()
DIR_METHOD = os.path.abspath(os.path.dirname(inspect.getfile(frame)))
del(os, inspect, frame)


config = 4


def configure(configuration):
    """
    Apply a selected fitting configuration.
    """
    global config
    config = configuration

    # Currents/conductances (and parameter order) and concentrations
    global required_currents, optional_currents, concentrations
    required_currents = [
        '_Na',
        '_CaL',
        '_Kr',
        '_Ks',
        '_to',
        '_NaCa',
        '_K1',
        '_NaK',
    ]
    optional_currents = [
        '_f',
        '_CaT',
    ]
    concentrations = {
        'Na_i': 10,
        'Na_o': 137,
        'K_i': 130,
        'K_o': 5.4,
        'Ca_i': 1e-5,
        'Ca_o': 2,
    }

    # Results dir
    global results
    results = None

    # Holding time & potential
    global t_hold, v_hold


    if config == 1:
        # Experimental concentrations, Ca clamped, no bg currents
        results = 'results-1'
        t_hold = 0
        v_hold = -80

    elif config == 2:
        # Adding 4s of pre-pacing
        results = 'results-2-pre-4s'
        t_hold = 4000
        v_hold = -80

    elif config == 3:
        # Adding background currents
        results = 'results-3-bg'
        t_hold = 4000
        v_hold = -80
        optional_currents += ['_NaB', '_CaB']

    elif config == 4:
        # Free calcium, fixed background currents, fitting icap
        results = 'results-4-ca'
        t_hold = 4000
        v_hold = -80
        concentrations['Ca_i'] = None
        optional_currents += ['_CaP']

    elif config == 5:
        # Free calcium, fitting background currents and ICaP
        results = 'results-5-ca-bg'
        t_hold = 4000
        v_hold = -80
        concentrations['Ca_i'] = None
        optional_currents += ['_CaP', '_NaB', '_CaB']

    else:
        raise ValueError(f'Unknown configuration {config}.')

    # Create results directory
    if results is not None:
        import os
        if not os.path.isdir(results):
            os.makedirs(results)


def configure_for_plots():
    """
    Set currents etc. for plotting. Must be called before importing ``models``.
    """
    global required_currents, optional_currents
    required_currents = [
        '_Na', '_CaL', '_Kr', '_Ks', '_to', '_NaCa', '_K1', '_NaK']
    optional_currents = [
        '_f', '_CaP', '_NaB', '_CaB', '_CaT']
    concentrations['Ca_i'] = None


# Apply configuration
configure(config)

