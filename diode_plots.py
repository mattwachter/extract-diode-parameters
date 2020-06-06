""" Plots of Diode behavior """
import matplotlib.pyplot as plt
import numpy as np
from diode_equations import  calc_rd_deriv_a



def plot_measurements_overview(meas_dict, plot_dir='/home/matt/Nextcloud/Studium/HauptseminarMikroNanoelektronik/Bericht/figures/plots'):
    """Plot all measured data as an overview

        meas_run(run_name, data) (tuple(string, dict)): Measurement run
        run_name (string): Name of test run, e.g. "T298.0K"
        data[phys_quantity: values] (dict[string: list]):
        Dictionary of diode capacitance ('C_CA'),
        current ('I_C) and voltage ('V_CA') values
    Args:
        measurement_dict (dict): 
            meas_run(run_name, data) (tuple(string, dict)): Measurement run
            run_name (string): Name of test run, e.g. "T298.0K"
            data[phys_quantity: values] (dict[string: list]):
            Dictionary of diode capacitance ('C_CA'),
            current ('I_C) and voltage ('V_CA') values
    """
    # Plot I_C over V_CA for all measurement runs
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('$ V_{CA} $ [V]')
    ax1.set_ylabel('$ I_{C} $ [A]')
    labelnames = []
    for meas in meas_dict.items():
        v_ca_a = meas[1]['V_CA'][:]
        i_c_a  = meas[1]['I_C'][:]
        label_temp = meas[0]
        ax1.semilogy(v_ca_a, i_c_a, '-', label=label_temp)
        labelnames.append(label_temp)
    ax1.legend(labelnames, loc='best')  # TODO Get rid of 'Line2D()'
    fname_plot = plot_dir + '/VCA_IC.png'
    plt.savefig(fname_plot)
    plt.clf()
    
    # Plot V_CA over V_CA for all measurement runs
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('$ V_{CA} $ [V]')
    ax1.set_ylabel('$ C_{CA} $ [F]')
    labelnames = []
    for meas in meas_dict.items():
        v_ca_a = meas[1]['V_CA'][:]
        c_ca_a  =  meas[1]['C_CA'][:]
        label_temp = meas[0]
        ax1.plot(v_ca_a, c_ca_a, '-', label=label_temp)
        labelnames.append(label_temp)
    ax1.legend(labelnames, loc='best')  # TODO Get rid of 'Line2D()'
    fname_plot = plot_dir + '/VCA_CCA.png'
    plt.savefig(fname_plot)
    plt.clf()


def plot_vca_cca(v_ca_a, i_c_a , c_ca, model, plot_dir='/home/matt/Nextcloud/Studium/HauptseminarMikroNanoelektronik/Bericht/figures/plots'):
    """Plot diode capacitance over diode voltage

    Args:
        v_ca_a (float array): Cathode-Anode voltage.
        i_c_a  (float array): Diode current.
        c_ca_a (float array): Diode capacitance.
        T (float): Temperature in Kelvin, defaults to 298.0
        plot_dir (string): Optional: Folder in which plots are saved.
    """
    i_c_r_a  = model.calc_ic_diode_ohmic_a(v_ca_a)
    c_ca_model = model.calc_c_diode_a(i_c_r_a )

    # Plot IC over V_CA
    fig, ax1 = plt.subplots()
    label_ic = ax1.plot(v_ca_a, i_c_a , '.', label='$I_C$')
    ax1.set_ylabel('$ I_C $ [A]')
    ax1.set_xlabel('$ V_{CA} $ [V]')

    # Prepare Labels of ax1
    labelnames =  label_ic
    labels = [l.get_label() for l in labelnames]
    ax1.legend(labelnames, labels, loc='best')  # TODO Get rid of 'Line2D()'

    # Plot C_CA over V_CA
    ax2 = ax1.twinx()
    label_cca = ax2.plot(v_ca_a, c_ca, 'rx', label='$C_{CA}$')
    lab_cca_model = ax2.plot(v_ca_a, c_ca_model, 'g-',
                             label=model.label_cca_model)
    ax2.set_ylabel('$ C_{CA} $ [F]')

    # Prepare Labels of ax2
    labelnames =  label_cca + lab_cca_model
    labels = [l.get_label() for l in labelnames]
    ax2.legend(labelnames, labels, loc='lower right')  # TODO Get rid of 'Line2D()'

    fname_plot = plot_dir + '/VCA_CCA_T' + "{:.0f}".format(model.T) + '.png'
    plt.savefig(fname_plot)
    plt.clf()


def plot_vca_ic(v_ca_a, i_c_a , model, plot_dir='/home/matt/Nextcloud/Studium/HauptseminarMikroNanoelektronik/Bericht/figures/plots'):
    """Plot diode current over diode voltage

    Args:
        v_ca_a (float array): Cathode-Anode voltage.
        i_c_a  (float array): Diode current.
        T (float): Temperature in Kelvin, defaults to 298.0
        plot_dir (string): Optional: Folder in which plots are saved.
    """
    # Calculate the model data
    i_c_ideal_diode_model_a = model.calc_ic_ideal_diode_a(v_ca_a)
    r_D_a = calc_rd_deriv_a(v_ca_a, i_c_a )
    i_c_r_a  = model.calc_ic_diode_ohmic_a(v_ca_a)

    # Plot I_C and models over V_CA
    fig, ax1 = plt.subplots()
    label_ic = ax1.semilogy(v_ca_a, i_c_a , '.', label='$I_C$')
    label_ic_model = ax1.semilogy(v_ca_a, i_c_ideal_diode_model_a, '-b',
                                  label=model.label_ideal_diode_model)
    label_ic_r_model = ax1.semilogy(v_ca_a, i_c_r_a , '--r',
                                    label=model.label_diode_ohmic)
    ax1.set_xlabel('$ V_{CA} $ [V]')
    ax1.set_ylabel('$ I_C $  [A]')

    # Prepare I_C Labels of ax1
    labelnames =  label_ic + label_ic_model + label_ic_r_model
    labels = [l.get_label() for l in labelnames]
    ax1.legend(labelnames, labels, loc='best')

    # Plot r_D over V_CA
    axr = ax1.twinx()
    axr.set_ylabel('$R$ [$\Omega$]')
    axr.set_ylim([0, 10])
    label_r = axr.plot(v_ca_a, r_D_a, 'g-.', label=model.label_r)
    axr.legend(label_r, loc='lower right')  # TODO: avoid 'Line2D()
    
    # Mark sections that form the basis of the i_c_r model
    plt.axvspan(model.vca_lim_lower_ic, model.vca_lim_upper_ic, color='b', alpha=0.3)
    plt.axvspan(model.vca_lim_lower_r, model.vca_lim_upper_r, color='g', alpha=0.3)

    title = 'T = ' + str(model.T) + 'K'
    fig.suptitle(title, fontsize=12)
    
    fname_plot = plot_dir + '/VCA_IC_T' + "{:.0f}".format(model.T) + '.png'
    plt.savefig(fname_plot)
    plt.clf()


def plot_vca_cca_for_presentation(v_ca_a, i_c_a , c_ca, model, plot_dir='/home/matt/Nextcloud/Studium/HauptseminarMikroNanoelektronik/Bericht/figures/plots'):
    """Plot diode capacitance over diode voltage

    Args:
        v_ca_a (float array): Cathode-Anode voltage.
        i_c_a  (float array): Diode current.
        c_ca_a (float array): Diode capacitance.
        T (float): Temperature in Kelvin, defaults to 298.0
        plot_dir (string): Optional: Folder in which plots are saved.
    """
    i_c_r_a  = model.calc_ic_diode_ohmic_a(v_ca_a)
    c_ca_model = model.calc_c_diode_a(i_c_r_a )

    # Plot IC over V_CA
    fig, ax1 = plt.subplots()
    label_ic = ax1.plot(v_ca_a, i_c_a , '.', label='$I_C$')
    ax1.set_ylabel('$ I_C $ [A]')
    ax1.set_xlabel('$ V_{CA} $ [V]')

    # Prepare Labels of ax1
    labelnames =  label_ic
    labels = [l.get_label() for l in labelnames]
    ax1.legend(labelnames, labels, loc='best')  # TODO Get rid of 'Line2D()'

    # Plot C_CA over V_CA
    ax2 = ax1.twinx()
    label_cca = ax2.plot(v_ca_a, c_ca, 'rx', label='$C_CA$ (measured)')
    lab_cca_model = ax2.plot(v_ca_a, c_ca_model, 'g-',
                             label='Curve Fitted Model $C_d$')
    ax2.set_ylabel('$ C_{CA} $ [F]')

    # Prepare Labels of ax2
    labelnames =  label_cca + lab_cca_model
    labels = [l.get_label() for l in labelnames]
    ax2.legend(labelnames, labels, loc='lower right')  # TODO Get rid of 'Line2D()'

    fname_plot = plot_dir + '/VCA_CCA_ T' + "{:.0f}".format(model.T) + '.png'
    plt.savefig(fname_plot)
    plt.clf()


def plot_vca_ic_ideal(v_ca_a, i_c_a , model, plot_dir='/home/matt/Nextcloud/Studium/HauptseminarMikroNanoelektronik/Bericht/figures/plots'):
    """Plot diode current over diode voltage

    Args:
        v_ca_a (float array): Cathode-Anode voltage.
        i_c_a  (float array): Diode current.
        model (DiodeModelIsotherm):
        plot_dir (string): Optional: Folder in which plots are saved.
    """
    log_vector = np.vectorize(np.log)
    i_c_a_log = log_vector(i_c_a)
    
    # Calculate the model data
    v_0_to_1 = np.linspace(0, 1.0, 201)
    i_c_ideal_diode_model_a = model.calc_ic_ideal_diode_a(v_0_to_1)
    i_c_ideal_diode_model_a_log = log_vector(i_c_ideal_diode_model_a)

    # Plot I_C and models over V_CA
    fig, ax1 = plt.subplots()
    label_ic = ax1.plot(v_ca_a, i_c_a_log , '.', label='Data')
    label_ic_model = ax1.plot(v_0_to_1, i_c_ideal_diode_model_a_log, '-r',
                                  label='Model')
    ax1.set_xlabel('$ U_{D} $ [V]')
    ax1.set_ylabel('ln($ I_D $)  [A]')
    title = 'T = ' + str(model.T) + 'K'
    fig.suptitle(title, fontsize=12)

    # Prepare I_C Labels of ax1
    labelnames =  label_ic + label_ic_model
    labels = [l.get_label() for l in labelnames]
    ax1.legend(labelnames, labels, loc='best')

    fname_plot = plot_dir + '/VCA_IC_ideal_T' + "{:.0f}".format(model.T) + '.png'
    plt.savefig(fname_plot)
    plt.clf()



def plot_vca_ic_r(v_ca_a, i_c_a , model, plot_dir='/home/matt/Nextcloud/Studium/HauptseminarMikroNanoelektronik/Bericht/figures/plots'):
    """Plot diode current over diode voltage

    Args:
        v_ca_a (float array): Cathode-Anode voltage.
        i_c_a  (float array): Diode current.
        T (float): Temperature in Kelvin, defaults to 298.0
        plot_dir (string): Optional: Folder in which plots are saved.
    """
    # Calculate the model data
    r_D_a = calc_rd_deriv_a(v_ca_a, i_c_a )

    # Plot I_C and models over V_CA
    fig, ax1 = plt.subplots()
    label_ic = ax1.plot(v_ca_a, i_c_a , '.', label='$I_D (measured)$')
    ax1.set_xlabel('$ U_{D} $ [V]')
    ax1.set_ylabel('$ I_D $  [A]')

    # Prepare I_C Labels of ax1
    labelnames =  label_ic
    labels = [l.get_label() for l in labelnames]
    ax1.legend(labelnames, labels, loc='best')

    # Plot r_D over V_CA
    axr = ax1.twinx()
    axr.set_ylabel('$R$ [$\Omega$]')
    axr.set_ylim([0, 10])
    label_r = axr.plot(v_ca_a, r_D_a, 'g-.', label='$r_S = dU_D/dI_D')
    axr.legend(label_r, loc='lower right')  # TODO: avoid 'Line2D()

    fname_plot = plot_dir + '/VCA_IC_R.png'
    plt.savefig(fname_plot)
    plt.clf()


def plot_T_is(T_a, i_s_a, model, plot_dir='/home/matt/Nextcloud/Studium/HauptseminarMikroNanoelektronik/Bericht/figures/plots'):
    """Plot diode saturation current over temperature

    Args:
        v_ca_a (float array): Cathode-Anode voltage.
        i_c_a  (float array): Diode current.
        T (float): Temperature in Kelvin, defaults to 298.0
        plot_dir (string): Optional: Folder in which plots are saved.
    """
    # Calculate the model data
    T_model_a, i_s_model_a = model.calc_i_s_temp_a()

    # Plot I_C and models over V_CA
    fig, ax1 = plt.subplots()
    label_is = ax1.semilogy(T_a, i_s_a, 'xr', label='$I_S$ (determined separately for each temperatures)')
    label_is_model = ax1.semilogy(T_model_a, i_s_model_a, '-b',
                                  label='$I_S(T)$ model')
    ax1.set_xlabel('T [K]')
    ax1.set_ylabel('$ I_S $ [A]')

    # Prepare I_C Labels of ax1
    labelnames =  label_is + label_is_model
    labels = [l.get_label() for l in labelnames]
    ax1.legend(labelnames, labels, loc='best')

    fname_plot = plot_dir + '/T_IS.png'
    plt.savefig(fname_plot)
    plt.clf()
