""" Plots of Diode behavior """
import matplotlib.pyplot as plt
from diode_equations import  calc_rd_deriv_a

def plot_vca_cca(v_ca_a, i_c_a , c_ca, model, plot_dir='/home/matt/Nextcloud/Studium/HauptseminarMikroNanoelektronik/Bericht/figures'):
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
    label_ic = ax1.plot(v_ca_a, i_c_a , '.', label='I_C')
    ax1.set_ylabel('I_C [A]')

    # Prepare Labels of ax1
    labelnames =  label_ic
    labels = [l.get_label() for l in labelnames]
    ax1.legend(labelnames, labels, loc='best')  # TODO Get rid of 'Line2D()'

    # Plot C_CA over V_CA
    ax2 = ax1.twinx()
    label_cca = ax2.plot(v_ca_a, c_ca, 'rx', label='C_CA')
    lab_cca_model = ax2.plot(v_ca_a, c_ca_model, 'g-',
                             label=model.label_cca_model)
    ax2.set_ylabel('C_CA [F]')

    # Prepare Labels of ax2
    labelnames =  label_cca + lab_cca_model
    labels = [l.get_label() for l in labelnames]
    ax2.legend(labelnames, labels, loc='lower right')  # TODO Get rid of 'Line2D()'

    fname_pdf = plot_dir + '/VCA_CCA_T' + "{:.1f}".format(model.T) + '.png'
    plt.savefig(fname_pdf)
    plt.clf()


def plot_vca_ic(v_ca_a, i_c_a , model, plot_dir='/home/matt/Nextcloud/Studium/HauptseminarMikroNanoelektronik/Bericht/figures'):
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
    label_ic = ax1.semilogy(v_ca_a, i_c_a , '.', label='I_C')
    label_ic_model = ax1.semilogy(v_ca_a, i_c_ideal_diode_model_a, '-b',
                                  label=model.label_ideal_diode_model)
    label_ic_r_model = ax1.semilogy(v_ca_a, i_c_r_a , '--r',
                                    label=model.label_diode_ohmic)
    ax1.set_xlabel('V_CA [V]')
    ax1.set_ylabel('I_C [A]')

    # Prepare I_C Labels of ax1
    labelnames =  label_ic + label_ic_model + label_ic_r_model
    labels = [l.get_label() for l in labelnames]
    ax1.legend(labelnames, labels, loc='best')

    # Plot r_D over V_CA
    axr = ax1.twinx()
    axr.set_ylabel('Resistance [Ohm]')
    axr.set_ylim([0, 10])
    label_r = axr.plot(v_ca_a, r_D_a, 'g-.', label=model.label_r)
    axr.legend(label_r, loc='lower right')  # TODO: avoid 'Line2D()

    fname_pdf = plot_dir + '/VCA_IC_T' + "{:.1f}".format(model.T) + '.png'
    plt.savefig(fname_pdf)
    plt.clf()

def plot_T_is(T_a, i_s_a, model, plot_dir='/home/matt/Nextcloud/Studium/HauptseminarMikroNanoelektronik/Bericht/figures'):
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
    label_is = ax1.semilogy(T_a, i_s_a, 'xr', label='I_S')
    label_is_model = ax1.semilogy(T_model_a, i_s_model_a, '-b',
                                  label='I_S model')
    ax1.set_xlabel('T [K]')
    ax1.set_ylabel('I_S [A]')

    # Prepare I_C Labels of ax1
    labelnames =  label_is + label_is_model
    labels = [l.get_label() for l in labelnames]
    ax1.legend(labelnames, labels, loc='best')

    fname_pdf = plot_dir + '/T_IS.png'
    plt.savefig(fname_pdf)
    plt.clf()
