#! /usr/bin/python3

# pylint: disable=unused-wildcard-import
import json
import numpy as np
from scipy.optimize import curve_fit
import scipy.constants as const
from scipy.special import lambertw
# Ensure custom functions can be found.
import sys
sys.path.append('/home/matt/Nextcloud/Studium/HauptseminarMikroNanoelektronik/PythonExtraction')
# Use diode functions of Python files in working directory
from diode_equations import  *  
from diode_plots import  *
from diode_utils import  *

class DiodeModelIsotherm:
    """
    TODO Define vca_lim variables at class level?
    """
    def __init__(self, v_ca_a, i_c_a , c_ca_a , T):
        self.params = diode_model_params_isotherm(v_ca_a, i_c_a , c_ca_a , T)
        self.i_s = self.params['I_S']
        self.m = self.params['m']
        self.r_s = self.params['R_S']
        self.tt = self.params['TT']
        self.T = self.params['T']
        self.vca_lim_lower_ic = self.params['vca_lim_lower_ic']
        self.vca_lim_upper_ic = self.params['vca_lim_upper_ic']
        self.vca_lim_lower_cca = self.params['vca_lim_lower_cca']
        self.vca_lim_upper_cca = self.params['vca_lim_upper_cca']
        self.vca_lim_lower_r = self.params['vca_lim_lower_r']
        self.vca_lim_upper_r = self.params['vca_lim_upper_r']
        # TODO: fname_pdf (ic, cca) also as class attribute?
        self.label_ideal_diode_model=(
            'I_C_ideal = I_S * (exp (V_CA/(V_T*m)) -1):\n I_S = ' +
            "{:.4g}".format(self.i_s) + ' A, m = ' + "{:.4g}".format(self.m) +
            '\n based on ' + str(self.vca_lim_lower_ic) + 'V <= V_CA <= ' +
            str(self.vca_lim_upper_ic) + 'V')
        self.label_diode_ohmic = ('I_C_model (R_S = ' +
            "{:.4g}".format(self.r_s) + ' $\Omega$)'+ '\n based on ' +
            str(self.vca_lim_lower_r) + 'V <= V_CA <= ' +
            str(self.vca_lim_upper_r) + 'V')
        self.label_cca_model = ('C_CA_model = TT * I_C_model\n TT =' +
            "{:.4g}".format(self.tt) + 's\n based on ' +
            str(self.vca_lim_lower_cca) + 'V <= V_CA <= ' +
            str(self.vca_lim_upper_cca) + 'V')
        # TODO: Makes sense as class attribute?
        self.label_r = 'r_D = d(V_CA)/d(I_C)'

    def calc_ic_ideal_diode_a(self, v_ca_a):
        """Ideal diode current array as a function of a diode voltage.

        TODO More efficient solution than np.vectorize()?
        Args:
            v_ca_a (float array): Cathode-Anode voltage [V].

        Returns:
            (float array): Diode current I_C [A]
        """
        # Define diode equation with fixed model parameters
        def ideal_diode_eq_self(v_ca):
            return ideal_diode_eq(v_ca, self.i_s, self.m, self.T)
        ideal_diode_eq_self_vec = np.vectorize(ideal_diode_eq_self)
        i_c_ideal_diode_a = ideal_diode_eq_self_vec(v_ca_a)
        return i_c_ideal_diode_a


    def calc_ic_diode_ohmic_a(self, v_ca_a):
        """Current array from an ideal diode in series with a resistor.

        TODO More efficient solution than np.vectorize()?
        Args:
            v_ca_a (float array): Cathode-Anode voltage [V].

        Returns:
            (float array): Diode current I_C [A]
        """
        def ic_diode_ohmic_self(v_ca):
            return ic_diode_ohmic(v_ca, self.i_s, self.m, self.T, self.r_s)
        ic_diode_ohmic_self_vec = np.vectorize(ic_diode_ohmic_self)
        i_c_r_a = ic_diode_ohmic_self_vec(v_ca_a)
        return i_c_r_a

    def calc_c_diode_a(self, i_c_r_a):
        """Capacitance array linearly dependent on diode current (model).

        TODO Check input
        Args:
            v_ca_a (float array): Cathode-Anode voltage [V].

        Returns:
            (float array): Diode capacitance C_CA [A]
        """
        def diode_capacitance_TT_eq_self(i_c_r):
            return diode_capacitance_TT_eq(i_c_r, self.tt)
        diode_capacitance_TT_eq_self_vec = np.vectorize(diode_capacitance_TT_eq_self)
        c_ca_model_a = diode_capacitance_TT_eq_self_vec(i_c_r_a)
        return c_ca_model_a


class DiodeModel(DiodeModelIsotherm):
    def __init__(self, v_ca_a, i_c_a , c_ca_a , T, T_i_s_a, i_s_temp_a ,):
        """Diode model with temperature dependence of saturation current.

        TODO Calculation
        Args:
            v_ca_a ([type]): [description]
            i_c_a  ([type]): [description]
            c_ca_a  ([type]): [description]
            T ([type]): [description]
            T_i_s_a (float array): 
                Temperatures [K] at which I_S was estimated
            i_s_temp_a  (float array): 
                Array of estimated I_S at temperatures T_i_s_a
        """
        # Extends __init__() of DiodeModelIsotherm
        DiodeModelIsotherm.__init__(self, v_ca_a, i_c_a , c_ca_a , T)
        # TODO Ensure reasonable results for I_S temperature coefficient.
        # self.i_s_temp_coeff = i_s_temp_dependence_model(T_i_s_a, 
        #                                                 i_s_temp_a)
        # self.params['i_s_temp_coeff'] = self.i_s_temp_coeff
        
    def calc_i_s_temp_a(self, T_lower = 250, T_upper = 450):
        T_a = np.linspace(T_lower, T_upper, num=100)
        i_s_a = np.zeros(len(T_a))
        for i in range(len(i_s_a)):
            i_s_a[i] = diode_saturation_current(self.i_s, T_a[i])
        return (T_a, i_s_a)


def process_diode_measurement(measurements_fname='data.json', 
    results_fname='model.json', plots_dir='figures' ):
    """Extract diode model parameters from JSON file with measurements.

    Structure of imported JSON File:
    Dict:
        meas_run(run_name, data) (tuple(string, dict)): Measurement run
        run_name (string): Name of test run, e.g. "T298.0K"
        data[phys_quantity: values] (dict[string: list]):
        Dictionary of diode capacitance ('C_CA'),
        current ('I_C) and voltage ('V_CA') values
    Args:
        measurements_fname (string): File name of a JSON file

    Returns:
        model_parameters_fname (string): File name of a new JSON file
    """
    with open(measurements_fname, 'r') as f:
        measurements = json.load(f)

    plot_measurements_overview(measurements)
    
    models = []
    # Prepare variables for temperature dependence of I_S
    i_s_temp_a  = np.zeros(len(measurements.items()))   # Saturation currents at
    T_i_s_a  = np.zeros(len(measurements.items()))      # different temperatures
    i = 0
    #TODO Guarantee correct order of dict entries for Python < 3.7
    for measurement in measurements.items():
        v_ca_a = measurement[1]['V_CA'][:]
        i_c_a  = measurement[1]['I_C'][:]
        c_ca_a  =  measurement[1]['C_CA'][:]
        T = float(measurement[0][1:5])     # e.g. measurement[0] = "T298.0K"
        model = DiodeModelIsotherm(v_ca_a, i_c_a , c_ca_a , T)
        plot_vca_ic(v_ca_a, i_c_a , model)
        plot_vca_cca(v_ca_a, i_c_a , c_ca_a , model)
        models.append(model)
        i_s_0 = diode_saturation_current_0(model.i_s, T)
        print('I_S = ', model.i_s, 'I_S0_model =', i_s_0)  
        i_s_model = diode_saturation_current(i_s_0, model.T)
        i_s_temp_a[i] = model.i_s
        T_i_s_a[i] = model.T 
        print(' For T = ' + str(model.T) + 'K: I_S = ' + str(model.i_s) +
              ' A, I_S_model(T) = ' + str(i_s_model) + ' A.')
        i += 1
        if (290. < model.T < 310.15):
            print('Measurement series at T =', model.T, 
                  'K has less than 10 K difference to T_nom = 300.15 K', 
                  'and will be used as reference.')
            # Set variables for base measurements (T ~ T_0 = 300.15K)
            v_ca_a_0 = v_ca_a
            i_c_a_0 = i_c_a
            c_ca_a_0 = c_ca_a
            # Plots for presentation, optional
            plot_vca_cca_for_presentation(v_ca_a, i_c_a , c_ca_a , model)
            plot_vca_ic_ideal(v_ca_a, i_c_a , model)
            plot_vca_ic_r(v_ca_a, i_c_a , model)

    base_model = DiodeModel(v_ca_a_0, i_c_a_0 , c_ca_a_0 , T, T_i_s_a, 
                            i_s_temp_a)
    # Plot for presentation, optional
    plot_T_is(T_i_s_a, i_s_temp_a, base_model)
    
    with open(results_fname, 'w') as f:
        json.dump(base_model.params, f, ensure_ascii=True)


def diode_model_params_isotherm(v_ca_a, i_c_a , c_ca_a , T):
    """Extract diode model parameters at a fixed temperature

    Args:
        v_ca_a (float array): Cathode-Anode voltage [V.]
        i_c_a  (float array): Diode current [A].
        c_ca_a  (float array): Diode capacitance [F].
        T (float): Temperature [K].

    Returns:
        (dict): Dictionary of model parameters T, I_S, m, R_S, TT
    """
    # Curve fit where I_c curve is purely exponential
    vca_lim_lower_ic = 0.65
    vca_lim_upper_ic = 0.75
    i_s, m = ideal_diode_model(v_ca_a, i_c_a , vca_lim_lower_ic, vca_lim_upper_ic, T)

    print('Model parameters for T = ' + str("{:.0f}".format(T)), 'K: I_S = ' +
          str(i_s) + ' A, m = ' + str(m))

    # Calculate R_S model
    # TODO: Is  mean of differential resistance better than simple
    # quotient of differences?
    # Simple difference between V_CA=0.9V, V_CA=1.0V
    # Curve fit where I_c curve is purely linear
    vca_lim_lower_r = 0.9
    vca_lim_upper_r = 1.0
    vca_lim_lower_r_i = np.where(np.isclose(v_ca_a, vca_lim_lower_r, rtol=1e-3))[0][0]
    vca_lim_upper_r_i =  np.where(np.isclose(v_ca_a, vca_lim_upper_r, rtol=1e-3))[0][0]
    r_ohm_simple = (v_ca_a[vca_lim_upper_r_i] - v_ca_a[vca_lim_lower_r_i])/(i_c_a [vca_lim_upper_r_i] - i_c_a [vca_lim_lower_r_i])
    print('R_S_simple =', str(r_ohm_simple))

    # Calculate C_CA model
    vca_lim_lower_cca = 0.65
    vca_lim_upper_cca = 1.0
    tt = diode_capacitance_model(v_ca_a, i_c_a , c_ca_a , vca_lim_lower_cca, vca_lim_upper_cca)

    print('Transit time for T = ' + str("{:.0f}".format(T)), 'K: TT =',
          "{:.4g}".format(tt), 's.')

    # TODO: Explanations of parameters?
    model_params = {'T': T, 'I_S': i_s, 'm': m, 'R_S': r_ohm_simple,
                    'TT': tt, 'vca_lim_lower_ic': vca_lim_lower_ic,
                    'vca_lim_upper_ic': vca_lim_upper_ic,
                    'vca_lim_lower_cca': vca_lim_lower_cca,
                    'vca_lim_upper_cca': vca_lim_upper_cca,
                    'vca_lim_lower_r': vca_lim_lower_r,
                    'vca_lim_upper_r': vca_lim_upper_r,
                    }
    return model_params


# TODO Separate files for modelling functions?!
def i_s_temp_dependence_model(T_i_s_a, i_s_temp_a ):
    """Determine temperature dependence of diode saturation currrent.

    Args:
        T_i_s_a ([type]): [description]
        i_s_temp_a ([type]): [description]

    Returns:
        [type]: [description]
    """
    log_vec = np.vectorize(np.log)
    p_opt, pcov = curve_fit(diode_saturation_current_log, T_i_s_a,
                            log_vec(i_s_temp_a))
    print(p_opt)
    i_s_temp_coeff = p_opt[0]   # Transit time
    return i_s_temp_coeff


def main():
  process_diode_measurement('/home/matt/Nextcloud/Studium/HauptseminarMikroNanoelektronik/PythonExtraction/data.json')


# Do not execute main() when imported as module
if __name__ == '__main__':
    main()
