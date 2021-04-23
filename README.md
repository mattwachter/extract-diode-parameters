# Automated extraction of diode model parameters

Extract model parameters from measurements of diode current, voltage and capacitance at different temperatures.
The model parameters can be used in an equivalent circuit. 
Plotting of measurement data and simulated diode behavior is also supported.

This project was developed for the seminar
[Hauptseminar Mikro-und Nanoelektronik](https://www.iee.et.tu-dresden.de/iee/eb/lehre/lv/lak/MIK_NANO_ET_lak.html) at TU Dresden by Matthias von Wachter.


## Getting started

No actual installation is required.
Just clone the repo

```bash
git clone https://gitlab.hrz.tu-chemnitz.de/s1760196--tu-dresden.de/diode_model_parameters.git/
```

and run the script

```bash
cd diode_model_parameters
python3 extract_diode_model_parameters.py
```

You can modify the values in `file_names.json` to set custom paths for the input and output files and the directory in which plots should be saved.

Example measurement data is in `data.json`.

The calculated model parameters are saved in `model.json` by default.

### System Requirements

 * git
 * Python 3.7+
 * matplotlib
 * Scipy
 * NumPy


## Background

The (German) presentation `DiodenModellparameterExtrahieren_pres.pdf` has more details on the genesis of the project.


### Equivalent circuit

Part of the model is the Shockley Diode Equation:

```
I_D = I_S * (exp(V_D/m*V_T)-1)
```

With the thermal voltage `V_T = kT/q`.

All of the diode equations that form the basis of the model used for this project are listed on slide 5 ("Grundgleichungen") of `DiodenModellparameterExtrahieren_pres.pdf`.
This slide also shows the equivalent circuit.


### Model parameters

The following parameters are saved in 'model.json'

| Parameter        | Description     | SI Unit  |
| ------------- |-------------| :-----:|
| T | Ambient temperature    |   K |
| I_S      | Saturation current | A |
| m | Ideality factor      |   - |
| R_S | Ohmic resistance      |    V/A |
| TT | Transfer time     |    s|


## Acknowledgements

I want to thank [Markus Müller](https://www.iee.et.tu-dresden.de/iee/eb/mitarbeiter/mit_mueller.html), my supervisor for this project, who did the measurements in `data.json` and had a lot of constructive advice and feedback.

The literature I used can also be found in the PDF Presentation.
