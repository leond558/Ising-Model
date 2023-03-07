from typing import Final
from CheckerboardIsingModel import CheckerboardIsingModel
from MultithreadedIsingModel import MultithreadedIsingModel
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import statsmodels.api as sm
import pandas as pd
import pickle as pkl
import time

"""
Graphs Plotted:
1. Critical temperature as a function of lattice size, N. Demonstrating that the value for critical temperature 
eventually reaches some convergence point that is in agreement with the theoretical T_c value.
2. A table showing the calculation of the variables associated with the critical temperature vs N fit and corresponding
error values.
"""

if __name__ == '__main__':
    """
    Want to investigate how critical temperature changes with lattice size and calculate the necessary errors. How 
    does T_c change with linear scaling of N?
    """
    use_existing_data = True
    # Because a lot of data is involved, even computing the mean and std of the resultant magnetisations
    # requires a lot of time, so we store the critical temperatures and the lattice size increments using
    # pickle so that we can jump right through to the plotting without having to repeat the computation
    # for finding the critical temperature values
    jump_to_plot = True

    # List of the values of temperature we sweep through
    temperature_increments = np.linspace(2.1, 2.6, 200)

    # We want the size increments to be both even and evenly spaced
    # CheckerboardIsingModel only takes even values of N
    size_increments = np.linspace(10, 120, 100)
    size_increments = np.around((size_increments / 2)) * 2
    size_increments = np.unique(size_increments.astype(int))

    magnetisations = []
    energies = []

    if use_existing_data:
        with open('scaling.pkl', 'rb') as f:
            magnetisations, energies = pkl.load(f)

    else:
        # This is the longest computation for the project so am curious to how long it will take overall
        total_time = 0

        # Initialise the models and time evolve the systems as usual
        for lattice_size in size_increments:
            initial = time.time()
            print(f"working on {lattice_size}")
            scaling_models = []

            for temperature in temperature_increments:
                scaling_models.append(CheckerboardIsingModel(lattice_size, 0, temperature, False))

            systems = MultithreadedIsingModel(scaling_models)
            systems.simultaneous_time_steps(400)
            scaling_models = systems.model_array

            magnetisation_for_lattice_size = []
            energy_for_lattice_size = []

            for model in scaling_models:
                magnetisation_for_lattice_size.append(model.magnetisation_array)
                energy_for_lattice_size.append(model.energy_array)

            magnetisations.append(magnetisation_for_lattice_size)
            energies.append(energy_for_lattice_size)

            final = time.time()
            total_time += final - initial
            # because this is a long computation, want to ensure that it's running smoothly
            # and if something goes wrong, what particular lattice size caused the error
            print(f"{lattice_size} took {final - initial:0.2f} seconds")

        with open('scaling.pkl', 'wb') as f:
            pkl.dump([magnetisations, energies], f)

        total_time_mins = np.trunc(total_time / 60)
        total_time_secs = total_time % 60
        # print out the overall time taken
        print(f"overall took {total_time_mins} minutes and {total_time_secs} seconds ")

        # took about 40 minutes for the N values estabished above

    magnetisation_mean_list = []
    magnetisation_deviation_list = []
    energy_mean_list = []
    energy_deviation_list = []

    # if the critical temperatures are already found, no need to compute them again
    if jump_to_plot:
        with open('scaling_critical.pkl', 'rb') as f:
            critical_temperatures, size_increments = pkl.load(f)

    else:

        for i, lattice_size in enumerate(size_increments):
            magnetisation_means = []
            magnetisation_deviations = []
            energy_means = []
            energy_deviations = []

            for j in range(len(temperature_increments)):
                # calculating the means of energy and magnetisation and finding the deviation which will be used
                # for error determination
                # 150 index for equilibration
                magnetisation_mean = np.abs(np.mean(magnetisations[i][j][150:]))
                magnetisation_deviation = np.std(magnetisations[i][j][150:])
                energy_mean = np.mean(energies[i][j][150:])
                energy_deviation = np.std(energies[i][j][150:])

                magnetisation_means.append(magnetisation_mean)
                magnetisation_deviations.append(magnetisation_deviation)
                energy_means.append(energy_mean)
                energy_deviations.append(energy_deviation)

            magnetisation_mean_list.append(magnetisation_means)
            magnetisation_deviation_list.append(magnetisation_deviations)
            energy_mean_list.append(energy_means)
            energy_deviation_list.append(energy_deviations)

        # Finding the critical temperatures

        smooth_means = []
        smooth_temperatures = []
        smooth_errors = []

        # applying some smoothing
        for (mag, dev) in zip(magnetisation_mean_list, magnetisation_deviation_list):
            smoothened_behaviour = sm.nonparametric.lowess(mag, temperature_increments, frac=0.3)
            smooth_means.append(smoothened_behaviour[:, 1])
            smooth_temperatures.append(smoothened_behaviour[:, 0])
            smoothened_deviation_behaviour = sm.nonparametric.lowess(dev, temperature_increments, frac=0.3)
            smooth_errors.append(smoothened_deviation_behaviour[:, 1])

        # finding the derivative function of magnetisatoin which is necessary to find critical temperature value
        smooth_derivatives_mean = []
        smooth_derivatives_temperatures = []
        for (mag, t) in zip(smooth_means, smooth_temperatures):
            derivative = np.diff(mag) / np.diff(t)
            smooth_derivative = sm.nonparametric.lowess(derivative, t[1:], frac=0.3)
            smooth_derivatives_mean.append(smooth_derivative[:, 1])
            smooth_derivatives_temperatures.append(smooth_derivative[:, 0])

        critical_temperatures = []
        error_critical_temperatures = []
        # theoretical value for critical temperature
        ONSAGER_CRITICAL_TEMPERATURE: Final = 2 / np.log(1 + np.sqrt(2))

        for (mag, temp) in zip(smooth_derivatives_mean, smooth_derivatives_temperatures):
            # crit temperature found at the point where the absolute value of the
            # derivative function is maximised
            # i.e. the trough of the curve
            critical_temperature_index = np.argmin((mag))
            critical_temperatures.append(temp[critical_temperature_index])

        with open('scaling_critical.pkl', 'wb') as f:
            pkl.dump([critical_temperatures, size_increments], f)

    # removing some anomalous results present due to the randomness of the
    # time evolution of the system and the spin flipping
    critical_temperatures = critical_temperatures[5:45]
    size_increments = size_increments[5:45]
    smooth_crit = sm.nonparametric.lowess(critical_temperatures, size_increments, frac=0.1)
    smooth_crit, smooth_size = smooth_crit[:, 1], smooth_crit[:, 0]


    # defining the function for critical temperature as given in the manual
    def critical_temperature_func(N, T_c_infinity, a, v):
        """
        Function representing the formula for critical temperature as given in the manual.
        :param N: lattice size
        :param T_c_infinity: critical temperature value at N= infinity
        :param a: coefficient
        :param v: critical v value
        :return: critical temperature for a lattice with size N
        """
        return T_c_infinity + a * np.power(N, (-1 / v))


    # Establish bounds to ensure the fit reflects the actual relationship of the data, bound are composed of the
    # approximate literature values of the critical values in thed function
    bounds = [2.5, 2, 1.1]

    # finding the function / fit from the data
    param, param_cov = scipy.optimize.curve_fit(critical_temperature_func, smooth_size, smooth_crit, bounds=(0, bounds))

    error_in_fit_variables = []
    # finding the errors
    for i, covariance in enumerate(param_cov):
        # diagonal elements in the covariance matrix correspond to the variance of the fitted variables
        variance_of_fit = covariance[i]
        std_of_fit = np.sqrt(variance_of_fit)
        error_of_fit = std_of_fit / np.sqrt(len(smooth_size))
        error_in_fit_variables.append(error_of_fit)

    # plotting the data points
    plt.scatter(smooth_size, smooth_crit, color='black', label='data')

    # plotting the curve_fit calculated fit
    sizes = np.linspace(10, 120, 400)
    plt.plot(sizes, critical_temperature_func(sizes, param[0], param[1], param[2]), color='red',
             linestyle='dashed', label='scipy curve_fit \n fit')

    plt.title('Inestigating the effect on the value of critical temperature \n with linear scaling of the size of the '
              'lattice')
    plt.xlabel('N, size of the lattice')
    plt.ylabel('$T_c$, Critical temperature / $($J$/k_B)$')

    param = np.around(param, 2)
    error_in_fit_variables = np.around(error_in_fit_variables, 2)

    plt.ylim(bottom=(param[0] - (param[0] / 98)))
    # plotting the T_c(infinity) asymptote
    plt.plot(sizes, np.full(400, param[0]), color='blue', linestyle='dotted', label='fitted $T_c(\infty)$')
    # plotting the equation of the fit on the graph and the value of T_c(infinity)
    plt.annotate(f'$T_c(\infty) = $ {param[0]} $\pm$ {error_in_fit_variables[0]} asymptotic convergence', (18, 2.26))
    plt.annotate(f'$T_c = ${param[0]} + {param[1]}*$N^{np.around(-1 / param[2])}$', (18, 2.45))
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('SCAL 1.png')
    plt.show()

    # T_c value from the manual, defined as a constant
    ONSAGER_CRITICAL_TEMPERATURE: Final = 2 / np.log(1 + np.sqrt(2))
    print(param[2])
    # calculating the percent difference between the fitted variables and the theoretical ones
    percentage_difference = [
        np.around((100 * (param[0] - ONSAGER_CRITICAL_TEMPERATURE)) / ONSAGER_CRITICAL_TEMPERATURE, 2), '-',
        np.around((100 * (1 - param[2])))]
    error_difference = [np.around((param[0] - ONSAGER_CRITICAL_TEMPERATURE) / error_in_fit_variables[0], 2), '-',
                        np.around((param[2] - 1) / error_in_fit_variables[2], 2)]

    # plotting the value of the fitted variables and their associated error
    # using pandas
    all_data = {
        "Fitted Variable": ['T_c(infinity)', 'alpha', 'v'],
        "Value": param,
        "Error in Value": error_in_fit_variables,
        "Percentage Difference \n from Theoretical Value": percentage_difference,
        "Error Difference \n from Theoretical Value": error_difference
    }

    df = pd.DataFrame(all_data)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    plt.tight_layout()
    plt.savefig("SCAL 2")
    plt.show()

    raise SystemExit(0)
