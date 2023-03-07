import matplotlib.pyplot as plt
import numpy as np
import scipy
from typing import Final
from CheckerboardIsingModel import CheckerboardIsingModel
from MultithreadedIsingModel import MultithreadedIsingModel
import statsmodels.api as sm
import pickle as pkl
import pandas as pd

"""
TEMP
1. Investigating how mean magnetisation fraction is affected by different temperatures for the lattice system.
2. Plotting a fit for the relationship between mean magnetisation against temperature for N=60. An R^2 value 
    is also calculated to show the strength of the fit.
3. Plotting the derivative of the magnetisation against temperature. The critical temperature is the temperature at 
    which magnetisation changes most rapidly and hence would be at the point where the 
    absolute value of the derivative is maximised.
    A fit is found for the derivative magnetisation and the critical temperature found.
4. A plot of the mean energy against temperature.
5. A plot of heat capacity from the data and from the dissipation fluctuation formula.

CRIT
1. A table of the critical temperatures for the considered lattice sizes and their associated errors.
2. A table of the mean critical temperature and percentage difference from the Onsager value.
"""
if __name__ == '__main__':

    use_existing_data = True
    lattice_size_list = []
    temperature_increments = []
    magnetisation_array_list = []
    energy_array_list = []

    # Using pickle to serialise and store data because gathering the required data for the above plots
    # has become very computationally expensive. pickle prevents having to compute data each time any
    # fine adjustment is to be made to a graph
    if use_existing_data:
        with open('testing.pkl', 'rb') as f:
            lattice_size_list, temperature_increments, magnetisation_array_list, energy_array_list = pkl.load(f)
        f.close()

    else:
        # defining parameters for the system
        lattice_size_list = [10, 40, 60, 100]
        temperature_increments = np.unique(
            np.concatenate([np.linspace(0, 1.5, 100), np.linspace(1.5, 2.5, 400), np.linspace(2.5, 3.5, 100)]))

        # standard creation and evolution of the system
        for lattice_size in lattice_size_list:
            dynamic_models = []
            for temperature in temperature_increments:
                dynamic_models.append(CheckerboardIsingModel(lattice_size, 0, temperature, False))

            systems = MultithreadedIsingModel(dynamic_models)
            systems.simultaneous_time_steps(1000)
            dynamic_models = systems.model_array

            magnetisation_for_lattice_size = []
            energy_for_lattice_size = []

            for model in dynamic_models:
                magnetisation_for_lattice_size.append(model.magnetisation_array)
                energy_for_lattice_size.append(model.energy_array)

            # storing both magnetisation and arrays for each lattice size for each system with a different
            # temperature value
            magnetisation_array_list.append(magnetisation_for_lattice_size)
            energy_array_list.append((energy_for_lattice_size))

        with open('temperature_dependence.pkl', 'wb') as f:
            pkl.dump([lattice_size_list, temperature_increments, magnetisation_array_list, energy_array_list], f)

    magnetisation_mean_list = []
    magnetisation_deviation_list = []
    energy_mean_list = []
    energy_deviation_list = []

    for i, lattice_size in enumerate(lattice_size_list):
        magnetisation_means = []
        magnetisation_deviations = []
        energy_means = []
        energy_deviations = []

        for j in range(len(temperature_increments)):
            # calculating the means of energy and magnetisation and finding the deviation which will be used
            # for error determination
            magnetisation_mean = np.abs(np.mean(magnetisation_array_list[i][j][150:]))
            magnetisation_deviation = np.std(magnetisation_array_list[i][j][150:])
            energy_mean = np.mean(energy_array_list[i][j][150:])
            energy_deviation = np.std(energy_array_list[i][j][150:])

            magnetisation_means.append(magnetisation_mean)
            magnetisation_deviations.append(magnetisation_deviation)
            energy_means.append(energy_mean)
            energy_deviations.append(energy_deviation)

        magnetisation_mean_list.append(magnetisation_means)
        magnetisation_deviation_list.append(magnetisation_deviations)
        energy_mean_list.append(energy_means)
        energy_deviation_list.append(energy_deviations)

    with open('sixty.pkl', 'wb') as f:
        pkl.dump([temperature_increments, magnetisation_mean_list[2], energy_mean_list[2]], f)
    f.close()

    # Plotting mean magnetisation against temperature
    plot_colours = ['red', 'black', 'blue', 'green']
    plt.figure(1, figsize=(8, 8))

    for (mag, size, colour) in zip(magnetisation_mean_list, lattice_size_list, plot_colours):
        # Applying some smoothening to the values
        smoothened_plot = sm.nonparametric.lowess(mag, temperature_increments, frac=0.3)
        plt.plot(smoothened_plot[:, 0], smoothened_plot[:, 1], label=f'N = {size}', color=colour)
    plt.xlabel('Temperature/ J/k_B')
    plt.ylabel('Mean magnetisation')
    plt.title('Investigating how mean magnetisation depends on temperature for \n different lattice sizes')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig('TEMPER 1')
    plt.show()

    # Finding the fit for magnetisation against temperature
    smooth_means = []
    smooth_temperatures = []
    smooth_errors = []

    # For the fit, we need to restrict the values considered to only those in the region where
    # the system is not equilibrated, i.e. the temperature values where there is a rapid
    # decrease in magnetisation
    for (mag, dev) in zip(magnetisation_mean_list, magnetisation_deviation_list):
        smoothened_behaviour = sm.nonparametric.lowess(mag, temperature_increments, frac=0.3)
        smooth_means.append(smoothened_behaviour[360:520, 1])
        smooth_temperatures.append(smoothened_behaviour[360:520, 0])
        smoothened_deviation_behaviour = sm.nonparametric.lowess(dev, temperature_increments, frac=0.3)
        smooth_errors.append(smoothened_deviation_behaviour[360:520, 1])

    # finding the fit for N=60
    sixty_mean = np.asarray(smooth_means[2])
    sixty_temperatures = np.asarray(smooth_temperatures[2])

    # func in the form M = M_0(T)^a
    # find the variables in the func using polyfit and rewriting the func as a linear equation
    # through taking the logarithm of the variables

    grad, c = np.polyfit(np.log(sixty_temperatures), np.log(sixty_mean), 1, w=np.sqrt(sixty_mean))


    # defining a function for the fit determined
    def ideal_magnetisation_fit(temperature_array):
        log_temperature_array = np.log(temperature_array)
        return np.exp(c + grad * log_temperature_array)


    # finding the R^2 value
    sixty_fit = ideal_magnetisation_fit(sixty_temperatures)
    squared_differences = np.square(sixty_mean - sixty_fit)
    squared_differences_from_mean = np.square(sixty_mean - np.mean(sixty_mean))
    rSquared = 1 - np.sum(squared_differences) / np.sum(squared_differences_from_mean)
    print(f"R² = {rSquared}")

    # plotting the fit and data for N=60 case of magnetisation vs temperature
    plt.figure(2, figsize=(6, 4))
    smoothened_sixty = sm.nonparametric.lowess(magnetisation_mean_list[2], temperature_increments, frac=0.3)
    plt.plot(smoothened_sixty[:, 0], smoothened_sixty[:, 1], label='Data', color='red')
    plt.plot(sixty_temperatures, sixty_fit, label='numpy.polyfit fit', color='black', linestyle='dashed')
    plt.xlabel('Temperature/ J/k_B')
    plt.ylabel('Mean magnetisation')
    plt.title('Mean magnetisation vs temperature \n plotted for both data and a fitted function \n using a lattice '
              'with N=60')
    plt.legend(loc='lower left')
    # including the R^2 value in the plot
    plt.annotate(f"R² = {rSquared:.3}", (2.3, 0.8))
    plt.tight_layout()
    plt.savefig('TEMPER 2')
    plt.show()


    # Plotting the derivative for N=60 vs fit

    # the fit for the derivative is just the derivative of the above fit
    def derivative_ideal_magnetisation(temperature_array):
        return np.exp(c) * grad * np.power(temperature_array, grad - 1)


    # temperature values that are to be passed onto the fit for the derivative
    temps = np.linspace(2.285, 2.72, 400)

    # finding the values for the derivative from the data
    smooth_derivatives_mean = []
    smooth_derivatives_temperatures = []
    for (mag, t) in zip(smooth_means, smooth_temperatures):
        derivative = np.diff(mag) / np.diff(t)
        smooth_derivative = sm.nonparametric.lowess(derivative, t[1:], frac=0.3)
        smooth_derivatives_mean.append(smooth_derivative[:, 1])
        smooth_derivatives_temperatures.append(smooth_derivative[:, 0])

    # the critical temperature is going to happen at the point that the derivative shows a trough
    # as this is the point of most rapid change of the magnetisation value
    # find the index at which this trough happens and find the corresponding temperature value
    critical_temperature_index_N60 = np.argmin(smooth_derivatives_mean[2])
    critical_temperature_N60 = smooth_derivatives_temperatures[2][critical_temperature_index_N60]

    # plot derivative of magneisation against temperature for both data and the fit for N=60
    plt.figure(3, figsize=(6, 4))
    plt.plot(smooth_derivatives_temperatures[2], smooth_derivatives_mean[2], label='Data', color='red')
    plt.plot(temps, derivative_ideal_magnetisation(temps), label='numpy.polyfit fit', color='black', linestyle='dashed')
    plt.xlabel('Temperature/ J/k_B')
    plt.ylabel('Derivative of mean magnetisation')
    plt.title('Derivative of mean magnetisation vs temperature \n plotted for both data and a fitted function \n '
              'using a lattice with N=60')
    plt.legend(loc='lower right')
    # annotate the plot with the critical temperature value at the trough of the derivative curve
    plt.annotate(f"$T_c$ = \n {critical_temperature_N60:.3}$J/k_b$", (2.2, -2))
    plt.tight_layout()
    plt.savefig('TEMPER 3')
    plt.show()

    # Plotting energy
    plot_colours = ['black', 'red', 'blue', 'green']
    plt.figure(4, figsize=(6, 4))

    for (eng, size, colour) in zip(energy_mean_list, lattice_size_list, plot_colours):
        # applying some smoothening
        smoothened_plot = sm.nonparametric.lowess(eng, temperature_increments, frac=0.3)
        plt.plot(smoothened_plot[:, 0], smoothened_plot[:, 1], label=f'N = {size}', color=colour)

    plt.xlabel('Temperature/ J/k_B')
    plt.ylabel('Energy/ J')
    plt.title('Investigating how energy depends on temperature for \n different lattice sizes')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('TEMPER 4')
    plt.show()

    # Finding the heat capacity of the system N=60
    sixty_energy = np.asarray(energy_mean_list[2])
    sixty_energy_deviations = np.asarray(energy_deviation_list[2])
    smooth_energy = sm.nonparametric.lowess(energy_mean_list[2], temperature_increments, frac=0.2)
    sixty_energy = smooth_energy[:, 1]
    sixty_energy_temperature = smooth_energy[:, 0]

    # calculating the heat capacity
    heat_capacity = (sixty_energy[1:] - sixty_energy[:-1]) / (
            temperature_increments[1:] - temperature_increments[:-1])
    # calculating the derivative of tempreautre
    sixty_energy_temperature_derivative = (sixty_energy_temperature[1:] + sixty_energy_temperature[:-1]) / 2

    # storing this heat capacity information
    with open('sixty_capacity.pkl', 'wb') as f:
        pkl.dump([heat_capacity, sixty_energy_temperature_derivative], f)
    f.close()

    # defining Boltzmann's constant, the final marker indicates that it is a constant
    BOLTZMANN_CONSTANT: Final = scipy.constants.value(key='Boltzmann constant')

    # calculate the heat capacity using the alternative method of the fluctuation_dissipation formula

    fluctuation_dissipation = (sixty_energy_deviations ** 2) / (temperature_increments ** 2)
    smooth_fluctuation_dissipation = sm.nonparametric.lowess(fluctuation_dissipation, temperature_increments, frac=0.2)

    # plotting heat capacity against temperature
    plt.figure(4, figsize=(6, 4))
    # plotting the data
    plt.plot(sixty_energy_temperature_derivative, heat_capacity, color='black', label='C from Energy')
    # plotting the fit
    plt.plot(smooth_fluctuation_dissipation[:, 0], smooth_fluctuation_dissipation[:, 1], color='red',
             label='C from fluctuation \n -dissipation theorem', linestyle='dashed')
    plt.legend(loc='upper left')
    plt.xlabel('Temperature / $($J$/k_B)$')
    plt.ylabel('Heat Capacity / k_B')
    plt.title('Heat capacity against temperature \n for a lattice of size N=60')
    plt.tight_layout()
    plt.savefig('TEMPER 5.png', bbox_inches='tight')
    plt.show()

    # finding critical temperatures

    with open('crit.pkl', 'wb') as f:
        pkl.dump([smooth_errors, smooth_means, smooth_derivatives_mean], f)
    f.close()

    # finding the error in the derivatives which are used in the error in critical temperature values
    smooth_derivatives_errors = []
    for (err, mag, der) in zip(smooth_errors, smooth_means, smooth_derivatives_mean):
        # error found through use of quadrature
        error = der * ((err[1:] / len(err[1:])) / mag[1:])
        smooth_derivatives_errors.append(error)

    critical_temperatures = []
    error_critical_temperatures = []

    # T_c value from the manual, defined as a constant
    ONSAGER_CRITICAL_TEMPERATURE: Final = 2 / np.log(1 + np.sqrt(2))

    for (mag, dev, temp) in zip(smooth_derivatives_mean, smooth_derivatives_errors, smooth_temperatures):
        # locating the critical temperature at the trough of the magnetisation derivative curve
        critical_temperature_index = np.argmin((mag))
        critical_temperatures.append(temp[critical_temperature_index])
        # extracting the associated error of the critical temperature
        error = dev[critical_temperature_index]
        error_critical_temperatures.append(error)

    # calculating the mean critical temperature
    mean_critical_temperature = np.mean(critical_temperatures)
    mean_error_critical_temperature = np.sum(error_critical_temperatures)

    # rounding
    critical_temperatures = np.around(critical_temperatures, 3)
    error_critical_temperatures = np.around(error_critical_temperatures, 3)

    mean_critical_temperature = np.around(mean_critical_temperature, 2)
    mean_error_critical_temperature = np.abs(np.around(mean_error_critical_temperature, 2))

    # calculating the percentage and error difference from the literature value of T_c
    percentage_difference = np.around(
        (ONSAGER_CRITICAL_TEMPERATURE - mean_critical_temperature) * 100 / ONSAGER_CRITICAL_TEMPERATURE, 2)
    error_difference = np.around(
        (ONSAGER_CRITICAL_TEMPERATURE - mean_critical_temperature) / mean_error_critical_temperature, 2)

    # presenting the data in a pandas table
    all_data = {
        "N, lattice size": lattice_size_list,
        "T_c, Critical Temperature": critical_temperatures,
        "dT_c, Error in Critical Temperature": error_critical_temperatures
    }
    df = pd.DataFrame(all_data)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    plt.tight_layout()
    plt.savefig("CRIT 1")

    mean_data = {
        "|T_c|, Mean Critical Temperature": [mean_critical_temperature],
        "d|T_c|, Error in \n Mean Critical Temperature": [mean_error_critical_temperature],
        "Percentage Difference from \nOnsager Critical Temperature": [percentage_difference],
        "Number of Errors from \nOnsager Critical Temperature": [error_difference]
    }
    df = pd.DataFrame(mean_data)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    plt.tight_layout()
    plt.savefig("CRIT 2")
    plt.show()
