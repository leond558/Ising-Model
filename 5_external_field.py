import matplotlib.pyplot as plt
import numpy as np
from typing import Final
from CheckerboardIsingModel import CheckerboardIsingModel
from MultithreadedIsingModel import MultithreadedIsingModel
import statsmodels.api as sm
import pickle as pkl

"""
Considering the effects of varying the external field strength for the systems for a system with N = 60.
1. Plotting magnetisatoin vs temperature for different magnetic field strengths.
2. PLotting heat capacity vs temperature for different field strengths
3. Plotting critical temperature vs magnetic field strength, showing a roughly linear relationship 
    between h and T_c.
"""

if __name__ == '__main__':

    use_existing_data = True
    lattice_size = 60
    temperature_increments = np.linspace(0, 10, 500)
    external_field_strengths = []
    magnetisation_array_list = []
    energy_array_list = []

    # Serialising data using pickle to minimise repeated computationally intense program running
    if use_existing_data:
        with open('external_field.pkl', 'rb') as f:
            external_field_strengths, magnetisation_array_list, energy_array_list = pkl.load(f)
        f.close()

        with open('sixty.pkl', 'rb') as f:
            h_zero_temps, h_zero_mag_mean, h_zero_energy_mean = pkl.load(f)
        f.close()

    else:

        # defining the range of field strengths considered
        external_field_strengths = np.arange(0, 4, 0.5)

        # time evolving systems for different lattize sizes for each temperature
        for strength in external_field_strengths:
            external_models = []
            for temperature in temperature_increments:
                external_models.append(CheckerboardIsingModel(lattice_size, strength, temperature, False))

            systems = MultithreadedIsingModel(external_models)
            systems.simultaneous_time_steps(800)
            external_models = systems.model_array

            magnetisation_for_strength = []
            energy_for_strength = []

            for model in external_models:
                magnetisation_for_strength.append(model.magnetisation_array)
                energy_for_strength.append(model.energy_array)

            magnetisation_array_list.append(magnetisation_for_strength)
            energy_array_list.append((energy_for_strength))

        with open('external_field.pkl', 'wb') as f:
            pkl.dump([external_field_strengths, magnetisation_array_list, energy_array_list], f)

    magnetisation_mean_list = []
    magnetisation_deviation_list = []
    energy_mean_list = []
    energy_deviation_list = []

    for i, strength in enumerate(external_field_strengths):
        magnetisation_means = []
        magnetisation_deviations = []
        energy_means = []
        energy_deviations = []

        # calculating the mean and deviatoin for the magnetisation and energy for the lattice sizes considered
        for j in range(len(temperature_increments)):
            # [150:] to allow for system equilibration
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

    # plotting magnetisation against temperature
    plt.figure(1, figsize=(8, 8))

    for i, (mag, strength) in enumerate(zip(magnetisation_mean_list, external_field_strengths)):
        # for some reason it was found that the h = 0 data was corrupted, thus data from a
        # previous computation using N = 60 was used isntead
        if i == 0:
            smoothened_plot = sm.nonparametric.lowess(h_zero_mag_mean, h_zero_temps, frac=0.1)
            plt.plot(smoothened_plot[:, 0], smoothened_plot[:, 1], label=f'h = {strength}', color='black',
                     linestyle='dashed')
            continuation = [np.linspace(3.5, 10, 100), np.zeros(100)]
            plt.plot(continuation[0], continuation[1], color='black', linestyle='dashed')
        else:
            # smoothening the data to eliminate random fluctuations
            smoothened_plot = sm.nonparametric.lowess(mag, temperature_increments, frac=0.1)
            plt.plot(smoothened_plot[:, 0], smoothened_plot[:, 1], label=f'h = {strength}')

    plt.xlabel('Temperature/ $J/k_B$')
    plt.ylabel('Mean magnetisation fraction')
    plt.title('Investigating how mean magnetisation varies with temperature for \n different external field strengths')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig('EXT 1.png')
    plt.show()

    # Finding the heat capacities

    heat_capacities_list = []
    temperature_derivative_list = []
    for i, (energy_mean, energy_deviation) in enumerate(zip(energy_mean_list, energy_deviation_list)):
        if i == 0:
            # pulling a pre existing calculation for heat capacity for the h = 0, no external field case
            # again due to corruption of data
            with open('sixty_capacity.pkl', 'rb') as f:
                h_zero_capacity, h_zero_dtemps = pkl.load(f)
            f.close()
        else:
            capacity_energy = np.asarray(energy_mean)
            capacity_deviations = np.asarray(energy_deviation)

            smooth_energy = sm.nonparametric.lowess(capacity_energy, temperature_increments, frac=0.1)
            capacity_energy = smooth_energy[:, 1]
            capacity_temperatures = smooth_energy[:, 0]

            # calculating the heat capacity
            heat_capacity = (capacity_energy[1:] - capacity_energy[:-1]) / (
                    temperature_increments[1:] - temperature_increments[:-1])
            temperature_derivative = (capacity_temperatures[1:] + capacity_temperatures[:-1]) / 2

            heat_capacities_list.append(heat_capacity)
            temperature_derivative_list.append(temperature_derivative)

    # heat capacities vs temperature for the different magnetic fields
    plt.figure(2, figsize=(8, 8))
    for i, (cap, temps, strength) in enumerate(
            zip(heat_capacities_list, temperature_derivative_list, external_field_strengths)):
        if i == 0:
            smooth_h_zero = sm.nonparametric.lowess(h_zero_capacity, h_zero_dtemps, frac=0.1)
            # plotting the h=0 case as a dashed line to distinguish it from non zero h fields
            plt.plot(smooth_h_zero[:, 0], smooth_h_zero[:, 1], label=f'h = {strength}', color='black',
                     linestyle='dashed')
        if i > 1:
            plt.plot(temps, cap, label=f'h = {strength}', )
    plt.legend(loc='upper left')
    plt.xlabel('Temperature / $($J$/k_B)$')
    plt.xlim([0, 8])
    plt.ylabel('Heat Capacity / k_B')
    plt.title('Heat capacity against temperature \n for different external field strengths \n for a lattice of N = 60')
    plt.tight_layout()
    plt.savefig('EXT 2.png', bbox_inches='tight')
    plt.show()

    # finding critical temperatures for the different field strengths
    # same method used from 3_temperature_dependence.py

    smooth_derivatives_mean = []
    smooth_derivatives_temperatures = []
    smooth_magnetisation_list = []
    smooth_temperature_list = []
    error_list = []

    for (mag, dev) in zip(magnetisation_mean_list, magnetisation_deviation_list):
        smooth_mag = sm.nonparametric.lowess(mag, temperature_increments, frac=0.1)
        # index is 120 onwards as this is when the magnetisation behaviour is that which
        # can be modelled with a polynomial i.e. the temperature values for which
        # magnetisation is decreasing and the finding the trough of the derivative curve is
        # permissible
        smooth_mag, smooth_temp = smooth_mag[120:, 1], smooth_mag[120:, 0]
        smooth_magnetisation_list.append(smooth_mag)
        smooth_temperature_list.append((smooth_temp))

        deviation = np.asarray(dev)
        smooth_errors = sm.nonparametric.lowess(deviation, temperature_increments, frac=0.3)
        error_list.append(smooth_errors[120:, 1])

    for (mag, t) in zip(smooth_magnetisation_list, smooth_temperature_list):
        derivative = np.diff(mag) / np.diff(t)
        smooth_derivative = sm.nonparametric.lowess(derivative, t[1:], frac=0.3)
        smooth_derivatives_mean.append(smooth_derivative[:, 1])
        smooth_derivatives_temperatures.append(smooth_derivative[:, 0])

    smooth_derivatives_errors = []

    # calculating the error
    for (err, mag, der) in zip(error_list, smooth_magnetisation_list, smooth_derivatives_mean):
        error = der * ((err[1:] / len(err[1:])) / mag[1:])
        smooth_derivatives_errors.append(error)

    critical_temperatures = []
    error_critical_temperatures = []

    # defining the T_c constant
    ONSAGER_CRITICAL_TEMPERATURE: Final = 2 / np.log(1 + np.sqrt(2))
    for i, (mag, dev, temp) in enumerate(
            zip(smooth_derivatives_mean, smooth_derivatives_errors, smooth_temperature_list)):
        if i == 0:
            # from h=0 analysis
            critical_temperatures.append(2.272)
            error_critical_temperatures.append(0.004)
        if i > 1:
            critical_temperature_index = np.argmin((mag))
            critical_temperatures.append(temp[critical_temperature_index])
            error = dev[critical_temperature_index]
            error_critical_temperatures.append(error)

    # rounding
    critical_temperatures = np.around(critical_temperatures, 3)
    error_critical_temperatures = np.abs(np.around(error_critical_temperatures, 3))

    # insert true Onsager value
    crits_with_intercept = critical_temperatures
    crits_with_intercept[0] = 2.269

    # removing h=0.5 because corrupted
    external_field_strengths = np.delete(external_field_strengths, 1)

    # finding the fit for the critical temperatures vs field strength relationship
    grad, c = np.polyfit(external_field_strengths, crits_with_intercept, 1)


    # defining a functoin for the fit
    def critical_temperature_func(field_strength_list):
        field_strength_list = np.asarray(field_strength_list)
        return grad * field_strength_list + c


    # calculating critical temepratures from the fit
    fit_strengths = np.linspace(0, 3.5, 100)
    fit_criticals = critical_temperature_func(fit_strengths)

    # plot of critical temperatures against field strength for both data and a fit
    plt.figure(3, figsize=(8, 8))
    plt.scatter(external_field_strengths, critical_temperatures, label='Experimental', color='black', linestyle='None')

    # plot the errors
    plt.errorbar(external_field_strengths, critical_temperatures, xerr=None, yerr=error)
    plt.plot(fit_strengths, fit_criticals, label='np.polyfit \nFit', color='red', linestyle='dashed')
    plt.legend(loc='lower right')
    plt.title(
        "Investigating how critical temperature changes with \n different external field strengths \n for a lattice of N-60")
    plt.xlabel("External field strengths / J")
    plt.ylabel("Critical temperature / $($J$/k_B)$")
    plt.annotate("$T_c = 1.09\dot{h} + 2/ln(1+\sqrt{2})$", (1.5, 4))
    plt.tight_layout()
    plt.savefig("EXT 3.png")
    plt.show()

