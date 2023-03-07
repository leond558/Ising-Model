from CheckerboardIsingModel import CheckerboardIsingModel
from MultithreadedIsingModel import MultithreadedIsingModel
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

"""
for a lattice of N=60:
1. Hysteresis curves for various temperatures.
2. Plot of energy dissipation against temperature with energy dissipation taken from area in the hysteresis curve.
"""
if __name__ == '__main__':
    use_existing_data = True

    # Generate list of temperatures and magnetic fields
    temperature_increments = np.unique(
        np.concatenate([np.linspace(0.05, 1.5, 30), np.linspace(1.5, 2.5, 100), np.linspace(2.5, 3.5, 30)]))
    external_field_strengths = np.concatenate(
        [np.linspace(0, 3.0, 50), np.linspace(3.0, -3.0, 100), np.linspace(-3.0, 3.0, 100),
         np.linspace(3.0, -3.0, 100)])

    magnetisation_array_list = []
    lattice_size = 60

    # iniitliase the systems and evolve them using multiprocessing
    if use_existing_data:
        with open('hysteresis.pkl', 'rb') as f:
            external_field_strengths, temperature_increments, magnetisation_array_list = pkl.load(f)
    else:
        hysteresis_models = []

        for temperature in temperature_increments:
            hysteresis_models.append(CheckerboardIsingModel(lattice_size, 0, temperature, False))

        systems = MultithreadedIsingModel(hysteresis_models)
        hysteresis_models = systems.model_array

        # to achieve the hysteresis effects, we must cycle through h field values
        for i, strength in enumerate(external_field_strengths[1:]):
            print('here')
            # we gradually time evolve the system by 1 step at each time, changing the h value at each step
            # so each time step, each model has a new value of h
            for model in hysteresis_models:
                model.h_tilde = strength
            systems = MultithreadedIsingModel(hysteresis_models)
            systems.simultaneous_time_steps(1)
            hysteresis_models = systems.model_array

        for model in hysteresis_models:
            magnetisation_array_list.append(model.magnetisation_array)

        with open('hysteresis.pkl', 'wb') as f:
            pkl.dump([external_field_strengths, temperature_increments, magnetisation_array_list], f)

    # Get temperatures closest to those specified to display on the plot
    plot_temperatures = [0.6, 1.2, 1.8, 2.4, 3]
    plot_indexes = [np.argmax(temperature_increments >= temperature) for temperature in plot_temperatures]

    plot_magnetisations = []
    for index in plot_indexes:
        plot_magnetisations.append(magnetisation_array_list[index])

    # calculate the area underneath the hysteresis curve to find energy dissipation
    areas_under_curve = []
    for mag in magnetisation_array_list:
        # indexes reflect the forward and backward motion of the hysteresis loop
        # 150: onwards is to allow time for equilibration
        area = ((np.sum(mag[150:250]) - np.sum(mag[250:350]))) / 2 / 3.0
        areas_under_curve.append(area)

    # Plot hysteresis curves at various temperatures

    plt.figure(1, figsize=(8, 8))
    data_cutoff = 100
    for mag, temp in zip(plot_magnetisations, plot_temperatures):
        plt.plot(external_field_strengths[data_cutoff:], np.negative(mag[data_cutoff:]), label=f'$T = {temp}$ J$/k_B$')
    plt.xlabel('External Field Strength / J')
    plt.ylabel('Magnetisation fraction')
    plt.title('Hysteresis Curves for varying temperatures of \n Ising Model systems with N=60')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('HYS 1.png')
    plt.show()

    # plotting energy dissipation acquired through area under the hysteresis curve against temperature
    plt.figure(2, figsize=(8, 6))
    plt.plot(temperature_increments, areas_under_curve, 'k')
    plt.xlabel('Temperature of system / $($J$/k_B)$')
    plt.ylabel('Energy lost through dissipation/ J')
    plt.title('Hysteresis curve area against temperature for \n Ising Model systems with N=60')
    plt.xlim([0.2, 3.5])
    plt.savefig('HYS 2.png')
    plt.show()
