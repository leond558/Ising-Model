import pickle as pkl
import time

import matplotlib.pyplot as plt
import numpy as np
from CheckerboardIsingModel import CheckerboardIsingModel
from MultithreadedIsingModel import MultithreadedIsingModel
import statsmodels.api as sm
"""
Code that serialises using pickle, some test results for the evolution of an Ising model system for different values of 
N and different temperatures. Used to experiment and test.
"""
if __name__ == '__main__':

    determine_time = True
    load = False
    lattice_size_list = []
    temperature_increments = []
    magnetisations = []
    energies = []

    if determine_time:
        initial = time.time()
        test_models = []
        for i in range(100):
            test_models.append(CheckerboardIsingModel(60, 0, 0.3, False))
        systems = MultithreadedIsingModel(test_models)
        systems.simultaneous_time_steps(100)
        test_models = systems.model_array
        final = time.time()

        print(final-initial)

    else:
        if load:
            with open('testing.pkl', 'rb') as f:
                lattice_size_list, temperature_increments, magnetisations, energies = pkl.load(f)

        else:
            temperature_increments = np.unique(
                np.concatenate([np.linspace(0, 1.5, 100), np.linspace(1.5, 2.5, 400), np.linspace(2.5, 3.5, 100)]))
            lattice_size_list = [10, 40, 60, 100]
            magnetisations = []
            energies = []

            for i, lattice_size in enumerate(lattice_size_list):
                print("here")
                test_models = []

                for j, temperature in enumerate(temperature_increments):
                    test_models.append(CheckerboardIsingModel(lattice_size, 0, temperature, False))
                systems = MultithreadedIsingModel(test_models)

                systems.simultaneous_time_steps(1000)
                test_models = systems.model_array

                magnetisation_array = []
                energy_array = []
                for model in test_models:
                    magnetisation_array.append(model.magnetisation_array)
                    energy_array.append(model.energy_array)
                magnetisations.append(magnetisation_array)
                energies.append(energy_array)

            with open('testing.pkl', 'wb') as f:
                pkl.dump([lattice_size_list, temperature_increments, magnetisations, energies], f)

        plt.figure(1, figsize=(8, 4))

        for i, lattice_size in enumerate(lattice_size_list):
            if i > 0:
                break
            for j, temperature in enumerate(temperature_increments):
                if j > 100:
                    break
                plt.plot(magnetisations[i][j], 'k')

        plt.show()

