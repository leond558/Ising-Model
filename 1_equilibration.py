import matplotlib.pyplot as plt
import numpy as np
from CheckerboardIsingModel import CheckerboardIsingModel
from MultithreadedIsingModel import MultithreadedIsingModel
import sys
import statsmodels.api as sm

"""
Graphs plotted:
1) Magnetisation against time for T=0.3. Shows some systems reach equilibration whilst some 
    end up with local minima of mis aligned magnetic domains.
2) A plot of the spin configuration of a system that didn't reach equilibration.
3) Plot of how the number of time steps required for equilibration changes with temperature
    for systems with initial alignment of spins and systems that have random initial spins.
"""

if __name__ == '__main__':
    '''
    Investigating the amount of time required for a system to reach equilibration. We plot
    magnetisation against time and then observe the spin configuration for a system that 
    failed to reach equilibration. The spin configuration will be one with bands corresponding 
    to local energy minima that are resistant to complete alignment.
    '''

    # We create 10 systems to investigate equilibration with
    equilibration_models = []
    for i in range(15):
        equilibration_models.append(CheckerboardIsingModel(60, 0, 0.3, False))

    # Using the multithreading approach so that we can evolve the systems in parallel
    systems = MultithreadedIsingModel(equilibration_models)
    systems.simultaneous_time_steps(300)

    # Store the evolved models in a new array
    equilibration_models = systems.model_array

    # Plotting magnetisation against time
    plt.figure(1, figsize=(8, 4))

    # Extracting the models from the array containing the evolved models
    for model in equilibration_models:
        plt.plot(model.magnetisation_array, 'k')

    plt.xlabel('Time / time steps')
    plt.ylabel('Magnetisation fraction')
    plt.title('Magnetisation from an initial unaligned random spin state at $T=0.3J/k_B$')
    plt.savefig('EQU 1.png')
    plt.show()

    # Plotting the spin configuration i.e. the spin lattice pattern for a system that failed to reach
    # equilibrium. We first extract the final magnetisations of all the systems.
    final_magnetisation_array = list(map(lambda m: np.abs(m.magnetisation), equilibration_models))

    # We then order the final magnetisations by the index in final_magnetisation_array, removing indices
    # that have a final magnetisation corresponding to equilibration i.e. those with an absolute value that's
    # not 1.
    indices_of_local_minima = list(
        filter(lambda o: final_magnetisation_array[o] < 1, np.argsort(final_magnetisation_array)))

    if len(indices_of_local_minima) > 0:
        # Plot the state of this system
        plt.figure(2, figsize=(4, 3))

        # We plot the spin configuration of the system with an index first in indices_of_nonequilibrated
        plt.imshow(equilibration_models[indices_of_local_minima[-1]].spin_configuration)
        plt.savefig('EQU 2.png')
        plt.show()

    else:
        print(f"No cases of systems in local minima. All systems have perfect spin alignment", file=sys.stderr)

    """We now consider how the number of time steps required for a system to equilibrate changes with temperature. 
    Given the nature of the CheckerboardIsingModel class, we can consider this for both states when the initial system 
    has spins all aligned and the case when the spins are in an initial random pattern. 
    
    In order to determine whether a system is equilibrated, we compare the root mean square fluctuations of the system
    to the mean magnetisation and standard deviation of an equivalent system that is at equilibrium.
    """

    # Initially define parameters corresponding to the size of the systems being investigated and
    # the number of temperature increments being considered
    lattice_size = 60
    plot_points = 50

    # Forms an array of the temperature increments being considered with the increments being linearly
    # spaced apart
    temperature_array = np.linspace(0, 3, plot_points)
    deviation_models = []

    # First we are trying to find the mean and standard deviation of systems at equilibrium:
    # Create a list of systems, initiated with the linearly spaced temperature increments
    for i, temperature in enumerate(temperature_array):
        # Can change whether aligned or not with the aligned boolean parameter
        deviation_models.append(CheckerboardIsingModel(lattice_size, 0, temperature, False))

    # Evolve the systems in parallel
    deviation_systems = MultithreadedIsingModel(deviation_models)
    deviation_systems.simultaneous_time_steps(400)
    deviation_models = deviation_systems.model_array

    # Determine the mean and standard deviation magnetisation of a system at equilibrium. Only consider magnetisations
    # past the 30th index as initial magnetisation values are volatile
    magnetisation_std = np.array([np.std(model.magnetisation_array[30:]) for model in deviation_models])
    magnetisation_mean = np.array([np.abs(np.mean(model.magnetisation_array[30:])) for model in deviation_models])

    # Now we run a high number of repeats for each system and find the number of time steps for equilibrium
    # Equilibrium of the system is determined by a certain criteria
    repeats = 1000
    time_steps = 100
    median_equilibration_time = np.zeros(plot_points)
    equilibration_percentage = np.zeros(plot_points)
    # The number of time steps that each system should be evolved through depends on the temperature.
    # It would be pointless running through high number of time steps for a system that equilibrates very quickly,
    # this would constitute wasted computational resources.
    for i, temperature in enumerate(temperature_array):
        equilibration_time_models = []

        # A suitable number of time steps were determined through experimenting.
        if temperature < 1:
            time_steps = 100

        if 1 <= temperature < 2:
            time_steps = 300

        if 2 <= temperature < 2.2:
            time_steps = 900

        # Systems with a temperature greater than 2.2, reach equilibrium very quickly
        if temperature >= 2.2:
            repeats = 5
            time_steps = 20

        # Evolve the systems for each temperature value for the number of repeats
        for _ in range(repeats):
            equilibration_time_models.append(CheckerboardIsingModel(lattice_size, 0, temperature, False))

        equilibration_time_systems = MultithreadedIsingModel(equilibration_time_models)
        equilibration_time_systems.simultaneous_time_steps(time_steps)
        equilibration_time_models = equilibration_time_systems.model_array

        # An array that stores the equilibration times for systems with that temperature values
        equilibration_time_array = []

        # Determine whether the model reaches equilibration and the associated value
        for model in equilibration_time_models:
            equilibration_marker = False
            time_for_equilibration = None

            # For the equilibration criteria, we consider across a time frame of t to t+14.
            for j in range(14, time_steps, 1):
                fourteen_magnetisation_array = np.array(model.magnetisation_array[j - 14:j])
                fourteen_standard_dev = np.std(fourteen_magnetisation_array)
                fourteen_mean = np.mean(fourteen_magnetisation_array)

                # Comparing the root-mean-square fluctuations within this time frame to the
                # magnetisation mean and standard deviation of a system already in
                # equilibrium
                if (fourteen_standard_dev < 2 * magnetisation_std[i]) & (
                        magnetisation_mean[i] - magnetisation_std[i] < fourteen_mean) & (
                        fourteen_mean < magnetisation_mean[i] + magnetisation_std[i]):
                    equilibration_marker = True
                    time_for_equilibration = j - 14
                    break

            # If the system indeed reaches equilibrium, store the time in which it reached it
            if equilibration_marker:
                equilibration_time_array.append(time_for_equilibration)

        # Store the equilibration times and percentages
        equilibration_time_array = np.asarray(equilibration_time_array)
        median_equilibration_time[i] = np.median(equilibration_time_array)
        equilibration_percentage[i] = (len(equilibration_time_array) / repeats) * 100

    # Due to te finite number of data points collected and the nature of matplotlib, plotting a graph
    # without some element of smoothening looks very jagged. Thus, we smooth the results somewhat.
    # The appropriate fraction of smoothening was determined through experimentation.
    smoothened_non_aligned = sm.nonparametric.lowess(median_equilibration_time, temperature_array, frac=0.4)

    # Plotting the figure
    plt.figure(3, figsize=(6, 4))
    plt.plot(smoothened_non_aligned[:, 0], smoothened_non_aligned[:, 1], color='red', ls='-',
             label='Initially non-aligned')
    plt.xlabel('Temperature / $($J$/k_B)$')
    plt.ylabel('Time to equilibrate / time steps')
    plt.title('Number of time steps for equilibration against temperature')
    # Tight layout prevents the axes titles from being cropped out
    plt.tight_layout()
    plt.legend(loc='lower left')
    plt.savefig('EQU 3.png')
    raise SystemExit(0)
