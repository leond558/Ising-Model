import matplotlib.pyplot as plt
import numpy as np
from CheckerboardIsingModel import CheckerboardIsingModel
from MultithreadedIsingModel import MultithreadedIsingModel
import statsmodels.api as sm

"""
Graphs plotted:
1) The autocorrelation function is plotted for different temperatures in the case of N=100.
2) The decorrelation time is plotted against temperatures for different lattice sizes.

"""
if __name__ == '__main__':
    """
    Investigating the autocorrelation function and then decorrelation time using the magnetisations
    of the systems. We observe behaviour that's expected by a ferromagnetic Ising Model system, principally 
    exponential decay of the autocorrelation function and a peak in the decorrelation time 
    around the critical temperature.
    """

    # Define the lists to contain the results calculated for the autocorrelation function and the decorrelation times
    autocorrelation = []
    decorrelation_times = []

    # Sizes of the systems that are to be considered
    lattice_size_list = [10, 40, 120]
    time_lags = np.arange(250).astype(int)

    # Behaviour of the system is more volatile around the critical temperature and thus, more increments are needed
    # around the critical temperature. np.unique prevents overlap of temperature values
    temperature_increments = np.unique(
        np.concatenate([np.linspace(0, 1.5, 100), np.linspace(1.5, 2.5, 400), np.linspace(2.5, 3.5, 100)]))

    for i, lattice_size in enumerate(lattice_size_list):
        print("here")
        correlation_models = []

        # Evolving systems with the necessary parameters in parallel
        for j, temperature in enumerate(temperature_increments):
            correlation_models.append(CheckerboardIsingModel(lattice_size, 0, temperature, False))
        systems = MultithreadedIsingModel(correlation_models)

        # 1000 time steps is a sufficient number to enable both equilibration and the computation of the auto
        # correlation function which needs a sufficient number of time lags for accuracy
        systems.simultaneous_time_steps(1000)
        correlation_models = systems.model_array

        # Extracting the magnetisations for all the systems
        magnetisation_array = []
        for model in correlation_models:
            magnetisation_array.append(model.magnetisation_array)
        magnetisation_array = np.asarray(magnetisation_array)

        # Computing the autocorrelation function
        # Assume that systems equilibrate in 150 steps and using 250 time lags
        time_to_equilibrate = 150
        auto_variance = np.zeros((len(time_lags), len(temperature_increments)))

        for j, time_lag in enumerate(time_lags):
            # Implementing the formula for the autocorrelation function as given in the manual
            mean_mag = np.mean(magnetisation_array[:, time_to_equilibrate:], axis=1).reshape((-1, 1))
            auto_variance[j, :] = np.mean((magnetisation_array[:, time_to_equilibrate + time_lag:] - mean_mag) * (
                    magnetisation_array[:, time_to_equilibrate:magnetisation_array.shape[1] - time_lag] - mean_mag),
                                          axis=1)

        # Populating autocorrelation list
        auto_variance = auto_variance / auto_variance[0, :]
        autocorrelation.append(auto_variance)

        print("done")

        # Determining the decorrelation times
        reciprocal_e_times = np.zeros(len(temperature_increments))
        for j in range(len(temperature_increments)):
            # Finding the time lag in which the autocorrelation function falls to 1 over e
            # 0.3679 is equivalent to 1/e to 4 decimal places
            reciprocal_e_times[j] = np.argmax(auto_variance[:, j] < 0.3679)
            # ????
            if reciprocal_e_times[j] == 0 and 2 < temperature_increments[j] < 3:
                reciprocal_e_times[j] = reciprocal_e_times[j - 1]
        decorrelation_times.append(reciprocal_e_times)

    print(autocorrelation)

    # Only plotting select temperatures out of all of those used in the calculation of the autocorrelation function
    plot_temperatures = [1.5, 1.7, 2.1, 2.2, 2.4, 2.7]
    plot_auto_correlations = []
    # Extracting the autocorrelation function values corresponding to the desired temperatures
    for cut_off_index in [np.argmax(temperature_increments > temperature) for temperature in plot_temperatures]:
        plot_auto_correlations.append(
            (np.mean(autocorrelation[-1][:, cut_off_index - 3:cut_off_index + 3], axis=1)[0:120]))
    # List of RGB values for the colours used in t he plotting
    colours = [(0, 0, 0), (1, 1, 0), (0, 1, 1), (1, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5)]

    # Plotting the autocorrelation function for various temperatures
    plt.figure(1, figsize=(6, 4))
    for (auto, temp, colour) in zip(plot_auto_correlations, plot_temperatures, colours):
        plt.plot(time_lags[0:120], auto, label=f'$T = {temp:0.1f}$ J$/k_B$', color=colour)
    plt.xlabel('Time Lag / time steps')
    plt.ylabel('Autocorrelation')
    plt.legend()
    plt.title('Autocorrelation Functions for Different Temperatures')
    plt.tight_layout()
    plt.savefig('AUTO 1 test.png')

    # Plotting the decorrelation time against time lags for different lattice sizes
    plot_temperatures = np.linspace(0, 3.5, len(temperature_increments))
    colours = [(0, 0, 1), (1, 0, 0), (0, 0, 0)]
    plt.figure(2, figsize=(6, 4))
    for (time, size, colour) in zip(decorrelation_times, lattice_size_list, colours):
        # Applying some smoothening to the graph
        smoothened_time = sm.nonparametric.lowess(time, plot_temperatures, frac=0.2)
        plt.plot(smoothened_time[:, 0], smoothened_time[:, 1], label=f'$N = {size}$', color=colour)
    plt.xlabel('Temperature / $($J$/k_B)$')
    plt.ylabel('Decorrelation time / time steps')
    plt.title('Decorrelation time against temperature of the system')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('AUTO 2 test.png')
