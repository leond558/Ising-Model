# Blind Grade Number: 6905T

import numpy as np
import sys


class CheckerboardIsingModel:
    '''
    Implementation of the Ising Model with associated calculations of states and properties using a checkerboard
    variant of  the Markov Chain Monte Carlo method known as the Metropolis Algorithm. The algorithm has been
    simplified using the benefits of vectorised computation.

    Note that the arguments passed to the constructor are in a form of reduced units and have factors of the exchange
    energy, the magnetic moment and the Boltzmann constant taken out.

    Args passed to the constructor:
        N (int): the total number of lattice points along one axis, thus the system has N^2 total lattices
        h_tilde (float): the strength of the external magnetic field
        T_tilde (float): the temperature of the system

    Attributes:
        N (int): the total number of lattice points along one axis, thus the system has N^2 total lattices
        h_tilde (float): the strength of the external magnetic field T_tilde (float): the temperature of the system
        energy (float): the energy of the system evaluated at the current time step in the reduced units defined above
        energy_array (float[]): array of energies calculated at each time step
        magnetisation (float): the magnetisation of the system evaluated at the current time step in the reduced units
            defined above
        magnetisation_array (float[]): array of the magnetisations calculated at each time step
        spin_configuration (np.int[]): matrix of the spins of each lattice point
        time_step_counter (int): an integer counter containing the number of time steps that the
            system has undergone evolution for
        checkerboard_pattern (string[]) : matrix of the parities of the lattice points, serving as a mesh
        store_configuration (bool): if set to true, stores each configuration in an array
        configuration_array ([np.int[]]): store of the spin configurations at each time step


    Methods:
        progress_time_step(): progresses the system forward by a single time step
        compute_properties(): calculates the magnetisation and energy of the system at that time step
        determine_neighbours(): returns a tuple of the neighbours of a lattice element ordered
            according to the order of the cardinal directions
    '''

    # Creating the constructor for the class
    def __init__(self, N: int, h_tilde: int, T_tilde: float, aligned: bool = True, store_configuration: bool = False):
        '''
        Args: N (int): the total number of lattice points along one axis, thus the system has N^2 total lattices
        h_tilde (float): the strength of the external magnetic field
        T_tilde (float): the temperature of the system
        aligned (bool): boolean indicating whether the initial spin configuration should be all spins aligned (True) or
                    randomly aligned (False). Default is aligned.
        store_configuration (bool): boolean flag indicating whether to store each spin configuration at each
                                    time step; this flag exists as storing each configuration would be very
                                    computationally intensive and require a large amount of RAM, thus for this
                                    reason, it is automatically set to False.

        '''
        # Creating the fields for the properties / attributes of the object
        self.h_tilde = h_tilde
        self.T_tilde = T_tilde
        self.N = N
        self.spin_configuration = np.ones((N, N), dtype=int)
        self.energy = None
        self.energy_array = []
        self.magnetisation = None
        self.magnetisation_array = []
        self.time_step_counter = 0
        self.checkerboard_pattern = None
        self.store_configuration = store_configuration
        self.configuration_array = []

        # The code is written to only handle lattices with even dimensions for lattice number on either two-dimensional
        # axis. To account for this, an Exception is thrown in the case N is odd. This is also logged. N is then changed
        # to the evn number 1 higher.
        if N % 2 != 0:
            self.N += 1
            print(f"Odd N input. Exception thrown.", file=sys.stderr)
            raise Exception("This implementation of the Metropolis algorithm only handles even lattice dimensions. ")

        # Start on a random alignment if aligned is set to false
        if not aligned:
            self.spin_configuration = np.random.choice([-1, 1], (self.N, self.N)).astype(int)

        # Create the checkerboard pattern to speed up the Metropolis Algorithm by flipping same parity simultaneously
        self.checkerboard_pattern = np.tile([["odd", "even"], ["even", "odd"]], (int(self.N / 2), int(self.N / 2)))

        # Calculate properties at the current time step of the simulation
        self.compute_properties()

    def determine_neighbours(self):
        '''
        Function to determine the neighbours of a lattice point
        :return: A tuple containing two elements. aligned_spins = the number of aligned spins for all lattices totaled.
                And aligned_spin_matrix = matrix where value at each lattice point is -2x the number of adjacent aligned
                spins.
        '''

        # The bitwise operation for determining whether two lattice points are aligned is the XOR gate
        # which can be implemented using the numpy bitwise functions. The lattice points share the same
        # spin if the resultant matrix position has a 0 element and a different spin with a -2 element.
        # np.roll allows us to get to the correct direction of neighbour using matrix shift parameters.

        north_neighbour = np.bitwise_xor(self.spin_configuration, np.roll(self.spin_configuration, 1, 0))
        east_neighbour = np.bitwise_xor(self.spin_configuration, np.roll(self.spin_configuration, -1, 1))
        south_neighbour = np.bitwise_xor(self.spin_configuration, np.roll(self.spin_configuration, -1, 0))
        west_neighbour = np.bitwise_xor(self.spin_configuration, np.roll(self.spin_configuration, 1, 1))

        # Need a matrix that enumerates the number of adjacent aligned spins for each individual lattice element
        # Have to first modify the format of neighbour matrices such that non-aligned are 0. Instead of checking each
        # element against a criteria and changing it, we can do vectorised arithmetic which is quicker.
        # Hence, an aligned spin is now represented by a +2 element.
        north_neighbour = north_neighbour + 2
        east_neighbour = east_neighbour + 2
        south_neighbour = south_neighbour + 2
        west_neighbour = west_neighbour + 2

        # Find the number of points that have aligned spins. +2 indicates spin alignment
        aligned_spins = np.count_nonzero(north_neighbour) + np.count_nonzero(east_neighbour) \
                        + np.count_nonzero(south_neighbour) + np.count_nonzero(west_neighbour)

        aligned_spin_matrix = north_neighbour + east_neighbour + south_neighbour + west_neighbour

        return aligned_spins, aligned_spin_matrix

    def compute_properties(self):
        """
        Computes the energy and magnetisation of the lattice.
        :return: Nothing. Modifies the values of energy and magnetisation of the object and appends these values
                to the corresponding arrays containing those values.
        """
        # Magnetisation calculation and appending to the array of magnetisation at each time step
        self.magnetisation = (2 * np.count_nonzero((self.spin_configuration + 1)) - self.N ** 2) / self.N ** 2
        self.magnetisation_array.append(self.magnetisation)

        # Energy calculation using the formula of the Ising Model

        # Fetch neighbours
        aligned_spins = self.determine_neighbours()[0]
        anti_aligned_spins = 4 * (self.N ** 2) - aligned_spins

        positive_spin = np.count_nonzero(self.spin_configuration == 1)
        negative_spin = self.N ** 2 - positive_spin

        exchange_value = anti_aligned_spins - aligned_spins
        magnetic_value = self.h_tilde * (positive_spin - negative_spin)

        self.energy = (exchange_value + magnetic_value) / self.N ** 2

        # Energy calculation and appending to the array of energies at each time step
        self.energy_array.append(self.energy)

    def progress_time_step(self, time_steps: int):
        '''
        Progresses the system through a provided number of time steps. A time step is taken when the spins are
        flipped for both parities.

        :param time_steps: int, number of time steps to progress the system through

        :return: Nothing. The CheckerboardIsingModel object evolved through the specified number of time steps with
                the fields of the object updated accordingly.
        '''
        # Do the necessary evolution actions for the specified number of time steps
        for _ in range(time_steps):

            # If the store_configuration marker is set to True, store the configuration as appropriate
            if self.store_configuration:
                self.configuration_array.append(self.spin_configuration.copy())

            # Creating a matrix of probabilities to use in determining whether a random flip in spin occurs
            probabilities = np.random.random((self.N, self.N))

            # One time step involves two processes, considering both even and odd parities.
            parities = ["odd", "even"]

            # Iterate across the two parities
            for parity in parities:
                aligned_spin_matrix = self.determine_neighbours()[1]

                # Note here the factor in front of the aligned spins is 2 instead of 4, this is because the
                # aligned_spin_matrix represents an aligned neighbour as a value of 2
                # !!!!!!!!!!!!!!!!!!
                energy_change = 2 * self.h_tilde - (self.spin_configuration+1) * 2 * self.h_tilde + 2 * aligned_spin_matrix - 8

                # Determining whether the spin of the lattice point should be flipped
                if self.T_tilde > 0:
                    flip_spin_matrix = (np.exp(-energy_change / self.T_tilde) > probabilities).astype(np.bool)
                else:
                    flip_spin_matrix = (energy_change < 0).astype(np.bool)

                # Removing instances where a flip is required for lattice elements of the parity not currently being
                # considered
                if parity == 'odd':
                    flip_spin_matrix = np.bitwise_and(flip_spin_matrix, self.checkerboard_pattern == 'odd')
                else:
                    flip_spin_matrix = np.bitwise_and(flip_spin_matrix, self.checkerboard_pattern == 'even')

                # Flipping the spins accordingly
                flip_spin_matrix = flip_spin_matrix.astype(np.int) * -1
                flip_spin_matrix[flip_spin_matrix == 0] = 1
                self.spin_configuration = flip_spin_matrix * self.spin_configuration

                # Troubleshooting the contents of the spin configuration field, should only be populated with 1 and -1
                if np.count_nonzero(self.spin_configuration + 2) != self.N ** 2:
                    print(f"Spin configuration is populated by something other than 1,-1s", file=sys.stderr)
                    raise Exception("Error in the spin configuration field of the system.")

            # Update the time, energy and magnetisation fields of the object
            self.time_step_counter += 1
            self.compute_properties()