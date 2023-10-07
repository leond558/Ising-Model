# Blind Grade Number: 6905T

import multiprocessing
import sys
import numpy as np
from CheckerboardIsingModel import CheckerboardIsingModel

class MultithreadedIsingModel:
    '''
    A class that enables for multiple Ising Model systems to be evolved through and considered simultaneously
    across several processor cores in parallel. This is necessary as computing systems sequentially for large
    time steps is computationally intensive. Running these computations with one model per thread allows for
    significant speed up at higher time step values.

     Args passed to the constructor:
        model_array ([CheckerboardIsingModel]): an array of the Ising Model systems that are to be evolved
                                        using multithreading and hence run in parallel by different processor threads

    Attributes:
        model_array ([CheckerboardIsingModel]): an array of the Ising Model systems that are to be evolved
                                        using multithreading and hence run in parallel by different processor threads
        number_of_models (int): the total number of models being evolved in parallel through multithreading,
                                equal to the length of model_array
    '''

    def __init__(self, model_array: [CheckerboardIsingModel]):
        '''
        The constructor that allows for parallel processing of the time evolution of the models.
        :param model_array: the models that are to be time-evolved in parallel
        '''

        # It is only efficient to use the multithreading class if multiple models are to be
        # investigated. Thus, an exception is thrown and an error is logged in the
        # instance that only a singular model is passed to the constructor

        single_system_flag = False
        if not (isinstance(model_array, list) or isinstance(model_array, np.ndarray)):
            single_system_flag = True

        if not single_system_flag:
            self.number_of_models = len(model_array)
            if self.number_of_models < 2:
                single_system_flag = True

        if single_system_flag:
            # Log an error
            print(f"Using multithreading class with only one system. This is inefficient! "
                  f"Only use MultithreadedIsingModel with multiple CheckerboardIsingModel objects!!!"
                  f"Exception thrown.", file=sys.stderr)

            # Throw an exception
            raise Exception("Do not use MultithreadingIsingModel with one model only.")

        # Set and calculate the necessary class properties accordingly
        self.model_array = model_array
        self.number_of_models = len(model_array)

    # Function that does the parallel time evolution of the models
    def simultaneous_time_steps(self, time_steps: int):
        '''
        Runs the multithreading_time_step in parallel with the models stored in model_array by
        using the multiprocessing package and the Pool() function.
        :param time_steps: the number of time steps the different models should be evolved by
        :return: updates model_array to contain the updated and evolved models
        '''

        # Use the multiprocessing package to evolve models in parallel
        # Pool represents a pool of worker processes that are handled simultaneously by different processors
        with multiprocessing.Pool() as pool:
            # Create a new array of the outputs of passing the models in model_array in parallel to the
            # multithreading_time_step function
            updated_models = pool.starmap(multithreading_time_step, [(model, time_steps) for model in self.model_array])
            updated_models = list(updated_models)

            # Update the field model_array with the new evolved models
            self.model_array = []
            for i in range(self.number_of_models):
                self.model_array.append(updated_models[i])



def multithreading_time_step(model: CheckerboardIsingModel, time_steps: int):
    '''
    A helper function that takes a CheckerboardIsingModel object and a specified number of time steps and
    evolves the lattice contained in the object by the time steps. This is defined non-locally as this is
    required for the starmap function. Multiple models are passed through this function concurrently to
    allow for multithreading of model evolution. This is the function run simultaneously across multiple
    cores.
    :param model: CheckerboardIsingModel,the model to be evolved through by the time steps
    :param time_steps: int, the integer number of time steps
    :return: The updated model object progressed through by the time steps specified.
    '''
    model.progress_time_step(time_steps)
    return model
