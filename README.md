# Ising-Model-BA-Natural-Sciences-Computing-Project
This repository contains a Python program that simulates a statistical mechanics model of ferromagnetism known as the Ising model. The Ising model is a lattice-based model that describes the interactions between spins in a magnetic material, and it has applications in a variety of fields, from condensed matter physics to neuroscience.

The main goal of this project was to develop an efficient and flexible program for simulating the Ising model and analyzing its properties. To achieve this, I implemented a specialized Monte Carlo algorithm that uses multithreading and vectorization to speed up the simulations and reduce the computational time. I also used clever data structures to optimize the memory usage and facilitate the analysis of the simulation results.

One important application of this program is in the study of the biological interactions of neurons in the brain. By modeling the activity of neurons as spins in an Ising-like system, we can use statistical mechanics and machine learning techniques to analyze the data and gain insights into the underlying mechanisms of brain function. This approach has been shown to be useful in Parkinson's research, where abnormal synchronization of neural activity is a hallmark of the disease.

Parkinson's Ising Model Paper:
Kusmartsev, V.F., Zhang, W., Kusmartseva, A.F., Balanov, A.G., Janson, N.B. and Kusmartsev, F., 2017. Ising model–an analysis, from opinions to neuronal states.

Another application is in modelling the stock market and herding behaviours. This use is inspired by the following papers:
https://iopscience.iop.org/article/10.1088/1742-6596/1113/1/012009/pdf
https://guava.physics.uiuc.edu/~nigel/courses/563/Essays_2005/PDF/wu1.pdf
https://www.sciencedirect.com/science/article/abs/pii/S0378437116303132

To demonstrate the capabilities of the program, I provide several examples of Ising model simulations and analysis. I show how to generate thermalized configurations of the system, compute thermodynamic properties such as the energy, magnetization, and specific heat, and perform linear regression to extract useful information from the data. We also compare our program's performance with other existing Ising model simulations and show that our implementation achieves a 2.7x speed increase on previous such programs.

<img width="383" alt="Screenshot 2023-10-13 at 21 36 13" src="https://github.com/leond558/Ising-Model-Project/assets/113116336/ef322a99-d212-4538-8ad2-bce5243d2c81">

<img width="349" alt="Screenshot 2023-10-13 at 21 36 24" src="https://github.com/leond558/Ising-Model-Project/assets/113116336/61652a77-3f81-4490-9c67-63c29e11e729">

All analyses and visualisations can be seen in the paper in the repository.
