# ABC_model_choice
Simple python implementation of Approximate Bayesian Computation for Model Choice (ABC-MC).

All the code in this file is referred to the toy example in the paper "Didelot, Xavier, et al. "Likelihood-free estimation of model evidence." Bayesian analysis 6.1 (2011): 49-76."
Specifically, the toy example experiment is reproduced. 

## Files

- `ABC_MC.py` contains the functions to run ABC-MC algorithm and some utilities. The summary statistics that are used are the ones referred to the above toy example. 
- `model_classes.py` contains the classes defining the models and used in the ABC-MC algorithms. For now, only the Geometric($\mu$) and Poisson($\lambda$) model, with $\mu \sim \text{Uniform}[0,1]$ and $\lambda \sim \text{Exponential}(1)$ are implemented, as in the toy example. 
- `ABC_MC_toy_example.ipynb` contains the results of the simulations. 


