# Projects 

In the [Overview talk]() we presented you three different methods that combine dynamicals systems with machine learning. Here, we have a lot of material for you to get started with them. 

## Neural Differential Equations 

[In this notebook](NeuralDifferentialEquations/NeuralDifferentialEquations.ipynb) and its accompyning material, we will get into Neural Differential Equations that integrate Artificial Neural Networks directly into the right hand side of a differential equations. This can be e.g. used to setup a hybrid model that combines prior knowledge with data-driven methods such as ANNs! For the project you'll learn how to use those to approximate chaotic system. You can of course also apply them to actual data. 

## Reservoir Computing 

[In the reservoir notebook](ReservoirComptuing/reservoir_computing.ipynb) we will use `ReservoirComputing.jl` for approximation dynamical system. The approach works very well for low-dimensional chaotic system and is computational less demanding than other ANNs. Apply it to a dynamical system of your choice or some your research data. 

## SINDy + Neural Differential Equations

[In the SINDy notebook](SINDy/SINDy.ipynb) we introduce SINDy to reconstruct the governing equations of dynamical systems from data. We also give you glimpse into genetic symbolic regression methods. There's also the possibility to combine these approaches with Neural Differential Equations to first fit a Neural DE to data and then use this as an input for a symbolic regression method. 
