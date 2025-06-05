# Pandemic Dark Matter Boltzmann solver 

#### Author: This code was originally written in the project https://iopscience.iop.org/article/10.1088/1475-7516/2013/10/044, and further developed by me, antonabr@uio.no. 

Major parts of this code served as my master thesis at the University of Oslo. 

Link to thesis: *link*

In this folder, you find several projects including 
 * single scalar theory
 * single dark photon theory
 *  dark photon and dark Higgs theory

The latter which contains the main results of the thesis.

## Structure
This code works as follows: 

* To run a single simulation, run "sterile_caller.py". 
* To run a full parameter scan, run "find_y.py".

The main assumption made for the code to work is dark sector equilibrium kept by $\nu_s \nu_s \leftrightarrow \Phi$ interactions.
We then use equilibrium phase-space distributions, and only need to solve for one of the chemical potentials, e.g. $\mu_s$, and the common dark sector temperature $T_d$. Hence, we only need to solve two Boltzmann equations -- the one for $n_d = n_s + 2n_X + 2n_h$ and $\rho_d = \rho_s + \rho_X + \rho_h$. We refer to these equations as "the coupled Boltzmann system". 

The coupled Boltzmann system is solved in "pandemolator.py". This file consists of two classes: 
* "TimeTempRelation"

which translates cosmological time into Standard Model temperature, and
* "Pandemolator"

which solved the coupled Boltzmann equations. 

The files "C_res_scalar.py" and "C_res_vector.py" implements the collision operators, and "scalar_mediator.py" and "vector_mediator.py" contains cross sections, decay rates, and matrix elements for 2-to-2 processes. The matrix elements for decay rates are in "sterile_caller.py". 
