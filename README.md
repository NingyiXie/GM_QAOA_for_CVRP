# GM_QAOA_for_CVRP

## Introduction
This is an implementation of Grover-Mixer Quantum Alternating Operator Ansatz (GM-QAOA) to address the capacitated vehicle routing problem (CVRP). 

We reformulate the CVRP and use a conditional objective function that allows us to apply the Grover mixer, and restrict the search space on the feasible solution set. For more details, refer to the manuscript ["A Feasibility-Preserved Quantum Approximate Solver for the Capacitated Vehicle Routing Problem"](https://arxiv.org/abs/2308.08785).

We offer a usage demo in `demo.ipynb`

## Requirements
* Qiskit 0.43.0
* Numpy 1.23.4
* Scipy 1.9.3
* Sympy 1.11.1
