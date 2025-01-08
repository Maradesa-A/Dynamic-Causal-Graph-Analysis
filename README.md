# Dynamic-Causal-Graph-Analysis
This repository provided step-by-step solutions to the PART1 of the problem 3 (which is based on Causal Dynamic Bayesian Optimization). Please check [docs](docs) folder to see the solution to corresponding problems in PART1. The tensorflow based implementation Aglietti's code is also reported in [Tensor](Tensor), the code is validated by performing both unit and integration test, please check the [Tensor](Tensor) to find [report](report) for code testing.
# Introduction

Causal graphs are powerful tools for understanding cause-effect relationships in dynamic systems, enabling effective intervention strategies in complex and evolving scenarios. The Dynamic Causal Bayesian Optimization (DCBO), a framework that combines causal inference, dynamic Bayesian networks, and Gaussian processes to identify optimal interventions in systems with time-evolving causal effects, was developed by Aglietti $\it {et}  {al}$ [1]. By leveraging both observational and interventional data, DCBO significantly accelerates the identification of optimal intervention sequences, outperforming traditional methods that overlook temporal and causal dynamics. While [1] focuses on simple causal graphs, real-world scenarios often involve more intricate causal relationships. To address this complexity, this repository examines a causal graph with 15 nodes at each time step consisting 7 non-manipulable variables (e.g., air quality, local weather, community infection rates), 7 manipulable variables (e.g., nurse-to-patient ratio, ICU bed allocation, availability of specialized doctors), and 1 target variable (e.g., patient recovery rates). Non-manipulable variables influence both manipulable variables and recovery rates, while manipulable variables can be adjusted to optimize outcomes. This structure reflects real-world challenges, such as those in hospital settings, where the goal is to maximize patient recovery rates through carefully planned interventions.

<img width="667" alt="image" src="https://github.com/user-attachments/assets/8d61e13a-9aa6-4659-9010-bf40bcc26196" />

## Dependencies
- tensorflow
- pygraphviz
- networkx
- itertools
- typing
- unittest/unittest2

## References
1. V. Aglietti, N. Dhir, J. Gonz´alez, Th. Damoulas. Dynamic causal bayesian optimization. Advances in Neural Information Processing Systems, 34:10549–10560, 2021.
  
