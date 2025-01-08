# Dynamic-Causal-Graph-Analysis

This repository provides step-by-step solutions to Part 1 of Problem 3, which focuses on Causal Dynamic Bayesian Optimization (DCBO). Detailed solutions to the corresponding problems in Part 1 can be found in the [docs](docs) folder. Following the code provided in [1], the tensorflow-based version was developed as reported in [Analyze_Causal_Graph](Analyze_Causal_Graph) folder. The code has been rigorously tested through both unit and integration tests. The results of these tests, along with the codes and reports, can be found in the [Test](Test) folder. 

# Introduction

Causal graphs are powerful tools for understanding cause-effect relationships in dynamic systems [1], enabling effective intervention strategies in complex and evolving scenarios. The Dynamic Causal Bayesian Optimization (DCBO), a framework that combines causal inference, dynamic Bayesian networks, and Gaussian processes to identify optimal interventions in systems with time-evolving causal effects, was developed by Aglietti $\it {et}  {al}$ [2]. By leveraging both observational and interventional data, DCBO significantly accelerates the identification of optimal intervention sequences, outperforming traditional methods that overlook temporal and causal dynamics. While [2] focuses on simple causal graphs, real-world scenarios often involve more complex causal relationships. This repository examines more complex causal graph with 15 nodes at each time step consisting 7 non-manipulable variables (e.g., air quality, local weather, community infection rates), 7 manipulable variables (e.g., nurse-to-patient ratio, ICU bed allocation, availability of specialized doctors), and 1 target variable (e.g., patient recovery rates). Non-manipulable variables influence both manipulable variables and recovery rates, while manipulable variables can be adjusted to optimize outcomes. This structure reflects real-world challenges, such as those in hospital settings, where the goal is to maximize patient recovery rates through carefully planned interventions. 

<img width="667" alt="image" src="https://github.com/user-attachments/assets/8d61e13a-9aa6-4659-9010-bf40bcc26196" />

## Dependencies
- tensorflow
- pygraphviz
- networkx
- itertools
- unittest/unittest2

## References
1. V. Aglietti, X. Lu, A. Paleyes, and Javier Gonz´alez. Causal bayesian optimization. In International
  Conference on Artificial Intelligence and Statistics, pages 3155–3164. PMLR, 2020
2. V. Aglietti, N. Dhir, J. Gonz´alez, Th. Damoulas. Dynamic causal bayesian optimization. Advances in Neural Information Processing Systems, 34:10549–10560, 2021.
3. V. Aglietti, X. Lu, A. Paleyes, and Javier Gonz´alez. Causal bayesian optimization. In International
 Conference on Artificial Intelligence and Statistics, pages 3155–3164. PMLR, 2020
  
