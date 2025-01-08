# Dynamic-Causal-Graph-Analysis
Tensorflow-based Implementation of Dynamic Causal Graph Analysis
# Introduction

Causal graphs are powerful tools for understanding cause-effect relationships in dynamic systems, enabling effective intervention strategies in complex and evolving scenarios. The Dynamic Causal Bayesian Optimization (DCBO) was introduced in [1], a framework that combines causal inference, dynamic Bayesian networks, and Gaussian processes to identify optimal interventions in systems with time-evolving causal effects. By leveraging both observational and interventional data, DCBO significantly accelerates the identification of optimal intervention sequences, outperforming traditional methods that overlook temporal and causal dynamics. While [1] focuses on simple causal graphs, real-world scenarios often involve more intricate causal relationships. To address this complexity, this repository examines a causal graph with 15 nodes at each time step consisting 7 non-manipulable variables (e.g., air quality, local weather, community infection rates), 7 manipulable variables (e.g., nurse-to-patient ratio, ICU bed allocation, availability of specialized doctors), and 1 target variable (e.g., patient recovery rates). Non-manipulable variables influence both manipulable variables and recovery rates, while manipulable variables can be adjusted to optimize outcomes. This structure reflects real-world challenges, such as those in hospital settings, where the goal is to maximize patient recovery rates through carefully planned interventions.

<img width="667" alt="image" src="https://github.com/user-attachments/assets/8d61e13a-9aa6-4659-9010-bf40bcc26196" />

# Dependencies
