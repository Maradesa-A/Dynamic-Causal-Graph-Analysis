import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import OrderedDict
import pygraphviz
from networkx.drawing import nx_agraph
from numpy.random import seed
from graphviz import Source
from numpy import repeat
# from itertools import cycle, chain
from networkx import MultiDiGraph
from typing import List, Union
from copy import deepcopy
from itertools import chain, combinations
from typing import Callable, Dict, Tuple
from tqdm import trange
from typing import Iterable, OrderedDict, Tuple
from numpy.core import hstack, vstack

from tqdm import trange
import pickle

import tensorflow as tf
# from collections import OrderedDict
# from typing import Callable
import networkx as nx
from networkx import MultiDiGraph

from itertools import chain, combinations, cycle

seed(seed=0)


def optimal_sequence_of_interventions(
    exploration_sets: List,
    interventional_grids: Dict,
    initial_structural_equation_model: Callable,
    structural_equation_model: Callable,
    graph: MultiDiGraph,
    time_steps: int = 3,
    model_variables: List = None,
    target_variable: str = None,
    task: str = "min",
) -> Tuple:
    """
    Determines the optimal sequence of interventions in a dynamic causal graph.

    Args:
        exploration_sets : (List) A list of variable sets to explore for interventions.
        interventional_grids : (Dict) A dictionary mapping each exploration set to its corresponding intervention grid.
        initial_structural_equation_model : (Callable) A function representing the static SEM.
        structural_equation_model : (Callable) A function representing the dynamic SEM.
        graph : (MultiDiGraph) The causal graph.
        time_steps : (int, optional) Number of time steps for the intervention sequence, by default 3.
        model_variables : (List, optional) List of model variables, by default None.
        target_variable : (str, optional) The variable to optimize, by default None.
        task : (str, optional) Optimization goal, either "min" or "max", by default "min".

    Returns:
    
        Tuple
        A tuple containing:
        best_s_values : (List) The best intervention values for each time step.
        best_s_sequence : (List) The sequence of exploration sets leading to the best outcome.
        best_objective_values : (List) The objective values corresponding to the best interventions.
        y_stars : (Dict) The computed outcomes for each intervention at each time step.
        optimal_interventions : (Dict) The optimal intervention values for each exploration set and time step.
        all_CE : (List) The cumulative causal effects computed during the optimization.

    Raises
    ------
    ValueError
        If an exploration set does not have a corresponding intervention grid.
    AssertionError
        If the target variable is not provided or is not in the model variables.
    """
    assert target_variable and target_variable in (model_variables or []), \
        "Target variable must be specified and part of the model variables."

    # Check if all exploration sets have corresponding grids
    missing_sets = [s for s in exploration_sets if s not in interventional_grids]
    if missing_sets:
        raise ValueError(f"Missing grids for exploration sets: {missing_sets}")

    # Initialize variables
    static_noise_model = {k: tf.zeros(time_steps) for k in (model_variables or ["X", "Z", "Y"])}
    range_t = range(time_steps)
    shift_range_t = range(time_steps - 1)
    best_s_sequence, best_s_values, best_objective_values = [], [], []

    optimal_interventions = {setx: [None] * time_steps for setx in exploration_sets}
    y_stars = deepcopy(optimal_interventions)
    all_ce = []
    blank_intervention_blanket = make_sequential_intervention_dict(graph, time_steps)

    for t in range_t:
        ce = {es: [] for es in exploration_sets}

        for s in exploration_sets:
            assert s in interventional_grids, f"Missing grid for exploration set: {s}"
            intervention_blanket = deepcopy(blank_intervention_blanket)

            if t > 0:
                for best_s, best_s_value, tt in zip(best_s_sequence, best_s_values, shift_range_t):
                    if len(best_s) == 1:
                        intervention_blanket[best_s[0]][tt] = float(best_s_value)
                    else:
                        for var, val in zip(best_s, best_s_value):
                            intervention_blanket[var][tt] = val

            for level in interventional_grids.get(s, []):
                if len(s) == 1:
                    intervention_blanket[s[0]][t] = float(level.item() if isinstance(level, np.ndarray) else level)
                elif len(s) > 1:
                    assert len(level) == len(s), f"Mismatch between grid dimensions and exploration set {s}"
                    for var, val in zip(s, level):
                        intervention_blanket[var][t] = float(val.item() if isinstance(val, np.ndarray) else val)
                else:
                    raise ValueError(f"Unexpected exploration set: {s}")

                # Use TensorFlow to compute samples
                intervention_samples = sequentially_sample_model(
                    initial_structural_equation_model,
                    structural_equation_model,
                    total_timesteps=time_steps,
                    interventions=intervention_blanket,
                    sample_count=1,
                    epsilon=static_noise_model,
                )

                # TensorFlow-based expectation computation
                out = get_monte_carlo_expectation(intervention_samples)
                ce[s].append(out[target_variable][t])

        # Evaluate local target values
        local_target_values = [
            (
                s,
                (tf.argmin(ce[s]).numpy() if task == "min" else tf.argmax(ce[s]).numpy()),
                ce[s][(tf.argmin(ce[s]).numpy() if task == "min" else tf.argmax(ce[s]).numpy())],
            )
            for s in exploration_sets
        ]

        for s, idx, value in local_target_values:
            y_stars[s][t] = value
            optimal_interventions[s][t] = interventional_grids[s][idx]

        best_s, best_idx, best_objective_value = min(local_target_values, key=lambda t: t[2])
        best_s_value = interventional_grids[best_s][best_idx]

        best_s_sequence.append(best_s)
        best_s_values.append(best_s_value)
        best_objective_values.append(best_objective_value)
        all_ce.append(ce)

    return (
        best_s_values,
        best_s_sequence,
        best_objective_values,
        y_stars,
        optimal_interventions,
        all_ce,
    )



class StationaryIndependentSEM:
    """
    A class representing Stationary Independent Structural Equation Models (SEMs).

    This class provides static and dynamic SEMs, which are mathematical models used to
    simulate causal relationships between variables.
    """

    @staticmethod
    def static():
        """
        Define a static SEM where variables X, Z, and Y are independent of previous time steps.

        Returns:
            OrderedDict: A dictionary defining the SEM equations for X, Z, and Y.
        """
        return OrderedDict({
            "X": lambda noise, t, sample: tf.cast(noise, tf.float32),
            "Z": lambda noise, t, sample: tf.cast(noise, tf.float32),
            "Y": lambda noise, t, sample: (
                -2 * tf.exp(
                    -((tf.cast(sample["X"][t], tf.float32) - 1) ** 2)
                    - (tf.cast(sample["Z"][t], tf.float32) - 1) ** 2
                )
                - tf.exp(
                    -((tf.cast(sample["X"][t], tf.float32) + 1) ** 2)
                    - (tf.cast(sample["Z"][t], tf.float32) ** 2)
                )
                + tf.cast(noise, tf.float32)
            ),
        })

    @staticmethod
    def dynamic():
        """
        Define a dynamic SEM where variables X, Z, and Y depend on their values from the previous time step.

        Returns:
            OrderedDict: A dictionary defining the SEM equations for X, Z, and Y.
        """
        return OrderedDict({
            "X": lambda noise, t, sample: -tf.cast(sample["X"][t - 1], tf.float32) + tf.cast(noise, tf.float32),
            "Z": lambda noise, t, sample: -tf.cast(sample["Z"][t - 1], tf.float32) + tf.cast(noise, tf.float32),
            "Y": lambda noise, t, sample: (
                -2 * tf.exp(
                    -((tf.cast(sample["X"][t], tf.float32) - 1) ** 2)
                    - (tf.cast(sample["Z"][t], tf.float32) - 1) ** 2
                )
                - tf.exp(
                    -((tf.cast(sample["X"][t], tf.float32) + 1) ** 2)
                    - (tf.cast(sample["Z"][t], tf.float32) ** 2)
                )
                + tf.cast(sample["Y"][t - 1], tf.float32)
                + tf.cast(noise, tf.float32)
            ),
        })


def sequentially_sample_model(
        static_sem,
        dynamic_sem,
        total_timesteps: int,
        initial_values=None,
        interventions=None,
        node_parents=None,
        sample_count: int = 100,
        epsilon=None,
        use_sem_estimate: bool = False,
        seed: int = None,
) -> dict:
    """
    Draws multiple samples from a Dynamic Bayesian Network (DBN) using TensorFlow.

    Args:
        static_sem (dict): Static Structural Equation Model (SEM) defining relationships at each time step.
        dynamic_sem (dict): Dynamic SEM defining temporal dependencies between variables.
        total_timesteps (int): Total number of time steps to sample.
        initial_values (dict, optional): Initial values for variables. Defaults to an empty dictionary.
        interventions (dict, optional): Dictionary of interventions to apply at specific time steps.
        node_parents (dict, optional): Mapping of nodes to their parents in the SEM.
        sample_count (int, optional): Number of samples to generate. Defaults to 100.
        epsilon (list or Tensor, optional): Noise terms for sampling. Defaults to None.
        use_sem_estimate (bool, optional): Whether to use estimated SEM instead of true SEM. Defaults to False.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        dict: Dictionary containing samples for each variable, formatted as n_samples x timesteps.
    """
    # Ensure initial_values is a dictionary
    initial_values = initial_values or {}
    if not isinstance(initial_values, dict):
        raise ValueError("initial_values must be a dictionary.")

    # Initialize samples dictionary
    samples = {var: [] for var in static_sem}

    # Iterate over the number of samples to generate
    for i in range(sample_count):
        epsilon_term = epsilon[i] if isinstance(epsilon, list) else epsilon

        if use_sem_estimate:
            # Use estimated SEM
            tmp = sequential_sample_from_SEM_hat(static_sem=static_sem, dynamic_sem=dynamic_sem, 
                                                 timesteps=total_timesteps, node_parents=node_parents, 
                                                 initial_values=initial_values, interventions=interventions,seed=seed,)
        else:
            # Use true SEM
            tmp = sequential_sample_from_true_SEM(static_sem=static_sem, dynamic_sem=dynamic_sem, timesteps=total_timesteps,
                                                  initial_values=initial_values, interventions=interventions, epsilon=epsilon_term, seed=seed,)

        # Append sampled values for each variable
        for var in static_sem:
            samples[var].append(tf.convert_to_tensor(tmp[var], dtype=tf.float32))

    # Stack results into TensorFlow tensors
    return {var: tf.stack(values, axis=0) for var, values in samples.items()}




def sequential_sample_from_true_SEM(static_sem: OrderedDict, dynamic_sem: OrderedDict, timesteps: int, initial_values: dict = None,
                                    interventions: dict = None, epsilon=None, seed=None,) -> OrderedDict:
    """
    Generates a single sample from the true Structural Equation Model (SEM) over multiple timesteps using TensorFlow.
    Args:
        static_sem (OrderedDict): Static SEM equations.
        dynamic_sem (OrderedDict): Dynamic SEM equations.
        timesteps (int): Number of timesteps.
        initial_values (dict): Initial values for each variable at t=0.
        interventions (dict): Dictionary specifying interventions at each timestep.
        epsilon (dict): Noise values for each variable and timestep.
        seed (int): Seed for reproducibility.

    Returns:
        OrderedDict: Samples for each variable across all timesteps.
    """
    if seed is not None:
        tf.random.set_seed(seed)

    # Initialize noise if not provided
    epsilon = epsilon or {k: tf.random.normal([timesteps]) for k in static_sem}
    # Initialize samples as TensorFlow tensors
    sample = OrderedDict((k, tf.zeros([timesteps], dtype=tf.float32)) for k in static_sem)
    initial_values = initial_values or {}

    if initial_values and not isinstance(initial_values, dict):
        raise ValueError("initial_values must be a dictionary.")

    if initial_values:
        assert set(sample.keys()) == set(initial_values.keys()), "Mismatch between sample and initial_values keys."

    for t in range(timesteps):
        # Use static SEM for t=0 or dynamic SEM otherwise
        target_sem = static_sem if t == 0 or dynamic_sem is None else dynamic_sem

        for var, function in target_sem.items():
            if interventions and interventions[var][t] is not None:
                sample[var] = tf.tensor_scatter_nd_update(
                    sample[var],
                    indices=[[t]],
                    updates=[tf.cast(interventions[var][t], tf.float32)],
                )
            elif t == 0 and initial_values.get(var) is not None:
                sample[var] = tf.tensor_scatter_nd_update(
                    sample[var],
                    indices=[[t]],
                    updates=[tf.cast(initial_values[var], tf.float32)],
                )
            else:
                sample[var] = tf.tensor_scatter_nd_update(
                    sample[var],
                    indices=[[t]],
                    updates=[
                        tf.cast(function(tf.cast(epsilon[var][t], tf.float32), t, sample), tf.float32)
                    ],
                )

    return sample



def create_n_dimensional_intervention_grid(limits: list, size_intervention_grid: int = 100):
    """
    Creates an n-dimensional grid for interventions using TensorFlow.

    Args:
        limits : (list) Ranges for each dimension of the grid.
        size_intervention_grid : (int, optional) Number of points per dimension (default is 100).

    Return:
    
        tf.Tensor: The n-dimensional intervention grid as a TensorFlow tensor.
    """
    # Check if the limits define a single-dimensional grid
    if not any(isinstance(el, list) for el in limits):
        return tf.expand_dims(
            tf.linspace(limits[0], limits[1], size_intervention_grid), axis=-1
        )

    # Prepare for multi-dimensional grid
    extrema = tf.constant(limits, dtype=tf.float32)
    inputs = [tf.linspace(i, j, size_intervention_grid) for i, j in zip(extrema[:, 0], extrema[:, 1])]

    # Create the meshgrid
    grids = tf.meshgrid(*inputs, indexing="ij")

    # Stack and reshape to create the intervention grid
    grid_tensor = tf.stack(grids, axis=-1)
    return tf.reshape(grid_tensor, [-1, len(inputs)])


def get_interventional_grids(exploration_sets, intervention_limits, size_intervention_grid=100) -> dict:
    """
    Builds n-dimensional interventional grids for each exploration set.

    Args:
    
        exploration_sets : (list) Sets of variables for which grids are created.
        intervention_limits : (dict) Ranges for each variable.
        size_intervention_grid : (int, optional) Number of points per dimension (default is 100).

    Returns
        dict: Dictionary of grids indexed by exploration sets.
    """
    grids = {}
    for es in exploration_sets:
        if len(es) == 1:  # Single-variable exploration sets
            grids[es] = create_n_dimensional_intervention_grid(
                intervention_limits[es[0]], size_intervention_grid
            )
        elif len(es) > 1:  # Multi-variable exploration sets
            limits = [intervention_limits[var] for var in es]
            grids[es] = create_n_dimensional_intervention_grid(
                limits, size_intervention_grid
            )
        else:
            raise ValueError(f"Unexpected exploration set: {es}")

        # Debugging output
        print(f"Created grid for exploration set {es}: shape {grids[es].shape}")
    return grids



def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))

def get_monte_carlo_expectation(intervention_samples):
    """
    Computes the Monte Carlo expectation of the interventions.
    Args:
        intervention_samples (dict): Dictionary containing sampled values.

    Returns:
        dict: Dictionary of expected values for each intervention.
    """
    new = {k: None for k in intervention_samples.keys()}
    for es in new.keys():
        new[es] = tf.reduce_mean(intervention_samples[es], axis=0)  # Use TensorFlow's reduce_mean
    return new


def make_sequential_intervention_dict(G: MultiDiGraph, T: int) -> dict:
    """
    Creates a dictionary for sequential interventions, specifying the variables to intervene on and at what time steps.

    Args:
    
        G : MultiDiGraph, A structural causal graph.
        T : (int) Total time-series length.

    Returns
        dict: Dictionary of sequential interventions with None values as placeholders.
    """
    # Extract variables from graph nodes
    variables = sorted(set("".join(G.nodes).strip()))
    
    # Create the intervention dictionary
    return {v: [None] * T for v in variables}


def setup_ind_scm(time_steps: int = 3):
    """
    Set up an independent Structural Causal Model (SCM) based on the SEM defined in the referenced paper.

    Args:
        time_steps (int, optional): Number of time steps for the SCM. Defaults to 3.

    Returns:
        tuple: Contains the following:
            - init_sem: Initial Structural Equation Model (SEM).
            - sem: Dynamic Structural Equation Model (SEM).
            - G_view: Visualization of the graphical model.
            - G: Directed Acyclic Graph (DAG) representation of the SCM.
            - exploration_sets: List of variable sets for intervention exploration.
            - intervention_domain: Domain of possible interventions for each variable.
            - true_objective_values: True objective values computed from optimal interventions.
    """
    # Load SEM from the specified paper (upper left quadrant in Figure 1)
    sem_model = StationaryIndependentSEM()
    init_sem, dynamic_sem = sem_model.static(), sem_model.dynamic()

    # Create base topology Directed Acyclic Graph (DAG)
    graphical_model_view = make_graphical_model(
        start_time=0,
        stop_time=time_steps - 1,
        topology="independent",
        nodes=["X", "Z", "Y"],
        target_node="Y",
        verbose=True,
    )
    dag = nx_agraph.from_agraph(pygraphviz.AGraph(graphical_model_view.source))

    # Define exploration sets and intervention domains
    exploration_sets = list(powerset(["X", "Z"]))
    intervention_domain = {"X": [-4, 1], "Z": [-3, 3]}

    # Generate interventional grids for exploration sets
    interventional_grids = get_interventional_grids(
        exploration_sets, intervention_domain, size_intervention_grid=100
    )

    # Compute true objective values for the interventions
    _, _, true_objective_values, _, _, _ = optimal_sequence_of_interventions(
        exploration_sets=exploration_sets,
        interventional_grids=interventional_grids,
        initial_structural_equation_model=init_sem,
        structural_equation_model=dynamic_sem,
        G=dag,
        T=time_steps,
        model_variables=["X", "Z", "Y"],
        target_variable="Y",
    )

    return init_sem, dynamic_sem, graphical_model_view, dag, exploration_sets, intervention_domain, true_objective_values



def make_graphical_model(start_time: int, stop_time: int, topology: str, nodes: List[str], target_node: str = None, verbose: bool = False,) -> Union[MultiDiGraph, str]:
    """
    Creates a temporal Bayesian network with two types of connections: spatial and temporal.

    Args:
    
        start_time : (int) Index of the first time-step.
        stop_time : (int) Index of the last time-step.
        topology : (str) "independent" or "dependent" causal topology.
        nodes : (list) List of nodes in the time-slice (e.g., ["X", "Z", "Y"]).
        target_node : (str, optional) Required for "independent" topology to specify the target node.
        verbose : (bool, optional) If True, returns DOT format of the graph for visualization.

    Returns
    
        Union: (MultiDiGraph, str) DOT format of the graph or a NetworkX object.
    """
    # Validations
    assert start_time <= stop_time, "Start time must be less than or equal to stop time."
    assert topology in ["dependent", "independent"], "Invalid topology."
    assert nodes, "Nodes cannot be empty."

    if topology == "independent":
        assert target_node, "Target node is required for independent topology."
        assert isinstance(target_node, str), "Target node must be a string."

    spatial_edges, ranking = [], []

    # Handle "independent" topology
    if topology == "independent":
        if target_node in nodes:
            nodes.remove(target_node)
        edge_pairs = [(node, target_node) for node in nodes]
    else:
        # Handle "dependent" topology
        edge_pairs = list(zip(nodes[:-1], nodes[1:]))

    # Add spatial edges
    for t in range(start_time, stop_time + 1):
        for src, tgt in edge_pairs:
            spatial_edges.append(f"{src}_{t} -> {tgt}_{t};")
        all_nodes = nodes + ([target_node] if topology == "independent" else [])
        ranking.append(f"{{ rank=same; {' '.join(f'{node}_{t}' for node in all_nodes)} }}")

    # Add temporal edges
    temporal_edges = []
    all_nodes = nodes + ([target_node] if topology == "independent" else [])
    for t in range(start_time, stop_time):
        for node in all_nodes:
            temporal_edges.append(f"{node}_{t} -> {node}_{t + 1};")

    # Combine edges into a Graphviz-compatible DOT graph
    graph = f"digraph {{ rankdir=LR; {' '.join(spatial_edges)} {' '.join(temporal_edges)} {' '.join(ranking)} }}"

    # Return either a DOT graph string or a visual representation using Graphviz
    return Source(graph) if verbose else graph


def fifteen_nodes_per_time_steps(timesteps: int = 3):
    """
    Creates a 15-node causal graph with connections across multiple time steps and computes 
    optimal interventions and causal effects.
    
    Args:
        start_time : (int) Index of the first time-step.

    Returns:
    -------
    tuple
        A tuple containing the following:
        - init_sem: Initial structural equation model.
        - sem: Dynamic structural equation model.
        - G: Directed causal graph as a networkx MultiDiGraph.
        - exploration_sets: List of exploration sets.
        - intervention_domain: Dictionary defining the intervention domain.
        - true_objective_values: True objective values for the interventions.
        - optimal_interventions: Optimal interventions as a dictionary.
        - all_causal_effects: All computed causal effects.
    """
    def powerset(iterable):
        """Generates all subsets of a set."""
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    # Initialize SEM
    sem_class = StationaryIndependentSEM() 
    init_sem, sem = sem_class.static(), sem_class.dynamic()

    # Define the graph edges to match the 15-node causal graph
    G = nx.MultiDiGraph()  # Use MultiDiGraph
    # Add edges for the X variables (chain-like structure over time)
    for i in range(1, 8):  # X1 to X7
        G.add_edge(f"X{i}_0", f"X{i}_1")
        G.add_edge(f"X{i}_1", f"X{i}_2")

    # Add edges for the Z variables (chain-like structure over time)
    for i in range(1, 8):  # Z1 to Z7
        G.add_edge(f"Z{i}_0", f"Z{i}_1")
        G.add_edge(f"Z{i}_1", f"Z{i}_2")

    # Add edges for the Y variables (chain-like structure over time)
    G.add_edge("Y_0", "Y_1")
    G.add_edge("Y_1", "Y_2")

    # Add connections from X and Z to Y at each time step
    for t in range(timesteps):
        for i in range(1, 8):  # X1 to X7 and Z1 to Z7
            G.add_edge(f"X{i}_{t}", f"Y_{t}")
            G.add_edge(f"Z{i}_{t}", f"Y_{t}")

    # Define exploration sets and intervention domains
    exploration_sets = [s for s in powerset(["X", "Z"]) if s]  # Filter empty sets
    intervention_domain = {"X": [-4, 1], "Z": [-3, 3]}  # Ensure all variables are included

    # Generate interventional grids
    interventional_grids = get_interventional_grids(exploration_sets, intervention_domain, size_intervention_grid=100)

    # Compute optimal interventions and causal effects
    (
        _,
        optimal_interventions,
        true_objective_values,
        _,
        _,
        all_causal_effects,
    ) = optimal_sequence_of_interventions(
        exploration_sets=exploration_sets,
        interventional_grids=interventional_grids,
        initial_structural_equation_model=init_sem,
        structural_equation_model=sem,
        G=G,
        T=timesteps,
        model_variables=["X", "Z", "Y"],
        target_variable="Y",
    )

    # Convert optimal_interventions to a dictionary if necessary
    if not isinstance(optimal_interventions, dict):
        optimal_interventions = {
            t: intervention
            for t, intervention in enumerate(optimal_interventions)
        }

    # Visualize the graph
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    plt.figure(figsize=(15, 8))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=10, font_weight="bold")
    plt.savefig("fifteen_nodes_per_time_step.svg")
    plt.savefig("fifteen_nodes_per_time_step.png")
    plt.show()

    return (
        init_sem,
        sem,
        G,
        exploration_sets,
        intervention_domain,
        true_objective_values,
        optimal_interventions,
        all_causal_effects,
    )


def run_methods_replicates(
        graph,
        structural_equation_model,
        make_sem_estimator,
        intervention_domain,
        methods_list,
        observational_samples,
        exploration_sets,
        base_target_variable,
        ground_truth=None,
        total_timesteps: int = 3,
        replicates: int = 3,
        number_of_trials: int = 3,
        number_of_trials_bo_abo=None,
        num_restarts: int = 1,
        save_data: bool = False,
        num_observations: int = 100,
        cost_structure: int = 1,
        optimal_assigned_blankets=None,
        debug_mode: bool = False,
        use_monte_carlo: bool = False,
        online: bool = False,
        concatenate: bool = False,
        use_di: bool = False,
        transfer_hp_outer: bool = False,
        transfer_hp_inner: bool = False,
        hp_inner_prior: bool = True,
        estimate_sem: bool = True,
        folder: str = None,
        num_anchor_points: int = 100,
        num_obs_t=None,
        sample_anchor_points: bool = True,
        seed: int = 0,
        controlled_experiment: bool = True,
        noise_experiment: bool = False,
        sem_args=None,
        manipulative_variables=None,
        change_points=None,
):
    """
    Run multiple replicates of optimization methods with the given parameters.

    Args:
        graph: The causal graph representing the system.
        structural_equation_model: Function to create the SEM.
        make_sem_estimator: Function to create SEM estimators.
        intervention_domain: Dictionary defining the intervention ranges for variables.
        methods_list: List of methods to optimize.
        observational_samples: Pre-sampled observational data (optional).
        exploration_sets: List of sets of variables to intervene on.
        base_target_variable: Target variable for optimization.
        ground_truth: Ground truth values for comparison (optional).
        total_timesteps (int): Number of timesteps for interventions. Default is 3.
        replicates (int): Number of replicates to run. Default is 3.
        number_of_trials (int): Number of trials for optimization. Default is 3.
        number_of_trials_bo_abo (int): Number of trials for Bayesian Optimization (optional).
        num_restarts (int): Number of restarts for optimization. Default is 1.
        save_data (bool): Whether to save the results. Default is False.
        num_observations (int): Number of observational samples. Default is 100.
        cost_structure (int): Cost structure for interventions. Default is 1.
        optimal_assigned_blankets: Predefined blankets for interventions (optional).
        debug_mode (bool): Enable debugging mode. Default is False.
        use_monte_carlo (bool): Whether to use Monte Carlo estimation. Default is False.
        online (bool): Use online learning methods. Default is False.
        concatenate (bool): Concatenate datasets. Default is False.
        use_di (bool): Use domain information. Default is False.
        transfer_hp_outer (bool): Transfer hyperparameters for outer loop. Default is False.
        transfer_hp_inner (bool): Transfer hyperparameters for inner loop. Default is False.
        hp_inner_prior (bool): Use inner hyperparameter prior. Default is True.
        estimate_sem (bool): Estimate the SEM. Default is True.
        folder (str): Directory to save results. Default is None.
        num_anchor_points (int): Number of anchor points. Default is 100.
        num_obs_t: Observational data per timestep (optional).
        sample_anchor_points (bool): Sample anchor points. Default is True.
        seed (int): Random seed for reproducibility. Default is 0.
        controlled_experiment (bool): Use controlled experiments. Default is True.
        noise_experiment (bool): Include noise in experiments. Default is False.
        sem_args: Arguments for SEM initialization (optional).
        manipulative_variables: Variables that can be manipulated (optional).
        change_points: Points of structural change in the SEM (optional).

    Returns:
        dict: Results of the replicates for each optimization method.
    """
    # Initialize structural equation model
    if sem_args is None and change_points is None:
        true_sem = structural_equation_model()
    elif sem_args and change_points is None:
        true_sem = structural_equation_model(*sem_args)
    else:
        true_sem = structural_equation_model(change_points.index(True))

    initial_sem, dynamic_sem = true_sem.static(), true_sem.dynamic()

    # Set up results containers
    results, opt_results_for_pickle = {}, {}
    tf.random.set_seed(seed)

    # Sample observational data if not provided
    if observational_samples is None:
        epsilon_list = None
        if noise_experiment:
            epsilon_list = [
                {
                    k: tf.random.normal([total_timesteps], mean=2.0, stddev=4.0)
                    for k in initial_sem.keys()
                }
                for _ in range(num_observations)
            ]
            for eps in epsilon_list:
                eps[base_target_variable] = tf.random.normal([total_timesteps])

        observation_samples = sequentially_sample_model(
            initial_sem,
            dynamic_sem,
            total_timesteps=total_timesteps,
            sample_count=num_observations,
            epsilon=epsilon_list,
        )
    else:
        observation_samples = observational_samples

    # Run replicates
    for ex in trange(replicates, desc="Experiment count"):
        seed_anchor_points = ex + 1 if controlled_experiment else None
        num_trials = number_of_trials_bo_abo or number_of_trials

        # Prepare input parameters for optimization
        input_params = {
            "graph": graph,
            "sem": structural_equation_model,
            "base_target_variable": base_target_variable,
            "observation_samples": observation_samples,
            "intervention_domain": intervention_domain,
            "intervention_samples": None,
            "number_of_trials": number_of_trials,
            "task": "min",
            "cost_type": cost_structure,
            "num_restarts": num_restarts,
            "debug_mode": debug_mode,
            "optimal_assigned_blankets": optimal_assigned_blankets,
            "num_anchor_points": num_anchor_points,
            "sample_anchor_points": sample_anchor_points,
            "seed_anchor_points": seed_anchor_points,
            "hp_inner_prior": hp_inner_prior,
            "sem_args": sem_args,
            "manipulative_variables": manipulative_variables,
            "change_points": change_points,
        }

        models, names = run_all_opt_models(
            methods_list,
            input_params,
            exploration_sets,
            online,
            use_di,
            transfer_hp_outer,
            transfer_hp_inner,
            concatenate,
            estimate_sem,
            use_monte_carlo,
            ground_truth,
            num_obs_t,
            make_sem_estimator,
            num_trials,
        )

        for i, key in enumerate(names):
            results.setdefault(key, []).append(models[i])

            if save_data:
                opt_results_for_pickle.setdefault(key, []).append(
                    (
                        models[i].per_trial_cost,
                        models[i].optimal_outcome_values_during_trials,
                        models[i].optimal_intervention_sets,
                        models[i].assigned_blanket,
                    )
                )

    # Save results
    if save_data:
        missing = isinstance(num_obs_t, list)
        file_name = (
            f"../data/{folder}/method_{''.join(methods_list)}_T_{total_timesteps}_"
            f"it_{number_of_trials}_reps_{replicates}_Nobs_{num_observations}_online_{online}_"
            f"concat_{concatenate}_transfer_{transfer_hp_inner}_usedi_{use_di}_"
            f"hpiprior_{hp_inner_prior}_missing_{missing}_noise_{noise_experiment}_"
            f"optimal_assigned_blanket_{optimal_assigned_blankets}_seed_{seed}.pickle"
        )

        with open(file_name, "wb") as handle:
            pickle.dump(opt_results_for_pickle, handle)

    return results



def build_sem_hat(G: MultiDiGraph, emission_fncs: dict, transition_fncs: dict = None) -> classmethod:
    """
    Create SEM-function estimates for the edges in a causal graph.

    Args:
    
        G : MultiDiGraph, Causal graphical model.
        emission_fncs : (dict) Fitted emission functions (horizontal edges in the DAG).
        transition_fncs : (dict) Fitted transition functions (vertical edges in the DAG).

    Returns:
    
    classmethod
        A SEM estimate for finding optimal intervention sets and values.
    """

    class SEMHat:
        def __init__(self):
            self.G = G
            self.n_t = tf.cast(len(G.nodes()) // G.T, tf.int32)

        @staticmethod
        def _make_marginal() -> Callable:
            """
            Create a marginal sampling function for emission functions.
            """
            return lambda t, margin_id: emission_fncs[t][margin_id].sample()

        @staticmethod
        def _make_emit_fnc(moment: int) -> Callable:
            """
            Create an emission function for the given moment (mean or variance).
            """
            return lambda t, _, emit_vars, sample: emission_fncs[t][emit_vars].predict(
                select_sample(sample, emit_vars, t)
            )[moment]

        @staticmethod
        def _make_trans_fnc(moment: int) -> Callable:
            """
            Create a transition function for the given moment (mean or variance).
            """
            return lambda t, trans_vars, _, sample: transition_fncs[t][trans_vars].predict(
                select_sample(sample, trans_vars, t - 1)
            )[moment]

        @staticmethod
        def _make_emit_plus_trans_fnc(moment: int) -> Callable:
            """
            Create a function combining emission and transition functions.
            """
            return lambda t, trans_vars, emit_vars, sample: (
                transition_fncs[t][trans_vars].predict(select_sample(sample, trans_vars, t - 1))[moment]
                + emission_fncs[t][emit_vars].predict(select_sample(sample, emit_vars, t))[moment]
            )

        def static(self, moment: int) -> OrderedDict:
            """
            Build static SEM functions for time t=0.

            Parameters
            ----------
            moment : int
                Moment to predict (mean or variance).

            Returns
            -------
            OrderedDict
                Static SEM functions.
            """
            assert moment in [0, 1], moment
            f = OrderedDict()
            for v in list(self.G.nodes)[: self.n_t]:
                vv = v.split("_")[0]
                if self.G.in_degree[v] == 0:
                    f[vv] = self._make_marginal()
                else:
                    f[vv] = self._make_emit_fnc(moment)
            return f

        def dynamic(self, moment: int) -> OrderedDict:
            """
            Build dynamic SEM functions for time t > 0.

            Parameters
            ----------
            moment : int
                Moment to predict (mean or variance).

            Returns
            -------
            OrderedDict
                Dynamic SEM functions.
            """
            assert moment in [0, 1], moment
            f = OrderedDict()
            for v in list(self.G.nodes)[self.n_t : 2 * self.n_t]:
                vv = v.split("_")[0]
                preds = list(self.G.predecessors(v))
                if self.G.in_degree[v] == 0:
                    f[vv] = self._make_marginal()
                elif all(
                    int(pred.split("_")[1]) + 1 == int(v.split("_")[1]) for pred in preds
                ):
                    f[vv] = self._make_trans_fnc(moment)
                elif all(pred.split("_")[1] == v.split("_")[1] for pred in preds):
                    f[vv] = self._make_emit_fnc(moment)
                else:
                    f[vv] = self._make_emit_plus_trans_fnc(moment)
            return f

    return SEMHat


def select_sample(sample: OrderedDict, input_variables: Iterable, outside_time: int) -> np.ndarray:
    """
    Select a subset of the sample for GP regression input.

    Args:
    
        sample : (OrderedDict) The sample being created.
        input_variables : Iterable, Input variables for the GP regression.
        outside_time : (int) Current time index being processed.

    Returns:
    
        np.ndarray: Formatted input as an ndarray of shape N x D.
    """

    if isinstance(input_variables, str):
        return sample[input_variables][outside_time].reshape(-1, 1)

    if len(input_variables) == 3 and isinstance(input_variables[1], int):
        # Handle input like (pa_V, ID, V)
        var, time = input_variables[0].split("_")[0], int(input_variables[0].split("_")[1])
        assert time == outside_time, (sample, input_variables, time, outside_time)
        return sample[var][time].reshape(-1, 1)

    # General case for tuple or list input
    return hstack([
        sample[node.split("_")[0]][int(node.split("_")[1])].reshape(-1, 1)
        for node in input_variables
        if int(node.split("_")[1]) == outside_time
    ])



def run_all_opt_models(methods_list, input_params, exploration_sets, online, use_di, transfer_hp_o, transfer_hp_i,
                       concat, estimate_sem, use_mc, ground_truth, n_obs_t, make_sem_estimator, number_of_trials_BO_ABO,):
    """
    Sequentially runs optimization models based on the given methods list.

    Args:
    
        methods_list : (list) List of methods to run (e.g., "ABO", "DCBO", "BO", "CBO").
        input_params : (dict) Parameters common to all methods.
        exploration_sets : (any) Exploration sets for optimization.
        online, use_di, transfer_hp_o, transfer_hp_i, concat, estimate_sem, use_mc : (bool) Flags for specific algorithm behaviors.
        ground_truth : Ground truth data.
        n_obs_t : Number of observed time points.
        make_sem_estimator : (callable) Function to create SEM estimators.
        number_of_trials_BO_ABO : (int) Number of trials for BO and ABO methods.

    Returns:
    
        models_list : (list) List of model instances for each method.
        names_list : (list) List of method names.
    """

    models_list, names_list = [], []

    for method in methods_list:
        assert method in ["ABO", "DCBO", "BO", "CBO"], f"Method not implemented: {method}"

        alg_input_params = deepcopy(input_params)
        names_list.append(method)

        if method in ["DCBO", "CBO"]:
            alg_input_params.update({
                "estimate_sem": estimate_sem,
                "exploration_sets": exploration_sets,
                "online": online,
                "ground_truth": ground_truth,
                "use_mc": use_mc,
                "n_obs_t": n_obs_t,
                "make_sem_estimator": make_sem_estimator,
            })

        if method == "DCBO":
            algorithm = DCBO
            alg_input_params.update({"use_di": use_di, "transfer_hp_o": transfer_hp_o, "transfer_hp_i": transfer_hp_i})
        elif method == "CBO":
            algorithm = CBO
            alg_input_params["concat"] = concat
        else:
            algorithm = ABO if method == "ABO" else BO
            alg_input_params["number_of_trials"] = number_of_trials_BO_ABO

        print(f"\n>>> {method}\n")
        model = algorithm(**alg_input_params)
        model.run()
        models_list.append(model)

    return models_list, names_list

