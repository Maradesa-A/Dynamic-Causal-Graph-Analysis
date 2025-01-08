import unittest
import tensorflow as tf
import Dynamic_Causal_graph
import importlib
importlib.reload(Dynamic_Causal_graph)

from Dynamic_Causal_graph import (
    optimal_sequence_of_interventions,
    StationaryIndependentSEM,
    sequentially_sample_model,
    create_n_dimensional_intervention_grid,
    get_interventional_grids,
    make_sequential_intervention_dict,
    setup_ind_scm,
    make_graphical_model,
    run_methods_replicates,
    build_sem_hat,
    select_sample,
    fifteen_nodes_per_time_steps,
    run_all_opt_models
)

from networkx import MultiDiGraph
from collections import OrderedDict

import unittest
from collections import OrderedDict
from networkx import MultiDiGraph
import tensorflow as tf

class TestDynamicCausalGraph(unittest.TestCase):

    def test_stationary_independent_sem_static(self):
        sem = StationaryIndependentSEM()
        static_sem = sem.static()
        self.assertIsInstance(static_sem, OrderedDict)
        self.assertIn("X", static_sem)
        self.assertIn("Z", static_sem)
        self.assertIn("Y", static_sem)

    def test_stationary_independent_sem_dynamic(self):
        sem = StationaryIndependentSEM()
        dynamic_sem = sem.dynamic()
        self.assertIsInstance(dynamic_sem, OrderedDict)
        self.assertIn("X", dynamic_sem)
        self.assertIn("Z", dynamic_sem)
        self.assertIn("Y", dynamic_sem)

    def test_make_graphical_model_independent(self):
        graph = make_graphical_model(0, 2, "independent", ["X", "Z", "Y"], target_node="Y", verbose=False)
        self.assertIsInstance(graph, str)

    def test_create_n_dimensional_intervention_grid(self):
        grid = create_n_dimensional_intervention_grid([-4, 1], size_intervention_grid=10)
        self.assertIsInstance(grid, tf.Tensor)
        self.assertEqual(grid.shape[0], 10)

    def test_get_interventional_grids(self):
        exploration_sets = [("X",), ("Z",), ("X", "Z")]
        intervention_limits = {"X": [-4, 1], "Z": [-3, 3]}
        grids = get_interventional_grids(exploration_sets, intervention_limits, size_intervention_grid=5)
        self.assertEqual(len(grids), 3)
        for key in exploration_sets:
            self.assertIn(key, grids)

    def test_setup_ind_scm(self):
        init_sem, dynamic_sem, _, G, exploration_sets, intervention_domain, true_objective_values = setup_ind_scm(3)
        self.assertIsInstance(init_sem, OrderedDict)
        self.assertIsInstance(dynamic_sem, OrderedDict)
        self.assertIsInstance(G, MultiDiGraph)
        self.assertIsInstance(exploration_sets, list)
        self.assertIsInstance(intervention_domain, dict)
        self.assertIsInstance(true_objective_values, list)

    def test_sequentially_sample_model(self):
        sem = StationaryIndependentSEM()
        static_sem, dynamic_sem = sem.static(), sem.dynamic()
        samples = sequentially_sample_model(static_sem, dynamic_sem, total_timesteps=3, sample_count=5)
        self.assertIsInstance(samples, dict)
        self.assertIn("X", samples)
        self.assertIn("Z", samples)
        self.assertIn("Y", samples)
        for key in samples:
            self.assertEqual(samples[key].shape[0], 5)
            self.assertEqual(samples[key].shape[1], 3)

    def test_make_sequential_intervention_dict(self):
        G = MultiDiGraph()
        G.add_nodes_from(["X", "Z", "Y"])
        intervention_dict = make_sequential_intervention_dict(G, 3)
        self.assertIsInstance(intervention_dict, dict)
        self.assertIn("X", intervention_dict)
        self.assertIn("Z", intervention_dict)
        self.assertIn("Y", intervention_dict)
        self.assertEqual(len(intervention_dict["X"]), 3)

    def test_optimal_sequence_of_interventions(self):
        # Initialize SEM
        sem = StationaryIndependentSEM()
        static_sem, dynamic_sem = sem.static(), sem.dynamic()

        # Define exploration sets and intervention grids
        exploration_sets = [("X",), ("Z",), ("X", "Z")]
        intervention_limits = {"X": [-4, 1], "Z": [-3, 3]}
        interventional_grids = get_interventional_grids(
            exploration_sets=exploration_sets,
            intervention_limits=intervention_limits,
            size_intervention_grid=5,
        )

        # Add assertion to ensure all grids are created
        for es in exploration_sets:
            self.assertIn(es, interventional_grids, f"Missing grid for exploration set: {es}")

        # Build causal graph
        G = MultiDiGraph()
        G.add_nodes_from(["X", "Z", "Y"])

        # Call optimal_sequence_of_interventions and check the results
        best_s_values, best_s_sequence, _, _, _, _ = optimal_sequence_of_interventions(
            exploration_sets=exploration_sets,
            interventional_grids=interventional_grids,
            initial_structural_equation_model=static_sem,
            structural_equation_model=dynamic_sem,
            G=G,
            T=3,
            model_variables=["X", "Z", "Y"],
            target_variable="Y",
        )

        # Assertions for outputs
        self.assertIsInstance(best_s_values, list)
        self.assertIsInstance(best_s_sequence, list)
        self.assertEqual(len(best_s_sequence), 3, "Expected a sequence of length equal to T.")
        
    def test_fifteen_nodes_per_time_steps(self):
        # Call fifteen_nodes_per_time_steps
        result = fifteen_nodes_per_time_steps(timesteps=3)

        # Unpack result
        init_sem, dynamic_sem, G, exploration_sets, intervention_domain, true_objective_values, optimal_interventions, all_causal_effects = result

        # Assertions
        self.assertIsInstance(init_sem, OrderedDict)
        self.assertIsInstance(dynamic_sem, OrderedDict)
        self.assertIsInstance(G, MultiDiGraph)
        self.assertIsInstance(exploration_sets, list)
        self.assertIsInstance(intervention_domain, dict)
        self.assertIsInstance(true_objective_values, list)
        self.assertIsInstance(optimal_interventions, dict)
        self.assertIsInstance(all_causal_effects, list)
        self.assertGreater(len(G.nodes), 0, "Graph should have nodes.")
        self.assertGreater(len(G.edges), 0, "Graph should have edges.")

if __name__ == "__main__":
    unittest.main()
