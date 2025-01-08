
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


class TestDynamicCausalGraphIntegration(unittest.TestCase):

    def test_full_workflow(self):
        """
        Comprehensive integration test to validate the full workflow.
        It ensures all components tested in unit tests work together seamlessly.
        """
        # Step 1: Initialize SEM
        sem = StationaryIndependentSEM()
        static_sem = sem.static()
        dynamic_sem = sem.dynamic()
        self.assertIsInstance(static_sem, OrderedDict, "Static SEM should be an OrderedDict")
        self.assertIn("X", static_sem, "Static SEM should contain variable X")
        self.assertIn("Z", static_sem, "Static SEM should contain variable Z")
        self.assertIn("Y", static_sem, "Static SEM should contain variable Y")
        self.assertIsInstance(dynamic_sem, OrderedDict, "Dynamic SEM should be an OrderedDict")
        self.assertIn("X", dynamic_sem, "Dynamic SEM should contain variable X")
        self.assertIn("Z", dynamic_sem, "Dynamic SEM should contain variable Z")
        self.assertIn("Y", dynamic_sem, "Dynamic SEM should contain variable Y")

        # Step 2: Create causal graph using make_graphical_model
        graph = make_graphical_model(0, 2, "independent", ["X", "Z", "Y"], target_node="Y", verbose=False)
        self.assertIsInstance(graph, str, "Graph representation should be a string")

        # Step 3: Create 15-node causal graph
        timesteps = 3
        result = fifteen_nodes_per_time_steps(timesteps=timesteps)
        (
            init_sem,
            sem,
            G,
            exploration_sets,
            intervention_domain,
            true_objective_values,
            optimal_interventions,
            all_causal_effects,
        ) = result

        self.assertIsInstance(G, MultiDiGraph, "Graph should be a MultiDiGraph")
        self.assertGreater(len(G.nodes), 0, "Graph should have nodes")
        self.assertGreater(len(G.edges), 0, "Graph should have edges")

        # Step 4: Generate exploration sets and intervention grids
        exploration_sets = [("X",), ("Z",), ("X", "Z")]
        intervention_limits = {"X": [-4, 1], "Z": [-3, 3]}
        interventional_grids = get_interventional_grids(
            exploration_sets=exploration_sets,
            intervention_limits=intervention_limits,
            size_intervention_grid=5
        )
        self.assertEqual(len(interventional_grids), 3, "Should create grids for all exploration sets")
        for es in exploration_sets:
            self.assertIn(es, interventional_grids, f"Exploration set {es} missing from grids")

        # Step 5: Create an n-dimensional intervention grid
        grid = create_n_dimensional_intervention_grid([-4, 1], size_intervention_grid=10)
        self.assertIsInstance(grid, tf.Tensor, "Intervention grid should be a TensorFlow tensor")
        self.assertEqual(grid.shape[0], 10, "Intervention grid should have 10 points")

        # Step 6: Sequentially sample from the model
        samples = sequentially_sample_model(
            static_sem,
            dynamic_sem,
            total_timesteps=3,
            sample_count=5
        )
        self.assertIsInstance(samples, dict, "Samples should be a dictionary")
        self.assertIn("X", samples, "Samples should include variable X")
        self.assertIn("Z", samples, "Samples should include variable Z")
        self.assertIn("Y", samples, "Samples should include variable Y")
        for key in samples:
            self.assertEqual(samples[key].shape[0], 5, f"Samples for {key} should have 5 entries")
            self.assertEqual(samples[key].shape[1], 3, f"Samples for {key} should have 3 timesteps")

        # Step 7: Create a sequential intervention dictionary
        intervention_dict = make_sequential_intervention_dict(G, timesteps)
        self.assertIsInstance(intervention_dict, dict, "Intervention dictionary should be a dictionary")
        self.assertIn("X", intervention_dict, "Intervention dictionary should include X")
        self.assertIn("Z", intervention_dict, "Intervention dictionary should include Z")
        self.assertIn("Y", intervention_dict, "Intervention dictionary should include Y")
        self.assertEqual(len(intervention_dict["X"]), 3, "Each variable in intervention dictionary should have 3 entries")

        # Step 8: Compute optimal interventions
        best_s_values, best_s_sequence, _, _, _, _ = optimal_sequence_of_interventions(
            exploration_sets=exploration_sets,
            interventional_grids=interventional_grids,
            initial_structural_equation_model=static_sem,
            structural_equation_model=dynamic_sem,
            G=G,
            T=timesteps,
            model_variables=["X", "Z", "Y"],
            target_variable="Y",
        )
        self.assertIsInstance(best_s_values, list, "Best S values should be a list")
        self.assertIsInstance(best_s_sequence, list, "Best S sequence should be a list")
        self.assertEqual(len(best_s_sequence), timesteps, "Best S sequence length should match timesteps")

        print("Comprehensive integration test for full workflow passed successfully.")

if __name__ == "__main__":
    unittest.main()

