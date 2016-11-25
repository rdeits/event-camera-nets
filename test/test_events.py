import unittest
import random
import numpy as np
from eventcnn.datasources.eventblock import SpatialDiscretization
from eventcnn.datasources.davis import DavisDataset

class TestVectorEncoding(unittest.TestCase):
    def test_round_trip(self):
        for i in range(100):
            dimension = np.random.randint(1, 4)
            lb = np.random.randn(dimension)
            ub = lb + np.random.rand(dimension)
            steps = np.random.randint(1, 10, dimension)
            discretization = SpatialDiscretization(lb, ub, steps)
            vector = np.random.rand(dimension) * (ub - lb) + lb
            one_hot = discretization.to_one_hot(vector)
            result = discretization.from_weights(one_hot)
            for i in range(dimension):
                self.assertTrue(
                    np.isclose(vector[i],
                               result[i],
                               atol=(ub[i] - lb[i]) / steps[i]))


class TestEventBlock(unittest.TestCase):
    def test_block_length(self):
        dataset = DavisDataset.named_dataset("shapes_translation")
        start = 10000
        length = 1000
        eventblock = dataset.event_block(start, length)
        self.assertEqual(len(eventblock.events), length)

    def test_initial_block(self):
        dataset = DavisDataset.named_dataset("shapes_translation")
        eventblock = dataset.event_block(0, 100)

    def test_delta_position(self):
        dataset = DavisDataset.named_dataset("shapes_translation")
        eventblock = dataset.event_block(100, 500)
        lb = np.zeros(3) + -1e-4
        ub = np.zeros(3) + 1e-4
        steps = [2, 2, 2]
        discretization = SpatialDiscretization(lb, ub, steps)
        one_hot = discretization.to_one_hot(eventblock.delta_position)
        # Sum to one
        self.assertEqual(np.sum(one_hot), 1)
        # Only one non-zero value
        self.assertEqual(np.sum(one_hot != 0), 1)

