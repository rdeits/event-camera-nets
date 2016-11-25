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
            resolution = np.random.randint(1, 10, dimension)
            discretization = SpatialDiscretization(lb, ub, resolution)
            vector = np.random.rand(dimension) * (ub - lb) + lb
            one_hot = discretization.to_one_hot(vector)
            result = discretization.from_weights(one_hot)
            for i in range(dimension):
                self.assertTrue(
                    np.isclose(vector[i],
                               result[i],
                               atol=(ub[i] - lb[i]) / resolution[i]))


class TestEventBlock(unittest.TestCase):
    def test_poster(self):
        dataset = DavisDataset.named_dataset("shapes_translation")
        start = 10000
        length = 1000
        eventblock = dataset.event_block(start, start + length)
        self.assertEqual(len(eventblock.events), length)
