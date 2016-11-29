import unittest
from eventcnn.datasources.eventsource import EventSource
from eventcnn.datasources.davis import DavisDataset


class TestEventSources(unittest.TestCase):
    def test_combination(self):
        set1 = DavisDataset.named_dataset("poster_translation")
        set2 = DavisDataset.named_dataset("shapes_translation")
        source1 = EventSource(set1,
                              testing_fraction=0.1,
                              validation_fraction=0.25,
                              events_per_block=100)
        source2 = EventSource(set2,
                              testing_fraction=0.1,
                              validation_fraction=0.25,
                              events_per_block=100)
        combined = source1 + source2

        it = iter(combined.training())
        for i in range(1000):
            next(it)
