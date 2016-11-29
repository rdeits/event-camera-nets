import unittest
from eventcnn.datasources.eventsource import take_randomly


class TestShuffling(unittest.TestCase):
    def test_take_randomly(self):
        iterables = [range(5), range(5, 9), range(9, 12)]
        found = [False for i in range(12)]
        for sample in take_randomly(iterables, [5, 4, 3]):
            self.assertFalse(found[sample])
            found[sample] = True
        self.assertTrue(all(found))
