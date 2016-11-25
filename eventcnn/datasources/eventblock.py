import numpy as np


class SpatialDiscretization:
    def __init__(self, lb, ub, steps):
        assert lb.size == ub.size
        assert lb.size == steps.size
        self.lb = lb
        self.ub = ub
        self.steps = steps

    @property
    def resolution(self):
        return (self.ub - self.lb) / self.steps

    @property
    def dimension(self):
        return self.steps.size

    def to_one_hot(self, position):
        assert position.size == self.steps.size
        index = (position - self.lb) / self.resolution
        index = np.clip(index, 0, self.steps - 1)
        # TODO: return sparse data structure
        out = np.zeros(self.steps)
        out[tuple(index.astype(np.int64))] = 1
        return out

    def from_weights(self, weights):
        result = np.zeros(self.dimension)
        for inds in np.indices(
                self.steps).reshape(
                (self.steps.size, -1)).transpose():
            coordinates = inds * self.resolution + self.lb
            result += coordinates * weights[tuple(inds)]
        return result


class EventBlock:
    def __init__(self, events, delta_position):
        self.events = events
        self.delta_position = delta_position
