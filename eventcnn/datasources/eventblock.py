import numpy as np
import tensorflow as tf


class SpatialDiscretization:
    def __init__(self, lb, ub, steps):
        lb = np.asarray(lb)
        ub = np.asarray(ub)
        steps = np.asarray(steps)
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

    def to_one_hot_sparse_tensor(self, position):
        assert position.size == self.steps.size
        index = (position - self.lb) / self.resolution
        index = np.clip(index, 0, self.steps - 1)
        index = index.astype(np.int64)
        return tf.SparseTensorValue(
            [index],
            [1],
            self.steps)

    def to_one_hot(self, position):
        sp = self.to_one_hot_sparse_tensor(position)
        out = np.zeros(self.steps)
        assert len(sp.indices) == 1
        out[tuple(sp.indices[0])] = 1
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
    def __init__(self, events, delta_position,
                 camera_config):
        self.events = events
        self.delta_position = delta_position
        self.camera_config = camera_config

    @property
    def num_events(self):
        return len(self.events)

    def events_as_sparse_tensor(self):
        # Extract indices in x, y, z order
        indices = self.events[["x", "y"]].as_matrix()
        indices = np.hstack(
            (indices,
             np.arange(self.num_events).reshape((-1, 1))))

        # convert polarity from {False, True} to {-0.5, 0.5}
        values = self.events["polarity"].as_matrix() - 0.5

        shape = [self.camera_config.cols,
                 self.camera_config.rows,
                 self.num_events]

        return tf.SparseTensorValue(indices,
                                    values,
                                    shape)
