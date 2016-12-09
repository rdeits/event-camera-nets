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

    def event_indices(self, n_layers=None, scaling=1):
        if n_layers is None:
            n_layers = self.num_events
        return np.hstack(((self.events[["x", "y"]] // scaling),
                          np.floor(np.arange(0, n_layers, n_layers / self.num_events).reshape((-1,1))))).astype(np.uint16)

    def events_rescaled(self, n_layers=None, scaling=1):
        if n_layers is None:
            n_layers = self.num_events
        # Construct the dense data matrix, with a trailing dimension
        # of length 1 to keep tensorflow's conv3d happy
        data = np.zeros((self.camera_config.cols // scaling,
                         self.camera_config.rows // scaling,
                         n_layers,
                         1))
        indices = self.event_indices(n_layers=n_layers, scaling=scaling)
        values = self.events["polarity"].as_matrix() - 0.5
        for (i, I) in enumerate(indices):
            data[I[0], I[1], I[2], 0] += values[i]
        return data

    def events_as_sparse_two_channel(self, n_layers=None, scaling=1):
        if n_layers is None:
            n_layers = self.num_events
        indices = self.event_indices(n_layers=n_layers, scaling=scaling)
        indices = np.hstack((indices, self.events["polarity"].values.reshape(
            (-1, 1))))
        return indices
        # return np.vstack(set(map(tuple, indices)))

    def events_as_dense_two_channel(self, n_layers=None, scaling=1):
        if n_layers is None:
            n_layers = self.num_events
        data = np.zeros((self.camera_config.cols // scaling,
                         self.camera_config.rows // scaling,
                         n_layers,
                         2), dtype=np.bool)
        indices = self.events_as_sparse_two_channel(n_layers=n_layers,
                                                    scaling=scaling)
        # flat_indices = np.ravel_multi_index(indices.T, data.shape)
        # data.flat[flat_indices] = 1
        for I in indices:
            data[I[0], I[1], I[2], I[3]] = 1
        return data

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

    def events_as_sparse_tensor_rescaled(self, n_layers, scaling):
	# Extract indices in x, y, z order
        indices = self.events[["x", "y"]].as_matrix() / scaling
        indices = np.hstack(
            (indices.round().astype(np.uint16),
             np.arange(0, n_layers-1, (n_layers-1)/self.num_events).round().reshape((-1, 1))))

        # convert polarity from {False, True} to {-0.5, 0.5}
        values = self.events["polarity"].as_matrix() - 0.5

        shape = [(np.int64)(self.camera_config.cols / scaling),
                 (np.int64)(self.camera_config.rows / scaling),
                 n_layers-1]

        return tf.SparseTensorValue(indices,
                                    values,
                                    shape)


    @property
    def time_span(self):
        return [self.events.iloc[0]["time"],
                self.events.iloc[-1]["time"]]

