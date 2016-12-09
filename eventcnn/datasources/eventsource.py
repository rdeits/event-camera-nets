import numpy as np
import tensorflow as tf
import os
import json
import h5py


class Split:
    def __init__(self,
                 training,
                 validation,
                 testing):
        self.training = training
        self.validation = validation
        self.testing = testing


def split_data(testing_fraction,
               validation_fraction,
               num_events,
               events_per_block=100,
               chunk_size=1000000):
    """
    Generate train-test-validate split from the data. Because
    motion at the beginning of the dataset may be different from
    motion at the end, it doesn't make sense to just take the
    first N rows as training data, the next M as validation, etc.
    Instead, we first divide the events into chunks, and then
    divide each chunk into training, validation, and test rows.

    events_per_block refers to the number of events in a single
    input to the network

    chunk_size is the number of events in a chunk, which will then
    be subdifived into training-test-validation examples.
    """
    assert testing_fraction > 0
    assert validation_fraction > 0
    assert testing_fraction + validation_fraction < 1
    assert chunk_size > events_per_block
    split_fractions = {
        "testing": testing_fraction,
        "validation": validation_fraction,
        "training": 1 - (testing_fraction + validation_fraction)
    }

    chunk_starts = np.arange(0, num_events - chunk_size + 1, chunk_size,
                             dtype=np.int64).reshape(
        (-1, 1))
    splits = {}
    for split_type, fraction in split_fractions.items():
        rows_in_chunk = np.arange(0,
                                  fraction * chunk_size - events_per_block - 1,
                                  events_per_block,
                                  dtype=np.int64).reshape(
            (1, -1))
        rows = (rows_in_chunk + chunk_starts).reshape((-1))
        splits[split_type] = rows

    return Split(splits["training"],
                 splits["validation"],
                 splits["testing"])


class CachedDenseEventSource:
    def __init__(self, folder):
        self.files = {"training": h5py.File(os.path.join(folder,
                                                         "training.h5")),
                      "testing": h5py.File(os.path.join(folder,
                                                        "testing.h5")),
                      "validation": h5py.File(os.path.join(folder,
                                                           "validation.h5"))}

    @property
    def rows(self):
        return self.files["training"]["events"].shape[2]

    @property
    def cols(self):
        return self.files["training"]["events"].shape[1]

    @property
    def event_slices(self):
        return self.files["training"]["events"].shape[3]

    @property
    def num_training(self):
        return self.files["training"]["events"].shape[0]

    @property
    def num_testing(self):
        return self.files["testing"]["events"].shape[0]

    @property
    def num_validation(self):
        return self.files["validation"]["events"].shape[0]

    def training(self):
        return self.files["training"]

    def testing(self):
        return self.files["testing"]

    def validation(self):
        return self.files["validation"]


class EventSourceBase:

    def cache_dense_events(self, folder, n_layers=4, scaling=1):
        sources = [("training", self.training(shuffle=True)),
                   ("validation", self.validation()),
                   ("testing", self.testing())]
        os.makedirs(folder, exist_ok=True)
        for (name, source) in sources:
            event_sets = []
            label_sets = []
            num_blocks = 0
            for block in source:
                num_blocks += 1
                if num_blocks % 100 == 0:
                    print(name, num_blocks)
                event_sets.append(
                    block.events_as_dense_two_channel(n_layers=n_layers,
                                                      scaling=scaling))
                label_sets.append(block.delta_position)
            events = np.stack(event_sets)
            labels = np.stack(label_sets)
            with h5py.File(os.path.join(folder, name + ".h5"), "w") as file:
                event_dset = file.create_dataset("events",
                                                 data=events,
                                                 compression="lzf",
                                                 chunks=(1, *events.shape[1:]))
                labels_dset = file.create_dataset("labels",
                                                  data=labels)

    def write_records(self, folder, n_layers=None, scaling=1):
        os.makedirs(folder, exist_ok=True)
        def make_example(block):
            events = block.events_as_dense_two_channel(
                n_layers=n_layers,
                scaling=scaling)
            label = block.delta_position
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'label': tf.train.Feature(
                            float_list=tf.train.FloatList(value=label)),
                        'events': tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                 value=events.flatten().tolist()))
                    }))
            return example

        sources = [("training", self.training(shuffle=False)),
                   ("validation", self.validation()),
                   ("testing", self.testing())]
        for (name, source) in sources:
            writer = tf.python_io.TFRecordWriter(
                os.path.join(folder, "{:s}.tfrecords".format(name)))
            num_blocks = 0
            for block in source:
                example = make_example(block)
                writer.write(example.SerializeToString())
                num_blocks += 1
                print(name, num_blocks)
                if num_blocks > 20:
                    break
        with open(os.path.join(folder, "metadata.json"), "w") as f:
            json.dump({
                "rows": self.rows // scaling,
                "cols": self.cols // scaling,
                "event_layers": n_layers,
                "channels": 2}, f)


class EventSource(EventSourceBase):
    def __init__(self, dataset,
                 testing_fraction,
                 validation_fraction,
                 events_per_block=100,
                 augmentation_functions=[lambda x: x]):
        self.dataset = dataset
        self.events_per_block = events_per_block
        self.augmentation_functions = augmentation_functions
        self.split = split_data(testing_fraction, validation_fraction,
                                self.dataset.num_events,
                                events_per_block=events_per_block)

    @property
    def rows(self):
        return self.dataset.camera_config.rows

    @property
    def cols(self):
        return self.dataset.camera_config.cols

    @property
    def num_training(self):
        return len(self.split.training)

    @property
    def num_validation(self):
        return len(self.split.validation)

    @property
    def num_testing(self):
        return len(self.split.testing)

    def training(self, shuffle=True):
        if shuffle:
            rows = np.random.permutation(self.split.training)
        else:
            rows = self.split.training
        for row in rows:
            for fun in self.augmentation_functions:
                yield fun(self.dataset.event_block(row, self.events_per_block))

    def testing(self):
        for row in self.split.testing:
            yield self.dataset.event_block(row, self.events_per_block)

    def validation(self):
        for row in self.split.validation:
            yield self.dataset.event_block(row, self.events_per_block)

    def __add__(self, other):
        assert self.events_per_block == other.events_per_block
        return EventSourceCombination([self, other])


def take_randomly(iterables, lengths):
    """
    Yield from a random shuffle of all input iterables
    without trying to store the entire list of all values in
    memory. Requires the length of each iterable to passed in,
    since some iterators don't provide len().
    """
    iterators = [iter(s) for s in iterables]

    # Selectors is a vector of length (sum(lengths)) with
    # values in range(0, length(iterables)). We will use the
    # selectors to choose from which iterable to yield at each step.
    selectors = np.random.permutation(
        np.hstack([
            np.tile(i, lengths[i])
            for i in range(len(iterables))]))
    for selector in selectors:
        yield next(iterators[selector])


class EventSourceCombination(EventSourceBase):
    def __init__(self, sources):
        self.sources = sources

    @property
    def cols(self):
        return self.sources[0].cols

    @property
    def rows(self):
        return self.sources[0].rows

    @property
    def events_per_block(self):
        return self.sources[0].events_per_block

    @property
    def num_training(self):
        return sum(source.num_training for source in self.sources)

    @property
    def num_testing(self):
        return sum(source.num_testing for source in self.sources)

    @property
    def num_validation(self):
        return sum(source.num_validation for source in self.sources)

    def training(self, shuffle=True):
        if shuffle:
            return self._training_shuffle()
        else:
            return self._training_noshuffle()

    def _training_shuffle(self):
        return take_randomly(
            [s.training(shuffle=True) for s in self.sources],
            [s.num_training for s in self.sources])

    def _training_noshuffle(self):
        for source in self.sources:
            yield from source.training(shuffle=False)

    def testing(self):
        for source in self.sources:
            yield from source.testing()

    def validation(self):
        for source in self.sources:
            yield from source.validation

    def __add__(self, other):
        assert self.events_per_block == other.events_per_block
        return EventSourceCombination([self, other])

