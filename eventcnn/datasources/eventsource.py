import numpy as np


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

    chunk_starts = np.arange(0, num_events, chunk_size)
    splits = {}
    for split_type, fraction in split_fractions.items():
        rows_in_chunk = np.arange(0,
                                  fraction * chunk_size - 1,
                                  events_per_block).reshape(
            (1, -1))
        rows = (rows_in_chunk + chunk_starts).reshape((-1))
        splits[split_type] = rows

    return Split(splits["training"],
                 splits["validation"],
                 splits["testing"])


class EventSource:
    def __init__(self, dataset,
                 testing_fraction,
                 validation_fraction,
                 events_per_block=100,
                 augmentation_functions=[lambda x: x]):
        self.dataset = dataset
        self.events_per_block = events_per_block
        self.augmentation_functions = augmentation_functions
        self.split = split_data(test_fraction, validation_fraction,
                                self.dataset.num_events,
                                events_per_block=events_per_block)

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
        for row in self.split.training:
            for fun in self.augmentation_functions:
                yield fun(self.dataset.event_block(row, self.events_per_block))

    def testing(self):
        for row in self.split.testing:
            yield self.dataset.event_block(row, self.events_per_block)

    def validation(self):
        for row in self.split.validation:
            yield self.dataset.event_block(row, self.events_per_block)


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


class EventSourceCombination:
    def __init__(self, sources):
        self.sources = sources

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
        return take_randomly(self.sources,
                             [s.num_training for s in self.sources])

    def _training_noshuffle(self):
        for source in self.sources:
            yield from source.training()

    def testing(self):
        for source in self.sources:
            yield from source.testing()

    def validation(self):
        for source in self.sources:
            yield from source.validation

