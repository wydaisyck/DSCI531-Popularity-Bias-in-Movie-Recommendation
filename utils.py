import numpy as np
from itertools import chain

class KFold():
    """A basic cross-validation iterator.

    Each fold is used once as a testset while the k - 1 remaining folds are
    used for training.

    See an example in the :ref:`User Guide <use_cross_validation_iterators>`.

    Args:
        n_splits(int): The number of folds.
        random_state(int, RandomState instance from numpy, or ``None``):
            Determines the RNG that will be used for determining the folds. If
            int, ``random_state`` will be used as a seed for a new RNG. This is
            useful to get the same splits over multiple calls to ``split()``.
            If RandomState instance, this same instance is used as RNG. If
            ``None``, the current RNG from numpy is used. ``random_state`` is
            only used if ``shuffle`` is ``True``.  Default is ``None``.
        shuffle(bool): Whether to shuffle the ratings in the ``data`` parameter
            of the ``split()`` method. Shuffling is not done in-place. Default
            is ``True``.
    """

    def __init__(self, n_splits=5, random_state=None, shuffle=True):

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, data):
        """Generator function to iterate over trainsets and testsets.

        Args:
            data(:obj:`Dataset<surprise.dataset.Dataset>`): The data containing
                ratings that will be divided into trainsets and testsets.

        Yields:
            tuple of (trainset, testset)
        """

        if self.n_splits > len(data.raw_ratings) or self.n_splits < 2:
            raise ValueError('Incorrect value for n_splits={0}. '
                             'Must be >=2 and less than the number '
                             'of ratings'.format(len(data.raw_ratings)))

        # We use indices to avoid shuffling the original data.raw_ratings list.
        indices = np.arange(len(data.raw_ratings))



        start, stop = 0, 0
        for fold_i in range(self.n_splits):
            start = stop
            stop += len(indices) // self.n_splits
            if fold_i < len(indices) % self.n_splits:
                stop += 1

            raw_trainset = [data.raw_ratings[i] for i in chain(indices[:start],
                                                               indices[stop:])]
            raw_testset = [data.raw_ratings[i] for i in indices[start:stop]]

            trainset = data.construct_trainset(raw_trainset)
            testset = data.construct_testset(raw_testset)

            yield trainset, testset

    def get_n_folds(self):

        return self.n_splits