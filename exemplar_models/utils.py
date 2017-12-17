import numpy as np
import tensorflow as tf

from abc import ABCMeta, abstractmethod
from collections import namedtuple

WEIGHT_DECAY_VARS = 'weight_decay_vars'


def linear(input, dout=None, bias=True, init_scale=1.0, name='', weight_decay=True):
    _, din = input.get_shape()
    init = init_scale/np.sqrt(int(din))
    Winit = np.random.uniform(low=-init, high=init, size=(din, dout)).astype(np.float32)
    W = tf.get_variable('W'+name, initializer=tf.constant(Winit))
    if weight_decay:
        tf.add_to_collection(WEIGHT_DECAY_VARS, W)

    b = 0
    if bias:
        b = tf.get_variable('b'+name, initializer=tf.constant(np.zeros(dout).astype(np.float32)))

    return tf.matmul(input, W)+b

def crelu(input):
    return tf.concat(1, [tf.nn.relu(input), tf.nn.relu(-input)])

def linear_id(input, bias=True, init_scale=1e-4, name='', weight_decay=True):
    _, din = input.get_shape()
    din = int(din)
    init = init_scale/np.sqrt(int(din))
    Winit = np.random.uniform(low=-init, high=init, size=(din, din)).astype(np.float32)
    W = tf.get_variable('W'+name, initializer=tf.constant(Winit))
    if weight_decay:
        tf.add_to_collection(WEIGHT_DECAY_VARS, W)
    W += tf.constant(np.eye(din).astype(np.float32))

    b = 0
    if bias:
        b = tf.get_variable('b'+name, initializer=tf.constant(np.zeros(din).astype(np.float32)))
    return tf.matmul(input, W)+b


def dump_graph(logdir='graph_log', graph=None):
    """
    Writes a graph into a protobuf file that can be visualized by tensorboard.
    """
    """
    if graph is None:
        graph = tf.get_default_graph()
    graphpb_txt = str(graph.as_graph_def())
    with open(fname, 'w') as f:
        f.write(graphpb_txt)
    """
    session = tf.get_default_session()

    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter(logdir, session.graph)
    #writer.add_summary(merged, 0)
    writer.flush()

def get_weight_decay_loss(scope):
    vars = tf.get_collection(WEIGHT_DECAY_VARS, scope=scope)
    return tf.reduce_sum([tf.reduce_sum(tf.square(var)) for var in vars])


def print_shape(**kwargs):
    for key in kwargs:
        print('%s:%s' % (key, kwargs[key].get_shape()))


def assert_shape(tensor, shape):
    assert tensor.get_shape().is_compatible_with(shape), "Bad shape. Tensor=%s, Expected:%s" \
                                                         % (tensor.get_shape(), shape)


class BatchSampler(object):
    def __init__(self, data, cast_to_list=False):
        self.data = data
        self.cast_to_list = cast_to_list

    def random_batch(self, batch_size=5):
        batch_idx = np.random.randint(0, len(self.data), size=batch_size)
        batch = [self.data[idx] for idx in batch_idx]
        if self.cast_to_list:
            batch = ListDataset(batch)
        return batch

    def with_replacement(self, batch_size=5, max_itr=float('inf')):
        itr = 0
        while True:
            yield self.random_batch(batch_size=batch_size)
            itr += 1
            if itr >= max_itr:
                break


def to_dataset(l):
    if isinstance(l, Dataset):
        return l
    return ListDataset(l)


class Dataset(object):
    """
    List-like object for managing lists of data.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError()

    @abstractmethod
    def as_list(self):
        raise NotImplementedError()

    @property
    def stack(self):
        """
        Returns a function that stacks all elements of a particular attribute.

        Ex.
        dataset.stack.attr1

        Outputs:
        np.array([ data[0].attr1, data[1].attr1, ... ])

        """
        data_list = self.as_list()
        class Dispatch(object):
            def __init__(self):
                pass
            def __getattr__(self, name):
                values = [getattr(data, name) for data in data_list]
                return np.array(values).astype(FLOAT_X)
        return Dispatch()

    @property
    def concat(self):
        """
        Returns a function that all elements of a particular attribute.

        Ex.
        dataset.concat.attr1

        Outputs:
        np.concatenate([ data[0].attr1, data[1].attr1, ... ])

        """
        data_list = self.as_list()

        class Dispatch(object):
            def __init__(self):
                pass

            def __getattr__(self, name):
                values = [getattr(data, name) for data in data_list]
                return np.concatenate(values).astype(FLOAT_X)

        return Dispatch()


class ListDataset(Dataset):
    """
    >>> from collections import namedtuple
    >>> DataPoint = namedtuple('DataPoint', ['x1', 'x2'])
    >>> dataset = ListDataset([DataPoint([1,1], [0,0]), DataPoint([1,2], [0,1])])
    >>> dataset.stack.x1
    array([[ 1.,  1.],
           [ 1.,  2.]])
    >>> dataset[0:1]
    ListDataset[DataPoint(x1=[1, 1], x2=[0, 0])]
    """
    def __init__(self, l):
        super(ListDataset, self).__init__()
        self.data = l

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return ListDataset(self.data[idx])
        return self.data[idx]

    def as_list(self):
        return self.data

    def __repr__(self):
        return 'ListDataset'+repr(self.data)


class FIFOBuffer(Dataset):
    """
    Limited capacity, first-in first-out buffer

    >>> buf = FIFOBuffer(capacity=2)
    >>> buf.append('a').append('b')
    Buffer['a', 'b']
    >>> buf.append('c')
    Buffer['c', 'b']
    >>> buf.append('d').append('e')
    Buffer['e', 'd']
    """
    def __init__(self, capacity=100):
        super(FIFOBuffer, self).__init__()
        self._buffer = [None] * int(capacity)
        self.C = int(capacity)
        self.active_idx = 0
        self.N = 0

    def append(self, datum):
        old_val = self._buffer[self.active_idx]
        self._buffer[self.active_idx] = datum
        self.active_idx = (self.active_idx+1) % self.C
        self.N = min(self.C, self.N+1)
        return old_val

    def append_all(self, collection):
        for data in collection:
            self.append(data)
        return self

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self._buffer[idx]

    def __repr__(self):
        return 'Buffer'+repr(self._buffer[:self.N])

    def __str__(self):
        return 'Buffer'+str(self._buffer[:self.N])

    def as_list(self):
        return self._buffer[:self.N]


