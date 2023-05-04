from abc import abstractmethod, ABCMeta

import numpy as np


class Array:

    def __init__(self, data):
        self.__data = data
        if hasattr(data, 'dtype'):
            self.dtype = data.dtype
        else:
            tmp = data
            while isinstance(tmp, (list, tuple)):
                tmp = tmp[0]
            self.dtype = type(tmp)

    @property
    def data(self):
        return self.__data

    def __len__(self):
        return len(self.__data)

    def get_data(self):
        return self.__data

    def __getidx__(self, arg):
        return self.__data[arg]

    def __sliceiter(self, arg):
        return iter(range(*arg.indices(len(self))))

    def __getitem__(self, arg):
        if isinstance(arg, list):
            idx = []
            for i in arg:
                if isinstance(i, slice):
                    idx.extend(iter(self.__sliceiter(i)))
                else:
                    idx.append(i)
            return np.fromiter((self.__getidx__(x) for x in idx), dtype=self.dtype)
        elif isinstance(arg, slice):
            return np.fromiter((self.__getidx__(x) for x in self.__sliceiter(arg)), dtype=self.dtype)
        elif isinstance(arg, tuple):
            return (self.__getidx__(arg[0]), self.__getidx__(arg[1]))
        else:
            return self.__getidx__(arg)


class AbstractSortedArray(Array, metaclass=ABCMeta):
    '''
    An abstract class for representing sorted array
    '''

    @abstractmethod
    def find_point(self, val):
        pass

    def get_data(self):
        return self

    def __lower(self, other):
        return self.find_point(other)

    def __upper(self, other):
        ins = self.__lower(other)
        while self[ins] == other:
            ins += 1
        return ins

    def __lt__(self, other):
        ins = self.__lower(other)
        return slice(0, ins)

    def __le__(self, other):
        ins = self.__upper(other)
        return slice(0, ins)

    def __gt__(self, other):
        ins = self.__upper(other)
        return slice(ins, len(self))

    def __ge__(self, other):
        ins = self.__lower(other)
        return slice(ins, len(self))

    @staticmethod
    def __sort(a):
        return a[0] if isinstance(a, tuple) else a

    def __eq__(self, other):
        if isinstance(other, list):
            ret = []
            for i in other:
                eq = self == i
                ret.append(eq)
            ret = sorted(ret, key=self.__sort)
            tmp = []
            for i in range(1, len(ret)):
                a, b = ret[i - 1], ret[i]
                if isinstance(a, tuple):
                    if isinstance(b, tuple) and a[1] >= b[0]:
                        b[0] = a[0]
                    elif isinstance(b, tuple) or b > a[1]:
                        tmp.append(slice(*a))
                    elif b == a[1]:
                        a[1] == b + 1
                    else:
                        ret[i] = a
                elif (
                    isinstance(b, tuple)
                    and a < b[0]
                    or not isinstance(b, tuple)
                    and b - a != 1
                ):
                    tmp.append(a)
                elif not isinstance(b, tuple):
                    ret[i] = (a, b)
            if isinstance(ret[-1], tuple):
                tmp.append(slice(*ret[-1]))
            else:
                tmp.append(ret[-1])
            return tmp
        elif isinstance(other, tuple):
            ge = self >= other[0]
            ge = ge.start
            lt = self < other[1]
            lt = lt.stop
            return ge if ge == lt else slice(ge, lt)
        else:
            lower = self.__lower(other)
            upper = self.__upper(other)
            d = upper - lower
            if d == 0:
                return None
            elif d == 1:
                return lower
            else:
                return slice(lower, upper)

    def __ne__(self, other):
        eq = self == other
        if isinstance(eq, tuple):
            return [slice(0, eq[0]), slice(eq[1], len(self))]
        else:
            return [slice(0, eq), slice(eq + 1, len(self))]


class SortedArray(AbstractSortedArray):
    '''
    A class for wrapping sorted arrays. This class overrides
    <,>,<=,>=,==, and != to leverage the sorted content for
    efficiency.
    '''

    def __init__(self, array):
        super().__init__(array)

    def find_point(self, val):
        return np.searchsorted(self.data, val)


class LinSpace(SortedArray):

    def __init__(self, start, stop, step):
        self.start = start
        self.stop = stop
        self.step = step
        self.dtype = float if any(isinstance(s, float) for s in (start, stop, step)) else int
        self.__len = int((stop - start) / step)

    def __len__(self):
        return self.__len

    def find_point(self, val):
        nsteps = (val - self.start) / self.step
        fl = int(nsteps)
        return fl if fl == nsteps else int(fl + 1)

    def __getidx__(self, arg):
        return self.start + self.step * arg
