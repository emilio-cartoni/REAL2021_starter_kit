import numpy as np
from heapq import heappush, heappop
import itertools

class PriorityQueue(object):
    # Adapted from https://docs.python.org/3/library/heapq.html
    def __init__(self):
        self.pq = []                         # list of entries arranged in a heap
        self.entry_finder = {}               # mapping of tasks to entries
        self.REMOVED = '<removed-data>'      # placeholder for a removed data
        self.counter = itertools.count()     # unique sequence count

    def is_empty(self):
        return not self.entry_finder

    def enqueue(self, data, value):
        'Add a new data or update the priority of an existing data'
        count = next(self.counter)
        entry = [value, count, data]
        self.entry_finder[data] = entry
        heappush(self.pq, entry)

    def enqueue_with_replace(self, data, value):
        'Add a new data or update the priority of an existing data'
        if data in self.entry_finder:
            self.remove_data(data)
        count = next(self.counter)
        entry = [value, count, data]
        self.entry_finder[data] = entry
        heappush(self.pq, entry)

    def remove_data(self, data):
        'Mark an existing data as REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(data)
        entry[-1] = self.REMOVED

    def dequeue(self):
        'Remove and return the lowest priority data. Raise KeyError if empty.'
        while self.pq:
            priority, count, data = heappop(self.pq)
            if data is not self.REMOVED:
                del self.entry_finder[data]
                return data, priority
        raise KeyError('pop from an empty priority queue')

    def get_queue(self):
        return self.pq.copy()

    def get_queue_values(self):
        return np.take(self.pq, 0, axis=1)

    def get_queue_data(self):
        return np.take(self.pq, 2, axis=1)

    def replace_if_better(self, data, value):
        if self.is_empty():
            return False
        else:
            if data in self.entry_finder:
                entry = self.entry_finder[data]
                if entry[0] > value:
                    self.remove_data(data)
                else:
                    return False
            count = next(self.counter)
            entry = [value, count, data]
            self.entry_finder[data] = entry
            heappush(self.pq, entry)
            return True
