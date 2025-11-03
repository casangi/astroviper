"""
Module to hold the ReturnDict class. The ReturnDict class is a convenience
class around a regular nested dictionary, allowing for more flexible indexing
and seaerching. By keeping it as a dict underneath, it remains generic
and lightweight.
"""

from collections import namedtuple, OrderedDict

# Define the key structure
Key = namedtuple("Key", ["time", "pol", "chan"])

# A namedtuple is used for the keys to ensure immutability and easy comparison
# It can be treated as a regular tuple for indexing and comparisons


class ReturnDict:
    def __init__(self):
        self._data = OrderedDict()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    def add(self, value, time, pol, chan):
        key = Key(time, pol, chan)
        self.data[key] = value

    def sel(self, time=None, pol=None, chan=None):
        """Get all items matching the given criteria"""
        matches = []
        for key, value in self.data.items():
            if (
                (time is None or key.time == time)
                and (pol is None or key.pol == pol)
                and (chan is None or key.chan == chan)
            ):
                matches.append(value)

        if len(matches) == 1:
            return matches[0]
        return matches if matches else None

    def to_dict(self):
        return self.data

    def __repr__(self):
        lines = []
        for key, value in self.data.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)
