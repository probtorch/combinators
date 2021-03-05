""" try to bring this back in"""
import torch

class Tracker():
    def __init__(self, size, device, from_dict=None):
        self._history = {}
        self._size = size
        self._fill_value = float('nan')
        self._device = device

        if from_dict is not None:
            for key, value in from_dict.items():
                self.create_entry(key, value.shape[len(size):])
                self._history[key][[slice(0, s) for s in value.shape]] = value

    def create_entry(self, key, shape):
        self._history[key] = torch.full((*self._size, *shape), fill_value=self._fill_value, device=self._device)

    def add_entry(self, pos, key, value):
        self._history[key][pos] = value.detach()

    def add(self, pos, **kwargs):
        for key, value in kwargs.items():
            if key not in self._history:
                self.create_entry(key, value.shape)
            self.add_entry(pos, key, value)

    def __getitem__(self, key):
        return self._history[key]

    def history(self):
        return self._history
