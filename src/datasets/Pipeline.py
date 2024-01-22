import torch

class Pipeline:
    def __init__(self, steps):
        """ Pre- and postprocessing pipeline. """
        self.steps = steps

    def transform(self, x, until=None):
        x = x.clone()
        for n, step in self.steps:
            if n == until:
                break
            x = step.transform(x)
        return x

    def inverse_transform(self, x, until=None):
        for n, step in self.steps[::-1]:
            if n == until:
                break
            x = step.inverse_transform(x)
        return x


class StandardScalerTS():
    """ Standard scales a given (indexed) input vector along the specified axis. """

    def __init__(self, axis=(1)):
        self.mean = None
        self.std = None
        self.axis = axis

    def transform(self, x):
        if self.mean is None:
            self.mean = torch.mean(x, dim=self.axis)
            self.std = torch.std(x, dim=self.axis)
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def inverse_transform(self, x):
        return x * self.std.to(x.device) + self.mean.to(x.device)

class MinMaxTS():
    """ Min-Max scales a given (indexed) input vector along the specified axis. """
    def __init__(self, axis=(1)):
        self.min = None
        self.max = None
        self.axis = axis

    def transform(self, x):
        if self.min is None:
            self.min = torch.min(x, dim=self.axis)
        if self.max is None:
            self.max = torch.max(x, dim=self.axis)

        return (x - self.min.to(x.device)) / (self.max - self.min).to(x.device)

    def inverse_transform(self, x):
        return x * (self.max - self.min).to(x.device) + self.min.to(x.device)