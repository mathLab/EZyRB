import numpy as np

class Scale(object):

    def __init__(self, method, axis=None):
        self.available_methods = {
            'standardisation': self.standardisation,
            'mean': self.mean,
            'minmax': self.minmax,
            'unitvector': self.unitvector
        }
        self.method = method
        self.axis = axis

    @property 
    def method(self):
        return self._method

    @method.setter
    def method(self, method):
        if method not in self.available_methods:
            raise ValueError(
                '{} not available for {} object.'.format(method, self.__class__.__name__),
                'Please chose one between {}.'.format(' '.join(available_methods.keys())))

        self._method = method
        
    def standardisation(self, x):
        """
        :math:`x_{\text{scaled}} = \frac{x - \overline{x}}{\delta}
        """
        return (x - np.mean(x, axis=self.axis, keepdims=True))/np.std(x, axis=self.axis, keepdims=True)

    def mean(self, x):
        return (x - np.mean(x, axis=self.axis, keepdims=True))/(np.max(x, axis=self.axis, keepdims=True) - np.min(x, axis=self.axis, keepdims=True))

    def minmax(self, x):
        self.attributes = {
            'min': np.min(x),
            'max': np.max(x)}

        return (x - np.min(x))/(np.max(x) - np.min(x))

    def inverse(self, x):
        a = self.attributes 
        return x * (a['max'] - a['min']) + a['min']
    def unitvector(self, x):
        return x/np.linalg.norm(x, axis=self.axis, keepdims=True)

    def __call__(self, x):
        print(x)
        return self.available_methods[self.method](x)
