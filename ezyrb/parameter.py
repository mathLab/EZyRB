""" Module for parameter object """
import logging
import numpy as np

logger = logging.getLogger(__name__)


class Parameter:
    """
    Class for representing a parameter in the reduced order model.
    
    This class encapsulates parameter values and provides validation
    to ensure parameters are 1-dimensional arrays.
    
    :param array_like values: The parameter values as a 1D array.
    
    :Example:
    
        >>> import numpy as np
        >>> from ezyrb import Parameter
        >>> param = Parameter([1.0, 2.5, 3.0])
        >>> print(param.values)
        [1.  2.5 3. ]
        >>> param2 = Parameter(np.array([4.5, 5.5]))
        >>> print(param2.values)
        [4.5 5.5]
    """

    def __init__(self, values):
        """
        Initialize a Parameter object.
        
        :param array_like values: The parameter values. Can be a Parameter
            instance or an array-like object that can be converted to a 1D numpy array.
        """
        logger.debug("Initializing Parameter with values type: %s",
                     type(values))
        if isinstance(values, Parameter):
            self.values = values.values
            logger.debug("Copied values from existing Parameter")
        else:
            self.values = values
            logger.debug("Created Parameter with shape: %s",
                         np.asarray(values).shape)

    @property
    def values(self):
        """ Get the snapshot values. """
        return self._values

    @values.setter
    def values(self, new_values):
        """
        Set the parameter values with validation.
        
        :param array_like new_values: The new parameter values.
        :raises ValueError: If the new values are not a 1D array.
        """
        if np.asarray(new_values).ndim != 1:
            logger.error("Invalid parameter dimension: %d (expected 1D)",
                         np.asarray(new_values).ndim)
            raise ValueError('only 1D array are usable as parameter.')

        self._values = np.asarray(new_values)
        logger.debug("Parameter values set with shape: %s",
                     self._values.shape)
