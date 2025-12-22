"""Module for the Plugin abstract class."""

from abc import ABC


class Plugin(ABC):
    """
    The abstract Plugin class for ROM preprocessing and postprocessing.
    
    All plugin classes should inherit from this class and override the
    methods corresponding to the stages where they need to intervene.
    """
    def fit_preprocessing(self, rom):
        """
        Execute before the fit process begins.
        
        :param ReducedOrderModel rom: The ROM instance.
        """
        pass

    def fit_before_reduction(self, rom):
        """
        Execute before the reduction step during fit.
        
        :param ReducedOrderModel rom: The ROM instance.
        """
        pass

    def fit_after_reduction(self, rom):
        """
        Execute after the reduction step during fit.
        
        :param ReducedOrderModel rom: The ROM instance.
        """
        pass
    
    def fit_before_approximation(self, rom):
        """
        Execute before the approximation step during fit.
        
        :param ReducedOrderModel rom: The ROM instance.
        """
        pass

    def fit_after_approximation(self, rom):
        """
        Execute after the approximation step during fit.
        
        :param ReducedOrderModel rom: The ROM instance.
        """
        pass

    def fit_postprocessing(self, rom):
        """
        Execute after the fit process completes.
        
        :param ReducedOrderModel rom: The ROM instance.
        """
        pass

    def predict_preprocessing(self, rom):
        """
        Execute before the prediction process begins.
        
        :param ReducedOrderModel rom: The ROM instance.
        """
        pass

    def predict_before_approximation(self, rom):
        """
        Execute before the approximation step during prediction.
        
        :param ReducedOrderModel rom: The ROM instance.
        """
        pass

    def predict_after_approximation(self, rom):
        """
        Execute after the approximation step during prediction.
        
        :param ReducedOrderModel rom: The ROM instance.
        """
        pass

    def predict_before_expansion(self, rom):
        """
        Execute before the expansion step during prediction.
        
        :param ReducedOrderModel rom: The ROM instance.
        """
        pass
    
    def predict_after_expansion(self, rom):
        """
        Execute after the expansion step during prediction.
        
        :param ReducedOrderModel rom: The ROM instance.
        """
        pass

    def predict_postprocessing(self, rom):
        """
        Execute after the prediction process completes.
        
        :param ReducedOrderModel rom: The ROM instance.
        """
        pass



