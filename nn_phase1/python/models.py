import logging
logger = logging.getLogger( __name__ )

import numpy as np
import copy

####
#### The neural network models
####
#### each *must* subclass from eitehr DiscriminantNN or GenerativeNN
#### and must override fit() and predict/sample

##=============================================================================

##
# A model specification is a packages set of things needed to 
# generate a new model object which can later be trained and used.
#
# This just packages a class and constructor arguments :)
class ModelSpecification( object ):
    def __init__( self, model_class, model_parameters, model_kwds = {} ):
        self.model_class = model_class
        self.model_parameters = model_parameters
        self.model_kwds = model_kwds

    def create( self, **kw ):
        return self.model_class( *self.model_parameters, **self.model_kwds)

    def __call__(self,**kw):
        return self.create( **kw )

    def __str__(self):
        return "Spec[{0}( {1}, {2} )]".format(
            self.model_class,
            self.model_parameters,
            self.model_kwds )

    def __repr__(self):
        return self.__str__()

##
# Untility class to easily create specs
def spec( model_class, *params, **kw ):
    return ModelSpecification( model_class, params, model_kwds=kw )

##=============================================================================

##
# Test models

class Constant( object ):
    def __init__(self, value):
        self.value = value
    def fit( self, dataset, **kw ):
        pass
    def predict( self, x, **kw ):
        return [ self.value ] * len(x)
    

class NearestNeighbor( object ):
    def __init__(self):
        self.ds = None

    def fit( self, dataset, visualize=False, **kw ):
        self.ds = copy.deepcopy( dataset )

    def predict( self, x, **kw ):
        y_hat = []
        for xi in x:
            y_hat.append( self._nn_y( xi ) )
        return y_hat

    def _nn_y( self, x ):
        min_dist = np.inf 
        min_sample = None
        for dx,dy in zip( self.ds.x, self.ds.y ):
            d = np.linalg.norm( np.array(x) - np.array(dx) )
            if d < min_dist:
                min_dist = d
                min_sample = ( dx, dy )
        return min_sample[1]

class BootstrapResampler( object ):
    def __init__(self):
        self.ds = None

    def fit( self, dataset, visualize=False, **kw ):
        self.ds = dataset
        
    def sample( self, n=None, **kw ):
        if n is None:
            return self._sample()
        return np.array([ self._sample() for i in xrange(n)])

    def _sample(self):
        n = len(self.ds.x)
        idx = np.random.choice(n)
        return ( self.ds.x[idx],
                 self.ds.y[idx] )

##=============================================================================

class NeuralNetwork( object ):
    
    ##
    # Takes in a dataset and fits the internal model parmaeters
    def fit( dataset, visualize=False, **kw ):
        raise NotImplementedError()
        

##=============================================================================

class DiscriminantNeuralNetwork( NeuralNetwork ):

    ##
    # Takes in a sequence of inputs x and returns 
    # sequence of outputs y_hat, one for each x
    def predict( x, **kw ):
        raise NotImplementedError()


##=============================================================================

class GenerativeNeuralNetwork( NeuralNetwork ):

    ##
    # Returns a sample from the generative model.
    def sample( n=None, **kw ):
        raise NotImplementedError()

##=============================================================================
##=============================================================================
##=============================================================================
##=============================================================================
##=============================================================================
##=============================================================================
##=============================================================================
##=============================================================================
##=============================================================================
##=============================================================================
##=============================================================================
##=============================================================================
##=============================================================================
##=============================================================================
##=============================================================================
##=============================================================================
##=============================================================================
##=============================================================================
##=============================================================================
##=============================================================================
