import logging
logger = logging.getLogger( __name__ )

import operator
import itertools
import numpy as np

import dataset

####
#### All of the evaluation functions
#### of the form funcname( dataset, nn_algorithm ) -> score
####


##=============================================================================

##
# Returns the dimnesionality of an object
def dimensionality( x ):
    if hasattr( x, '__len__' ):
        return len(x)
    return 1

##=============================================================================

##
# Completely flatten a structured object.
# This can only lfatten through lists, tuples, and np.ndarray objects
# and iterables
def _can_flatten( x ):
    if isinstance( x, (list, tuple, np.ndarray) ):
        return True
def flatten( x ):
    res = []
    if _can_flatten( x ):
        for e in x:
            res.extend( flatten( e ) )
    else:
        res.append( x )
    return res

##=============================================================================

##
# A base class for objective functions
class Objective( object ):

    def __call__( self, dataset, model, **kw ):
        return self.evaluate( dataset, model, **kw )
    
    def evaluate( self, dataset, model, **kw ):
        raise NotImplementedError()

##=============================================================================

##
# An objective useful for discriminant models
#
# For discriminant objectives, we in fact most often compute
# a loss function that is based on the original and predicted
# ys hence we build that into this base class.
# Subclasses probably only need to override the loss() function
class DiscriminantModelObjective( Objective ):
    
    ##
    # Return the loss of the prediction from the model for hte dataset x
    # to the ys in the dataset
    def evaluate( self, dataset, model, **kw ):
        y_hat = model.predict( dataset.x )
        return self.loss( dataset.x, dataset.y, y_hat, **kw )

    ##
    # Given a sequence of y and a sequence of y_hat
    # to be treated as y=truth, y_hat=prediction
    # we also supply the sequence of x in needed
    #
    # @param x : List[ input ]
    # @param y : List[ output ]
    # @param y_hat : List[ prediction ]
    # @return loss for y_hat predictions of x given y is "truth"
    def loss( self, x, y, y_hat, **kw ):
        raise NotImplementedError()

##=============================================================================

##
# An object useful for generatirve models
class GenerativeModelObjective( Objective ):
    pass

##=============================================================================

##
# Eucledian distance squared
def _l2_distance( y, y_hat ):
    return np.linalg.norm( np.array(y) - np.array(y_hat) )

##
# A Standard total-sum of distance objective.
# This is most often used in regression problems with the l2 norm
# of the eucledian distance
class SumOfDistanceObjective( DiscriminantModelObjective ):
    
    def __init__( self, distance_function=_l2_distance ):
        self.distance_function = distance_function
        
    def loss( self, x, y, y_hat, **kw ):
        return np.sum([ self.distance_function( y0, y_hat0 ) 
                        for (y0, y_hat0) in zip( y, y_hat ) ])

##=============================================================================

##
# The Kolmologrov-Smirnov test between samples to determine
# if they both come from the same distribution.
# This is a standard test to check if a generative model
# has a distribution close to that of a dataset of samples.
#
# A multidimensional version of the Kolmorogov-Sminov Test
# G. Fasano and A Franceshini
# http://mnras.oxfordjournals.org/content/225/1/155.full.pdf+html
class ContinuousKolmorogovSmirnovObjective( GenerativeModelObjective ):
    
    def __init__(self):
        pass

    ##
    # Run the 2-samples Kolmogorov-Smirnov test with the dataset samples
    # and samples taken from the generative model.
    # We return the resulting statistic of the test, with lower meaning that
    # the samples came from the same distribution.
    # For critical values of the statistic, one must run monte-carlo simulation
    # to generate critical points, hence we only return hte statistic as the 
    # objective since we want to minimize it.
    def evaluate( self, dataset, model, **kw ):
        
        # What we want to evaluate is the maximum difference in 
        # the emprirical CDF from both hte dataset and the 
        # model

        # take samples from hte model
        n = len(dataset.x)
        model_samples = map(flatten, model.sample( n ) )

        # join samples of hte dataset
        ds_samples = map(flatten, zip( dataset.x, dataset.y ) )

        # We need to worry about the direciton of the CDF for 
        # multidimensional data
        if dimensionality( ds_samples[0] ) > 1:
            return self._multidimenaional_ks_test( ds_samples, model_samples )
        else:
            return self._onedimensional_ks_test( ds_samples, model_samples )


    ##
    # The original one-dimensional Kolmogorov-Smirnov test
    def _onedimensional_ks_test( self, a_samples, b_samples ):
        max_diff = 0.0
        for x in a_samples + b_samples:
            a_cdf = self._cdf( a_samples, x, [operator.le] )
            b_cdf = self._cdf( b_samples, x, [operator.le] )
            diff = np.abs( a_cdf - b_cdf )
            if diff > max_diff:
                max_diff = diff
        return max_diff

    ##
    # We take the maximum difference in the empirical CDFs
    # taking into account *all* possible CDF directions :-)
    # so this takes longer the higher dimensional our samples
    # are.
    def _multidimenaional_ks_test( self, a_samples, b_samples ):
        dim = dimensionality( a_samples[0] )
        max_diff = 0.0
        for x in a_samples + b_samples:
            for ops in itertools.product( [ operator.le, operator.ge ],
                                          repeat = dim ):
                a_cdf = self._cdf( a_samples, x, ops )
                b_cdf = self._cdf( b_samples, x, ops )
                diff = abs( a_cdf - b_cdf )
                if diff > max_diff:
                    max_diff = diff
        return max_diff

    ##
    # Computes the empirical CDF from samples for a particular x
    # We need to give the orintation of the CDF by giving the
    # comparison operators wanted for each dimension of the samples
    def _cdf( self, samples, x, ops ):
        num_found = 0
        n = len(samples)
        for s in samples:
            if self._all_ops( s, x, ops ):
                num_found += 1
        return float( num_found ) / float( n )



    ##
    # Apply the given comparator per dimension to the given values,
    # returns if all resulting in true
    def _all_ops( self, a, b, ops ):
        return np.all(map(lambda (a0,b0,ops0): ops0(a0,b0),
                          zip( a, b, ops ) ) )
        


    ##
    # Computes an approximate p-value for the kolmogorov-smirnov
    # test.
    # Here, we take *many* repeated sets of samples from the
    # generative model and compute the empirical CDF for the
    # given statistic
    #
    # @param max_d_stat : The max distance statistic, result of
    #                     a KS test between dataset and model
    # @param dataset : a dataset of asmples
    # @param model : A generative model (has a sample() method)
    #
    # @return pvalue
    def approximate_pvalue( self, max_d_stat, ds, model, iterations=100 ):
        n = len( ds.x )
        stats = []
        for i in xrange(iterations):
            
            # sample a new test dataset from model
            model_samples = model.sample(n=n)
            test_ds = dataset.dataset_from_samples( model_samples )
            
            # compute statistic for this dataset to model.
            # Note that these are *equal* distributions hence the null
            # hypothesis of the test is true
            stat = self.evaluate( test_ds, model )

            # store this statistic
            stats.append( stat )
        
        # Ok, approximate the statistic under the null hypothesis CDF
        # from the samples :-)
        num_more = 0
        for s in stats:
            if s >= max_d_stat:
                num_more += 1
        return float( num_more ) / float( iterations )

##=============================================================================
##=============================================================================
##=============================================================================
##=============================================================================
