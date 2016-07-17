import logging 
logger = logging.getLogger( __name__ )

import collections

import numpy as np
import scipy.stats


####
#### This module contians a bunch of generate_* functions which return
#### Dataset objects of different types/lengths/etc.
####
#### General Usage:
####     import dataset
####     gen = dataset.Polynomial( <params> )
####     dset = gen( num_samples )


##=============================================================================

##
# A dataset of paired (x,y).
# The type will be denoted Dataset[(x-type,y-type)] or
# Dataset[*] to mean any dataset
Dataset = collections.namedtuple( "Dataset",
                                  ['x',
                                   'y',
                                   'identifier'] )

##=============================================================================
##=============================================================================

##
# Set the seed for future random number generator calls
def _set_seed( seed ):
    np.random.seed( seed )

##=============================================================================

##
# A generator is an object which generates datasets.
#
# Dataset Generators have type: # -> Dataset[*]
# where # stands for numebr of samples and the resulting object is
# a Dataset object with some form x,y pairs inside (* means anything)
# Subclasses will have specific resutlign Dataset types
class DatasetGenerator( object ):

    def __call__( self,
                  num_samples,
                  **kw ):
        return self.generate( num_samples, **kw )
    
    def generate( self,
                  num_samples, 
                  seed=None,
                  **kw ):
        raise NotImplementedError()

##=============================================================================

##
# A generator for datasets sampled from a polynomial
#
# The type of this generator is: # -> Dataset[( float in domain, float )]
class Polynomial( DatasetGenerator ):
    def __init__( self, 
                  poly,
                  noise_sigma = 1.0,
                  domain = [ -2.0, 2.0 ],
                  identifier = None ):
        self.poly = poly
        self.noise_sigma = noise_sigma
        self.domain = domain
        self.identifier = identifier

    def generate( self,
                  num_samples,
                  seed=None,
                  **kw ):
        return _generate_from_polynomial(
            num_samples,
            self.poly,
            noise_sigma = self.noise_sigma,
            domain = self.domain,
            seed = seed,
            identifier = self.identifier )

##=============================================================================

##
# A generator from a random polynomial of the given order
#
# The type of this generator is: # -> Dataset[( float in domain, float )]
class RandomPolynomial( DatasetGenerator ):
    def __init__( self,
                  order = 3,
                  noise_sigma = 1.0,
                  domain = [ -2.0, 2.0 ],
                  identifier = None ):

    def generate( self,
                  num_samples,
                  seed=None,
                  **kw ):
        return _generate_from_random_polynomial(
            num_samples,
            order = self.order,
            noise_sigma = self.noise_sigma,
            domain = self.domain,
            seed = seed,
            identifier = self.identifier )
        

##=============================================================================

##
# A generator which create a dataset from a random mixture of
# the given generators.
# The resuting dataset for this generator contains Y's which are
# pairs of the original Y and the INDEX of the generator used to
# create that particular sample
#
# For genrators of type: # -> Dataset[(X,Y)]
# This becomes a generator of type: # -> Dataset[( X, (Y, index) )]
#
# This is a *structured* addition of an index to Y not a flat
# appending to Y in order to allow for arbitrary
# mixtures of generator types :-)
class Mixture( DatasetGenerator ):
    
    def __init__( self,
                  generators,
                  weights = 1.0 ):
        self.generators = generators
        if isinstance( weights, (list,tuple,np.ndarray) ):
            self.weights = weights
        else:
            self.weights = [ weights ] * len(generators)
        self.norm_weights = np.array( self.weights )
        self.norm_weights /= np.sum( self.norm_weights )


    def generate( self,
                  num_samples,
                  seed = None,
                  **kw):
        _set_seed( seed )
        GN = len(self.generators)

        # generate number of samples proportional to the weight
        # of teh generator
        samples = []
        identifiers = []
        for gen_i in xrange( GN ):
            num_i = int( np.ceil( num_samples * self.norm_weights[ gen_i ] ) )
            ds = self.generators[ gen_i ].generate( num_i, seed=None )
            identifiers.append( ds.identifier )
            for (x,y) in zip( ds.x, ds.y ):
                
                # append generator index to orignal y
                samples.append( ( x, ( y, num_i ) ) )
        
        # create composite identifier
        identifier = "Mix{"
        for gen_i, ident in enumerate(identifiers):
            identifiers += "{0}/{1:0.3f},".format( 
                ident,
                self.norm_weights[ gen_i ] )
        identifier += "}"

        # shuffle the data
        np.random.shuffle( samples )
        
        # cut off any ceiling overshoots
        samples = samples[:num_samples]
        
        # return new dataset
        return Dataset(
            identifier = identifier,
            x = map(lambda s: s[0], samples),
            y = map(lambda s: s[1], samples) )
    

##=============================================================================
##=============================================================================
##=============================================================================
##=============================================================================
##=============================================================================
##=============================================================================
##=============================================================================
##=============================================================================

##
# Generate a datasset from a given polynomial
# with gaussian additive independent noise
# The domain of hte polynomial can also be given
def _generate_from_polynomial(
        num_samples,
        poly,
        noise_sigma = 1.0,
        domain = [0.0, 10.0],
        seed = None,
        identifier = None):

    if identifier is None:
        identifier = "Poly({0}, sigma={1}{2}, #{3})".format(
            poly,
            noise_sigma,
            "@{0}".format(seed) if seed is not None else "",
            num_samples )
    _set_seed( seed )

    x_rv = scipy.stats.uniform( loc=domain[0], scale=domain[1]-domain[0] )
    noise_rv = scipy.stats.norm( scale=noise_sigma )

    return _generate_dataset_aux(
        identifier,
        num_samples,
        poly,
        lambda: x_rv.rvs(),
        lambda: noise_rv.rvs() )

##=============================================================================

##
# Generate dataset from a *random* polynomial of the given order.
# Add given independent additive gaussian noise with given sigma.
# The domain for the polynomial can be given
def _generate_from_random_polynomial(
        num_samples,
        order = 3,
        coeff_domain = [ -10.0, 10.0 ],
        domain = [-2.0, 2.0],
        noise_sigma = 1.0,
        seed = None ):
    
    _set_seed( seed )

    # generat teh polynomial
    cdelta = coeff_domain[1] - coeff_domain[0]
    poly = np.poly1d( coeff_domain[0] + np.random.random( order ) * cdelta )

    identifier = "RandomPoly({0}, sigma={1}{2}, #{3})".format(
        poly,
        noise_sigma,
        "@{0}".format(seed) if seed is not None else "",
        num_samples )


    # return dataset
    return generate_from_polynomial(
        num_samples,
        poly,
        domain = domain,
        noise_sigma = noise_sigma,
        seed = seed,
        identifier = identifier )

##=============================================================================
##=============================================================================
##=============================================================================

##
# Generate a dataset from a python function,
# a domain variable sampler,
# and a noise sampler
def _generate_dataset_aux(
        identifier,
        num_samples,
        f,
        x_sampler,
        noise_sampler ):

    xs = []
    ys = []
    for i in xrange( num_samples ):
        x = np.array( x_sampler() )
        y = np.array( f( x ) )
        e = np.array( noise_sampler() )
        xs.append( x )
        ys.append( y + e )
        
    return Dataset(
        identifier = identifier,
        x = np.array( xs ),
        y = np.array( ys ) )
        
    
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
##=============================================================================
