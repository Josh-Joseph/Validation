import logging
logger = logging.getLogger( __name__ )

import datetime

import numpy as np
import scipy.stats
import collections


##========================================================================

##
# The gibbs sampler for the simple idealizes generative model for
# ow a classifier's accuracy is defined.
#
# This is the first graphical model in
# http://jmlr.org/proceedings/papers/v48/platanios16.pdf
class FlatIndependentAccuracyGibbsSampler( object ):

    ##
    # The samples returned by this
    SampleType = collections.namedtuple( 
        'FlatIndependentAccuracyGibbsSamplerSample',
        [ 'p',
          'l',
          'e' ] )

    ##
    #
    # @param classifiers : A List[ f(data_poitn) ] where the function returns
    #                      the class, 0 or 1, for the given data_point
    # @param data : a List[ data_point ] to be fed to the classifiers
    def __init__( self, 
                  classifiers, 
                  data, 
                  prior_alpha_p = 1.0,
                  prior_beta_p = 1.0,
                  prior_alpha_e = 1.0, 
                  prior_beta_e = 1.0,):
        self.classifiers = classifiers
        self.data = data
        self.prior_beta_p = prior_beta_p
        self.prior_alpha_p = prior_alpha_p
        self.prior_alpha_e = prior_alpha_e
        self.prior_beta_e = prior_beta_e

        # debg logging stas misc
        self.num_gibbs_sweeps = 0

        # counts/max indeices
        self.S = len(data)
        self.N = len(classifiers)
        
        # things we are reasoning about
        self.rv_p = None
        self.rv_labels = [ None ] * self.S
        self.rv_es = [ None ] * self.N
        self._initialize_rvs()

        # observations :-)
        self.f = np.zeros( (len(data), len(classifiers)) )
        for i,s in enumerate(data):
            for j,c in enumerate(classifiers):
                self.f[ i, j ] = c( s )

        
    ##
    #
    def _initialize_rvs(self):
        self.rv_p = scipy.stats.beta(self.prior_alpha_p, self.prior_beta_p).rvs()
        self.rv_labels = scipy.stats.bernoulli(self.rv_p).rvs(size=self.S)
        self.rv_es = scipy.stats.beta(self.prior_alpha_e,self.prior_beta_e).rvs(size=self.N)


    ##
    #
    def _single_gibbs_sweep(self):

        start_time = datetime.datetime.now()

        # sample p
        sigma_l = np.sum(self.rv_labels)
        self.rv_p = scipy.stats.beta(
            self.prior_alpha_p + sigma_l,
            self.prior_beta_p + self.S - sigma_l ).rvs()
        logger.debug( "p ~ Beta( {0}, {1} ), sigma_l = {2} , p = {3}".format(
            self.prior_alpha_p + sigma_l,
            self.prior_beta_p + self.S - sigma_l,
            sigma_l,
            self.rv_p ) )

        # Sample *each* label independently
        for i in xrange(self.S):

            # compute the probability of the label (bernoulli) in log space
            log_i = 0.0
            log_pi_i = 0.0
            for j in xrange(self.N):
                ej = self.rv_es[j]
                if self.f[i,j] != self.rv_labels[i]:
                    log_pi_i += np.log( ej )
                else:
                    log_pi_i += np.log( 1.0 - ej )
            if self.rv_labels[i] == 1:
                log_i = np.log( self.rv_p ) + log_pi_i
            else:
                log_i = np.log( 1.0 - self.rv_p ) + log_pi_i
            
            # sample from bernoulli
            p_i = np.exp( log_i )
            self.rv_labels[ i ] = scipy.stats.bernoulli(p_i).rvs()
            logger.debug( "l_{0} ~ Bern( {1} )  log(pi_{0}) = {2} , l_{0} = {3}".format(
                i,
                p_i,
                log_pi_i,
                self.rv_labels[i]) )

        # Sample teh classifier accuracies
        for j in xrange(self.N):
            sigma_j = 0.0
            for i in xrange(self.S):
                if self.f[i,j] != self.rv_labels[i]:
                    sigma_j += 1.0
            self.rv_es[j] = scipy.stats.beta(
                self.prior_alpha_e + sigma_j,
                self.prior_beta_e + self.S - sigma_j ).rvs()
            logger.debug( "e_{0} ~ Beta( {1}, {2} ) , sigma_{0} = {3}, e_{0} = {4}".format(
                j,
                self.prior_alpha_e + sigma_j,
                self.prior_beta_e + self.S - sigma_j,
                sigma_j,
                self.rv_es[j]))

        end_time = datetime.datetime.now()
        self.num_gibbs_sweeps += 1

        # log stuff
        logger.info( "gibbs sweep {0}: {1} seconds".format(
            self.num_gibbs_sweeps,
            (end_time - start_time).total_seconds() ) )
        

    ##
    #
    def sample(self, n=None, every_n=100):
        if n is None:
            self._single_gibbs_sweep()
            return self._create_sample_from_state()
        samples = []
        skip_counter = 0
        num = 0
        while num < n:
            self._single_gibbs_sweep()
            skip_counter += 1
            if skip_counter >= every_n:
                samples.append( self._create_sample_from_state() )
                skip_counter = 0
                num += 1
        return samples

    ##
    #
    def _create_sample_from_state(self):
        return FlatIndependentAccuracyGibbsSampler.SampleType(
            p = np.copy(self.rv_p),
            l = np.copy(self.rv_labels),
            e = np.copy(self.rv_es) )
        

##========================================================================
##========================================================================

##
# Generate sample fake data
#
# p is probability of label being 1
def generate_fake_data( n = 1000, p = 0.5 ):
    data = []
    for i in xrange(n ):
        x = np.random.random()
        l = 0
        if x < p:
            l = 1
        data.append( (x, l ) )
    return data

##========================================================================

##
# generates a fake classifier fro the dake data with given accuracy
def generate_fake_classifier( wanted_accuracy ):
    rv = scipy.stats.bernoulli( 1.0 - wanted_accuracy )
    return lambda x: (x[1] + rv.rvs()) % 2

##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
##========================================================================
