import logging
logger = logging.getLogger( __name__ )

import numpy as np
import itertools

import dataset
import validation
import models


##=============================================================================

##
# Runs a given model with a given dataset using a number of random
# split cross-validation runs
#
# This will return a list of the results of objective for the
# different cross-validation iterations.
def _random_cross_validation(
        dataset_generator,
        model_spec,
        objective,
        split = 0.5,
        total_data_samples = 1000,
        iterations = 10,
        seed = None):

    # first, set the seed
    np.random.seed( seed )

    # store resuts of objective
    results = []
    
    # iterate over the cross-validation runs
    for i in xrange( iterations ):
        
        # Generate test and train datasets
        num_train_samples = int( np.ceil(split * total_data_samples))
        num_test_samples = int( np.ceil( (1.0-split) * total_data_samples))
        train_ds = dataset_generator( num_train_samples )
        test_ds = dataset_generator( num_test_samples )
        
        # Create a new model and train it
        model = model_spec()
        model.fit( train_ds )
        
        # evaluate objective
        res = objective( test_ds, model )
        
        # store result
        results.append( res )

    # return results
    return results
        

##=============================================================================

##
# A specification for a batch of experiments
#
# A *batch* is defiend as the product combination of a set of
# datasets, models, objectives and cross-validation inputs.
class BatchSpecification( object ):

    ##
    # The dataset generators, model psecifications, objectives,
    # and cross validation arguments for a batch.
    # All arguments *must* be lists since a batch is defiend as 
    # the product of all of these arguments together :)
    #
    # If all arguments are 1-element lists, then this defiens a 
    # batch with only a single fixed experiment to run 
    def __init__( self,
                  dataset_generators,
                  model_specs,
                  objectives,
                  crossval_iterations,
                  crossval_samples,
                  crossval_splits,
                  seeds,
                  identifier = None):
        
        self.dataset_generators = dataset_generators
        self.model_specs = model_specs
        self.objectives = objectives
        self.crossval_iterations = crossval_iterations
        self.crossval_samples = crossval_samples
        self.crossval_splits = crossval_splits
        self.seeds = seeds
        self.identifier = identifier


##=============================================================================


##
# Runs a single batch of experiments
def _run_batch( spec ):
    results = []
    iteration = 0
    for ( ds_gen, 
          model_spec, 
          objective, 
          cv_iters, 
          cv_samples, 
          cv_split, 
          seed ) in itertools.product(
              spec.dataset_generators,
              spec.model_specs,
              spec.objectives,
              spec.crossval_iterations,
              spec.crossval_samples,
              spec.crossval_splits,
              spec.seeds ):
        
        res = _random_cross_validation(
            ds_gen,
            model_spec,
            objective,
            iterations = cv_iters,
            split = cv_split,
            total_data_samples = cv_samples,
            seed = seed )

        logger.info( "{0}Finished experiment run #{1}".format( 
            "[{0}] ".format( spec.identifier ) if spec.identifier is not None else "",
            iteration ) )

        results.append( {
            'batch' : (ds_gen, 
                       model_spec, 
                       objective, 
                       cv_iters, 
                       cv_samples, 
                       cv_split, 
                       seed),
            'result' : res } )

        iteration += 1

    return results

##=============================================================================

##
# Runs a set of batched experiments.
def run_experiments( batch_specs ):
    return map(_run_batch, batch_specs )

##=============================================================================
##=============================================================================

def test():

    specs = [

        # first batch
        BatchSpecification(

            # datasets to test
            [ dataset.RandomPolynomial(),
              dataset.RandomPolynomial(order=1) ],

            # models to test
            [ models.spec( models.NearestNeighbor ),
              models.spec( models.Constant, 1.0 ) ],
            
            # objectives to test
            [ validation.SumOfDistanceObjective(),
              validation.SumOfDistanceObjective( 
                  distance_function = lambda y, y_hat: y - y_hat ) ],

            # cross-validation parameters to test
            [ 2, 10, 30 ],
            [ 10, 20, 100 ],
            [ 0.1, 0.5, 0.7 ],

            # seeds
            [ None ],

            identifier = "TestBatch_0 (Discriminants)" ),

        # Second Batch
        BatchSpecification(

            # datasets to test
            [ dataset.RandomPolynomial(),
              dataset.RandomPolynomial(order=1) ],

            # models to test
            [ models.spec( models.BootstrapResampler ) ],
            
            # objectives to test
            [ validation.ContinuousKolmorogovSmirnovObjective() ],

            # cross-validation parameters to test
            [ 2, 5 ],
            [ 10, 20, 100 ],
            [ 0.1, 0.5, 0.7 ],

            # seeds
            [ None ],

            identifier = "TestBatch_1 (Generatives)" ),
    ]

    # Ok, run the specs
    results = run_experiments( specs )
    
    # Print means
    for batch_res in results:
        for res in batch_res:
            print "{0} : {1}".format(
                str(res['batch']),
                np.mean( res['result'] ) )

            

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
