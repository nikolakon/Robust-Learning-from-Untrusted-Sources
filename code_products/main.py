import tensorflow as tf
import numpy as np
import sys

import functions as f

if __name__ == "__main__":
    
    # Run an experiment with the products data
    
    # Define default parameters:
    # m - number of samples per task,
    # min_pos_reviews - minimum number of reviews of each type for a product to be selected  
    # K - parameter for the K-fold CV
    # CV_with_disc - if True, discrepancies are recomputed every time a part of the data
    #                is taken out for cross-validation
    
    m = 100
    min_pos_reviews = 300
    K = 5
    CV_with_disc = True
    
    
    # If specific is True, the test with the books and non-books is run (it uses specific products)
    # If not, then the experment on all products is performed
    try:
        specific = bool(int(sys.argv[1]))
    except:
        specific = True
        
        
    # If specific == False, the index of the target task can be taken as input (ind)
    # If specific == True, one can pass the desired values of non-books n
    #                       and the index of the experiment through ind
    try:
        ind = int(sys.argv[2])
        if specific == False:
 
            n = 0
            p = 0
            i = ind
            sigma = 0
            attack_type = 'none'       
        
        else:
            n = int(ind/1000)    
            i = ind%1000
            
            p = 0
            attack_type = 'none'
            sigma = 0
            
            
    except:
        n = 0
        i = 0
        p = 0
        attack_type = 'none'
        sigma = 0
        
    # Set a different random seed for each experiment
    np.random.seed(ind)
    print(specific, attack_type, sigma, n, p, i)
    
    pathname = 'paths/to/product/dataset/'
    
    #############################################################################################
    
    # Run an experiment 
    
    errs, errs_zero, errs_inf, errs_robust_log_regr_soft, errs_robust_probs,\
            errs_geometric_median, errs_component_median, errs_batch_norm, all_alphas,\
            selected_lambdas, poisoned_tasks \
            = f.test_on_products(m, min_pos_reviews, pathname,\
                K, CV_with_disc, n, p, attack_type, specific, i)
      
    #############################################################################################
      
    # Save values; if saved this way, the provided Jupyter notebook can then be used to
    # visualize the results
    if specific:
        folder = 'path/to/where/you/want/results_products_specific/'
    else:
        folder = 'path/to/where/you/want/results_products_all/'
            
        
    name = folder + 'errs' \
        + '_' + str(attack_type) + '_'  + str(sigma) + '_' + str(n) + '_' + str(p) + '_' + str(i) + '.npy'
    np.save(name, errs)
    name = folder + 'errs_zero' \
        + '_' + str(attack_type) + '_'  + str(sigma) + '_' + str(n) + '_' + str(p) + '_' + str(i) + '.npy'
    np.save(name, errs_zero)
    name = folder + 'errs_inf' \
        + '_' + str(attack_type) + '_'  + str(sigma) + '_' + str(n) + '_' + str(p) + '_' + str(i) + '.npy'
    np.save(name, errs_inf)

    name = folder + 'errs_robust_log_regr_soft' \
        + '_' + str(attack_type) + '_'  + str(sigma) + '_' + str(n) + '_' + str(p) + '_' + str(i) + '.npy'
    np.save(name, errs_robust_log_regr_soft)       
    name = folder + 'errs_robust_probs' \
        + '_' + str(attack_type) + '_'  + str(sigma) + '_' + str(n) + '_' + str(p) + '_' + str(i) +'.npy'
    np.save(name, errs_robust_probs)  
    name = folder + 'errs_geometric_median' \
        + '_' + str(attack_type) + '_'  + str(sigma) + '_' + str(n) + '_' + str(p) + '_' + str(i) + '.npy'
    np.save(name, errs_geometric_median)  
    name = folder + 'errs_component_median' \
        + '_' + str(attack_type) + '_'  + str(sigma) + '_' + str(n) + '_' + str(p) + '_' + str(i) + '.npy'
    np.save(name, errs_component_median)  
    name = folder + 'errs_batch_norm' \
        + '_' + str(attack_type) + '_'  + str(sigma) + '_' + str(n) + '_' + str(p) + '_' + str(i) + '.npy'
    np.save(name, errs_batch_norm)  
 
    name = folder + 'alphas' \
        + '_' + str(attack_type) + '_'  + str(sigma) + '_' + str(n) + '_' + str(p) + '_' + str(i) + '.npy'
    np.save(name, all_alphas)
    name = folder + 'lambdas' \
        + '_' + str(attack_type) + '_'  + str(sigma) + '_' + str(n) + '_' + str(p) + '_' + str(i) + '.npy'
    np.save(name, selected_lambdas)
    name = folder + 'poisoned_tasks' \
        + '_' + str(attack_type) + '_'  + str(sigma) + '_' + str(n) + '_' + str(p) + '_' + str(i) + '.npy'
    np.save(name, poisoned_tasks)

