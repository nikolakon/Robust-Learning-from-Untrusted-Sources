import tensorflow as tf
import numpy as np
import sys

import functions as f

if __name__ == "__main__":
    
    # Run an example experiment with the animals data. Here, run all experiments for a single attribute.
    
    # Define default parameters:
    # m - number of samples per source
    # T - number of sources
    # K - parameter for the K-fold CV
    # CV_with_disc - if True, discrepancies are recomputed every time a part of the data
    #                is taken out for cross-validation    
    
    m = 500
    T = 60
    K = 5
    CV_with_disc = True
    
    # Receive the target attribute as an argument
    try:
        target_attr = int(sys.argv[1])
    except:
        target_attr = 0
        
    # Through the ind argument, one can select n, p (proportion of corrupted samples per source) and the type of corruption
    try:
        ind = int(sys.argv[2])
        if ind < 12600:
            
            ind_attack = int(ind/2100)
            attack_type = ['enforce_label', 'inputs_same', 'shuffle_labels', 'blured', 'pixels', 'RGB'][ind_attack]
            ind = ind%2100
            ind_p = int(ind/700)
            p = [0.2, 0.5, 1.][ind_p]
            
            ind = ind%700
            n_ind = int(ind/100)
            n = [10,20,30,40,50,55,59][n_ind]
            
            # Do a 100 runs for each setup
            i = ind%100
        else:
            # Can also do runs with no corruption
            ind = ind - 12600
            attack_type = 'none'
            n = 0
            p = 0
            
            i = ind%100         
    
    except:
        ind = 0
        p = 1.
        n = 20
        i = 0
        target_attr = 0
        attack_type = 'enforce_label'
        
    # Pass the relevant value of sigma (noise for blurring or % dead pixels)
    if attack_type == 'blured':
        sigma = 6
    elif attack_type == 'pixels':
        sigma = 30
    else:
        sigma = 0
        
    np.random.seed(i)
    print(attack_type, target_attr, sigma, n, p, i)

    pathname = 'paths/to/extracted/features/'
    
    #############################################################################################
    # Run test
    errs, errs_zero, errs_inf, errs_robust_log_regr_soft, errs_robust_probs,\
        errs_geometric_median, errs_component_median, errs_batch_norm, all_alphas, selected_lambdas, poisoned_tasks =\
        f.test_on_animals(m, T, pathname,K, CV_with_disc, n, p, attack_type,\
                    target_attr, sigma)
        

    
    #############################################################################################
    
    # Save values; if saved this way, the provided Jupyter notebook can then be used to
    # visualize the results
    
    folder = 'path/to/where/you/want/results_animals/'
        
    name = folder + 'errs' \
        + '_' + str(attack_type) + '_'  + str(sigma) + '_' + str(n) + '_' + str(p) + '_' + str(i) + '_' + str(target_attr) + '.npy'
    np.save(name, errs)
    name = folder + 'errs_zero' \
        + '_' + str(attack_type) + '_'  + str(sigma) + '_' + str(n) + '_' + str(p) + '_' + str(i) + '_' + str(target_attr) + '.npy'
    np.save(name, errs_zero)
    name = folder + 'errs_inf' \
        + '_' + str(attack_type) + '_'  + str(sigma) + '_' + str(n) + '_' + str(p) + '_' + str(i) + '_' + str(target_attr) + '.npy'
    np.save(name, errs_inf)

    name = folder + 'errs_robust_log_regr_soft' \
        + '_' + str(attack_type) + '_'  + str(sigma) + '_' + str(n) + '_' + str(p) + '_' + str(i) + '_' + str(target_attr) + '.npy'
    np.save(name, errs_robust_log_regr_soft)       
    name = folder + 'errs_robust_probs' \
        + '_' + str(attack_type) + '_'  + str(sigma) + '_' + str(n) + '_' + str(p) + '_' + str(i) + '_' + str(target_attr) + '.npy'
    np.save(name, errs_robust_probs)  
    name = folder + 'errs_geometric_median' \
        + '_' + str(attack_type) + '_'  + str(sigma) + '_' + str(n) + '_' + str(p) + '_' + str(i) + '_' + str(target_attr) + '.npy'
    np.save(name, errs_geometric_median)  
    name = folder + 'errs_component_median' \
        + '_' + str(attack_type) + '_'  + str(sigma) + '_' + str(n) + '_' + str(p) + '_' + str(i) + '_' + str(target_attr) + '.npy'
    np.save(name, errs_component_median)  
    name = folder + 'errs_batch_norm' \
        + '_' + str(attack_type) + '_'  + str(sigma) + '_' + str(n) + '_' + str(p) + '_' + str(i) + '_' + str(target_attr) + '.npy'
    np.save(name, errs_batch_norm)  
    
    name = folder + 'alphas' \
        + '_' + str(attack_type) + '_'  + str(sigma) + '_' + str(n) + '_' + str(p) + '_' + str(i) + '_' + str(target_attr) + '.npy'
    np.save(name, all_alphas)
    name = folder + 'lambdas' \
        + '_' + str(attack_type) + '_'  + str(sigma) + '_' + str(n) + '_' + str(p) + '_' + str(i) + '_' + str(target_attr) + '.npy'
    np.save(name, selected_lambdas)
    name = folder + 'poisoned_tasks' \
        + '_' + str(attack_type) + '_'  + str(sigma) + '_' + str(n) + '_' + str(p) + '_' + str(i) + '_' + str(target_attr) + '.npy'
    np.save(name, poisoned_tasks)
