'''
This file contains the functions we need.
'''

import tensorflow as tf
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.model_selection import PredefinedSplit
from scipy.stats import binom


# Loading the data

'''
get_data_clean imports and preprocesses the CLEAN data for the experiments with Animals with Attributes,
creating train and test datasets for each product.
'''

def get_data_clean(pathname, m, T, target_feature):

    # Load the data; Need to have all clean features and all labels saved in such files
    features = np.load(pathname + '/files/all_features.npy')
    labels = np.load(pathname + '/files/all_labels.npy')    
    labels = labels[:, target_feature]
    
    # Store the training data in lists of T elements, each for one of the T sources
    # At each source we have m data points
    Inputs_train = []
    Outputs_train = []
    
    # Shuffle the data
    shuffle = np.random.permutation(features.shape[0])
    features = features[shuffle]
    labels = labels[shuffle]
    
    for i in range(T):
        Inputs_train.append(features[(i*m):((i+1)*m), :])
        Outputs_train.append(labels[(i*m):((i+1)*m)])
    
    Inputs_test = features[(m*T):, :]
    Outputs_test = labels[(m*T):]
    
    return((Inputs_train, Outputs_train, Inputs_test, Outputs_test))

  
'''
get_data_poisoned imports and preprocesses the data for the experiments with Animals with Attributes,
creating train and test datasets for each product.
It also corrupts a fraction of p samples from n random sources, by inserting the features
of the corrupted images, instead of the clean ones.
This is how corruptions are created for the synthetic attacks.
'''

def get_data_poisoned(pathname, m, T, target_feature, n, p, attack_type, sigma = 0):

    # Load the data:  
    features = np.load(pathname + '/files/all_features.npy')
    labels = np.load(pathname + '/files/all_labels.npy')    
    labels = labels[:, target_feature]
    
    # Store the training data in lists of T elements, each for one of the T sources
    # At each source we have m data points
    Inputs_train = []
    Outputs_train = []
    
    # Shuffle the data
    shuffle = np.random.permutation(features.shape[0])
    features = features[shuffle]
    labels = labels[shuffle]
    
    for i in range(T):
        Inputs_train.append(features[(i*m):((i+1)*m), :])
        Outputs_train.append(labels[(i*m):((i+1)*m)])
    
    Inputs_test = features[(m*T):, :]
    Outputs_test = labels[(m*T):]
    
    # Load poisoned features (again, need to have those files pre-created)
    poisoned_features = np.load(pathname + '/files/all_features_' + attack_type + '_' + str(sigma) + '.npy')
    
    poisoned_tasks = np.random.choice(T, n, replace = False)
    # Insert the poisoned features into the data
    for ind in poisoned_tasks:
        n_poisoned = int(m*p)
        poisoned_indexes = np.random.choice(m, n_poisoned, replace = False)
        poisoned_indexes_global = (ind)*m + poisoned_indexes
        Inputs_train[ind][poisoned_indexes, :] = poisoned_features[poisoned_indexes_global, :]
    
    return((Inputs_train, Outputs_train, Inputs_test, Outputs_test, poisoned_tasks))
    
#########################################################################################
# Implementations of different corruptions
        
'''
poison_by_flip poisons a proportion p of the labels of n tasks, by inverting the labels
of all poisoned points
'''

def poison_by_flip(Outputs_train, n, p):
    T = len(Outputs_train)
    poisoned_tasks = np.random.choice(T, n, replace = False)
    
    for ind in poisoned_tasks:
        n_points = Outputs_train[ind].shape[0]
        n_poisoned = int(n_points*p)
        poisoned_points = np.random.choice(n_points, n_poisoned, replace = False)
        Outputs_train[ind][poisoned_points] = 1 - Outputs_train[ind][poisoned_points]
        
    return(Outputs_train, poisoned_tasks)
    
    
'''
poison_by_enforcing_label poisons a proportion p of the labels of n tasks, by setting the 
labels of all poisoned points to 1.
'''


def poison_by_enforcing_label(Outputs_train, n, p):
    T = len(Outputs_train)
    poisoned_tasks = np.random.choice(T, n, replace = False)
    
    for ind in poisoned_tasks:
        n_points = Outputs_train[ind].shape[0]
        n_poisoned = int(n_points*p)
        poisoned_points = np.random.choice(n_points, n_poisoned, replace = False)
        Outputs_train[ind][poisoned_points] = 1
        
    return(Outputs_train, poisoned_tasks)
    
 
    
'''
poison_by_shuffling_labels selects a fraction p of the samples from n sources and
shuffles their labels randomly (within each source)
'''   
    
def poison_by_shuffling_labels(Outputs_train, n, p):
    T = len(Outputs_train)
    poisoned_tasks = np.random.choice(T, n, replace = False)
    
    for ind in poisoned_tasks:
        n_points = Outputs_train[ind].shape[0]
        n_poisoned = int(n_points*p)
        poisoned_points = np.random.choice(n_points, n_poisoned, replace = False)
        reordered = np.random.permutation(poisoned_points)
        Outputs_train[ind][poisoned_points] = Outputs_train[ind][reordered]
        
    return(Outputs_train, poisoned_tasks)


'''
poison_by_permutation_same applieds the same random permutation to a proportion p of the inputs of n tasks.
'''   
def poison_by_permutation_same(Inputs_train, n, p):
    T = len(Inputs_train)
    d = Inputs_train[0].shape[1]
    poisoned_tasks = np.random.choice(T, n, replace = False)
    
    perm = np.random.permutation(d-1)
    
    for ind in poisoned_tasks:
        n_points = Inputs_train[ind].shape[0]
        n_poisoned = int(n_points*p)
        poisoned_points = np.random.choice(n_points, n_poisoned, replace = False)
        for point in poisoned_points:
            Inputs_train[ind][point, :-1] = Inputs_train[ind][point, perm]
        
    return(Inputs_train, poisoned_tasks)    
    
##########################################################################################
# Functions for estimating the discrepancies and optimizing the source weights


'''
get_disc_sq estimates the empirical discrepancies for a task by finding a linear classifier that performs well on
it and badly on another. We use the linear classifier that minimises a SQUARE LOSS.
'''

def get_disc_sq(Inputs, Outputs, task_ind):
    # We store the estimates of the discrepancies in the Disc numpy array
    Disc = np.zeros((len(Inputs)))
    
    # We loop over all the other tasks
    X1 = Inputs[task_ind]
    for j in range(len(Inputs)):
        # We set up Inputs X and outputs Y as a classification problem
        X2 = Inputs[j]
        m1 = X1.shape[0]
        m2 = X2.shape[0]
        X = np.concatenate((X1, X2))
        Y = np.concatenate((Outputs[task_ind], -Outputs[j]+1))
        
        # We find the optimal parameter values and the fitted values using the usual
        # weighted linear regression formulas
        
        weights = np.concatenate((np.repeat(1/m1, m1), np.repeat(1/m2, m2)))
        bla = np.multiply(np.transpose(X), weights)
        w = np.linalg.solve(bla@X + 0.000001*np.eye(X.shape[1]),bla@Y)
        fitted = X@w
        
        # We estimate the discrepancy by evaluating the emperical discrepancy at this optimal
        # (for square loss) linear classifier
        loss = np.sum(((Y==1)*(fitted<0.5) + (Y==0)*(fitted>=0.5))*weights) - 1
        
        Disc[j] = np.abs(loss)

    return(Disc)
    
    
    
'''
optimize_weights finds the best values for the alphas, that minimize our bound. This is done for a particular
task index and discrepancies. lamb controls the strength of regularization.
'''

def optimize_weights(disc_for_task, ms, lamb, nepochs = 1001, learn_rate = 0.01):
    
    tf.reset_default_graph()
    
    # Initiate the weights randomly
    T = disc_for_task.shape[0]
    with tf.name_scope('Parameters'):
        logits = tf.Variable(tf.random_normal([T], stddev = 1/np.sqrt(T), dtype = tf.float32))
        alphas = tf.nn.softmax(logits) # Takes soft-max for every row
    
    # Find an expression for the optimisation target
    with tf.name_scope('loss'):
        loss = lamb*tf.sqrt(tf.reduce_sum(tf.square(alphas)*(1/ms))) + 2*tf.reduce_sum(alphas*(disc_for_task))
    
    # Run gradient descent to optimize the weights:
    optimizer = tf.train.AdamOptimizer(learn_rate)
    train = optimizer.minimize(loss)
    
    with tf.Session() as sess:
      
        sess.run(tf.global_variables_initializer()) # reset values to initial random values   
     
        # Train the linear model
        for i in range(nepochs):
            sess.run(train)
            #if i%100 == 0:
                #print(sess.run(loss))
        alph = sess.run(alphas)
        return alph
    
    
    
###############################################################################################
# Training (weighted) linear models


'''
get_model_with_cv finds the best linear predictor for a task based on the log loss, with cross-validation:
'''
def get_lr_model_with_cv(X_train_all, Y_train_all):
    lr = LogisticRegressionCV(fit_intercept = False)
    lr.fit(X_train_all, Y_train_all)
    best_w = lr.coef_[0]
    return best_w

'''
get_lr_model finds the best linear predictor for a task by minimizing an alpha-weighted log loss
(no regularization here)
'''

def get_lr_model(alphas, X_train_all, Y_train_all, ms):
    all_alphas = np.repeat(alphas/ms, ms)
    
    lr = LogisticRegression(C = 100, fit_intercept = False)
    lr.fit(X_train_all, Y_train_all, sample_weight=all_alphas)
    
    best_w = lr.coef_[0]
    return best_w

'''
get_lr_model_with_cv_on_clean minimizes an alpha-weighted log-loss and chooses a regularization
constant by cross-validation on the clean data only.
'''

def get_lr_model_with_cv_on_clean(alphas, X_train_all, Y_train_all, ms, clean_task):
    # Spread the weights on sample level and define the CV split
    all_alphas = np.repeat(alphas/ms, ms)
    indexes_cv = (-1)*np.ones(X_train_all.shape[0])
    clean_begins = np.sum(ms[:clean_task])
    curr_m = ms[clean_task]
    all_alphas[clean_begins:(clean_begins + curr_m)] = all_alphas[clean_begins:(clean_begins + curr_m)]*(5/4)
    for l in range(5):
        indexes_cv[(clean_begins + l*(int(curr_m/5))):(clean_begins + (l+1)*int((curr_m/5)))] = l
        
    ps = PredefinedSplit(indexes_cv)
    
    # Train on all data, with 5-fold CV on the clean data
    lr = LogisticRegressionCV(fit_intercept = False, cv = ps)
    lr.fit(X_train_all, Y_train_all, sample_weight=all_alphas)
    best_w = lr.coef_[0]
    return best_w


########################################################################################
# Baselines

'''
robust_log_regr minimizes the robust squared loss of Pregibon et al. (1982), for a fixed
regularization constant
'''
    
def robust_log_regr(X_train_all, Y_train_all, c, l, learn_rate = 0.005, nepochs = 100, soft = True):
    
    tf.reset_default_graph()
    
    # Initiate the weights randomly
    d = X_train_all.shape[1]
    with tf.name_scope('Parameters'):
        w = tf.Variable(tf.random_normal([d,1], stddev = 1/np.sqrt(d), dtype = tf.float32))
        Y = tf.placeholder(dtype = tf.float32)
        X = tf.placeholder(dtype = tf.float32)
        
    # Find an expression for the optimisation target
    with tf.name_scope('loss'):
        neg_log_likelihood = tf.log(1+tf.exp(-(2*Y-1)*tf.reshape(tf.matmul(X,w), [tf.shape(X)[0]])))
        
        if soft:
            loss = tf.reduce_mean(neg_log_likelihood*(tf.cast(tf.less(neg_log_likelihood, c), tf.float32)) + \
                    (2*tf.sqrt(c*neg_log_likelihood) - c)*(1-tf.cast(tf.less(neg_log_likelihood, c), tf.float32))) \
                                  + l*(tf.norm(w)**2)
        
        else:
            loss = tf.reduce_mean(neg_log_likelihood*(tf.cast(tf.less(neg_log_likelihood, c), tf.float32)) + \
                    c*(1-tf.cast(tf.less(neg_log_likelihood, c), tf.float32))) + l*(tf.norm(w)**2)
            
    # Run gradient descent to optimize the weights:
    optimizer = tf.train.GradientDescentOptimizer(learn_rate)
    train = optimizer.minimize(loss)
    
    with tf.Session() as sess:
      
        sess.run(tf.global_variables_initializer()) # reset values to initial random values   
        w_now = sess.run(w)
        
        for i in range(nepochs):
     
            sess.run(train, {X: X_train_all, Y: Y_train_all})
            if i%10 == 0:
                loss_now = sess.run(loss, {X: X_train_all, Y: Y_train_all})
                if loss_now == loss_now:
                    w_now = sess.run(w)
                    #print(loss_now)
                else:
                    #print(loss_now)
                    best_w = [i[0] for i in w_now]
                    return best_w
        
        best_w = [i[0] for i in w_now]
        return best_w


'''
robust_log_regr_with_CV does cross-validation on the clean data only, for the robust log loss
'''
def robust_log_regr_with_CV(X_train_all, Y_train_all, task_ind, m, T, soft):
    poss_l = 2**(np.arange(-5, 5, dtype=float))
    val_errs = np.zeros((len(poss_l), 5))
    
    c = 1.345**2
    
    for l_ind, l in enumerate(poss_l):
        for k in range(5):
            # Create a training and validation set
            mask_val = np.all([np.arange(0, m*T) >= m*task_ind + k*(m/5), np.arange(0, m*T) < m*task_ind + (k+1)*(m/5)], axis = 0)
            X_val_now = X_train_all[mask_val, :]
            X_train_now = X_train_all[~mask_val, :]
            Y_val_now = Y_train_all[mask_val]
            Y_train_now = Y_train_all[~mask_val]
            
            curr_w = robust_log_regr(X_train_now, Y_train_now, c, l, soft = soft)
            predictions = ((1/(1+np.exp(-X_val_now@curr_w))) > 0.5).astype(int)
            val_errs[l_ind, k] = np.mean(predictions != Y_val_now)
    
    l = poss_l[np.argmin(np.sum(val_errs, axis = 1))]
    best_w = robust_log_regr(X_train_all, Y_train_all, c, l, learn_rate = 0.002, nepochs = 500, soft = soft)
    print(l)
    return best_w


'''
robust_mixtures implements the three robust ensemble baselines
'''

def robust_mixtures(Inputs_train, Outputs_train, Inputs_test, Outputs_test, clean_task, T):

    d = Inputs_train[0].shape[1]
    
    probs = np.zeros((T, Outputs_test.shape[0]))
    models = np.zeros((T, Inputs_test.shape[1]))
    
    # Train local models
    
    lr = LogisticRegression(C = 100, fit_intercept = False)
    for i in range(T):
        if len(np.unique(Outputs_train[i])) > 1:
            lr.fit(Inputs_train[i], Outputs_train[i])
            best_w = lr.coef_[0]
            probs[i,:] = (1/(1+np.exp(-Inputs_test@best_w)))
            models[i,:] = best_w
        else:
            best_w = np.zeros(d)
            if np.unique(Outputs_train[i])[0] == 0:
                best_w[-1] = -1
            else:
                best_w[-1] = 1
            
            probs[i,:] = (1/(1+np.exp(-Inputs_test@best_w)))
            models[i,:] = best_w
    
    # Compute median of predicted probabilities
    # Estimate the test performance of the predictor here for simplicity      
    robust_probs = np.median(probs, axis = 0)
    robust_preds = (robust_probs>0.5).astype(int)
    success_robust_preds = np.mean(robust_preds!= Outputs_test)
    
    # Component-wise median of the local models (inspired by Yin et al.)
    component_median_model = np.median(models, axis = 0)
    
    # Find the geometric median by gradient descent
    # (minimize the sum of sqrt of L2-distances to the learned weight vectors)
    learn_rate = 0.05
    nepochs = 1000
    tf.reset_default_graph()
    
    # Initiate the weights randomly
    d = (Inputs_train[0]).shape[1]
    with tf.name_scope('Parameters'):
        w = tf.Variable(tf.random_normal([1,d], stddev = 1/np.sqrt(d), dtype = tf.float32))
        vectors = tf.constant(models, dtype = tf.float32)
        
    # Find an expression for the optimisation target
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.norm(vectors - w, axis = 1))
    
    # Run gradient descent to optimize the weights:
    optimizer = tf.train.GradientDescentOptimizer(learn_rate)
    train = optimizer.minimize(loss)
    
    with tf.Session() as sess:
      
        sess.run(tf.global_variables_initializer()) # reset values to initial random values   
        w_now = sess.run(w)
        
        for i in range(nepochs):
     
            sess.run(train)
            if i%10 == 0:
                loss_now = sess.run(loss)
                if loss_now == loss_now:
                    w_now = sess.run(w)
                    #print(loss_now)
                else:
                    #print(loss_now)
                    break
        
        geometric_median_model = w_now[0]
    
    return(success_robust_preds, component_median_model, geometric_median_model)
        

def batch_norm(Inputs_train, Outputs_train, task_ind, m, T):
    
    T = len(Inputs_train)
    d = Inputs_train[0].shape[1]
    
    # Compute the mean and variance of the clean dataset
    
    mu = np.mean(Inputs_train[task_ind][:,:-1], axis = 0)
    sigma = np.std(Inputs_train[task_ind][:,:-1], axis = 0)
    
    # Standardise all data
    for i in range(T):
        Inputs_train[i][:,:-1] = (Inputs_train[i][:,:-1] - np.mean(Inputs_train[i][:,:-1], axis = 0))/(np.std(Inputs_train[i][:,:-1], axis = 0))
    
    
    # Store all data in big matrices
    X_train_all = np.zeros((1,d))
    Y_train_all = np.zeros(1)
    for i in range(T):
        X_train_all = np.concatenate((X_train_all, Inputs_train[i]), axis = 0)
        Y_train_all = np.concatenate((Y_train_all, Outputs_train[i]))
    X_train_all = X_train_all[1:,:]
    Y_train_all = Y_train_all[1:]
    
    ms = np.repeat(m, T)
    alphas = np.ones(T)*(1/T)
    
    # Train a logistic regression model on this data, with cross-validation on the reference dataset
    best_w = get_lr_model_with_cv_on_clean(alphas, X_train_all, Y_train_all, ms, task_ind)
    return (best_w, mu, sigma)
            
#########################################################################################
    
    
'''
test_on_animals implements the Animals with Attributes 2 experiments, described in the paper. 
It selects random subsets of the data for sources and testing then evaluates our method and the
baselines, with a random clean source as reference data. 

A random proportion p of the samples in n sources are corrupted, with attack_type specifying 
the type of corruption.
'''

def test_on_animals(m, T, pathname, K, CV_with_disc, n, p, attack_type, target_feature,\
                           sigma):
    
    if (attack_type != 'blured') and (attack_type != 'RGB') and (attack_type != 'pixels'):
        # Load data
        Inputs_train, Outputs_train, Inputs_test, Outputs_test = get_data_clean(pathname, m, T, target_feature)
            
        # Poison the data
        if attack_type == 'none':
            poisoned_tasks = np.array([])
        if attack_type == 'flip_labels':
            Outputs_train, poisoned_tasks = poison_by_flip(Outputs_train, n, p)
        if attack_type == 'inputs_same':
            Inputs_train, poisoned_tasks = poison_by_permutation_same(Inputs_train, n, p)
        if attack_type == 'enforce_label':
            Outputs_train, poisoned_tasks = poison_by_enforcing_label(Outputs_train, n, p)
        if attack_type == 'shuffle_labels':
            Outputs_train, poisoned_tasks = poison_by_shuffling_labels(Outputs_train, n, p)
            
    else:
        # If corruptions are on the image level, load the corrupted features directly
        Inputs_train, Outputs_train, Inputs_test, Outputs_test, poisoned_tasks = get_data_poisoned(pathname\
                                                                , m, T, target_feature, n, p, attack_type, sigma)
        
        
    d = Inputs_train[0].shape[1]
    
    # Store all data in big matrices
    X_train_all = np.zeros((1,d))
    Y_train_all = np.zeros(1)
    for i in range(T):
        X_train_all = np.concatenate((X_train_all, Inputs_train[i]), axis = 0)
        Y_train_all = np.concatenate((Y_train_all, Outputs_train[i]))
    X_train_all = X_train_all[1:,:]
    Y_train_all = Y_train_all[1:]

    # Possible values for lambda
    lambdas = np.concatenate((np.array([0]), 2**(np.arange(-10,10, dtype=float)), np.array([10000])))
    
    # We only test on clean tasks
    clean_tasks = np.array([i for i in np.arange(T) if not i in poisoned_tasks])
    task_ind = np.random.choice(clean_tasks, 1, replace = False)[0]
    

    # Select an optimal value of lambda by using K-fold validation. Calculate the
    # test error for this value of lambda and for all baselines
    # We also report the selected values of lambda and their corresponding alphas
    disc_for_task = get_disc_sq(Inputs_train, Outputs_train, task_ind)
    val_errs = np.zeros((K, len(lambdas)))
    for k in range(K):
        # Create a training and validation set
        mask_val = np.all([np.arange(0, m*T) >= m*task_ind + k*(m/K), np.arange(0, m*T) < m*task_ind + (k+1)*(m/K)], axis = 0)
        mask_train = np.all([np.arange(0, m*T) >= m*task_ind, np.arange(0, m*T) < m*(task_ind+1), ~mask_val], axis = 0)
        X_val_now = X_train_all[mask_val, :]
        X_train_now = X_train_all[~mask_val, :]
        Y_val_now = Y_train_all[mask_val]
        Y_train_now = Y_train_all[~mask_val]
        
        ms = np.repeat(m, T)
        ms[task_ind] = int(m - m/K)
        
        # If CV_with_disc == True, discrepancies are recalculated without the help of the validation set
        if CV_with_disc == True:
            
            Inputs_train2 = Inputs_train.copy()
            Inputs_train2[task_ind] = X_train_all[mask_train,:]
            Outputs_train2 = Outputs_train.copy()
            Outputs_train2[task_ind] = Y_train_all[mask_train]
            disc_for_task = get_disc_sq(Inputs_train2, Outputs_train2, task_ind)
        
        # For each value of lambda, calculate the validation error
        for i, lamb in enumerate(lambdas):
            
            # For a given lambda, find optimal alphas
            alphas = optimize_weights(disc_for_task, ms, lamb, nepochs = 1001, learn_rate = 0.01)
            
            # Compute the best predictor based on the training data only
            best_w = get_lr_model(alphas, X_train_now, Y_train_now, ms)
            
            # Evaluate the predictor on the validation set
            predictions = ((1/(1+np.exp(-X_val_now@best_w))) > 0.5).astype(int)
            val_errs[k, i] = np.mean(predictions != Y_val_now)
    
    # Select the optimal value of lambda
    lamb = lambdas[np.argmin(np.sum(val_errs, axis = 0))]
    selected_lambdas = lamb
    ms = np.repeat(m, T)
    
    # Obtain the best linear predictor based on the full training set, wiht the chosen value of lambda
    disc_for_task = get_disc_sq(Inputs_train, Outputs_train, task_ind)
    alphas = optimize_weights(disc_for_task, ms, lamb, nepochs = 10001, learn_rate = 0.01)
    best_w  = get_lr_model_with_cv_on_clean(alphas, X_train_all, Y_train_all, ms, task_ind)
    
    # Evaluate the predictor on the test data from the task
    predictions = ((1/(1+np.exp(-Inputs_test@best_w))) > 0.5).astype(int)
    print("Our Algo:", lamb, np.mean(predictions != Outputs_test))
    errs = np.mean(predictions != Outputs_test)
    all_alphas = alphas
    
    # Compare with learning on clean data only
    if len(np.unique(Outputs_train[task_ind]))>1:
        best_w = get_lr_model_with_cv(Inputs_train[task_ind], Outputs_train[task_ind])   
    else:
        best_w = np.zeros(d)
        if np.unique(Outputs_train[task_ind])[0] == 0:
            best_w[-1] = -1
        else:
            best_w[-1] = 1
    # Evaluate the predictor on the test data from the task
    predictions = ((1/(1+np.exp(-Inputs_test@best_w))) > 0.5).astype(int)
    print("Single source:", np.mean(predictions != Outputs_test))
    errs_zero = np.mean(predictions != Outputs_test)
    
    # Compare to simply learning on all data, validating only on the clean data
    
    alphas = np.ones(T)*(1/T)
    best_w = get_lr_model_with_cv_on_clean(alphas, X_train_all, Y_train_all, ms, task_ind)
    # Evaluate the predictor on the test data from the task
    predictions = ((1/(1+np.exp(-Inputs_test@best_w))) > 0.5).astype(int)
    print("All data:", np.mean(predictions != Outputs_test))
    errs_inf = np.mean(predictions != Outputs_test)

    # Compare to robust baselines
    
    # Compare to robust logistic regression:
    best_w = robust_log_regr_with_CV(X_train_all, Y_train_all, task_ind, m, T, soft = True)
    predictions = (Inputs_test@best_w > 0).astype(int)
    print("Robust loss:", np.mean(predictions != Outputs_test))
    errs_robust_log_regr_soft = np.mean(predictions != Outputs_test)
    
    # Compare to robust ensembles
    robust_probs, component_median_model, geometric_median_model = robust_mixtures(Inputs_train, Outputs_train, Inputs_test, Outputs_test, task_ind, T)
    print("Median of probs:", robust_probs)
    errs_robust_probs = robust_probs
    
    predictions = ((1/(1+np.exp(-Inputs_test@geometric_median_model))) > 0.5).astype(int)
    print("Geometric Median:", np.mean(predictions != Outputs_test))
    errs_geometric_median = np.mean(predictions != Outputs_test)
    
    predictions = ((1/(1+np.exp(-Inputs_test@component_median_model))) > 0.5).astype(int)
    print("Componentwise Median:", np.mean(predictions != Outputs_test))
    errs_component_median = np.mean(predictions != Outputs_test)
    
    # Compare to batch norm
    best_w, mu, sigma = batch_norm(Inputs_train, Outputs_train, task_ind, m, T)
    Inputs_test_normalized = np.copy(Inputs_test)
    Inputs_test_normalized[:,:-1] = (Inputs_test[:,:-1]-mu)/sigma
    predictions = ((1/(1+np.exp(-Inputs_test_normalized@best_w))) > 0.5).astype(int)
    print("Batch norm:", np.mean(predictions != Outputs_test))
    errs_batch_norm = np.mean(predictions != Outputs_test)
        
    return (errs, errs_zero, errs_inf, errs_robust_log_regr_soft, errs_robust_probs,\
            errs_geometric_median, errs_component_median, errs_batch_norm, all_alphas, selected_lambdas, poisoned_tasks)