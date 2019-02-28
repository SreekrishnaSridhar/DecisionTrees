# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Programming Assignment 1 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np
import math


def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """
    s = {v:(x==v).nonzero()[0] for v in np.unique(x)}
    return s

    # INSERT YOUR CODE HERE
    raise Exception('Function not yet implemented!')


def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """
    value,count = np.unique(y,return_counts = True)
    p = count.astype('float')/len(y)
    hy = 0.0
    for k in np.nditer(p,op_flags = ['readwrite']):
    	hy+=(- k)*(math.log(k)/math.log(2))
    	
    return hy
    

    # INSERT YOUR CODE HERE
    raise Exception('Function not yet implemented!')


def mutual_information(x, y):
	"""
	Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
	over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
	the weighted-average entropy of EACH possible split.

	Returns the mutual information: I(x, y) = H(y) - H(y | x)
	"""
    
	valuex, countx = np.unique(x,return_counts = True)
	px = countx.astype('float')/len(x)
	hyx = 0.0
	for  pxval,xval in zip(px,valuex):
		hyx+=(pxval)*entropy(y[x==xval])
	hy  = entropy(y)
	ixy = hy -hyx
	return ixy
    # INSERT YOUR CODE HERE
	raise Exception('Function not yet implemented!')


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    root = {}
    if attribute_value_pairs is None:
     attribute_value_pairs= np.vstack([[(i,v) for v in np.unique(x[:,i])] for i in range(x.shape[1])])
      
    yvalues, ycounts = np.unique(y, return_counts = True)
    
    if(len(yvalues)==1): #1st terminating condition
     return yvalues[0]
     
    if(len(attribute_value_pairs)==0) or depth == max_depth: #2nd and 3rd terminating condition
     return yvalues[np.argmax(ycounts)]
     
     
     # Best v attribute and value
     # astype to get 0/1 values. Default is true or false.
     ListOfMutualInformation = np.array([mutual_information(np.array(x[:,i]==v).astype(int),y) for (i,v) in attribute_value_pairs])
     (bestattr,bestval) = attribute_value_pairs[np.argmax(ListofMutualInformation)]
     # Based on best attribute and value, splitting in true or false
     partitioning = partition(np.array(x[:,bestattr]==bestval).astype(int))
     # Removing the best attribute and value to split on
     dropIndex = np.all(attribute_value_pairs==(bestattr,bestvalue), axis = 1)
     attribute_value_pairs = np.delete(attribute_value_pairs,np.argwhere(dropIndex),0)
     # Removing those values that were split on  and recursively calling with the new data.
     for splitIndex, indices in partitioning.items():
      xsubset = x.take(indices,axis = 0)
      ysubset = y.take(indices,axis = 0)
      decision = bool(splitIndex)
      root[(bestattr,bestval,decision)] = id3(xsubset,ysubset,attribute_value_pairs = attribute_value_pairs,max_depth = max_depth, depth = depth+1)
      print(root)
      
     return root
    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
     raise Exception('Function not yet implemented!')


def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """
    for criteria, subtree in tree.items():
     attribute = criteria[0]
     value = criteria[1]
     decision = criteria[2]
     if decision == (x[attribute]== value):
      if type(subtree) is dict:
       label = predict_example(x,subtree)
      else:
       label = subtree
     return label

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    raise Exception('Function not yet implemented!')


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """
    size = len(y_true)
    err = [y_true[i]!=y_pred[i] for i in range(n)]
    return sum(err)/n
    # INSERT YOUR CODE HERE
    raise Exception('Function not yet implemented!')


def visualize(tree, depth=0):
    """
    Pretty prints (kinda ugly, but hey, it's better than nothing) the decision tree to the console. Use print(tree) to
    print the raw nested dictionary representation.
    DO NOT MODIFY THIS FUNCTION!
    """

    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

        # Print the children
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


if __name__ == '__main__':
    # Load the training data
    M = np.genfromtxt('/Users/sreekrishnasridhar/Desktop/data/monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('/Users/sreekrishnasridhar/Desktop/data//monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    # Learn a decision tree of depth 3
    decision_tree = id3(Xtrn, ytrn, max_depth=3)
    print(decision_tree)
    
    visualize(decision_tree)

    # Compute the test error
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)

    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
