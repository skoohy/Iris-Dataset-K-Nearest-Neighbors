import numpy as np
from collections import Counter

def knn(train, test, k):
    """k-Nearest-Negihbor algorithm
    
       train: Training data to be used (Built on dictionaries)
       test: Test data to be evaluated
       k: Number of desired neighbors
       
       Returns the prediction/result for 'test' and the confidence
       """ 
    distances = []
    for group in train:
        for features in train[group]:
            eucliean_distance = np.linalg.norm(np.array(features)-np.array(test))
            distances.append([eucliean_distance, group])     
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_results = Counter(votes).most_common(1)[0][0]
    
    confidence = Counter(votes).most_common(1)[0][1] / k
    
    return vote_results, confidence