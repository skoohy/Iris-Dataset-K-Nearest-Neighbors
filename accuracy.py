from knn import knn as knn

def accuracy(train, test, ki):
    """Returns the accuracy for a k value in the knn algorithm
    
       train: Training data to be used (Built on dictionaries)
       test: Test data to be evaluated
       ki: Number of desired neighbors"""
    accuracy = 0
    total = 0
    for group in test:
        for data in test[group]:
            results, confidence = knn(train, data, ki)
            if group == results:
                accuracy += 1 
            total += 1
    accuracy = accuracy / total
    return accuracy
