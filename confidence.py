from knn import knn as knn

# Confidence: The count of the most common class / k
# If k = 5 then we hope the most common is 5, however this isnt always the case 
# We calculte this when the algoirthms guess was wrong

def confidence1(train, test, ki):
    """Returns a list for the confidence values, while also storing k
       for each confidence value calculated 
    
       train: Training data to be used (Built on dictionaries)
       test: Test data to be evaluated
       ki: Number of desired neighbors"""
    confidence_lst = []
    for group in test:
        for data in test[group]:
            results, confidence = knn(train, data, k=ki)
            if group != results:
                confidence_lst.append([confidence, ki])
    return confidence_lst

def confidence2(train, test, ki):
    """Returns a list for the confidence values, does not store k
    
       train: Training data to be used (Built on dictionaries)
       test: Test data to be evaluated
       ki: Number of desired neighbors"""
    confidence_lst = []
    for group in test:
        for data in test[group]:
            results, confidence = knn(train, data, k=ki)
            if group != results:
                confidence_lst.append(confidence)
    return confidence_lst