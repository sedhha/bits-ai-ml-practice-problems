# importing dependencies
import numpy as np
import matplotlib.pyplot as plt

'''
    Get the random corelated data for testing of Multi Level Perceptron.
'''
def get_random_data(
    # The List shows the mean of the two variables in the form [mean(X), mean(Y)]
    mean: np.array = np.array([5.0,6.0]),
    
    cov: np.array = np.array(
        '''
        The List shows covariance between the variables in the form: 
            [
                [Cov(X,X), Cov(X,Y)],
                [Cov(Y,X), Cov(Y,Y)]
            ]
            
            Taking a set of closely related variables help in having better results.
        '''
        [
        [1,0.95],
        [0.95, 1.2]
    ]
        ),
    total_samples: int = 8000
    ):
    # Based on above constraints, generatitng X and Y data.
    data = np.random.multivariate_normal(mean, cov, total_samples)
    '''
        If we want we can uncomment the below lines to
        visualize the data that we have generated using 
        np.random.multivariate_normal function.
    '''
    # plt.scatter(data[:500, 0], data[:500, 1], marker='.')
    # plt.show()
    return data

'''
    Assumes Dataset has 2 features and hence spits out
    the result in form of X_Train, X_Test, Y_Train, Y_Test
'''
def split_test_and_train_data(
    data, 
    split_factor = 0.9 # Ratio of train data to total data
    ):
    # Creating a bias of 1 for all the data points to begin with
    
    # rows should be same as number of rows in the data
    number_of_bias_rows = data.shape[0]
    
    # column should be 1 as there's only one bias term
    number_of_bias_column = 1
    
    bias = np.ones(number_of_bias_rows, number_of_bias_column)
    
    # Combining the bias with the dataset features
    combined_data = np.hstack((bias, data))
    
    # Split point
    split_point = int(split_factor * number_of_bias_rows)
    
    # Data Split Operation
    '''
        We need to keep rows until split point.
        For training, we need to keep bias and X only and hence 
        we need to keep everything except the last column.
        
        For testing, we need to keep everything after split point along with bias.
    '''
    X_Train = data[:split_point,:-1]
    X_Test = data[split_point,:-1]
    
    '''
        We need to keep last row for Y_Train until split point.
        For training, we need to keep last row after split point.
        Since we're keeping last row it will give us a NX1 Row vector.
        We need to transform it to Nx[1] format.
    '''
    
    Y_Train = data[split_point:, -1].reshape(-1, 1)
    Y_Test = data[split_point:, -1].reshape(-1, 1)
    
    return X_Train, X_Test, Y_Train, Y_Test

# Hypothesis function to predict output based on thetha
def hypothesis(X: np.array, thetha: np.array):
    return np.dot(X, thetha)

# Compute Gradient for the given delta
def gradient(X: np.ndarray, thetha: np.array, y: np.ndarray):
    X0 = hypothesis(X,thetha)
    delta = X0 - y
    return np.dot(X.T, delta)

# Compute the overall cost function
def costFunction(X: np.ndarray, thetha: np.array, y: np.array):
    X0 = hypothesis(X,thetha)
    delta = X0 - y
    cost = np.dot(delta, delta.T)
    cost /= 2
    return cost[0]

    
    