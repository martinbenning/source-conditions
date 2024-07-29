from numpy import argmax, copy, eye, maximum, minimum, ones, sign, where, zeros
from numpy import abs as absval
from numpy import amin as argminimum
from numpy import inf as infinity
from numpy import sum as nsum
from numpy.linalg import norm, solve

def solve_constrained_linear_problem(matrix, data, subgradient, stepsizes=[0.25, 4], \
                                     threshold=10**(-10), maximum_no_of_iterations=10000, output=False):
    
    shifted_gram_matrix = eye(matrix.shape[1]) + stepsizes[0] * (matrix.T @ matrix)
    right_hand_side = matrix.T @ data
    primal = zeros((matrix.shape[1], 1))
    dual = copy(primal)
    counter = 0
    criterion = infinity
    while (counter < maximum_no_of_iterations) and (criterion > threshold):
        previous_primal = copy(primal)
        primal = solve(shifted_gram_matrix, primal + stepsizes[0] * (right_hand_side - \
                                                                    subgradient * dual))
        dual = minimum(0, dual + stepsizes[1] * subgradient * (2*primal - previous_primal))
        #criterion = norm(primal * (matrix.T @ (matrix @ primal - data)))
        criterion = norm((matrix.T @ (matrix @ primal - data)) + dual * subgradient)
        counter += 1
        if output:
            print("Iteration {i}/{m}, tolerance {c}/{t}.".format(i = counter, \
                        m=maximum_no_of_iterations, c=criterion, t=threshold))
    if output:    
        print("Iteration completed after {i} iterations with tolerance {t}.".format(i = \
                    counter, t=criterion))
        
    return primal, dual

def adaptive_inverse_scale_space_method(matrix, data, noise_level=10**(-5), \
                                        maximum_no_of_iterations=100, tolerance=10**(-8), output=True):
    gram_matrix = matrix.T @ matrix
    data_projection = matrix.T @ data
    times = []
    times.append(1 / norm(data_projection, infinity))
    subgradient = times[0] * data_projection
    index_set = where(absval(subgradient) >= (1 - tolerance))[:][0]
    index_set_complement = where(absval(subgradient) < (1 - tolerance))[:][0]
    reconstruction = zeros((matrix.shape[1], 1))
    
    counter = 0
    discrepancy = 1/2 * nsum((data_projection) ** 2)
    new_times = True
    while (counter < maximum_no_of_iterations) and (discrepancy > noise_level) and new_times:
        
        ### Solve linear system ###
        reduced_matrix = matrix[:, index_set]
        linear_system_solution, _ = solve_constrained_linear_problem(reduced_matrix, \
                                                            data, subgradient[index_set])
        reconstruction[index_set] = linear_system_solution
        
        ### Find next time step ###
        residual = data_projection - (gram_matrix @ reconstruction)
        shifted_residual = subgradient - times[counter] * residual
        times1 = (1 - shifted_residual)/residual
        times1 = times1[index_set_complement]
        times1 = times1[times1 >= times[counter]]
        times2 = (-1 - shifted_residual)/residual
        times2 = times2[index_set_complement]
        times2 = times2[times2 >= times[counter]]
        if (times1.size == 0) and (times2.size == 0):
            new_times = False
        elif (times1.size == 0):
            times.append(argminimum(times2))
        elif (times2.size == 0):
            times.append(argminimum(times1))
        else:
            times.append(minimum(argminimum(times1), argminimum(times2)))
        
        ### Update subgradient ###
        if new_times:
            temporary_variable = subgradient[index_set]
            subgradient = subgradient + (times[counter + 1] - times[counter]) * residual
            subgradient[index_set] = temporary_variable
        
        ### Update index set ###
        index_set = where(absval(subgradient) >= (1 - tolerance))[:][0]
        index_set_complement = where(absval(subgradient) < (1 - tolerance))[:][0]
        
        ### Update counter and discrepancy ###
        counter += 1
        discrepancy = 1/2 * nsum((matrix @ reconstruction - data) ** 2)
        
        if output:
            print("Iteration {i} with discrepancy {d}.".format(i = counter, d=discrepancy))

    if output:
        print("Iteration completed after {i} iterations with discrepancy {d}.".format(i = counter, d=discrepancy))
        
    return reconstruction

def soft_thresholding(argument, threshold):
    return sign(argument) * maximum(absval(argument) - threshold, 0)

def fista(matrix, data, regularisation_parameter=1, maximum_no_of_iterations=100000, tolerance=10**(-8), \
          acceleration=True, output=True):
    step_size = 1/(norm(matrix, 2) ** 2)
    adaptive_step_size = 0
    weights = zeros((matrix.shape[1], 1))
    previous_weights = copy(weights)
    optimality_criterion = infinity
    counter = 0
    while (counter < maximum_no_of_iterations) and (optimality_criterion > tolerance):  
        argument = (1 + adaptive_step_size) * weights - adaptive_step_size * previous_weights
        gradient = matrix.T @ (matrix @ argument - data)
        previous_weights = copy(weights)
        weights = soft_thresholding(argument - step_size * gradient.reshape(-1, 1), step_size * \
                                    regularisation_parameter)
        optimality_criterion = norm(weights - soft_thresholding(weights + 1/regularisation_parameter * \
                                                           matrix.T @ (data - matrix @ weights), 1))
        counter += 1
        
        if acceleration == True:
            adaptive_step_size = (counter - 1)/(counter + 3)
        
        if output == True:
            print("Iteration no. {i}, optimality criterion: {g}".format(i = counter, g = optimality_criterion))
    
    if output == True:
        print("Iteration completed after {i} iterations with optimality criterion {g}.".format(i = counter, \
                                                                                    g = optimality_criterion))
    return weights