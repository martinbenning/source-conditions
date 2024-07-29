def gradient_descent(gradient, initial_argument, step_size=1, \
                        max_no_of_iterations=100, print_output=10, \
                        tolerance=1e-6, acceleration=False):

    from numpy import copy, Inf
    from numpy.linalg import norm

    argument = copy(initial_argument)
    previous_argument = copy(argument)
    counter = 0
    adaptive_step_size = 0
    sensitivity = Inf
    norm_list = []
    while (counter < max_no_of_iterations) and (sensitivity > tolerance):
        new_argument = (1 + adaptive_step_size) * argument - \
                        adaptive_step_size * previous_argument
        computed_gradient = gradient(new_argument)
        sensitivity = norm(computed_gradient.reshape(-1, 1, order='F'), 2)
        #sensitivity = norm(gradient(argument).reshape(-1, 1, order='F'), 2)
        norm_list.append(sensitivity)
        previous_argument = copy(argument)
        argument -= step_size * computed_gradient
        counter += 1
        if counter % print_output == 0:
            print("Iteration {k}/{m}, norm of gradient = {g}.".format(k=counter, \
                    m=max_no_of_iterations, g=sensitivity))
        if acceleration == True:
            adaptive_step_size = (counter - 1)/(counter + 3)
    print("Iteration completed after {k}/{m}, norm of gradient = {g}.".format( \
            k=counter, m=max_no_of_iterations, g=sensitivity))
    return argument, norm_list

def proximal_gradient_descent(gradient, proximal_map, initial_argument, \
                        step_size=1, max_no_of_iterations=100, print_output=10, \
                        tolerance=1e-6, acceleration=False):
    
    from numpy import copy, Inf
    from numpy.linalg import norm

    argument = copy(initial_argument)
    previous_argument = copy(argument)
    counter = 0
    adaptive_step_size = 0
    sensitivity = Inf
    norm_list = []
    while (counter < max_no_of_iterations) and (sensitivity > tolerance):
        new_argument = (1 + adaptive_step_size) * argument - \
                        adaptive_step_size * previous_argument
        computed_gradient = gradient(new_argument)
        sensitivity = norm((argument - proximal_map(argument - \
                        computed_gradient)).reshape(-1, 1, order='F'), 2)
        norm_list.append(sensitivity)
        previous_argument = copy(argument)
        argument = proximal_map(argument - step_size * computed_gradient)
        counter += 1
        if print_output is not None:
            if counter % print_output == 0:
                print("Iteration {k}/{m}, norm of gradient = {g}.".format(k=counter, \
                    m=max_no_of_iterations, g=sensitivity))
        if acceleration == True:
            adaptive_step_size = (counter - 1)/(counter + 3)
    #if print_output is not None:
    print("Iteration completed after {k}/{m}, norm of gradient = {g}.".format( \
            k=counter, m=max_no_of_iterations, g=sensitivity))
    return argument, norm_list

def linearised_bregman_iteration(gradient, proximal_map, initial_argument, \
                        step_size=1, max_no_of_iterations=100, print_output=10, \
                        tolerance=1e-6, acceleration=False):

    # Note: there is no convergence guarantee for the Nesterov accelerated version 
    # for general gradients

    from numpy import copy, Inf
    from numpy.linalg import norm

    dual_argument = copy(initial_argument)
    previous_dual_argument = copy(dual_argument)
    counter = 0
    adaptive_step_size = 0
    sensitivity = Inf
    norm_list = []
    while (counter < max_no_of_iterations) and (sensitivity > tolerance):
        new_dual_argument = (1 + adaptive_step_size) * dual_argument - \
                        adaptive_step_size * previous_dual_argument
        primal_argument = proximal_map(new_dual_argument)
        computed_gradient = gradient(primal_argument)
        sensitivity = norm(computed_gradient.reshape(-1, 1, order='F'), 2)
        #sensitivity = norm(gradient(primal_argument).reshape(-1, 1, order='F'), 2)
        norm_list.append(sensitivity)
        previous_dual_argument = copy(dual_argument)
        dual_argument -= step_size * computed_gradient
        counter += 1
        if counter % print_output == 0:
            print("Iteration {k}/{m}, norm of gradient = {g}.".format(k=counter, \
                    m=max_no_of_iterations, g=sensitivity))
        if acceleration == True:
            adaptive_step_size = (counter - 1)/(counter + 3)
    print("Iteration completed after {k}/{m}, norm of gradient = {g}.".format( \
            k=counter, m=max_no_of_iterations, g=sensitivity))
    return primal_argument, dual_argument, norm_list

def PALM(gradients, proximal_maps, initial_arguments, step_sizes=[1, 1], max_no_of_iterations=100, \
            print_output=10, tolerance=1e-6):

    from numpy import copy, Inf
    from numpy.linalg import norm

    arguments = [copy(initial_arguments[0]), copy(initial_arguments[1])]
    counter = 0
    sensitivity = Inf
    norm_list = []

    while (counter < max_no_of_iterations) and (sensitivity > tolerance):
        previous_arguments = [copy(arguments[0]), copy(arguments[1])]
        arguments[0] = proximal_maps[0](arguments[0] - step_sizes[0] * gradients[0](arguments))
        arguments[1] = proximal_maps[1](arguments[1] - step_sizes[1] * gradients[1](arguments))
        if counter > 0:
            sensitivity = norm(arguments[0].flatten() - previous_arguments[0].flatten(), 2) / \
                        norm(arguments[0].flatten(), 2) + norm(arguments[1].flatten() - \
                        previous_arguments[1].flatten(), 2) / norm(arguments[1].flatten(), 2)
        norm_list.append(sensitivity)
        counter += 1
        if counter % print_output == 0:
            print("Iteration {k}/{m}, sensitivity = {s}.".format(k=counter, \
                    m=max_no_of_iterations, s=sensitivity))
    print("Iteration completed after {k}/{m}, sensitivity = {s}.".format( \
            k=counter, m=max_no_of_iterations, s=sensitivity))
    return arguments, norm_list


def PDHG(operator, proximal_maps, initial_arguments, step_sizes=[1, 1], \
            max_no_of_iterations=1000, print_output=100, tolerance=1e-6):

    from numpy import copy, Inf
    from numpy.linalg import norm

    arguments = [copy(initial_arguments[0]), copy(initial_arguments[1])]    
    counter = 0
    sensitivity = Inf
    norm_list = []
    while (counter < max_no_of_iterations) and (sensitivity > tolerance):
        previous_arguments = [copy(arguments[0]), copy(arguments[1])]
        arguments[0] = proximal_maps[0](arguments[0] - step_sizes[0] * (operator.T @ arguments[1]))
        arguments[1] = proximal_maps[1](arguments[1] + step_sizes[1] * \
                        (operator @ (2*arguments[0] - previous_arguments[0])))
        if counter > 0:
            sensitivity = norm(arguments[0].flatten() - previous_arguments[0].flatten(), 2) / \
                        norm(arguments[0].flatten(), 2) + norm(arguments[1].flatten() - \
                        previous_arguments[1].flatten(), 2) / norm(arguments[1].flatten(), 2)
        norm_list.append(sensitivity)
        counter += 1
        if counter % print_output == 0:
            print("Iteration {k}/{m}, sensitivity = {s}.".format(k=counter, \
                    m=max_no_of_iterations, s=sensitivity))
    print("Iteration completed after {k}/{m}, sensitivity = {s}.".format( \
            k=counter, m=max_no_of_iterations, s=sensitivity))
    return arguments, norm_list

def Generalised_PDHG_torch_v4(operator, proximal_maps, func, initial_arguments, \
                        step_sizes=[1, 1], max_no_of_iterations=1000, print_output=100, \
                        tolerance=1e-6):

    import torch

    arguments = [initial_arguments[0], initial_arguments[1]]    
    counter = 0
    sensitivity = float("inf")
    norm_list = []
    while (counter < max_no_of_iterations) and (sensitivity > tolerance):
        previous_arguments = [arguments[0].clone(), arguments[1].clone()]
        computed_gradient = torch.autograd.grad(func(arguments[0]),arguments[0],create_graph=False)[0]
        arguments[0] = proximal_maps[0](arguments[0] - step_sizes[0] * computed_gradient)
        arguments[1] = proximal_maps[1](arguments[1] + step_sizes[1] * \
                        operator @ (2*arguments[0] - previous_arguments[0]))
        if counter > 0:
            sensitivity = torch.norm(arguments[0] - previous_arguments[0],p=2) / \
                        torch.norm(arguments[0],p=2) + torch.norm(arguments[1] - \
                        previous_arguments[1],p=2) / torch.norm(arguments[1],p=2)
        norm_list.append(sensitivity)
        counter += 1
        if counter % print_output == 0:
            print("Iteration {k}/{m}, sensitivity = {s}.".format(k=counter, \
                    m=max_no_of_iterations, s=sensitivity))
    print("Iteration completed after {k}/{m}, sensitivity = {s}.".format( \
            k=counter, m=max_no_of_iterations, s=sensitivity))
    return arguments, norm_list
