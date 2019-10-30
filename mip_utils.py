import numpy as np

from sentian_miami import get_solver

def get_numvars(solver, lb, ub, N):
    return np.array([solver.NumVar(lb, ub) for _ in range(N)])

def abs_var(solver, val):
    x = solver.NumVar(0)
    solver.Add(x >= val)
    solver.Add(x >= -val)
    return x


def L1_energy(solver, P):
    """L1 energy is integral of abs of derivative
    In discrete case
    sum(|p_{i+1} - p{i}| for i in range(N-1))
    """
    change_vars = [abs_var(solver, p0 - p1) for p0, p1 in zip(P, P[1:])]
    return solver.Sum(change_vars)


def inverse_cumulative(solver, values, S, max_sum):
    """At what index does the cumulative sum of `values`
    surpass S? 

    `values` is a vector of MIP vars, assumed to be positive
    `S` is a float, assumed to be less than sum(values)
    `max_sum` is a float, representing the max value of sum(values)

    sum(values) < S, will return all zeros
    sum(values) > max_sum, will make model infeasible
    """
    N = len(values)
    # cumulative sums
    c_values = [solver.Sum(values[:(i+1)]) for i in range(N)]
    return index_of_surpass(solver, c_values, S, max_sum)


def index_of_surpass(solver, values, S, max_value):
    """Returns one-hot that indicate the first value of
    `values` that is larger than `S`.

    Assumes that `values` is monotonously increasing.
    Assumes that all values are less than `max_value`.
    """
    N = len(values)
    # indicates if the cumulative value is larger than S
    indicators = [solver.IntVar(0, 1) for _ in range(N)]

    # derivative of indicators - these will be returned
    thresholds = [solver.IntVar(0, 1) for _ in range(N)]

    # bind indicators
    tol = 1e-6  # resolve ambiguity when c_val == S for some c_val
    for ind, val in zip(indicators, values):
        solver.Add(ind <= val / S)
        solver.Add(ind >= (val - S) / (max_value + tol - S) + tol)

    # bind thresholds
    for thres, ind0, ind1 in zip(thresholds, [0] + indicators, indicators):
        solver.Add(thres == ind1 - ind0)

    return thresholds


def element_from_one_hot(solver, values, indicators, max_val):
    """Given a one-hot vector `indicators`, 
    and a vector of elements `values`, return the element
    of `values` on the index where `indicators` is 1. 

    All values in `values` are assumed to be positive.
    """
    H = solver.NumVar(lb=0)

    for ind, Hi in zip(indicators, values):
        solver.Add(H >= Hi - max_val * (1 - ind))
        solver.Add(H <= Hi + max_val * (1 - ind))

    return H


def inverse_cumulative_element(solver, values, densities, 
                                mass_threshold, max_mass, max_value):
    """What is the value at the index of `values`,
    where the cumulative of `densities` surpass `mass_threshold`?

    For the district heating network, 
    `values` represent a history of in-temperatures,
    `densities` represent a history of flow speeds, and
    `mass_threshold` represents the length of a pipe. 
    The returned value represents the current out-temperature.

    Need to supply two bound values, 
    `max_mass`: maximum flow speed times time horizon, and
    `max_value`: the max temperature.
    These are required for the model to be feasible and correct.
    In practice, these can be calculated from model constraints.
    """

    thresholds = inverse_cumulative(solver, densities, mass_threshold, max_mass)
    return element_from_one_hot(solver, values, thresholds, max_value), thresholds


def test_inverse_cumulative():
    solver = get_solver("CBC")
    densities = [10, 1, 1, 1, 1]
    mass_threshold = 10
    max_mass = 30

    thresholds = inverse_cumulative(solver, densities, mass_threshold, max_mass)

    solver.Solve(time_limit=10)

    values_solve = [solver.solution_value(thres) for thres in thresholds]
    print(values_solve)


def test_element_from_one_hot():
    max_value = 10

    solver = get_solver("CBC")
    values = [1, 2, 3, 4, 7]
    indicators = [0, 0, 0, 0, 1]

    H = element_from_one_hot(solver, values, indicators, max_value)

    solver.Solve(time_limit=10)

    H_solve = solver.solution_value(H)

    print(H_solve)


def test_inverse_cumulative_element():
    values = [1, 2, 3, 4, 7]
    max_value = 10
    
    densities = [10, 1, 1, 1, 1]
    mass_threshold = 10
    max_mass = 30

    solver = get_solver("CBC")

    H, thresholds = inverse_cumulative_element(solver, values, densities, 
                                    mass_threshold, max_mass, max_value)

    solver.Solve(time_limit=10)

    H_solve = solver.solution_value(H)
    print(H_solve)

    thres_solved = [solver.solution_value(thres) for thres in thresholds]
    print(thres_solved)


if __name__ == '__main__':
    test_inverse_cumulative()
    #test_element_from_one_hot()
    test_inverse_cumulative_element()
