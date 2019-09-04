import numpy as np
import matplotlib.pyplot as plt

from sentian_miami import get_solver

def get_demand_forecast():
    # unit in hypothetical MW
    return np.array([30, 30, 30, 30, 30, 30,
                     40, 50, 50, 50, 40, 30,
                     30, 30, 40, 40, 50, 50,
                     60, 60, 60, 50, 40, 30]*5)


def bind_delay_line(solver, X):
    for x0, x1 in zip(X[:-1], X[1:]):
        for x0t_1, x1t in zip(x0[:-1], x1[1:]):
            solver.Add(x1t == x0t_1)
            # version with temp loss:
            #solver.Add(x1[t] == x0[t-1] - loss_coeff*(x0[t-1] - T_out))


def bind_heat_exchange(solver, X0, X1, vals):
    for x0, x1, val in zip(X0[:-1], X1[1:], vals):
        solver.Add(x1 == x0 + val)


def plan():
    solver = get_solver("CBC")

    # parameters
    demand = get_demand_forecast()
    T = len(demand) # planning horizon in hours
    n_plants = 1
    max_production = 100 # MW
    max_temp = 100
    delay = 2 # time from production to consumer in hours

    # decision variable: production (MW)
    P = [solver.NumVar(0, max_production) for _ in range(T)]
    # state variables: temperature (C) in different parts of the network
    X = [[solver.NumVar(0, max_temp) for _ in range(T)] 
         for _ in range(delay * 2)]
    X_c = X[delay-1] # temperature that reaches customer
    X_pre_customer = X[:delay]
    X_post_customer = X[delay:]

    # initial conditions
    for x in X_pre_customer:
        solver.Add(x[0] == 80)

    for x in X_post_customer:
        solver.Add(x[0] == 50)

    # bindings
    bind_delay_line(solver, X_pre_customer)
    bind_delay_line(solver, X_post_customer)

    power2temp = 1 # simplify 1 MW <=> 1 degree difference
    bind_heat_exchange(solver, 
                       X_post_customer[-1], X_pre_customer[0], 
                       P * power2temp)
    bind_heat_exchange(solver, 
                       X_pre_customer[-1], X_post_customer[0], 
                       demand * -1)

    # customer promise constraint
    #for x in X_c:
    #    solver.Add(x >= 70)

    # objective
    cost = solver.Sum(P)
    solver.SetObjective(cost, maximize=False)

    solver.Solve(time_limit=10)

    P_solved = [solver.solution_value(p) for p in P]
    T_solved = [solver.solution_value(x) for x in X[0]]

    plt.plot(P_solved, color='b')
    plt.plot(demand, color='r')
    plt.plot(T_solved, color='g')

    plt.legend(["Planned production", "Demand", "Forward temperature"])
    plt.show()


if __name__ == '__main__':
    plan()