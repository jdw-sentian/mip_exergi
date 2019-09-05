import numpy as np
import matplotlib.pyplot as plt

from sentian_miami import get_solver

num_days = 10

class Accumulator:
    def __init__(self, solver, T):
        max_flow = 20
        max_capacity = 20
        self.in_flow = [solver.NumVar(0, max_flow) for _ in range(T)]
        self.out_flow = [solver.NumVar(0, max_flow) for _ in range(T)]
        self.balance = [solver.NumVar(0, max_capacity) for _ in range(T)]

        solver.Add(self.balance[0] == 0)
        for t in range(1, len(self.balance)):
            solver.Add(self.out_flow[t] <= self.balance[t])
            solver.Add(self.balance[t] == self.balance[t-1] \
                                          + self.in_flow[t-1] \
                                          - self.out_flow[t])


def get_demand_forecast():
    # unit in hypothetical MW
    return np.array([30, 30, 30, 30, 30, 30,
                     40, 50, 50, 50, 40, 30,
                     30, 30, 40, 40, 50, 50,
                     60, 60, 60, 50, 40, 30]*num_days)


'''
def bind_delay_line(solver, X):
    for x0, x1 in zip(X[:-1], X[1:]):
        for x0t_1, x1t in zip(x0[:-1], x1[1:]):
            #solver.Add(x1t == x0t_1)
            # version with temp loss:
            loss_coeff = 0.01
            T_out = 0
            solver.Add(x1t == x0t_1 - loss_coeff*(x0t_1 - T_out))
'''

def bind_line_vars(solver, x0, x1, diff=None):
    if diff is None:
        diff = [0]*len(x0)
    for x0t_1, x1t, d in zip(x0, x1[1:], diff):
        #solver.Add(x1t == x0t_1 + d)
        t_in = x0t_1 + d
        Q_loss = 0.01
        T_out = 0
        solver.Add(x1t == t_in - Q_loss * (t_in - T_out))

def bind_delay_line(solver, X, acc_idx=None):
    if acc_idx is None:
        acc_idx = len(X)
    else:
        assert acc_idx > 0
        assert acc_idx < len(X) - 1

    acc = Accumulator(solver, len(X[0]))
    for idx, (x0, x1) in enumerate(zip(X, X[1:])):
        if idx == acc_idx:
            bind_line_vars(solver, x0, x1, [-var for var in acc.in_flow])
        elif idx == acc_idx + 1:
            bind_line_vars(solver, x0, x1, acc.out_flow)
        else:
            bind_line_vars(solver, x0, x1)

    return acc


def bind_heat_exchange(solver, X0, X1, vals):
    for x0, x1, val in zip(X0[:-1], X1[1:], vals):
        solver.Add(x1 == x0 + val)


def abs_diff_var(solver, val):
    x = solver.NumVar(0)
    solver.Add(x >= val)
    solver.Add(x >= -val)
    return x


def production_inertia_cost(solver, P):
    change_vars = [abs_diff_var(solver, p0 - p1) for p0, p1 in zip(P[:-1], P[1:])]
    return solver.Sum(change_vars)


def plan():
    solver = get_solver("CBC")

    # parameters
    demand = get_demand_forecast()
    T = len(demand) # planning horizon in hours
    n_plants = 1
    max_production = 100 # MW
    max_temp = 100
    delay = 5 # time from production to consumer in hours

    init_pre_temp = 90
    init_post_temp = 50

    min_forward_temp = 75
    max_forward_temp = 90

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
        solver.Add(x[0] == init_pre_temp)

    solver.Add(X_pre_customer[0][-1] == init_pre_temp)
    #for p0, p1 in zip(P[:-24], P[24:]):
    #    solver.Add(p0 <= p1 + 5)
    #    solver.Add(p0 >= p1 - 5)

    for x in X_post_customer:
        solver.Add(x[0] == init_post_temp)

    # bindings
    acc = bind_delay_line(solver, X_pre_customer, acc_idx=1)
    bind_delay_line(solver, X_post_customer)

    power2temp = 1 # simplify 1 MW <=> 1 degree difference
    bind_heat_exchange(solver, 
                       X_post_customer[-1], X_pre_customer[0], 
                       P * power2temp)
    bind_heat_exchange(solver, 
                       X_pre_customer[-1], X_post_customer[0], 
                       demand * -1)

    # customer promise constraint
    for x in X_c:
        solver.Add(x >= min_forward_temp)
        solver.Add(x <= max_forward_temp)

    for x in X[0]:
        solver.Add(x >= min_forward_temp)
        solver.Add(x <= max_forward_temp)


    # objective
    prod_base_cost = solver.Sum(P)
    prod_inertia_cost = production_inertia_cost(solver, P)
    cost = 1 * prod_base_cost + 0.9 * prod_inertia_cost
    solver.SetObjective(cost, maximize=False)

    solver.Solve(time_limit=10)

    P_solved = [solver.solution_value(p) for p in P]
    T_solved = [solver.solution_value(x) for x in X[delay-1]]
    acc_in = [solver.solution_value(a) for a in acc.in_flow]
    acc_out = [solver.solution_value(a) for a in acc.out_flow]
    acc_balance = [solver.solution_value(a) for a in acc.balance]
    T_pipe_solved = [[solver.solution_value(xt) for xt in x] for x in X]
    T_pipe_solved = np.array(T_pipe_solved)

    x = list(range(T))
    fig, (ax_plot, ax_img) = plt.subplots(nrows=2, sharex=True)
    ax_plot.step(x, P_solved, color='b')
    ax_plot.step(x, demand, color='r')
    ax_plot.step(x, T_solved, color='g')
    ax_plot.step(x, acc_in)
    ax_plot.step(x, acc_balance, linewidth=3)
    ax_plot.step(x, acc_out)

    ax_plot.legend(["Planned production", "Demand", "Forward temperature", 
                "Accumulator in", "Acc balance", "Acc out"], loc=1)
    ax_plot.set_xlabel("Time / h")
    ax_plot.set_ylabel("Temp C / Power MW")
    #ax_plot.set_title("")

    #_, ax_img = plt.subplots()
    ax_img.imshow(T_pipe_solved, aspect="auto", cmap="coolwarm")
    ax_img.set_xlabel("Time / h")
    ax_img.set_ylabel("Piece of line")
    ax_img.set_title("Heat propagation from plant (top),\n\
                      to customer (middle), and back (bottom)")

    plt.show()


if __name__ == '__main__':
    plan()