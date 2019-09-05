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
    demand = np.array([30, 30, 30, 30, 30, 30,
                     40, 50, 50, 50, 40, 30,
                     30, 30, 40, 40, 50, 50,
                     60, 60, 60, 50, 40, 30]*num_days) * 1.0
    demand += np.random.normal(size=len(demand)) * 2
    return demand


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

    np.random.seed(0)
    # parameters
    demand = get_demand_forecast()
    T = len(demand) # planning horizon in hours
    n_plants = 1
    max_production = 100 # MW
    max_buy = 20 # MW
    max_sell = 20 # MW
    prod_price = 1 # € / MW
    prod_inertia = 0.2 # € / dMW/dt
    buy_price = 1.1 # € / MW
    sell_price = 0.9 # € / MW
    max_temp = 100
    delay = 5 # time from production to consumer in hours

    init_pre_temp = 90
    init_post_temp = 50

    min_forward_temp = 75
    max_forward_temp = 90

    # decision variable: production (MW)
    Prod = [solver.NumVar(0, max_production) for _ in range(T)]
    Buy = [solver.NumVar(0, max_buy) for _ in range(T)]
    Sell = [solver.NumVar(0, max_sell) for _ in range(T)]
    P = [p+b-s for p, b, s in zip(Prod, Buy, Sell)] # Total power, MW
    for p in P:
        solver.Add(p >= 0)
        solver.Add(p <= max_production)
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
    prod_base_cost = solver.Sum(Prod)
    prod_inertia_cost = production_inertia_cost(solver, Prod)
    prod_cost = prod_price*prod_base_cost + prod_inertia*prod_inertia_cost
    cost = prod_cost + buy_price * solver.Sum(Buy) - sell_price * solver.Sum(Sell)
    solver.SetObjective(cost, maximize=False)

    solver.Solve(time_limit=10)

    P_solved = [solver.solution_value(p) for p in P]
    Prod_solved = [solver.solution_value(p) for p in Prod]
    T_solved = [solver.solution_value(x) for x in X[delay-1]]
    acc_in = [solver.solution_value(a) for a in acc.in_flow]
    acc_out = [solver.solution_value(a) for a in acc.out_flow]
    acc_balance = [solver.solution_value(a) for a in acc.balance]
    T_pipe_solved = [[solver.solution_value(xt) for xt in x] for x in X]
    T_pipe_solved = np.array(T_pipe_solved)
    buy_solved = [solver.solution_value(b) for b in Buy]
    sell_solved = [solver.solution_value(s) for s in Sell]

    cost_solved = solver.solution_value(cost)
    print("Total cost: {0:.1f}".format(cost_solved))

    x = list(range(T))
    fig, (ax_power, ax_acc, ax_market, ax_img) = plt.subplots(nrows=4, sharex=True)
    ax_power.step(x, Prod_solved, color='b')
    ax_power.step(x, P_solved, color='b', linestyle="--")
    ax_power.step(x, demand, color='r')
    ax_power.step(x, T_solved, color='g')
    ax_acc.step(x, acc_in)
    ax_acc.step(x, acc_balance, linewidth=3)
    ax_acc.step(x, acc_out)
    ax_market.step(x, sell_solved)
    ax_market.step(x, buy_solved)

    ax_power.legend(["Production", "Production + Market", "Demand", "Forward temperature"], loc=1)
    #ax_power.set_xlabel("Time / h")
    ax_power.set_ylabel("Power MW / Temp C")
    ax_power.set_title("Total cost: {0:.1f}".format(cost_solved))

    ax_acc.legend(["Accumulator in", "Acc balance", "Acc out"], loc=1)
    ax_acc.set_ylabel("Power MW")

    ax_market.legend(["Sold power", "Bought power"], loc=1)
    ax_market.set_ylabel("Power MW")

    #_, ax_img = plt.subplots()
    ax_img.imshow(T_pipe_solved, aspect="auto", cmap="coolwarm")
    ax_img.set_xlabel("Time / h")
    ax_img.set_ylabel("Piece of line")
    #ax_img.set_title("Heat propagation from plant (top),\n\
    #                  to customer (middle), and back (bottom)")

    plt.show()


if __name__ == '__main__':
    plan()