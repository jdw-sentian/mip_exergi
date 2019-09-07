import time
import numpy as np
import matplotlib.pyplot as plt

import networkx as nx

from sentian_miami import get_solver

num_days = 20

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


def get_node_vals(solver, lb, ub, N):
    return [solver.NumVar(lb, ub) for _ in range(N)]


def network2nfg(solver, G, max_capacity, max_flow, T):
    """nfg stands for normal flow graph
    In an nfg, all nodes are the same 'type'.
    Edges may be marked as 'passive' to note that
    they split the remainder of free flow.
    """

    for _, data in G.nodes(data=True):
        if "vals" not in data:
            data["vals"] = get_node_vals(solver, 0, max_capacity, T)

    for _, _, data in G.edges(data=True):
        if "flow" not in data:
            data["flow"] = get_node_vals(solver, 0, max_flow, T)


def constraints_from_nfg(solver, nfg):
    """Makes the associated MIP model constraints
    for an nfg
    """

    # constraints for flows
    # actives are free variables, bound only by resource constraint
    # passives split remaining flow equally
    for x_name in nfg:
        x = nfg.nodes[x_name]["vals"]
        out_edges = nfg.out_edges(nbunch=x_name, data=True)
        try:
            _, _, out_edges_data = zip(*out_edges)
        except ValueError:
            # if there are no out edges, skip constraints
            continue
        active = [data["flow"] for data in out_edges_data if data["active"]]
        passive = [data["flow"] for data in out_edges_data if not data["active"]]
        if len(active):
            pulled = [solver.Sum(pull_t) for pull_t in zip(*active)]
        else:
            pulled = [0]*len(x)
        remainder = [x_t - pull_t for x_t, pull_t in zip(x, pulled)]
        [solver.Add(rem_t >= 0) for rem_t in remainder]
        n_passive = len(passive)
        for passive_t, rem_t in zip(zip(*passive), remainder):
            for pass_t in passive_t:
                solver.Add(pass_t == rem_t / n_passive)

    # constraints for nodes
    # the value of a node is just the sum of 
    for x_name in nfg:
        x = nfg.nodes[x_name]["vals"]
        in_edges = nfg.in_edges(nbunch=x_name, data=True)
        try:
            _, _, in_edges_data = zip(*in_edges)
        except ValueError:
            # if there are no in edges, skip constraints
            continue
        for t in range(len(x)):
            flows = [data["flow"] for data in in_edges_data]
            delays = [data["delay"] for data in in_edges_data]
            # first version: no heat loss
            in_flows = [flow[t - d] for flow, d in zip(flows, delays) if t - d >= 0]
            if len(in_flows):
                in_flow = solver.Sum(in_flows)
            else:
                in_flow = nfg.nodes[x_name]["initial"]
            solver.Add(in_flow == x[t])
            ''' 
            # this bug was caused by trying to do:
                solver.Add(x[t] == in_flow)
            this is *different* than doing
                solver.Add(in_flow == x[t])
            and causes an exception

            except Exception as e:
                print("Num in_flows:", len(in_flows))
                print("x_name:", x_name)
                print("x[t]:", x[t])
                print("t:", t)
                raise e
            '''
            # what about state, you say? self-flow is possible!


def abs_diff_var(solver, val):
    x = solver.NumVar(0)
    solver.Add(x >= val)
    solver.Add(x >= -val)
    return x


def production_inertia_cost(solver, P):
    change_vars = [abs_diff_var(solver, p0 - p1) for p0, p1 in zip(P[:-1], P[1:])]
    return solver.Sum(change_vars)


def plan():
    t0 = time.time()
    solver = get_solver("CBC")

    np.random.seed(0)
    # parameters
    demand = get_demand_forecast()
    T = len(demand) # planning horizon in time units
    n_plants = 1
    max_production = 100 # MW
    max_buy = 0 # MW
    max_sell = 0 # MW
    prod_price = 1 # € / MW
    prod_inertia = 0.2 # € / dMW/dt
    buy_price = 1.1 # € / MW
    sell_price = 0.9 # € / MW
    max_temp = 100
    delay = 5 # time from production to consumer in time units

    init_pre_temp = 80
    init_post_temp = 50

    min_forward_temp = 78
    max_forward_temp = 82

    Prod = get_node_vals(solver, 0, max_production, T)
    Buy = get_node_vals(solver, 0, max_buy, T)
    Sell = get_node_vals(solver, 0, max_sell, T)
    P = [p+b-s for p, b, s in zip(Prod, Buy, Sell)] # Total power, MW
    for p in P:
        solver.Add(p >= 0)
        solver.Add(p <= max_production)

    # build network graph
    G = nx.DiGraph()
    G.add_node("production", **{"vals": P})
    G.add_node("consumer", **{"vals": demand})
    
    #G.add_edge("production", "consumer", **{"delay": delay, "active": True})

    
    G.add_node("x0", **{"initial": init_post_temp})
    G.add_node("xc" ,**{"initial": init_pre_temp})

    G.add_edge("production", "x0", **{"delay": 0, "active": False})
    G.add_edge("xc", "consumer", **{"delay": 0, "active": True})
    G.add_edge("x0", "xc", **{"delay": delay, "active": False})
    G.add_edge("xc", "x0", **{"delay": delay, "active": False})
    

    network2nfg(solver, G, max_temp, max_temp, T)
    constraints_from_nfg(solver, G)

    for x_t in G.nodes["xc"]["vals"]:
        solver.Add(x_t >= min_forward_temp)
        solver.Add(x_t <= max_forward_temp)

    # objective
    prod_base_cost = solver.Sum(Prod)
    prod_inertia_cost = production_inertia_cost(solver, Prod)
    prod_cost = prod_price*prod_base_cost + prod_inertia*prod_inertia_cost
    cost = prod_cost + buy_price * solver.Sum(Buy) - sell_price * solver.Sum(Sell)
    solver.SetObjective(cost, maximize=False)

    t1 = time.time()
    print("Build time: {0:.3f}".format(t1 - t0))

    # solving
    solver.Solve(time_limit=10)

    # presenting
    P_solved = [solver.solution_value(p) for p in P]
    Prod_solved = [solver.solution_value(p) for p in Prod]
    T_solved = [solver.solution_value(x) for x in G.nodes["xc"]["vals"]]
    Tx0_solved = [solver.solution_value(x) for x in G.nodes["x0"]["vals"]]
    #acc_in = [solver.solution_value(a) for a in acc.in_flow]
    #acc_out = [solver.solution_value(a) for a in acc.out_flow]
    #acc_balance = [solver.solution_value(a) for a in acc.balance]
    #T_pipe_solved = [[solver.solution_value(xt) for xt in x] for x in X]
    #T_pipe_solved = np.array(T_pipe_solved)
    buy_solved = [solver.solution_value(b) for b in Buy]
    sell_solved = [solver.solution_value(s) for s in Sell]

    cost_solved = solver.solution_value(cost)
    print("Total cost: {0:.1f}".format(cost_solved))
    print("Sum of demand: {0:.3f}".format(sum(demand)))
    print("Sum of production: {0:.3f}".format(sum(P_solved)))

    x = list(range(T))
    #fig, (ax_power, ax_acc, ax_market, ax_img) = plt.subplots(nrows=4, sharex=True)
    fig, ax_power = plt.subplots(nrows=1, sharex=True)
    ax_power.step(x, Prod_solved, color='b')
    ax_power.step(x, P_solved, color='b', linestyle="--")
    ax_power.step(x, demand, color='r')
    ax_power.step(x, T_solved, color='g')
    #ax_power.step(x, Tx0_solved)    
    #ax_acc.step(x, acc_in)
    #ax_acc.step(x, acc_balance, linewidth=3)
    #ax_acc.step(x, acc_out)
    #ax_market.step(x, sell_solved)
    #ax_market.step(x, buy_solved)

    ax_power.legend(["Production", "Production + Market", "Demand", "Forward temperature"], loc=1)
    #ax_power.set_xlabel("Time / h")
    ax_power.set_ylabel("Power MW / Temp C")
    ax_power.set_title("Total cost: {0:.1f}".format(cost_solved))

    #ax_acc.legend(["Accumulator in", "Acc balance", "Acc out"], loc=1)
    #ax_acc.set_ylabel("Power MW")

    #ax_market.legend(["Sold power", "Bought power"], loc=1)
    #ax_market.set_ylabel("Power MW")

    #_, ax_img = plt.subplots()
    #ax_img.imshow(T_pipe_solved, aspect="auto", cmap="coolwarm")
    #ax_img.set_xlabel("Time / h")
    #ax_img.set_ylabel("Piece of line")
    #ax_img.set_title("Heat propagation from plant (top),\n\
    #                  to customer (middle), and back (bottom)")

    plt.show()


if __name__ == '__main__':
    plan()