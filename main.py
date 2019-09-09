import time
import numpy as np
import matplotlib.pyplot as plt

import networkx as nx

from sentian_miami import get_solver

num_days = 10

'''
class Accumulator:
    def __init__(self, solver, T, name="acc"):
        self.name = name
        self.max_flow = 5
        self.max_capacity = 5
        self.in_flow = [solver.NumVar(0, self.max_flow) for _ in range(T)]
        self.out_flow = [solver.NumVar(0, self.max_flow) for _ in range(T)]
        self.balance = [solver.NumVar(0, self.max_capacity) for _ in range(T)]

    def integrate(self, solver, G, in_node, out_node, in_delay, out_delay):
        G.add_node(self.name, **{"initial": 0})
        G.add_edge(self.name, self.name, **{"delay": 1, "active": False, "flow": self.balance})
        # see if we can make the input delay 0 instead
        G.add_edge(in_node, self.name, **{"delay": in_delay, "active": True, "flow": self.in_flow})
        G.add_edge(self.name, out_node, **{"delay": out_delay, "active": True, "flow": self.out_flow})
'''

def get_demand_forecast():
    # unit in hypothetical MW
    demand = np.array([30, 30, 30, 30, 30, 30,
                     40, 50, 50, 50, 40, 30,
                     30, 30, 40, 40, 50, 50,
                     60, 60, 60, 50, 40, 30]*num_days) * 1.0
    demand += np.random.normal(size=len(demand)) * 5
    return demand


def get_numvars(solver, lb, ub, N):
    return np.array([solver.NumVar(lb, ub) for _ in range(N)])


def set_divs(G, divs, default_div):
    for x_name, data in G.nodes(data=True):
        data["div"] = divs.get(x_name, default_div())


def set_flows(solver, G, flows, default_flow):
    for x_name, y_name, data in G.edges(data=True):
        data["flow"] = flows.get((x_name, y_name), default_flow())


def bind_out_flows(solver, G, T):
    """Constraints for out flows
    actives are free variables, bound only by resource constraint
    passives split remaining flow equally
    """
    out_flows = {}
    for x_name in G:
        out_edges = G.out_edges(nbunch=x_name, data=True)
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
            pulled = [0]*T
        #remainder = [x_t - pull_t for x_t, pull_t in zip(x, pulled)]
        remainder = [solver.NumVar(0, 100) for _ in range(T)]
        #[solver.Add(rem_t >= 0) for rem_t in remainder]
        n_passive = len(passive)
        for passive_t, rem_t in zip(zip(*passive), remainder):
            for pass_t in passive_t:
                solver.Add(pass_t == rem_t / n_passive)
        out_flows[x_name] = [rem_t + pull_t for rem_t, pull_t in zip(remainder, pulled)]

    return out_flows


def bind_in_flows(solver, G, T):
    """Constraints for in flows
    """
    in_flows = {}
    for x_name in G:
        in_edges = G.in_edges(nbunch=x_name, data=True)
        try:
            _, _, in_edges_data = zip(*in_edges)
        except ValueError:
            # if there are no in edges, skip constraints
            continue
        in_flow = []
        for t in range(T):
            flows = [data["flow"] for data in in_edges_data]
            delays = [data["delay"] for data in in_edges_data]
            # first version: no heat loss
            in_flows_t = [flow[t - d] for flow, d in zip(flows, delays) if t - d >= 0]
            if len(in_flows_t):
                in_flow_t = solver.Sum(in_flows_t)
            else:
                in_flow_t = 80 #G.nodes[x_name]["initial"]
            #solver.Add(in_flow == x[t])
            in_flow.append(in_flow_t)
        in_flows[x_name] = in_flow

    return in_flows


def divergence_constraints(solver, G, in_flows, out_flows, default_flow_val, burn_in=0):
    """Constrain so that 
    out_flow - in_flow = divergence
    """
    for x_name in G:
        in_flow = in_flows.get(x_name, default_flow_val())
        out_flow = out_flows.get(x_name, default_flow_val())
        div = G.nodes[x_name]["div"]
        for t, (in_t, out_t, div_t) in enumerate(zip(in_flow, out_flow, div)):
            if t < burn_in:
                continue
            solver.Add(out_t - in_t == div_t)

    # remember the out_flow, for constraints
    for x_name in G:
        out_flow = out_flows.get(x_name, default_flow_val())
        G.nodes[x_name]["out_flow"] = out_flow


def abs_diff_var(solver, val):
    x = solver.NumVar(0)
    solver.Add(x >= val)
    solver.Add(x >= -val)
    return x


def production_inertia_cost(solver, P):
    change_vars = [abs_diff_var(solver, p0 - p1) for p0, p1 in zip(P[:-1], P[1:])]
    return solver.Sum(change_vars)


def set_objective(solver, G, prod_price, prod_inertia, buy_price, sell_price):
    Plant = G.nodes["plant"]["div"]
    Buy = G.nodes["buy"]["div"]
    Sell = -G.nodes["sell"]["div"]
    Prod = G.edges["production", "x0"]["flow"]
    prod_base_cost = solver.Sum(Plant)
    prod_inertia_cost = production_inertia_cost(solver, Plant)
    prod_cost = prod_price*prod_base_cost + prod_inertia*prod_inertia_cost
    cost = prod_cost + buy_price * solver.Sum(Buy) - sell_price * solver.Sum(Sell)
    solver.SetObjective(cost, maximize=False)
    return cost


def get_structure(delay):
    G = nx.DiGraph()
    G.add_node("plant")
    G.add_node("buy")
    G.add_node("sell")
    G.add_node("production")
    G.add_node("consumer")
    G.add_node("x0")
    G.add_node("xc")

    G.add_edge("plant", "production", **{"delay": 0, "active": False})
    G.add_edge("buy", "production", **{"delay": 0, "active": False})
    G.add_edge("production", "sell", **{"delay": 0, "active": True})
    G.add_edge("production", "x0", **{"delay": 0, "active": False})
    G.add_edge("xc", "consumer", **{"delay": 0, "active": True})
    G.add_edge("x0", "xc", **{"delay": delay, "active": False})
    G.add_edge("xc", "x0", **{"delay": delay, "active": False})
    
    # accumulator
    G.add_node("acc")
    G.add_edge("acc", "acc", **{"delay": 1, "active": False})
    G.add_edge("x0", "acc", **{"delay": 0, "active": True})
    G.add_edge("acc", "xc", **{"delay": delay, "active": True})

    return G


def plan():
    t0 = time.time()
    solver = get_solver("CBC")

    np.random.seed(0)
    # parameters
    demand = get_demand_forecast()
    T = len(demand) # planning horizon in time units
    n_plants = 1
    max_production = 100 # MW
    max_buy = 20 # MW
    max_sell = 20 # MW
    prod_price = 1 # € / MW
    prod_inertia = 0.2 # € / dMW/dt
    buy_price = 1.1 # € / MW
    sell_price = 0.9 # € / MW
    max_temp = 100
    max_flow = 100
    delay = 5 # time from production to consumer in time units

    min_forward_temp = 75
    max_forward_temp = 90

    acc_max_balance = 5
    acc_max_flow = 5

    G = get_structure(delay)

    divs = {"plant": get_numvars(solver, 0, max_production, T),
            "buy": get_numvars(solver, 0, max_buy, T),
            "sell": -get_numvars(solver, 0, max_sell, T),
            "consumer": -demand
            }
    flows = {("acc", "acc"): get_numvars(solver, 0, acc_max_balance, T),
             ("x0", "acc"): get_numvars(solver, 0, acc_max_flow, T),
             ("acc", "xc"): get_numvars(solver, 0, acc_max_flow, T)
            }
    default_div = lambda: [0]*T
    default_flow = lambda: get_numvars(solver, 0, max_flow, T)
    default_flow_val = lambda: [0]*T
    set_divs(G, divs, default_div)
    set_flows(solver, G, flows, default_flow)
    in_flows = bind_in_flows(solver, G, T)
    out_flows = bind_out_flows(solver, G, T)
    divergence_constraints(solver, G, in_flows, out_flows,
                           default_flow_val, burn_in=1)

    # custormer satisfaction constraint
    for x_t in G.nodes["xc"]["out_flow"]:
        solver.Add(x_t >= min_forward_temp)
        solver.Add(x_t <= max_forward_temp)

    cost = set_objective(solver, G, prod_price, prod_inertia, buy_price, sell_price)

    t1 = time.time()
    print("Build time: {0:.3f}".format(t1 - t0))

    # solving
    solver.Solve(time_limit=10)

    # presenting
    Plant = G.nodes["plant"]["div"]
    Buy = G.nodes["buy"]["div"]
    Sell = -G.nodes["sell"]["div"]
    Prod = G.edges["production", "x0"]["flow"]
    acc_in_flow = G.edges["x0", "acc"]["flow"]
    acc_out_flow = G.edges["acc", "xc"]["flow"]
    acc_balance = G.edges["acc", "acc"]["flow"]
    Plant_solved = [solver.solution_value(p) for p in Plant]
    Prod_solved = [solver.solution_value(p) for p in Prod]
    T_solved = [solver.solution_value(x) for x in G.nodes["xc"]["out_flow"]]
    acc_in = [solver.solution_value(a) for a in acc_in_flow]
    acc_out = [solver.solution_value(a) for a in acc_out_flow]
    acc_balance = [solver.solution_value(a) for a in acc_balance]
    #manual_balance = np.cumsum(np.array(acc_in[:-4]) - np.array(acc_out[4:]))
    #print(manual_balance)
    #T_pipe_solved = [[solver.solution_value(xt) for xt in x] for x in X]
    #T_pipe_solved = np.array(T_pipe_solved)
    buy_solved = [solver.solution_value(b) for b in Buy]
    sell_solved = [solver.solution_value(s) for s in Sell]

    cost_solved = solver.solution_value(cost)
    print("Total cost: {0:.1f}".format(cost_solved))
    print("Sum of demand: {0:.3f}".format(sum(demand)))
    print("Sum of production: {0:.3f}".format(sum(Prod_solved)))

    x = list(range(48, T-24))
    fig, (ax_power, ax_acc, ax_market) = plt.subplots(nrows=3, sharex=True)
    #fig, ax_power = plt.subplots(nrows=1, sharex=True)
    ax_power.step(x, Plant_solved[48:-24], color='b')
    ax_power.step(x, Prod_solved[48:-24], color='b', linestyle="--")
    ax_power.step(x, demand[48:-24], color='r')
    ax_power.step(x, T_solved[48:-24], color='g')
    #ax_power.step(x, Tx0_solved)    
    ax_acc.step(x, acc_in[48:-24])
    ax_acc.step(x, acc_balance[48:-24], linewidth=3)
    ax_acc.step(x, acc_out[48:-24])
    #ax_acc.step(x[4:], manual_balance, color="r")
    ax_market.step(x, sell_solved[48:-24])
    ax_market.step(x, buy_solved[48:-24])

    ax_power.legend(["Production", "Production + Market", "Demand", "Forward temperature"], loc=1)
    #ax_power.set_xlabel("Time / h")
    ax_power.set_ylabel("Power MW / Temp C")
    ax_power.set_title("Total cost: {0:.1f}".format(cost_solved))

    ax_acc.legend(["Accumulator in", "Acc balance", "Acc out"], loc=1)
    ax_acc.set_ylabel("Power MW")

    ax_market.legend(["Sold power", "Bought power"], loc=1)
    ax_market.set_ylabel("Power MW")

    #_, ax_img = plt.subplots()
    #ax_img.imshow(T_pipe_solved, aspect="auto", cmap="coolwarm")
    #ax_img.set_xlabel("Time / h")
    #ax_img.set_ylabel("Piece of line")
    #ax_img.set_title("Heat propagation from plant (top),\n\
    #                  to customer (middle), and back (bottom)")

    plt.show()


if __name__ == '__main__':
    plan()