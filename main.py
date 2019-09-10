import time
import numpy as np
import matplotlib.pyplot as plt

import networkx as nx

from sentian_miami import get_solver

from dhn_graph import DHNGraph

def get_demand_forecast(num_days=1):
    # unit in hypothetical MW
    demand = np.array([30, 30, 30, 30, 30, 30,
                     40, 50, 50, 50, 40, 30,
                     30, 30, 40, 40, 50, 50,
                     60, 60, 60, 50, 40, 30]*num_days) * 1.0
    demand += np.random.normal(size=len(demand)) * 2
    return demand


def get_numvars(solver, lb, ub, N):
    return np.array([solver.NumVar(lb, ub) for _ in range(N)])


def get_T(G):
    for _, div in G.nodes(data="div"):
        return len(div)

    for _, _, flow in G.edges(data="flow"):
        return len(flow)

    return None


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


def plan(demand, delay, params,
         t_start=None, t_end=None, burn_in=None,
         custom_divs=None, custom_flows=None,
         legacy=None, solver=None):
    if custom_divs is None:
        custom_divs = {}
    if custom_flows is None:
        custom_flows = {}
    if solver is None:
        solver = get_solver("CBC")
        solve_all = True
    else:
        solve_all = False
    T = len(demand)
    G = get_structure(delay)
    G = DHNGraph(G, params)
    divs = {"plant": get_numvars(solver, 0, G.max_production, T),
            "buy": get_numvars(solver, 0, G.max_buy, T),
            "sell": -get_numvars(solver, 0, G.max_sell, T),
            "consumer": -demand
            }
    divs.update(custom_divs)
    flows = {("acc", "acc"): get_numvars(solver, 0, G.acc_max_balance, T),
             ("x0", "acc"): get_numvars(solver, 0, G.acc_max_flow, T),
             ("acc", "xc"): get_numvars(solver, 0, G.acc_max_flow, T)
            }
    flows.update(custom_flows)
    G.set_T_from(divs, flows)
    G.set_divs(divs)
    G.set_flows(solver, flows)
    G.merge(legacy)
    if burn_in is None:
        if not legacy:
            burn_in = 1
        else:
            burn_in = sum([G_l.T for G_l in legacy])
    G.divergence_constraints(solver, burn_in)
    G.forward_temp_constraint(solver, "xc", burn_in)
    cost = G.get_objective(solver, "plant", "buy", "sell")
    if solve_all:
        solver.SetObjective(cost, maximize=False)
        solver.Solve(time_limit=10)    
        return G.extract_interval(solver, t_start, t_end)
    else:
        return G, cost


def main():
    np.random.seed(0)
    demand = get_demand_forecast(num_days=10)

    # parameters
    params = {}
    params["max_production"] = 100 # MW
    params["max_buy"] = 20 # MW
    params["max_sell"] = 20 # MW
    params["prod_price"] = 1 # € / MW
    params["prod_inertia"] = 0.2 # € / dMW/dt
    params["buy_price"] = 1.1 # € / MW
    params["sell_price"] = 0.9 # € / MW
    params["max_temp"] = 100
    params["max_flow"] = 100
    delay = 5 # time from production to consumer in time units

    params["min_forward_temp"] = 75
    params["max_forward_temp"] = 90

    params["acc_max_balance"] = 5
    params["acc_max_flow"] = 5

    G_legacy = plan(demand, delay, params, t_start=24, t_end=72)
    #print("G.T:", get_T(G_legacy))
    _, axes_legacy = plt.subplots(nrows=3, sharex=True)
    present(axes_legacy, G_legacy)
    
    t0 = time.time()
    solver = get_solver("CBC")
    costs = []
    custom_divs = {"plant": get_numvars(solver, 0, params["max_production"], 1),
                   "buy": get_numvars(solver, 0, params["max_buy"], 1),
                   "sell": -get_numvars(solver, 0, params["max_sell"], 1)
                   }
    for _ in range(1):
        demand = get_demand_forecast(num_days=2)
        dem0, dem_rest = demand[:1], demand[1:]
        G0, _ = plan(solver=solver, demand=dem0, delay=delay, legacy=[G_legacy], 
                     custom_divs=custom_divs, burn_in=get_T(G_legacy), params=params)
        G, cost = plan(solver=solver, demand=dem_rest, delay=delay, legacy=[G0], 
                       burn_in=get_T(G_legacy), params=params)
        costs.append(cost)
    solver.SetObjective(solver.Sum(costs), maximize=False)
    t1 = time.time()
    print("Build time: {0:.3f}".format(t1 - t0))

    solver.Solve(time_limit=10)

    G_solved = G.extract_interval(solver)

    _, axes = plt.subplots(nrows=3, sharex=True)
    present(axes, G_solved)
    '''
    '''
    plt.show()

def present(axes, G):
    ax_power, ax_acc, ax_market = axes
    plant = G.nodes["plant"]["div"]
    buy = G.nodes["buy"]["div"]
    sell = -G.nodes["sell"]["div"]
    prod = G.edges["production", "x0"]["flow"]
    acc_in = G.edges["x0", "acc"]["flow"]
    acc_out = G.edges["acc", "xc"]["flow"]
    acc_balance = G.edges["acc", "acc"]["flow"]
    customer_temp = G.edges["xc", "x0"]["flow"] + G.edges["xc", "consumer"]["flow"]
    demand = -G.nodes["consumer"]["div"]

    x = list(range(len(plant)))
    where = "post"
    ax_power.step(x, plant, color='b', where=where)
    ax_power.step(x, prod, color='b', linestyle="--", where=where)
    ax_power.step(x, demand, color='r', where=where)
    ax_power.step(x, customer_temp, color='g', where=where)
    ax_acc.step(x, acc_in, where=where)
    ax_acc.step(x, acc_balance, linewidth=3, where=where)
    ax_acc.step(x, acc_out, where=where)
    ax_market.step(x, sell, where=where)
    ax_market.step(x, buy, linestyle="--", where=where)

    ax_power.legend(["Production", "Production + Market", "Demand", "Forward temperature"], loc=1)
    #ax_power.set_xlabel("Time / h")
    ax_power.set_ylabel("Power MW / Temp C")
    #ax_power.set_title("Total cost: {0:.1f}".format(cost_solved))

    ax_acc.legend(["Accumulator in", "Acc balance", "Acc out"], loc=1)
    ax_acc.set_ylabel("Power MW")

    ax_market.legend(["Sold power", "Bought power"], loc=1)
    ax_market.set_ylabel("Power MW")


if __name__ == '__main__':
    main()
