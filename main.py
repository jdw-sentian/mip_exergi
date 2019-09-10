import time
import numpy as np
import matplotlib.pyplot as plt

import networkx as nx

from sentian_miami import get_solver

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


def set_divs(G, divs, default_div):
    #T = get_T(G)
    for x_name, data in G.nodes(data=True):
        data["div"] = divs.get(x_name, default_div())


def set_flows(solver, G, flows, default_flow):
    #T = get_T(G)
    for x_name, y_name, data in G.edges(data=True):
        data["flow"] = flows.get((x_name, y_name), default_flow())


def merge(G, pre=None, post=None):
    if pre is None:
        pre = []
    if post is None:
        post = []

    for x_name, data in G.nodes(data=True):
        pre_data = [p.nodes[x_name]["div"] for p in pre]
        post_data = [p.nodes[x_name]["div"] for p in post]
        data["div"] = np.concatenate(pre_data + [data["div"]] + post_data)
        #pre_data = [p.nodes[x_name]["out_flow"] for p in pre]
        #post_data = [p.nodes[x_name]["out_flow"] for p in post]
        #data["out_flow"] = np.concatenate(pre_data + [data["out_flow"]] + post_data)

    for x_name, y_name, data in G.edges(data=True):
        pre_data = [p.edges[x_name, y_name]["flow"] for p in pre]
        post_data = [p.edges[x_name, y_name]["flow"] for p in post]
        data["flow"] = np.concatenate(pre_data + [data["flow"]] + post_data)


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


def get_T(G):
    """Assumes that all data have the same length in G
    """
    #for _, data in G.nodes(data=True):
    #    return len(data["div"])
    return len(G.nodes["consumer"]["div"])


def divergence_constraints(solver, G, in_flows, out_flows, default_flow_val, burn_in=0):
    """Constrain so that 
    out_flow - in_flow = divergence
    """
    #T = get_T(G)
    for x_name in G:
        in_flow = in_flows.get(x_name, default_flow_val())
        out_flow = out_flows.get(x_name, default_flow_val())
        div = G.nodes[x_name]["div"]
        for t, (in_t, out_t, div_t) in enumerate(zip(in_flow, out_flow, div)):
            if t < burn_in:
                continue
            solver.Add(out_t - in_t == div_t)

    # remember the out_flow, for constraints
    #for x_name in G:
    #    out_flow = out_flows.get(x_name, default_flow_val())
    #    G.nodes[x_name]["out_flow"] = out_flow


def forward_temp_constraint(solver, G, min_forward_temp, max_forward_temp, burn_in):
    # customer satisfaction constraint
    out_flow = G.edges["xc", "x0"]["flow"] + G.edges["xc", "consumer"]["flow"]
    for t, x_t in enumerate(out_flow):
        if t < burn_in:
            continue
        solver.Add(x_t >= min_forward_temp)
        solver.Add(x_t <= max_forward_temp)


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


def extract_interval(solver, G, t_start=None, t_end=None):
    G = G.copy()
    if t_start is None:
        t_start = 0
    if t_end is None:
        for _, data in G.nodes(data=True):
            t_end = len(data["div"])
            break

    for x_name, data in G.nodes(data=True):
        data["div"] = np.array([solver.solution_value(d) for d in data["div"][t_start:t_end]])
        #data["out_flow"] = np.array([solver.solution_value(d) for d in data["out_flow"][t_start:t_end]])

    for x_name, y_name, data in G.edges(data=True):
        data["flow"] = np.array([solver.solution_value(f) for f in data["flow"][t_start:t_end]])

    return G


def plan(demand,
         max_production, max_buy, max_sell,
         prod_price, prod_inertia, buy_price, sell_price,
         max_temp, max_flow, delay,
         min_forward_temp, max_forward_temp,
         acc_max_flow, acc_max_balance,
         t_start=None, t_end=None,
         custom_divs=None, custom_flows=None,
         pre_legacy=None, post_legacy=None, solver=None,
         add_constraints=True):
    #t0 = time.time()
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
    divs = {"plant": get_numvars(solver, 0, max_production, T),
            "buy": get_numvars(solver, 0, max_buy, T),
            "sell": -get_numvars(solver, 0, max_sell, T),
            "consumer": -demand
            }
    divs.update(custom_divs)
    flows = {("acc", "acc"): get_numvars(solver, 0, acc_max_balance, T),
             ("x0", "acc"): get_numvars(solver, 0, acc_max_flow, T),
             ("acc", "xc"): get_numvars(solver, 0, acc_max_flow, T)
            }
    flows.update(custom_flows)
    if pre_legacy is None:
        T_full = T
    else:
        T_full = T #sum([get_T(G_l) for G_l in pre_legacy])
    default_div = lambda: np.array([0]*T_full)
    default_flow = lambda: get_numvars(solver, 0, max_flow, T_full)
    default_flow_val = lambda: np.array([0]*T_full)
    set_divs(G, divs, default_div)
    set_flows(solver, G, flows, default_flow)
    if (pre_legacy is not None) or (post_legacy is not None):
        merge(G, pre=pre_legacy, post=post_legacy)
    #print("Post merge T:", get_T(G))
    T_full = get_T(G)
    in_flows = bind_in_flows(solver, G, T_full)
    out_flows = bind_out_flows(solver, G, T_full)
    if not pre_legacy:
        burn_in = 1
    else:
        for G_legacy in pre_legacy:
            burn_in = 0
            for x_name in G_legacy:
                burn_in += len(G_legacy.nodes[x_name]["div"])
                break
    if add_constraints:
        divergence_constraints(solver, G, in_flows, out_flows,
                               default_flow_val, burn_in=burn_in)
        forward_temp_constraint(solver, G, min_forward_temp, max_forward_temp, burn_in)

    cost = set_objective(solver, G, prod_price, prod_inertia, buy_price, sell_price)
    if solve_all:
        solver.Solve(time_limit=10)    
        return extract_interval(solver, G, t_start, t_end)
    else:
        return G, cost

    #t1 = time.time()
    #print("Build time: {0:.3f}".format(t1 - t0))


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
    params["delay"] = 5 # time from production to consumer in time units

    params["min_forward_temp"] = 75
    params["max_forward_temp"] = 90

    params["acc_max_balance"] = 5
    params["acc_max_flow"] = 5

    G_legacy = plan(demand, t_start=24, t_end=72, **params)
    #_, axes_legacy = plt.subplots(nrows=3, sharex=True)
    #present(axes_legacy, G_legacy)
    
    solver = get_solver("CBC")
    costs = []
    custom_divs = {"plant": get_numvars(solver, 0, params["max_production"], 1),
                   "buy": get_numvars(solver, 0, params["max_buy"], 1),
                   "sell": -get_numvars(solver, 0, params["max_sell"], 1)
                   }
    for _ in range(10):
        demand = get_demand_forecast(num_days=2)
        dem0, dem_rest = demand[:1], demand[1:]
        G0, _ = plan(solver=solver, demand=dem0, pre_legacy=[G_legacy], 
                     custom_divs=custom_divs, add_constraints=True, **params)
        G, cost = plan(solver=solver, demand=dem_rest, pre_legacy=[G0], **params)
        costs.append(cost)
    solver.SetObjective(solver.Sum(costs), maximize=False)

    #G = plan(demand, t_start=0, t_end=48, pre_legacy=[G_legacy], **params)
    solver.Solve(time_limit=10)

    G_solved = extract_interval(solver, G)

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
    #print("T:", len(plant))
    #print(plant)
    #fig, ax_power = plt.subplots(nrows=1, sharex=True)
    where = "post"
    ax_power.step(x, plant, color='b', where=where)
    ax_power.step(x, prod, color='b', linestyle="--", where=where)
    ax_power.step(x, demand, color='r', where=where)
    ax_power.step(x, customer_temp, color='g', where=where)
    #ax_power.step(x, Tx0_solved)    
    ax_acc.step(x, acc_in, where=where)
    ax_acc.step(x, acc_balance, linewidth=3, where=where)
    ax_acc.step(x, acc_out, where=where)
    #ax_acc.step(x[4:], manual_balance, color="r")
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

    '''
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

    # customer satisfaction constraint
    for x_t in G.nodes["xc"]["out_flow"]:
        solver.Add(x_t >= min_forward_temp)
        solver.Add(x_t <= max_forward_temp)

    cost = set_objective(solver, G, prod_price, prod_inertia, buy_price, sell_price)

    t1 = time.time()
    print("Build time: {0:.3f}".format(t1 - t0))

    # solving
    solver.Solve(time_limit=10)
    '''

    '''
    Plant_solved = [solver.solution_value(p) for p in Plant]
    Prod_solved = [solver.solution_value(p) for p in Prod]
    acc_in = [solver.solution_value(a) for a in acc_in_flow]
    acc_out = [solver.solution_value(a) for a in acc_out_flow]
    acc_balance = [solver.solution_value(a) for a in acc_balance]
    '''
    #manual_balance = np.cumsum(np.array(acc_in[:-4]) - np.array(acc_out[4:]))
    #print(manual_balance)
    #T_pipe_solved = [[solver.solution_value(xt) for xt in x] for x in X]
    #T_pipe_solved = np.array(T_pipe_solved)
    #buy_solved = [solver.solution_value(b) for b in Buy]
    #sell_solved = [solver.solution_value(s) for s in Sell]

    '''
    cost_solved = solver.solution_value(cost)
    print("Total cost: {0:.1f}".format(cost_solved))
    print("Sum of demand: {0:.3f}".format(sum(demand)))
    print("Sum of production: {0:.3f}".format(sum(Prod_solved)))
    '''
