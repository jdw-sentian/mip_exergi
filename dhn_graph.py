# District heating network graph
from copy import deepcopy

import numpy as np
import networkx as nx

class DHNGraph(nx.DiGraph):
    def __init__(self, G, params):
        super(DHNGraph, self).__init__(G)

        self.G = G
        self.max_production = params["max_production"]
        self.max_buy = params["max_buy"] 
        self.max_sell = params["max_sell"]
        self.prod_price = params["prod_price"] 
        self.prod_inertia = params["prod_inertia"] 
        self.buy_price = params["buy_price"] 
        self.sell_price = params["sell_price"]
        self.max_temp = params["max_temp"] 
        self.max_flow = params["max_flow"] 
        self.min_forward_temp = params["min_forward_temp"] 
        self.max_forward_temp = params["max_forward_temp"]
        self.acc_max_flow = params["acc_max_flow"] 
        self.acc_max_balance = params["acc_max_balance"]
        self.cold_init = 80 # inital value on cold start

    def set_T_from(self, divs=None, flows=None):
        """Looks for data in self and outside to find T
        """

        if divs is None:
            divs = dict()
        if flows is None:
            flows = dict()

        for div in divs.values():
            self.T = len(div)
            return

        for flow in flows.values():
            self.T = len(flow)
            return

        for _, div in self.nodes(data="div"):
            self.T = len(div)
            return

        for _, _, flow in self.edges(data="flow"):
            self.T = len(flow)
            return

    def default_div(self, T):
        return np.array([0]*T)

    def default_flow(self, solver, T):
        return np.array([solver.NumVar(0, self.max_flow) for _ in range(T)])

    def default_flow_val(self, T):
        return np.array([0]*T)

    def set_divs(self, divs):
        for x_name, data in self.nodes(data=True):
            data["div"] = divs.get(x_name, self.default_div(self.T))

    def set_flows(self, solver, flows):
        for x_name, y_name, data in self.edges(data=True):
            data["flow"] = flows.get((x_name, y_name), self.default_flow(solver, self.T))    

    def merge(self, legacy=None):
        if legacy is None:
            legacy = []

        for x_name, data in self.nodes(data=True):
            legacy_data = [p.nodes[x_name]["div"] for p in legacy]
            data["div"] = np.concatenate(legacy_data + [data["div"]])

        for x_name, y_name, data in self.edges(data=True):
            legacy_data = [p.edges[x_name, y_name]["flow"] for p in legacy]
            data["flow"] = np.concatenate(legacy_data + [data["flow"]])

        self.set_T_from()

    def bind_out_flows(self, solver, T):
        """Constraints for out flows
        actives are free variables, bound only by resource constraint
        passives split remaining flow equally
        """
        out_flows = {}
        for x_name in self:
            out_edges = self.out_edges(nbunch=x_name, data=True)
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
            remainder = [solver.NumVar(0, 100) for _ in range(T)]
            n_passive = len(passive)
            for passive_t, rem_t in zip(zip(*passive), remainder):
                for pass_t in passive_t:
                    solver.Add(pass_t == rem_t / n_passive)
            out_flows[x_name] = [rem_t + pull_t for rem_t, pull_t in zip(remainder, pulled)]

        return out_flows

    def bind_in_flows(self, solver, T):
        """Constraints for in flows
        """
        in_flows = {}
        for x_name in self:
            in_edges = self.in_edges(nbunch=x_name, data=True)
            try:
                _, _, in_edges_data = zip(*in_edges)
            except ValueError:
                # if there are no in edges, skip constraints
                continue
            in_flow = []
            flows = [data["flow"] for data in in_edges_data]
            delays = [data["delay"] for data in in_edges_data]
            for t in range(T):
                # first version: no heat loss
                in_flows_t = [flow[t - d] for flow, d in zip(flows, delays) if t - d >= 0]
                if len(in_flows_t):
                    in_flow_t = solver.Sum(in_flows_t)
                else:
                    in_flow_t = self.cold_init #only relevant for cold starts
                in_flow.append(in_flow_t)
            in_flows[x_name] = in_flow

        return in_flows

    def divergence_constraints(self, solver, burn_in):
        """Constrain so that 
        out_flow - in_flow = divergence
        """
        in_flows = self.bind_in_flows(solver, self.T)
        out_flows = self.bind_out_flows(solver, self.T)
        for x_name in self:
            in_flow = in_flows.get(x_name, self.default_flow_val(self.T))
            out_flow = out_flows.get(x_name, self.default_flow_val(self.T))
            div = self.nodes[x_name]["div"]
            for t, (in_t, out_t, div_t) in enumerate(zip(in_flow, out_flow, div)):
                if t < burn_in:
                    continue
                solver.Add(out_t - in_t == div_t)

    def forward_temp_constraint(self, solver, xc, burn_in):
        _, _, out_flows = zip(*self.edges(nbunch=xc, data="flow"))
        out_flow = sum(out_flows)
        for t, x_t in enumerate(out_flow):
            if t < burn_in:
                continue
            solver.Add(x_t >= self.min_forward_temp)
            solver.Add(x_t <= self.max_forward_temp)

    def get_objective(self, solver, plant, buy, sell):
        """Assuming just 1 plant, 1 buy entry, and 1 sell exit
        """
        Plant = self.nodes[plant]["div"]
        Buy = self.nodes[buy]["div"]
        Sell = -self.nodes[sell]["div"]
        Prod = Plant + Buy - Sell
        prod_base_cost = solver.Sum(Plant)
        prod_inertia_cost = production_inertia_cost(solver, Plant)
        prod_cost = self.prod_price*prod_base_cost + self.prod_inertia*prod_inertia_cost
        cost = prod_cost + self.buy_price * solver.Sum(Buy) - self.sell_price * solver.Sum(Sell)
        return cost

    def extract_interval(self, solver, t_start=None, t_end=None):
        G = nx.DiGraph(self)
        if t_start is None:
            t_start = 0
        if t_end is None:
            t_end = self.T

        for x_name, data in G.nodes(data=True):
            data["div"] = np.array([solver.solution_value(d) for d in data["div"][t_start:t_end]])

        for x_name, y_name, data in G.edges(data=True):
            data["flow"] = np.array([solver.solution_value(f) for f in data["flow"][t_start:t_end]])

        #self.set_T_from()

        return G

# the below are abstract, and could be moved to a utils module

def abs_diff_var(solver, val):
    x = solver.NumVar(0)
    solver.Add(x >= val)
    solver.Add(x >= -val)
    return x

def production_inertia_cost(solver, P):
    change_vars = [abs_diff_var(solver, p0 - p1) for p0, p1 in zip(P[:-1], P[1:])]
    return solver.Sum(change_vars)