# District heating network graph
from copy import deepcopy

from math import ceil
import numpy as np
import networkx as nx

from mip_utils import L1_energy, get_numvars, inverse_cumulative_element

class DHNGraph(nx.DiGraph):
    def __init__(self, structure, policy):
        super(DHNGraph, self).__init__(structure)

        self.max_production = policy["max_production"]
        self.max_buy = policy["max_buy"]
        self.max_sell = policy["max_sell"]
        self.prod_price = policy["prod_price"]
        self.prod_inertia = policy["prod_inertia"]
        self.buy_price = policy["buy_price"]
        self.sell_price = policy["sell_price"]
        self.max_temp = policy["max_temp"]
        self.max_flow = policy["max_flow"]
        self.max_speed = policy["max_speed"]
        self.min_speed = policy["min_speed"]
        self.min_forward_temp = policy["min_forward_temp"]
        self.max_forward_temp = policy["max_forward_temp"]
        self.acc_max_flow = policy["acc_max_flow"]
        self.acc_max_balance = policy["acc_max_balance"]
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

    def set_divs(self, solver, divs, demand):
        for x_name, data in self.nodes(data=True):
            if x_name in divs:
                div_conf = divs[x_name]
                if "data" in div_conf:
                    div = demand[div_conf["data"]]
                else:
                    div = get_numvars(solver, 
                                      div_conf["min"], div_conf["max"], self.T)
            else:
                div = self.default_div(self.T)
            data["div"] = div

    def set_flows(self, solver, flows):
        for x_name, y_name, data in self.edges(data=True):
            if (x_name, y_name) in flows:
                flow_conf = flows[(x_name, y_name)]
                flow = get_numvars(solver, 
                                   flow_conf["min"], flow_conf["max"], self.T)
            else:
                flow = self.default_flow(solver, self.T)
            data["flow"] = flow

    def set_speeds(self, solver):
        self.speeds = get_numvars(solver, 
                                  self.min_speed, self.max_speed, self.T)
        #max_delay = max(self.edges(data="delay"))
        #self.max_horizon = ceil(max_delay / self.min_speed)
        #self.max_total_flow = max_horizon * self.max_speed
        #self.max_temp_value = self.max_flow

    def merge(self, legacy=None):
        if legacy is None:
            return

        for x_name, data in self.nodes(data=True):
            legacy_data = [p.nodes[x_name]["div"] for p in legacy]
            data["div"] = np.concatenate(legacy_data + [data["div"]])

        for x_name, y_name, data in self.edges(data=True):
            legacy_data = [p.edges[x_name, y_name]["flow"] for p in legacy]
            data["flow"] = np.concatenate(legacy_data + [data["flow"]])

        self.set_T_from()

    def bind_out_flows(self, solver):
        """Structural constraints for out flows
        actives are free variables, bound only by resource constraint
        passives split remaining flow equally
        """
        out_flows = {}
        for x_name in self:
            out_edges = self.out_edges(nbunch=x_name, data=True)
            try:
                _, _, out_edges_data = zip(*out_edges)
            except ValueError:
                # there are no out edges, skip constraints
                continue
            active = [data["flow"] for data in out_edges_data if data["active"]]
            passive = [data["flow"] for data in out_edges_data if not data["active"]]
            if len(active):
                pulled = [solver.Sum(pull_t) for pull_t in zip(*active)]
            else:
                pulled = [0]*self.T
            remainder = get_numvars(solver, 0, self.max_flow, self.T)
            n_passive = len(passive)
            for passive_t, rem_t in zip(zip(*passive), remainder):
                for pass_t in passive_t:
                    solver.Add(pass_t == rem_t / n_passive)
            out_flows[x_name] = [rem_t + pull_t 
                                 for rem_t, pull_t in zip(remainder, pulled)]

        return out_flows

    def bind_in_flows(self, solver):
        """Structural constraints for in flows
        """
        in_flows = {}
        for x_name in self:
            in_edges = self.in_edges(nbunch=x_name, data=True)
            try:
                _, _, in_edges_data = zip(*in_edges)
            except ValueError:
                # there are no in edges, skip constraints
                continue
            in_flow = []
            flows = [data["flow"] for data in in_edges_data]
            delays = [data["delay"] for data in in_edges_data]
            h_losses = [data["heat_loss"] for data in in_edges_data]
            for t in range(self.T):
                # add speed modulation here
                #in_flows_t = [flow[t - d] * (1 - h_loss)
                #                 if t - d >= 0]

                in_flows_t = []
                for idx, (flow, d, h_loss) in enumerate(zip(flows, delays, h_losses)):
                    if d == 0:
                        in_flows_t.append(flow * (1 - h_loss))
                    else:
                        horizon = ceil(d / self.min_speed)
                        if t < horizon:
                            continue
                        temps = flows[idx][t-horizon:t]
                        speeds = self.speeds[t-horizon:t]
                        max_total_flow = horizon * self.max_speed
                        max_temp_value = self.max_flow
                        f = inverse_cumulative_element
                        flow_t, _ = f(solver, temps, speeds,
                                      d, max_total_flow,
                                      max_temp_value)
                        in_flows_t.append(flow_t * (1 - h_loss))
                        # I just remembered the heat loss depends on 
                        # transfer time, in a nonlinear way. 
                        # Well, that'll be a pain for another time. 
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
        in_flows = self.bind_in_flows(solver)
        out_flows = self.bind_out_flows(solver)
        for x_name in self:
            in_flow = in_flows.get(x_name, self.default_flow_val(self.T))
            out_flow = out_flows.get(x_name, self.default_flow_val(self.T))
            div = self.nodes[x_name]["div"]
            for t, (in_t, out_t, div_t) in enumerate(zip(in_flow, out_flow, div)):
                if t < burn_in:
                    continue
                solver.Add(out_t - in_t == div_t)

    def forward_temp_constraint(self, solver, consumers, burn_in):
        """Takes in a container (set, list, etc) of customer nodes, 
        constrains the total out flow 
        of heat from this node to be within an interval
        """
        _, _, out_flows = zip(*self.edges(nbunch=consumers, data="flow"))
        out_flow = sum(out_flows)
        for t, x_t in enumerate(out_flow):
            if t < burn_in:
                continue
            solver.Add(x_t >= self.min_forward_temp)
            solver.Add(x_t <= self.max_forward_temp)

    def get_objective(self, solver, plants, buyers, sellers):
        """Assuming just 1 plant, 1 buy entry, and 1 sell exit
        """
        Plants = [self.nodes[plant]["div"] for plant in plants]
        Buyers = [self.nodes[buy]["div"] for buy in buyers]
        Sellers = [-self.nodes[sell]["div"] for sell in sellers]
        prod_base_cost = solver.Sum([solver.Sum(Plant) for Plant in Plants])
        prod_inertia_cost = solver.Sum([L1_energy(solver, Plant) for Plant in Plants])
        cost =   self.prod_price*prod_base_cost \
               + self.prod_inertia*prod_inertia_cost \
               + self.buy_price * solver.Sum([solver.Sum(Buy) for Buy in Buyers]) \
               - self.sell_price * solver.Sum([solver.Sum(Sell) for Sell in Sellers])
        return cost

    def extract_interval(self, solver, t_start=None, t_end=None):
        """
        Return a graph with solution values, over a time interval
        """
        G = nx.DiGraph(self)
        if t_start is None:
            t_start = 0
        if t_end is None:
            t_end = self.T

        for x_name, data in G.nodes(data=True):
            data["div"] = np.array([solver.solution_value(d) 
                                    for d in data["div"][t_start:t_end]])

        for x_name, y_name, data in G.edges(data=True):
            data["flow"] = np.array([solver.solution_value(f) 
                                    for f in data["flow"][t_start:t_end]])

        return G
