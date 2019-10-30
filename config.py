import networkx as nx

def get_structure(structure="structure_1"):
    """In the future, these definitions should come from files 
    instead of functions.
    """
    if structure == "structure_1":
        return _get_structure_1()
    elif structure == "structure_debug":
        return _get_structure_debug()

def _get_structure_1():
    delay = 5  # time from production to consumer in time units

    G = nx.DiGraph()
    G.add_node("plant")
    G.add_node("buy")
    G.add_node("sell")
    G.add_node("production")
    G.add_node("consumer")
    G.add_node("x0")
    G.add_node("xc")

    G.add_edge("plant", "production", 
            **{"delay": 0, "heat_loss": 0, "active": False})
    G.add_edge("buy", "production", 
            **{"delay": 0, "heat_loss": 0, "active": False})
    G.add_edge("production", "sell", 
            **{"delay": 0, "heat_loss": 0, "active": True})
    G.add_edge("production", "x0", 
            **{"delay": 0, "heat_loss": 0, "active": False})
    G.add_edge("xc", "consumer", 
            **{"delay": 0, "heat_loss": 0, "active": True})
    G.add_edge("x0", "xc", 
            **{"delay": delay, "heat_loss": 0.02, "active": False})
    G.add_edge("xc", "x0", 
            **{"delay": delay, "heat_loss": 0.02, "active": False})
    
    # accumulator
    G.add_node("acc")
    G.add_edge("acc", "acc", 
            **{"delay": 1, "heat_loss": 0, "active": False})
    G.add_edge("x0", "acc", 
            **{"delay": 0, "heat_loss": 0, "active": True})
    G.add_edge("acc", "xc", 
            **{"delay": delay, "heat_loss": 0.02, "active": True})

    return G

def _get_structure_debug():
    delay = 5  # time from production to consumer in time units

    G = nx.DiGraph()
    G.add_node("plant")
    G.add_node("consumer")
    G.add_node("x0")
    G.add_node("xc")

    G.add_edge("plant", "x0", 
            **{"delay": 0, "heat_loss": 0, "active": False})
    G.add_edge("xc", "consumer", 
            **{"delay": 0, "heat_loss": 0, "active": True})
    G.add_edge("x0", "xc", 
            **{"delay": delay, "heat_loss": 0, "active": False})
    G.add_edge("xc", "x0", 
            **{"delay": delay, "heat_loss": 0, "active": False})
    
    return G

def get_policy(policy="policy_1"):
    """In the future, these definitions should come from files 
    instead of functions.
    """
    if policy == "policy_1":
        return _get_policy_1()
    elif policy == "policy_debug":
        return _get_policy_debug()

def _get_policy_1():
    policy = {}
    policy["max_production"] = 100  # MW
    policy["max_buy"] = 20  # MW
    policy["max_sell"] = 20  # MW
    policy["prod_price"] = 1  # € / MW
    policy["prod_inertia"] = 0.2  # € / dMW/dt
    policy["buy_price"] = 1.1  # € / MW
    policy["sell_price"] = 0.9  # € / MW
    policy["max_temp"] = 100
    policy["max_flow"] = 100

    policy["max_speed"] = 2
    policy["min_speed"] = 1

    policy["min_forward_temp"] = 75
    policy["max_forward_temp"] = 90

    policy["acc_max_balance"] = 5
    policy["acc_max_flow"] = 5

    policy["divs"] = {"plant": {"min": 0, "max": policy["max_production"]},
                      "buy": {"min": 0, "max": policy["max_buy"]},
                      "sell": {"min": 0, "max": policy["max_sell"]},
                      "consumer": {"data": "demand_0"},
                      }

    policy["flows"] = {("acc", "acc"): {"min": 0, "max": policy["acc_max_balance"]},
                       ("x0", "acc"): {"min": 0, "max": policy["acc_max_flow"]},
                       ("acc", "xc"): {"min": 0, "max": policy["acc_max_flow"]},
                       }

    policy["consumers"] = ["xc"]
    policy["plants"] = ["plant"]
    policy["buyers"] = ["buy"]
    policy["sellers"] = ["sell"]

    return policy


def _get_policy_debug():
    policy = {}
    policy["max_production"] = 100  # MW
    policy["max_buy"] = 0  # MW
    policy["max_sell"] = 0  # MW
    policy["prod_price"] = 0  # € / MW
    policy["prod_inertia"] = 1  # € / dMW/dt
    policy["buy_price"] = 0  # € / MW
    policy["sell_price"] = 0  # € / MW
    policy["max_temp"] = 100
    policy["max_flow"] = 100

    policy["max_speed"] = 2
    policy["min_speed"] = 1

    policy["min_forward_temp"] = 80
    policy["max_forward_temp"] = 80

    policy["acc_max_balance"] = 5
    policy["acc_max_flow"] = 5

    policy["divs"] = {"plant": {"min": 0, "max": policy["max_production"]},
                      "consumer": {"data": "demand_0"},
                      }

    policy["flows"] = {}

    policy["consumers"] = ["xc"]
    policy["plants"] = ["plant"]
    policy["buyers"] = []
    policy["sellers"] = []

    return policy