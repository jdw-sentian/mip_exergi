import networkx as nx

def get_structure(structure="structure_1"):
    """In the future, these definitions should come from files 
    instead of functions.
    """
    if structure == "structure_1":
        return _get_structure_1()
    elif structure == "structure_test":
        return _get_structure_test()

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

def _get_structure_test():
    G = nx.DiGraph()

    return G

def get_parameters(parameters="parameters_1"):
    """In the future, these definitions should come from files 
    instead of functions.
    """
    if parameters == "parameters_1":
        return _get_parameters_1()
    elif parameters == "parameters_test":
        return _get_parameters_test()

def _get_parameters_1():
    params = {}
    params["max_production"] = 100  # MW
    params["max_buy"] = 20  # MW
    params["max_sell"] = 20  # MW
    params["prod_price"] = 1  # € / MW
    params["prod_inertia"] = 0.2  # € / dMW/dt
    params["buy_price"] = 1.1  # € / MW
    params["sell_price"] = 0.9  # € / MW
    params["max_temp"] = 100
    params["max_flow"] = 100

    params["min_forward_temp"] = 75
    params["max_forward_temp"] = 90

    params["acc_max_balance"] = 5
    params["acc_max_flow"] = 5

    return params

def _get_parameters_test():
    params = {}

    return params
