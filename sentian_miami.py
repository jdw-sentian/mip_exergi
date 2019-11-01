import time
from ortools.linear_solver import pywraplp
import pyomo.environ as pe

status2str = ["OPTIMAL", "FEASIBLE", "INFEASIBLE", "UNBOUNDED", "ABNORMAL", "MODEL_INVALID", "NOT_SOLVED"]

def get_solver(solver_version):
    if solver_version == "CBC":
        return SolverOrtools()
    elif solver_version.upper() == "COUENNE":
        return SolverPyomo()

def solution_value(variable):
    return variable.solution_value()

class SolverOrtools(pywraplp.Solver):
    def __init__(self):
        super().__init__("", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        self.inf = self.infinity()

    def IntVar(self, lb=None, ub=None, name=""):
        if lb is None:
            lb = self.inf
        if ub is None:
            ub = self.inf
        return super().IntVar(lb, ub, name)

    def NumVar(self, lb=None, ub=None, name=""):
        if lb is None:
            lb = self.inf
        if ub is None:
            ub = self.inf
        return super().NumVar(lb, ub, name)

    def Dot(self, X, Y):
        return self.Sum([x * y for x, y in zip(X, Y)])

    def SetObjective(self, expr, maximize):
        if maximize:
            self.Maximize(expr)
        else:
            self.Minimize(expr)

    def Solve(self, time_limit=0, print_output=True):
        t0 = time.time()
        self.set_time_limit(time_limit*1000)
        status = super().Solve()
        t1 = time.time()
        if print_output:
            print("Status:", status2str[status])
            print("Time: {0:.3f}".format(t1 -t0))

    def solution_value(self, variable):
        try:
            return variable.solution_value()
        except AttributeError:
            return variable

class SolverPyomo:
    """the pyomo syntax is so ugly
    gotta cover it with a brown paper bag
    """

    def __init__(self):
        self.model = pe.ConcreteModel()
        self.solver = pe.SolverFactory("couenne")
        self.top = 0

    def IntVar(self, lb=None, ub=None, name=None):
        return self.Var(lb, ub, name, within=pe.Integers)

    def NumVar(self, lb=None, ub=None, name=None):
        return self.Var(lb, ub, name, within=pe.Reals)

    def Var(self, lb=None, ub=None, name=None, **kwargs):
        var = pe.Var(bounds=(lb, ub), **kwargs)
        self.top += 1
        if name is None:
            name = "var_{}".format(self.top)
        setattr(self.model, name, var)
        return var

    def Sum(self, variables):
        return sum(variables)

    def Dot(self, X, Y):
        return sum([x*y for x, y in zip(X, Y)])

    def Add(self, expr):
        constraint = pe.Constraint(expr=expr)
        self.top += 1
        name = "constraint_{}".format(self.top)
        setattr(self.model, name, constraint)
        return constraint

    def Solve(self, time_limit=None, verbose=False):
        res = self.solver.solve(self.model)
        if verbose:
            res.write()

    def SetObjective(self, expr, maximize):
        if maximize:
            sense = pe.maximize
        else:
            sense = pe.minimize
        self.model.objective = pe.Objective(expr=expr, sense=sense)

    def solution_value(self, variable):
        try:
            return variable()
        except TypeError:
            return variable


def test_couenne_basic():
    solver = get_solver("couenne")

    a = solver.NumVar(lb=11, ub=20)
    b = 10
    #b = solver.NumVar(0, 2)
    c = a + b
    #d = 2*a + b
    solver.Add(a <= 20)
    solver.SetObjective(a, maximize=True)
    #solver.model.pprint()

    solver.Solve()
    #solver.model.pprint()

    print("a:", solver.solution_value(a))
    print("b:", solver.solution_value(b))

    c_val = solver.solution_value(c)
    #print(type(c_val))
    print("c:", c_val)


def test_couenne_nonlinear():

    solver = get_solver("couenne")

    a = solver.NumVar(0.2, 2)
    b = solver.NumVar(0.2, 2)

    s = a + b
    p = a * b

    solver.Add(solver.Sum([a, b, a, a]) <= 3)
    solver.SetObjective(p, maximize=True)

    solver.Solve()

    print("a:", solver.solution_value(a))
    print("b:", solver.solution_value(b))


if __name__ == '__main__':
    #test_couenne_basic()
    test_couenne_nonlinear()