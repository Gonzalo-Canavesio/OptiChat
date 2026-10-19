from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from pyomo.opt import SolverFactory

# label (lowercase) -> (Pyomo solver name, option that sets the time limit in seconds)
SOLVERS: dict[str, tuple[str, str]] = {
    "gurobi": ("gurobi", "TimeLimit"),
    "glpk": ("glpk", "tmlim"),
    "cbc": ("cbc", "seconds"),
    "highs": ("appsi_highs", "time_limit"),
    "scip": ("scip", "limits/time"),
    "cplex": ("cplex", "timelimit"),
}
DEFAULT_SOLVER = "gurobi"

# label (lowercase) -> GAMS solver name, for GAMSPy models (GAMS has no GLPK)
GAMSPY_SOLVERS: dict[str, str] = {
    "gurobi": "gurobi",
    "cplex": "cplex",
    "cbc": "cbc",
    "highs": "highs",
    "scip": "scip",
}

_active_solver: ContextVar[str] = ContextVar("optichat_solver", default=DEFAULT_SOLVER)


class SolverNotAvailableError(RuntimeError):
    pass


def normalize_solver(label: str | None) -> str:
    key = (label or DEFAULT_SOLVER).strip().lower()
    if key not in SOLVERS:
        raise ValueError(
            f"Unknown solver '{label}'. Available: {', '.join(sorted(SOLVERS))}."
        )
    return key


@contextmanager
def use_solver(label: str | None) -> Iterator[str]:
    key = normalize_solver(label)
    token = _active_solver.set(key)
    try:
        yield key
    finally:
        _active_solver.reset(token)


def get_active_solver() -> str:
    return _active_solver.get()


def get_solver_pyomo_name() -> str:
    return SOLVERS[get_active_solver()][0]


def get_solver_gamspy_name() -> str:
    """GAMS solver name for the active solver, or SolverNotAvailableError."""
    active = get_active_solver()
    if active not in GAMSPY_SOLVERS:
        raise SolverNotAvailableError(
            f"The solver '{active}' cannot be used with GAMSPy models. "
            f"Choose one of: {', '.join(sorted(GAMSPY_SOLVERS))}."
        )
    return GAMSPY_SOLVERS[active]


class _Solver:
    """Pyomo solver that reports a missing solution instead of raising.

    Pyomo loads the solution into the model by default. When there is none (e.g. an
    infeasible model) Gurobi only warns, but the APPSI solvers (HiGHS) raise. Solving
    with load_solutions=False and loading afterwards, only if a solution exists,
    makes every solver return its termination condition the same way.
    """

    def __init__(self, solver):
        self._solver = solver

    def __getattr__(self, name):
        return getattr(self._solver, name)

    def solve(self, model, load_solutions: bool = True, **kwargs):
        results = self._solver.solve(model, load_solutions=False, **kwargs)
        if load_solutions and len(results.solution) > 0:
            model.solutions.load_from(results)
        return results


def make_solver(time_limit: float | None = None) -> _Solver:
    """Pyomo solver object for the active solver, or SolverNotAvailableError."""
    active = get_active_solver()
    pyomo_name, time_limit_option = SOLVERS[active]
    solver = SolverFactory(pyomo_name)
    if not solver.available(exception_flag=False):
        raise SolverNotAvailableError(
            f"The solver '{active}' is not installed or not licensed on the "
            "machine running the backend. Choose another solver or install this one."
        )
    if time_limit is not None:
        solver.options[time_limit_option] = time_limit
    return _Solver(solver)
