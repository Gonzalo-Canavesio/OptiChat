import math
import os
import re

import gamspy as gp

from optichat.backend.config import workspace
from optichat.backend.config.solver import (
    SolverNotAvailableError,
    get_solver_gamspy_name,
)

EPS = 1e-5

_OPTIMAL = {
    gp.ModelStatus.OptimalGlobal,
    gp.ModelStatus.OptimalLocal,
    gp.ModelStatus.Solved,
    gp.ModelStatus.SolvedUnique,
    gp.ModelStatus.SolvedSingular,
}
_FEASIBLE = {gp.ModelStatus.Feasible, gp.ModelStatus.Integer}
_UNBOUNDED = {gp.ModelStatus.Unbounded, gp.ModelStatus.UnboundedNoSolution}
_INFEASIBLE = {
    gp.ModelStatus.InfeasibleGlobal,
    gp.ModelStatus.InfeasibleLocal,
    gp.ModelStatus.InfeasibleIntermed,
    gp.ModelStatus.InfeasibleNoSolution,
    gp.ModelStatus.IntegerInfeasible,
}
_DISCRETE = {gp.Problem.MIP, gp.Problem.MINLP, gp.Problem.MIQCP}
_SOLVE_FAILURES = {
    gp.SolveStatus.CapabilityError,
    gp.SolveStatus.SetupError,
    gp.SolveStatus.SolverError,
    gp.SolveStatus.InternalError,
    gp.SolveStatus.SystemError,
}
_SENSES = {"=e=": "==", "=l=": "<=", "=g=": ">="}
# Symbols OptiChat adds to a model's container (robustness analysis, relaxation)
HELPER_PREFIX = "optichat_"
_WORD = re.compile(r"[A-Za-z_]\w*")


def save_gamspy_model(model: gp.Model, version: str) -> str:
    """Serialize the model's container to <workspace>/model_objects/<version>.zip.

    GAMSPy containers cannot be pickled; serialization keeps every symbol with its
    records and every model with its equation definitions, so the model can be
    modified and solved again after restore_gamspy_model.
    """
    folder = workspace.model_objects_dir()
    os.makedirs(folder, exist_ok=True)
    path = str(folder / f"{version}.zip")
    # gp.deserialize restores the statuses as ints, and gp.serialize expects the
    # enums, so a restored model that was not solved again could not be saved
    for other in model.container.models.values():
        if isinstance(other._status, int):
            other._status = gp.ModelStatus(other._status)
        if isinstance(other._solve_status, int):
            other._solve_status = gp.SolveStatus(other._solve_status)
    gp.serialize(model.container, path)
    return path


def restore_gamspy_model(path: str, model_name: str) -> gp.Model:
    """The model named model_name from a container saved by save_gamspy_model."""
    return gp.deserialize(path).models[model_name]


def solve_gamspy_model(model: gp.Model, time_limit: float | None = None) -> str:
    """Solve with the active solver; returns the status named like Pyomo's
    termination conditions ('optimal', 'infeasible', ...)."""
    options = gp.Options(time_limit=time_limit) if time_limit is not None else None
    model.solve(solver=get_solver_gamspy_name(), options=options)
    if (
        model.status == gp.ModelStatus.LicenseError
        or model.solve_status == gp.SolveStatus.LicenseError
    ):
        raise SolverNotAvailableError(
            f"The solver '{get_solver_gamspy_name()}' is not licensed for this "
            "GAMSPy model. Choose another solver."
        )
    if model.solve_status in _SOLVE_FAILURES:
        raise RuntimeError(
            f"The solver '{get_solver_gamspy_name()}' failed "
            f"({model.solve_status.name})."
        )
    return _termination_condition(model)


def _termination_condition(model: gp.Model) -> str:
    status = model.status
    if status in _OPTIMAL:
        return "optimal"
    if status in _FEASIBLE:
        return "feasible"
    if status in _UNBOUNDED:
        return "unbounded"
    if status in _INFEASIBLE:
        return "infeasible"
    if status == gp.ModelStatus.NotSolved:
        return "unknown"
    return "other"


def _number(value, integer: bool = False):
    """JSON-safe number: NaN (NA/undefined) is None, infinities are 'inf'/'-inf'."""
    if value is None or math.isnan(value):
        return None
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return int(round(value)) if integer else float(value)


def _bound(value):
    """Variable and equation bounds: None when unbounded."""
    return (
        None
        if value is None or math.isnan(value) or math.isinf(value)
        else float(value)
    )


def _records(symbol):
    """(index labels, attribute values) of each record, as lists."""
    records = symbol.records
    if records is None or records.empty:
        return []
    dimension = symbol.dimension
    return [
        (list(row[:dimension]), row[dimension:])
        for row in records.itertuples(index=False, name=None)
    ]


def _name(symbol, labels) -> str:
    return f"{symbol.name}[{','.join(map(str, labels))}]" if labels else symbol.name


def _symbols_in(text: str, names: set[str]) -> list[str]:
    return sorted(set(_WORD.findall(text)) & names)


def _extract_sets(container: gp.Container) -> dict:
    info = {}
    for symbol in [*container.getSets(), *container.getAliases()]:
        members = [
            labels[0] if len(labels) == 1 else labels for labels, _ in _records(symbol)
        ]
        entry = {"component_type": "set", "members": members}
        if symbol.description:
            entry["doc"] = symbol.description
        if any(name != "*" for name in symbol.domain_names):
            entry["domain"] = symbol.domain_names
        if isinstance(symbol, gp.Alias):
            alias_with = symbol.alias_with
            entry["alias_with"] = (
                alias_with if isinstance(alias_with, str) else alias_with.name
            )
        info[symbol.name] = entry
    return info


def _extract_parameters(container: gp.Container) -> dict:
    info = {}
    for symbol in container.getParameters():
        if symbol.name.startswith(HELPER_PREFIX):
            continue  # evaluation helpers of gamspy_analysis, not model data
        records = _records(symbol)
        if symbol.dimension == 0:  # a scalar at its default is stored without records
            records = [([], (symbol.toValue() or 0.0,))]
        for labels, (value, *_) in records:
            entry = {"component_type": "parameter", "value": _number(value)}
            if symbol.description:
                entry["doc"] = symbol.description
            info[_name(symbol, labels)] = entry
    return info


def _extract_variables(
    container: gp.Container, autogen: set[str], solved: bool
) -> dict:
    info = {}
    for symbol in container.getVariables():
        if symbol.name in autogen:
            continue
        integer = symbol.type in ("integer", "binary")
        for labels, (level, _marginal, lower, upper, *_) in _records(symbol):
            entry = {
                "component_type": "variable",
                "solution": _number(level, integer) if solved else "unknown",
                "lb": _bound(lower),
                "ub": _bound(upper),
                "domain": symbol.type,
            }
            if symbol.description:
                entry["doc"] = symbol.description
            info[_name(symbol, labels)] = entry
    return info


def _is_binding(level, lower, upper) -> bool:
    return any(
        bound is not None and abs(level - bound) < EPS * max(1.0, abs(bound))
        for bound in (_bound(lower), _bound(upper))
    )


def _extract_constraints(
    model: gp.Model,
    autogen: set[str],
    solved: bool,
    var_names: set[str],
    param_names: set[str],
) -> dict:
    info = {}
    with_duals = solved and model.problem not in _DISCRETE
    for symbol in model.equations:
        if symbol.name in autogen:
            continue
        definition = symbol.getDefinition()
        senses = [_SENSES[op] for op in re.findall(r"=[elg]=", definition)]
        indexed = symbol.dimension > 0
        for labels, (level, marginal, lower, upper, *_) in _records(symbol):
            entry = {
                "component_type": "constraint",
                "activity": _number(level) if solved else "unknown",
                "lb": _bound(lower),
                "ub": _bound(upper),
                "is_binding": _is_binding(level, lower, upper) if solved else "unknown",
                "dual": _number(marginal) if with_duals else "unknown",
            }
            if indexed:
                if symbol.description:
                    entry["doc"] = symbol.description
            else:
                entry.update(
                    _definition_fields(
                        symbol, definition, senses, var_names, param_names
                    )
                )
            info[_name(symbol, labels)] = entry
        if indexed:
            # Every record of an indexed equation shares one symbolic definition
            info[f"{symbol.name}[*]"] = {
                "component_type": "constraint",
                **_definition_fields(
                    symbol, definition, senses, var_names, param_names
                ),
            }
    return info


def _definition_fields(symbol, definition, senses, var_names, param_names) -> dict:
    fields = {
        "expression": definition,
        "sense": senses[0] if len(senses) == 1 else senses,
    }
    fields["vars_in"] = _symbols_in(definition, var_names)
    fields["params_in"] = _symbols_in(definition, param_names)
    if symbol.description:
        fields["doc"] = symbol.description
    return fields


def _extract_objective(
    model: gp.Model,
    termination_condition: str,
    var_names: set[str],
    param_names: set[str],
) -> dict:
    objective = model._objective
    if objective is None:
        expression = ""
    elif isinstance(objective, gp.Variable):
        expression = objective.name
    else:
        expression = objective.gamsRepr()
    return {
        "obj": {
            "component_type": "objective",
            "sense": {"MIN": "MINIMIZE", "MAX": "MAXIMIZE"}.get(
                model.sense.name, model.sense.name
            ),
            "expression": expression,
            "sol_status": termination_condition,
            "value": (
                _number(model.objective_value)
                if termination_condition == "optimal"
                else "unknown"
            ),
            "params_in": _symbols_in(expression, param_names),
            "vars_in": _symbols_in(expression, var_names),
        }
    }


def extract_gamspy_model_info(model: gp.Model, termination_condition: str) -> dict:
    """Components of a solved GAMSpy model in the format of extract_model_info.

    Indexed symbols get one entry per record, named like Pyomo's ('x[a,b]'). Records
    are sparse: an index that GAMS did not generate has no entry. The variable and
    equation that GAMSpy creates for the objective are reported as 'obj' instead.
    """
    container = model.container
    autogen = set(model._autogen_symbols)
    solved = termination_condition == "optimal"
    var_names = {v.name for v in container.getVariables()} - autogen
    param_names = {
        p.name
        for p in container.getParameters()
        if not p.name.startswith(HELPER_PREFIX)
    }

    info = {}
    info.update(_extract_sets(container))
    info.update(_extract_parameters(container))
    info.update(_extract_variables(container, autogen, solved))
    info.update(_extract_constraints(model, autogen, solved, var_names, param_names))
    info.update(
        _extract_objective(model, termination_condition, var_names, param_names)
    )
    return info
