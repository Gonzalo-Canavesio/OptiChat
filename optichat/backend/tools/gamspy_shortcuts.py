"""GAMSPy counterparts of the Pyomo operations in shortcut_functions."""

import operator

import gamspy as gp
import pandas as pd

_OPERATIONS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
}


def is_gamspy_model(model) -> bool:
    return isinstance(model, gp.Model)


def is_discrete_gamspy_model(model: gp.Model) -> bool:
    return model.problem in {gp.Problem.MIP, gp.Problem.MINLP, gp.Problem.MIQCP}


def _new_value(current: float, operation: str, delta) -> float:
    if operation == "=":
        return float(delta)
    if operation not in _OPERATIONS:
        raise ValueError(f"Unknown operation '{operation}'; use one of +, -, *, /, =.")
    return float(_OPERATIONS[operation](current, float(delta)))


def _label_tuple(index) -> tuple[str, ...]:
    """Index as given to modify_and_solve -> GAMS labels (always strings)."""
    if index is None:
        return ()
    if isinstance(index, (tuple, list)):
        return tuple(str(i) for i in index)
    return (str(index),)


def _target_labels(existing: list[tuple[str, ...]], dimension: int, indexes) -> list:
    """Labels a modification applies to.

    An exact index applies even if the symbol has no record there (GAMS stores no
    zeros); a slice or predicate only selects among the existing records.
    """
    if dimension == 0:
        return [()]
    if callable(indexes):
        return [
            labels
            for labels in existing
            if indexes(labels[0] if dimension == 1 else labels)
        ]
    if isinstance(indexes, slice):
        return list(existing)
    if isinstance(indexes, tuple) and any(isinstance(i, slice) for i in indexes):
        fixed = {
            position: str(i)
            for position, i in enumerate(indexes)
            if not isinstance(i, slice)
        }
        return [
            labels
            for labels in existing
            if all(labels[p] == label for p, label in fixed.items())
        ]
    return [_label_tuple(indexes)]


def _index_text(labels: tuple[str, ...]) -> str:
    return f"[{','.join(labels)}]" if labels else ""


def _modify_parameter(symbol: gp.Parameter, indexes, operation, delta) -> list[str]:
    if symbol.dimension == 0:
        current = float(symbol.toValue() or 0.0)
        new = _new_value(current, operation, delta)
        symbol.setRecords(new)
        return [f"  {symbol.name} changed to {new} (was {current})"]

    records = symbol.records
    values = {}
    if records is not None and not records.empty:
        for row in records.itertuples(index=False, name=None):
            values[tuple(map(str, row[: symbol.dimension]))] = float(
                row[symbol.dimension]
            )
    log = []
    for labels in _target_labels(list(values), symbol.dimension, indexes):
        current = values.get(labels, 0.0)
        new = _new_value(current, operation, delta)
        values[labels] = new
        log.append(
            f"  {symbol.name}{_index_text(labels)} changed to {new} (was {current})"
        )
    if log:
        columns = [*symbol.domain_names, "value"]
        if records is not None and not records.empty:
            columns = list(records.columns[: symbol.dimension + 1])
        symbol.setRecords(
            pd.DataFrame(
                [(*labels, value) for labels, value in values.items()], columns=columns
            )
        )
    return log


def _fix_variable(symbol: gp.Variable, indexes, operation, delta) -> list[str]:
    records = symbol.records
    columns = [*symbol.domain_names, "level", "marginal", "lower", "upper", "scale"]
    rows = {}
    if records is not None and not records.empty:
        columns = list(records.columns)
        for row in records.itertuples(index=False, name=None):
            rows[tuple(map(str, row[: symbol.dimension]))] = list(
                row[symbol.dimension :]
            )
    log = []
    for labels in _target_labels(list(rows), symbol.dimension, indexes):
        level, marginal, _lower, _upper, scale = rows.get(
            labels, [0.0, 0.0, 0.0, 0.0, 1.0]
        )
        new = _new_value(float(level), operation, delta)
        rows[labels] = [new, marginal, new, new, scale]
        log.append(f"  {symbol.name}{_index_text(labels)} fixed to {new} (was {level})")
    if log:
        symbol.setRecords(
            pd.DataFrame(
                [(*labels, *attrs) for labels, attrs in rows.items()], columns=columns
            )
        )
    return log


def apply_modification(
    model: gp.Model, component_name: str, indexes, operation, delta
) -> list[str]:
    """Change a Parameter or fix a Variable of a GAMSPy model; returns a change log."""
    container = model.container
    if component_name not in container.data:
        print(
            f"Warning: '{component_name}' not found in the GAMSPy container. Skipping."
        )
        return []
    symbol = container[component_name]
    if isinstance(symbol, gp.Parameter):
        log = _modify_parameter(symbol, indexes, operation, delta)
    elif isinstance(symbol, gp.Variable):
        log = _fix_variable(symbol, indexes, operation, delta)
    else:
        print(f"Warning: '{component_name}' is neither a Parameter nor a Variable.")
        return []
    if not log:
        print(f"Warning: index {indexes!r} for '{component_name}' matched no records.")
    return log


def _unique_model_name(container: gp.Container, base: str) -> str:
    name, k = base, 2
    while name in container.models or name in container.data:
        name, k = f"{base}_{k}", k + 1
    return name


def extend_gamspy_model(
    model: gp.Model, add_equations=(), remove_equations=(), name=None
) -> gp.Model:
    """A new model in the same container: the model's equations minus remove_equations
    (names or Equation objects) plus add_equations, with the same problem type,
    sense and objective."""
    removed = {e if isinstance(e, str) else e.name for e in remove_equations}
    known = {e.name for e in model.equations}
    unknown = removed - known
    if unknown:
        raise ValueError(f"Equations not in model '{model.name}': {sorted(unknown)}")
    equations = [e for e in model.equations if e.name not in removed]
    equations += [e for e in add_equations if e.name not in {q.name for q in equations}]
    extended = gp.Model(
        model.container,
        name=_unique_model_name(model.container, name or f"{model.name}_ext"),
        equations=equations,
        problem=model.problem,
        sense=model.sense,
        objective=model._objective,
    )
    # The objective variable and equation GAMSPy generated for the original model
    # are still internal symbols, not components of the problem
    extended._autogen_symbols |= set(model._autogen_symbols)
    return extended
