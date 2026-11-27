"""Track common data preparation operations without executing user code."""

import ast
from dataclasses import dataclass
from itertools import count


PREPROCESSOR_MODULES = {
    "compose", "decomposition", "feature_selection", "impute",
    "kernel_approximation", "pipeline", "preprocessing", "random_projection",
}
PREPROCESSOR_FACTORIES = {
    "make_column_transformer", "make_pipeline", "make_union",
}
STATISTICS = {
    "mean", "median", "mode", "std", "var", "min", "max", "quantile",
    "value_counts",
}
PANDAS_MUTATORS = {"fillna", "replace", "where", "mask", "clip"}
STATELESS_PREPROCESSORS = {"Binarizer", "FunctionTransformer", "Normalizer"}
FIT_METHODS = {"fit", "fit_transform", "partial_fit"}


@dataclass(frozen=True, order=True)
class Diagnostic:
    cell: int
    line: int
    column: int
    code: str
    message: str


@dataclass(frozen=True)
class Value:
    key: int
    lineage: frozenset[int]
    statistics: frozenset[Diagnostic]
    partitions: frozenset[str]
    qualified: str = ""
    preprocessor: bool = False
    items: tuple["Value", ...] = ()


class Analyzer(ast.NodeVisitor):
    def __init__(self, index_splits=()):
        self.cell = 0
        self.index_splits = frozenset(index_splits)
        self.bindings: dict[str, Value] = {}
        self.diagnostics: set[Diagnostic] = set()
        self.fits: dict[int, set[Diagnostic]] = {}
        self.fitted_inputs: dict[int, Value] = {}
        self.next_key = count()

    def value(self, parts=(), *, partitions=(), qualified="", preprocessor=False,
              statistics=(), items=()):
        parts = tuple(parts)
        key = next(self.next_key)
        return Value(
            key,
            frozenset({key}).union(*(part.lineage for part in parts)),
            frozenset(statistics).union(*(part.statistics for part in parts)),
            frozenset(partitions).union(*(part.partitions for part in parts)),
            qualified,
            preprocessor,
            tuple(items),
        )

    def location(self, node, code, message):
        return Diagnostic(self.cell, node.lineno, node.col_offset + 1, code, message)

    def name(self, name):
        if name not in self.bindings:
            self.bindings[name] = self.value()
        return self.bindings[name]

    def visit_Import(self, node):
        for alias in node.names:
            root = alias.name.split(".")[0]
            self.bindings[alias.asname or root] = self.value(
                qualified=alias.name if alias.asname else root
            )

    def visit_ImportFrom(self, node):
        if node.level == 0 and node.module:
            for alias in node.names:
                self.bindings[alias.asname or alias.name] = self.value(
                    qualified=f"{node.module}.{alias.name}"
                )

    def visit_Assign(self, node):
        result = self.expression(node.value)
        for target in node.targets:
            self.bind(target, result)

    def visit_AnnAssign(self, node):
        if node.value is not None:
            self.bind(node.target, self.expression(node.value))

    def visit_AugAssign(self, node):
        self.bind(node.target, self.value((self.expression(node.target),
                                          self.expression(node.value))))

    def visit_Expr(self, node):
        self.expression(node.value)

    def visit_Return(self, node):
        if node.value is not None:
            self.expression(node.value)

    def visit_Call(self, node):
        self.expression(node)

    def visit_FunctionDef(self, node):
        outer_bindings, outer_fits = self.bindings, self.fits
        outer_fitted_inputs = self.fitted_inputs
        self.bindings, self.fits = self.bindings.copy(), self.fits.copy()
        self.fitted_inputs = self.fitted_inputs.copy()
        arguments = node.args
        parameters = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
        parameters.extend(arg for arg in (arguments.vararg, arguments.kwarg) if arg)
        for argument in parameters:
            self.bindings[argument.arg] = self.value()
        for statement in node.body:
            self.visit(statement)
        self.bindings, self.fits = outer_bindings, outer_fits
        self.fitted_inputs = outer_fitted_inputs
        self.bindings[node.name] = self.value()

    visit_AsyncFunctionDef = visit_FunctionDef

    def bind(self, target, value):
        if isinstance(target, ast.Name):
            self.bindings[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            matched = len(target.elts) == len(value.items)
            for index, element in enumerate(target.elts):
                self.bind(element, value.items[index] if matched else value)
        elif isinstance(target, (ast.Subscript, ast.Attribute)):
            base = target.value
            while isinstance(base, (ast.Subscript, ast.Attribute)):
                base = base.value
            if isinstance(base, ast.Name):
                original = self.name(base.id)
                updated = self.value((original, value))
                for name, binding in self.bindings.items():
                    if binding is original:
                        self.bindings[name] = updated

    def expression(self, node):
        if isinstance(node, ast.Name):
            return self.name(node.id)
        if isinstance(node, ast.Attribute):
            owner = self.expression(node.value)
            qualified = f"{owner.qualified}.{node.attr}" if owner.qualified else ""
            return self.value((owner,), qualified=qualified)
        if isinstance(node, ast.Call):
            return self.call(node)
        if isinstance(node, ast.Subscript):
            owner = self.expression(node.value)
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
                index = node.slice.value
                if -len(owner.items) <= index < len(owner.items):
                    return owner.items[index]
            return self.value((owner, self.expression(node.slice)))
        if isinstance(node, (ast.List, ast.Tuple)):
            items = tuple(self.expression(element) for element in node.elts)
            return self.value(items, items=items)
        if isinstance(node, ast.NamedExpr):
            value = self.expression(node.value)
            self.bind(node.target, value)
            return value
        if isinstance(node, ast.Lambda):
            return self.value()
        return self.value(self.expression(child) for child in ast.iter_child_nodes(node))

    def call(self, node):
        method = node.func.attr if isinstance(node.func, ast.Attribute) else ""
        owner = self.expression(node.func.value) if method else None
        qualified = (f"{owner.qualified}.{method}" if owner.qualified else "") if owner else (
            self.expression(node.func).qualified
        )
        args = [self.expression(arg) for arg in node.args]
        keywords = {kw.arg: self.expression(kw.value) for kw in node.keywords}
        parts = [*args, *keywords.values()]
        if qualified == "sklearn.model_selection.train_test_split":
            return self.split(args)
        if qualified in self.index_splits:
            source = args[0] if args else keywords.get("X")
            return self.split([source]) if source is not None else self.value()

        package, _, member = qualified.rpartition(".")
        if package.startswith("sklearn.") and package.split(".")[1] in PREPROCESSOR_MODULES:
            if member[:1].isupper() or member in PREPROCESSOR_FACTORIES:
                return self.value(preprocessor=member not in STATELESS_PREPROCESSORS)
            if package == "sklearn.preprocessing" and member in {
                "scale", "minmax_scale", "maxabs_scale", "robust_scale", "quantile_transform",
                "power_transform",
            }:
                source = args[0] if args else keywords.get("X")
                if source is not None:
                    return self.fit(node, method or member, source)

        if owner and owner.preprocessor:
            source = args[0] if args else keywords.get("X")
            if method in FIT_METHODS and source is not None:
                transformed = self.fit(node, method, source)
                self.fitted_inputs[owner.key] = transformed
                return transformed if method == "fit_transform" else owner
            if method == "transform" and source is not None:
                fitted = self.fitted_inputs.get(owner.key)
                if fitted is not None and not fitted.partitions:
                    return self.value((source, fitted))
                return self.value((source,))

        if owner and method in STATISTICS and not (owner.qualified or owner.partitions):
            # Reducing columns within a row does not learn across samples.
            axis = next((kw.value for kw in node.keywords if kw.arg == "axis"), None)
            axis_position = 1 if method == "quantile" else 0
            if axis is None and method != "value_counts" and len(node.args) > axis_position:
                axis = node.args[axis_position]
            if isinstance(axis, ast.Constant) and axis.value in (1, "columns"):
                return self.value((owner, *parts))
            diagnostic = self.location(
                node, "PG003", f".{method}() contributes a statistic computed before the split"
            )
            return self.value((owner, *parts), statistics=(diagnostic,))

        result = self.value(([owner] if owner else []) + parts)
        if owner and method in PANDAS_MUTATORS and any(
            kw.arg == "inplace" and isinstance(kw.value, ast.Constant) and kw.value.value is True
            for kw in node.keywords
        ):
            self.bind(node.func.value, result)
        return result

    def fit(self, node, method, source):
        if "test" in source.partitions:
            self.diagnostics.add(self.location(
                node, "PG002", f"{method} learns from a held-out test partition"
            ))
        else:
            diagnostic = self.location(node, "PG001", f"{method} learns from data before it is split")
            self.fits[source.key] = self.fits.get(source.key, set()) | {diagnostic}
        return self.value((source,))

    def split(self, args):
        items = []
        for source in args:
            for key in source.lineage:
                self.diagnostics.update(self.fits.get(key, ()))
            self.diagnostics.update(source.statistics)
            items.extend(self.value(partitions=(partition,)) for partition in ("train", "test"))
        return self.value(items, items=items)
