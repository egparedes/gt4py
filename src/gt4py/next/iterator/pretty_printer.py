# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""A pretty printer for the functional IR.

Inspired by P. Yelland, “A New Approach to Optimal Code Formatting”, 2015

Dimensions are printed by a short name followed by their kind suffix (`ₕ` horizontal, `ᵥ`
vertical, `ₗ` local). A header at the top of the printed root node maps each short name to the
dimension's full spelling: its tag `module:qualname`, kind suffix and category suffix (`ᵃ`
Cartesian axis, `ˡ` local, none otherwise). `pretty_parser` rebuilds the exact
`DimensionRef`s from it, without importing anything:

    dim IDim = my.model.dims:IDimₕᵃ;
    dim Staggered[KDim] = gt4py.next.common:Staggered[my.model.dims:KDim]ᵥᵃ;
    c⟨ IDimₕ: [0, 10[, Staggered[KDim]ᵥ: [0, 81[ ⟩

The short name is the qualname. Different dimensions sharing one are told apart by the
shortest module suffix that distinguishes them (`dims.IDim`, `other.dims.IDim`). Dimensions
that share their whole tag (they differ only in kind or category) have no distinguishing
short name and are printed in full in place.
"""

# TODO(tehrengruber): add support for printing the types of itir.Sym nodes
from __future__ import annotations

import types as _types
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Final

from gt4py import eve
from gt4py.eve import NodeTranslator
from gt4py.next import common
from gt4py.next.iterator import ir
from gt4py.next.type_system import type_specifications as ts, type_translation


# replacements for builtin binary operations
BINARY_OPS: Final = {
    "plus": "+",
    "minus": "-",
    "multiplies": "×",  # noqa: RUF001 [ambiguous-unicode-character-string]
    "divides": "/",
    "eq": "==",
    "less": "<",
    "less_equal": "<=",
    "greater": ">",
    "greater_equal": ">=",
    "and_": "∧",
    "or_": "∨",  # noqa: RUF001 [ambiguous-unicode-character-string]
}

# replacements for builtin unary operations
UNARY_OPS: Final = {"deref": "·", "lift": "↑", "not_": "¬", "as_fieldop": "⇑"}

# operator precedence
PRECEDENCE: Final = {
    "__lambda__": 0,
    "if_": 1,
    "or_": 2,
    "and_": 3,
    "eq": 4,
    "less": 4,
    "less_equal": 4,
    "greater": 4,
    "greater_equal": 4,
    "plus": 5,
    "minus": 5,
    "multiplies": 6,
    "divides": 6,
    "deref": 7,
    "not_": 7,
    "lift": 7,
    "as_fieldop": 7,
    "tuple_get": 8,
    "__call__": 8,
}


#: Surface spelling of the scalar types, LLVM/MLIR style.
SCALAR_TYPE_NAMES: Final[Mapping[ts.ScalarKind, str]] = _types.MappingProxyType(
    {
        ts.ScalarKind.BOOL: "i1",
        ts.ScalarKind.INT8: "i8",
        ts.ScalarKind.UINT8: "u8",
        ts.ScalarKind.INT16: "i16",
        ts.ScalarKind.UINT16: "u16",
        ts.ScalarKind.INT32: "i32",
        ts.ScalarKind.UINT32: "u32",
        ts.ScalarKind.INT64: "i64",
        ts.ScalarKind.UINT64: "u64",
        ts.ScalarKind.FLOAT32: "f32",
        ts.ScalarKind.FLOAT64: "f64",
    }
)

SCALAR_TYPE_KINDS: Final[Mapping[str, ts.ScalarKind]] = _types.MappingProxyType(
    {name: kind for kind, name in SCALAR_TYPE_NAMES.items()}
)


def format_type(type_: ts.TypeSpec) -> str:
    if isinstance(type_, ts.TupleType):
        return f"{{{', '.join(format_type(t) for t in type_.types)}}}"
    if isinstance(type_, ts.ScalarType) and type_.kind in SCALAR_TYPE_NAMES:
        name = SCALAR_TYPE_NAMES[type_.kind]
        if type_.shape is None:
            return name
        return f"{name}[{', '.join(str(s) for s in type_.shape)}]"
    raise NotImplementedError(f"No pretty-printed form for type '{type_}'.")


def implied_literal_type(value: str) -> ts.ScalarType:
    """Type the bare, unannotated lexeme `value` denotes.

    `pretty_parser` assigns exactly this to an unannotated lexeme, so the printer
    leaves the annotation off when it would agree.  The lexeme is read back to a
    Python value and typed by `type_translation.from_value`, which is what the
    frontend applies to a constant (`ffront/foast_passes/type_deduction.py`,
    `TypeDeducer.visit_Constant`), so the two cannot disagree.

    Raises:
        ValueError: If `value` is not a literal lexeme; neither a bare nor an
            annotated spelling of it parses back.
    """
    py_value: bool | int | float
    if value in ("True", "False"):
        py_value = value == "True"
    else:
        try:
            py_value = int(value)
        except ValueError:
            try:
                py_value = float(value)
            except ValueError:
                raise ValueError(
                    f"Invalid literal value '{value}'; expected a boolean, an integer or a"
                    " floating point lexeme."
                ) from None
    type_ = type_translation.from_value(py_value)
    assert isinstance(type_, ts.ScalarType)
    return type_


DEFAULT_INDENT: Final = 2
DEFAULT_WIDTH: Final = 100


_AXIS_KIND_SUFFIX: Final = {
    common.DimensionKind.HORIZONTAL: "ₕ",
    common.DimensionKind.VERTICAL: "ᵥ",
    common.DimensionKind.LOCAL: "ₗ",
}
#: Printed after the kind suffix; the default category prints nothing. Together with the tag
#: and the kind it is enough to rebuild the `DimensionRef` without importing anything.
_AXIS_CATEGORY_SUFFIX: Final = {
    common.DimensionCategory.CARTESIAN_AXIS: "ᵃ",
    common.DimensionCategory.LOCAL: "ˡ",
    common.DimensionCategory.OTHER: "",
}

#: Keyword of a dimension header entry: `dim <short name> = <full axis literal>;`.
DIMENSION_HEADER_KEYWORD: Final = "dim"


def format_full_axis_literal(dim: common.DimensionRef) -> str:
    """The self-contained spelling of a dimension: tag, kind suffix and category suffix."""
    return dim.tag + _AXIS_KIND_SUFFIX[dim.kind] + _AXIS_CATEGORY_SUFFIX[dim.category]


def _module_depth(dim: common.DimensionRef) -> int:
    # A staggered dimension is told apart by its base: its own module is always `common`.
    return _module_depth(dim.base) if dim.base is not None else dim.module.count(".") + 1


def _short_name(dim: common.DimensionRef, depth: int) -> str:
    """The qualname, prefixed by the last `depth` components of the module path."""
    if dim.base is not None:
        owner = dim.qualname.partition("[")[0]
        return f"{owner}[{_short_name(dim.base, depth)}]"
    module_suffix = dim.module.split(".")[-depth:] if depth else []
    return ".".join([*module_suffix, dim.qualname])


def short_dimension_names(dims: Iterable[common.DimensionRef]) -> dict[common.DimensionRef, str]:
    """
    Assign each of `dims` a distinct short name for printing.

    The short name is the qualname. Where several dimensions would share one, each of them is
    prefixed by one more component of its module path, until the names differ or the module
    paths are exhausted. Dimensions that still share a name (their tags are equal, they differ
    in kind or category only) get none.

    Returns:
        The short name of each dimension that has one; deterministic for a given set of `dims`.
    """
    unique = sorted(set(dims), key=lambda dim: (dim.tag, dim.kind.value, dim.category.value))
    depths = dict.fromkeys(unique, 0)
    while True:
        groups: dict[str, list[common.DimensionRef]] = {}
        for dim in unique:
            groups.setdefault(_short_name(dim, depths[dim]), []).append(dim)
        deepened = False
        for group in groups.values():
            if len(group) > 1:
                for dim in group:
                    if depths[dim] < _module_depth(dim):
                        depths[dim] += 1
                        deepened = True
        if not deepened:
            return {group[0]: name for name, group in groups.items() if len(group) == 1}


class PrettyPrinter(NodeTranslator):
    def __init__(
        self,
        indent: int = DEFAULT_INDENT,
        width: int = DEFAULT_WIDTH,
    ) -> None:
        super().__init__()
        self.indent: int = indent
        self.width: int = width
        #: Dimensions printed by short name; any other is printed in full in place.
        self.short_names: Mapping[common.DimensionRef, str] = {}

    @staticmethod
    def _hmerge(*blocks: list[str]) -> list[str]:
        if not blocks:
            return []

        def impl(a: list[str], b: list[str]) -> list[str]:
            if not a:
                return b
            if not b:
                return a
            spaces = len(a[-1]) * " "
            return a[:-1] + [a[-1] + b[0]] + [spaces + line for line in b[1:]]

        res = blocks[0]
        for b in blocks[1:]:
            res = impl(res, b)
        return res

    @staticmethod
    def _vmerge(*blocks: list[str]) -> list[str]:
        return [s for b in blocks for s in b]

    def _prec_parens(self, block: list[str], prec: int, op_prec: int) -> list[str]:
        if prec > op_prec:
            return self._hmerge(["("], block, [")"])
        return block

    def _indent(self, block: list[str]) -> list[str]:
        return [" " * self.indent + line for line in block]

    def _cost(self, block: list[str]) -> int:
        if not block:
            return 0
        max_line_length = max(len(line) for line in block)
        return (
            # preferring blocks of fewer lines:
            len(block)
            # strongly preferring blocks with lines not longer than self.width:
            + max(max_line_length - self.width, 0) * 100
            # preferring blocks with lines of uniform length:
            + sum(max_line_length - len(line) for line in block)
        )

    def _optimum(self, a: list[str], b: list[str]) -> list[str]:
        return a if self._cost(a) < self._cost(b) else b

    def _hinterleave(
        self, blocks: Sequence[list[str]], sep: str, *, indent: bool = False
    ) -> Iterator[list[str]]:
        if blocks:
            do_indent = self._indent if indent else lambda x: x
            yield from (do_indent(self._hmerge(block, [sep])) for block in blocks[:-1])
            yield do_indent(blocks[-1])

    def visit_Sym(self, node: ir.Sym, *, prec: int) -> list[str]:
        return [node.id]

    def visit_Literal(self, node: ir.Literal, *, prec: int) -> list[str]:
        # Unconditional, as it also rejects a value that is no literal lexeme: printing such a
        # value, annotated or not, yields text that does not parse back.
        implied_type = implied_literal_type(node.value)
        return [
            node.value if implied_type == node.type else f"{node.value}:{format_type(node.type)}"
        ]

    def visit_InfinityLiteral(self, node: ir.InfinityLiteral, *, prec: int) -> list[str]:
        if node == ir.InfinityLiteral.POSITIVE:
            return ["∞"]
        elif node == ir.InfinityLiteral.NEGATIVE:
            return ["-∞"]
        raise AssertionError()

    def visit_OffsetLiteral(self, node: ir.OffsetLiteral, *, prec: int) -> list[str]:
        return [str(node.value) + "ₒ"]

    def visit_CartesianOffset(self, node: ir.CartesianOffset, *, prec: int) -> list[str]:
        (domain,) = self.visit(node.domain, prec=0)
        (codomain,) = self.visit(node.codomain, prec=0)
        return [f"{domain}→{codomain}"]

    def visit_AxisLiteral(self, node: ir.AxisLiteral, *, prec: int) -> list[str]:
        # NOTE: the literal carries its `DimensionRef`, so printing needs no import. A short
        # name stands for the full spelling given in the header (see `apply`).
        dim = node.dim
        if (short_name := self.short_names.get(dim)) is not None:
            return [short_name + _AXIS_KIND_SUFFIX[dim.kind]]
        return [format_full_axis_literal(dim)]

    def visit_SymRef(self, node: ir.SymRef, *, prec: int) -> list[str]:
        return [node.id]

    def visit_Lambda(self, node: ir.Lambda, *, prec: int) -> list[str]:
        params = self.visit(node.params, prec=0)
        expr = self.visit(node.expr, prec=0)

        start, bridge = ["λ("], [") → "]
        if not params:
            params = self._hmerge(start, bridge)
        else:
            hparams = self._hmerge(start, *self._hinterleave(params, ", "), bridge)
            vparams = self._vmerge(start, *self._hinterleave(params, ",", indent=True), bridge)
            params = self._optimum(hparams, vparams)

        hbody = self._hmerge(params, expr)
        vbody = self._vmerge(params, self._indent(expr))
        return self._prec_parens(self._optimum(hbody, vbody), prec, PRECEDENCE["__lambda__"])

    def visit_FunCall(self, node: ir.FunCall, *, prec: int) -> list[str]:
        if isinstance(node.fun, ir.SymRef):
            fun_name = node.fun.id
            if fun_name in BINARY_OPS and len(node.args) == 2:
                # replacing binary ops: plus(x, y) → x + y etc.
                op = BINARY_OPS[fun_name]
                lhs = self.visit(node.args[0], prec=PRECEDENCE[fun_name])
                rhs = self.visit(node.args[1], prec=PRECEDENCE[fun_name] + 1)
                h = self._hmerge(lhs, [" " + op + " "], rhs)
                v = self._vmerge(lhs, self._hmerge([op + " "], rhs))
                return self._prec_parens(self._optimum(h, v), prec, PRECEDENCE[fun_name])
            if fun_name in UNARY_OPS and len(node.args) == 1:
                # replacing unary ops: deref(x) → ·x etc.
                op = UNARY_OPS[fun_name]
                res = self._hmerge([op], self.visit(node.args[0], prec=PRECEDENCE[fun_name]))
                return self._prec_parens(res, prec, PRECEDENCE[fun_name])
            if fun_name == "tuple_get" and len(node.args) == 2:
                # tuple_get(i, x) → x[i]
                idx, tup = self.visit(node.args, prec=PRECEDENCE[fun_name])
                res = self._hmerge(tup, ["["], idx, ["]"])
                return self._prec_parens(res, prec, PRECEDENCE[fun_name])
            if fun_name == "shift":
                shifts = self.visit(node.args, prec=0)
                return self._hmerge(["⟪"], *self._hinterleave(shifts, ", "), ["⟫"])
            if fun_name == "named_range" and len(node.args) == 3:
                # named_range(dim, start, stop) → dim: [star, stop)
                dim, start, end = self.visit(node.args, prec=0)
                res = self._hmerge(
                    dim, [": ["], start, [", "], end, ["["]
                )  # to get matching parenthesis of functions
                return self._prec_parens(res, prec, PRECEDENCE["__call__"])
            if fun_name == "cartesian_domain" and len(node.args) >= 1:
                # cartesian_domain(x, y, ...) → c{ x × y × ... } # noqa: RUF003 [ambiguous-unicode-character-comment]
                args = self.visit(node.args, prec=PRECEDENCE["__call__"])
                return self._hmerge(["c⟨ "], *self._hinterleave(args, ", "), [" ⟩"])
            if fun_name == "unstructured_domain" and len(node.args) >= 1:
                # unstructured_domain(x, y, ...) → u{ x × y × ... } # noqa: RUF003 [ambiguous-unicode-character-comment]
                args = self.visit(node.args, prec=PRECEDENCE["__call__"])
                return self._hmerge(["u⟨ "], *self._hinterleave(args, ", "), [" ⟩"])
            if fun_name == "if_" and len(node.args) == 3:
                # if_(x, y, z) → if x then y else z
                ifb, thenb, elseb = self.visit(node.args, prec=PRECEDENCE["if_"])
                hblocks = self._hmerge(["if "], ifb, [" then "], thenb, [" else "], elseb)
                vblocks = self._vmerge(
                    self._hmerge(["if   "], ifb),
                    self._hmerge(["then "], thenb),
                    self._hmerge(["else "], elseb),
                )
                return self._prec_parens(self._optimum(hblocks, vblocks), prec, PRECEDENCE["if_"])

        fun = self.visit(node.fun, prec=PRECEDENCE["__call__"])
        args = self.visit(node.args, prec=0)

        if not args:
            args = [""]
        else:
            hargs = self._hmerge(*self._hinterleave(args, ", "))
            vargs = self._vmerge(*self._hinterleave(args, ","))
            args = self._optimum(hargs, vargs)

        if node.fun == ir.SymRef(id="make_tuple"):
            # make_tuple(args...) → {args...}
            htup = self._hmerge(["{"], args, ["}"])
            vtup = self._vmerge(["{"], self._indent(args), ["}"])
            return self._optimum(htup, vtup)

        hfun = self._hmerge(fun, ["("], args, [")"])
        vfun = self._vmerge(self._hmerge(fun, ["("]), self._indent(args), [")"])
        return self._prec_parens(self._optimum(hfun, vfun), prec, PRECEDENCE["__call__"])

    def visit_FunctionDefinition(self, node: ir.FunctionDefinition, prec: int) -> list[str]:
        assert prec == 0
        params = self.visit(node.params, prec=0)
        expr = self._hmerge(self.visit(node.expr, prec=0), [";"])

        start, bridge = [node.id + " = λ("], [") → "]
        if not params:
            params = self._hmerge(start, bridge)
        else:
            hparams = self._hmerge(start, *self._hinterleave(params, ", "), bridge)
            vparams = self._vmerge(start, *self._hinterleave(params, ",", indent=True), bridge)
            params = self._optimum(hparams, vparams)

        hbody = self._hmerge(params, expr)
        vbody = self._vmerge(params, self._indent(expr))
        return self._optimum(hbody, vbody)

    def visit_Temporary(self, node: ir.Temporary, *, prec: int) -> list[str]:
        start, end = [node.id + " = temporary("], [");"]
        args = []
        if node.domain is not None:
            args.append(self._hmerge(["domain="], self.visit(node.domain, prec=0)))
        if node.dtype is not None:
            args.append(self._hmerge(["dtype="], [format_type(node.dtype)]))
        hargs = self._hmerge(*self._hinterleave(args, ", "))
        vargs = self._vmerge(*self._hinterleave(args, ","))
        oargs = self._optimum(hargs, vargs)
        h = self._hmerge(start, oargs, end)
        v = self._vmerge(start, self._indent(oargs), end)
        return self._optimum(h, v)

    def visit_SetAt(self, node: ir.SetAt, *, prec: int) -> list[str]:
        expr = self.visit(node.expr, prec=0)
        domain = self.visit(node.domain, prec=0)
        target = self.visit(node.target, prec=0)

        head = self._hmerge(target, [" @ "], domain)
        foot = self._hmerge([" ← "], expr, [";"])

        h = self._hmerge(head, foot)
        v = self._vmerge(
            head,
            self._indent(self._indent(foot)),
        )
        return self._optimum(h, v)

    def visit_IfStmt(self, node: ir.IfStmt, *, prec: int) -> list[str]:
        cond = self.visit(node.cond, prec=0)
        true_branch = self._vmerge(*self.visit(node.true_branch, prec=0))
        false_branch = self._vmerge(*self.visit(node.false_branch, prec=0))

        hhead = self._hmerge(["if ("], cond, [") {"])
        vhead = self._vmerge(["if ("], cond, [") {"])
        head = self._optimum(hhead, vhead)

        return self._vmerge(
            head, self._indent(true_branch), ["} else {"], self._indent(false_branch), ["}"]
        )

    def visit_Program(self, node: ir.Program, *, prec: int) -> list[str]:
        assert prec == 0
        function_definitions = self.visit(node.function_definitions, prec=0)
        body = self.visit(node.body, prec=0)
        declarations = self.visit(node.declarations, prec=0)
        params = self.visit(node.params, prec=0)

        hparams = self._hmerge([node.id + "("], *self._hinterleave(params, ", "), [") {"])
        vparams = self._vmerge(
            [node.id + "("], *self._hinterleave(params, ",", indent=True), [") {"]
        )
        params = self._optimum(hparams, vparams)

        function_definitions = self._vmerge(*function_definitions)
        declarations = self._vmerge(*declarations)
        body = self._vmerge(*body)

        return self._vmerge(
            params,
            self._indent(function_definitions),
            self._indent(declarations),
            self._indent(body),
            ["}"],
        )

    @classmethod
    def apply(cls, node: ir.Node, indent: int, width: int) -> str:
        """Print `node`, preceded by the header declaring the short dimension names it uses."""
        printer = cls(indent=indent, width=width)
        dims = eve.walk_values(node).if_isinstance(ir.AxisLiteral).getattr("dim")
        printer.short_names = short_dimension_names(dims)
        header = [
            f"{DIMENSION_HEADER_KEYWORD} {name} = {format_full_axis_literal(dim)};"
            for dim, name in sorted(printer.short_names.items(), key=lambda item: item[1])
        ]
        return "\n".join([*header, *printer.visit(node, prec=0)])


def pformat(x: ir.Node, indent: int = DEFAULT_INDENT, width: int = DEFAULT_WIDTH) -> str:
    return PrettyPrinter.apply(x, indent, width)


def pprint(x: ir.Node, indent: int = DEFAULT_INDENT, width: int = DEFAULT_WIDTH) -> None:
    print(pformat(x, indent, width))
