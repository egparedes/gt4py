from __future__ import annotations

import enum
from typing import Union

import eve
from eve import Coerced, OpNode, SymbolName, SymbolRef, datamodels
from eve.traits import SymbolTableCreatorTrait
from eve.type_definitions import StrEnum


# from functional.iterator.util.sym_validation import validate_symbol_refs


@enum.unique
class GridType(StrEnum):
    CARTESIAN = "cartesian"
    UNSTRUCTURED = "unstructured"


class Sym(OpNode):  # helper
    id: Coerced[SymbolName]  # noqa: A003


class Expr(OpNode):
    ...


class UnaryExpr(Expr):
    op: str
    expr: Expr


class BinaryExpr(Expr):
    op: str
    lhs: Expr
    rhs: Expr


class TernaryExpr(Expr):
    cond: Expr
    true_expr: Expr
    false_expr: Expr


class Literal(Expr):
    value: str
    type: str  # noqa: A003


class OffsetLiteral(Expr):
    value: Union[int, str]


class SymRef(Expr):
    id: Coerced[SymbolRef]  # noqa: A003


class Lambda(Expr, SymbolTableCreatorTrait):
    params: Coerced[eve.Block[Sym]]
    expr: Expr


class FunCall(Expr):
    fun: Expr  # VType[Callable]
    args: Coerced[eve.Block[Expr]]


class TemplatedFunCall(Expr):
    fun: Expr  # VType[Callable]
    template_args: Coerced[eve.Block[Expr]]
    args: Coerced[eve.Block[Expr]]


class FunctionDefinition(OpNode, SymbolTableCreatorTrait):
    id: Coerced[SymbolName]  # noqa: A003
    params: Coerced[eve.Block[Sym]]
    expr: Expr


class Backend(OpNode):
    domain: Union[SymRef, FunCall]  # TODO(havogt) `FunCall` only if domain will be part of the IR


class StencilExecution(OpNode):
    backend: Backend
    stencil: SymRef  # TODO should be list of assigns for canonical `scan`
    output: SymRef
    inputs: Coerced[eve.Block[SymRef]]


BUILTINS = {
    "deref",
    "shift",
    "tuple",
    "get",
    "can_deref",
    "domain",  # TODO(havogt) decide if domain is part of IR
    "named_range",
}


class FencilDefinition(OpNode, eve.traits.SymbolTableTrait):
    id: Coerced[SymbolName]  # noqa: A003
    params: Coerced[eve.Block[Sym]]
    function_definitions: Coerced[eve.Block[FunctionDefinition]]
    executions: Coerced[eve.Block[StencilExecution]]
    offset_declarations: Coerced[eve.Block[str]]
    grid_type: GridType

    builtin_functions: eve.FrozenBlock[Sym] = datamodels.field(
        default=eve.frozenblock(*(Sym(id=name) for name in BUILTINS)), repr=False
    )

    # _validate_symbol_refs = validate_symbol_refs()
