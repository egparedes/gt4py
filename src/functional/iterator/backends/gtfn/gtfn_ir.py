from __future__ import annotations

import enum
from typing import Union

import eve
from eve import OpNode, SymbolName, SymbolRef, datamodels
from eve.traits import SymbolTableCreatorTrait
from eve.type_definitions import StrEnum


# from functional.iterator.util.sym_validation import validate_symbol_refs


@enum.unique
class GridType(StrEnum):
    CARTESIAN = "cartesian"
    UNSTRUCTURED = "unstructured"


class Sym(OpNode):  # helper
    id: SymbolName = datamodels.coerced_field()  # noqa: A003


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
    id: SymbolRef = datamodels.coerced_field()  # noqa: A003


class Lambda(Expr, SymbolTableCreatorTrait):
    params: eve.Block[Sym] = datamodels.coerced_field()
    expr: Expr


class FunCall(Expr):
    fun: Expr  # VType[Callable]
    args: eve.Block[Expr] = datamodels.coerced_field()


class TemplatedFunCall(Expr):
    fun: Expr  # VType[Callable]
    template_args: eve.Block[Expr] = datamodels.coerced_field()
    args: eve.Block[Expr] = datamodels.coerced_field()


class FunctionDefinition(OpNode, SymbolTableCreatorTrait):
    id: SymbolName = datamodels.coerced_field()  # noqa: A003
    params: eve.Block[Sym] = datamodels.coerced_field()
    expr: Expr


class Backend(OpNode):
    domain: Union[SymRef, FunCall]  # TODO(havogt) `FunCall` only if domain will be part of the IR


class StencilExecution(OpNode):
    backend: Backend
    stencil: SymRef  # TODO should be list of assigns for canonical `scan`
    output: SymRef
    inputs: eve.Block[SymRef] = datamodels.coerced_field()


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
    id: SymbolName = datamodels.coerced_field()  # noqa: A003
    params: eve.Block[Sym] = datamodels.coerced_field()
    function_definitions: eve.Block[FunctionDefinition] = datamodels.coerced_field()
    executions: eve.Block[StencilExecution] = datamodels.coerced_field()
    offset_declarations: eve.Block[str] = datamodels.coerced_field()
    grid_type: GridType

    builtin_functions: eve.FrozenBlock[Sym] = datamodels.field(
        default=eve.frozenblock(*(Sym(id=name) for name in BUILTINS)), repr=False
    )

    # _validate_symbol_refs = validate_symbol_refs()
