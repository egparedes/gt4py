import enum
from typing import List, Union

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
    id: SymbolName = datamodels.field(converter=True)  # noqa: A003


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
    id: SymbolRef = datamodels.field(converter=True)  # noqa: A003


class Lambda(Expr, SymbolTableCreatorTrait):
    params: List[Sym]
    expr: Expr


class FunCall(Expr):
    fun: Expr  # VType[Callable]
    args: List[Expr]


class TemplatedFunCall(Expr):
    fun: Expr  # VType[Callable]
    template_args: List[Expr]
    args: List[Expr]


class FunctionDefinition(OpNode, SymbolTableCreatorTrait):
    id: SymbolName = datamodels.field(converter=True)  # noqa: A003
    params: List[Sym]
    expr: Expr


class Backend(OpNode):
    domain: Union[SymRef, FunCall]  # TODO(havogt) `FunCall` only if domain will be part of the IR


class StencilExecution(OpNode):
    backend: Backend
    stencil: SymRef  # TODO should be list of assigns for canonical `scan`
    output: SymRef
    inputs: List[SymRef]


class FencilDefinition(OpNode, eve.traits.SymbolTableTrait):
    id: SymbolName = datamodels.field(converter=True)  # noqa: A003
    params: List[Sym]
    function_definitions: List[FunctionDefinition]
    executions: List[StencilExecution]
    offset_declarations: List[str]
    grid_type: GridType

    builtin_functions = list(
        Sym(id=name)
        for name in [
            "deref",
            "shift",
            "tuple",
            "get",
            "can_deref",
            "domain",  # TODO(havogt) decide if domain is part of IR
            "named_range",
        ]
    )

    # _validate_symbol_refs = validate_symbol_refs()
