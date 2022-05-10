from typing import List, Union

import eve
from eve import datamodels
from eve.concepts import SymbolName, SymbolRef
from eve.traits import SymbolTableCreatorTrait
from eve.utils import noninstantiable


# from functional.iterator.util.sym_validation import validate_symbol_refs


@noninstantiable
class Node(eve.OpNode):
    def __str__(self) -> str:
        from functional.iterator.pretty_printer import pformat

        return pformat(self)


class Sym(Node):  # helper
    id: SymbolName = datamodels.field(converter=True)  # noqa: A003


@noninstantiable
class Expr(Node):
    ...


class Literal(Expr):
    value: str
    type: str  # noqa: A003


class NoneLiteral(Expr):
    _none_literal: int = 0


class OffsetLiteral(Expr):
    value: Union[int, str]


class AxisLiteral(Expr):
    value: str


class SymRef(Expr):
    id: SymbolRef = datamodels.field(converter=True)  # noqa: A003


class Lambda(Expr, SymbolTableCreatorTrait):
    params: List[Sym]
    expr: Expr


class FunCall(Expr):
    fun: Expr  # VType[Callable]
    args: List[Expr]


class FunctionDefinition(Node, SymbolTableCreatorTrait):
    id: SymbolName = datamodels.field(converter=True)  # noqa: A003
    params: List[Sym]
    expr: Expr


class StencilClosure(Node):
    domain: Expr
    stencil: Expr
    output: SymRef  # we could consider Expr for cases like make_tuple(out0,out1)
    inputs: List[SymRef]


BUILTINS = {
    "domain",
    "named_range",
    "lift",
    "make_tuple",
    "tuple_get",
    "reduce",
    "deref",
    "can_deref",
    "shift",
    "scan",
    "plus",
    "minus",
    "multiplies",
    "divides",
    "eq",
    "less",
    "greater",
    "if_",
    "not_",
    "and_",
    "or_",
}


class FencilDefinition(Node, eve.traits.SymbolTableTrait):
    id: SymbolName = datamodels.field(converter=True)  # noqa: A003

    function_definitions: List[FunctionDefinition]
    params: List[Sym]
    closures: List[StencilClosure]

    builtin_functions = [Sym(id=name) for name in BUILTINS]

    # _validate_symbol_refs = validate_symbol_refs()
