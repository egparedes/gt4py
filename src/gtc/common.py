# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import enum
from typing import Any, ClassVar, Dict, Generic, List, Optional, Type, TypeVar, Union, cast

from eve import (
    IntEnum,
    Node,
    NodeVisitor,
    SourceLocation,
    Str,
    StrEnum,
    SymbolTableTrait,
)
from eve.datamodels import Attribute, DataModel, field, property_field, root_validator, validator
from eve import exceptions as eve_exceptions
from eve.type_definitions import SymbolRef
from eve.typingx import RootValidatorType, RootValidatorValuesType
from gtc.utils import flatten_list


class GTCPreconditionError(eve_exceptions.EveError, RuntimeError):
    message_template = "GTC pass precondition error: [{info}]"

    def __init__(self, *, expected: str, **kwargs: Any) -> None:
        super().__init__(expected=expected, **kwargs)  # type: ignore


class GTCPostconditionError(eve_exceptions.EveError, RuntimeError):
    message_template = "GTC pass postcondition error: [{info}]"

    def __init__(self, *, expected: str, **kwargs: Any) -> None:
        super().__init__(expected=expected, **kwargs)  # type: ignore


class AssignmentKind(StrEnum):
    """Kind of assignment: plain or combined with operations."""

    PLAIN = "="
    ADD = "+="
    SUB = "-="
    MUL = "*="
    DIV = "/="


@enum.unique
class UnaryOperator(StrEnum):
    """Unary operator indentifier."""

    POS = "+"
    NEG = "-"
    NOT = "not"


@enum.unique
class ArithmeticOperator(StrEnum):
    """Arithmetic operators."""

    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"


@enum.unique
class ComparisonOperator(StrEnum):
    """Comparison operators."""

    GT = ">"
    LT = "<"
    GE = ">="
    LE = "<="
    EQ = "=="
    NE = "!="


@enum.unique
class LogicalOperator(StrEnum):
    """Logical operators."""

    AND = "and"
    OR = "or"


@enum.unique
class DataType(IntEnum):
    """Data type identifier."""

    # IDs from gt4py
    INVALID = -1
    AUTO = 0
    DEFAULT = 1
    BOOL = 10
    INT8 = 11
    INT16 = 12
    INT32 = 14
    INT64 = 18
    FLOAT32 = 104
    FLOAT64 = 108


@enum.unique
class LoopOrder(StrEnum):
    """Loop order identifier."""

    PARALLEL = "parallel"
    FORWARD = "forward"
    BACKWARD = "backward"


@enum.unique
class BuiltInLiteral(StrEnum):
    MAX_VALUE = "max"
    MIN_VALUE = "min"
    ZERO = "zero"
    ONE = "one"
    TRUE = "true"
    FALSE = "false"


@enum.unique
class NativeFunction(StrEnum):
    ABS = "abs"
    MIN = "min"
    MAX = "max"
    MOD = "mod"

    SIN = "sin"
    COS = "cos"
    TAN = "tan"
    ARCSIN = "arcsin"
    ARCCOS = "arccos"
    ARCTAN = "arctan"

    SQRT = "sqrt"
    POW = "pow"
    EXP = "exp"
    LOG = "log"

    ISFINITE = "isfinite"
    ISINF = "isinf"
    ISNAN = "isnan"
    FLOOR = "floor"
    CEIL = "ceil"
    TRUNC = "trunc"

    IR_OP_TO_NUM_ARGS: ClassVar[Dict[NativeFunction, int]]

    @property
    def arity(self) -> int:
        return self.IR_OP_TO_NUM_ARGS[self]


NativeFunction.IR_OP_TO_NUM_ARGS = {
    NativeFunction.ABS: 1,
    NativeFunction.MIN: 2,
    NativeFunction.MAX: 2,
    NativeFunction.MOD: 2,
    NativeFunction.SIN: 1,
    NativeFunction.COS: 1,
    NativeFunction.TAN: 1,
    NativeFunction.ARCSIN: 1,
    NativeFunction.ARCCOS: 1,
    NativeFunction.ARCTAN: 1,
    NativeFunction.SQRT: 1,
    NativeFunction.POW: 2,
    NativeFunction.EXP: 1,
    NativeFunction.LOG: 1,
    NativeFunction.ISFINITE: 1,
    NativeFunction.ISINF: 1,
    NativeFunction.ISNAN: 1,
    NativeFunction.FLOOR: 1,
    NativeFunction.CEIL: 1,
    NativeFunction.TRUNC: 1,
}


@enum.unique
class LevelMarker(StrEnum):
    START = "start"
    END = "end"


@enum.unique
class ExprKind(IntEnum):
    SCALAR = 0
    FIELD = 1


class LocNode(Node):
    loc: Optional[SourceLocation] = field(default=None, compare=False)


class Expr(LocNode, instantiable=False):
    """
    Expression base class.

    All expressions have
    - an optional `dtype`
    - an expression `kind` (scalar or field)
    """

    dtype: Optional[DataType]
    kind: ExprKind


class Stmt(LocNode, instantiable=False):
    pass


def verify_condition_is_boolean(parent_node_cls: Node, cond: Expr) -> Expr:
    if cond.dtype and cond.dtype is not DataType.BOOL:
        raise ValueError("Condition in `{}` must be boolean.".format(parent_node_cls.__name__))
    return cond


def verify_and_get_common_dtype(
    node_cls: Type[Node], values: List[Expr], *, strict: bool = True
) -> Optional[DataType]:
    assert len(values) > 0
    if all(v.dtype is not None for v in values):
        dtypes: List[DataType] = [v.dtype for v in values]  # type: ignore # guaranteed to be not None
        dtype = dtypes[0]
        if strict:
            if all(dt == dtype for dt in dtypes):
                return dtype
            else:
                raise ValueError(
                    "Type mismatch in `{}`. Types are ".format(node_cls.__name__)
                    + ", ".join(dt.name for dt in dtypes)
                )
        else:
            # upcasting
            return max(dt for dt in dtypes)
    else:
        return None


def compute_kind(values: List[Expr]) -> ExprKind:
    if any(v.kind == ExprKind.FIELD for v in values):
        return cast(ExprKind, ExprKind.FIELD)  # see https://github.com/GridTools/gtc/issues/100
    else:
        return cast(ExprKind, ExprKind.SCALAR)  # see https://github.com/GridTools/gtc/issues/100


class Literal(Node):
    # TODO(havogt) reconsider if `str` is a good representation for value,
    # maybe it should be Union[float,int,str] etc?
    value: Union[BuiltInLiteral, Str]
    dtype: DataType
    kind: ExprKind = cast(
        ExprKind, ExprKind.SCALAR
    )  # cast shouldn't be required, see https://github.com/GridTools/gtc/issues/100


StmtT = TypeVar("StmtT", bound=Stmt)
ExprT = TypeVar("ExprT", bound=Expr)
TargetT = TypeVar("TargetT", bound=Expr)


class CartesianOffset(Node):
    i: int
    j: int
    k: int

    @classmethod
    def zero(cls) -> CartesianOffset:
        return cls(i=0, j=0, k=0)

    def to_dict(self) -> Dict[str, int]:
        return {"i": self.i, "j": self.j, "k": self.k}


class ScalarAccess(LocNode):
    name: SymbolRef
    kind = ExprKind.SCALAR


class FieldAccess(LocNode):
    name: SymbolRef
    offset: CartesianOffset
    kind = ExprKind.FIELD

    @classmethod
    def centered(cls, *, name: str, loc: SourceLocation = None) -> "FieldAccess":
        return cls(name=name, loc=loc, offset=CartesianOffset.zero())


class BlockStmt(Node, SymbolTableTrait, Generic[StmtT]):
    body: List[StmtT]


class IfStmt(Node, Generic[StmtT, ExprT]):
    """
    Generic if statement.

    Verifies that `cond` is a boolean expr (if `dtype` is set).
    """

    cond: ExprT
    true_branch: StmtT
    false_branch: Optional[StmtT]

    @validator("cond")
    def condition_is_boolean(self, cond: Expr) -> None:
        verify_condition_is_boolean(self.__class__, cond)


class AssignStmt(Node, Generic[TargetT, ExprT]):
    left: TargetT
    right: ExprT


def assign_stmt_dtype_validation(*, strict: bool) -> RootValidatorType:
    def _impl(cls: Type[Node], assign_stmt: AssignStmt) -> None:
        verify_and_get_common_dtype(cls, [assign_stmt.left, assign_stmt.right], strict=strict)

    return root_validator(_impl)


class UnaryOp(Node, Generic[ExprT]):
    """
    Generic unary operation with type propagation.

    The generic `UnaryOp` already contains logic for type propagation.
    """

    op: UnaryOperator
    expr: ExprT

    @root_validator
    def dtype_propagation(cls, instance: UnaryOp) -> None:
        instance.dtype = instance.expr.dtype

    @root_validator
    def kind_propagation(cls, instance: UnaryOp) -> None:
        instance.kind = instance.expr.kind

    @root_validator
    def op_to_dtype_check(cls, instance: UnaryOp) -> None:
        if instance.expr.dtype:
            if instance.op == UnaryOperator.NOT:
                if not instance.expr.dtype == DataType.BOOL:
                    raise ValueError("Unary operator `NOT` only allowed with boolean expression.")
            else:
                if instance.expr.dtype == DataType.BOOL:
                    raise ValueError(
                        f"Unary operator `{instance.name}` not allowed with boolean expression."
                    )


class BinaryOp(Node, Generic[ExprT]):
    """Generic binary operation with type propagation.

    The generic BinaryOp already contains logic for
    - strict type checking if the `dtype` for `left` and `right` is set.
    - type propagation (taking `operator` type into account).
    """

    # consider parametrizing on op
    op: Union[ArithmeticOperator, ComparisonOperator, LogicalOperator]
    left: ExprT
    right: ExprT
    kind: ExprKind = property_field(lambda self: compute_kind([self.left, self.right]))
    # @root_validator(pre=True)
    # def kind_propagation(cls, values: RootValidatorValuesType) -> RootValidatorValuesType:
    #     values["kind"] = compute_kind([values["left"], values["right"]])
    #     return values


def binary_op_dtype_propagation(*, strict: bool) -> RootValidatorType:
    def _impl(cls: Type[Node], instance: Node) -> None:
        common_dtype = verify_and_get_common_dtype(
            cls, [instance.left, instance.right], strict=strict
        )

        if common_dtype:
            if isinstance(instance.op, ArithmeticOperator):
                if common_dtype is not DataType.BOOL:
                    instance.dtype = common_dtype
                else:
                    raise ValueError("Boolean expression is not allowed with arithmetic operation.")
            elif isinstance(instance.op, LogicalOperator):
                if common_dtype is DataType.BOOL:
                    instance.dtype = DataType.BOOL
                else:
                    raise ValueError("Arithmetic expression is not allowed in boolean operation.")
            elif isinstance(instance.op, ComparisonOperator):
                instance.dtype = DataType.BOOL

    return root_validator(_impl)


class TernaryOp(Node, Generic[ExprT]):
    """
    Generic ternary operation with type propagation.

    The generic TernaryOp already contains logic for
    - strict type checking if the `dtype` for `true_expr` and `false_expr` is set.
    - type checking for `cond`
    - type propagation.
    """

    # consider parametrizing cond type and expr separately
    cond: ExprT
    true_expr: ExprT
    false_expr: ExprT

    @property_field()
    def kind(self) -> ExprKind:
        return compute_kind([self.true_expr, self.false_expr])

    @validator("cond")
    def condition_is_boolean(self, cond: ExprT) -> None:
        verify_condition_is_boolean(self.__class__, cond)

    # @root_validator(pre=True)
    # def kind_propagation(cls, values: RootValidatorValuesType) -> RootValidatorValuesType:
    #     values["kind"] = compute_kind([values["true_expr"], values["false_expr"]])
    #     return values


def ternary_op_dtype_propagation(*, strict: bool) -> RootValidatorType:
    def _impl(cls: Type[Node], instance: Node) -> None:
        common_dtype = verify_and_get_common_dtype(
            cls, [instance.true_expr, instance.false_expr], strict=strict
        )
        if common_dtype:
            instance.dtype = common_dtype

    return root_validator(_impl)


class Cast(Node, Generic[ExprT]):
    dtype: DataType
    expr: ExprT

    @property_field
    def kind(self) -> ExprKind:
        return compute_kind([self.expr])

    # @root_validator(pre=True)
    # def kind_propagation(cls, values: RootValidatorValuesType) -> RootValidatorValuesType:
    #     values["kind"] = compute_kind([values["expr"]])
    #     return values


class NativeFuncCall(Node, Generic[ExprT]):
    func: NativeFunction
    args: List[ExprT]

    @property_field
    def kind(self) -> ExprKind:
        return compute_kind([self.args])

    # @root_validator(pre=True)
    # def kind_propagation(cls, values: RootValidatorValuesType) -> RootValidatorValuesType:
    #     values["kind"] = compute_kind(values["args"])
    #     return values

    @root_validator
    def arity_check(cls, instance: NativeFuncCall) -> None:
        if instance.func.arity != len(instance.args):
            raise ValueError(
                f"'{instance.func}'' accepts {instance.arity} arguments, {instance.args} were passed."
            )


def native_func_call_dtype_propagation(*, strict: bool = True) -> RootValidatorType:
    def _impl(cls: Type[Node], instance: Node) -> None:
        # assumes all NativeFunction args have a common dtype
        common_dtype = verify_and_get_common_dtype(cls, instance.args, strict=strict)
        if common_dtype:
            instance.dtype = common_dtype

    return root_validator(_impl)


def validate_dtype_is_set() -> RootValidatorType:
    def _impl(cls: Type[Node], instance: Node) -> None:
        dtype_nodes: List[Node] = []
        for v in flatten_list(instance.to_dict().values()):
            if isinstance(v, Node):
                dtype_nodes.extend(v.iter_tree().if_hasattr("dtype"))

        nodes_without_dtype = []
        for node in dtype_nodes:
            if not node.dtype:
                nodes_without_dtype.append(node)

        if len(nodes_without_dtype) > 0:
            raise ValueError(f"Nodes without dtype detected {nodes_without_dtype}")

    return root_validator(_impl)


def validate_symbol_refs() -> RootValidatorType:
    """Works only, if only the root node has a symbol table."""

    def _impl(cls: Type[Node], instance: Node) -> None:
        class SymtableValidator(NodeVisitor):
            def __init__(self) -> None:
                self.missing_symbols: List[str] = []

            def visit_Node(self, node: Node, *, symtable: Dict[str, Any], **kwargs: Any) -> None:
                for name, metadata in node.__node_children__.items():
                    if isinstance(metadata["definition"].type_, type) and issubclass(
                        metadata["definition"].type_, SymbolRef
                    ):
                        if getattr(node, name) not in symtable:
                            self.missing_symbols.append(getattr(node, name))

                if isinstance(node, SymbolTableTrait):
                    symtable = {**symtable, **node.symtable_}
                self.generic_visit(node, symtable=symtable, **kwargs)

            @classmethod
            def apply(cls, node: Node, *, symtable: Dict[str, Any]) -> List[str]:
                instance = cls()
                instance.visit(node, symtable=symtable)
                return instance.missing_symbols

        missing_symbols = []
        for v in instance.to_dict().values():
            missing_symbols.extend(SymtableValidator.apply(v, symtable=instance.symtable_))

        if len(missing_symbols) > 0:
            raise ValueError(f"Symbols {missing_symbols} not found.")

    return root_validator(_impl)


class AxisBound(Node):
    level: LevelMarker
    offset: int = 0

    @classmethod
    def from_start(cls, offset: int) -> AxisBound:
        return cls(level=LevelMarker.START, offset=offset)

    @classmethod
    def from_end(cls, offset: int) -> AxisBound:
        return cls(level=LevelMarker.END, offset=offset)

    @classmethod
    def start(cls) -> AxisBound:
        return cls.from_start(0)

    @classmethod
    def end(cls) -> AxisBound:
        return cls.from_end(0)
