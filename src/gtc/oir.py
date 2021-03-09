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

"""
Optimizable Intermediate Representation (working title).

OIR represents a computation at the level of GridTools stages and multistages,
e.g. stage merging, staged computations to compute-on-the-fly, cache annotations, etc.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from eve.datamodels import Attribute, DataModel, property_field, root_validator, validator
from eve import Str, SymbolName, SymbolRef, SymbolTableTrait
from gtc import common
from gtc.common import AxisBound, LocNode


class Expr(common.Expr, instantiable=False):
    dtype: Optional[common.DataType]


class Stmt(common.Stmt, instantiable=False):
    pass


class Literal(common.Literal, Expr):  # type: ignore
    pass


class ScalarAccess(common.ScalarAccess, Expr):  # type: ignore
    pass


class FieldAccess(common.FieldAccess, Expr):  # type: ignore
    pass


class AssignStmt(common.AssignStmt[Union[ScalarAccess, FieldAccess], Expr], Stmt):
    @validator("left")
    def no_horizontal_offset_in_assignment(self, v: Union[ScalarAccess, FieldAccess]) -> None:
        if isinstance(v, FieldAccess) and (v.offset.i != 0 or v.offset.j != 0):
            raise ValueError("Lhs of assignment must not have a horizontal offset.")

    _dtype_validation = common.assign_stmt_dtype_validation(strict=True)


# TODO(havogt) consider introducing BlockStmt
# class BlockStmt(common.BlockStmt[Stmt], Stmt):
#     pass

# TODO(havogt) should we have an IfStmt or is masking the final solution?
# class IfStmt(common.IfStmt[List[Stmt], Expr], Stmt):  # TODO replace List[Stmt] by BlockStmt?
#     pass


class UnaryOp(common.UnaryOp[Expr], Expr):
    pass


class BinaryOp(common.BinaryOp[Expr], Expr):
    _dtype_propagation = common.binary_op_dtype_propagation(strict=True)


class TernaryOp(common.TernaryOp[Expr], Expr):
    _dtype_propagation = common.ternary_op_dtype_propagation(strict=True)


class Cast(common.Cast[Expr], Expr):  # type: ignore
    pass


class NativeFuncCall(common.NativeFuncCall[Expr], Expr):
    _dtype_propagation = common.native_func_call_dtype_propagation(strict=True)


class Decl(LocNode, instantiable=False):
    name: SymbolName
    dtype: common.DataType


class FieldDecl(Decl):
    # TODO dimensions
    pass


class ScalarDecl(Decl):
    pass


class LocalScalar(Decl):
    pass


class Temporary(Decl):
    pass


class HorizontalExecution(LocNode):
    body: List[Stmt]
    mask: Optional[Expr]
    declarations: List[LocalScalar]

    @validator("mask")
    def mask_is_boolean_field_expr(self, v: Optional[Expr]) -> None:
        if v:
            if v.dtype != common.DataType.BOOL:
                raise ValueError("Mask must be a boolean expression.")


class Interval(LocNode):
    start: AxisBound
    end: AxisBound

    @root_validator
    def check(cls, instance: Interval) -> None:
        start, end = instance.start, instance.end
        if start.level == common.LevelMarker.END and end.level == common.LevelMarker.START:
            raise ValueError("Start level must be smaller or equal end level")
        if start.level == end.level and not start.offset < end.offset:
            raise ValueError(
                "Start offset must be smaller than end offset if start and end levels are equal"
            )


class CacheDesc(LocNode):
    name: SymbolRef


class IJCache(CacheDesc):
    pass


class KCache(CacheDesc):
    fill: bool
    flush: bool


class VerticalLoopSection(LocNode):
    interval: Interval
    horizontal_executions: List[HorizontalExecution]


class VerticalLoop(LocNode):
    loop_order: common.LoopOrder
    sections: List[VerticalLoopSection]
    caches: List[CacheDesc]

    @validator("sections")
    def nonempty_loop(self, v: List[VerticalLoopSection]) -> None:
        if not v:
            raise ValueError("Empty vertical loop is not allowed")

    @root_validator
    def valid_section_intervals(cls, instance: VerticalLoop) -> None:
        loop_order, sections = instance["loop_order"], instance["sections"]
        starts, ends = zip(*((s.interval.start, s.interval.end) for s in sections))
        if loop_order == common.LoopOrder.BACKWARD:
            starts, ends = starts[:-1], ends[1:]
        else:
            starts, ends = starts[1:], ends[:-1]

        if not all(
            start.level == end.level and start.offset == end.offset
            for start, end in zip(starts, ends)
        ):
            raise ValueError("Loop intervals not contiguous or in wrong order")
        return instance


class Stencil(LocNode, SymbolTableTrait):
    name: Str
    # TODO: fix to be List[Union[ScalarDecl, FieldDecl]]
    params: List[Decl]
    vertical_loops: List[VerticalLoop]
    declarations: List[Temporary]

    _validate_dtype_is_set = common.validate_dtype_is_set()
    _validate_symbol_refs = common.validate_symbol_refs()
