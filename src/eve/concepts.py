# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
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

"""Definitions of basic Eve concepts."""


from __future__ import annotations

import abc
import ast
import re

from attr import frozen

from . import datamodels, trees, type_definitions, utils
from .datamodels import validators as dm_validators
from .extended_typing import (
    Any,
    Dict,
    Final,
    FrozenDict,
    FrozenList,
    Generator,
    Generic,
    List,
    Mapping,
    NoArgsCallable,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
    no_type_check,
)
from .type_definitions import NOTHING, ConstrainedStr, IntEnum, StrEnum


_SYMBOL_NAME_RE: Final = re.compile(r"^[a-zA-Z_]\w*$")


class SymbolName(ConstrainedStr, regex=_SYMBOL_NAME_RE):
    """String value containing a valid symbol name for typical programming conventions."""

    __slots__ = ()


class SymbolRef(ConstrainedStr, regex=_SYMBOL_NAME_RE):
    """Reference to a symbol name."""

    __slots__ = ()


@datamodels.datamodel(slots=True, frozen=True)
class SourceLocation:
    """Source code location (line, column, source)."""

    line: int = datamodels.field(validator=dm_validators.ge(1))
    column: int = datamodels.field(validator=dm_validators.ge(1))
    source: str
    end_line: Optional[int] = datamodels.field(
        validator=dm_validators.optional(dm_validators.ge(1))
    )
    end_column: Optional[int] = datamodels.field(
        validator=dm_validators.optional(dm_validators.ge(1))
    )

    @classmethod
    def from_AST(cls, ast_node: ast.AST, source: Optional[str] = None) -> SourceLocation:
        if (
            not isinstance(ast_node, ast.AST)
            or getattr(ast_node, "lineno", None) is None
            or getattr(ast_node, "col_offset", None) is None
        ):
            raise ValueError(
                f"Passed AST node '{ast_node}' does not contain a valid source location."
            )
        if source is None:
            source = f"<ast.{type(ast_node).__name__} at 0x{id(ast_node):x}>"
        return cls(
            ast_node.lineno,
            ast_node.col_offset + 1,
            source,
            end_line=ast_node.end_lineno,
            end_column=ast_node.end_col_offset + 1 if ast_node.end_col_offset is not None else None,
        )

    def __init__(
        self,
        line: int,
        column: int,
        source: str,
        *,
        end_line: Optional[int] = None,
        end_column: Optional[int] = None,
    ) -> None:
        assert end_column is None or end_line is not None
        self.__auto_init__(
            line=line, column=column, source=source, end_line=end_line, end_column=end_column
        )

    def __str__(self) -> str:
        src = self.source or ""

        end_part = ""
        if self.end_line is not None:
            end_part += f" to Line {self.end_line}"
        if self.end_column is not None:
            end_part += f", Col {self.end_column}"

        return f"<'{src}': Line {self.line}, Col {self.column}{end_part}>"


@datamodels.datamodel(slots=True, frozen=True)
class SourceLocationGroup:
    """A group of merged source code locations (with optional info)."""

    locations: Tuple[SourceLocation, ...] = datamodels.field(validator=dm_validators.non_empty())
    context: Optional[Union[str, Tuple[str, ...]]]

    def __init__(
        self, *locations: SourceLocation, context: Optional[Union[str, Tuple[str, ...]]] = None
    ) -> None:
        self.__auto_init__(locations=locations, context=context)

    def __str__(self) -> str:
        locs = ", ".join(str(loc) for loc in self.locations)
        context = f"#{self.context}#" if self.context else ""
        return f"<{context}[{locs}]>"


AnySourceLocation = Union[SourceLocation, SourceLocationGroup]


AnyNode = TypeVar("AnyNode", bound="Node")
ValueNode = Union[bool, bytes, int, float, str, IntEnum, StrEnum]
LeafNode = Union[AnyNode, ValueNode]
CollectionNode = Union[List[LeafNode], Dict[Any, LeafNode], Set[LeafNode]]
TreeNode = Union[AnyNode, CollectionNode]


class Node(datamodels.DataModel, trees.Tree):
    """Base class representing a node in a syntax tree.

    Implemented as a :class:`eve.datamodels.DataModel` with some extra features.

    Field values should be either:

        * builtin types: `bool`, `bytes`, `int`, `float`, `str`
        * enum.Enum types
        * other :class:`Node` subclasses
        * other :class:`eve.datamodels.DataModel` subclasses
        * supported collections (:class:`List`, :class:`Dict`, :class:`Set`)
            of any of the previous items

    """

    @property
    def annex(self) -> utils.Namespace:
        if "__annex__" not in self.__dict__:
            self.__dict__["__annex__"] = utils.Namespace()
        return self.__dict__["__annex__"]

    def iter_children_items(self) -> Generator[Tuple[str, Any], None, None]:
        for name in self.__fields__:
            yield name, getattr(self, name)

    def iter_children_values(self) -> Generator[Any, None, None]:
        for name in self.__fields__:
            yield getattr(self, name)

    pre_iter_tree_items = trees.pre_walk_tree_items
    pre_iter_tree_values = trees.pre_walk_tree_values

    post_iter_tree_items = trees.post_walk_tree_items
    post_iter_tree_values = trees.post_walk_tree_values

    iter_tree_items = trees.walk_tree_items
    iter_tree_values = trees.walk_tree_values


class FrozenNode(Node):
    pass


_T = TypeVar("_T")


class Block(Node, Generic[_T]):
    items: List[_T]

    def __getitem__(self, /, item: int) -> _T:
        return self.items[item]


class FrozenBlock(Block[_T]):
    items: FrozenList[_T]


_KeyT = TypeVar("_KeyT")
_ValueT = TypeVar("_ValueT")


class Table(Node, Generic[_KeyT, _ValueT]):
    items: Dict[_KeyT, _ValueT]

    def __getitem__(self, /, key: _KeyT) -> _ValueT:
        return self.items[key]


class FrozenTable(Table[_KeyT, _ValueT]):
    items: FrozenDict[_KeyT, _ValueT]


# -- Misc --
class VType(datamodels.FrozenModel):

    # VType fields
    #: Unique name
    name: str

    # def __init__(self, name: str) -> None:
    #     super().__auto_init__(name=name)
