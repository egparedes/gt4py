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

import ast
import re
import types

from . import datamodels, iterators, type_definitions, utils
from .datamodels import validators as dm_validators
from .extended_typing import (
    Any,
    Dict,
    Final,
    Generator,
    List,
    NoArgsCallable,
    Optional,
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


_EVE_METADATA_KEY = "_EVE_META_"

AnyNode = TypeVar("AnyNode", bound="BaseNode")
ValueNode = Union[bool, bytes, int, float, str, IntEnum, StrEnum]
LeafNode = Union[AnyNode, ValueNode]
CollectionNode = Union[List[LeafNode], Dict[Any, LeafNode], Set[LeafNode]]
TreeNode = Union[AnyNode, CollectionNode]


class BaseNode(datamodels.DataModel):
    """Base class representing an IR node.

    Implemented as a :class:`eve.datamodels.DataModel` with some extra features.

    Field values should be either:

        * builtin types: `bool`, `bytes`, `int`, `float`, `str`
        * enum.Enum types
        * other :class:`Node` subclasses
        * other :class:`eve.datamodels.DataModel` subclasses
        * supported collections (:class:`List`, :class:`Dict`, :class:`Set`)
            of any of the previous items

    """

    @type_definitions.classproperty
    def __fields__(cls):
        return cls.__datamodel_fields__

    @property
    def annex(self) -> utils.Namespace:
        return self.__dict__.setdefault("__annex__", utils.Namespace())

    @property
    def annex(self) -> utils.Namespace:
        return self.__dict__.setdefault("__annex__", utils.Namespace())

    def iter_annex(self) -> Generator[Tuple[str, Any], None, None]:
        names = self.__annex__.__dict__.keys()
        for name in names:
            yield name, getattr(self, name)

    def iter_children(self) -> Generator[Tuple[str, Any], None, None]:
        field_names = self.__datamodel_fields__.keys()
        for name in field_names:
            yield name, getattr(self, name)

    def iter_children_values(self) -> Generator[Any, None, None]:
        field_names = self.__datamodel_fields__.keys()
        for name in field_names:
            yield getattr(self, name)

    def iter_tree_pre(self) -> utils.XIterable:
        return iterators.iter_tree_pre(self)

    def iter_tree_post(self) -> utils.XIterable:
        return iterators.iter_tree_post(self)

    def iter_tree_levels(self) -> utils.XIterable:
        return iterators.iter_tree_levels(self)

    iter_tree = iter_tree_pre


# class GenericNode(BaseNode, pydantic.generics.GenericModel):
#     pass


class Node(BaseNode):
    """Default public name for a base node class."""

    pass


class FrozenNode(Node, frozen=True):
    """Default public name for an inmutable base node class."""


# -- Misc --
class VType(datamodels.FrozenModel):

    # VType fields
    #: Unique name
    name: str

    # def __init__(self, name: str) -> None:
    #     super().__auto_init__(name=name)
