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

from . import datamodels, iterators, utils
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
from .type_definitions import NOTHING, IntEnum, StrEnum, ConstrainedStr


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


_SYMBOL_NAME_RE: Final = re.compile(r"^[a-zA-Z_]\w*$")


class SymbolName(ConstrainedStr, regex=_SYMBOL_NAME_RE):
    """String value containing a valid symbol name for typical programming conventions."""

    __slots__ = ()


class SymbolRef(ConstrainedStr, regex=_SYMBOL_NAME_RE):
    """Reference to a symbol name."""

    __slots__ = ()


# -- Fields --
# class ImplFieldMetadataDict(TypedDict, total=False):
#     info: pydantic.fields.FieldInfo


# NodeImplFieldMetadataDict = Dict[str, ImplFieldMetadataDict]


# class FieldKind(StrEnum):
#     INPUT = "input"
#     OUTPUT = "output"


# class FieldConstraintsDict(TypedDict, total=False):
#     vtype: Union[VType, Tuple[VType, ...]]


# class FieldMetadataDict(TypedDict, total=False):
#     constraints: FieldConstraintsDict
#     kind: FieldKind
#     definition: pydantic.fields.ModelField


# NodeChildrenMetadataDict = Dict[str, FieldMetadataDict]


_EVE_METADATA_KEY = "_EVE_META_"


# def field(
#     default: Any = NOTHING,
#     *,
#     default_factory: Optional[NoArgsCallable] = None,
#     kind: Optional[FieldKind] = None,
#     constraints: Optional[FieldConstraintsDict] = None,
#     schema_config: Dict[str, Any] = None,
# ) -> pydantic.fields.FieldInfo:
#     metadata = {}
#     for key in ["kind", "constraints"]:
#         value = locals()[key]
#         if value:
#             metadata[key] = value
#     kwargs = schema_config or {}
#     kwargs[_EVE_METADATA_KEY] = metadata

#     if default is NOTHING:
#         field_info = pydantic.Field(default_factory=default_factory, **kwargs)
#     else:
#         field_info = pydantic.Field(default, default_factory=default_factory, **kwargs)
#     assert isinstance(field_info, pydantic.fields.FieldInfo)

#     return field_info


# in_field = functools.partial(field, kind=FieldKind.INPUT)
# out_field = functools.partial(field, kind=FieldKind.OUTPUT)


# -- Models --
# class Model(pydantic.BaseModel):
#     class Config:
#         extra = "forbid"


# class FrozenModel(pydantic.BaseModel):
#     class Config:
#         allow_mutation = False


# -- Nodes --
_EVE_NODE_INTERNAL_SUFFIX = "__"
_EVE_NODE_IMPL_SUFFIX = "_"

AnyNode = TypeVar("AnyNode", bound="BaseNode")
ValueNode = Union[bool, bytes, int, float, str, IntEnum, StrEnum]
LeafNode = Union[AnyNode, ValueNode]
CollectionNode = Union[List[LeafNode], Dict[Any, LeafNode], Set[LeafNode]]
TreeNode = Union[AnyNode, CollectionNode]


# class NodeMetaclass(pydantic.main.ModelMetaclass):
#     """Custom metaclass for Node classes.

#     Customize the creation of Node classes adding Eve specific attributes.

#     """

#     @no_type_check
#     def __new__(mcls, name, bases, namespace, **kwargs):
#         # Optional preprocessing of class namespace before creation:
#         cls = super().__new__(mcls, name, bases, namespace, **kwargs)

#         # Postprocess created class:
#         # Add metadata class members
#         impl_fields_metadata = {}
#         children_metadata = {}
#         for name, model_field in cls.__fields__.items():
#             if not name.endswith(_EVE_NODE_INTERNAL_SUFFIX):
#                 if name.endswith(_EVE_NODE_IMPL_SUFFIX):
#                     impl_fields_metadata[name] = {"definition": model_field}
#                 else:
#                     children_metadata[name] = {
#                         "definition": model_field,
#                         **model_field.field_info.extra.get(_EVE_METADATA_KEY, {}),
#                     }

#         cls.__node_impl_fields__ = impl_fields_metadata
#         cls.__node_children__ = children_metadata

#         return cls


class BaseNode(datamodels.DataModel):
    """Base class representing an IR node.

    It is currently implemented as a pydantic Model with some extra features.

    Field values should be either:

        * builtin types: `bool`, `bytes`, `int`, `float`, `str`
        * enum.Enum types
        * other :class:`Node` subclasses
        * other :class:`pydantic.BaseModel` subclasses
        * supported collections (:class:`List`, :class:`Dict`, :class:`Set`)
            of any of the previous items

    Field naming scheme:

        * Field names starting with "_" are ignored by pydantic and Eve. They
            will not be considered as `fields` and thus none of the pydantic
            features will work (type coercion, validators, etc.).
        * Field names ending with "__" are reserved for internal Eve use and
            should NOT be defined by regular users. All pydantic features will
            work on these fields anyway but they will be invisible for Eve users.
        * Field names ending with "_" are considered implementation fields
            not children nodes. They are intended to be defined by users when needed,
            typically to cache derived, non-essential information on the node.

    """

    def iter_impl_fields(self) -> Generator[Tuple[str, Any], None, None]:
        for name in self.__node_impl_fields__.keys():
            yield name, getattr(self, name)

    def iter_children(self) -> Generator[Tuple[str, Any], None, None]:
        for name in self.__node_children__.keys():
            yield name, getattr(self, name)

    def iter_children_values(self) -> Generator[Any, None, None]:
        for name in self.__node_children__.keys():
            yield getattr(self, name)

    def iter_tree_pre(self) -> utils.XIterable:
        return iterators.iter_tree_pre(self)

    def iter_tree_post(self) -> utils.XIterable:
        return iterators.iter_tree_post(self)

    def iter_tree_levels(self) -> utils.XIterable:
        return iterators.iter_tree_levels(self)

    iter_tree = iter_tree_pre

    # class Config(Model.Config):
    #     pass


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
