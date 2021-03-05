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

import functools

import pydantic
import pydantic.generics

from . import datamodels, iterators, utils
from .datamodels import field
from .type_definitions import NOTHING, IntEnum, StrEnum
from .typingx import (
    Any,
    AnyNoArgCallable,
    ClassVar,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
    no_type_check,
)


_EVE_METADATA_KEY = "_EVE_META_"


# -- Nodes --
_EVE_NODE_INTERNAL_SUFFIX = "__"
_EVE_NODE_IMPL_SUFFIX = "_"

AnyNode = TypeVar("AnyNode", bound="BaseNode")
ValueNode = Union[bool, bytes, int, float, str, IntEnum, StrEnum]
LeafNode = Union[AnyNode, ValueNode]
CollectionNode = Union[List[LeafNode], Dict[Any, LeafNode], Set[LeafNode]]
TreeNode = Union[AnyNode, CollectionNode]


class BaseNode(datamodels.DataModel):
    """Base class representing an IR node.

    Implemented as a Data Model with some extra features.

    Field naming scheme:

        * Field names starting with "_" are ignored by Eve. They
            will not be considered as `fields` and thus none of the datamodel
            features will work (validators, etc.).
        * Field names ending with "_" are considered implementation fields
            not children nodes. They are intended to be defined by users when needed,
            typically to cache derived, non-essential information on the node.
        * Field names ending with "__" are reserved for internal Eve use and
            should **NOT** be defined by regular users.
    """

    def __init_subclass__(cls, /, **kwargs: Any) -> None:
        assert kwargs.pop("skip_private", True)
        return super().__init_subclass__(skip_private=True, **kwargs)

    # Node fields
    #: Unique node-id (implementation field)
    id_: str = field(default_factory=utils.UIDGenerator.sequential_id)

    def iter_impl_fields(self) -> Generator[Tuple[str, Any], None, None]:
        for name in self.__datamodel_fields__.keys():
            if name.endswith(_EVE_NODE_IMPL_SUFFIX) and not name.endswith(
                _EVE_NODE_INTERNAL_SUFFIX
            ):
                yield name, getattr(self, name)

    def iter_children(self) -> Generator[Tuple[str, Any], None, None]:
        for name in self.__datamodel_fields__.keys():
            if not (
                name.endswith(_EVE_NODE_IMPL_SUFFIX) or name.endswith(_EVE_NODE_INTERNAL_SUFFIX)
            ):
                yield name, getattr(self, name)

    def iter_children_values(self) -> Generator[Any, None, None]:
        for _, node in self.iter_children():
            yield node

    def iter_tree_pre(self) -> utils.XIterator:
        return iterators.iter_tree_pre(self)

    def iter_tree_post(self) -> utils.XIterator:
        return iterators.iter_tree_post(self)

    def iter_tree_levels(self) -> utils.XIterator:
        return iterators.iter_tree_levels(self)

    def iter_tree(
        self, *, traversal_order: Optional[iterators.TraversalOrder] = None
    ) -> utils.XIterator:
        return iterators.iter_tree_levels(
            self, traversal_order=traversal_order or iterators.TraversalOrder.PRE_ORDER
        )

    def to_dict(self) -> Dict[str, Any]:
        return datamodels.asdict(self)


class Node(BaseNode):
    """Default public name for a base node class."""

    pass


class FrozenNode(BaseNode, datamodels.FrozenDataModel):
    """Default public name for an inmutable base node class."""

    pass


# -- Misc --
class VType(datamodels.FrozenDataModel):

    # VType fields
    #: Unique name
    name: str
