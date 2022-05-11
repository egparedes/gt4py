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

"""Iterator utils."""


from __future__ import annotations

import abc
from enum import Enum

from . import concepts, utils
from .extended_typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)


class TraversalOrder(Enum):
    PRE_ORDER = "pre"
    POST_ORDER = "post"
    LEVELS_ORDER = "levels"


Key = Union[int, str]


def _pre_walk_items(node: Any, *, __key__: Optional[Key] = None) -> Iterable[Tuple[Key, Any]]:
    """Create a pre-order tree traversal iterator of (key, value) pairs."""
    yield __key__, node
    if (iter_child_items := getattr(node, "iter_child_items", None)) is not None:
        for key, child in iter_child_items():
            yield from _pre_walk_items(child, __key__=key)


def _pre_walk_values(node: Any) -> Iterable[Tuple[Any]]:
    """Create a pre-order tree traversal iterator of values."""
    yield node
    if (iter_child_values := getattr(node, "iter_child_values", None)) is not None:
        for child in iter_child_values():
            yield from _pre_walk_values(child)


pre_walk_items = utils.as_xiter(_pre_walk_items)
pre_walk_values = utils.as_xiter(_pre_walk_values)


def _post_walk_items(node: Any, *, __key__: Optional[Key] = None) -> Iterable[Tuple[Key, Any]]:
    """Create a post-order tree traversal iterator of (key, value) pairs."""
    yield __key__, node
    if (iter_child_items := getattr(node, "iter_child_items", None)) is not None:
        for key, child in iter_child_items():
            yield from _post_walk_items(child, __key__=key)


def _post_walk_values(node: Any) -> Iterable[Tuple[Any]]:
    """Create a post-order tree traversal iterator of values."""
    if (iter_child_values := getattr(node, "iter_child_values", None)) is not None:
        for child in iter_child_values():
            yield from _post_walk_values(child)
    yield node


post_walk_items = utils.as_xiter(_post_walk_items)
post_walk_values = utils.as_xiter(_post_walk_values)


def _bfs_walk_items(
    node: Any, *, __key__: Optional[Any] = None, __queue__: Optional[List] = None
) -> Iterable[Tuple[Key, Any]]:
    """Create a tree traversal iterator of (key, value) pairs by tree levels (Breadth-First Search)."""
    __queue__ = __queue__ or []
    yield __key__, node
    if (iter_child_items := getattr(node, "iter_child_items", None)) is not None:
        __queue__.extend(iter_child_items())
    if __queue__:
        key, child = __queue__.pop(0)
        yield from _bfs_walk_items(child, __key__=key, __queue__=__queue__)


def _bfs_walk_values(node: Any, *, __queue__: Optional[List] = None) -> Iterable[Tuple[Key, Any]]:
    """Create a tree traversal iterator of values by tree levels (Breadth-First Search)."""
    __queue__ = __queue__ or []
    yield node
    if (iter_child_values := getattr(node, "iter_child_values", None)) is not None:
        __queue__.extend(iter_child_values())
    if __queue__:
        child = __queue__.pop(0)
        yield from _bfs_walk_values(child, __queue__=__queue__)


bfs_walk_items = utils.as_xiter(_bfs_walk_items)
bfs_walk_values = utils.as_xiter(_bfs_walk_values)


def walk_items(
    node: TreeNode, traversal_order: TraversalOrder = TraversalOrder.PRE_ORDER
) -> utils.XIterable[Tuple[Key, Any]]:
    """Create a tree traversal iterator of (key, value) pairs.

    Args:
        traversal_order: Tree nodes traversal order.
    """
    if traversal_order is traversal_order.PRE_ORDER:
        return pre_walk_items(node=node)
    elif traversal_order is traversal_order.POST_ORDER:
        return post_walk_items(node=node)
    elif traversal_order is traversal_order.LEVELS_ORDER:
        return bfs_walk_items(node=node)
    else:
        raise ValueError(f"Invalid '{traversal_order}' traversal order.")


def walk_values(
    node: TreeNode, traversal_order: TraversalOrder = TraversalOrder.PRE_ORDER
) -> utils.XIterable[Any]:
    """Create a tree traversal iterator of values.

    Args:
        traversal_order: Tree nodes traversal order.
    """
    if traversal_order is traversal_order.PRE_ORDER:
        return pre_walk_values(node=node)
    elif traversal_order is traversal_order.POST_ORDER:
        return post_walk_values(node=node)
    elif traversal_order is traversal_order.LEVELS_ORDER:
        return bfs_walk_values(node=node)
    else:
        raise ValueError(f"Invalid '{traversal_order}' traversal order.")


TreeNodeT = TypeVar("TreeNodeT", bound="TreeNode")

TreeNodeKey = Union[int, str]
TreeNodeValue = Any
TreeNodeItem = Tuple[TreeNodeKey, TreeNodeValue]


class TreeNode(abc.ABC):
    __slots__ = ()

    @classmethod
    @abc.abstractmethod
    def from_child_items(
        cls: Type[TreeNodeT], items: Dict[TreeNodeKey, TreeNodeValue]
    ) -> TreeNodeT:
        return NotImplemented

    @classmethod
    def from_child_values(cls: Type[TreeNodeT], values: Iterable[TreeNodeValue]) -> TreeNodeT:
        return cls.from_child_items({key: value for key, value in enumerate(values)})

    @property
    @abc.abstractmethod
    def num_children(self) -> int:
        return len(self.iter_child_values)

    @abc.abstractmethod
    def iter_child_items(self) -> Generator[TreeNodeItem, None, None]:
        return None

    @abc.abstractmethod
    def iter_child_values(self) -> Generator[TreeNodeValue, None, None]:
        return None

    pre_walk_items = pre_walk_items
    pre_walk_values = pre_walk_values

    post_walk_items = post_walk_items
    post_walk_values = post_walk_values

    bfs_walk_items = bfs_walk_items
    bfs_walk_values = bfs_walk_values

    walk_items = walk_items
    walk_values = walk_values
