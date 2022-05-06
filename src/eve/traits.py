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

"""Definitions of Trait classes."""


from __future__ import annotations

import abc
import collections
import contextlib

from . import concepts, datamodels, visitors
from .concepts import SymbolName
from .extended_typing import (
    Any,
    Dict,
    Generic,
    Iterator,
    Protocol,
    Type,
    TypeVar,
    runtime_checkable,
)


class _CollectSymbols(visitors.NodeVisitor):
    def __init__(self) -> None:
        self.collected: Dict[str, concepts.Node] = {}

    def visit_Node(self, node: concepts.Node) -> None:
        for name, metadata in node.__node_children__.items():
            if isinstance(metadata["definition"].type_, type) and issubclass(
                metadata["definition"].type_, SymbolName
            ):
                symbol_name = getattr(node, name)
                if symbol_name in self.collected:
                    raise ValueError(f"Multiple definitions of symbol '{symbol_name}'")
                self.collected[symbol_name] = node
        if not isinstance(node, SymbolTableTrait):
            # don't recurse into a new scope (i.e. node with SymbolTableTrait)
            self.generic_visit(node)

    @classmethod
    def apply(cls, node: concepts.Node) -> Dict[str, concepts.Node]:
        instance = cls()
        instance.generic_visit(node)
        return instance.collected


class SymbolTableTrait:
    def __post_init__(self):
        super(SymbolTableTrait, self).__post_init__()
        self.collect_symbols()

    def collect_symbols(self: concepts.BaseNode) -> None:
        self.annex.symtable = _CollectSymbols.apply(self)


# Visitors
_OutT = TypeVar("_OutT", covariant=True)
_KwargsT = TypeVar("_KwargsT", contravariant=True)


@runtime_checkable
class NodeVisitorTrait(Protocol[_OutT, _KwargsT]):
    def visit(self, node: concepts.Node, /, **kwargs: _KwargsT) -> _OutT:
        ...


class SymbolTableVisitorTrait(Generic[_OutT, _KwargsT]):
    """Update or add the symtable to kwargs in the visitor calls.

    This is a context manager that, when included to the contexts classvar, will
    automatically pass 'symtable' as a keyword argument to visitor methods.
    """

    def visit(self, node: concepts.Node, /, **kwargs: _KwargsT) -> _OutT:
        kwargs.setdefault("symtable", collections.ChainMap())
        if new_scope := isinstance(node, SymbolTableTrait):
            kwargs["symtable"] = kwargs["symtable"].new_child(node.annex.symtable_)

        result = super(SymbolTableVisitorTrait, self).visit(node, **kwargs)

        if new_scope:
            kwargs["symtable"] = kwargs["symtable"].parents

        return result
