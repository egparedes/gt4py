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
        self.collected_symbols: Dict[str, concepts.Node] = {}

    def visit_Node(self, node: concepts.Node) -> None:
        for field_name, attribute in node.__datamodel_fields__.items():
            if isinstance(attribute.type, type) and issubclass(attribute.type, SymbolName):
                symbol_name = getattr(node, field_name)
                if symbol_name in self.collected_symbols:
                    raise ValueError(f"Multiple definitions of symbol '{symbol_name}'")
                self.collected_symbols[symbol_name] = node
        if not isinstance(node, SymbolTableTrait):
            # Recurse only if the node does not open a new scope (i.e. node with SymbolTableTrait)
            self.generic_visit(node)

    @classmethod
    def apply(cls, node: concepts.Node) -> Dict[str, concepts.Node]:
        collector = cls()
        collector.generic_visit(node)
        return collector.collected_symbols


@datamodels.datamodel
class SymbolTableTrait:
    @datamodels.root_validator
    def collect_symbols(cls: Type[SymbolTableTrait], self: SymbolTableTrait) -> None:
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
            kwargs["symtable"] = kwargs["symtable"].new_child(node.annex.symtable)

        result = super(SymbolTableVisitorTrait, self).visit(node, **kwargs)

        if new_scope:
            kwargs["symtable"] = kwargs["symtable"].parents

        return result
