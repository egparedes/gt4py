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

import collections

from . import concepts, datamodels, exceptions, visitors
from .extended_typing import (
    Any,
    ClassVar,
    Dict,
    Final,
    Generic,
    Iterator,
    Set,
    Protocol,
    Type,
    TypeVar,
    get_origin,
    runtime_checkable,
)


# ---- Symbol Tables ----
class SymbolNamesCollector(visitors.NodeVisitor):
    def __init__(self) -> None:
        self.collected_symbols: Dict[str, concepts.Node] = {}

    def visit_Node(self, node: concepts.Node) -> None:
        for value in node.iter_child_values():
            if isinstance(value, concepts.SymbolName):
                if value in self.collected_symbols:
                    raise exceptions.EveValueError(f"Multiple definitions of symbol '{value}'")
                self.collected_symbols[value] = node
        if not isinstance(node, SymbolTableCreatorTrait):
            # Stop recursion if the node opens a new scope (i.e. node with SymbolTableTrait)
            self.generic_visit(node)

    @classmethod
    def apply(cls, node: concepts.Node) -> Dict[str, concepts.Node]:
        collector = cls()
        collector.generic_visit(node)
        return collector.collected_symbols


@concepts.register_annex_key("symtable")
@datamodels.datamodel
class SymbolTableCreatorTrait:
    __slots__ = ()

    @datamodels.root_validator
    def _collect_symbol_names(cls: Type[SymbolTableCreatorTrait], node: concepts.Node) -> None:
        node.annex.symtable = SymbolNamesCollector.apply(node)


_OutT = TypeVar("_OutT", covariant=True)
_KwargsT = TypeVar("_KwargsT", contravariant=True)


class SymbolRefsValidator(visitors.NodeVisitor):
    def __init__(self) -> None:
        self.missing_symbols: Set[str] = set()

    def visit_Node(self, node: concepts.Node, *, symtable: Dict[str, Any], **kwargs: Any) -> None:
        for value in node.iter_child_values():
            if isinstance(value, concepts.SymbolRef):
                if value not in symtable:
                    self.missing_symbols.add(value)

        if isinstance(node, SymbolTableCreatorTrait):
            # Append symbols from nested scope for nested nodes
            symtable = {**symtable, **node.annex.symtable}
        self.generic_visit(node, symtable=symtable, **kwargs)

    @classmethod
    def apply(cls, node: concepts.Node, *, symtable: Dict[str, Any]) -> Set[str]:
        validator = cls()
        validator.visit(node, symtable=symtable)
        return validator.missing_symbols


@datamodels.datamodel
class SymbolRefsValidatorTrait:
    __slots__ = ()

    @datamodels.root_validator
    def _validate_symbol_refs(cls: Type[SymbolRefsValidatorTrait], node: concepts.Node) -> None:
        validator = SymbolRefsValidator()
        symtable = node.annex.symtable
        for child_node in node.iter_child_values():
            validator.visit(child_node, symtable=symtable)

        if validator.missing_symbols:
            raise exceptions.EveValueError(
                "Symbols {} not found.".format(validator.missing_symbols)
            )


@datamodels.datamodel
class SymbolTableTrait(SymbolRefsValidatorTrait, SymbolTableCreatorTrait):
    __slots__ = ()


#######


@runtime_checkable
class NodeVisitorTrait(Protocol[_OutT, _KwargsT]):
    def visit(self, node: concepts.Node, /, **kwargs: _KwargsT) -> _OutT:
        ...


class VisitorWithSymbolTableTrait(Generic[_OutT, _KwargsT]):
    """Update or add the symtable to kwargs in the visitor calls.

    This is a context manager that, when included to the contexts classvar, will
    automatically pass 'symtable' as a keyword argument to visitor methods.
    """

    def visit(self, node: concepts.Node, /, **kwargs: _KwargsT) -> _OutT:
        kwargs.setdefault("symtable", collections.ChainMap())
        if new_scope := isinstance(node, SymbolTableCreatorTrait):
            kwargs["symtable"] = kwargs["symtable"].new_child(node.annex.symtable)

        result = super(VisitorWithSymbolTableTrait, self).visit(node, **kwargs)

        if new_scope:
            kwargs["symtable"] = kwargs["symtable"].parents

        return result
