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
    Protocol,
    Set,
    Type,
    TypeVar,
    get_origin,
    runtime_checkable,
)


# ---- Symbol Tables ----
class SymbolNamesCollector(visitors.NodeVisitor):
    def __init__(self) -> None:
        self.collected_symbols: Dict[str, concepts.Node] = {}

    def visit_OpNode(self, op_node: concepts.OpNode) -> None:
        for field_name, attribute in op_node.__datamodel_fields__.items():
            if isinstance(attribute.type, type) and issubclass(attribute.type, concepts.SymbolName):
                symbol_name = getattr(op_node, field_name)
                if symbol_name in self.collected_symbols:
                    raise exceptions.EveValueError(
                        f"Multiple definitions of symbol '{symbol_name}'"
                    )
                self.collected_symbols[symbol_name] = op_node
        if not isinstance(op_node, SymbolTableCreatorTrait):
            # Stop recursion if the node opens a new scope (i.e. node with SymbolTableCreatorTrait)
            self.generic_visit(op_node)

    @classmethod
    def apply(cls, node: concepts.Node) -> Dict[str, concepts.Node]:
        collector = cls()
        collector.generic_visit(node)
        return collector.collected_symbols


@concepts.register_annex_user("symtable", Dict[str, concepts.Node], shared=True)
@datamodels.datamodel
class SymbolTableCreatorTrait:
    __slots__ = ()

    @datamodels.root_validator
    def _collect_symbol_names(cls: Type[SymbolTableCreatorTrait], node: concepts.Node) -> None:
        collected_symbols = SymbolNamesCollector.apply(node)
        node.annex.symtable = collected_symbols


_OutT = TypeVar("_OutT", covariant=True)
_KwargsT = TypeVar("_KwargsT", contravariant=True)


class SymbolRefsValidator(visitors.NodeVisitor):
    def __init__(self) -> None:
        self.missing_symbols: Set[str] = set()

    def visit_OpNode(
        self, op_node: concepts.OpNode, *, symtable: Dict[str, Any], **kwargs: Any
    ) -> None:
        for field_name, attribute in op_node.__datamodel_fields__.items():
            if isinstance(attribute.type, type) and issubclass(attribute.type, concepts.SymbolRef):
                symbol_name = getattr(op_node, field_name)
                if symbol_name not in symtable:
                    self.missing_symbols.add(symbol_name)

        if isinstance(op_node, SymbolTableCreatorTrait):
            # Append symbols from nested scope for nested nodes
            symtable = {**symtable, **op_node.annex.symtable}
        self.generic_visit(op_node, symtable=symtable, **kwargs)

    @classmethod
    def apply(cls, node: concepts.Node, *, symtable: Dict[str, Any]) -> Set[str]:
        validator = cls()
        validator.visit(node, symtable=symtable)
        return validator.missing_symbols


@concepts.register_annex_user("symtable", Dict[str, concepts.Node], shared=True)
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
        new_scope = False
        if isinstance(node, concepts.Node):
            if new_scope := ("symtable" in node.annex):
                kwargs["symtable"] = kwargs["symtable"].new_child(node.annex.symtable)

        result = super(VisitorWithSymbolTableTrait, self).visit(node, **kwargs)

        if new_scope:
            kwargs["symtable"] = kwargs["symtable"].parents

        return result
