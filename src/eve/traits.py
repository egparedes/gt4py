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

from eve import datamodels

from . import concepts, visitors
from .type_definitions import SymbolName
from .typingx import Any, Dict, Type


class _CollectSymbols(visitors.NodeVisitor):
    @classmethod
    def apply(cls, node: concepts.TreeNode) -> Dict[str, Any]:
        instance = cls()
        instance.visit(node)
        return instance.collected

    def __init__(self) -> None:
        self.collected: Dict[str, Any] = {}

    def visit_Node(self, node: concepts.Node) -> None:
        for name, field_info in node.__datamodel_fields__.items():
            if isinstance(t := field_info.type, type) and issubclass(t, SymbolName):
                self.collected[getattr(node, name)] = node
        if not isinstance(node, SymbolTableTrait):
            # don't recurse into a new scope (i.e. node with SymbolTableTrait)
            self.generic_visit(node)


class SymbolTableTrait(datamodels.DataModel):
    symtable_: Dict[str, Any] = datamodels.field(default_factory=dict, compare=False)

    @datamodels.root_validator
    def _collect_symbols_validator(cls: Type[SymbolTableTrait], instance: SymbolTableTrait) -> None:
        instance.collect_symbols()

    def collect_symbols(self) -> None:
        self.symtable_ = _CollectSymbols.apply(self)
