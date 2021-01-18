# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
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

from typing import Dict

from eve import Node, NodeTranslator
from gtc import gtir
from gtc.gtir import Expr


def _upcast_node(target_dtype, node: Expr):
    return node if node.dtype == target_dtype else gtir.Cast(dtype=target_dtype, expr=node)


def _upcast_nodes(*exprs):
    target_dtype = max([e.dtype for e in exprs])
    return map(lambda e: _upcast_node(target_dtype, e), exprs)


def _update_node(node: Node, updated_children: Dict[str, Node]):
    # create new node only if children changed
    old_children = node.dict(include={*updated_children.keys()})
    if any([old_children[k] != updated_children[k] for k in updated_children.keys()]):
        return node.copy(update=updated_children)
    else:
        return node


class _GTIRUpcasting(NodeTranslator):
    """
    Introduces Cast nodes (upcasting) for expr involving different datatypes.

    Precondition: all dtypes are resolved (no `None`, `Auto`, `Default`)
    Postcondition: all dtype transitions are explicit via a `Cast` node
    """

    def visit_BinaryOp(self, node: gtir.BinaryOp, **kwargs):
        left, right = _upcast_nodes(self.visit(node.left), self.visit(node.right))
        return _update_node(node, {"left": left, "right": right})

    def visit_TernaryOp(self, node: gtir.TernaryOp, **kwargs):
        true_expr, false_expr = _upcast_nodes(
            self.visit(node.true_expr), self.visit(node.false_expr)
        )
        return _update_node(
            node, {"true_expr": true_expr, "false_expr": false_expr, "cond": self.visit(node.cond)}
        )

    def visit_NativeFuncCall(self, node: gtir.NativeFuncCall, **kwargs):
        args = [*_upcast_nodes(*self.visit(node.args))]
        return _update_node(node, {"args": args})

    def visit_ParAssignStmt(self, node: gtir.ParAssignStmt, **kwargs):
        right = _upcast_node(node.left.dtype, self.visit(node.right))
        return _update_node(node, {"right": right})


def upcast(node: gtir.Stencil):
    return _GTIRUpcasting().visit(node)