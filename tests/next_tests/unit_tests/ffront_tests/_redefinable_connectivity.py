# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
A connectivity that `test_connectivity_identity_at_call` redefines with `importlib.reload`.

It shares the local dimension of `_stable_mesh.V2E`, so redefining it creates no new dimension
class: only the declaration class changes.
"""

from typing import TypeAlias

import gt4py.next as gtx

from ._stable_mesh import V2E, Cell, Vertex


class V2C(gtx.NeighborConnectivity[Vertex, Cell]):
    Local: TypeAlias = V2E.Local
