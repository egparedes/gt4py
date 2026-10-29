# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""A mesh that `test_connectivity_identity_at_call` does not redefine."""

import gt4py.next as gtx


class Vertex(gtx.DimensionIndex): ...


class Edge(gtx.DimensionIndex): ...


class Cell(gtx.DimensionIndex): ...


class V2E(gtx.NeighborConnectivity[Vertex, Edge], max_neighbors=1):
    class Local(gtx.LocalDimensionIndex): ...
