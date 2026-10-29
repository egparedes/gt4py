# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Classes at the Python edge, refs inside: a redefined connectivity is rejected when called.

The connectivity counterpart of `test_dimension_identity_at_call`. A notebook cell re-run
redefines a declaration with the same dimensions and counts, so the new class has a
`ConnectivityRef` *equal* to the old one. Here the redefined declaration shares another one's
local dimension, so no dimension class changes and only the declaration identity tells the two
apart: a program written with the old declaration, called with a table that the offset provider
binds to the new one, is rejected at the argument check.
"""

import importlib

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import common, errors

from . import _redefinable_connectivity as conns
from ._stable_mesh import V2E, Cell, Vertex


_DIFFERENT = "two different classes named"

# --- the first run: the program is written with the original declaration
OldV2C = conns.V2C


@gtx.field_operator
def _old_first_cell(
    c: gtx.Field[gtx.Dims[Cell], gtx.float64],
) -> gtx.Field[gtx.Dims[Vertex], gtx.float64]:
    return c(OldV2C[0])


@gtx.program
def old_first_cell(
    c: gtx.Field[gtx.Dims[Cell], gtx.float64], out: gtx.Field[gtx.Dims[Vertex], gtx.float64]
):
    _old_first_cell(c, out=out)


# --- the re-run: a new declaration class with an equal ref
conns = importlib.reload(conns)
V2C = conns.V2C


@gtx.field_operator
def _first_cell(
    c: gtx.Field[gtx.Dims[Cell], gtx.float64],
) -> gtx.Field[gtx.Dims[Vertex], gtx.float64]:
    return c(V2C[0])


@gtx.program
def first_cell(
    c: gtx.Field[gtx.Dims[Cell], gtx.float64], out: gtx.Field[gtx.Dims[Vertex], gtx.float64]
):
    _first_cell(c, out=out)


def _arguments():
    c = gtx.as_field([Cell], np.arange(3.0))
    out = gtx.as_field([Vertex], np.zeros(2))
    v2c = gtx.as_connectivity(
        domain={Vertex: 2, V2E.Local: 1}, codomain=Cell, data=np.array([[2], [0]], np.int32)
    )
    return c, out, {V2C.offset_tag: v2c}


BACKENDS = pytest.mark.parametrize("backend", [None, gtx.gtfn_cpu], ids=["embedded", "gtfn_cpu"])


def test_redefinition_keeps_an_equal_ref():
    assert OldV2C is not V2C
    assert OldV2C.offset_tag == V2C.offset_tag == V2C.tag
    assert common.connectivity_ref(OldV2C) == common.connectivity_ref(V2C)
    # the table under the shared key is bound to the most recent declaration
    _, _, offset_provider = _arguments()
    assert common.bound_declaration(V2C.offset_tag, offset_provider[V2C.offset_tag]) is V2C


@BACKENDS
def test_program_with_the_old_declaration(backend):
    c, out, offset_provider = _arguments()
    with pytest.raises(errors.DSLError, match=f"connectivity 'V2C'.*{_DIFFERENT}"):
        old_first_cell.with_backend(backend)(c, out, offset_provider=offset_provider)


@BACKENDS
def test_field_operator_with_the_old_declaration(backend):
    c, out, offset_provider = _arguments()
    with pytest.raises(errors.DSLError, match=_DIFFERENT):
        _old_first_cell.with_backend(backend)(c, out=out, offset_provider=offset_provider)


def test_the_current_declaration_is_accepted_embedded():
    c, out, offset_provider = _arguments()
    first_cell.with_backend(None)(c, out, offset_provider=offset_provider)
    assert np.allclose(out.asnumpy(), [2.0, 0.0])
