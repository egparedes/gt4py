# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2020, CSCS - Swiss National Supercomputing Center, ETH Zurich
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


from __future__ import annotations

import re

import pytest

import eve


def test_sentinel():
    from eve.type_definitions import NOTHING

    values = [0, 1, 2, NOTHING, 4, 6]

    assert values.index(NOTHING) == 3
    assert values[values.index(NOTHING)] is NOTHING


def test_symbol_types():
    from eve.type_definitions import SymbolName

    assert SymbolName("valid_name_01A") == "valid_name_01A"
    assert SymbolName("valid_name_01A") == "valid_name_01A"
    with pytest.raises(ValueError, match="does not satisfies RE constraint"):
        SymbolName("$name_01A")
    with pytest.raises(ValueError, match="does not satisfies RE constraint"):
        SymbolName("0name_01A")
    with pytest.raises(ValueError, match="does not satisfies RE constraint"):
        SymbolName("name_01A ")

    class LettersOnlySymbol(SymbolName, regex=re.compile(r"[a-zA-Z]+$")):
        __slots__ = ()

    assert LettersOnlySymbol("validNAME") == "validNAME"
    with pytest.raises(ValueError, match="does not satisfies RE constraint"):
        LettersOnlySymbol("name_a")
    with pytest.raises(ValueError, match="does not satisfies RE constraint"):
        LettersOnlySymbol("name01")
