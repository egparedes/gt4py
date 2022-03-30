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

"""Typing definitions working across different Python versions (uses typing_extensions)."""


from sys import version_info as __version_info
# Definitions in typing_extensions take priority over typing
from typing import *
from typing_extensions import *

__IS_PYTHON_AT_LEAST_3_8 = __version_info >= (3, 8)
__IS_PYTHON_AT_LEAST_3_9 = __version_info >= (3, 9)
__IS_PYTHON_AT_LEAST_3_10 = __version_info >= (3, 10)
__IS_PYTHON_AT_LEAST_3_11 = __version_info >= (3, 11)
__IS_PYTHON_3_8 = __IS_PYTHON_AT_LEAST_3_8 and not __IS_PYTHON_AT_LEAST_3_9
__IS_PYTHON_3_9 = __IS_PYTHON_AT_LEAST_3_9 and not __IS_PYTHON_AT_LEAST_3_10
__IS_PYTHON_3_10 = __IS_PYTHON_AT_LEAST_3_10 and not __IS_PYTHON_AT_LEAST_3_11
__IS_PYTHON_3_11 = __IS_PYTHON_AT_LEAST_3_11 and not __version_info > (3, 11)


if __IS_PYTHON_AT_LEAST_3_9:
    # Implements PEP 585 (Type Hinting Generics In Standard Collections)
    Tuple = tuple
    List = list
    Dict = dict
    Set = set
    FrozenSet = frozenset
    Type = type

    from collections import deque, defaultdict, OrderedDict, Counter, ChainMap
    from collections.abc import (
        Awaitable,
        Coroutine,
        AsyncIterable,
        AsyncIterator,
        AsyncGenerator,
        Iterable,
        Iterator,
        Generator,
        Reversible,
        Container,
        Collection,
        Callable,
        Set as AbstractSet,
        MutableSet,
        Mapping,
        MutableMapping,
        Sequence,
        MutableSequence,
        ByteString,
        MappingView,
        KeysView,
        ItemsView,
        ValuesView,
    )
    from contextlib import (
        AbstractContextManager as ContextManager,
        AbstractAsyncContextManager as AsyncContextManager,
    )
    from re import Match, Pattern


# These fallbacks are useful for public symbols not exported by default.
# Again, definitions in typing_extensions take priority over typing
def __getattr__(name: str) -> Any:
    import sys, typing_extensions

    result = SENTINEL = object()
    if not (name.startswith("__") and name.endswith("__")):
        result = getattr(typing_extensions, name, SENTINEL)
        if result is SENTINEL:
            import typing

            result = getattr(typing, name, SENTINEL)

    if result is SENTINEL:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    setattr(sys.modules[__name__], name, result)

    return result


def __dir__() -> List[str]:
    if not hasattr(self_func := (globals()["__dir__"]), "__cached_dir"):
        import typing, typing_extensions

        orig_dir = typing.__dir__()
        self_func.__cached_dir = orig_dir + [
            name for name in typing_extensions.__dir__() if name not in orig_dir
        ]

    return self_func.__cached_dir
