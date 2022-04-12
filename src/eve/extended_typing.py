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

"""Typing definitions working across different Python versions (via `typing_extensions`)."""


from sys import version_info as __version_info, modules as __sys_modules
import types as __types
import typing as __typing

# Definitions in typing_extensions take priority over typing
from typing import *

from typing_extensions import *

from .python_info import IS_PYTHON_AT_LEAST_3_9


if IS_PYTHON_AT_LEAST_3_9:
    # Standard library already supports PEP 585 (Type Hinting Generics In Standard Collections)
    from builtins import (
        tuple as Tuple,
        list as List,
        dict as Dict,
        set as Set,
        frozenset as FrozenSet,
        type as Type,
    )

    from collections import (
        ChainMap as ChainMap,
        Counter as Counter,
        OrderedDict as OrderedDict,
        defaultdict as defaultdict,
        deque as deque,
    )
    from collections.abc import (
        AsyncGenerator as AsyncGenerator,
        AsyncIterable as AsyncIterable,
        AsyncIterator as AsyncIterator,
        Awaitable as Awaitable,
        ByteString as ByteString,
        Callable as Callable,
        Collection as Collection,
        Container as Container,
        Coroutine as Coroutine,
        Generator as Generator,
        ItemsView as ItemsView,
        Iterable as Iterable,
        Iterator as Iterator,
        KeysView as KeysView,
        Mapping as Mapping,
        MappingView as MappingView,
        MutableMapping as MutableMapping,
        MutableSequence as MutableSequence,
        MutableSet as MutableSet,
        Reversible as Reversible,
        Sequence as Sequence,
    )
    from collections.abc import Set as AbstractSet
    from collections.abc import ValuesView as ValuesView
    from contextlib import AbstractAsyncContextManager as AsyncContextManager
    from contextlib import AbstractContextManager as ContextManager
    from re import Match as Match, Pattern as Pattern


# These fallbacks are useful for public symbols not exported by default.
# Again, definitions in typing_extensions take priority over typing
def __getattr__(name: str) -> Any:
    import sys

    import typing_extensions

    result = SENTINEL = object()
    if not (name.startswith("__") and name.endswith("__")):
        result = getattr(typing_extensions, name, SENTINEL)
        if result is SENTINEL:
            import typing

            result = getattr(typing, name, SENTINEL)

    if result is SENTINEL:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    setattr(sys.modules[__name__], name, result)  # cache result

    return result


def __dir__() -> List[str]:
    if not hasattr(self_func := (globals()["__dir__"]), "__cached_dir"):
        import typing

        import typing_extensions

        orig_dir = typing.__dir__()
        self_func.__cached_dir = orig_dir + [
            name for name in typing_extensions.__dir__() if name not in orig_dir
        ]

    return self_func.__cached_dir


def is_protocol(tp: type) -> bool:
    """Check if a type is a Protocol definition."""
    this_module = __sys_modules[is_protocol.__module__]
    return isinstance(tp, this_module._ProtocolMeta) and tp.__bases__[-1] is this_module.Protocol


def is_namedtuple(tp: type) -> bool:
    """Check if a type is a NamedTuple class."""
    this_module = __sys_modules[is_namedtuple.__module__]
    return isinstance(tp, this_module.NamedTupleMeta)


# Common type aliases
_T_co = TypeVar("_T_co", covariant=True)

FrozenList: TypeAlias = Tuple[_T_co, ...]

# Typing of annotations
_TypingGenericAliasType: TypeAlias = (
    Union[__types.GenericAlias, __typing.GenericAlias, __typing._SpecialGenericAlias]
    if IS_PYTHON_AT_LEAST_3_9
    else __typing._GenericAlias
)

_TypingSpecialFormType = __typing._SpecialForm


TypingAnnotation = Union[Type, ForwardRef, _TypingGenericAliasType, _TypingSpecialFormType]
RawTypingAnnotation = Union[str, TypingAnnotation]

# Third party protocols
class DevToolsPrettyPrintable(Protocol):
    """Used by python-devtools: https://python-devtools.helpmanual.io/"""

    def __pretty__(self, fmt: Callable[[Any], Any], **kwargs: Any) -> Generator[Any, None, None]:
        ...
