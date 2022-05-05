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

"""Definitions of useful field and general types."""


from __future__ import annotations

import ast
import functools
import re
import sys
from enum import Enum as Enum, IntEnum as IntEnum

from boltons.typeutils import classproperty as classproperty  # noqa: F401
from frozendict import frozendict as _frozendict  # noqa: F401

from .extended_typing import (
    Any,
    Callable,
    ClassVar,
    Final,
    Generator,
    NamedTuple,
    NoReturn,
    Optional,
    Tuple,
    Type,
    Union,
    final,
)


# from pydantic import validator  # noqa
# from pydantic import (  # noqa: F401
#     NegativeFloat,
#     NegativeInt,
#     PositiveFloat,
#     PositiveInt,
#     StrictBool as Bool,
#     StrictFloat as Float,
#     StrictInt as Int,
#     StrictStr as Str,
# )
# from pydantic.types import ConstrainedStr



frozenlist: Final = tuple
frozendict: Final = _frozendict if sys.version_info >= (3, 9) else xtyping.FrozenDict


@final
class NothingType(type):
    """Metaclass of :class:`NOTHING` setting its bool value to False."""

    def __bool__(cls) -> bool:
        return False


@final
class NOTHING(metaclass=NothingType):
    """Marker to avoid confusion with `None` in contexts where `None` could be a valid value."""

    def __new__(cls: type) -> NoReturn:  # type: ignore[misc]  # should return an instance
        raise TypeError(f"{cls.__name__} is used as a sentinel value and cannot be instantiated.")


class StrEnum(str, Enum):
    """:class:`enum.Enum` subclass whose members are considered as real strings."""

    pass


class ConstrainedStr(str):
    """Base string subclass allowing to restrict values to those satisfying a regular expression.

    Subclasses should define the specific constraint pattern in the ``regex``
    class keyword argument.

    Examples:
        >>> class OnlyLetters(ConstrainedStr, regex=re.compile(r"^[a-zA-Z]*$")): pass
        >>> OnlyLetters("aabbCC")
        'aabbCC'

        >>> OnlyLetters("aabbCC33")
        Traceback (most recent call last):
            ...
        ValueError: OnlyLetters('aabbCC33') does not satisfies RE constraint re.compile('^[a-zA-Z]*$').

    """

    __slots__ = ()

    regex: ClassVar[re.Pattern]

    def __new__(cls, value: str) -> ConstrainedStr:
        if cls is ConstrainedStr:
            raise TypeError(f"{cls} cannot be directly instantiated, it should be subclassed.")
        instance = super().__new__(cls, value)
        if not cls.regex.fullmatch(instance):
            raise ValueError(
                f"{cls.__name__}('{instance}') does not satisfies RE constraint {cls.regex}."
            )

        return instance

    def __init_subclass__(cls, *, regex: Optional[re.Pattern] = None, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if regex is None and "regex" in cls.__dict__:
            # regex has been defined as a class var either in this class or in the parents
            assert isinstance(cls.regex, re.Pattern)
            return
        if not isinstance(regex, re.Pattern):
            raise TypeError(
                f"Invalid regex pattern ({regex}) for '{cls.__name__}' ConstrainedStr subclass."
            )
        cls.regex = regex


_SYMBOL_NAME_RE: Final = re.compile(r"^[a-zA-Z_]\w*$")


class SymbolName(ConstrainedStr, regex=_SYMBOL_NAME_RE):
    """String value containing a valid symbol name for typical programming conventions."""

    __slots__ = ()


class SymbolRef(ConstrainedStr, regex=_SYMBOL_NAME_RE):
    """Reference to a symbol name."""

    __slots__ = ()


class IntRange(NamedTuple):
    start: Optional[int] = None
    stop: Optional[int] = None
    step: Optional[int] = 1

    def __contains__(self, item: int) -> bool:
        if self.start and item < self.start:
            return False
        if self.stop and item >= self.stop:
            return False
        if self.step and (item - (self.start or 0)) % self.step:
            return False

        return True


class ConstrainedInt(int):
    """Base int subclass allowing to restrict values to specific ranges.

    Subclasses should define the specific constraint pattern in the ``range``
    class keyword argument.

    Examples:
        >>> class EvenIntNumber(ConstrainedInt, range=IntRange(None, None, 2)): pass
        >>> EvenIntNumber(2)
        2

        >>> EvenIntNumber(3)
        Traceback (most recent call last):
            ...
        ValueError: EvenIntNumber(3) does not satisfies range constraint IntRange(start=None, stop=None, step=2).

    """

    __slots__ = ()

    range: ClassVar[IntRange]

    def __new__(cls, value: int) -> ConstrainedInt:
        if cls is ConstrainedInt:
            raise TypeError(f"{cls} cannot be directly instantiated, it should be subclassed.")
        instance = super().__new__(cls, value)
        if instance not in cls.range:
            raise ValueError(
                f"{cls.__name__}({instance}) does not satisfies range constraint {cls.range}."
            )

        return instance

    def __init_subclass__(cls, *, range: IntRange, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if not isinstance(range, IntRange):
            raise TypeError(
                f"Invalid range constraint ({range}) for '{cls.__name__}' ConstrainedInt subclass."
            )
        cls.range = range


class PositiveInt(ConstrainedInt, range=IntRange(0, None)):
    """Int subclass constrained to positive values (x >= 0)."""

    __slots__ = ()


class NegativeInt(ConstrainedInt, range=IntRange(None, 0)):
    """Int subclass constrained to strictly negative values (x < 0)."""

    __slots__ = ()
