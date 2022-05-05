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
import ast
import enum
from enum import IntEnum as IntEnum
import functools
import re
import sys

import pydantic
import xxhash
from boltons.typeutils import classproperty as classproperty  # noqa: F401
from frozendict import frozendict as _frozendict  # noqa: F401

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

from . import extended_typing as xtyping
from .extended_typing import (
    Any,
    Callable,
    ClassVar,
    Final,
    Generator,
    NoReturn,
    Optional,
    Tuple,
    Type,
    Union,
    final,
)


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


# #: Typing definitions for `__get_validators__()` methods
# # (defined but not exported in `pydantic.typing`)
# PydanticCallableGenerator = Generator[Callable[..., Any], None, None]


class StrEnum(str, enum.Enum):
    """:class:`enum.Enum` subclass whose members are considered as real strings."""

    pass


class ConstrainedStr(str):
    """Base string subclass allowing to restrict values to those satisfying a regular expression.

    Subclasses should define the specific constraint pattern in the ``regex``
    class keyword argument.

    Examples:
        >>> class OnlyLetters(ConstrainedStr, regex=re.compile(r"^[a-zA-Z]*$")): pass
        >>> OnlyLetters("aabbCC")
        "aabbCC"

        >>> OnlyLetters("aabbCC33")
        "aabbCC"

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

    def __init_subclass__(cls, *, regex: re.Pattern, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if not isinstance(regex, re.Pattern):
            raise TypeError(
                f"Invalid regex pattern ({regex}) for '{cls.__name__}' ConstrainedStr subclass."
            )
        cls.regex = regex


class SymbolName(ConstrainedStr, regex=re.compile(r"^[a-zA-Z_]\w*$")):
    """String value containing a typically valid symbol name."""

    pass


class SymbolRef(ConstrainedStr):
    """Reference to a symbol name.

    Instance validation only happens automatically within a Pydantic
    model validation context.

    """

    regex = re.compile(r"^[a-zA-Z_]\w*$")

    @classmethod
    def from_string(cls, name: str) -> SymbolRef:
        name = cls.validate(name)
        return cls(name)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({super().__repr__()})"


class SourceLocation(pydantic.BaseModel):
    """Source code location (line, column, source)."""

    line: PositiveInt
    column: PositiveInt
    source: Str
    end_line: Optional[PositiveInt]
    end_column: Optional[PositiveInt]

    @classmethod
    def from_AST(cls, ast_node: ast.AST, source: Optional[str] = None) -> SourceLocation:
        if (
            not isinstance(ast_node, ast.AST)
            or getattr(ast_node, "lineno", None) is None
            or getattr(ast_node, "col_offset", None) is None
        ):
            raise ValueError(
                f"Passed AST node '{ast_node}' does not contain a valid source location."
            )
        if source is None:
            source = f"<ast.{type(ast_node).__name__} at 0x{id(ast_node):x}>"
        return cls(
            ast_node.lineno,
            ast_node.col_offset + 1,
            source,
            end_line=ast_node.end_lineno,
            end_column=ast_node.end_col_offset + 1 if ast_node.end_col_offset is not None else None,
        )

    def __init__(
        self,
        line: int,
        column: int,
        source: str,
        *,
        end_line: Optional[int] = None,
        end_column: Optional[int] = None,
    ) -> None:
        assert end_column is None or end_line is not None
        super().__init__(
            line=line, column=column, source=source, end_line=end_line, end_column=end_column
        )

    def __str__(self) -> str:
        src = self.source or ""

        end_part = ""
        if self.end_line is not None:
            end_part += f" to Line {self.end_line}"
        if self.end_column is not None:
            end_part += f", Col {self.end_column}"

        return f"<'{src}': Line {self.line}, Col {self.column}{end_part}>"

    class Config:
        extra = "forbid"
        allow_mutation = False


class SourceLocationGroup(pydantic.BaseModel):
    """A group of merged source code locations (with optional info)."""

    locations: Tuple[SourceLocation, ...]
    context: Optional[Union[str, Tuple[str, ...]]]

    def __init__(
        self, *locations: SourceLocation, context: Optional[Union[str, Tuple[str, ...]]] = None
    ) -> None:
        super().__init__(locations=locations, context=context)

    def __str__(self) -> str:
        locs = ", ".join(str(loc) for loc in self.locations)
        context = f"#{self.context}#" if self.context else ""
        return f"<{context}[{locs}]>"

    @validator("locations")
    def non_empty_tuple(cls, v: Tuple[SourceLocation, ...]) -> Tuple[SourceLocation, ...]:
        if not v:
            raise ValueError("At least one location should be provided")
        return v
