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

import dataclasses
import enum
import sys
import typing

import pytest

from eve import type_validation as type_val
from eve.extended_typing import (
    Any,
    Final,
    List,
    Optional,
    Sequence,
    Set,
    SourceTypingAnnotation,
    Tuple,
    Union,
)


VALIDATORS: Final = [type_val.simple_type_validator]
FACTORIES: Final = [type_val.simple_type_validator_factory]


class SampleEnum(enum.Enum):
    FOO = "foo"
    BLA = "bla"


class SampleEmptyClass:
    pass


@dataclasses.dataclass
class SampleDataClass:
    a: int


# Each item contains: (annotation: Any, valid_values: Sequence, wrong_values: Sequence)
SAMPLE_TYPE_DEFINITIONS: List[Tuple[Any, Sequence, Sequence]] = [
    (bool, [True, False], [1, "True"]),
    (int, [1, -1], [1.0, True, "1"]),
    (float, [1.0], [1, "1.0"]),
    (str, ["", "one"], [1, ("one",)]),
    (complex, [1j], [1, 1.0, "1j"]),
    (bytes, [b"bytes", b""], ["string", ["a"]]),
    (typing.Any, ["any"], tuple()),
    (typing.Literal[1, 1.0, True], [1, 1.0, True], [False]),
    (typing.Tuple[int, str], [(3, "three")], [(), (3, 3)]),
    (typing.Tuple[int, ...], [(1, 2, 3), ()], [3, (3, "three")]),
    (typing.List[int], ([1, 2, 3], []), (1, [1.0])),
    (typing.Set[int], ({1, 2, 3}, set()), (1, [1], (1,), {1: None})),
    (typing.Dict[int, str], ({}, {3: "three"}), ([(3, "three")], 3, "three", [])),
    (typing.Sequence[int], ([1, 2, 3], [], (1, 2, 3), tuple()), (1, [1.0], {1})),
    (typing.MutableSequence[int], ([1, 2, 3], []), ((1, 2, 3), tuple(), 1, [1.0], {1})),
    (typing.Set[int], ({1, 2, 3}, set()), (1, [1], (1,), {1: None})),
    (typing.Union[int, float, str], [1, 3.0, "one"], [[1], [], 1j]),
    (typing.Optional[int], [1, None], [[1], [], 1j]),
    (
        typing.Dict[Union[int, float, str], Union[Tuple[int, Optional[float]], Set[int]]],
        [{1: (2, 3.0)}, {1.0: (2, None)}, {"1": {1, 2}}],
        [{(1, 1.0, "1"): set()}, {1: [1]}, {"1": (1,)}],
    ),
    (SampleEnum, [SampleEnum.FOO, SampleEnum.BLA], [SampleEnum, "foo", "bla"]),
    (
        SampleEmptyClass,
        [SampleEmptyClass(), SampleEmptyClass()],
        [object(), "", None, SampleDataClass(1), SampleEmptyClass],
    ),
    (
        SampleDataClass,
        [SampleDataClass(1), SampleDataClass(-42)],
        [object(), int(1), "1", SampleDataClass],
    ),
]

if sys.version_info >= (3, 10):

    @dataclasses.dataclass(slots=True)
    class SampleSlottedDataClass:
        b: float

    SAMPLE_TYPE_DEFINITIONS.append(
        (
            SampleSlottedDataClass,
            [SampleSlottedDataClass(1.0), SampleSlottedDataClass(1)],
            [object(), float(1.2), int(1), "1.2", SampleSlottedDataClass],
        )
    )


@pytest.mark.parametrize("validator", VALIDATORS)
@pytest.mark.parametrize(["type_hint", "valid_values", "wrong_values"], SAMPLE_TYPE_DEFINITIONS)
def test_validators(
    validator: type_val.TypeValidator,
    type_hint: SourceTypingAnnotation,
    valid_values: Sequence,
    wrong_values: Sequence,
):
    for value in valid_values:
        validator(value, type_hint, "<value>")

    for value in wrong_values:
        with pytest.raises((TypeError, ValueError), match="'<value>'"):
            validator(value, type_hint, "<value>")


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize(["type_hint", "valid_values", "wrong_values"], SAMPLE_TYPE_DEFINITIONS)
def test_validator_factories(
    factory: type_val.TypeValidatorFactory,
    type_hint: SourceTypingAnnotation,
    valid_values: Sequence,
    wrong_values: Sequence,
):
    validator = factory(type_hint, name="<value>")
    for value in valid_values:
        validator(value)

    for value in wrong_values:
        with pytest.raises((TypeError, ValueError), match="'<value>'"):
            validator(value)


# @pytest.mark.parametrize("factory", FACTORIES)
# def test_invalid_annotation(factory: eve_tv.TypeValidatorFactory):


# @pytest.mark.parametrize("factory", FACTORIES)
# def test_forward_refs(factory: eve_tv.TypeValidatorFactory):

#     validator = factory("", name="<value>")
