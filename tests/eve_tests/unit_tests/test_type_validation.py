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
import typing

import pytest

from eve import type_validation as eve_tv
from eve.extended_typing import (
    Final,
    List,
    Optional,
    Sequence,
    Set,
    SourceTypingAnnotation,
    Tuple,
    Union,
)


VALIDATORS: Final = [eve_tv.simple_type_validator]
FACTORIES: Final = [eve_tv.simple_type_validator_factory]


sample_type_data: Final[List[Tuple[str, Sequence, Sequence]]] = [
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
]


@pytest.mark.parametrize("validator", VALIDATORS)
@pytest.mark.parametrize(["type_hint", "valid_values", "wrong_values"], sample_type_data)
def test_validators(
    validator: eve_tv.TypeValidator,
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
@pytest.mark.parametrize(["type_hint", "valid_values", "wrong_values"], sample_type_data)
def test_validator_factories(
    factory: eve_tv.TypeValidatorFactory,
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


# T = TypeVar("T")


# class SampleEnum(enum.Enum):
#     FOO = "foo"
#     BLA = "bla"


# class SampleEmptyClass:
#     pass


# @dataclasses.dataclass
# class SampleDataClass:
#     a: int


# @dataclasses.dataclass(slots=True)
# class SampleSlottedDataClass:
#     b: float


# # Reuse sample_type_data from test_field_type_hint
# @pytest.mark.parametrize(["type_hint", "valid_values", "wrong_values"], sample_type_data)
# def test_concrete_field_type_validation(
#     type_hint: str, valid_values: Sequence[Any], wrong_values: Sequence[Any]
# ):
#     concrete_type: Type = eval(type_hint)
#     Model: Type[datamodels.DataModelTP] = typing.get_origin(GenericModel[concrete_type])  # type: ignore[valid-type,assignment]

#     for value in valid_values:
#         Model(value=value)

#     for value in wrong_values:
#         with pytest.raises((TypeError, ValueError), match="'value'"):
#             Model(value=value)
