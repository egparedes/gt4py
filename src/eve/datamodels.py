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
# version. See the LICENSE.txt file at the top-l directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Data Model class and related utils.

Data Models can be considered as an extension to ``dataclasses`` providing
run-time validation utils for fields. Values assigned to fields at
initialization are validated with automatic type checkings using the
field type definition. Custom field validation methods can also be added with
the :func:`validator` decorator, and global instance validation methods with
:func:`root_validator`.

The datamodels API is intentionally copying ``dataclasses`` API as close as
possible. Actually, Data Model classes are also ``dataclasses``, so all functions
dealing with ``dataclasses`` should also work with Data Models.

Notes:
    Since the current implementation uses `attrs <https://www.attrs.org/>`_ internally,
    Data Model classes are also ``attrs`` classes.

Examples:
    >>> @datamodel
    ... class SampleModel:
    ...     name: str
    ...     value: int
    ...
    ...     @validator('name')
    ...     def _name_validator(self, attribute, value):
    ...         if len(value) < 5:
    ...             raise ValueError(
    ...                 f"Provided value '{value}' for '{attribute.name}' field is too short."
    ...             )
    ...
    >>> model = SampleModel(name="Diane", value=32)  # use keyword arguments by default
    >>> print(model)
    SampleModel(name='Diane', value=32)
    >>> SampleModel(name="Diane", value=1.32)
    Traceback (most recent call last):
        ...
    TypeError: 'value' must be <class 'int'> (got '1.32' that is a <class 'float'>).

    >>> class AnotherSampleModel(DataModel, kw_only=False):
    ...     name: str
    ...     friends: List[str]
    ...
    ...     @root_validator
    ...     def _root_validator(cls, instance):
    ...         if instance.name in instance.friends:
    ...             raise ValueError("'name' value cannot appear in 'friends' list.")
    ...
    >>> model = AnotherSampleModel("John", ["Kate", "Alice"])  # positional arguments are ok when 'kw_only=False'
    >>> print(model)
    AnotherSampleModel(name='John', friends=['Kate', 'Alice'])
    >>> AnotherSampleModel("John", ["Kate", "Alice", "John"])
    Traceback (most recent call last):
        ...
    ValueError: 'name' value cannot appear in 'friends' list.
"""

from __future__ import annotations

import abc
import collections
import dataclasses
import sys
import typing
import warnings
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    ForwardRef,
    Generator,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import attr
from attr import validators

from eve import typingx, utils
from eve.type_definitions import NOTHING
from eve.typingx import NonDataDescriptor


# Typing
T = TypeVar("T")
V = TypeVar("V")


Attribute = attr.Attribute


class _AttrClassTp(Protocol):
    __attrs_attrs__: ClassVar[Tuple[Attribute, ...]]


class _DataClassTp(Protocol):
    __dataclass_fields__: ClassVar[Dict[str, dataclasses.Field]]
    __dataclass_params__: ClassVar[dataclasses._DataclassParams]

    def __post_init__(self) -> None:
        ...


class _DevToolsPrettyPrintable(Protocol):
    def __pretty__(self, fmt: Callable[[Any], Any], **kwargs: Any) -> Generator[Any, None, None]:
        ...


class DataModelTp(_AttrClassTp, _DataClassTp, _DevToolsPrettyPrintable, Protocol):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...

    __datamodel_fields__: ClassVar[utils.FrozenNamespace[FieldInfo]]
    __datamodel_initializers__: ClassVar[Tuple[Callable[[DataModelTp], None], ...]]
    __datamodel_options__: ClassVar[DataModelOptions]
    __datamodel_validators__: ClassVar[Tuple[RootValidatorType, ...]]


class GenericDataModelTp(DataModelTp, Protocol):
    __args__: ClassVar[Tuple[Union[Type, TypeVar]]]
    __parameters__: ClassVar[Tuple[TypeVar]]
    __class__: ClassVar[DataModelTp]  # type: ignore[assignment]

    @classmethod
    def __class_getitem__(
        cls: Type[GenericDataModelTp], args: Union[Type, Tuple[Type]]
    ) -> GenericDataModelAlias:
        ...


ValidatorType = Callable[[DataModelTp, Attribute, T], None]
RootValidatorType = Callable[[Type[DataModelTp], DataModelTp], None]


@typing.runtime_checkable
class TypeWithAttrValidatorTp(Protocol):
    """Protocol for classes defining its own custom validator (when used as fields)."""

    @classmethod
    @abc.abstractmethod
    def __type_validator__(self) -> ValidatorType:
        raise NotImplementedError()


class GenericDataModelAlias(typing._GenericAlias, _root=True):  # type: ignore[call-arg,name-defined]  # typing._GenericAlias not visible
    """Custom class alias compatible with aliases created by ``typing.Generic``.

    This class emulates the :class:`typing._GenericAlias` behavior, to be
    compatible with the mechanism of the :mod:`typing` module for ``Generic``
    types. Basically, a :class:`typing._GenericAlias` instance is a class
    proxy which stores a reference to the original class (``__origin__``),
    the generic type parameters (``__parameters__``) and the concrete
    types passed at creation (``__args__``).

    Both :class:`typing._GenericAlias` and this class implement a
    ``__mro__entries__()`` method (PEP 560) and, therefore, when
    instances of these classes are found in the list of bases of
    a new class, they are automatically substituted by the original
    Python class, and the new type is created as usual.

    Instances of this class work exactly in the same way, but also
    create new actual Data Model classes during the `concretization`
    of generic models, which are stored instead of the original
    generic models in the ``__origin__`` attribute.

    The new concrete class can be accessed as usual using
    :class:`typing.get_origin` or by using the custom :attr:`__class__`
    shortcut provided by this class.

    Examples:
        >>> from typing import Generic, get_origin
        >>> @datamodel
        ... class Model(Generic[T]):
        ...     value: T
        ...
        >>> print(Model.__parameters__)
        (~T,)
        >>> hasattr(Model, '__args__')
        False

        >>> assert isinstance(Model[int], GenericDataModelAlias)
        >>> assert issubclass(get_origin(Model[int]), Model)
        >>> assert Model[int].__class__ is get_origin(Model[int])
        >>> print(Model[int].__class__.__name__)
        Model__int

        >>> print(Model[int].__parameters__)
        ()
        >>> hasattr(Model[int], '__args__')
        True
        >>> print(Model[int].__args__)
        (<class 'int'>,)

    Notes:
        For the full picture check also related PEPs:

            - `PEP 526 - Syntax for Variable Annotations <https://www.python.org/dev/peps/pep-0526>`_
            - `PEP 560 - Core support for typing module and generic types <https://www.python.org/dev/peps/pep-0560>`_
    """

    __origin__: Type[GenericDataModelTp]

    def __getitem__(self, args: Union[Type, Tuple[Type]]) -> GenericDataModelAlias:
        origin_model: Type[GenericDataModelTp] = self.__origin__
        assert isinstance(origin_model, type) and is_generic(origin_model)
        return origin_model.__class_getitem__(args)  # equivalent to: self.__origin__[args]

    @property  # type: ignore[misc]  # Read-only property cannot override read-write property
    def __class__(self) -> Type:
        """Return the concrete class represented by this instance."""
        assert isinstance(self.__origin__, type)
        return self.__origin__


# Implementation
_DERIVED_FIELD_TAG = "_DERIVED_FIELD_TAG_"
_FIELD_INITIALIZERS = "__datamodel_initializers__"
_FIELD_VALIDATOR_TAG = "_FIELD_VALIDATOR_TAG_"
_MODEL_FIELDS = "__datamodel_fields__"
_MODEL_OPTIONS = "__datamodel_options__"
_ROOT_VALIDATOR_TAG = "_ROOT_VALIDATOR_TAG_"
_ROOT_VALIDATORS = "__datamodel_validators__"
_STRICT_TYPE_VALIDATOR_TAG = "_STRICT_TYPE_VALIDATOR_TAG_"


# -- Validators --
@dataclasses.dataclass
class _ForwardRefValidator:
    """Implementation of ``attr.s`` type validator for ``ForwardRef`` typings."""

    #: Actual type validators created after resolving the forward references.
    validator: Optional[ValidatorType] = None

    def __call__(self, instance: DataModelTp, attribute: Attribute, value: Any) -> None:
        if self.validator is None:
            model_cls = instance.__class__
            update_forward_refs(model_cls)
            self.validator = strict_type_attrs_validator(
                getattr(getattr(model_cls, _MODEL_FIELDS), attribute.name).type
            )

        self.validator(instance, attribute, value)


@dataclasses.dataclass
class _TupleValidator:
    """Implementation of ``attr.s`` type validator for ``Tuple`` typings."""

    #: Collection of validators.
    validators: Tuple[ValidatorType, ...]
    #: Class used in the container ``isintance()`` check.
    tuple_type: Type[Tuple]

    def __call__(self, instance: DataModelTp, attribute: Attribute, value: Any) -> None:
        if not isinstance(value, self.tuple_type):
            raise TypeError(
                f"In '{attribute.name}' validation, got '{value}' that is a {type(value)} instead of {self.tuple_type}."
            )
        if len(value) != len(self.validators):
            raise TypeError(
                f"In '{attribute.name}' validation, got '{value}' tuple which contains {len(value)} elements instead of {len(self.validators)}."
            )

        _i = None
        item_value = ""
        try:
            for _i, (item_value, item_validator) in enumerate(zip(value, self.validators)):
                item_validator(instance, attribute, item_value)
        except Exception as e:
            raise TypeError(
                f"In '{attribute.name}' validation, tuple '{value}' contains invalid value '{item_value}' at position {_i}."
            ) from e


@dataclasses.dataclass
class _OrValidator:
    """Implementation of ``attr.s`` validator composing multiple validators together using OR."""

    #: Collection of validators.
    validators: Tuple[ValidatorType, ...]
    #: Exception class for validation errors.
    error_type: Type[Exception]

    def __call__(self, instance: DataModelTp, attribute: Attribute, value: Any) -> None:
        passed = False
        for v in self.validators:
            try:
                v(instance, attribute, value)
                passed = True
                break
            except Exception:
                pass

        if not passed:
            raise self.error_type(
                f"In '{attribute.name}' validation, provided value '{value}' fails for all the possible validators."
            )


@dataclasses.dataclass
class _LiteralValidator:
    """Implementation of ``attr.s`` type validator for ``Literal`` typings."""

    literal: Any

    def __call__(self, instance: DataModelTp, attribute: Attribute, value: Any) -> None:
        if isinstance(self.literal, bool):
            valid = value is self.literal
        else:
            valid = value == self.literal
        if not valid:
            raise ValueError(
                f"Provided value '{value}' field does not match {self.literal} during '{attribute.name}' validation."
            )


def empty_attrs_validator() -> ValidatorType:
    """Create an ``attr.s`` empty validator which always succeeds."""

    def _empty_validator(instance: DataModelTp, attribute: Attribute, value: Any) -> None:
        pass

    return _empty_validator


def forward_ref_type_attrs_validator() -> ValidatorType:
    """Create an ``attr.s`` strict type validator for ``ForwardRef`` typings.

    The generated validator will resolve the field type to an actual type
    the first time is called.
    """
    return _ForwardRefValidator()


def instance_of_int_attrs_validator() -> ValidatorType:
    """Create an ``attr.s`` validator for ``int`` values which fails with ``bool`` values."""

    def _int_validator(instance: DataModelTp, attribute: Attribute, value: Any) -> None:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(
                f"'{attribute.name}' must be {int} (got '{value}' that is a {type(value)})."
            )

    return _int_validator


def or_attrs_validator(*validators: ValidatorType, error_type: Type[Exception]) -> ValidatorType:
    """Create an ``attr.s`` validator combinator where only one of the validators needs to pass."""
    if len(validators) == 1:
        return validators[0]
    else:
        return _OrValidator(validators, error_type=error_type)


def literal_type_attrs_validator(*type_args: Type) -> ValidatorType:
    """Create an ``attr.s`` strict type validator for ``Literal`` typings."""
    return or_attrs_validator(*(_LiteralValidator(t) for t in type_args), error_type=ValueError)


def tuple_type_attrs_validator(*type_args: Type, tuple_type: Type = tuple) -> ValidatorType:
    """Create an ``attr.s`` strict type validator for ``Tuple`` typings."""
    if len(type_args) == 2 and (type_args[1] is Ellipsis):
        # Tuple as an immutable sequence type: Tuple[int, ...]
        if not issubclass(tuple_type, tuple):
            raise TypeError(f"Invalid 'tuple' subclass '{tuple_type}'.")
        member_type_hint = type_args[0]
        return attr.validators.deep_iterable(
            member_validator=strict_type_attrs_validator(member_type_hint),
            iterable_validator=attr.validators.instance_of(tuple_type),
        )
    else:
        # Tuple as a heterogeneous container: Tuple[int, float]
        return _TupleValidator(
            tuple(strict_type_attrs_validator(t) for t in type_args),
            tuple_type,
        )


def union_type_attrs_validator(*type_args: Type) -> ValidatorType:
    """Create an ``attr.s`` strict type validator for Union typings."""
    if len(type_args) == 2 and (type_args[1] is type(None)):  # noqa: E721  # use isinstance()
        non_optional_validator = strict_type_attrs_validator(type_args[0])
        return attr.validators.optional(non_optional_validator)
    else:
        return or_attrs_validator(
            *(strict_type_attrs_validator(t) for t in type_args),
            error_type=TypeError,
        )


def strict_type_attrs_validator(type_hint: Any) -> ValidatorType:
    """Create an ``attr.s`` strict type validator for a specific typing hint."""
    type_args = typing.get_args(type_hint)

    # Custom type validator
    if isinstance(type_hint, TypeWithAttrValidatorTp):
        return type_hint.__type_validator__()

    # Non-generic types
    if isinstance(type_hint, type) and type_hint is not type(None):  # noqa: E721  # use isinstance
        assert not type_args
        if type_hint is int:
            return instance_of_int_attrs_validator()
        else:
            return attr.validators.instance_of(type_hint)
    if isinstance(type_hint, typing.TypeVar):
        if type_hint.__bound__:
            return attr.validators.instance_of(type_hint.__bound__)
        else:
            return empty_attrs_validator()
    if isinstance(type_hint, ForwardRef):
        return forward_ref_type_attrs_validator()
    if type_hint is Any:
        return empty_attrs_validator()

    # Generic and parametrized type hints
    origin_type = typing.get_origin(type_hint)

    if origin_type is typing.Literal:
        return literal_type_attrs_validator(*type_args)
    if origin_type is typing.Union:
        return union_type_attrs_validator(*type_args)
    if isinstance(origin_type, type):
        # Deal with generic collections
        if issubclass(origin_type, tuple):
            return tuple_type_attrs_validator(*type_args, tuple_type=origin_type)
        if issubclass(origin_type, (collections.abc.Sequence, collections.abc.Set)):
            assert len(type_args) == 1
            member_type_hint = type_args[0]
            return attr.validators.deep_iterable(
                member_validator=strict_type_attrs_validator(member_type_hint),
                iterable_validator=attr.validators.instance_of(origin_type),
            )
        if issubclass(origin_type, collections.abc.Mapping):
            assert len(type_args) == 2
            key_type_hint, value_type_hint = type_args
            return attr.validators.deep_mapping(
                key_validator=strict_type_attrs_validator(key_type_hint),
                value_validator=strict_type_attrs_validator(value_type_hint),
                mapping_validator=attr.validators.instance_of(origin_type),
            )

    raise TypeError(f"Type description '{type_hint}' is not supported.")


@dataclasses.dataclass
class AutoTypeValidator:
    validator: ValidatorType

    def __call__(self, instance: DataModelTp, attribute: Attribute, value: Any) -> None:
        return self.validator(instance, attribute, value)


def make_auto_type_validator(type_hint: Any) -> ValidatorType:
    return AutoTypeValidator(strict_type_attrs_validator(type_hint))


# -- DataModel --
def _collect_field_validators(cls: Type, *, delete_tag: bool = True) -> Dict[str, ValidatorType]:
    result = {}
    for member in cls.__dict__.values():
        if hasattr(member, _FIELD_VALIDATOR_TAG):
            field_name = getattr(member, _FIELD_VALIDATOR_TAG)
            result.setdefault(field_name, []).append(member)
            if delete_tag:
                delattr(member, _FIELD_VALIDATOR_TAG)

    result = {
        name: (v_list[0] if len(v_list) == 0 else attr.validators.and_(*v_list))
        for name, v_list in result.items()
    }

    return result


def _collect_root_validators(cls: Type, *, delete_tag: bool = True) -> List[RootValidatorType]:
    result = []
    for base in reversed(cls.__mro__[1:]):
        for validator in getattr(base, _ROOT_VALIDATORS, []):
            if validator not in result:
                result.append(validator)

    for member in cls.__dict__.values():
        if hasattr(member, _ROOT_VALIDATOR_TAG):
            result.append(member)
            if delete_tag:
                delattr(member, _ROOT_VALIDATOR_TAG)

    return result


def _collect_derived_fields(
    cls: Type,
    canonicalized_annotations: Dict[str, Any],
    skip_private: bool,
    *,
    delete_tag: bool = True,
) -> List[Tuple[str, Any, attr._make._CountingAttr, Callable[[DataModelTp], None]]]:
    fields = []
    for name, member_func in cls.__dict__.items():
        if skip_private and name.startswith("_"):
            continue

        if hasattr(member_func, _DERIVED_FIELD_TAG):
            assert callable(member_func)
            options = getattr(member_func, _DERIVED_FIELD_TAG)

            field_type_hint = canonicalized_annotations.get(name, None)
            if "ClassVar[" in repr(typing.get_origin(field_type_hint)):
                raise TypeError(
                    f"Invalid type annotation for '{name}' derived field ({field_type_hint})."
                )

            return_type_hint = typingx.get_canonical_type_hints(member_func).get("return", None)
            if field_type_hint is None and return_type_hint is None:
                raise TypeError(f"Missing type annotation for '{name}' derived type initializer.")
            if field_type_hint is None:
                field_type_hint = return_type_hint
            elif return_type_hint is not None and field_type_hint != return_type_hint:
                raise TypeError(
                    f"Wrong return type for '{name}' derived type initializer: "
                    f"'{return_type_hint}' (expected '{field_type_hint}')."
                )

            member_validator = make_auto_type_validator(field_type_hint)
            if name in cls.__datamodel_fields__.keys():
                inherited_validator = getattr(cls.__datamodel_fields__, name).validator
                if isinstance(inherited_validator, attr._make._AndValidator):
                    custom_validators = [
                        v
                        for v in inherited_validator._validators
                        if not isinstance(v, AutoTypeValidator)
                    ]
                    print(f"{custom_validators=}")
                    if custom_validators:
                        member_validator = attr.validators.and_(
                            member_validator, *custom_validators
                        )
                else:
                    member_validator = attr.validators.and_(member_validator, inherited_validator)

            field_attr_def = attr.ib(
                init=False,
                repr=options.repr,
                hash=options.hash,
                eq=options.compare,
                order=options.compare,
                validator=member_validator,
                metadata=options.metadata,
                on_setattr=attr.setters.frozen,
            )
            fields.append(
                (
                    name,
                    member_func.__annotations__.get("return", None),
                    field_attr_def,
                    _make_derived_field_initializer(name, member_func, member_validator),
                )
            )

            if delete_tag:
                delattr(member_func, _DERIVED_FIELD_TAG)

    return fields


def _make_derived_field_initializer(
    field_name: str, init_func: Callable[[DataModelTp], V], validator: ValidatorType
) -> Callable[[DataModelTp], None]:
    def _derived_field_initializer(
        instance: DataModelTp, instance_values: Optional[utils.FrozenNamespace] = None
    ) -> None:
        if instance_values is None:
            instance_values = fields_view(instance)

        value = init_func(instance_values)
        object.__setattr__(instance, field_name, value)
        validator(instance, getattr(instance.__datamodel_fields__, field_name), value)

    return _derived_field_initializer


def _get_attribute_from_bases(
    name: str, mro: Tuple[Type, ...], annotations: Optional[Dict[str, Any]] = None
) -> Optional[Attribute]:
    for base in mro:
        for base_field_attrib in getattr(base, "__attrs_attrs__", []):
            if base_field_attrib.name == name:
                if annotations is not None:
                    annotations[name] = base.__annotations__[name]
                return typing.cast(Attribute, base_field_attrib)

    return None


def _substitute_typevars(
    type_hint: Type, type_params_map: Mapping[TypeVar, Union[Type, TypeVar]]
) -> Tuple[Union[Type, TypeVar], bool]:
    if isinstance(type_hint, typing.TypeVar):
        assert type_hint in type_params_map
        return type_params_map[type_hint], True
    elif getattr(type_hint, "__parameters__", []):
        return type_hint[tuple(type_params_map[tp] for tp in type_hint.__parameters__)], True
    else:
        return type_hint, False


def _make_counting_attr_from_attribute(
    field_attrib: Attribute, *, include_type: bool = False, **kwargs: Any
) -> Any:  # attr.s lies on purpose in some typings
    members = [
        "default",
        "validator",
        "repr",
        "eq",
        "order",
        "hash",
        "init",
        "metadata",
        "converter",
        "kw_only",
        "on_setattr",
    ]
    if include_type:
        members.append("type")

    return attr.ib(**{key: getattr(field_attrib, key) for key in members}, **kwargs)  # type: ignore[call-overload]  # too hard for mypy


def _make_dataclass_params_from_cls(cls: DataModelTp) -> dataclasses._DataclassParams:
    assert hasattr(cls, _MODEL_FIELDS)
    model_options = getattr(cls, _MODEL_OPTIONS)
    return dataclasses._DataclassParams(
        **{
            key: getattr(model_options, key)
            for key in ("init", "repr", "eq", "order", "unsafe_hash", "frozen")
        }
    )


def _make_dataclass_field_from_attr(field_attrib: Attribute) -> dataclasses.Field:
    MISSING = getattr(dataclasses, "MISSING", NOTHING)
    default = MISSING
    default_factory = MISSING
    if isinstance(field_attrib.default, attr.Factory):  # type: ignore[arg-type]  # attr.s lies on purpose in some typings
        default_factory = field_attrib.default.factory  # type: ignore[union-attr]  # attr.s lies on purpose in some typings
    elif field_attrib.default is not attr.NOTHING:
        default = field_attrib.default

    assert field_attrib.eq == field_attrib.order  # dataclasses.compare == (attr.eq and attr.order)

    dataclasses_field = dataclasses.Field(  # type: ignore[call-arg,var-annotated]  # dataclasses.Field signature seems invisible to mypy
        default=default,
        default_factory=default_factory,
        init=field_attrib.init,
        repr=field_attrib.repr if not callable(field_attrib.repr) else None,
        hash=field_attrib.hash,
        compare=field_attrib.eq,
        metadata=field_attrib.metadata,
    )
    dataclasses_field.name = field_attrib.name
    assert field_attrib.type is not None
    dataclasses_field.type = field_attrib.type
    dataclasses_field._field_type = dataclasses._FIELD  # type: ignore[attr-defined] # dataclasses._FIELD not visible

    return dataclasses_field


def _make_non_instantiable_init() -> Callable[..., None]:
    def __init__(self: DataModelTp, *args: Any, **kwargs: Any) -> None:
        raise TypeError(f"Trying to instantiate '{type(self).__name__}' abstract class.")

    return __init__


def _make_post_init(has_post_init: bool) -> Callable[[DataModelTp], None]:
    if has_post_init:

        def call_post_init(self):
            self.__post_init__()

    else:

        def call_post_init(self):
            pass

    def __attrs_post_init__(self: DataModelTp) -> None:
        cls = type(self)
        for initializer in cls.__datamodel_initializers__:
            initializer(self)
        if attr._config._run_validators is True:  # type: ignore[attr-defined]  # attr._config is not visible for mypy
            for validator in cls.__datamodel_validators__:
                validator(cls, self)
        call_post_init(self)

    return __attrs_post_init__


def _make_devtools_pretty() -> Callable[
    [DataModelTp, Callable[[Any], Any]], Generator[Any, None, None]
]:
    def __pretty__(
        self: DataModelTp, fmt: Callable[[Any], Any], **kwargs: Any
    ) -> Generator[Any, None, None]:
        """Provide a human readable representation for `devtools <https://python-devtools.helpmanual.io/>`_.

        Note:
            Adapted from `pydantic <https://github.com/samuelcolvin/pydantic>`_.
        """
        yield self.__class__.__name__ + "("
        yield 1
        for name in self.__datamodel_fields__.keys():
            yield name + "="
            yield fmt(getattr(self, name))
            yield ","
            yield 0
        yield -1
        yield ")"

    return __pretty__


def _make_data_model_class_getitem() -> classmethod:
    def __class_getitem__(
        cls: Type[GenericDataModelTp], args: Union[Type, Tuple[Type]]
    ) -> GenericDataModelAlias:
        """Return an instance compatible with aliases created by :class:`typing.Generic` classes.

        See :class:`GenericDataModelAlias` for further information.
        """
        type_args: Tuple[Type] = args if isinstance(args, tuple) else (args,)
        concrete_cls = concretize(cls, *type_args)
        return GenericDataModelAlias(concrete_cls, type_args)

    return classmethod(__class_getitem__)


def _make_datamodel(
    cls: Type,
    *,
    repr: bool,  # noqa: A002   # shadowing 'repr' python builtin
    eq: bool,
    order: bool,
    unsafe_hash: bool,
    frozen: bool,
    kw_only: bool,
    instantiable: bool,
    skip_private: bool,
) -> Type:
    """Actual implementation of the Data Model creation.

    See :func:`datamodel` for the description of the parameters.
    """
    if "__annotations__" not in cls.__dict__:
        cls.__annotations__ = {}
    annotations: Dict[str, Any] = cls.__dict__["__annotations__"]
    mro_bases: Tuple[Type, ...] = cls.__mro__[1:]
    canonicalized_annotations: Dict[str, Any] = typingx.get_canonical_type_hints(cls)

    # Create attrib definitions with automatic type validators (and converters)
    # for the annotated fields. The original annotations are used for iteration
    # since the resolved annotations may also contain superclasses' annotations
    for key in annotations:
        if skip_private and key.startswith("_"):
            continue
        type_hint = annotations[key] = canonicalized_annotations[key]
        if typing.get_origin(type_hint) is not ClassVar:
            type_validator = make_auto_type_validator(type_hint)
            if key not in cls.__dict__:
                setattr(cls, key, attr.ib(validator=type_validator))
            elif not isinstance(default_value := cls.__dict__[key], attr._make._CountingAttr):  # type: ignore[attr-defined]  # attr._make is not visible for mypy
                if not hasattr(default_value, _DERIVED_FIELD_TAG):
                    # If it is an derived field, it will be added later
                    setattr(cls, key, attr.ib(default=default_value, validator=type_validator))
            else:
                # A field() function has been used to customize the definition:
                # prepend the type validator to the list of provided validators (if any)
                cls.__dict__[key]._validator = (
                    type_validator
                    if cls.__dict__[key]._validator is None
                    else attr.validators.and_(type_validator, cls.__dict__[key]._validator)  # type: ignore[attr-defined]  # attr._make is not visible for mypy
                )

                # TODO(egparedes): implement type coercing
                # TODO: if cls.__dict__[key].converter is True:
                # TODO:    cls.__dict__[key].converter = _make_type_coercer(type_hint)

    # Add derived field definitions
    derived_fields_data = _collect_derived_fields(cls, canonicalized_annotations, skip_private)
    derived_fields = set()
    initializers = []
    for name, annotation, attr_def, initializer in derived_fields_data:
        if annotation:
            annotations.setdefault(name, annotation)
        setattr(cls, name, attr_def)
        derived_fields.add(name)
        initializers.append(initializer)

    setattr(cls, _FIELD_INITIALIZERS, tuple(initializers))
    if initializers:

        def attr_on_setattr(instance: DataModelTp, attribute: Attribute, value: Any) -> Any:
            new_values = fields_view(instance)
            object.__setattr__(new_values, attribute.name, value)

            # update derived fields using new value
            for initializer in type(instance).__datamodel_initializers__:
                initializer(instance, new_values)

            return value

    else:
        attr_on_setattr = None

    # All fields should be annotated with type hints
    for key, value in cls.__dict__.items():
        if skip_private and key.startswith("_"):
            continue
        if isinstance(value, attr._make._CountingAttr) and (  # type: ignore[attr-defined]  # attr._make is not visible for mypy
            key not in annotations
        ):
            raise TypeError(f"Missing type annotation in '{key}' field.")

    # Collect and validators
    field_validators = _collect_field_validators(cls)
    for field_name, field_validator in field_validators.items():
        field_c_attr = cls.__dict__.get(field_name, None)
        if not field_c_attr:
            # Field has not been defined in the current class namespace,
            # look for field definition in the base classes.
            base_field_attr = _get_attribute_from_bases(field_name, mro_bases, annotations)
            if base_field_attr:
                # Create a new field in the current class cloning the existing
                # definition and add the new validator (attrs recommendation)
                field_c_attr = _make_counting_attr_from_attribute(
                    base_field_attr,
                )
                setattr(cls, field_name, field_c_attr)
            else:
                raise TypeError(f"Validator assigned to non existing '{field_name}' field.")

        # Add field validator using field_attr.validator
        assert isinstance(field_c_attr, attr._make._CountingAttr)  # type: ignore[attr-defined]  # attr._make is not visible for mypy
        field_c_attr.validator(field_validator)

    root_validators = _collect_root_validators(cls)
    setattr(cls, _ROOT_VALIDATORS, tuple(root_validators))

    # Update class with attr.s features
    if "__init__" in cls.__dict__:
        raise TypeError(
            "datamodel(init=True) is incompatible with custom '__init__' methods, use '__post_init__' instead."
        )

    if not instantiable:
        cls.__init__ = _make_non_instantiable_init()
    else:
        # For dataclasses emulation, __attrs_post_init__ calls __post_init__ (if it exists)
        cls.__attrs_post_init__ = _make_post_init(has_post_init="__post_init__" in cls.__dict__)

    cls.__class_getitem__ = _make_data_model_class_getitem()

    attr_settings = {"auto_attribs": False, "slots": False}
    hash_arg = None if not unsafe_hash else True
    new_cls = attr.define(  # type: ignore[attr-defined]  # attr.define is not visible for mypy
        **attr_settings,
        init=instantiable,
        repr=repr,
        eq=eq,
        order=order,
        frozen=frozen,
        kw_only=kw_only,
        hash=hash_arg,
        on_setattr=attr_on_setattr,
    )(cls)
    assert new_cls is cls

    # Final postprocessing
    cls.__pretty__ = _make_devtools_pretty()
    setattr(
        cls,
        _MODEL_OPTIONS,
        DataModelOptions(
            init=True,
            repr=repr,
            eq=eq,
            order=order,
            unsafe_hash=unsafe_hash,
            frozen=frozen,
            kw_only=kw_only,
            instantiable=instantiable,
        ),
    )
    setattr(
        cls,
        _MODEL_FIELDS,
        utils.FrozenNamespace(
            **{
                f_attr.name: FieldInfo(
                    name=f_attr.name,
                    default=f_attr.default,
                    init=f_attr.init,
                    repr=f_attr.repr,
                    hash=f_attr.hash,
                    compare=f_attr.eq and f_attr.order,
                    kw_only=f_attr.kw_only,
                    metadata=f_attr.metadata,
                    type=f_attr.type,
                    converter=f_attr.converter,
                    validator=f_attr.validator,
                    inherited=f_attr.inherited,
                    derived=f_attr.name in derived_fields,
                    inherit_validators=False,
                    attrib_index=i,
                )
                for i, f_attr in enumerate(cls.__attrs_attrs__)
            }
        ),
    )

    # dataclasses emulation
    cls.__dataclass_params__ = _make_dataclass_params_from_cls(cls)
    cls.__dataclass_fields__ = {
        field_attr.name: _make_dataclass_field_from_attr(field_attr)
        for field_attr in cls.__attrs_attrs__
    }

    return cls


@utils.optional_lru_cache(maxsize=None, typed=True)
def _make_concrete_with_cache(
    datamodel_cls: Type[GenericDataModelTp],
    *type_args: Type,
    class_name: Optional[str] = None,
    module: Optional[str] = None,
) -> Type[DataModelTp]:
    if not is_generic(datamodel_cls):
        raise TypeError(f"'{datamodel_cls.__name__}' is not a generic model class.")
    for t in type_args:
        if not (isinstance(t, (type, type(None))) or getattr(t, "__module__", None) == "typing"):
            raise TypeError(
                f"Only 'type' and 'typing' definitions can be passed as arguments "
                f"to instantiate a generic model class (received: {type_args})."
            )
    if len(type_args) != len(datamodel_cls.__parameters__):
        raise TypeError(
            f"Instantiating '{datamodel_cls.__name__}' generic model with a wrong number of parameters "
            f"({len(type_args)} used, {len(datamodel_cls.__parameters__)} expected)."
        )

    # Replace field definitions with the new actual types for generic fields
    type_params_map = dict(zip(datamodel_cls.__parameters__, type_args))
    model_fields = getattr(datamodel_cls, _MODEL_FIELDS)
    model_attrs = getattr(datamodel_cls, "__attrs_attrs__")
    new_annotations = {}
    new_field_c_attrs = {}
    for field_name, field_type in typing.get_type_hints(datamodel_cls).items():
        new_annotation, replaced = _substitute_typevars(field_type, type_params_map)
        if replaced:
            new_annotations[field_name] = new_annotation
            field_attrib = model_attrs[getattr(model_fields, field_name).attrib_index]
            new_field_c_attrs[field_name] = _make_counting_attr_from_attribute(field_attrib)

    # Create new concrete class
    if not class_name:
        arg_names = []
        for tp_var in datamodel_cls.__parameters__:
            arg_string = tp_var.__name__
            if tp_var in type_params_map:
                concrete_arg = type_params_map[tp_var]
                if isinstance(concrete_arg, type):
                    arg_string = concrete_arg.__name__
                else:
                    arg_string = utils.slugify(
                        str(concrete_arg).replace("typing.", "").replace("...", "ellipsis")
                    )

            arg_names.append(arg_string)

        class_name = f"{datamodel_cls.__name__}__{'_'.join(arg_names)}"

    namespace = {
        "__annotations__": new_annotations,
        "__module__": module if module else datamodel_cls.__module__,
        **new_field_c_attrs,
    }

    concrete_cls = type(class_name, (datamodel_cls,), namespace)
    assert concrete_cls.__module__ == module or not module

    if _MODEL_FIELDS not in concrete_cls.__dict__:
        # If original model does not inherit from GenericModel,
        # _make_datamodel() hasn't been called yet, so call it now
        params = getattr(datamodel_cls, _MODEL_OPTIONS)
        concrete_cls = _make_datamodel(
            concrete_cls,
            **{
                name: getattr(params, name)
                for name in (
                    "repr",
                    "eq",
                    "order",
                    "unsafe_hash",
                    "frozen",
                    "kw_only",
                    "instantiable",
                )
            },
        )

    return concrete_cls


def is_datamodel(obj: Any) -> bool:
    """Return True if `obj` is a Data Model class or an instance of a Data Model."""
    cls = obj if isinstance(obj, type) else obj.__class__
    return hasattr(cls, _MODEL_FIELDS)


def is_generic(model: Union[DataModelTp, Type[DataModelTp]]) -> bool:
    """Return True if `model` is a generic Data Model class or an instance of a generic Data Model."""
    if not is_datamodel(model):
        raise TypeError(f"Invalid datamodel instance or class: '{model}'.")

    return len(getattr(model, "__parameters__", [])) > 0


def is_instantiable(model: Type[DataModelTp]) -> bool:
    """Return True if `model` is a instantiable Data Model class or an instance of a Data Model."""
    if not is_datamodel(model):
        raise TypeError(f"Invalid datamodel instance or class: '{model}'.")

    options = getattr(model, _MODEL_OPTIONS)
    assert hasattr(options, "instantiable") and isinstance(options.instantiable, bool)
    return options.instantiable


@typing.overload
def get_fields(
    model: Union[DataModelTp, Type[DataModelTp]], *, as_dataclass: Literal[False] = False
) -> utils.FrozenNamespace:
    ...


@typing.overload
def get_fields(
    model: Union[DataModelTp, Type[DataModelTp]], *, as_dataclass: Literal[True]
) -> Tuple[dataclasses.Field, ...]:
    ...


def get_fields(
    model: Union[DataModelTp, Type[DataModelTp]], *, as_dataclass: bool = False
) -> Union[utils.FrozenNamespace, Tuple[dataclasses.Field, ...]]:
    """Return the field meta-information of a Data Model.

    Arguments:
        model: A Data Model class or instance.

    Keyword Arguments:
        as_dataclass: If ``True`` (the default is ``False``), field information is returned
            as :class:`dataclass.Field` instances instead of :class:`Attribute`.

    Examples:
        >>> from typing import List
        >>> @datamodel
        ... class Model:
        ...     amount: int = 1
        ...     name: str
        ...     numbers: List[float]
        >>> fields(Model)  # doctest:+ELLIPSIS
        FrozenNamespace(amount=FieldInfo(name='amount', default=1, ...),\
 name=FieldInfo(name='name', default=NOTHING, ...),\
 numbers=FieldInfo(name='numbers', default=NOTHING, ...))

        >>> fields(Model, as_dataclass=True)  # doctest:+ELLIPSIS
        (Field(name='amount',type=<class 'int'>,default=1,default_factory=...),\
 Field(name='name',type=<class 'str'>,default=...),\
 Field(name='numbers',type=typing.List[float],default=...))

    """  # noqa: RST201  # doctest conventions confuse RST validator
    if not is_datamodel(model):
        raise TypeError(f"Invalid datamodel instance or class: '{model}'.")
    if not isinstance(model, type):
        model = model.__class__

    if as_dataclass:
        return dataclasses.fields(model)
    else:
        ns = getattr(model, _MODEL_FIELDS)
        assert isinstance(ns, utils.FrozenNamespace)
        return ns


fields = get_fields


def asdict(
    instance: DataModelTp,
    *,
    dict_factory: Type[Mapping[Any, Any]] = dict,
    retain_collection_types: bool = False,
) -> Dict[str, Any]:
    """Return the contents of a Data Model instance as a new mapping from field names to values.

    Arguments:
        instance: Data Model instance.

    Keyword Arguments:
        dict_factory: A callable to produce ``dict`` instances from.
        retain_collection_types: Do not convert to ``list`` when encountering an
            attribute whose type is ``tuple`` or ``set``.

    Examples:
        >>> @datamodel
        ... class C:
        ...     x: int
        ...     y: int
        >>> c = C(x=1, y=2)
        >>> assert asdict(c) == {'x': 1, 'y': 2}
    """  # noqa: RST301  # sphinx.napoleon conventions confuse RST validator
    if not is_datamodel(instance) or isinstance(instance, type):
        raise TypeError(f"Invalid datamodel instance: '{instance}'.")
    return attr.asdict(
        instance,
        dict_factory=dict_factory,
        recurse=True,
        retain_collection_types=retain_collection_types,
    )


as_dict = asdict


def astuple(
    instance: DataModelTp,
    *,
    tuple_factory: Type[Sequence[Any]] = tuple,
    retain_collection_types: bool = False,
) -> Tuple[Any, ...]:
    """Return the contents of a Data Model instance as a new tuple of field values.

    Arguments:
        instance: Data Model instance.

    Keyword Arguments:
        tuple_factory: A callable to produce ``tuple`` instances from.
        retain_collection_types: Do not convert to ``list`` or ``dict`` when
            encountering an attribute which type is ``tuple``, ``dict``
            or ``set``.

    Examples:
        >>> @datamodel
        ... class C:
        ...     x: int
        ...     y: int
        >>> c = C(x=1, y=2)
        >>> assert astuple(c) == (1, 2)
    """  # noqa: RST301  # sphinx.napoleon conventions confuse RST validator
    if not is_datamodel(instance) or isinstance(instance, type):
        raise TypeError(f"Invalid datamodel instance: '{instance}'.")
    return attr.astuple(
        instance,
        tuple_factory=tuple_factory,
        recurse=True,
        retain_collection_types=retain_collection_types,
    )


as_tuple = astuple


def fields_view(instance: DataModelTp) -> utils.FrozenNamespace:
    if not is_datamodel(instance) or isinstance(instance, type):
        raise TypeError(f"Invalid datamodel instance: '{instance}'.")

    fields_meta = getattr(instance, _MODEL_FIELDS)
    assert isinstance(fields_meta, utils.FrozenNamespace)

    return utils.FrozenNamespace(
        **{field_name: getattr(instance, field_name, NOTHING) for field_name in fields_meta.keys()}
    )


def update_forward_refs(
    model: Union[DataModelTp, Type[DataModelTp]],
    local_ns: Optional[Dict[str, Any]] = None,
    *,
    fields: Optional[List[str]] = None,
) -> None:
    """Update Data Model class meta-information replacing forwarded type annotations with actual types.

    Arguments:
        local_ns: locals dict used in the evaluation of the annotations
            (globals are automatically taken from model.__module__).

    Keyword Arguments:
        fields: list with the names of the fields to be updated. If none provide,
            all fields will be checked.
    """
    if not is_datamodel(model):
        raise TypeError(f"Invalid datamodel instance or class: '{model}'.")
    if not isinstance(model, type):
        model = model.__class__

    if not fields:
        fields = list(model.__datamodel_fields__.keys())

    datamodel_fields_ns = getattr(model, _MODEL_FIELDS)
    updated_fields: Dict[str, Attribute] = {}
    try:
        field_attr = None
        for field_name in fields:
            field_attr = getattr(datamodel_fields_ns, field_name)
            if "ForwardRef(" in repr(field_attr.type):
                actual_type = typingx.resolve_type(
                    field_attr.type,
                    sys.modules[model.__module__].__dict__,
                    local_ns,
                    allow_partial=False,
                )
                new_attr = dataclasses.replace(field_attr, type=actual_type)
                object.__setattr__(datamodel_fields_ns, field_name, new_attr)
                updated_fields[field_name] = new_attr

    except Exception as e:
        raise TypeError(
            f"Unexpected error trying to solve '{field_name}' field annotation ('{getattr(field_attr, 'type', None)}')"
        ) from e

    if updated_fields:
        model.__attrs_attrs__ = tuple(updated_fields.get(a.name, a) for a in model.__attrs_attrs__)
        for f_name, f_info in model.__dataclass_fields__.items():
            if f_name in updated_fields:
                actual_type = updated_fields[f_name].type  # type: ignore[assignment]  # updated_fiels[...].type is never None
                f_info.type = actual_type


def concretize(
    datamodel_cls: Type[GenericDataModelTp],
    /,
    *type_args: Type,
    class_name: Optional[str] = None,
    module: Optional[str] = None,
    support_pickling: bool = True,  # noqa
    overwrite_definition: bool = True,
) -> Type[DataModelTp]:
    """Generate a new concrete subclass of a generic Data Model.

    Arguments:
        datamodel_cls: Generic Data Model to be subclassed.
        type_args: Type defintitions replacing the `TypeVars` in
            ``datamodel_cls.__parameters__``.

    Keyword Arguments:
        class_name: Name of the new concrete class. The default value is the
            same of the generic Data Model replacing the `TypeVars` by the provided
            `type_args` in the name.
        module: Value of the ``__module__`` attribute of the new class.
            The default value is the name of the module containing the generic Data Model.
        support_pickling: If ``True``, support for pickling will be added
            by actually inserting the new class into the target `module`.
        overwrite_definition: If ``True``, a previous definition of the class in
            the target module will be overwritten.

    """  # noqa: RST301  # doctest conventions confuse RST validator
    concrete_cls = _make_concrete_with_cache(
        datamodel_cls, *type_args, class_name=class_name, module=module
    )
    assert isinstance(concrete_cls, type) and is_datamodel(concrete_cls)

    # For pickling to work, the new class has to be added to the proper module
    if support_pickling:
        class_name = concrete_cls.__name__
        reference_module_globals = sys.modules[concrete_cls.__module__].__dict__
        if reference_module_globals.get(class_name, None) is not concrete_cls:
            if class_name in reference_module_globals and not overwrite_definition:
                warnings.warn(
                    f"Existing '{class_name}' symbol in module '{module}' contains a reference"
                    "to a different object.",
                    RuntimeWarning,
                )
            else:
                reference_module_globals[class_name] = concrete_cls

    return concrete_cls


def validator(name: str) -> Callable[[ValidatorType], ValidatorType]:
    """Define a custom field validator for a specific field (decorator function).

    Arguments:
        name: Name of the field to be validated by the decorated function.

    The decorated functions should have the following signature:
    ``def _validator_function(self, attribute, value) -> None:``
    where ``self`` will be the model instance being validated, ``attribute``
    the definition information of the attribute (the value of
    ``__datamodel_fields__.field_name``) and ``value`` the actual value
    received for this field.
    """
    assert isinstance(name, str)

    def _field_validator_maker(func: ValidatorType) -> ValidatorType:
        setattr(func, _FIELD_VALIDATOR_TAG, name)
        return func

    return _field_validator_maker


def root_validator(func: RootValidatorType, /) -> RootValidatorType:
    """Define a custom root validator (decorator function).

    The decorated functions should have the following signature:
    ``def _root_validator_function(cls, instance) -> None:``
    where ``cls`` will be the class of the model and ``instance`` the
    actual instance being validated.
    """
    setattr(func, _ROOT_VALIDATOR_TAG, None)
    return func


def derived_field(
    initializer: Optional[Callable[[DataModelTp], V]] = None,
    *,
    repr: bool = True,  # noqa: A002   # shadowing 'repr' python builtin
    hash: Optional[bool] = None,  # noqa: A002   # shadowing 'hash' python builtin
    compare: bool = True,
    metadata: Optional[Mapping[Any, Any]] = None,
) -> Union[
    Callable[[DataModelTp], V], Callable[[Callable[[DataModelTp], V]], Callable[[DataModelTp], V]]
]:
    def _derived_field_maker(initializer: Callable[[DataModelTp], V]) -> Callable[[DataModelTp], V]:
        if not callable(initializer):
            raise TypeError(
                f"Invalid derived field initializer function: '{initializer}' (expected callable)."
            )

        setattr(
            initializer,
            _DERIVED_FIELD_TAG,
            utils.FrozenNamespace(repr=repr, hash=hash, compare=compare, metadata=metadata),
        )
        return initializer

    return _derived_field_maker(initializer) if initializer else _derived_field_maker


def field(
    *,
    default: Any = NOTHING,
    default_factory: Callable[[None], Any] = NOTHING,
    # default_if_none: bool = False,
    init: bool = True,
    repr: bool = True,  # noqa: A002   # shadowing 'repr' python builtin
    hash: Optional[bool] = None,  # noqa: A002   # shadowing 'hash' python builtin
    compare: bool = True,
    kw_only: bool = None,
    metadata: Optional[Mapping[Any, Any]] = None,
) -> Any:  # attr.s lies on purpose in some typings
    """Define a new attribute on a class with advanced options.

    Keyword Arguments:
        default: If provided, this will be the default value for this field.
            This is needed because the ``field()`` call itself replaces the
            normal position of the default value.
        default_factory: If provided, it must be a zero-argument callable that will
            be called when a default value is needed for this field. Among other
            purposes, this can be used to specify fields with mutable default values.
            It is an error to specify both `default` and `default_factory`.
        init: If ``True``, this field is included as a parameter to the
            generated ``__init__()`` method.
        repr: If ``True``, this field is included in the string returned
            by the generated ``__repr__()`` method.
        hash: This can be a ``bool`` or ``None``. If ``True``, this field is included
            in the generated ``__hash__()`` method. If ``None``, use the value of
            `compare`, which would normally be the expected behavior: a field
            should be considered in the `hash` if it’s used for comparisons.
            Setting this value to anything other than ``None`` is `discouraged`.
        compare: If ``True``, this field is included in the generated equality and
            comparison methods (__eq__(), __gt__(), et al.).
        kw_only: Make this attribute keyword-only in the generated ``__init__``.
        metadata: An arbitrary mapping, not used at all by Data Models, and provided
            only as a third-party extension mechanism. Multiple third-parties can each
            have their own key, to use as a namespace in the metadata.

    Examples:
        >>> from typing import List
        >>> @datamodel
        ... class C:
        ...     mylist: List[int] = field(default_factory=lambda : [1, 2, 3])
        >>> c = C()
        >>> c.mylist
        [1, 2, 3]

    Note:
        Currently implemented using :func:`attr.ib` from `attrs <https://www.attrs.org/>`_
    """
    defaults_kwargs = {}
    if default is not NOTHING and default_factory is not NOTHING:
        raise ValueError("Cannot specify both 'default' and 'default_factory'.")

    if default is not NOTHING:
        defaults_kwargs["default"] = default
    if default_factory is not NOTHING:
        defaults_kwargs["factory"] = default_factory

    # if default_if_none:
    #     if not defaults_kwargs:
    #         raise ValueError(
    #             "Cannot specify 'default_if_none' without 'default' or 'default_factory'."
    #         )
    #     converter = attr.converters.default_if_none(**defaults_kwargs)
    # else:
    #     converter = None

    return attr.ib(  # type: ignore[call-overload]  # attr.s lies on purpose in some typings
        **defaults_kwargs,
        init=init,
        repr=repr,
        hash=hash,
        eq=compare,
        order=compare,
        kw_only=kw_only,
        # converter=converter,
        metadata=metadata,
    )


def datamodel(
    cls: Type = None,
    /,
    *,
    repr: bool = True,  # noqa: A002   # shadowing 'repr' python builtin
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    kw_only: bool = True,
    instantiable: bool = True,
    skip_private: bool = False,
) -> Union[Type, Callable[[Type], Type]]:
    """Add generated special methods to classes according to the specified attributes (class decorator).

    Examines PEP 526 ``__annotations__`` to determine field types and creates
    strict type validation functions for the fields.

    Supported type annotations are:

        * specific types (e.g. ``int``, ``float``, ``str``, ``MyClass``, ...)
        * ``Any``
        * ``TypeVar`` (implemented as ``Any`` or `TypeVar`'s  upper bound)
        * ``Literal`` types
        * ``Union`` types
        * Collection types implementing the ``Sequence``, ``Set`` or
          ``Mapping`` interfaces

    Arguments:
        cls: Original class definition.

    Keyword Arguments:
        repr: If ``True``, a ``__repr__()`` method will be generated.
            If the class already defines ``__repr__()``, it will be overwritten.
        eq: If ``True``, ``__eq__()`` and ``__ne__()`` methods will be generated.
            This method compares the class as if it were a tuple of its fields.
            Both instances in the comparison must be of identical type.
        order:  If ``True``, add ``__lt__()``, ``__le__()``, ``__gt__()``,
            and ``__ge__()`` methods that behave like `eq` above and allow instances
            to be ordered. If ``None`` mirror value of `eq`.
        unsafe_hash: If ``False``, a ``__hash__()`` method is generated in a safe way
            according to how ``eq`` and ``frozen`` are set, or set to ``None`` (disabled)
            otherwise. If ``True``, a ``__hash__()`` method is generated anyway
            (use with care). See :func:`dataclasses.dataclass` for the complete explanation
            (or other sources like: `<https://hynek.me/articles/hashes-and-equality/>`_).
        frozen: If ``True``, assigning to fields will generate an exception.
            This emulates read-only frozen instances. The ``__setattr__()`` and
            ``__delattr__()`` methods should not be defined in the class.
        kw_only: Make all attributes keyword-only in the generated ``__init__``.
        instantiable: If ``False`` the class will contain an invalid ``__init__()``
            method that raises an exception.
        skip_private: If ``True``, skip private field definitions (names starting with `_`).

    Note:
        Currently implemented using :func:`attr.s` from `attrs <https://www.attrs.org/>`_
    """

    def _decorator(cls: Type) -> Type:
        return _make_datamodel(
            cls,
            repr=repr,
            eq=eq,
            order=order,
            unsafe_hash=unsafe_hash,
            frozen=frozen,
            kw_only=kw_only,
            instantiable=instantiable,
            skip_private=skip_private,
        )

    # This works for both @datamodel or @datamodel() decorations
    return _decorator(cls) if cls is not None else _decorator


@dataclasses.dataclass(frozen=True)
class DataModelOptions:
    init: bool
    repr: bool
    eq: bool
    order: bool
    unsafe_hash: bool
    frozen: bool
    kw_only: bool
    instantiable: bool


@dataclasses.dataclass(frozen=True)
class FieldInfo:
    name: str
    default: Any
    init: bool
    repr: bool
    hash: Optional[bool]
    compare: bool
    kw_only: bool
    metadata: Optional[Mapping[Any, Any]]
    type: Optional[Union[str, Type, TypeVar, ForwardRef]]
    converter: Optional[Callable]
    validator: Optional[ValidatorType]
    inherited: bool
    derived: bool
    inherit_validators: bool
    attrib_index: int


class DataModel(DataModelTp):
    """Base class to automatically convert any subclass into a Data Model.

    Inheriting from this class is equivalent to apply the :func:`datamodel`
    decorator to a class, except that all descendants will be also converted
    automatically in Data Models (which does not happen when explicitly
    applying the decorator).

    See :func:`datamodel` for the description of the parameters.
    """

    @classmethod
    def __init_subclass__(
        cls,
        /,
        *,
        repr: bool = True,  # noqa: A002   # shadowing 'repr' python builtin
        eq: bool = True,
        order: bool = False,
        unsafe_hash: bool = False,
        frozen: bool = False,
        kw_only: bool = True,
        instantiable: bool = True,
        skip_private: bool = False,
        **kwargs: Any,
    ) -> None:
        skip_datamodel_init = kwargs.pop("__skip_datamodel_init__", False)
        super().__init_subclass__(**kwargs)  # type: ignore[call-arg]  # super() does not need to be object
        if not skip_datamodel_init:
            _make_datamodel(
                cls,
                repr=repr,
                eq=eq,
                order=order,
                unsafe_hash=unsafe_hash,
                frozen=frozen,
                kw_only=kw_only,
                instantiable=instantiable,
                skip_private=skip_private,
            )


class FrozenDataModel(DataModel, __skip_datamodel_init__=True):
    """Base class to automatically convert any subclass into a frozen Data Model.

    See :class:`DataModel` for further reference.
    """

    @classmethod
    def __init_subclass__(
        cls,
        /,
        *,
        repr: bool = True,  # noqa: A002   # shadowing 'repr' python builtin
        eq: bool = True,
        order: bool = False,
        unsafe_hash: bool = False,
        kw_only: bool = True,
        instantiable: bool = True,
        skip_private: bool = False,
        **kwargs: Any,
    ) -> None:
        assert "frozen" not in kwargs
        super().__init_subclass__(
            repr=repr,
            eq=eq,
            order=order,
            unsafe_hash=unsafe_hash,
            frozen=True,
            kw_only=kw_only,
            instantiable=instantiable,
            skip_private=skip_private,
            **kwargs,
        )


# TODO(egparedes): implement type coercing
# TODO: def _make_type_coercer(type_hint: Type[T]) -> Callable[[Any], T]:
# TODO:     return type_hint if isinstance(type_hint, type) else lambda x: x  # type: ignore


# TODO(egparedes): implement full instance freezing
# TODO: def _frozen_setattr(instance, attribute, value):    # noqa: E800
# TODO:      raise attr.exceptions.FrozenAttributeError(
# TODO:         f"Trying to modify immutable '{attribute.name}' attribute in '{type(instance).__name__}' instance."
# TODO:      )


# TODO(egparedes): implement validation on attribute assignment
# TODO: def _valid_setattr(instance, attribute, value):
# TODO:     print("SET", attribute, value)
