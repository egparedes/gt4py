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

Examples: (Doctests disabled)
    <<< @datamodel
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


    <<< class AnotherSampleModel(DataModel):
    ...     name: str
    ...     friends: List[str]
    ...
    ...     @root_validator
    ...     def _root_validator(cls, instance):
    ...         if instace.name in instance.friends:
    ...             raise ValueError("'name' value cannot appear in 'friends' list.")
"""

from __future__ import annotations

import abc
import dataclasses
import functools
import sys
import types
import typing
import warnings

import attr
import attrs

from eve import extended_typing as xtyping
from eve import utils
from eve.extended_typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Final,
    ForwardRef,
    Generator,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from eve.type_definitions import NOTHING


# Typing
T = TypeVar("T")
V = TypeVar("V")

#
# class _AttrClassTp(Protocol):
#     __attrs_attrs__: ClassVar[Tuple[attr.Attribute, ...]]


# class _DataClassTp(Protocol):
#     __dataclass_fields__: ClassVar[Dict[str, dataclasses.Field]]

#     def __post_init__(self) -> None:
#         ...


# class _DevToolsPrettyPrintable(Protocol):
#     def __pretty__(self, fmt: Callable[[Any], Any], **kwargs: Any) -> Generator[Any, None, None]:
#         ...


# Attribute = attr.Attribute


# class DataModelTp(_AttrClassTp, _DataClassTp, _DevToolsPrettyPrintable, Protocol):
#     def __init__(self, *args: Any, **kwargs: Any) -> None:
#         ...

#     __datamodel_fields__: ClassVar[utils.FrozenNamespace[Attribute]]
#     __datamodel_params__: ClassVar[utils.FrozenNamespace[Type]]
#     __datamodel_root_validators__: ClassVar[
#         Tuple[typingx.NonDataDescriptor[DataModelTp, BoundRootValidatorType], ...]
#     ]


# class GenericDataModelTp(DataModelTp, Protocol):
#     __args__: ClassVar[Tuple[Union[Type, TypeVar]]]
#     __parameters__: ClassVar[Tuple[TypeVar]]
#     __class__: ClassVar[DataModelTp]  # type: ignore[assignment]

#     @classmethod
#     def __class_getitem__(
#         cls: Type[GenericDataModelTp], args: Union[Type, Tuple[Type]]
#     ) -> GenericDataModelAlias:
#         ...

Attribute = attr.Attribute
DataModelTp = Any

ValidatorType = Callable[[DataModelTp, Attribute, T], None]
BoundValidatorType = Callable[[Attribute, T], None]

RootValidatorType = Callable[[Type[DataModelTp], DataModelTp], None]
BoundRootValidatorType = Callable[[DataModelTp], None]

TypeValidationFactory = Callable[[xtyping.SourceTypingAnnotation], ValidatorType]


@typing.runtime_checkable
class TypeWithAttrValidatorTp(Protocol):
    """Protocol for classes defining its own custom validator (when used as fields)."""

    @classmethod
    @abc.abstractmethod
    def __type_validator__(self) -> ValidatorType:
        raise NotImplementedError()


# Implementation
_FIELD_VALIDATOR_TAG: Final = "__DATAMODEL_FIELD_VALIDATOR_TAG"
_ROOT_VALIDATOR_TAG: Final = "__DATAMODEL_ROOT_VALIDATOR_TAG"

_MODEL_FIELDS: Final = "__datamodel_fields__"
_MODEL_PARAMS: Final = "__datamodel_params__"
_ROOT_VALIDATORS: Final = "__datamodel_root_validators__"

_KNOWN_MUTABLE_TYPES: Final = (list, dict, set)
_CACHE_HASH_THRESHOLD: Final = 6


# -- Data Models --
T = TypeVar("T")


@typing.overload
def datamodel(
    cls: Literal[None] = None,
    /,
    *,
    repr: bool = True,  # noqa: A002  # shadowing 'repr' python builtin
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    match_args: bool = True,
    kw_only: bool = False,
    slots: bool = False,
    type_validation_factory: Optional[TypeValidationFactory] = None,
) -> Callable[[Type[T]], Type[T]]:
    ...


@typing.overload
def datamodel(
    cls: Type[T],
    /,
    *,
    repr: bool = True,  # noqa: A002  # shadowing 'repr' python builtin
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    match_args: bool = True,
    kw_only: bool = False,
    slots: bool = False,
    type_validation_factory: Optional[TypeValidationFactory] = None,
) -> Type[T]:
    ...


def datamodel(
    cls: Type[T] = None,
    /,
    *,
    repr: bool = True,  # noqa: A002  # shadowing 'repr' python builtin
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    match_args: bool = True,
    kw_only: bool = False,
    slots: bool = False,
    type_validation_factory: Optional[TypeValidationFactory] = None,
) -> Union[Type[T], Callable[[Type[T]], Type[T]]]:
    """Add generated special methods to classes according to the specified attributes (class decorator).

    Examines PEP 526 ``__annotations__`` to determine field types and creates
    strict type validation functions for the fields.

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

    Note:
        Currently implemented using :func:`attr.s` from `attrs <https://www.attrs.org/>`_
    """

    kwargs = {
        "repr": repr,
        "eq": eq,
        "order": order,
        "unsafe_hash": unsafe_hash,
        "frozen": frozen,
        "match_args": match_args,
        "kw_only": kw_only,
        "slots": slots,
        "type_validation_factory": type_validation_factory,
    }

    if cls is None:  # called as @datamodel()
        return functools.partial(_make_datamodel, **kwargs)
    else:  # called as @datamodel
        return _make_datamodel(cls, **kwargs, stacklevel_offset=1)


# class DataModel(DataModelTp):
class DataModel:
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
        match_args: bool = True,
        kw_only: bool = False,
        slots: bool = False,
        type_validation_factory: Optional[TypeValidationFactory] = None,
        **kwargs: Any,
    ) -> None:
        super(DataModel, cls).__init_subclass__(
            **kwargs
        )  # type: ignore[call-arg]  # super() does not need to be object
        _make_datamodel(
            cls,
            repr=repr,
            eq=eq,
            order=order,
            unsafe_hash=unsafe_hash,
            frozen=frozen,
            match_args=match_args,
            kw_only=kw_only,
            slots=slots,
            type_validation_factory=type_validation_factory,
            stacklevel_offset=1,
        )


def field(
    *,
    default: Any = NOTHING,
    default_factory: Optional[Callable[[None], Any]] = None,
    init: bool = True,
    repr: bool = True,  # noqa: A002   # shadowing 'repr' python builtin
    hash: Optional[bool] = None,  # noqa: A002   # shadowing 'hash' python builtin
    compare: bool = True,
    metadata: Optional[Mapping[Any, Any]] = None,
    kw_only: bool = False,
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
        metadata: An arbitrary mapping, not used at all by Data Models, and provided
            only as a third-party extension mechanism. Multiple third-parties can each
            have their own key, to use as a namespace in the metadata.

    Examples:  (Doctests disabled)
        <<< from typing import List
        <<< @datamodel
        ... class C:
        ...     mylist: List[int] = field(default_factory=lambda : [1, 2, 3])
        <<< c = C()
        <<< c.mylist
        [1, 2, 3]

    Note:
        Currently implemented using :func:`attr.ib` from `attrs <https://www.attrs.org/>`_
    """
    if default is not NOTHING and default_factory is not None:
        raise ValueError("Cannot specify both 'default' and 'default_factory'.")

    if default is not NOTHING:
        defaults_kwargs = {"default": default}
    elif default_factory is not None:
        defaults_kwargs = {"factory": default_factory}
    else:
        defaults_kwargs = {}

    return attrs.field(  # type: ignore[call-overload]  # attrs lies on purpose in some typings
        **defaults_kwargs,
        init=init,
        repr=repr,
        hash=hash,
        eq=compare,
        order=compare,
        metadata=metadata,
        kw_only=kw_only,
    )


def validator(name: str) -> Callable[[Callable], Callable]:
    """Define a custom field validator for a specific field (decorator function).

    Arguments:
        name: Name of the field to be validated by the decorated function.

    The decorated functions should have the following signature:
    ``def _validator_function(self, attribute, value):``
    where ``self`` will be the model instance being validated, ``attribute``
    the definition information of the attribute (the value of
    ``__datamodel_fields__.field_name``) and ``value`` the actual value
    received for this field.
    """
    assert isinstance(name, str)

    def _field_validator_maker(func: Callable) -> Callable:
        setattr(func, _FIELD_VALIDATOR_TAG, name)
        return func

    return _field_validator_maker


def root_validator(func: Callable, /) -> classmethod:
    """Define a custom root validator (decorator function).

    The decorated functions should have the following signature:
    ``def _root_validator_function(cls, instance):``
    where ``cls`` will be the class of the model and ``instance`` the
    actual instance being validated.
    """
    cls_method = classmethod(func)
    setattr(cls_method, _ROOT_VALIDATOR_TAG, None)
    return cls_method


# -- Utils --
def is_datamodel(obj: Any) -> bool:
    """Return True if `obj` is a Data Model class or an instance of a Data Model."""
    cls = obj if isinstance(obj, type) else obj.__class__
    return hasattr(cls, _MODEL_FIELDS)


def is_generic(model: Union[DataModelTp, Type[DataModelTp]]) -> bool:
    """Return True if `model` is a generic Data Model class or an instance of a generic Data Model."""
    if not is_datamodel(model):
        raise TypeError(f"Invalid datamodel instance or class: '{model}'.")

    return len(getattr(model, "__parameters__", [])) > 0


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

    Examples: (Doctests disabled)
        <<< from typing import List
        <<< @datamodel
        ... class Model:
        ...     amount: int = 1
        ...     name: str
        ...     numbers: List[float]
        <<< fields(Model)  # doctest:+ELLIPSIS
        FrozenNamespace(amount=Attribute(name='amount', default=1, ...),\
 name=Attribute(name='name', default=NOTHING, ...),\
 numbers=Attribute(name='numbers', default=NOTHING, ...))

        <<< fields(Model, as_dataclass=True)  # doctest:+ELLIPSIS
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

    Examples:  (Doctests disabled)
        <<< @datamodel
        ... class C:
        ...     x: int
        ...     y: int
        <<< c = C(x=1, y=2)
        <<< assert asdict(c) == {'x': 1, 'y': 2}
    """  # noqa: RST301  # sphinx.napoleon conventions confuse RST validator
    if not is_datamodel(instance) or isinstance(instance, type):
        raise TypeError(f"Invalid datamodel instance: '{instance}'.")
    return attr.asdict(
        instance,
        dict_factory=dict_factory,
        recurse=True,
        retain_collection_types=retain_collection_types,
    )


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

    Examples:  (Doctests disabled)
        <<< @datamodel
        ... class C:
        ...     x: int
        ...     y: int
        <<< c = C(x=1, y=2)
        <<< assert astuple(c) == (1, 2)
    """  # noqa: RST301  # sphinx.napoleon conventions confuse RST validator
    if not is_datamodel(instance) or isinstance(instance, type):
        raise TypeError(f"Invalid datamodel instance: '{instance}'.")
    return attr.astuple(
        instance,
        tuple_factory=tuple_factory,
        recurse=True,
        retain_collection_types=retain_collection_types,
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
            if isinstance((field_attr.type), ForwardRef):
                actual_type = xtyping.eval_forward_ref(
                    field_attr.type,
                    sys.modules[model.__module__].__dict__,
                    local_ns,
                    include_extras=True,
                )
                new_attr = field_attr.evolve(type=actual_type)
                object.__setattr__(datamodel_fields_ns, field_name, new_attr)
                updated_fields[field_name] = new_attr

    except Exception as e:
        raise TypeError(
            f"Unexpected error trying to solve '{field_name}' field annotation ('{getattr(field_attr, 'type', None)}')"
        ) from e

    if updated_fields:
        model.__attrs_attrs__ = tuple(updated_fields.get(a.name, a) for a in model.__attrs_attrs__)


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
        type_args: Type definitions replacing the `TypeVars` in
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
        if not (cls_in_module := (class_name in reference_module_globals)) or overwrite_definition:
            reference_module_globals[class_name] = concrete_cls
        elif cls_in_module and reference_module_globals[class_name] is not concrete_cls:
            warnings.warn(
                f"Existing '{class_name}' symbol in module '{module}' contains a reference"
                "to a different object.",
                RuntimeWarning,
            )

    return concrete_cls


# -- Helpers --
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


def _collect_field_validators(cls: Type) -> Dict[str, ValidatorType]:
    result = {}
    for member in cls.__dict__.values():
        if hasattr(member, _FIELD_VALIDATOR_TAG):
            field_name = getattr(member, _FIELD_VALIDATOR_TAG)
            result[field_name] = member
            delattr(member, _FIELD_VALIDATOR_TAG)

    return result


def _collect_root_validators(cls: Type) -> List[RootValidatorType]:
    result = []
    for base in reversed(cls.__mro__[1:]):
        for validator in getattr(base, _ROOT_VALIDATORS, []):
            if validator not in result:
                result.append(validator)

    for member in cls.__dict__.values():
        if hasattr(member, _ROOT_VALIDATOR_TAG):
            result.append(member)
            delattr(member, _ROOT_VALIDATOR_TAG)

    return result


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


def _make_post_init(has_post_init: bool) -> Callable[[DataModelTp], None]:
    # Duplicated code to facilitate the source inspection of the generated `__init__()` method
    if has_post_init:

        def __attrs_post_init__(self: DataModelTp) -> None:
            if attr._config._run_validators is True:  # type: ignore[attr-defined]  # attr._config is not visible for mypy
                for validator in self.__datamodel_root_validators__:
                    validator.__get__(self)(self)

            self.__post_init__()

    else:

        def __attrs_post_init__(self: DataModelTp) -> None:
            if attr._config._run_validators is True:  # type: ignore[attr-defined]  # attr._config is not visible for mypy
                for validator in type(self).__datamodel_root_validators__:
                    validator.__get__(self)(self)

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


if sys.version_info >= (3, 9):
    GenericTypeAlias: Final = types.GenericAlias
else:
    GenericTypeAlias: Final = typing._GenericAlias


def _make_data_model_class_getitem() -> classmethod:
    def __class_getitem__(
        cls: Type[GenericDataModelTp], args: Union[Type, Tuple[Type]]
    ) -> GenericTypeAlias:
        """Return an instance compatible with aliases created by :class:`typing.Generic` classes.

        See :class:`GenericDataModelAlias` for further information.
        """
        type_args: Tuple[Type] = args if isinstance(args, tuple) else (args,)
        concrete_cls = concretize(cls, *type_args)
        return GenericTypeAlias(concrete_cls, type_args)

    return classmethod(__class_getitem__)


def typeguard_validation_factory(annotation) -> Callable:
    import typeguard

    def _validator(cls, attrib, value):
        print(f"{attrib=}, {value=}")
        assert typeguard.check_type(attrib.name, value, attrib.type)

    return _validator


def _make_datamodel(
    cls: Type[T],
    *,
    repr: bool,  # noqa: A002   # shadowing 'repr' python builtin
    eq: bool,
    order: bool,
    unsafe_hash: bool,
    frozen: bool,
    match_args: bool,
    kw_only: bool,
    slots: bool,
    type_validation_factory: Optional[TypeValidationFactory] = None,
    stacklevel_offset: int = 0,
) -> Type[T]:
    """Actual implementation of the Data Model creation.

    See :func:`datamodel` for the description of the parameters.
    """
    if "__annotations__" not in cls.__dict__:
        cls.__annotations__ = {}
    annotations = cls.__dict__["__annotations__"]
    mro_bases: Tuple[Type, ...] = cls.__mro__[1:]
    partial_annotations = xtyping.get_partial_type_hints(cls)

    # Create attrib definitions with automatic type validators (and converters)
    # for the annotated fields. The original annotations are used for iteration
    # since the resolved annotations may also contain superclasses' annotations
    for key in annotations:
        type_hint = annotations[key] = partial_annotations[key]
        if typing.get_origin(type_hint) is not ClassVar:
            type_validator = type_validation_factory(type_hint) if type_validation_factory else None
            cls_attr_value = cls.__dict__.get(key, NOTHING)
            if cls_attr_value is NOTHING:
                # Missing definition
                setattr(cls, key, attrs.field(validator=type_validator))
            elif not isinstance(cls.__dict__[key], attr._make._CountingAttr):  # type: ignore[attr-defined]  # attr._make is not visible for mypy
                # Default value
                if isinstance(cls_attr_value, _KNOWN_MUTABLE_TYPES):
                    raise warnings.warn(
                        f"'{cls_attr_value.__class__.__name__}' value used as default in '{cls.__name__}.{key}'.\n"
                        "Mutable types should not be normally used as field defaults (use 'default_factory' instead).",
                        stacklevel=2 + stacklevel_offset,
                    )
                setattr(cls, key, attrs.field(default=cls_attr_value, validator=type_validator))
            else:
                # A field() function has been used to customize the definition:
                #   prepend the type validator to the list of provided validators (if any)
                cls.__dict__[key]._validator = (
                    type_validator
                    if cls_attr_value._validator is None
                    else attr._make.and_(type_validator, cls_attr_value._validator)  # type: ignore[attr-defined]  # attr._make is not visible for mypy
                )

                # TODO(egparedes): implement type coercing
                # TODO: if cls.__dict__[key].converter is True:
                # TODO:    cls.__dict__[key].converter = _make_type_coercer(type_hint)

    # All fields should be annotated with type hints
    num_attrs = 0
    for key, value in cls.__dict__.items():
        if isinstance(value, attr._make._CountingAttr):  # type: ignore[attr-defined]  # attr._make is not visible for mypy
            num_attrs += 1
            if (
                key not in annotations
                and typing.get_origin(partial_annotations.get(key, None)) is not ClassVar
            ):
                raise TypeError(f"Missing type annotation in '{key}' field.")

    # Validator processing
    root_validators = _collect_root_validators(cls)
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

    setattr(cls, _ROOT_VALIDATORS, tuple(root_validators))

    # Apply attrs magic
    if "__pre_init__" in cls.__dict__:
        cls.__attrs_pre_init__ = cls.__pre_init__

    cls.__attrs_post_init__ = _make_post_init(has_post_init="__post_init__" in cls.__dict__)
    cls.__class_getitem__ = _make_data_model_class_getitem()

    new_cls = attrs.define(  # type: ignore[attr-defined]  # attr.define is not visible for mypy
        auto_attribs=True,
        cache_hash=bool(frozen) and num_attrs >= _CACHE_HASH_THRESHOLD,
        repr=repr,
        eq=eq,
        order=order,
        hash=None if not unsafe_hash else True,
        frozen=frozen,
        match_args=match_args,
        kw_only=kw_only,
        slots=slots,
    )(cls)
    assert new_cls is cls or slots
    if "__attrs_init__" in new_cls.__dict__:
        new_cls.__auto_init__ = new_cls.__attrs_init__

    # Final postprocessing
    cls.__pretty__ = _make_devtools_pretty()
    setattr(
        cls,
        _MODEL_PARAMS,
        utils.FrozenNamespace(
            init=True,
            repr=repr,
            eq=eq,
            order=order,
            unsafe_hash=unsafe_hash,
            frozen=frozen,
        ),
    )
    setattr(
        cls,
        _MODEL_FIELDS,
        utils.FrozenNamespace(
            **{field_attr.name: field_attr for field_attr in cls.__attrs_attrs__}
        ),
    )

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
    new_annotations = {}
    new_field_c_attrs = {}
    for field_name, field_type in typing.get_type_hints(datamodel_cls).items():
        new_annotation, replaced = _substitute_typevars(field_type, type_params_map)
        if replaced:
            new_annotations[field_name] = new_annotation
            field_attrib = getattr(model_fields, field_name)
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
        params = getattr(datamodel_cls, _MODEL_PARAMS)
        concrete_cls = _make_datamodel(
            concrete_cls,
            **{
                name: getattr(params, name)
                for name in ("repr", "eq", "order", "unsafe_hash", "frozen")
            },
        )

    return concrete_cls
