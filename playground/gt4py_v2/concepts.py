"""Minimal stand-ins for the GT4Py concepts used in the GT4Py V2 notebooks.
Terminology, as in the "Semantics of domains, values and fields" notebook: a field maps every point
of its *domain* to a *value* of its *codomain* (the value type). The pairs `(point, value)`
form its *graph*. The *extent* of a stencil along a dimension is the `(min, max)` range of the
offsets at which it reads its arguments.

Staggered axes: an `Axis` has a staggered partner, `Staggered[K]`, the same axis at its
half-integer positions (e.g. the interfaces between the full levels, the layer
midpoints), an involution (`flip_staggered`), with extents declared, never both in one domain.
Half offsets, `K + 0.5`, premap fields between the integer and the half-integer positions of
an axis (see `Staggered`, `HalfOffset`).

Connectivities: on an unstructured mesh, the neighbors of a point are not found by offsets
but through a *relation*, e.g. `C2E` from cells to their edges, whose *neighbor table* is
provided at runtime by a `with Mesh(...)` block. Reading a field through a relation adds a
*neighbor* dimension, `Neighbor[C2E]`: the position of the neighbor in the row of the table.
Reads can be chained, `v(E2V)(C2E)`: each neighbor dimension knows its parent, the dimension it
hangs from. Through a *partial* relation, some neighbors are missing (at the boundary of the
mesh): a read either excludes them from its domain, `c(C2C)`, or reads a boundary value there,
`c(C2C.boundary(0.0))`. Relations are composed, `C2E @ E2C`, and transposed, `C2V.T` (the
footprint algebra); a composition of relations is a derived relation, read with one neighbor
dimension, `c(C2E @ E2C)`, and relations can be declared from such expressions, with a
`Stencil` (an ordered table, derived by the mesh) or a `reach`. The neighbors of a relation
flatten to its *pairs*, `Pairs[C2E]`, one point per neighbor, not per skip position (a sparse
field), with the *functional* relations `C2E.pair_source` and `C2E.pair_target` from each pair
to its ends. The `Extents` of a DSL operator are chains of reads, through which domains are
propagated.

Local meshes: on a rank of a distributed run, `with local_mesh:` provides a `LocalMesh`,
the part of the mesh on that rank, with its own numbering: the points it owns and a halo
around them, their global indices, and the named `Region`s, e.g. the lateral boundary. In
it, `Owned(Cell)` and `region(LateralBoundary)` are domains, e.g. to request results on. A
*local program* (in the implementation) runs on every rank, each in its local mesh: its fields
are values, valid on their domains, and it calls the *collectives* `exchange`, which fills the
halo from the owners, and `global_sum`, which DSL operators cannot call (`DSLRules`).

Logical meshes: a `Schema` brands the locations of its meshes; its families `Face[M]`,
`Edge[M]` and `Node[M]` (UGRID's names) are interned location classes, usable in
annotations and at runtime, and its roles (`class Coarse(Plate): ...`) declare their own
relations. A `LogicalMesh` is the whole mesh, with global indices and global semantics:
its `extents`, `regions` (semantic boundary regions, global index sets) and identity, the
`mesh_id`, brand its fields and its ranks; the decomposition (which rank owns what) is
separate, and built with `decompose` (in the implementation), as in L1. Its fields, the
`MeshField`s, are allocated through it, e.g. `grid.zeros(Cell, K)` or `grid.field(Cell,
data=...)`, and carry it: their premaps read its tables without a `with` block, and they are
combined only with fields of the same mesh (the brand, its `mesh_id`). Its `geometry` holds
named fields on the whole of their dimensions, e.g. the cell areas. A `Distribution` is how a
logical mesh is split into ranks, its local meshes: configuration, given or derived from the
reads of programs (in the implementation), never part of the mesh's identity.

Programs: a `@program` (in the implementation) is a Python function with global semantics on
the logical mesh of its field arguments: it calls DSL operators, possibly on requested domains,
e.g. `op[region(East)](t)` (in a program, `region` gives the global domain from the logical mesh,
`program_mesh`), sums fields over a `Location` with `sum_over(f, axis=Cell)` (a global,
reproducible sum), and combines fields point by point. It sees no decomposition (`Owned`, the
collectives and the field constructors are not in `PROGRAM_RULES`), and it does not look inside
fields: premaps, subscripts, writes and neighbor reductions at program level raise a `TypeError`
(compute with fields in field operators), so that the program means the same on any
decomposition. The implementation runs it directly on the logical mesh (the reference).

Only the happy path is supported. Checked: the neighbor tables (once, given or derived), the
rules of the kinds of dimensions (`Axis`, `Location`, `Label`, `Neighbor`), and that
restricting a field stays within its domain. Fields with different dimensions are broadcast
in arithmetic (see `Field`).
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import contextvars
import dataclasses
import enum
import functools
import inspect
import itertools
import textwrap
import types
from collections.abc import Callable, Iterator, Mapping
from types import EllipsisType, get_original_bases
from typing import (
    Any,
    ClassVar,
    Final,
    Literal,
    Protocol,
    Self,
    TypeAliasType,
    TypeIs,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
    runtime_checkable,
)

# --- Dimensions ---------------------------------------------------------------


class _DimensionType(type):
    """Metaclass of the dimensions, so that `K + 1` is the offset `(K, 1)`, for an `Axis` `K`.

    Checked at runtime only: all the kinds share this metaclass, so type checkers accept
    `Cell + 1` too. A float offset is a half offset to the staggered partner of the axis
    (see `Staggered`): `K + 0.5` is `(K, 0.5)`, of the type `HalfOffset`.
    """

    @overload
    def __add__(cls, offset: int) -> Offset: ...
    @overload
    def __add__(cls, offset: float) -> HalfOffset: ...
    def __add__(cls, offset: float) -> Offset | HalfOffset:
        if issubclass(cls, (Location, Label, Neighbor)):
            raise TypeError(f"'{cls.__name__}' has no offsets: only `Axis` dimensions have them.")
        if isinstance(offset, float):
            if (2 * offset) % 2 != 1:  # a half-integer: an odd multiple of one half
                raise ValueError(
                    f"'{cls.__name__} + {offset}': an offset is an integer, or a half-integer "
                    "to the staggered partner, e.g. `K + 0.5`."
                )
            return (cast(Dimension, cls), offset)
        return (cast(Dimension, cls), offset)

    @overload
    def __sub__(cls, offset: int) -> Offset: ...
    @overload
    def __sub__(cls, offset: float) -> HalfOffset: ...
    def __sub__(cls, offset: float) -> Offset | HalfOffset:
        return cls + -offset


class DimensionIndex(int, metaclass=_DimensionType):
    """An integer position along a dimension.

    A dimension is declared by subclassing one of its kinds, `Axis`, `Location` or `Label` (the
    fourth kind, `Neighbor`, is built from relations): `class IDim(Axis): ...`. Then `IDim.at(3)`
    is position 3 along `IDim` and `IDim.at(0):IDim.at(4)` is a slice of it. `IDim + 1` is an
    offset, used to premap fields: `f(IDim + 1)`; for an `Axis`, a half offset `K + 0.5`
    premaps to the staggered partner `Staggered[K]` (see `Staggered`). Subscripting a
    dimension only builds types, e.g. `Neighbor[C2E]`: positions and ranges are always
    `IDim.at(...)`.

    A plain subclass, `class IDim(DimensionIndex): ...`, is still allowed (legacy): it may be
    used like any kind, and none of the rules of the kinds are checked for it, except that
    `sum_over`, in the implementation, sums over `Neighbor` dimensions only.
    """

    @classmethod
    def at(cls, *values: int) -> Self | range:
        """`IDim.at(3)` is position 3 along `IDim`; `IDim.at(0, 4)` is `range(0, 4)`."""
        if len(values) == 1:
            return cls(values[0])
        start, stop = values
        return range(start, stop)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({int(self)})"


# The kinds of dimensions, the index spaces of fields (a `Neighbor` is the fourth kind, see
# "Connectivities"). Each kind says which domains, offsets and relations it allows.


class Location(DimensionIndex):
    """An unstructured index space, the locations of a mesh of one kind, e.g. its cells:
    `class Cell(Location): ...`.

    Its domains are ranges or index sets. It has no offsets: neighbors are found through the
    relations it is the source or target of.
    """


class Axis(DimensionIndex):
    """A Cartesian dimension, e.g. `class K(Axis): ...`: domains are ranges only, and it has
    offsets, `K + 1`. A half offset, `K + 0.5`, reads the field along its staggered partner
    `Staggered[K]` (see `Staggered`): `w(K + 0.5)` is `w` at `Staggered[K](i + c)`. It is
    never the source or target of a relation."""


class Label(DimensionIndex):
    """A finite dimension without topology, e.g. the tracers of a model: domains are ranges or
    index sets; no offsets, and never the source or target of a relation."""


class Staggered[A: Axis](Axis):
    """The staggered partner of a declared `Axis` `A`, at its half-integer positions:
    `Staggered[K](i)` is at position `i - 1/2`, e.g. the interfaces between the full levels
    `K` (the layer midpoints) of a vertical axis.

    An involution: `Staggered[Staggered[K]] is K` at runtime, and `flip_staggered(K) is
    Staggered[K]`. Extents are declared, never derived (a `K` of `nlev` full levels has
    `nlev + 1` interfaces), and a domain never holds both an axis and its staggered partner
    (see `check_domain`). Half offsets `K + 0.5` premap fields between them (see
    `Field.__call__`); integer offsets along `Staggered[K]` are ordinary shifts. Static
    limitation: `Staggered[Staggered[K]]` is a different nominal type from `K` for type
    checkers (write `K`).
    """

    base: ClassVar[Dimension]

    def __class_getitem__(cls, dim: Dimension) -> Dimension:  # type: ignore[override]
        return flip_staggered(dim)


def flip_staggered(dim: Dimension) -> Dimension:
    """Return the staggered partner of `dim`: its own `base` if `dim` is staggered, else the
    interned `Staggered[dim]` (see `Staggered`). A `TypeError` if `dim` is not an `Axis` or is
    `Staggered` itself."""
    if dim is Staggered or not issubclass(dim, Axis):
        raise TypeError(f"'{dim.__name__}' has no staggered partner: only an `Axis` has one.")
    if issubclass(dim, Staggered):
        return dim.base
    name = f"Staggered[{dim.__name__}]"
    namespace = {"base": dim, "__qualname__": name, "__module__": dim.__module__}
    return _staggered_partners.setdefault(dim, type(name, (Staggered,), namespace))


_staggered_partners: dict[Dimension, Dimension] = {}


type Dimension = type[DimensionIndex]
# The indices of the points of a domain along a `Location` or `Label` dimension, e.g. `Cell`,
# need not be consecutive: e.g. the cells around some edges. Sorted, without duplicates.
type IndexSet = tuple[int, ...]
# The domain along each dimension: a `range`, or an `IndexSet` except on `Axis` and `Neighbor`
# dimensions (see `check_domain`). The points of the domain are all the combinations.
type Domain = dict[Dimension, range | IndexSet]
type Offset = tuple[Dimension, int]  # e.g. `I + 1`
# A half offset, to the staggered partner of an `Axis` (see `Staggered`): `K + 0.5`.
type HalfOffset = tuple[Dimension, float]
# What a field can be premapped by: an offset (integer, or half to the staggered partner of
# an `Axis`, see `Staggered`), a relation, or a relation with a boundary value.
type Premap = Offset | HalfOffset | type[Relation[Any, Any]] | Boundary
# Field dimensions in order; also the type of a position in the domain, e.g. `(I(1), J(2))`.
type Dims[*Ds] = tuple[*Ds]
# A field subscript: dimension slices like `I.at(1):I.at(3)`, a `Domain`, e.g.
# `{Cell: (0, 2, 5)}`, or `...` for everything.
type Key = slice | tuple[slice, ...] | Domain | EllipsisType


def check_domain(domain: Domain) -> None:
    """Raise a `ValueError` if `domain` breaks the rules of the kinds of its dimensions: an
    index set on an `Axis` or a `Neighbor`, an index set that is not sorted without
    duplicates, or a staggered dimension together with its `base`."""
    for dim, indices in domain.items():
        if isinstance(indices, range):
            if issubclass(dim, Staggered) and dim.base in domain:
                raise ValueError(
                    f"A domain cannot have both '{dim.base.__name__}' and '{dim.__name__}': "
                    "they are the integer and the half-integer positions of one axis."
                )
            continue
        if issubclass(dim, (Axis, Neighbor)) or list(indices) != sorted(set(indices)):
            ranges_only = issubclass(dim, (Axis, Neighbor))
            rule = "are not allowed" if ranges_only else "must be sorted, without duplicates,"
            raise ValueError(f"Index sets {rule} along '{dim.__name__}'.")


# --- Fields -------------------------------------------------------------------


@runtime_checkable
class Field[D: tuple[DimensionIndex, ...], T](Protocol):
    """A value of the codomain `T` at every point of a `domain` with dimensions `D`.

    Annotate as `Field[Dims[I, J], float]`; `D` and `T` are only used in annotations.
    Arithmetic works on the values point by point; the domain of the result is computed
    forward: two fields are combined on the intersection of their domains. Fields with
    different dimensions are broadcast: the common dimensions are intersected, and the result
    has the dimensions of the left operand followed by the missing ones of the right, e.g.
    `(Cell, K) * (Cell,)` and `ef(V2E) * vf(V2V)` -> `(Vertex, Neighbor[V2E], Neighbor[V2V])`.
    Each field operator returns its result in the dimension order of its annotated result.
    """

    @property
    def domain(self) -> Domain: ...

    @overload
    def __call__(self, *premaps: Offset) -> Field[D, T]: ...
    @overload
    def __call__(self, *premaps: Premap) -> Field[Any, T]: ...
    def __call__(self, *premaps: Premap) -> Field[Any, T]:
        """Premap by offsets and relations, applied in order: `v(E2V, C2E)` is `v(E2V)(C2E)`.

        - `f(I + 1)` is the field whose value at `I(i)` is the value of `f` at `I(i + 1)`. Only
          the domain changes: it is shifted by `-1` along `I`. The values are the same.
        - A half offset `f(K + 0.5)` is the field whose value at `K(i)` is the value of `f` at
          `flip_staggered(K)(i + c)`, with `c = h + 1/2` for a declared axis and `c = h - 1/2`
          for a staggered one (`h` the offset, e.g. `0.5`): it premaps a field along one cell
          class of an axis to the other, e.g. the interfaces above and below each full level,
          `w(K - 0.5)` and `w(K + 0.5)`. Its domain along `K` is the field's domain along the
          partner, shifted by `-c`; the data is unchanged (see `Staggered`).
        - `e(C2E)`, with `e: Field[Dims[Edge, K], float]` and `C2E` a `Relation[Cell, Edge]`, is
          the field with dimensions `(Cell, Neighbor[C2E], K)` whose value at `(c, n, k)` is the
          value of `e` at the `n`-th neighbor edge of `c` and `k`: a gather through the neighbor
          table of `C2E` in the current mesh (see `Mesh`). Its domain along `Cell` is computed
          forward: the cells whose neighbors are all in the domain of `e`. Its values at skip
          positions (see `Relation`) are unspecified. The dimensions are in this order, but a
          field operator returns its result in the order of its annotation, e.g.
          `Field[Dims[K, Cell], float]`.
        - Through a partial relation `C2C` (see `Relation`), `c(C2C)` is defined on the cells
          whose neighbors are all present and in the domain of `c`. `c(C2C.boundary(value))`
          reads the scalar `value` at the missing neighbors: it is defined on the cells whose
          other neighbors are all in the domain of `c`, with the same `Neighbor[C2C]`. Only on
          fields without neighbor dimensions (a `NotImplementedError` otherwise).
        - A gather `f(R)`, with `R` a `Relation[O, T]`, on a field with neighbor dimensions
          hanging from `T` first renames them, recursively: `Neighbor[X, T]` becomes
          `Neighbor[X, Neighbor[R, O]]` (the data is the same). Then `T` is replaced by `O`,
          followed by `Neighbor[R, O]`, as above. So `v(E2V)(C2E)` has the dimensions
          `(Cell, Neighbor[C2E], Neighbor[E2V, Neighbor[C2E]])`, and `vf(V2V)(V2V)` has
          `(Vertex, Neighbor[V2V], Neighbor[V2V, Neighbor[V2V]])`.
        - A composition of relations is a relation (see "Footprint algebra"): `c(C2E @ E2C)` has
          the dimensions `(Cell, Neighbor[C2E @ E2C])`, one neighbor position per path. A staged
          expression with a transpose, e.g. `c(C2V.T)`, is a `TypeError`: declare a relation
          with it.
        - Through a functional relation (see `Relation`), e.g. `C2E.pair_target`, from
          `Pairs[C2E]` to `Edge`, `e(C2E.pair_target)` has the dimensions `(Pairs[C2E], K)`: the
          target is replaced by the source, without a neighbor dimension. Only on fields without
          neighbor dimensions (a `NotImplementedError` otherwise).

        In DSL operators, each premap is a step of a read of the `Extents`: the backends
        propagate domains through it, forward as above, or backward.
        """
        ...

    def __getitem__(self, key: Key) -> Field[D, T]:
        """Restrict to a sub-domain, e.g. `f[I.at(1):I.at(3)]` or `f[{Cell: (0, 2)}]`; `f[...]` is
        the whole field.

        Raise a `ValueError` if `key` selects points outside the domain, along the dimensions
        of the field; keys on other dimensions are ignored.
        """
        ...

    def __add__(self, other: Field | Scalar) -> Field: ...

    def __radd__(self, other: Field | Scalar) -> Field: ...

    def __sub__(self, other: Field | Scalar) -> Field: ...

    def __rsub__(self, other: Field | Scalar) -> Field: ...

    def __mul__(self, other: Field | Scalar) -> Field: ...

    def __rmul__(self, other: Field | Scalar) -> Field: ...


@runtime_checkable
class NDArrayField[D: tuple[DimensionIndex, ...], T](Field[D, T], Protocol):
    """A `Field` whose values are stored in an `NDArray`, one entry per point of the domain."""

    @property
    def data(self) -> NDArray:
        """The values, with one array axis per dimension of the domain, in the same order."""
        ...

    def __getitem__(self, key: Key) -> NDArrayField[D, T]: ...

    def __setitem__(self, key: Key, value: Field[D, T] | Scalar) -> None:
        """Write values into a sub-domain, e.g. `f[...] = other` or `f[I.at(1):I.at(3)] = 0`.

        Only the values of a field `value` on that sub-domain are read. The sub-domain must be
        within the domains of both fields, as for `__getitem__`. Implementations may forbid it
        where fields are values, e.g. in a running local program (a `RuntimeError`): a field
        must not claim values it does not have, e.g. a buffer of the local points written on
        the owned ones only.
        """
        ...


def dims_of(annotation: Any) -> tuple[Dimension, ...]:
    """Return the dimensions of a resolved field annotation: `Field[Dims[I, J], float]` -> `(I, J)`."""
    dims, _ = get_args(annotation)
    return get_args(dims)


# --- Connectivities -----------------------------------------------------------
# On an unstructured mesh, e.g. of cells, edges and vertices, each kind of location is a
# `Location` dimension (`class Cell(Location): ...`) and neighbors are found through relations.


class _RelationType(type):
    """Metaclass of the relations, for their footprint algebra, `R1 @ R2` and `R.T` (see
    `compose` and `Transpose`), and the derived relations of their pairs, `R.to_pairs`,
    `R.pair_source` and `R.pair_target` (see `Pairs`)."""

    @overload
    def __matmul__(cls, other: type[Relation[Any, Any]]) -> type[Relation[Any, Any]]: ...
    @overload
    def __matmul__(cls, other: Compose | Transpose) -> Compose: ...
    def __matmul__(cls, other: Expression) -> Expression:
        return compose(cast(type[Relation[Any, Any]], cls), other)

    @property
    def T(cls) -> Transpose:
        return Transpose(cast(type[Relation[Any, Any]], cls))

    @property
    def to_pairs(cls) -> type[Relation[Any, Any]]:
        return pair_relation(cast(type[Relation[Any, Any]], cls), "to_pairs")

    @property
    def pair_source(cls) -> type[Relation[Any, Any]]:
        return pair_relation(cast(type[Relation[Any, Any]], cls), "pair_source")

    @property
    def pair_target(cls) -> type[Relation[Any, Any]]:
        return pair_relation(cast(type[Relation[Any, Any]], cls), "pair_target")


class Relation[S: DimensionIndex, T: DimensionIndex](metaclass=_RelationType):
    """A relation from each point of the source dimension `S` to its neighbors along `T`.

    Declared by subclassing, with the most neighbors per source point, whether rows can have
    fewer of them (`skip_values`, e.g. the edges of the cells of a mesh of triangles and
    quadrilaterals: a position of a row holding `SKIP_VALUE` is a *skip position*, no neighbor),
    and whether neighbors can be missing at the boundary of the mesh
    (`partial`, e.g. the cells beside each cell of a strip, without a left one at its end):

        class C2E(Relation[Cell, Edge], max_neighbors=4, skip_values=True): ...
        class C2C(Relation[Cell, Cell], max_neighbors=2, partial=True): ...

    Reading through a partial relation needs a domain without the missing neighbors, `c(C2C)`,
    or a boundary value for them, `c(C2C.boundary(0.0))` (see `Field.__call__`).

    A relation is a type, never instantiated: it only names the connectivity. Its neighbor
    table, a `NeighborTable`, is only known at runtime, from the current `Mesh`. `S` and `T`
    are `Location` dimensions (or legacy ones): an `Axis`, `Label` or `Neighbor` raises a
    `TypeError` (at runtime only). A missing `max_neighbors` is a `TypeError` at runtime only
    too: the keywords are optional, since occurrences have none (and mypy does not check them:
    it skips the class keywords of classes with a metaclass, here the one of the footprint
    algebra): a `stencil` that is not a `Stencil`, or a `reach` that is not an `Expression`, is
    a `TypeError` at runtime.

    A relation may also carry its construction from other relations (see "Footprint algebra"):
    a `stencil`, from which the mesh derives its table, or a `reach`, the expression whose
    footprint holds the neighbors of each row, for tables in an order that cannot be derived:

        class C2E2C(Relation[Cell, Cell], max_neighbors=3, partial=True,
                    stencil=Stencil(C2E @ E2C, paths="exclude_self")): ...
        class C2E2Cr(Relation[Cell, Cell], max_neighbors=3, partial=True, reach=C2E @ E2C): ...

    Occurrences are nominal: a subclass of a relation, `class C2E_b(C2E): ...`, is another
    occurrence of the same relation: it inherits everything, and its table is the one of its
    `declared` relation, but it has its own neighbor dimension. So `e(C2E) * e(C2E_b)` pairs
    every two edges of a cell (an outer product per cell), while `e(C2E) * e(C2E)` pairs each
    edge with itself.

    A `functional` relation has exactly one neighbor per source (`max_neighbors=1`, neither
    skip values nor partial; else a `TypeError`): a gather through it replaces the target by
    the source, without a neighbor dimension, e.g. `e(C2E.pair_target)` (see `Pairs`).
    """

    source: ClassVar[Dimension]  # `S`
    target: ClassVar[Dimension]  # `T`
    max_neighbors: ClassVar[int]
    skip_values: ClassVar[bool]
    partial: ClassVar[bool]
    functional: ClassVar[bool]
    declared: ClassVar[type[Relation[Any, Any]]]  # itself, or the one it is an occurrence of
    stencil: ClassVar[Stencil | None]
    reach: ClassVar[Expression | None]
    # `(R, role)` for the derived relations of the pairs of `R`, e.g. `(C2E, "to_pairs")` for
    # `C2E.to_pairs` (see `pair_relation`), whose tables the mesh derives.
    pairs_role: ClassVar[tuple[type[Relation[Any, Any]], str] | None] = None

    def __init_subclass__(
        cls,
        *,
        max_neighbors: int | None = None,
        skip_values: bool | None = None,
        partial: bool | None = None,
        functional: bool | None = None,
        stencil: Stencil | None = None,
        reach: Expression | None = None,
    ) -> None:
        super().__init_subclass__()
        base = get_original_bases(cls)[0]
        if get_origin(base) is not Relation:  # an occurrence: everything is inherited
            if (max_neighbors, skip_values, partial, functional, stencil, reach) != (None,) * 6:
                raise TypeError(f"The occurrence '{cls.__name__}' inherits its configuration.")
            return
        if max_neighbors is None:
            raise TypeError(f"The relation '{cls.__name__}' needs `max_neighbors`.")
        if functional and (max_neighbors != 1 or skip_values or partial):
            raise TypeError(
                f"The functional relation '{cls.__name__}' has one neighbor per source: "
                "`max_neighbors=1`, without skip values, not partial."
            )
        cls.source, cls.target = get_args(base)
        for dim in (cls.source, cls.target):
            if issubclass(dim, (Axis, Label, Neighbor)):
                raise TypeError(f"'{dim.__name__}' cannot be in a relation: use a `Location`.")
        expression = reach is None or is_relation(reach) or isinstance(reach, Compose | Transpose)
        if not (isinstance(stencil, Stencil | None) and expression):
            raise TypeError(
                f"'{cls.__name__}' needs a `Stencil` as `stencil` and an expression of relations "
                f"as `reach`, not {stencil!r} and {reach!r}."
            )
        for construction in (stencil.expr if stencil else None, reach):
            if construction is not None and (construction.source, construction.target) != (
                cls.source,
                cls.target,
            ):
                raise TypeError(
                    f"'{cls.__name__}' cannot be built from `{construction}`: it relates "
                    f"'{construction.source.__name__}' to '{construction.target.__name__}'."
                )
        cls.max_neighbors = max_neighbors
        cls.skip_values = bool(skip_values)
        cls.partial = bool(partial)
        cls.functional = bool(functional)
        cls.declared = cls
        cls.stencil, cls.reach = stencil, reach

    def __init__(self) -> None:
        raise TypeError("A relation is a type: it is never instantiated.")

    @classmethod
    def boundary(cls, value: Scalar) -> Boundary:
        """The premap reading through this partial relation with `value` at the missing
        neighbors: `c(C2C.boundary(0.0))`. A `TypeError` if the relation is not partial."""
        if not cls.partial:
            raise TypeError(f"'{cls.__name__}' is not partial: it has no boundary.")
        return Boundary(cls, value)


# Footprint algebra: relations are composed, `C2E @ E2C` (the cells across the edges of each
# cell), and transposed, `C2V.T` (the cells around each vertex). These expressions are staged
# values, not types: annotations use declared relations, e.g. `Neighbor[C2E2C]`. A composition
# of relations is itself a relation, derived, anonymous and interned (the same class each time):
# `c(C2E @ E2C)` has ONE neighbor dimension, `Neighbor[C2E @ E2C]`, of width `w1 * w2`, one
# position per path (the cell itself is there once per edge), while the chained `c(E2C)(C2E)`
# has two. A transpose, or a composition with one, only has a table on a given mesh, whose width
# depends on the mesh: it is only used in the declarations of relations (`stencil=`, `reach=`).
# Footprints are sets: the neighbors of some sources, without order or multiplicity (`neighbors`
# in the implementation). Stencils are ordered: the tables of `Stencil`s.


class _Staged:
    """The footprint algebra of the staged expressions that are not relations."""

    def __matmul__(self, other: Expression) -> Compose:
        return cast(Compose, compose(cast(Expression, self), other))

    @property
    def T(self) -> Transpose:
        return Transpose(cast(Expression, self))


@dataclasses.dataclass(frozen=True, repr=False)
class Compose(_Staged):
    """`first @ second`, when one of them is not a relation, e.g. `C2V @ C2V.T`: the targets of
    `second` from the targets of `first`, along all the paths (see `compose`)."""

    first: Expression
    second: Expression

    @property
    def source(self) -> Dimension:
        return self.first.source

    @property
    def target(self) -> Dimension:
        return self.second.target

    def __repr__(self) -> str:
        second = expression_name(self.second)
        return f"{expression_name(self.first)} @ {f'({second})' if ' @ ' in second else second}"


@dataclasses.dataclass(frozen=True, repr=False)
class Transpose(_Staged):
    """`expr.T`: the sources of `expr` reaching each of its targets, in increasing order, padded
    with `SKIP_VALUE` (the table needs the extent of `expr.target`, see `Mesh`)."""

    expr: Expression

    @property
    def source(self) -> Dimension:
        return self.expr.target

    @property
    def target(self) -> Dimension:
        return self.expr.source

    def __repr__(self) -> str:
        name = expression_name(self.expr)
        return f"({name}).T" if " @ " in name else f"{name}.T"


# A staged expression of the footprint algebra: a relation (declared, an occurrence, or a
# derived composition), or a `Compose` or `Transpose` of them.
type Expression = type[Relation[Any, Any]] | Compose | Transpose


def expression_name(expr: Expression) -> str:
    return expr.__name__ if is_relation(expr) else repr(expr)


type _Relation = type[Relation[Any, Any]]
_compositions: dict[tuple[_Relation, _Relation], _Relation] = {}


def compose(first: Expression, second: Expression) -> Expression:
    """Return `first @ second`: `first`, then `second` (a `TypeError` unless `first.target is
    second.source`).

    Of two relations, with widths `w1` and `w2`, it is the derived relation, interned, whose row
    for a source lists, for each position of the row of `first`, the `w2` targets of `second`
    from its neighbor: `w1 * w2` positions, one per path, in this order; a skip position or
    missing neighbor of `first` gives `w2` of them. It has skip values or is partial if `first`
    or `second` is. Its table is derived by the mesh, from its `stencil`,
    `Stencil(Compose(first, second))`. Otherwise, it is a `Compose`, only used in declarations.
    """
    if first.target is not second.source:
        raise TypeError(
            f"Cannot compose `{expression_name(first)}` with `{expression_name(second)}`: "
            f"'{first.target.__name__}' is not '{second.source.__name__}'."
        )
    if not (is_relation(first) and is_relation(second)):
        return Compose(first, second)
    if (first, second) not in _compositions:
        name = repr(Compose(first, second))
        configuration = {
            "max_neighbors": first.max_neighbors * second.max_neighbors,
            "skip_values": first.skip_values or second.skip_values,
            "partial": first.partial or second.partial,
            "stencil": Stencil(Compose(first, second)),
        }
        namespace = {"__qualname__": name, "__module__": first.__module__}
        base = cast(Any, Relation)[first.source, second.target]
        relation = types.new_class(name, (base,), configuration, lambda ns: ns.update(namespace))
        # interned as in `_neighbor_dims`
        _compositions.setdefault((first, second), cast(_Relation, relation))
    return _compositions[first, second]


@dataclasses.dataclass(frozen=True)
class Stencil:
    """The construction of an ordered table from the paths of `expr`, the table of `expr`.

    `paths` says which entries of each row are kept:
    - "all": all of them, one per path, in order;
    - "exclude_self": for a composition `R1 @ R2` from a dimension to itself, the source is
      dropped from the targets of `R2` from each neighbor of `R1` (it must be there once),
      keeping the order, the skip positions and the missing neighbors: ICON's `C2E2C`, from
      `C2E @ E2C`. A skip position or missing neighbor of `R1` gives `w2 - 1` of them (`w2` the
      width of `R2`);
    - "unique": each neighbor once (the missing one too), in the order of first occurrence.
    With `include_self`, the source is prepended (ICON's `C2E2CO`); with "unique" too, before
    deduplicating: the source comes first, once. The rows are then padded
    with `SKIP_VALUE` to the `max_neighbors` of the relation (fewer is a `ValueError`).
    """

    expr: Expression
    paths: Literal["all", "exclude_self", "unique"] = "all"
    include_self: bool = False

    def __post_init__(self) -> None:
        to_itself = self.expr.source is self.expr.target
        if (self.paths == "exclude_self" or self.include_self) and not to_itself:
            raise TypeError(f"`{expression_name(self.expr)}` does not go back to its source.")
        if self.paths == "exclude_self" and composition_of(self.expr) is None:
            raise TypeError(f"`{expression_name(self.expr)}` is not a composition.")


def composition_of(expr: Expression) -> Compose | None:
    """Return `expr` as a `Compose`, also if it is a composition of relations, e.g. `C2E @ E2C`
    (a relation, built from all the paths of its `Compose`); `None` if it is not one."""
    stencil = expr.stencil if is_relation(expr) else None
    if stencil is not None and (stencil.paths, stencil.include_self) == ("all", False):
        return composition_of(stencil.expr)
    return expr if isinstance(expr, Compose) else None


@dataclasses.dataclass(frozen=True)
class Boundary:
    """A gather through the partial `relation` reading `value` at the missing neighbors, built
    by `relation.boundary(value)`. As a read step (see `Extents`), it has no value:
    `Boundary(C2C)`, since the value may be an argument, only known when called."""

    relation: type[Relation[Any, Any]]
    value: Scalar | None = None


def is_relation(obj: object) -> TypeIs[type[Relation[Any, Any]]]:
    return isinstance(obj, type) and issubclass(obj, Relation)


class Neighbor[R: Relation[Any, Any], P: DimensionIndex = Any](DimensionIndex):
    """The neighbor dimension of the relation `R`: the neighbors of each source of `R`.

    A point of `Neighbor[R]` is a position in a row of the table of `R`, not a point of
    `R.target`: the same target can be at several positions, e.g. the same cell on several
    paths of `C2E @ E2C`, or the same edge in the rows of several cells, and a skip position
    (see `Relation`) is not a neighbor of anything.

    `Neighbor[C2E]` is a dimension, of the fourth kind, one per relation and parent (the same
    class each time): its domains are ranges only, it has no offsets and is never in a relation.
    `e(C2E)` has the dimensions `(Cell, Neighbor[C2E], ...)`: its value at `(c, n, ...)` comes
    from the `n`-th neighbor of `c`. Reductions remove it, e.g. the DSL builtins
    `sum_over`, `max_over` and `min_over` of the implementation:
    `sum_over(e(C2E), axis=Neighbor[C2E])`.

    The parent `P` is the dimension the neighbors hang from: `R.source` (the default, a root),
    or a neighbor dimension `N` with `N.relation.target is R.source`. E.g. in `v(E2V)(C2E)`, the
    vertices hang from the edges of each cell: `Neighbor[E2V, Neighbor[C2E]]`. At runtime,
    `Neighbor[C2E]` is `Neighbor[C2E, Cell]`; type checkers only see `P` and cannot check it.
    """

    relation: ClassVar[type[Relation[Any, Any]]]
    parent: ClassVar[Dimension]

    def __class_getitem__(  # type: ignore[override]
        cls, params: type[Relation[Any, Any]] | tuple[type[Relation[Any, Any]], Dimension]
    ) -> Dimension:
        relation, parent = params if isinstance(params, tuple) else (params, None)
        return neighbor_of(relation, parent)


# Interned with `setdefault`: one class per key, also if the ranks of a local program, threads in
# the implementation, intern the same key at the same time.
_neighbor_dims: dict[tuple[type[Relation[Any, Any]], Dimension], Dimension] = {}


def neighbor_of(relation: type[Relation[Any, Any]], parent: Dimension | None = None) -> Dimension:
    """Return `Neighbor[relation, parent]`, e.g. for a relation known at runtime only.

    Raise a `TypeError` if `parent` is neither `relation.source` (the default) nor a neighbor
    dimension of a relation to `relation.source`.
    """
    parent = relation.source if parent is None else parent
    if parent is not relation.source and not (
        issubclass(parent, Neighbor) and parent.relation.target is relation.source
    ):
        raise TypeError(
            f"'{parent.__name__}' cannot be the parent of a neighbor of '{relation.__name__}': "
            f"it must be '{relation.source.__name__}' or a neighbor of a relation to it."
        )
    if (relation, parent) not in _neighbor_dims:
        name = relation.__name__ + ("" if parent is relation.source else f", {parent.__name__}")
        namespace = {
            "relation": relation,
            "parent": parent,
            "__qualname__": f"Neighbor[{name}]",
            "__module__": relation.__module__,
        }
        _neighbor_dims.setdefault(
            (relation, parent), cast(Dimension, type(f"Neighbor[{name}]", (Neighbor,), namespace))
        )
    return _neighbor_dims[relation, parent]


def root_of(dim: Dimension) -> Dimension:
    """Return the root of a neighbor dimension, following its parents (`Cell` for
    `Neighbor[E2V, Neighbor[C2E]]`), or `dim` itself if it is not a neighbor dimension."""
    while issubclass(dim, Neighbor):
        dim = dim.parent
    return dim


class Pairs[R: Relation[Any, Any]](Location):
    """The pairs of the relation `R`, flattened: a `Location` dimension with one point per
    neighbor in the table of `R` (not per skip position), in row-major order: the sources in the
    order of the table, then their neighbors in order. One per relation (the same class each
    time); a partial relation has none (a `TypeError`).

    A field with dimensions `(R.source, Neighbor[R], ...)`, padded at the skip positions, is a
    field with dimensions `(Pairs[R], ...)`, flattened, without them (a sparse field, like COO);
    the DSL builtins of the implementation `flatten(f, axis=Neighbor[R])` and `unflatten(p,
    axis=Pairs[R])` convert between them. A pair is a neighbor position, a path, not a distinct
    neighbor: through a composition, `Pairs[C2E @ E2C]` has a point per path, even if paths have
    the same ends.

    Three derived relations of `R` (see `pair_relation`), whose tables the mesh derives from the
    table of `R`: `R.to_pairs`, from `R.source` to `Pairs[R]`, the order-preserving map from the
    neighbors of `R` to the pairs (`SKIP_VALUE` at the skip positions), and the functional
    relations `R.pair_source` and `R.pair_target`, from each pair to its source and to its
    target along `R.target`: e.g. `e(C2E.pair_target) * c(C2E.pair_source)` has the dimensions
    `(Pairs[C2E],)`.
    """

    relation: ClassVar[type[Relation[Any, Any]]]

    def __class_getitem__(  # type: ignore[override]
        cls, relation: type[Relation[Any, Any]]
    ) -> Dimension:
        return pairs_of(relation)


_pairs: dict[_Relation, Dimension] = {}


def pairs_of(relation: _Relation) -> Dimension:
    """Return `Pairs[relation]`; a `TypeError` if `relation` is partial."""
    if relation.partial:
        raise TypeError(
            f"'{relation.__name__}' is partial: it has no pairs, since its missing neighbors "
            "have no target."
        )
    if relation not in _pairs:
        name = f"Pairs[{relation.__name__}]"
        namespace = {"relation": relation, "__qualname__": name, "__module__": relation.__module__}
        _pairs.setdefault(relation, cast(Dimension, type(name, (Pairs,), namespace)))
    return _pairs[relation]


type _Role = Literal["to_pairs", "pair_source", "pair_target"]
_pair_relations: dict[tuple[_Relation, _Role], _Relation] = {}


def pair_relation(relation: _Relation, role: _Role) -> _Relation:
    """Return the derived relation `relation.<role>` of the pairs of `relation` (see `Pairs`),
    interned, e.g. `C2E.to_pairs`: of the width of `relation`, with its skip values, or, for
    `pair_source` and `pair_target`, functional. A `TypeError` if `relation` is partial."""
    pairs = pairs_of(relation)
    if (relation, role) not in _pair_relations:
        name = expression_name(relation)
        name = f"({name}).{role}" if " @ " in name else f"{name}.{role}"
        source, target = {
            "to_pairs": (relation.source, pairs),
            "pair_source": (pairs, relation.source),
            "pair_target": (pairs, relation.target),
        }[role]
        configuration: dict[str, Any] = (
            {"max_neighbors": relation.max_neighbors, "skip_values": relation.skip_values}
            if role == "to_pairs"
            else {"max_neighbors": 1, "functional": True}
        )
        namespace = {"__qualname__": name, "__module__": relation.__module__}
        base = cast(Any, Relation)[source, target]
        derived = types.new_class(name, (base,), configuration, lambda ns: ns.update(namespace))
        cast(_Relation, derived).pairs_role = (relation, role)
        _pair_relations.setdefault((relation, role), cast(_Relation, derived))
    return _pair_relations[relation, role]


# The neighbor table of a relation `R`: an integer field with dimensions
# `(R.source, Neighbor[R])` whose value at `(s, k)` is the index along `R.target` of the `k`-th
# neighbor of `s`, or `SKIP_VALUE` if `s` has fewer neighbors (only if `R.skip_values`: the row
# is ragged), or `MISSING_VALUE` if the neighbor is missing at the boundary of the mesh (only if
# `R.partial`). Reductions skip the skip positions only: a missing neighbor read with a boundary
# value counts.
# In a `LocalMesh` only, `TRUNCATED` is a neighbor outside the local support, in a halo row; an
# implementation sentinel, never observable: a read reaching it is not computable (forward, the
# row is not in the domain; backward, an error), and it is neither skipped nor missing.
type NeighborTable = NDArrayField[Any, int]
SKIP_VALUE: Final = -1
MISSING_VALUE: Final = -2
TRUNCATED: Final = -3


def check_neighbor_table(
    relation: type[Relation[Any, Any]],
    table: NeighborTable,
    extents: Mapping[Dimension, int] | None = None,
    truncated: bool = False,
) -> None:
    """Raise a `ValueError` if `table` cannot be the neighbor table of `relation`.

    With the numbers of points of some dimensions, `extents` (see `Mesh`): if known, the table
    has a row per point of `relation.source`, and the neighbor indices are below the extent of
    `relation.target`. `TRUNCATED` is only valid with `truncated` (in a `LocalMesh`).
    """
    extents = extents or {}
    dims = (relation.source, neighbor_of(relation))
    if tuple(table.domain) != dims:
        raise ValueError(f"The table of '{relation.__name__}' must have the dimensions {dims}.")
    if table.domain[neighbor_of(relation)] != range(relation.max_neighbors):
        raise ValueError(
            f"The table of '{relation.__name__}' must have `max_neighbors` columns, "
            f"{relation.max_neighbors}."
        )
    source_extent = extents.get(relation.source)
    if source_extent is not None and table.domain[relation.source] != range(source_extent):
        raise ValueError(
            f"The table of '{relation.__name__}' must have a row per point of "
            f"'{relation.source.__name__}', {source_extent}."
        )
    target_extent = extents.get(relation.target)
    data = cast(Any, table.data)
    xp = data.__array_namespace__()
    valid = (
        (data >= 0)
        | ((data == SKIP_VALUE) & relation.skip_values)
        | ((data == MISSING_VALUE) & relation.partial)
        | ((data == TRUNCATED) & truncated)
    )
    if not bool(xp.all(valid)):
        raise ValueError(f"The table of '{relation.__name__}' has invalid neighbor indices.")
    if target_extent is not None and not bool(xp.all(data < target_extent)):
        raise ValueError(
            f"The table of '{relation.__name__}' has neighbor indices beyond the extent of "
            f"'{relation.target.__name__}', {target_extent}."
        )


class Mesh(Mapping[type[Relation[Any, Any]], NeighborTable]):
    """The neighbor tables of some relations, each checked once, when the mesh is built.
    A `LogicalMesh` is a mesh with a schema, regions and an identity (Phase 2), whose fields
    carry it (`MeshField`).

    A mesh is a context manager: it provides its tables to the DSL operators called in its
    `with` block, where `current_mesh()` returns it:

        with Mesh({C2E: c2e_table, E2V: e2v_table}, extents={Edge: 8}):
            result = op(e)

    or `m = Mesh(...)`, then `with m: ...` (`with m as the_mesh:` gives the mesh itself). An
    inner block, of another mesh or of the same one, provides its mesh until it ends; then the
    outer one is current again. Each thread has its own current mesh: a mesh can be entered in
    several threads at once. Lazy backends keep the mesh of the call to compute the values later.

    `mesh[R]` is the table of `R.declared`: an occurrence uses the table of its relation, and
    cannot be a key of `tables` (a `TypeError`). `extents` gives the number of points of some
    dimensions, e.g. `{Cell: 5}`: the tables from and to them are checked against it.

    The table of a relation with a `stencil` (e.g. a composition `C2E @ E2C`) is derived from
    the other tables when first used, then kept (checked like the given ones, a `ValueError` if
    it is wider than `max_neighbors`); a transpose needs the extent of its source, never
    inferred from the tables. A given table of such a relation must be the derived one (the
    same neighbors, in the same order, with the same skip positions and missing neighbors), and
    the rows of a given table of a relation with a `reach` must be within the footprint of its
    expression: else a `ValueError`. The tables of the relations of the pairs of a relation
    (`R.to_pairs`, `R.pair_source`, `R.pair_target`, see `Pairs`) are derived from its table in
    the same way. The derivation is provided by the implementation of the fields (the concepts
    have none): `construct`. The mesh of one rank of a distributed run is a `LocalMesh`.
    """

    # Derive the table of a relation with a `stencil` or a `pairs_role`, or check a given table
    # against its construction and return it, e.g. `implementation.construct_table`.
    construct: ClassVar[
        Callable[[Mesh, type[Relation[Any, Any]], NeighborTable | None], NeighborTable]
    ]
    # A `LocalMesh`: `TRUNCATED` is valid, given tables are not checked against their
    # constructions (done on the global mesh), and transposes are not derived.
    _local: ClassVar[bool] = False

    def __init__(
        self,
        tables: Mapping[type[Relation[Any, Any]], NeighborTable],
        extents: Mapping[Dimension, int] | None = None,
    ) -> None:
        self.extents: Mapping[Dimension, int] = dict(extents or {})
        for relation, table in tables.items():
            if relation.declared is not relation:
                raise TypeError(
                    f"'{relation.__name__}' is an occurrence: give the table of "
                    f"'{relation.declared.__name__}' instead."
                )
            check_neighbor_table(relation, table, self.extents, self._local)
        self._tables = dict(tables)
        self._derived: dict[type[Relation[Any, Any]], NeighborTable] = {}
        for relation, table in tables.items():
            if not self._local and (_constructed(relation) or relation.reach is not None):
                self.construct(relation, table)

    def __getitem__(self, relation: type[Relation[Any, Any]]) -> NeighborTable:
        declared = relation.declared
        if declared in self._tables:
            return self._tables[declared]
        if not _constructed(declared):
            raise LookupError(
                f"No neighbor table for '{relation.__name__}': use `with Mesh(...)` to provide it."
            )
        if declared not in self._derived:
            if self._local and declared.stencil and _has_transpose(declared.stencil.expr):
                raise LookupError(
                    f"A local mesh does not derive the table of '{declared.__name__}', from "
                    f"`{declared.stencil.expr}`: the halo rows of a transpose would miss the "
                    "sources outside the local support. Derive it on the global mesh: "
                    "`decompose(..., derived=[...])`."
                )
            table = self.construct(declared, None)
            check_neighbor_table(declared, table, self.extents, self._local)
            self._derived[declared] = table
        return self._derived[declared]

    def __iter__(self) -> Iterator[type[Relation[Any, Any]]]:
        return iter(self._tables)

    def __len__(self) -> int:
        return len(self._tables)

    def __enter__(self) -> Self:
        """Provide this mesh in the `with` block: `current_mesh()` returns it there."""
        _mesh_tokens.set((*_mesh_tokens.get(), (self, _current_mesh.set(self))))
        return self

    def __exit__(self, *exc_info: object) -> None:
        """End the `with` block of this mesh: a `RuntimeError` if it is not the innermost mesh
        entered in this thread, e.g. never entered, or exited out of order (by hand, or by a
        generator suspended in a `with` block). After an out-of-order exit, this mesh stays
        entered in this thread: `current_mesh()` can still return it after its block."""
        entries = _mesh_tokens.get()
        if not entries or entries[-1][0] is not self:
            if any(m is self for m, _ in entries):
                raise RuntimeError(
                    "Meshes exited out of order: this mesh is not the innermost one entered in "
                    "this thread. Exit them in the reverse order (nest their `with` blocks)."
                )
            raise RuntimeError(
                "Cannot exit a mesh not entered in this thread: use it in a `with` block."
            )
        _mesh_tokens.set(entries[:-1])
        _current_mesh.reset(entries[-1][1])


def _constructed(relation: type[Relation[Any, Any]]) -> bool:
    """Whether the mesh derives the table of `relation`: from a `stencil` or of pairs."""
    return relation.stencil is not None or relation.pairs_role is not None


def _has_transpose(expr: Expression) -> bool:
    """Whether the staged expression `expr` has a `Transpose` (not inside its relations)."""
    if isinstance(expr, Compose):
        return _has_transpose(expr.first) or _has_transpose(expr.second)
    return isinstance(expr, Transpose)


_current_mesh: contextvars.ContextVar[Mesh | None] = contextvars.ContextVar("mesh", default=None)
# The meshes entered and not yet exited, innermost last, each with the token resetting
# `_current_mesh` when it exits: per thread (each has its own context), so a mesh can be entered
# in several threads at once.
type _MeshEntry = tuple[Mesh, contextvars.Token[Mesh | None]]
_mesh_tokens: contextvars.ContextVar[tuple[_MeshEntry, ...]] = contextvars.ContextVar(
    "mesh_tokens", default=()
)


def current_mesh() -> Mesh:
    """Return the mesh of the innermost `with` block of a `Mesh`: an empty mesh outside them."""
    the_mesh = _current_mesh.get()
    return Mesh({}) if the_mesh is None else the_mesh  # a mesh without tables is falsy


# --- Local meshes -------------------------------------------------------------
# In a distributed run, each rank computes on its part of the mesh, a `LocalMesh`: the points it
# owns, and a halo of points owned by other ranks, which its reads need. Results are requested
# on `Owned(Cell)`, or on regions, e.g. `intersect(Owned(Cell), region(LateralBoundary))`.
# A local program, run on all the ranks, computes values valid on their domains: a result on
# `Owned(Cell)` gets its halo with the collective `exchange`, before reads that need it (see
# `@local_program` and `run_local` in the implementation).


class Region[D: DimensionIndex]:
    """A named set of points of the `Location` dimension `D`, e.g. the lateral boundary of a
    limited-area mesh: `class LateralBoundary(Region[Cell]): ...`.

    Regions are nominal (one class per region) and are types, never instantiated: their points
    are only known at runtime, from the mesh. In a `LocalMesh`, a region holds the local points
    whose global indices are in it, halo points too: `region(LateralBoundary)` is that domain.
    `D` is a `Location` dimension (or a legacy one): an `Axis`, `Label` or `Neighbor` raises a
    `TypeError` (at runtime only). A subclass of a region is another region, on the same `D`.
    """

    dim: ClassVar[Dimension]  # `D`

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        base = get_original_bases(cls)[0]
        if get_origin(base) is Region:
            (cls.dim,) = get_args(base)
            if issubclass(cls.dim, (Axis, Label, Neighbor)):
                raise TypeError(f"'{cls.dim.__name__}' cannot have regions: use a `Location`.")

    def __init__(self) -> None:
        raise TypeError("A region is a type: it is never instantiated.")


class LocalMesh(Mesh):
    """The mesh of one `rank` of a distributed run, with local indices: built once, checked,
    immutable, and provided like any `Mesh`, by `with local_mesh: ...`.

        LocalMesh(tables, extents={Cell: 5, Edge: 5}, owned={Cell: range(3), Edge: range(3)},
                  global_index={Cell: (0, 1, 2, 3, 5), Edge: (0, 1, 2, 3, 5)},
                  regions={LateralBoundary: (0, 4)}, mesh_id="ring", rank=0)

    - `tables`: the neighbor tables of the local points, with local indices: a row per local
      point, the halo points too. A neighbor outside the local points is `TRUNCATED`: a read
      reaching it is not computable (see `NeighborTable`).
    - `extents`: the number of local points of each distributed dimension.
    - `owned`: the local indices of the points this rank owns, along each of them.
    - `global_index`: the global index of each local point (an injection: distinct indices).
    - `regions`: the local indices of the points of each `Region` (see `Region`).
    - `mesh_id`: the identity of the global mesh, the same on all its ranks.

    A `ValueError` if they do not fit together. The given tables are not checked against their
    constructions, which a local mesh cannot do for halo rows: `decompose`, in the
    implementation, derives or checks them on the global mesh. For the same reason, a local mesh
    does not derive the tables of stencils with transposes (a `LookupError`); other stencils,
    and pairs, are derived from the local tables, except that the rows of `paths="unique"`
    stencils with a truncated neighbor are all `TRUNCATED` (distinct truncated neighbors cannot
    be told apart): to get them exactly, derive them on the global mesh (`decompose(...,
    derived=[...])`). The sources and targets of the tables are distributed dimensions (else a
    `ValueError`). Its attributes are immutable; the arrays of its tables are not copied.
    """

    _local = True

    def __init__(
        self,
        tables: Mapping[type[Relation[Any, Any]], NeighborTable],
        *,
        extents: Mapping[Dimension, int],
        owned: Mapping[Dimension, range | IndexSet],
        global_index: Mapping[Dimension, IndexSet],
        regions: Mapping[type[Region[Any]], range | IndexSet],
        mesh_id: object,
        rank: int,
    ) -> None:
        super().__init__(tables, extents)
        table_dims = (d for r in tables for d in (r.source, r.target))
        for dim in dict.fromkeys([*owned, *global_index, *(r.dim for r in regions), *table_dims]):
            n = self.extents.get(dim)
            indices = global_index.get(dim, ())
            if n is None or dim not in owned or len(indices) != n or len(set(indices)) != n:
                raise ValueError(
                    f"'{dim.__name__}' needs a local extent, its owned points, and a distinct "
                    "global index per local point."
                )
        for name, sets in (("owned", owned), ("region", regions)):
            for key, points in sets.items():
                dim = key.dim if isinstance(key, type) and issubclass(key, Region) else key
                check_domain({dim: points})
                if not all(0 <= i < self.extents[dim] for i in points):
                    raise ValueError(f"The {name} points of '{key.__name__}' must be local.")
        self.extents = types.MappingProxyType(dict(self.extents))
        self.owned = types.MappingProxyType(dict(owned))
        self.global_index = types.MappingProxyType({d: tuple(i) for d, i in global_index.items()})
        self.regions = types.MappingProxyType(dict(regions))
        self.mesh_id, self.rank = mesh_id, rank
        self._frozen = True

    def __setattr__(self, name: str, value: Any) -> None:
        if "_frozen" in self.__dict__:
            raise AttributeError("A `LocalMesh` is immutable.")
        super().__setattr__(name, value)


def _local_mesh() -> LocalMesh:
    """The current mesh, a `LocalMesh` (else a `TypeError`)."""
    the_mesh = current_mesh()
    if not isinstance(the_mesh, LocalMesh):
        raise TypeError("No local mesh: use `with local_mesh:` to provide it.")
    return the_mesh


def Owned(dim: Dimension) -> Domain:
    """The domain of the points of `dim` owned by this rank, in the current local mesh, e.g.
    `op[Owned(Cell)](...)`. A `TypeError` outside a local mesh, a `LookupError` if `dim` is not
    distributed there. The halo points are not nameable. Not in programs (a `TypeError`): their
    semantics is global."""
    if program_mesh.get() is not None:
        raise TypeError(
            "`Owned` is not in programs: their semantics is global, the decomposition is not "
            "visible. Request regions, `region(...)`."
        )
    owned = _local_mesh().owned
    if dim not in owned:
        raise LookupError(f"'{dim.__name__}' is not distributed in the local mesh.")
    return {dim: owned[dim]}


def region(the_region: type[Region[Any]]) -> Domain:
    """The domain of the local points of `the_region` (see `Region`) in the current local mesh,
    e.g. `{Cell: (0, 4)}`. Combine with `intersect` and `difference` (in the implementation),
    e.g. `difference(Owned(Cell), region(LateralBoundary))`. A `TypeError` outside a local mesh,
    a `LookupError` if the local mesh has no points for `the_region`. In a running program, the
    global domain of `the_region` in the program's logical mesh (see `program_mesh`)."""
    the_mesh = program_mesh.get()
    if the_mesh is not None:
        return the_mesh.region(the_region)
    regions = _local_mesh().regions
    if the_region not in regions:
        raise LookupError(f"The local mesh has no region '{the_region.__name__}'.")
    return {the_region.dim: regions[the_region]}


# --- Logical meshes -----------------------------------------------------------
# Proposal B's model on one process: a logical mesh is the whole mesh, with global indices
# and global semantics (see the design note). Its locations are the families of its schema:
# `Face[M]`, `Edge[M]` and `Node[M]` (UGRID's names), interned per family and schema, nominal
# classes usable in annotations and at runtime. A *role* is a subclass of a schema, e.g.
# `class Coarse(Plate): ...`: `Face[Coarse]` is a location distinct from `Face[Plate]`, and
# relations are declared per role. The logical mesh also knows its `extents`, its `regions`
# (semantic boundary regions, global index sets) and its identity, the `mesh_id`: the brand
# of its fields and of its ranks. The decomposition (which rank owns what) is separate, and
# built with `decompose` (in the implementation), as in L1. A field on a logical mesh, a
# `MeshField`, is the pair (mesh, values): allocated through the mesh, branded by its `mesh_id`.


class Schema:
    """A brand for the locations of its meshes: a type, never instantiated.

    A schema is declared by subclassing, e.g. `class Plate(Schema): ...`, and its subclasses
    are *roles*, e.g. `class Coarse(Plate): ...`: `Face[Coarse]` and `Face[Fine]` are distinct
    locations of one family, and mypy rejects one for the other (but, statically only, accepts
    a field of a role where a field of its schema is expected). The families of a schema are
    its locations: `Cell = Face[Plate]` is an alias, usable in annotations and at runtime.
    Relations, regions and axes are declared on its families as in L1 (a schema only brands).
    """

    def __init__(self) -> None:
        raise TypeError("A schema is a type: it is never instantiated.")


class SchemaLocation(Location):
    """The base of the families of a schema, e.g. `Face[M]`, `Edge[M]` and `Node[M]`
    (UGRID's 2-, 1- and 0-dimensional locations; e.g. ICON's cells, edges and vertices are
    `Face[Icon]`, `Edge[Icon]` and `Node[Icon]`). A family is a `Location` subclass, interned
    per family and schema with `location_of`. A schema brands its locations; the roles of a
    schema are its subclasses, and each role redeclares its relations.
    """

    schema: ClassVar[type[Schema]]

    def __class_getitem__(cls, schema: type[Schema]) -> Dimension:  # type: ignore[override]
        return location_of(cls, schema)


class Face[M: Schema](SchemaLocation):
    """The 2-dimensional locations of the schema `M`, e.g. ICON's cells: `Face[Icon]`."""


class Edge[M: Schema](SchemaLocation):
    """The 1-dimensional locations of the schema `M`, e.g. ICON's edges: `Edge[Icon]`."""


class Node[M: Schema](SchemaLocation):
    """The 0-dimensional locations of the schema `M`, e.g. ICON's vertices: `Node[Icon]`."""


def location_of(family: type[SchemaLocation], schema: type[Schema]) -> Dimension:
    """Return the interned `family[schema]`, e.g. `Face[Plate]` for `family` `Face` and
    `schema` `Plate`: the same class each time, with `schema` as its `schema` attribute.

    A `TypeError` if `family` is not one of `Face`, `Edge`, `Node`, or `schema` is not a
    strict subclass of `Schema`.
    """
    if family not in (Face, Edge, Node):
        raise TypeError(
            f"'{family.__name__}' is not a family: use `Face`, `Edge` or `Node`, e.g. "
            "`Face[Plate]`."
        )
    if not (isinstance(schema, type) and issubclass(schema, Schema) and schema is not Schema):
        raise TypeError(
            f"'{schema.__name__ if isinstance(schema, type) else schema}' is not a schema: "
            "subclass `Schema`, e.g. `class Icon(Schema): ...`."
        )
    name = f"{family.__name__}[{schema.__name__}]"
    if (family, schema) not in _locations:
        namespace = {
            "schema": schema,
            "__qualname__": name,
            "__module__": schema.__module__,
        }
        _locations.setdefault((family, schema), cast(Dimension, type(name, (family,), namespace)))
    return _locations[family, schema]


_locations: dict[tuple[type[SchemaLocation], type[Schema]], Dimension] = {}


class LogicalMesh(Mesh):
    """The whole mesh, with global indices and global semantics: `schema`, `tables`,
    `extents` and `regions` (semantic boundary regions, global index sets), `geometry` (named
    fields, e.g. `{"area": ...}`), with an identity, the `mesh_id` (see `Mesh`).

    Immutable after construction, like a `LocalMesh`. Its tables are checked like the ones of
    any `Mesh` (once, given or derived), and its locations must be families of its schema
    (else a `ValueError`). The `signature` is the compile key: the schema, the given
    relations and the regions, as classes, sorted by `__qualname__` (never the instance, the
    extents or the values).

    Its fields, `MeshField`s, are allocated through it: `grid.zeros(Cell, K)`,
    `grid.full(Cell, fill_value=1.0)`, `grid.field(Cell, data=[...])`, on the whole of the
    given dimensions. Each geometry field must be on the whole of its dimensions (else a
    `ValueError`), and is kept as a `MeshField` of this mesh.
    """

    # Create a `MeshField` of this mesh from a domain and its values, a scalar (broadcast) or
    # an array of the shape of the domain: called `self.make_field(domain, data)`, like
    # `Mesh.construct`, and provided by the implementation, e.g. `implementation.mesh_field`.
    make_field: ClassVar[Callable[[LogicalMesh, Domain, Any], MeshField[Any, Any]]]

    def __init__(
        self,
        schema: type[Schema],
        tables: Mapping[type[Relation[Any, Any]], NeighborTable],
        *,
        extents: Mapping[Dimension, int],
        regions: Mapping[type[Region[Any]], range | IndexSet] | None = None,
        geometry: Mapping[str, NDArrayField[Any, Any]] | None = None,
        mesh_id: object = None,
    ) -> None:
        if not (isinstance(schema, type) and issubclass(schema, Schema) and schema is not Schema):
            raise TypeError(
                f"'{schema.__name__ if isinstance(schema, type) else schema}' is not a schema: "
                "subclass `Schema`, e.g. `class Icon(Schema): ...`."
            )
        # Every `Location` dimension of the tables (sources and targets), of the regions
        # (`R.dim`) and of the keys of `extents` is a family of `schema`; the `Axis` and
        # `Label` keys of `extents` are allowed: they size `zeros` etc. (see 2b).
        dims = {d for r in tables for d in (r.source, r.target)}
        dims |= {r.dim for r in regions or {}}
        for dim in dict.fromkeys([*dims, *extents]):
            if issubclass(dim, Location) and not (
                issubclass(dim, SchemaLocation) and dim.schema is schema
            ):
                raise ValueError(
                    f"'{dim.__name__}' is not a location of the schema '{schema.__name__}': use "
                    f"`Face[{schema.__name__}]`, `Edge[{schema.__name__}]` or "
                    f"`Node[{schema.__name__}]`."
                )
        for dim in dict.fromkeys([*dims, *extents]):
            if dim not in extents:
                raise ValueError(f"The logical mesh needs the extent of '{dim.__name__}'.")
        super().__init__(tables, extents)
        for the_region, points in (regions or {}).items():
            check_domain({the_region.dim: points})
            extent = self.extents.get(the_region.dim, 0)
            if not all(0 <= p < extent for p in points):
                raise ValueError(
                    f"The points of the region '{the_region.__name__}' must be within the extent "
                    f"of '{the_region.dim.__name__}', {extent}."
                )
        self.schema = schema
        self.extents = types.MappingProxyType(dict(self.extents))
        self.regions = types.MappingProxyType(dict(regions or {}))
        self.mesh_id = (
            f"{schema.__name__}-{next(_logical_mesh_ids)}" if mesh_id is None else mesh_id
        )
        for name, field in (geometry or {}).items():
            if field.domain != self.domain(*field.domain):
                raise ValueError(
                    f"The geometry field '{name}' must be on the whole of its dimensions."
                )
        self.geometry: Mapping[str, MeshField[Any, Any]] = types.MappingProxyType(
            {name: self.make_field(f.domain, f.data) for name, f in (geometry or {}).items()}
        )
        self._frozen = True

    def __setattr__(self, name: str, value: Any) -> None:
        if "_frozen" in self.__dict__:
            raise AttributeError("A `LogicalMesh` is immutable.")
        super().__setattr__(name, value)

    def domain(self, *dims: Dimension) -> Domain:
        """Return `{d: range(extents[d]) for d in dims}`; a `LookupError` if the logical mesh
        has no extent for a dimension."""
        missing = [d for d in dims if d not in self.extents]
        if missing:
            raise LookupError(f"The logical mesh has no extent for '{missing[0].__name__}'.")
        return {d: range(self.extents[d]) for d in dims}

    def region(self, the_region: type[Region[Any]]) -> Domain:
        """Return `{the_region.dim: regions[the_region]}`; a `LookupError` if the logical mesh
        has no region `the_region`."""
        if the_region not in self.regions:
            raise LookupError(f"The logical mesh has no region '{the_region.__name__}'.")
        return {the_region.dim: self.regions[the_region]}

    def zeros(self, *dims: Dimension) -> MeshField[Any, float]:
        """A field of zeros on the whole of `dims`, e.g. `grid.zeros(Cell, K)`; a `LookupError`
        if the logical mesh has no extent for one of them (see `domain`)."""
        return self.make_field(self.domain(*dims), 0.0)

    def full(self, *dims: Dimension, fill_value: Scalar) -> MeshField[Any, Any]:
        """A field of `fill_value` on all of `dims`, e.g. `grid.full(Cell, fill_value=1.0)`."""
        return self.make_field(self.domain(*dims), fill_value)

    def field(self, *dims: Dimension, data: Any) -> MeshField[Any, Any]:
        """A field with the values `data` on the whole of `dims`, e.g. `grid.field(Cell,
        data=[...])`: an array of their shape (else a `ValueError`), or a scalar."""
        return self.make_field(self.domain(*dims), data)

    @property
    def signature(self) -> tuple[Any, ...]:
        """`(schema, the given relations and the regions, as classes, sorted by __qualname__)`:
        the compile key (never the instance, the extents or the values). Classes with the same
        `__qualname__` (e.g. from two modules) are ordered by module, then identity, so the order
        never depends on the order of the given tables or regions."""
        return (
            self.schema,
            tuple(sorted(self, key=_sort_key)),
            tuple(sorted(self.regions, key=_sort_key)),
        )

    def __repr__(self) -> str:
        return f"LogicalMesh('{self.mesh_id}')"


_logical_mesh_ids = itertools.count()


def _sort_key(cls: type) -> tuple[str, str, int]:
    """The `signature` sort key of a class (see `LogicalMesh`): its `__qualname__`, then its
    module and identity, to order distinct classes with the same name."""
    return (cls.__qualname__, cls.__module__, id(cls))


@runtime_checkable
class MeshField[D: tuple[DimensionIndex, ...], T](NDArrayField[D, T], Protocol):
    """An `NDArrayField` on a logical mesh: the pair (`mesh`, values), with global indices.

    Allocated through its mesh, e.g. `grid.zeros(Cell, K)` or `grid.field(Cell, data=...)`
    (see `LogicalMesh`). Its premaps, e.g. `t(C2C)`, and the neighbor reductions of its values
    use the tables of its mesh, without a `with` block. It is combined (arithmetic, writes)
    only with fields of the same mesh, the same `mesh_id` (else a `ValueError`); a result keeps
    the brand. DSL operators called with mesh fields run in their mesh.
    """

    @property
    def mesh(self) -> LogicalMesh:
        """The logical mesh of the field, its brand."""
        ...


# The logical mesh of the running program: set by the implementation while the body of a
# program runs (`region` gives global domains there, `Owned` raises), not while the DSL operators
# it calls run (they are L1 code, on the current mesh).
program_mesh: Final[contextvars.ContextVar[LogicalMesh | None]] = contextvars.ContextVar(
    "program_mesh", default=None
)


@dataclasses.dataclass(frozen=True)
class Distribution:
    """How the logical `mesh` is distributed: the `LocalMesh` of each rank, `local_meshes[r]`
    of rank `r`, with the points it owns and its halo.

    Configuration, never in the source of programs, and not part of the mesh's identity: a
    mesh keeps its `mesh_id` (its brand) under any distribution, and the local meshes carry
    it. Built by `distribute` (the given mode: a halo of `hops` over relations) or
    `derive_distribution` (the derived mode: the halo the reads of some programs need), in the
    implementation. A `ValueError` if there is no local mesh, if they are not the ranks 0, 1,
    ... in order, or if one is not of `mesh` (another `mesh_id`).
    """

    mesh: LogicalMesh
    local_meshes: tuple[LocalMesh, ...]

    def __post_init__(self) -> None:
        ranks = [local_mesh.rank for local_mesh in self.local_meshes]
        ids = {local_mesh.mesh_id for local_mesh in self.local_meshes}
        if not ranks or ranks != list(range(len(ranks))) or ids != {self.mesh.mesh_id}:
            raise ValueError(
                f"The local meshes are not the ranks 0, 1, ... of '{self.mesh.mesh_id}'."
            )


# --- Values -------------------------------------------------------------------

type Scalar = float | int  # simplified
type NumericValue = Scalar | Field  # simplified for the example
type FunctionValue = Callable[..., Any]  # simplified for the example
type PrimitiveValue = NumericValue | FunctionValue

type NestedTuple[T] = tuple[T | NestedTuple[T], ...]


class NamedCollection(Protocol):
    """Any dataclass instance whose field values are `AnyValue`s (not checked)."""

    __dataclass_fields__: ClassVar[dict[str, Any]]


type CompositeValue = NestedTuple[PrimitiveValue] | NamedCollection
type AnyValue = PrimitiveValue | CompositeValue


# --- DSL definitions ----------------------------------------------------------


class InspectableCallable[**P, R](Protocol):
    """A callable whose call operator's Python source can be retrieved with `getsource`."""

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...


@runtime_checkable
class CallableWithSource[**P, R](InspectableCallable[P, R], Protocol):
    """An `InspectableCallable` carrying its own source code, e.g. a generated function."""

    __source__: str


# Definitions are plain Python callables: nothing is added to them. Their kind is known from
# the operator built from them (see `DSLOperator.__gt_operator_kind__`).


class DSLOperatorDefinition[**P, R](InspectableCallable[P, R], Protocol):
    """Arguments and source code follow the rules of one of the DSLs below.

    See `is_dsl_operator_definition`.
    """


class FieldOperatorDefinition[**P, R](DSLOperatorDefinition[P, R], Protocol):
    """Arguments and source follow `FIELD_OPERATOR_RULES`; see `is_field_operator_definition`."""


class ScanOperatorDefinition[**P, R](DSLOperatorDefinition[P, R], Protocol):
    """Arguments and source follow `SCAN_OPERATOR_RULES`; see `is_scan_operator_definition`."""


def call_operator(c: Callable) -> Callable:
    """Return the function run by `c(...)`: `c` itself, or the `__call__` of its class."""
    return c if inspect.isfunction(c) else type(c).__call__


@functools.singledispatch
def getsource(object: InspectableCallable[..., Any]) -> str:
    """Return the Python source code of the call operator of `object`.

    Wrappers, like DSL operators wrapping their definition, are unwrapped first (`__wrapped__`).
    Uses `object.__source__` if present (see `CallableWithSource`), else `inspect.getsource`.
    Other ways of providing the source for a class `C` can be added with `@getsource.register`:

        @getsource.register
        def _(object: C) -> str: ...
    """
    object = inspect.unwrap(object)
    if isinstance(object, CallableWithSource):
        return object.__source__
    return textwrap.dedent(inspect.getsource(call_operator(object)))


def is_inspectable(obj: object) -> TypeIs[InspectableCallable[..., Any]]:
    if not callable(obj):
        return False
    try:
        getsource(obj)
    except (OSError, TypeError):
        return False
    return True


@dataclasses.dataclass(frozen=True)
class DSLRules:
    """What a DSL accepts: argument types and the Python syntax of the body (simplified).

    `forbidden` names cannot be used in the body, e.g. the collectives of local programs
    (`COLLECTIVES`), nor private ones, `_x`: the internals of the implementation, e.g. its private
    write path into fields, are not in the DSL. Checked on the free names of the body (not its
    parameters and local variables: `x, _ = t` is fine) and on the names of their values, in the
    globals and closure of the definition, so aliases are rejected too (`ex = exchange`,
    `from implementation import _write as w`), and on attribute names (`implementation._write`).
    A name given as a string, `getattr(m, "_write")`, or a closure variable assigned only after the
    definition is decorated, is beyond these static checks. The body
    does not write into subscripts or attributes either, e.g. `f[...] = 0.0` or `f.data[0] = 1`:
    DSL operators return new values. Only `top_level` definitions (programs) give a `domain` in
    their calls, e.g. `op(t, domain=...)`; `op[domain](t)` is accepted in all of them.
    """

    arguments: tuple[type, ...]  # annotation origins; `NamedCollection` stands for any dataclass
    syntax: frozenset[type[ast.AST]]  # allowed statement and expression nodes
    forbidden: frozenset[str] = frozenset()
    # Programs are top level: their calls may request a `domain`.
    top_level: bool = False

    def accepts(self, c: InspectableCallable[..., Any]) -> bool:
        return self.violation(c) is None

    def violation(self, c: InspectableCallable[..., Any]) -> str | None:
        """Why `c` does not follow these rules, e.g. "uses the collective 'exchange'": the first
        violation found, or `None` if it follows them."""
        hints = get_type_hints(call_operator(c))
        parameters = inspect.signature(c).parameters
        for name in parameters:
            if name not in hints or not self._accepts_type(hints[name]):
                return f"the type of the parameter '{name}' is not accepted"
        body = ast.parse(getsource(c)).body[0].body  # statements of the function definition
        nodes = [node for statement in body for node in ast.walk(statement)]
        for node in nodes:
            if isinstance(node, (ast.stmt, ast.expr)) and type(node) not in self.syntax:
                return f"the syntax `{type(node).__name__}` is not accepted"
            # The `domain` of a DSL operator is only given in top-level calls, from Python.
            if not self.top_level and isinstance(node, ast.keyword) and node.arg == "domain":
                return "a `domain` is only given in top-level calls"
            if isinstance(node, (ast.Subscript, ast.Attribute)) and isinstance(node.ctx, ast.Store):
                target = "a subscript" if isinstance(node, ast.Subscript) else "an attribute"
                return f"assignment to {target}"
        local = {n.id for n in nodes if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        free = {n.id for n in nodes if isinstance(n, ast.Name)} - local - set(parameters)
        names = {name: name for name in free}  # the name in the source, by the name checked
        names |= {node.attr: node.attr for node in nodes if isinstance(node, ast.Attribute)}
        names |= {value: name for name, value in _value_names(c, free).items()}
        for checked, name in sorted(names.items()):
            alias = "" if checked == name else f" (as '{name}')"
            if checked in self.forbidden:
                kind = "the collective" if checked in COLLECTIVES else "the forbidden name"
                return f"uses {kind} '{checked}'{alias}"
            if checked.startswith("_"):
                return f"uses the private name '{checked}'{alias}"
        return None

    def _accepts_type(self, annotation: Any) -> bool:
        if isinstance(annotation, TypeAliasType):  # e.g. `type IField = Field[Dims[I], float]`
            annotation = annotation.__value__
        origin = get_origin(annotation) or annotation
        return (NamedCollection if dataclasses.is_dataclass(origin) else origin) in self.arguments


def _value_names(c: InspectableCallable[..., Any], names: set[str]) -> dict[str, str]:
    """The `__name__` of the value of each free name in `names` of the definition `c`, found in
    its closure, its globals or the builtins, e.g. `{"ex": "exchange"}` after `ex = exchange`.
    A closure variable not assigned yet (assigned after the definition) is left out."""
    function = inspect.unwrap(call_operator(c))
    if not inspect.isfunction(function):  # e.g. a generated callable: no namespace
        return {}
    namespace = {**vars(builtins), **function.__globals__}
    for name, cell in zip(function.__code__.co_freevars, function.__closure__ or (), strict=True):
        with contextlib.suppress(ValueError):  # an empty cell: not assigned yet
            namespace[name] = cell.cell_contents
    values = {name: getattr(namespace[name], "__name__", None) for name in names & namespace.keys()}
    return {name: value for name, value in values.items() if isinstance(value, str)}


_EXPRESSIONS = {ast.Name, ast.Constant, ast.Attribute, ast.Subscript, ast.Tuple, ast.Call}
_OPERATORS = {ast.BinOp, ast.UnaryOp, ast.Compare}

# The collectives of local programs (in the implementation), called by all the ranks together:
# only local programs call them, never DSL operators, which compute point by point.
COLLECTIVES: Final = frozenset({"exchange", "global_sum"})

FIELD_OPERATOR_RULES = DSLRules(
    arguments=(Field, int, float, tuple, NamedCollection),
    syntax=frozenset(
        {ast.Expr, ast.Assign, ast.AnnAssign, ast.Return, ast.If, ast.IfExp}
        | _EXPRESSIONS
        | _OPERATORS
    ),
    forbidden=COLLECTIVES,
)
# Scan operators take the carry and the values of one level: scalars and tuples of them.
SCAN_OPERATOR_RULES = DSLRules(
    arguments=(int, float, tuple), syntax=FIELD_OPERATOR_RULES.syntax, forbidden=COLLECTIVES
)
# Programs (`@program`, in the implementation) see no decomposition: no collectives, `Owned`,
# local meshes or local programs (layering rule 3); they get their fields as arguments (a field
# built in the body would pair local and global indices once lowered: no field constructors; at
# run time, in the direct and the lowered run alike, a field from a global or a closure, e.g.
# geometry, and a field inside a container argument of the program or of an operator it calls
# are rejected: pass fields as separate arguments; results are fields, scalars or tuples of
# them); and they do not look inside fields: no attributes, e.g. `t.data`. Loops are allowed.
PROGRAM_RULES = DSLRules(
    arguments=(Field, int, float, tuple, NamedCollection),
    syntax=frozenset(
        {ast.Expr, ast.Assign, ast.AnnAssign, ast.Return, ast.If, ast.For, ast.IfExp}
        | (_EXPRESSIONS - {ast.Attribute})
        | _OPERATORS
    ),
    forbidden=COLLECTIVES
    | {
        "Owned",
        "current_mesh",
        "run_local",
        "local_program",
        "scatter",
        "NumPyArrayField",
        "DeferredField",
        "mesh_field",
        "gather_owned",
    },
    top_level=True,
)


# The checks keep the signature `P -> R` of the checked callable in the narrowed type. A
# callable can follow the rules of both DSLs: its kind is chosen when building an operator.


def is_field_operator_definition[**P, R](
    c: Callable[P, R],
) -> TypeIs[FieldOperatorDefinition[P, R]]:
    return is_inspectable(c) and FIELD_OPERATOR_RULES.accepts(c)


def is_scan_operator_definition[**P, R](c: Callable[P, R]) -> TypeIs[ScanOperatorDefinition[P, R]]:
    return is_inspectable(c) and SCAN_OPERATOR_RULES.accepts(c)


def is_dsl_operator_definition[**P, R](c: Callable[P, R]) -> TypeIs[DSLOperatorDefinition[P, R]]:
    return is_field_operator_definition(c) or is_scan_operator_definition(c)


# --- DSL operators ------------------------------------------------------------
# The callables gt4py builds from DSL definitions, e.g. with a decorator, plus a backend.


@dataclasses.dataclass(frozen=True)
class Reduction:
    """A read step: a reduction over the neighbor dimension `axis`, e.g. by `sum_over`."""

    axis: Dimension


@dataclasses.dataclass(frozen=True)
class ScanAlong:
    """A read step: a scan along `axis`, which reads all the previous levels."""

    axis: Dimension


@dataclasses.dataclass(frozen=True)
class Flatten:
    """A read step: `(relation.source, Neighbor[relation], ...)` flattened to `(Pairs[relation],
    ...)`, e.g. by `flatten` (see `Pairs`)."""

    relation: type[Relation[Any, Any]]


@dataclasses.dataclass(frozen=True)
class Unflatten:
    """A read step: `(Pairs[relation], ...)` back to `(relation.source, Neighbor[relation], ...)`,
    e.g. by `unflatten` (see `Pairs`)."""

    relation: type[Relation[Any, Any]]


# For each argument name, how the operator reads it, relative to the point being computed: a
# list of reads, each a chain of steps applied in order. Reading `a(I + 1)` is `((I, 1),)`, a
# plain `a` is `()`, and `sum_over(v(E2V)(C2E), axis=...)` is `(E2V, C2E, Reduction(...))`;
# a scan operator reads each argument with `(ScanAlong(axis),)`, `c(C2C.boundary(x))` is
# `(Boundary(C2C),)`, without the value `x`, and `unflatten(e(C2E.pair_target), axis=...)` is
# `(C2E.pair_target, Unflatten(C2E))` (`Flatten(C2E)` for `flatten`). Reads are followed through
# local variables and calls to other DSL operators; a step is only added to the reads it
# applies to, e.g. in `(e * k)(C2E)` with `k: Field[Dims[K], float]`, not to the read of `k`.
# Extents are used for forward and backward domain computation, step by step: forward, an
# offset shifts the domain, a gather keeps the sources whose neighbors are all in it (from the
# tables of the current `Mesh`; for a `Boundary`, all but the missing ones), a reduction removes
# its neighbor dimension, a scan changes nothing; backward, the inverse, where a scan needs its
# whole axis, a reduction does not restrict its neighbor dimension (the argument keeps all its
# neighbors), and a plain gather raises a `ValueError` on missing neighbors (any gather on
# `TRUNCATED` ones, in a local mesh: outside the local support). A gather through a functional
# relation changes the root, without a neighbor dimension; `Flatten` and `Unflatten` go through
# the table of `R.to_pairs` (see `Pairs`). Half offsets are steps too (see `Staggered`): forward
# and backward, they shift the domain by the half-integer offset. Along a dimension, the extent
# is the `(min, max)` range of the offsets.
type Step = (
    Offset
    | HalfOffset
    | type[Relation[Any, Any]]
    | Boundary
    | Reduction
    | ScanAlong
    | Flatten
    | Unflatten
)
type Read = tuple[Step, ...]
type Extents = dict[str, list[Read]]

# The requested domain of the result of a DSL operator, with the structure of the result: a
# `Domain` per field. A single `Domain` applies to all the fields of a tuple.
type DomainSpec = Domain | NestedTuple[Domain]


def restrict[T](result: T, domain: DomainSpec) -> T:
    """Restrict every field of `result`, a field or a (nested) tuple of fields, to `domain`.

    As `Field.__getitem__`: a `ValueError` if `domain` is not within the domain of a field.
    """
    if isinstance(result, tuple):
        domains = domain if isinstance(domain, tuple) else (domain,) * len(result)
        return cast(T, tuple(restrict(r, d) for r, d in zip(result, domains, strict=True)))
    return cast(T, cast(Field, result)[cast(Domain, domain)])


class DSLOperatorKind(enum.Enum):
    """The kind of a DSL operator: the DSL of its definition."""

    FIELD_OPERATOR = "FieldOperator"
    SCAN_OPERATOR = "ScanOperator"


@dataclasses.dataclass(frozen=True)
class ScanConfig:
    """How a scan operator runs: along `axis`, `forward` or backward, from the carry `init`."""

    axis: Dimension
    forward: bool = True
    init: Scalar = 0.0


class DSLOperator[**P, R](Protocol):
    """Any callable built by gt4py from a `DSLOperatorDefinition`, of kind `__gt_operator_kind__`.

    `op[domain](*args)` returns `restrict(op(*args), domain)`: the result restricted to the
    requested `domain`, which must be within the result's domain. Implementations may use
    `domain` to compute only the requested points (backward domain propagation).
    `op(*args, domain=domain)` is accepted at runtime too, but not by type checkers: a keyword
    argument cannot be added to the signature `P`.
    """

    @property
    def __gt_operator_kind__(self) -> DSLOperatorKind:
        """Usually a class attribute: the kind is the same for all the operators of a class."""
        ...

    @property
    def __gt_definition__(self) -> DSLOperatorDefinition[..., Any]: ...

    @property
    def __gt_extents__(self) -> Extents: ...

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...

    def __getitem__(self, domain: DomainSpec) -> Callable[P, R]: ...


class FieldOperator[**P, R](DSLOperator[P, R], Protocol):
    """A field operator built from `__gt_definition__`, with the same signature `P -> R`.

    Callable from Python and from the source of field operators and scan operators.
    """

    @property
    def __gt_operator_kind__(self) -> Literal[DSLOperatorKind.FIELD_OPERATOR]: ...

    @property
    def __gt_definition__(self) -> FieldOperatorDefinition[P, R]: ...


class ScanOperator[**P, R](DSLOperator[P, R], Protocol):
    """A scan operator built from `__gt_definition__`, scanning as `__gt_scan_config__` says.

    The definition computes one level from the carry and the values on that level; the scan
    operator takes and returns fields, so its signature `P -> R` differs from the definition's.
    Callable from Python and from the source of field operators.
    """

    @property
    def __gt_operator_kind__(self) -> Literal[DSLOperatorKind.SCAN_OPERATOR]: ...

    @property
    def __gt_definition__(self) -> ScanOperatorDefinition[..., Any]: ...

    @property
    def __gt_scan_config__(self) -> ScanConfig: ...


class CompiledDSLOperator[**P, R](DSLOperator[P, R], Protocol):
    """A DSL operator that can also be compiled ahead of time."""

    def compile(self, *args: P.args, **kwargs: P.kwargs) -> Self:
        """Compile for arguments like `args` and `kwargs` (their types, domains, ...) and
        return `self`, which then runs the compiled code when called with such arguments."""
        ...


class CompiledFieldOperator[**P, R](FieldOperator[P, R], CompiledDSLOperator[P, R], Protocol):
    """A `FieldOperator` that can also be compiled ahead of time."""


class CompiledScanOperator[**P, R](ScanOperator[P, R], CompiledDSLOperator[P, R], Protocol):
    """A `ScanOperator` that can also be compiled ahead of time."""


# --- DSL processors -----------------------------------------------------------
# Objects turning DSL operators into other callables, e.g. the operators of another backend,
# with one method per kind of operator. Calling a processor dispatches on the kind of the
# operator: `processor(operator)`. The protocols implement `__call__`, for the classes
# subclassing them explicitly.


def dispatch(processor: DSLProcessor, operator: DSLOperator[..., Any]) -> Any:
    """Call the method of `processor` for the kind of `operator`."""
    match operator.__gt_operator_kind__:
        case DSLOperatorKind.FIELD_OPERATOR:
            return processor.field_operator(cast(FieldOperator[..., Any], operator))
        case DSLOperatorKind.SCAN_OPERATOR:
            return processor.scan_operator(cast(ScanOperator[..., Any], operator))


@runtime_checkable
class DSLProcessor(Protocol):
    """Any object turning each kind of DSL operator into a callable."""

    def field_operator[**P, R](self, operator: FieldOperator[P, R]) -> Callable[..., Any]: ...

    def scan_operator(self, operator: ScanOperator[..., Any]) -> Callable[..., Any]: ...

    @overload
    def __call__[**P, R](self, operator: FieldOperator[P, R]) -> Callable[..., Any]: ...
    @overload
    def __call__(self, operator: ScanOperator[..., Any]) -> Callable[..., Any]: ...
    def __call__(self, operator: DSLOperator[..., Any]) -> Any:
        return dispatch(self, operator)


@runtime_checkable
class DSLInterpreter(DSLProcessor, Protocol):
    """A `DSLProcessor` returning DSL operators with the signature of their definition.

    The operators may add keyword arguments with default values, e.g. `out=`: such a
    callable still matches the signature `P -> R` of the definition. Scan operators
    are the exception: they take and return fields, unlike their definition (see
    `ScanOperator`).
    """

    def field_operator[**P, R](self, operator: FieldOperator[P, R]) -> FieldOperator[P, R]: ...

    def scan_operator(self, operator: ScanOperator[..., Any]) -> ScanOperator[..., Any]: ...

    @overload
    def __call__[**P, R](self, operator: FieldOperator[P, R]) -> FieldOperator[P, R]: ...
    @overload
    def __call__(self, operator: ScanOperator[..., Any]) -> ScanOperator[..., Any]: ...
    def __call__(self, operator: DSLOperator[..., Any]) -> Any:
        return dispatch(self, operator)


@runtime_checkable
class DSLCompiler(DSLInterpreter, Protocol):
    """A `DSLInterpreter` returning DSL operators that can also be compiled ahead of time."""

    def field_operator[**P, R](
        self, operator: FieldOperator[P, R]
    ) -> CompiledFieldOperator[P, R]: ...

    def scan_operator(self, operator: ScanOperator[..., Any]) -> CompiledScanOperator[..., Any]: ...

    @overload
    def __call__[**P, R](self, operator: FieldOperator[P, R]) -> CompiledFieldOperator[P, R]: ...
    @overload
    def __call__(self, operator: ScanOperator[..., Any]) -> CompiledScanOperator[..., Any]: ...
    def __call__(self, operator: DSLOperator[..., Any]) -> Any:
        return dispatch(self, operator)


def definition_of(fun: Callable[..., Any]) -> DSLOperatorDefinition[..., Any]:
    """Return the DSL definition of `fun`: `fun.__gt_definition__` if `fun` is a DSL operator,
    else `fun` itself. Raise a `TypeError` if it is not a valid DSL definition."""
    definition = getattr(fun, "__gt_definition__", fun)
    if not is_dsl_operator_definition(definition):
        raise TypeError(f"'{fun}' has no valid DSL definition.")
    return definition


type Transform[A, B] = Callable[[A], B]


@dataclasses.dataclass
class GTIR:
    """Dummy stand-in for GTIR, the AST of a DSL definition: a tree of dataclass nodes.

    Simplified: the nodes mirror the Python `ast` nodes of the definition, e.g.
    `GTIR("BinOp", {"left": GTIR("Name", ...), "op": GTIR("Add"), "right": ...})`.
    """

    kind: str  # e.g. "FunctionDef", "BinOp", "Name"
    fields: dict[str, Any] = dataclasses.field(default_factory=dict)  # child nodes or values


@runtime_checkable
class GTIRCompiler(DSLCompiler, Protocol):
    """A `DSLCompiler` going through GTIR: operator -> GTIR -> compiled DSL operator.

    Subclass it explicitly to inherit `field_operator` and `scan_operator`, which compose the
    two transforms. The frontend transform gets the operator, for its definition, kind and
    configuration; the backend transform returns the same kind of compiled operator.
    """

    frontend_transform: Transform[DSLOperator[..., Any], GTIR]
    backend_transform: Transform[GTIR, CompiledDSLOperator[..., Any]]

    def field_operator[**P, R](self, operator: FieldOperator[P, R]) -> CompiledFieldOperator[P, R]:
        compiled = self.backend_transform(self.frontend_transform(operator))
        return cast(CompiledFieldOperator[P, R], compiled)

    def scan_operator(self, operator: ScanOperator[..., Any]) -> CompiledScanOperator[..., Any]:
        compiled = self.backend_transform(self.frontend_transform(operator))
        return cast(CompiledScanOperator[..., Any], compiled)


# --- Allocation ---------------------------------------------------------------
# Creation functions follow the Python Array API standard:
# https://data-apis.org/array-api/latest/API_specification/creation_functions.html

type DType = Any  # simplified: a dtype of the array namespace, e.g. `np.float64`
type Device = Any  # simplified: a device of the array namespace
type Shape = int | tuple[int, ...]


@runtime_checkable
class NDArray(Protocol):
    """An array of the Array API standard, e.g. a `numpy.ndarray`."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> DType: ...

    def __array_namespace__(self, *, api_version: Any = None) -> Any: ...  # an `ArrayNamespace`


@runtime_checkable
class NDArrayAllocator(Protocol):
    """Any object, including a module, with the `empty*()`, `zeros*()`, `ones*()` and `full*()`
    creation functions of the Array API standard, e.g. `numpy`: `numpy.zeros((4, 3))`."""

    def empty(self, shape: Shape, *, dtype: DType = None, device: Device = None) -> NDArray: ...

    def empty_like(
        self, x: NDArray, /, *, dtype: DType = None, device: Device = None
    ) -> NDArray: ...

    def zeros(self, shape: Shape, *, dtype: DType = None, device: Device = None) -> NDArray: ...

    def zeros_like(
        self, x: NDArray, /, *, dtype: DType = None, device: Device = None
    ) -> NDArray: ...

    def ones(self, shape: Shape, *, dtype: DType = None, device: Device = None) -> NDArray: ...

    def ones_like(
        self, x: NDArray, /, *, dtype: DType = None, device: Device = None
    ) -> NDArray: ...

    def full(
        self, shape: Shape, fill_value: Scalar, *, dtype: DType = None, device: Device = None
    ) -> NDArray: ...

    def full_like(
        self, x: NDArray, /, fill_value: Scalar, *, dtype: DType = None, device: Device = None
    ) -> NDArray: ...


@runtime_checkable
class ArrayNamespace(NDArrayAllocator, Protocol):
    """Any object, including a module, implementing the Array API standard, e.g. `numpy`.

    Only the version attribute and the creation functions are listed; the rest of the
    standard is assumed.
    """

    @property
    def __array_api_version__(self) -> str: ...


@runtime_checkable
class NDArrayFieldAllocator(Protocol):
    """Like an `NDArrayAllocator`, but taking a `Domain` instead of a shape and returning
    `NDArrayField`s: `allocator.zeros({I: range(0, 4), J: range(0, 3)})`."""

    def empty(
        self, domain: Domain, *, dtype: DType = None, device: Device = None
    ) -> NDArrayField: ...

    def empty_like(
        self, x: NDArrayField, /, *, dtype: DType = None, device: Device = None
    ) -> NDArrayField: ...

    def zeros(
        self, domain: Domain, *, dtype: DType = None, device: Device = None
    ) -> NDArrayField: ...

    def zeros_like(
        self, x: NDArrayField, /, *, dtype: DType = None, device: Device = None
    ) -> NDArrayField: ...

    def ones(
        self, domain: Domain, *, dtype: DType = None, device: Device = None
    ) -> NDArrayField: ...

    def ones_like(
        self, x: NDArrayField, /, *, dtype: DType = None, device: Device = None
    ) -> NDArrayField: ...

    def full(
        self, domain: Domain, fill_value: Scalar, *, dtype: DType = None, device: Device = None
    ) -> NDArrayField: ...

    def full_like(
        self, x: NDArrayField, /, fill_value: Scalar, *, dtype: DType = None, device: Device = None
    ) -> NDArrayField: ...


type Allocator = NDArrayAllocator | NDArrayFieldAllocator


@runtime_checkable
class AllocatorProvider(Protocol):
    """Any object providing an `Allocator`, e.g. a backend or a DSL operator."""

    @property
    def __gt_allocator__(self) -> Allocator: ...


# --- Backends -----------------------------------------------------------------


@runtime_checkable
class Backend(DSLInterpreter, AllocatorProvider, Protocol):
    """A `DSLInterpreter` that is also an `AllocatorProvider`, e.g. to allocate the `out` fields
    of its operators."""
