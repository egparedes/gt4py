"""Prototype of the GT4Py V2 decorator interface, built on the stand-ins in `concepts`.

Each step and the artifact it produces:

1. `@field_operator()` or `@scan_operator(...)` checks a Python function, the definition, and
   wraps it -> a `FieldOperatorWrapper` or a `ScanOperatorWrapper`: a `FieldOperator` or a
   `ScanOperator` running the definition in Python (embedded execution).
2. Optionally, `backend(operator)` re-targets it, dispatching on its kind to
   `backend.field_operator(operator)` or `backend.scan_operator(operator)` -> the backend's
   own `FieldOperator` or `ScanOperator`. All the backends are `Backend`s: `eager`, `lazy`,
   `lazy_backward` and `column_major` interpret the definitions, and `compiled` is a
   `GTIRCompiler`, whose operators can also be compiled ahead of time with `.compile(...)`.
3. Calling the operator -> the resulting `Field`. `op[domain](...)`, or `op(..., domain=...)`,
   restricts the result to `domain`; the backends may compute only that domain.

The backends differ in how they compute the domain of the result and in when they compute
its values (see the "Semantics of domains, values and fields" notebook). All of them compute
the values operation by operation: the interpreters by running the Python definition,
`compiled` by running Python code generated from the GTIR of the definition.

| Backend         | Domain computation                     | Values computed  | Result          |
|-----------------|----------------------------------------|------------------|-----------------|
| `eager`         | forward                                | when called      | `Field`         |
| `lazy`          | forward, then sliced to the requested  | when first used  | `DeferredField` |
| `lazy_backward` | backward, from the requested domain    | when first used  | `DeferredField` |
| `column_major`  | forward                                | when called      | `Field` (*)     |
| `compiled`      | forward                                | when called      | `Field`         |

A `DeferredField` gets a default domain, computed forward at call time, which a requested
domain or `[...]` can still narrow before any value is computed. On all the backends, a
requested domain outside the domain of the result raises a `ValueError`, as for `f[key]`.
Scan operators are computed eagerly by all the backends, level by level; `compiled` runs
generated code for each level.

Fields: `NumPyArrayField` implements the `NDArrayField` concept with a numpy array, and
`DeferredField` is a `NumPyArrayField` whose values are computed on first use.
Allocation: every backend is an `AllocatorProvider`. Its `__gt_allocator__` is an
`NDArrayFieldAllocator` creating `NumPyArrayField`s: `numpy_field_allocator`, row-major (the
numpy default), except for `column_major`, whose `column_major_field_allocator` lays out the
values column-major (*): its results are stored in fields from that allocator.
Connectivities: `neighbor_table(relation, rows)` builds the neighbor table of a relation; the
meshes derive the tables of the relations declared with a `Stencil`, and of compositions like
`C2E @ E2C`, with `construct_table` (from `paths_table`, the table of an expression), and check
the given ones against their construction. In a `with Mesh({...})` block, fields are read
through relations, `e(C2E)` (a gather), possibly chained, `v(E2V)(C2E)`, or with a boundary
value through partial relations, `c(C2C.boundary(0.0))`, and the DSL builtins `sum_over`,
`max_over` and `min_over` sum, take the largest and the smallest value over neighbor
dimensions, leaves first, skipping the skip positions along their ancestry
(`neighbor_indices`). The DSL builtins `flatten` and `unflatten` convert between the neighbors
of a relation and its pairs, `Pairs[C2E]` (the meshes derive the tables of `C2E.to_pairs`,
`C2E.pair_source` and `C2E.pair_target` with `pairs_rows`), and a gather through a functional
relation adds no neighbor dimension: `e(C2E.pair_target)`. Half offsets premap fields between
the integer and the half-integer positions of an axis (`_half_domain`, `Staggered[K]` in
`concepts`): `w(K - 0.5)` reads the interfaces of each level. Arithmetic broadcasts fields with
different dimensions; field operators return their results in the dimension order of their
annotation (`as_annotated`).
Extents: `read_extents` finds the reads of each argument, as chains of steps, by a small
symbolic evaluation of the definition, through local variables and nested operator calls. The
lazy backends propagate domains through them step by step, as index sets through relations:
forward (`forward_domain`, with `sources_within`), and backward (`backward_domain`, with
`neighbors`), so that `lazy_backward` computes only the points needed; backward, a plain read
of missing neighbors raises a `ValueError`: the request is beyond the computable domain.
Local meshes: `decompose` builds the `LocalMesh` of each rank from a partition of a global mesh
(the tables derived and checked globally, then localized, `TRUNCATED` outside the local
support), and `scatter` and `gather_owned` move global fields to the ranks and back. The
operators run unchanged on a rank, in `with local_mesh:`, e.g. on `Owned(Cell)`; domains
are combined with `intersect` and `difference`. A read beyond the local support is not
computable: forward, it is not in the domain; backward, it raises a `ValueError`.
Local programs: `@local_program` marks a Python function that `run_local` runs once per rank,
as threads, each in `with local_mesh:`. Its fields are values, valid on their domains
(writing into a field raises there): the collective `exchange` returns a field on the owned
points with its halo, from the owners, and `global_sum` sums on owned points, bitwise the same
for any decomposition. DSL operators, on any backend, are called in it as anywhere.
Fields on a logical mesh: a `NumPyMeshField` is a `NumPyArrayField` with its `LogicalMesh`, a
`MeshField`, allocated through the mesh (`grid.zeros(Cell, K)`, `grid.field(Cell, data=...)`,
with `mesh_field`). Its premaps and neighbor reductions use the tables of its mesh without a
`with` block; arithmetic keeps the brand, also with a plain field on the left, and rejects
fields of another mesh (a `ValueError`); DSL operators called with mesh fields run in their mesh
and brand their results, except the `DeferredField`s of the lazy backends (branding would
compute them).
Distributions: a `Distribution` of a logical mesh is given, `distribute(mesh, owner,
support=..., hops=...)` (a halo of `hops` over relations, as `decompose`), or derived,
`derive_distribution(mesh, owner, programs)`: each rank's halo is exactly the `footprints` of its
owned points through the reads of the programs' operators; `check_coverage` checks that a
distribution covers them. For the lowering, the private lenient `_exchange(..., strict=False)` and
`_scatter_valid` move values defined on part of the mesh (e.g. on the interior).
Programs: `@program()` checks a Python function with `PROGRAM_RULES` and wraps it in a
`Program`, run directly on the logical mesh of its field arguments (the reference, global
semantics). Each DSL operator call in its body is handed to the running program (`_call`), which
runs the operator outside the program, in an operator-level window, and materializes its
result there; `sum_over` over a `Location` is a reproducible global sum (`_ordered_total`). The
guard: at program level, premaps, subscripts, writes, neighbor reductions, `flatten` and
`unflatten` raise a `TypeError` (pointwise arithmetic is allowed): programs compute with fields
in field operators.
The lowering: `program.lower(distribution)` checks the coverage of the program's operators and
returns a `LoweredProgram`, which runs the program on every rank as a local program
(`run_local`), on its local fields (the inputs given with `_scatter_valid`): before each operator
call, every argument read through a relation is exchanged (lenient `_exchange`), unless it is
fresh (already an exchange); results are computed on the owned points; `sum_over` over a
`Location` is a `global_sum`; the results are gathered (`gather_owned`) and branded. The rules
are rank-independent, so all ranks call the same collectives. The simulation condition: the
lowered run gives bitwise the results of the direct run, for any distribution covering the reads.
"""

from __future__ import annotations

import ast
import contextlib
import contextvars
import dataclasses
import functools
import inspect
import itertools
import operator
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import (
    Any,
    ClassVar,
    Concatenate,
    Literal,
    Self,
    TypeAliasType,
    TypeIs,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

import numpy as np

from concepts import (
    FIELD_OPERATOR_RULES,
    GTIR,
    MISSING_VALUE,
    PROGRAM_RULES,
    SCAN_OPERATOR_RULES,
    SKIP_VALUE,
    TRUNCATED,
    Backend,
    Boundary,
    CompiledDSLOperator,
    Compose,
    Device,
    Dimension,
    DimensionIndex,
    Distribution,
    Domain,
    DomainSpec,
    DSLOperator,
    DSLOperatorDefinition,
    DSLOperatorKind,
    DType,
    Expression,
    Extents,
    Field,
    FieldOperator,
    FieldOperatorDefinition,
    Flatten,
    GTIRCompiler,
    HalfOffset,
    IndexSet,
    Key,
    LocalMesh,
    Location,
    LogicalMesh,
    Mesh,
    NDArrayField,
    NDArrayFieldAllocator,
    Neighbor,
    NeighborTable,
    Pairs,
    Premap,
    Read,
    Reduction,
    Region,
    Relation,
    Scalar,
    ScanAlong,
    ScanConfig,
    ScanOperator,
    ScanOperatorDefinition,
    Staggered,
    Stencil,
    Step,
    Transform,
    Transpose,
    Unflatten,
    call_operator,
    check_domain,
    composition_of,
    current_mesh,
    definition_of,
    dims_of,
    expression_name,
    flip_staggered,
    getsource,
    is_field_operator_definition,
    is_inspectable,
    is_relation,
    is_scan_operator_definition,
    neighbor_of,
    pairs_of,
    program_mesh,
    restrict,
    root_of,
)

# --- Fields -------------------------------------------------------------------


class NumPyArrayField[D: tuple[DimensionIndex, ...], T]:
    """An `NDArrayField` storing its values in the numpy array `data`.

    Along a dimension whose domain is an `IndexSet`, the values are in the order of the indices.
    The domain is checked with `check_domain`. A field on a logical mesh is a `NumPyMeshField`.
    """

    def __init__(self, domain: Domain, data: Any) -> None:
        check_domain(domain)
        self.domain = domain
        self.data = np.asarray(data)

    def __repr__(self) -> str:
        return f"Field({format_domain(self.domain)}: {self.data.tolist()})"

    def __call__(self, *premaps: Premap) -> NumPyArrayField[Any, T]:
        """Premap (see `Field.__call__`); not at program level (a `TypeError`)."""
        _not_in_program("A premap")
        field: NumPyArrayField[Any, T] = self
        for premap in premaps:
            _check_premap(premap)
            if _is_gather(premap):
                field = field._gather(premap)
            elif _is_half(premap):
                dim, offset = premap
                partner = flip_staggered(dim)
                if partner not in field.domain:
                    raise TypeError(
                        f"`({dim.__name__}, {offset})` reads a field along "
                        f"'{partner.__name__}', not {format_domain(field.domain)}."
                    )
                field = NumPyArrayField(_half_domain(field.domain, (dim, offset)), field.data)
            else:
                field = field._shift(*premap)
        return field

    def _shift(self, dim: Dimension, offset: int) -> NumPyArrayField[Any, T]:
        """Premap by an integer offset `dim + offset` (see `Field.__call__`)."""
        return NumPyArrayField({**self.domain, dim: shift(self.domain[dim], -offset)}, self.data)

    def _gather(self, premap: type[Relation[Any, Any]] | Boundary) -> NumPyArrayField[Any, T]:
        """Read the values at the neighbors given by the table of the relation in the current
        mesh, on the sources whose neighbors are all in this field's domain, or with a
        `Boundary`, the boundary value at the missing ones (see `Field.__call__`)."""
        relation = _gathered(premap)
        if isinstance(premap, Boundary) and (premap.value is None or not relation.partial):
            raise TypeError(
                f"Read with `{_parenthesized(relation)}.boundary(value)`, through a partial "
                f"relation: not {premap}."
            )
        source, target = relation.source, relation.target
        domain = _forward_step(self.domain, premap)  # `(source, Neighbor[R], *rest)`, reparented
        table = np.asarray(current_mesh()[relation][{source: domain[source]}].data)
        axis = list(self.domain).index(target)
        data = self.data
        if data.shape[axis] == 0:  # only missing neighbors: take from a zero pad, all masked
            data = np.zeros((*data.shape[:axis], 1, *data.shape[axis + 1 :]), dtype=data.dtype)
        positions = np.where(table >= 0, _positions(self.domain[target], table), 0)
        values = np.moveaxis(np.take(data, positions, axis=axis), (axis, axis + 1), (0, 1))
        table = table.reshape(table.shape + (1,) * (values.ndim - 2))
        values = np.where(table >= 0, values, 0)  # 0 at the skip positions
        if isinstance(premap, Boundary):
            values = np.where(table == MISSING_VALUE, cast(Scalar, premap.value), values)
        if relation.functional:  # the only neighbor, without a neighbor dimension
            values = values[:, 0]
        return NumPyArrayField(domain, values)

    def __getitem__(self, key: Key) -> NumPyArrayField[D, T]:
        """Restrict to `key` (see `Field.__getitem__`); not at program level (a `TypeError`)."""
        _not_in_program("A subscript")
        return self._restricted(key)

    def _restricted(self, key: Key) -> NumPyArrayField[D, T]:
        """The unguarded restriction of `__getitem__`, for internal callers that run at program
        level, e.g. pointwise arithmetic (`_apply`)."""
        return NumPyArrayField(self.subdomain(key), self.data[self._array_index(key)])

    def subdomain(self, key: Key) -> Domain:
        """Return this field's domain restricted by `key`, along its own dimensions (keys on
        others are ignored). Raise a `ValueError` if `key` selects points outside the domain."""
        wanted = _key_domain(key)
        for dim, own in self.domain.items():
            if dim in wanted and not _within(wanted[dim], own):
                outside = format_domain({dim: wanted[dim]})
                raise ValueError(
                    f"{outside} is not within the domain {format_domain(self.domain)}."
                )
        return {dim: wanted.get(dim, own) for dim, own in self.domain.items()}

    def __setitem__(self, key: Key, value: Field[D, T] | Scalar) -> None:
        """Write values into a sub-domain (see `NDArrayField`); a `RuntimeError` in a running
        local program, where fields are values (see `run_local`), and a `TypeError` at program
        level."""
        _not_in_program("A write")
        if _running.get() is not None:
            raise RuntimeError(
                "Fields are values in a local program: compute a new field instead of writing "
                f"into {format_domain(self.domain)}, e.g. `exchange(f)` for its halo."
            )
        _write(self, key, value)

    def _array_index(self, key: Key) -> Any:
        """Translate a subscript into a numpy index into `data`: slices along ranges, else
        integer arrays (an open mesh, `np.ix_`)."""
        wanted = _key_domain(key)
        index = [
            _positions(own, wanted[dim]) if dim in wanted else slice(None)
            for dim, own in self.domain.items()
        ]
        if all(isinstance(i, slice) for i in index):
            return tuple(index)
        arrays = (
            np.arange(n)[i] if isinstance(i, slice) else i for i, n in zip(index, self.data.shape)
        )
        return np.ix_(*arrays)

    def _apply(self, op: Callable[[Any, Any], Any], other: Field | Scalar) -> NumPyArrayField:
        """Apply `op` point by point, broadcasting fields with different dimensions (see
        `Field`): `intersect` keeps the dimensions of `self`, then the missing ones of `other`.
        The brand: if `self` or `other` is a `NumPyMeshField`, the result is one of its mesh; a
        `ValueError` if they are on different logical meshes. At program level, both fields must
        be the program's (see `Program`), and the result is one."""
        run = _program_run.get()
        if run is not None:
            run.check(self, "an operand of pointwise arithmetic")
            run.check(other, "an operand of pointwise arithmetic")
        mesh = _mesh_of((self, other))
        result: NumPyArrayField
        if not isinstance(other, NDArrayField):
            result = NumPyArrayField(self.domain, op(self.data, other))
        else:
            domain = intersect(self.domain, other.domain)
            dims = list(domain)
            # unguarded restrictions: pointwise arithmetic is allowed at program level
            part = (
                other._restricted(domain) if isinstance(other, NumPyArrayField) else other[domain]
            )
            left = _aligned(self._restricted(domain).data, list(self.domain), dims)
            right = _aligned(np.asarray(part.data), list(other.domain), dims)
            result = NumPyArrayField(domain, op(left, right))
        result = result if mesh is None else mesh_field(mesh, result.domain, result.data)
        return result if run is None else run.remember(result)

    def __add__(self, other: Field | Scalar) -> NumPyArrayField:
        return self._apply(operator.add, other)

    def __sub__(self, other: Field | Scalar) -> NumPyArrayField:
        return self._apply(operator.sub, other)

    def __rsub__(self, other: Field | Scalar) -> NumPyArrayField:
        return self._apply(lambda a, b: b - a, other)

    def __mul__(self, other: Field | Scalar) -> NumPyArrayField:
        return self._apply(operator.mul, other)

    __radd__ = __add__
    __rmul__ = __mul__


def _write(field: NumPyArrayField, key: Key, value: Field | Scalar) -> None:
    """The private write path into `field`, also in local programs, for fields being built,
    e.g. by `store`. The DSL rules reject it in the source of DSL definitions, also aliased (see
    `DSLRules`; a string `getattr` is beyond their static checks)."""
    field.subdomain(key)  # raises outside the domain
    field.data[field._array_index(key)] = (
        value[key].data if isinstance(value, NDArrayField) else value
    )


def _parenthesized(relation: type[Relation[Any, Any]]) -> str:
    """The name of `relation`, in parentheses if it is a composition: `(C2E @ E2C)`."""
    return f"({relation.__name__})" if " @ " in relation.__name__ else relation.__name__


def _is_half(step: Step) -> TypeIs[HalfOffset]:
    """Whether `step` is a half offset `(D, h)`, a float: a premap between the two cell
    classes of the axis `D` (see `Staggered` in `concepts`)."""
    return isinstance(step, tuple) and isinstance(step[1], float)


def _half_shift(dim: Dimension, offset: float) -> int:
    """The `c` of D3: `h + 1/2` for a declared axis, `h - 1/2` for a staggered one, rounded
    to an integer (the position convention `Staggered[K](i)` at `i - 1/2`)."""
    return int(offset + 0.5) if not issubclass(dim, Staggered) else int(offset - 0.5)


def _half_domain(domain: Domain, step: HalfOffset) -> Domain:
    """Rename the staggered partner of `step[0]` to `step[0]` in `domain`, in place, with its
    indices shifted by `-c` (see `_half_shift`): the domain of the field premaped by the half
    offset. The partner keeps its position in the order of `domain`, so the axes of `data`
    still match."""
    dim, offset = step
    partner = flip_staggered(dim)
    if partner not in domain:
        return domain
    indices = domain[partner]
    shifted = shift(indices, -_half_shift(dim, offset))
    return {dim if d is partner else d: shifted if d is partner else i for d, i in domain.items()}


def _reparented(dim: Dimension, root: Dimension, parent: Dimension) -> Dimension:
    """Rename `dim` for a gather from `root`: `root` becomes `parent`, and every neighbor
    dimension hanging from `root` is renamed, recursively: `Neighbor[X, root]` ->
    `Neighbor[X, parent]`."""
    if dim is root:
        return parent
    if issubclass(dim, Neighbor):
        return neighbor_of(dim.relation, _reparented(dim.parent, root, parent))
    return dim


def _aligned(data: Any, own: list[Dimension], dims: list[Dimension]) -> Any:
    """Return `data`, with one axis per dimension in `own`, transposed to the order of `dims`,
    which holds them all, with axes of length 1 for the others: ready to broadcast."""
    order = [own.index(dim) for dim in dims if dim in own]
    missing = tuple(i for i, dim in enumerate(dims) if dim not in own)
    return np.expand_dims(np.transpose(data, order), missing)


def as_annotated[R](result: R, annotation: Any) -> R:
    """Return `result` in the dimension order of `annotation`, the result annotation of a
    definition, if it is a `Field` with the same dimensions (they may differ in order after
    broadcasting). Tuples are transposed element by element."""
    annotation = _resolve(annotation)
    if get_origin(annotation) is tuple and isinstance(result, tuple):
        elements = zip(result, get_args(annotation), strict=True)
        return cast(R, tuple(as_annotated(r, a) for r, a in elements))
    if get_origin(annotation) is not Field or not isinstance(result, NumPyArrayField):
        return result
    own, dims = list(result.domain), list(dims_of(annotation))
    if own == dims or set(own) != set(dims):
        return result
    domain = {dim: result.domain[dim] for dim in dims}
    return cast(R, NumPyArrayField(domain, _aligned(result.data, own, dims)))


def index_set(indices: Iterable[int]) -> range | IndexSet:
    """Return `indices` sorted, without duplicates: a `range` if they are consecutive."""
    values = sorted({int(i) for i in indices})
    if not values:
        return range(0)
    if values[-1] - values[0] + 1 == len(values):
        return range(values[0], values[-1] + 1)
    return tuple(values)


def shift(indices: range | IndexSet, offset: int) -> range | IndexSet:
    """Return `indices` shifted by `offset`."""
    if isinstance(indices, range):
        return range(indices.start + offset, indices.stop + offset)
    return tuple(i + offset for i in indices)


def intersect(a: Domain, b: Domain) -> Domain:
    """Return the intersection of two domains; a dimension in only one of them is kept as is."""
    return {
        dim: _intersect(a[dim], b[dim])
        if dim in a and dim in b
        else (a[dim] if dim in a else b[dim])
        for dim in dict.fromkeys([*a, *b])
    }


def difference(a: Domain, b: Domain) -> Domain:
    """Return the points of `a` that are not in `b`, e.g. `difference(Owned(Cell),
    region(LateralBoundary))`: `b` must have exactly one dimension, one of `a` (else a
    `ValueError`: the difference would not be a domain)."""
    if len(b) != 1 or next(iter(b)) not in a:
        raise ValueError(f"{format_domain(b)} must have one dimension, of {format_domain(a)}.")
    (dim,) = b
    return {**a, dim: index_set(set(a[dim]) - set(b[dim]))}


def _within(indices: range | IndexSet, own: range | IndexSet) -> bool:
    """Return whether all the `indices` are in `own`."""
    if isinstance(indices, range) and isinstance(own, range):
        return not indices or own.start <= indices.start and indices.stop <= own.stop
    return set(indices) <= set(own)


def _intersect(a: range | IndexSet, b: range | IndexSet) -> range | IndexSet:
    if isinstance(a, range) and isinstance(b, range):
        return range(max(a.start, b.start), min(a.stop, b.stop))
    return index_set(set(a) & set(b))


def union(parts: list[range | IndexSet]) -> range | IndexSet:
    """Return the union of `parts`: the smallest range holding them if they are all ranges."""
    if all(isinstance(p, range) for p in parts):
        return range(min(p.start for p in parts), max(p.stop for p in parts))  # type: ignore[union-attr]
    return index_set(i for p in parts for i in p)


def format_domain(domain: Domain) -> str:
    """Format a domain as `I[0:4], J[0:3]`, with index sets as `Cell{0, 2, 5}`."""
    return ", ".join(
        f"{dim.__name__}[{r.start}:{r.stop}]"
        if isinstance(r, range)
        else f"{dim.__name__}{{{', '.join(map(str, r))}}}"
        for dim, r in domain.items()
    )


def _key_domain(key: Key) -> Domain:
    """Normalize a field subscript to the domain it selects, along some dimensions."""
    if key is Ellipsis:
        return {}
    if isinstance(key, dict):
        return key
    slices = key if isinstance(key, tuple) else (key,)
    for s in slices:
        if not isinstance(s.start, DimensionIndex) or type(s.stop) is not type(s.start):
            raise TypeError("Slice bounds are positions along a dimension: `I.at(1):I.at(3)`.")
    return {type(s.start): range(s.start, s.stop) for s in slices}


def _positions(own: range | IndexSet, indices: Any) -> Any:
    """Return the positions in `own` of `indices`, in `own` (happy path): a slice if both are
    ranges, else an integer array of the shape of `indices`."""
    if isinstance(own, range):
        if isinstance(indices, range):
            return slice(indices.start - own.start, indices.stop - own.start)
        return np.asarray(indices, dtype=int) - own.start
    return np.searchsorted(np.asarray(own, dtype=int), np.asarray(indices, dtype=int))


class DeferredField(NumPyArrayField):
    """A `NumPyArrayField` whose values are computed on first use, only on its (possibly
    narrowed) domain, in the operator-level window (see "Programs"): the computation is the
    operator's, also when the values are first used at program level."""

    def __init__(self, compute: Callable[[Domain], Field], domain: Domain) -> None:
        check_domain(domain)
        self.compute = compute
        self.domain = domain

    @functools.cached_property
    def data(self) -> Any:  # type: ignore[override]  # a cached property is writable too
        # Computing an operator's values is operator-level work, also when first used at program
        # level (a deferred field made outside the program): in the window, not guarded.
        with _operator_level():
            result = self.compute(self.domain)
            assert isinstance(result, NDArrayField)  # happy path: operators return NDArrayFields
            order = [list(result.domain).index(dim) for dim in self.domain]
            # in the order of the dimensions of `domain`
            return np.transpose(np.asarray(result.data), order)

    def _restricted(self, key: Key) -> NumPyArrayField:
        """Narrow the domain; once the values are computed, only slice them (`f[key]` too, which
        is guarded, see `NumPyArrayField.__getitem__`)."""
        if "data" in self.__dict__:  # computed: the cached property is in the instance dict
            return super()._restricted(key)
        return DeferredField(self.compute, self.subdomain(key))


class NumPyMeshField[D: tuple[DimensionIndex, ...], T](NumPyArrayField[D, T]):
    """A `MeshField`: a `NumPyArrayField` on the logical mesh `mesh`, with global indices,
    created through the mesh (`grid.zeros(Cell)`, ..., see `mesh_field`).

    Premaps use the tables of `mesh` without a `with` block, and the results of premaps,
    subscripts, arithmetic and neighbor reductions are mesh fields of `mesh` (the brand). The
    overrides go through the methods of `NumPyArrayField` (`super()`), so their checks apply.
    """

    def __init__(self, domain: Domain, data: Any, mesh: LogicalMesh) -> None:
        super().__init__(domain, data)
        self.mesh = mesh

    def __call__(self, *premaps: Premap) -> NumPyMeshField[Any, T]:
        """Premap as `NumPyArrayField`, through the tables of `mesh` (in `with self.mesh:`)."""
        with self.mesh:
            result = super().__call__(*premaps)
        return mesh_field(self.mesh, result.domain, result.data)

    def __getitem__(self, key: Key) -> NumPyMeshField[D, T]:
        result = super().__getitem__(key)
        return mesh_field(self.mesh, result.domain, result.data)

    def __setitem__(self, key: Key, value: Field[D, T] | Scalar) -> None:
        """Write as `NumPyArrayField`; a `ValueError` if `value` is a field of another logical
        mesh."""
        _mesh_of((self, value))
        super().__setitem__(key, value)


def mesh_field(mesh: LogicalMesh, domain: Domain, data: Any) -> NumPyMeshField[Any, Any]:
    """Return the field of `mesh` with the values `data` on `domain`: a scalar `data` is
    broadcast, an array must have the shape of `domain` (else a `ValueError`). The hook of
    `LogicalMesh.make_field`, behind `grid.zeros(...)`, `grid.full(...)` and `grid.field(...)`."""
    shape = _shape(domain)
    values = np.full(shape, data) if np.ndim(data) == 0 else np.asarray(data)
    if values.shape != shape:
        raise ValueError(
            f"The values of a field on {format_domain(domain)} must have the shape {shape}, "
            f"not {values.shape}."
        )
    return NumPyMeshField(domain, values, mesh)


def _mesh_of(values: Iterable[Any]) -> LogicalMesh | None:
    """The logical mesh of the `NumPyMeshField`s among `values` (the brand), `None` if there
    are none; a `ValueError` if they are on different logical meshes (different `mesh_id`s)."""
    meshes: dict[Any, LogicalMesh] = {}
    for value in values:
        if isinstance(value, NumPyMeshField):
            meshes.setdefault(value.mesh.mesh_id, value.mesh)
    if len(meshes) > 1:
        *first, last = sorted(f"'{mesh_id}'" for mesh_id in meshes)
        raise ValueError(
            f"The fields are on different logical meshes, {', '.join(first)} and {last}."
        )
    return next(iter(meshes.values()), None)


def _field_on(mesh: LogicalMesh | None, domain: Domain, data: Any) -> NumPyArrayField[Any, Any]:
    """A plain `NumPyArrayField` if `mesh` is `None`, else a field of `mesh` (`mesh_field`)."""
    return NumPyArrayField(domain, data) if mesh is None else mesh_field(mesh, domain, data)


def _branded[R](result: R, mesh: LogicalMesh) -> R:
    """Brand every field of `result`, a field or a (nested) tuple of fields, with `mesh`, as
    `restrict` restricts them; `DeferredField`s are kept (branding would compute them), and
    fields of a mesh already carry one."""
    if isinstance(result, tuple):
        return cast(R, tuple(_branded(r, mesh) for r in result))
    if isinstance(result, NumPyArrayField) and not isinstance(
        result, (DeferredField, NumPyMeshField)
    ):
        return cast(R, mesh_field(mesh, result.domain, result.data))
    return result


# --- Connectivities -----------------------------------------------------------


def neighbor_table(relation: type[Relation[Any, Any]], rows: Any) -> NumPyArrayField[Any, int]:
    """Return a neighbor table for `relation`: `rows[s]` lists the neighbors of the source `s`,
    for `s` in `0, ..., len(rows) - 1`, padded with `SKIP_VALUE`, with `MISSING_VALUE` for the
    missing ones. Without rows (e.g. on a rank that owns nothing), the table has the shape
    `(0, max_neighbors)`."""
    data = np.asarray(rows, dtype=int)
    if len(data) == 0:  # no rows: `(0, max_neighbors)`, not `(0,)` (other shapes are checked)
        data = data.reshape(0, relation.max_neighbors)
    domain: Domain = {
        relation.source: range(len(data)),
        neighbor_of(relation): range(relation.max_neighbors),
    }
    return NumPyArrayField(domain, data)


def paths_table(expr: Expression, the_mesh: Mesh) -> Any:
    """Return the table of the staged expression `expr` on `the_mesh` (see "Footprint algebra"
    in `concepts`), as an array with a row per source, from 0 (happy path): of a relation, its
    table; of a composition, the targets of the second from each position of the first, along
    all the paths (a skip position or missing neighbor of the first gives as many of them); of a
    transpose, the sources reaching each target, in increasing order, padded with `SKIP_VALUE`,
    for the extent of its source (a `ValueError` without it)."""
    if is_relation(expr):
        return np.asarray(the_mesh[expr].data)
    if isinstance(expr, Compose):
        first, second = paths_table(expr.first, the_mesh), paths_table(expr.second, the_mesh)
        groups = np.where(first[..., None] >= 0, second[np.maximum(first, 0)], first[..., None])
        return groups.reshape(len(first), -1)
    expr = cast(Transpose, expr)
    if expr.source not in the_mesh.extents:
        raise ValueError(
            f"`{expr}` needs the extent of '{expr.source.__name__}': give it to the mesh, "
            f"`extents={{{expr.source.__name__}: ...}}`."
        )
    table = paths_table(expr.expr, the_mesh)
    reaching = [
        np.flatnonzero((table == t).any(axis=1)) for t in range(the_mesh.extents[expr.source])
    ]
    width = max((len(sources) for sources in reaching), default=0)
    rows = [
        np.pad(sources, (0, width - len(sources)), constant_values=SKIP_VALUE)
        for sources in reaching
    ]
    return np.asarray(rows, dtype=int).reshape(len(rows), width)


def stencil_rows(relation: type[Relation[Any, Any]], the_mesh: Mesh) -> list[list[int]]:
    """Return the rows of the table of `relation` built from its `stencil` (see `Stencil`)."""
    stencil = cast(Stencil, relation.stencil)
    if stencil.paths == "exclude_self":
        parts = cast(Compose, composition_of(stencil.expr))
        first = paths_table(parts.first, the_mesh)
        second = paths_table(parts.second, the_mesh)
        rows = []
        for source, hops in enumerate(first.tolist()):
            row = []
            for hop in hops:  # a group per position of the row of the first relation
                if hop < 0:  # skip or missing: `w2 - 1` of them
                    row += [hop] * (second.shape[1] - 1)
                    continue
                group = second[hop].tolist()
                if group.count(source) != 1:
                    raise ValueError(
                        f"Cannot exclude {format_domain({relation.source: (source,)})} from "
                        f"`{expression_name(parts)}`: it is {group.count(source)} times among "
                        f"the neighbors {group} of {format_domain({parts.first.target: (hop,)})}."
                    )
                row += [target for target in group if target != source]
            rows.append(row)
    else:
        rows = paths_table(stencil.expr, the_mesh).tolist()
    if stencil.include_self:
        rows = [[source, *row] for source, row in enumerate(rows)]
    if stencil.paths == "unique":  # after the source: first, once
        rows = [list(dict.fromkeys(t for t in row if t != SKIP_VALUE)) for row in rows]
        # in a local mesh, distinct truncated neighbors would merge into one: a row with one is
        # never computable, and all truncated, so the others keep the width
        rows = [[TRUNCATED] * relation.max_neighbors if TRUNCATED in row else row for row in rows]
    if any(len(row) > relation.max_neighbors for row in rows):
        width = max(len(row) for row in rows)
        raise ValueError(
            f"The stencil of '{relation.__name__}' has {width} neighbors, more than its "
            f"`max_neighbors`, {relation.max_neighbors}."
        )
    return [row + [SKIP_VALUE] * (relation.max_neighbors - len(row)) for row in rows]


def pairs_rows(relation: type[Relation[Any, Any]], the_mesh: Mesh) -> Any:
    """Return the rows of the table of a relation of the pairs of `R` (see `Pairs`), as an
    array: of `R.to_pairs`, the rank of each neighbor of `R` in row-major order, and
    `SKIP_VALUE` at the skip positions; of `R.pair_source` and `R.pair_target`, the source and the
    target of each pair."""
    base, role = cast(tuple[type[Relation[Any, Any]], str], relation.pairs_role)
    table = paths_table(base, the_mesh)
    valid = table != SKIP_VALUE
    if role == "to_pairs":
        return np.where(valid, np.cumsum(valid).reshape(valid.shape) - 1, SKIP_VALUE)
    sources = np.nonzero(valid)[0]
    return (sources if role == "pair_source" else table[valid])[:, None]


def construct_table(
    the_mesh: Mesh, relation: type[Relation[Any, Any]], table: NeighborTable | None
) -> NeighborTable:
    """Derive the table of `relation` on `the_mesh` from its `stencil`, or of pairs (see
    `Pairs`), or check a given `table` against its construction and return it (see `Mesh`): a
    `ValueError` if it is not the derived one, or if a row has a neighbor outside the footprint
    of its `reach`."""
    derive = pairs_rows if relation.pairs_role is not None else stencil_rows
    if table is None:
        return neighbor_table(relation, derive(relation, the_mesh))
    given = np.asarray(table.data)
    if relation.stencil is not None or relation.pairs_role is not None:
        rows = derive(relation, the_mesh)
        derived = np.asarray(rows, dtype=int).reshape(len(rows), relation.max_neighbors)
        construction = "its stencil" if relation.stencil is not None else "its pairs"
        if given.shape != derived.shape:
            raise ValueError(
                f"The table of '{relation.__name__}' must have the shape {derived.shape}, as "
                f"{construction}."
            )
        wrong = np.flatnonzero((given != derived).any(axis=1))
        if len(wrong):
            raise ValueError(
                f"The table of '{relation.__name__}' is not the one of {construction}: the row "
                f"of {wrong[0]} is {given[wrong[0]].tolist()}, not {derived[wrong[0]].tolist()}."
            )
    if relation.reach is not None:
        footprints = paths_table(relation.reach, the_mesh).tolist()
        for s, (row, footprint) in enumerate(zip(given.tolist(), footprints, strict=True)):
            valid = [t for t in row if t >= 0]
            outside = set(valid) - set(footprint)
            if MISSING_VALUE in row and MISSING_VALUE not in footprint:
                outside.add(MISSING_VALUE)  # not at the boundary through its reach
            if outside or len(valid) != len(set(valid)):
                raise ValueError(
                    f"The table of '{relation.__name__}' has the row {row} for {s}: its neighbors "
                    f"must be distinct, within its reach, `{expression_name(relation.reach)}`, "
                    f"{sorted(set(footprint) - {SKIP_VALUE})}, missing only if it is."
                )
    return table


Mesh.construct = construct_table  # how meshes derive and check the tables (see `Mesh`)
LogicalMesh.make_field = mesh_field  # how logical meshes create their fields (see `LogicalMesh`)


def neighbors(
    relation: type[Relation[Any, Any]], sources: range | IndexSet, boundary: bool = False
) -> range | IndexSet:
    """Backward propagation through `relation`: the indices of all the neighbors of `sources`,
    from the table of `relation` in the current mesh (given or derived), their footprint: a set,
    without order or multiplicity. A `ValueError` if some are missing, unless they are read
    with a `boundary` value, or truncated, in a local mesh (outside its support: a coverage
    error)."""
    table = np.asarray(current_mesh()[relation][{relation.source: sources}].data)
    if (table == TRUNCATED).any():
        truncated = np.asarray(sources, dtype=int)[(table == TRUNCATED).any(axis=1)]
        raise ValueError(
            f"{format_domain({relation.source: index_set(truncated)})} is beyond the local "
            f"support: it has neighbors through '{relation.__name__}' outside it (decompose "
            "with more `hops`)."
        )
    if not boundary and (table == MISSING_VALUE).any():
        missing = np.asarray(sources, dtype=int)[(table == MISSING_VALUE).any(axis=1)]
        raise ValueError(
            f"{format_domain({relation.source: index_set(missing)})} is beyond the computable "
            f"domain: it has missing neighbors through '{relation.__name__}' (read with "
            f"`{_parenthesized(relation)}.boundary(value)` there)."
        )
    return index_set(table[table >= 0])


def sources_within(
    relation: type[Relation[Any, Any]], targets: range | IndexSet, boundary: bool = False
) -> range | IndexSet:
    """Forward propagation through `relation`: the sources in the table of `relation`, in the
    current mesh, whose neighbors are all in `targets`; with a `boundary` value, all but the
    missing ones. A truncated neighbor, in a local mesh, is never in `targets`."""
    table = current_mesh()[relation]
    data = np.asarray(table.data)
    inside = np.isin(data, np.asarray(targets, dtype=int)) | (data == SKIP_VALUE)
    if boundary:
        inside |= data == MISSING_VALUE
    return index_set(np.asarray(table.domain[relation.source], dtype=int)[inside.all(axis=1)])


def _reduce[T](
    field: Field[Any, T],
    axis: Dimension,
    reduction: Callable[..., Any],
    identity: Callable[[DType], Any],
) -> NumPyArrayField[Any, T]:
    """The shared body of the neighbor reductions, after the `axis` check of each: the leaf rule
    (the same `ValueError`), the skip mask along the path of `axis` (a point is a skip position
    if any position on its path, from the root of `axis` down to `axis`, holds `SKIP_VALUE`),
    the masked values replaced by `identity(dtype)`, then `reduction(data, axis=...)`, `axis`
    removed from the domain of the result (see `sum_over`, `max_over` and `min_over`). A
    `NumPyMeshField` is reduced in its mesh (its tables), and the result is one of its mesh. Not
    at program level (a `TypeError`): reduce over neighbors in field operators."""
    _not_in_program("A reduction over a neighbor dimension")
    values = cast(NumPyArrayField[Any, T], field)
    dims = list(values.domain)
    children = [dim.__name__ for dim in dims if issubclass(dim, Neighbor) and dim.parent is axis]
    if children:
        raise ValueError(f"Reduce over {children} before '{axis.__name__}', which they hang from.")
    domain = {dim: indices for dim, indices in values.domain.items() if dim is not axis}
    data = values.data
    path = _path(axis)
    mesh = values.mesh if isinstance(values, NumPyMeshField) else None
    if any(cast("type[Neighbor[Any]]", dim).relation.skip_values for dim in path[1:]):
        with mesh if mesh is not None else contextlib.nullcontext():  # the tables of its mesh
            _, valid = neighbor_indices(values.domain, axis)
        data = np.where(_aligned(valid, path, dims), data, identity(data.dtype))
    result = reduction(data, axis=dims.index(axis))
    return _field_on(mesh, domain, result)


@overload
def sum_over[T](  # type: ignore[overload-overlap]
    field: Field[Any, T], /, *, axis: type[Location], domain: Domain | None = None
) -> T: ...
@overload
def sum_over[T](field: Field[Any, T], /, *, axis: Dimension) -> NumPyArrayField[Any, T]: ...
def sum_over(field: Field[Any, Any], /, *, axis: Dimension, domain: Domain | None = None) -> Any:
    """DSL builtin: sum the values of `field` over its neighbor dimension `axis`, a
    `Neighbor[R, P]`, e.g. `sum_over(e(C2E), axis=Neighbor[C2E])`; in a program, over a
    `Location`, a global sum.

    `axis` must be a `Neighbor` dimension (else a `TypeError`) and a leaf: no other neighbor
    dimension of `field` hangs from it (else a `ValueError`). The skip positions, from the tables
    in the current mesh, are not summed: a point is a skip position if any position on its
    path, from the root of `axis` down to `axis`, holds `SKIP_VALUE`. Missing neighbors are not
    skipped: they hold the value of a boundary read (see `Field.__call__`). The sum of a point
    with no neighbors is 0. `axis` is removed from the dimensions of the result. Not at program
    level (the guard's `TypeError`): reduce over neighbors in field operators.

    In a running program (at program level, not in the operators it calls), `sum_over(f,
    axis=Cell, domain=...)` over a `Location` is the scalar sum of `f` on `domain` (all of `f`
    by default), a global reduction: the values are added one by one in the order of their
    global indices (`_ordered_total`), bitwise the same for any decomposition. Phase 2 sums
    fields with one dimension, the `axis` (else a `NotImplementedError`). Elsewhere a
    `TypeError`; `domain` is only for these sums (a `TypeError` with a `Neighbor` axis).
    """
    if not issubclass(axis, Neighbor):
        run = _program_run.get()
        if run is not None and issubclass(axis, Location):
            return run.total(field, axis, domain)
        raise TypeError(
            f"`sum_over` sums over a neighbor dimension, `Neighbor[R]`, not '{axis.__name__}'. "
            "Over a `Location`, call it in a `@program` (global semantics), or `global_sum` in a "
            "local program."
        )
    if domain is not None:
        raise TypeError("`domain` is for sums over a `Location`, in a program.")
    return _reduce(field, axis, np.sum, lambda dtype: 0)


def max_over[T](field: Field[Any, T], /, *, axis: Dimension) -> NumPyArrayField[Any, T]:
    """DSL builtin: the largest value of `field` over its neighbor dimension `axis`, a
    `Neighbor[R, P]`, e.g. `max_over(e(C2E), axis=Neighbor[C2E])`.

    `axis` must be a `Neighbor` dimension (else a `TypeError`) and a leaf, as `sum_over`: no
    other neighbor dimension of `field` hangs from it (else a `ValueError`). The skip positions
    are skipped as in `sum_over`: a point is a skip position if any position on its path, from
    the root of `axis` down to `axis`, holds `SKIP_VALUE`. The identity of a point with no
    neighbors is the lowest value of the dtype: `-inf` for floats, `np.iinfo(dtype).min` for
    ints. Missing neighbors are not skipped: they hold the value of a boundary read (see
    `Field.__call__`). `axis` is removed from the dimensions of the result.
    """
    if not issubclass(axis, Neighbor):
        raise TypeError(
            f"`max_over` reduces over a neighbor dimension, `Neighbor[R]`, not '{axis.__name__}'."
        )
    return _reduce(
        field,
        axis,
        np.max,
        lambda dtype: -np.inf if dtype.kind == "f" else np.iinfo(dtype).min,
    )


def min_over[T](field: Field[Any, T], /, *, axis: Dimension) -> NumPyArrayField[Any, T]:
    """DSL builtin: the smallest value of `field` over its neighbor dimension `axis`, a
    `Neighbor[R, P]`, e.g. `min_over(e(C2E), axis=Neighbor[C2E])`.

    `axis` must be a `Neighbor` dimension (else a `TypeError`) and a leaf, as `sum_over`: no
    other neighbor dimension of `field` hangs from it (else a `ValueError`). The skip positions
    are skipped as in `sum_over`: a point is a skip position if any position on its path, from
    the root of `axis` down to `axis`, holds `SKIP_VALUE`. The identity of a point with no
    neighbors is the highest value of the dtype: `+inf` for floats, `np.iinfo(dtype).max` for
    ints. Missing neighbors are not skipped: they hold the value of a boundary read (see
    `Field.__call__`). `axis` is removed from the dimensions of the result.
    """
    if not issubclass(axis, Neighbor):
        raise TypeError(
            f"`min_over` reduces over a neighbor dimension, `Neighbor[R]`, not '{axis.__name__}'."
        )
    return _reduce(
        field,
        axis,
        np.min,
        lambda dtype: np.inf if dtype.kind == "f" else np.iinfo(dtype).max,
    )


# The DSL builtins that are reductions over their `axis` keyword: read steps `Reduction(axis)`.
REDUCTIONS: tuple[Callable[..., Any], ...] = (sum_over, max_over, min_over)


def flatten[T](field: Field[Any, T], /, *, axis: Dimension) -> NumPyArrayField[Any, T]:
    """DSL builtin: flatten `field`, with dimensions `(R.source, Neighbor[R], ...)`, to the pairs
    of `R`, `(Pairs[R], ...)`: the values at the neighbors, not at the skip positions, in row-major
    order (see `Pairs`), through the table of `R.to_pairs` in the current mesh, or in the mesh of
    a `NumPyMeshField` (the result is then one of its mesh). Read step: `Flatten(R)`.

    `axis` is a `Neighbor[R]` hanging from `R.source` (else a `TypeError`), and no other
    neighbor dimension of `field` may hang from `R.source` or from `axis` (else a `ValueError`:
    reduce them first). Not at program level (a `TypeError`).
    """
    _not_in_program("`flatten`")
    values = cast(NumPyArrayField[Any, T], field)
    relation = cast("type[Neighbor[Any]]", axis).relation if issubclass(axis, Neighbor) else None
    if relation is None or axis is not neighbor_of(relation):
        raise TypeError(
            f"Flatten the neighbors of a relation, `Neighbor[R]`, not '{axis.__name__}'."
        )
    source, dims = relation.source, list(values.domain)
    others = [d.__name__ for d in dims if d not in (source, axis) and root_of(d) is source]
    if others:
        raise ValueError(f"Sum over {others} before flattening '{axis.__name__}'.")
    mesh = values.mesh if isinstance(values, NumPyMeshField) else None
    with mesh if mesh is not None else contextlib.nullcontext():  # the tables of its mesh
        ranks = _pair_ranks(relation, values.domain)
    valid = ranks >= 0
    data = np.moveaxis(values.data, (dims.index(source), dims.index(axis)), (0, 1))[valid]
    rest = {dim: indices for dim, indices in values.domain.items() if dim not in (source, axis)}
    return _field_on(mesh, {pairs_of(relation): index_set(ranks[valid]), **rest}, data)


def _pair_ranks(relation: type[Relation[Any, Any]], domain: Domain) -> Any:
    """The table of `R.to_pairs` on the sources and neighbors of `domain`: the pair of each
    neighbor, or `SKIP_VALUE`."""
    to_pairs = relation.to_pairs
    key = {
        relation.source: domain[relation.source],
        neighbor_of(to_pairs): domain[neighbor_of(relation)],
    }
    return np.asarray(current_mesh()[to_pairs][key].data)


def unflatten[T](field: Field[Any, T], /, *, axis: Dimension) -> NumPyArrayField[Any, T]:
    """DSL builtin: the inverse of `flatten`, from `(Pairs[R], ...)` to `(R.source,
    Neighbor[R], ...)` with 0 at the skip positions, on the sources whose pairs are all in the
    domain of `field`: the gather `field(R.to_pairs)`, with `Neighbor[R.to_pairs]` renamed
    `Neighbor[R]`. Read step: `Unflatten(R)`. A `TypeError` if `axis` is not a `Pairs[R]`, a
    `ValueError` if a neighbor dimension of `field` hangs from it (reduce it first). A
    `NumPyMeshField` is gathered in its mesh, and the result is one of its mesh. Not at program
    level (a `TypeError`)."""
    _not_in_program("`unflatten`")
    if not issubclass(axis, Pairs):
        raise TypeError(f"Unflatten pairs, `Pairs[R]`, not '{axis.__name__}'.")
    relation = cast("type[Pairs[Any]]", axis).relation
    gathered = cast(NumPyArrayField[Any, T], field)(relation.to_pairs)  # in its mesh, if any
    mesh = gathered.mesh if isinstance(gathered, NumPyMeshField) else None
    return _field_on(mesh, _unflattened(gathered.domain, relation), gathered.data)


def _unflattened[V](
    domain: dict[Dimension, V], relation: type[Relation[Any, Any]]
) -> dict[Dimension, V]:
    """Rename `Neighbor[R.to_pairs]` to `Neighbor[R]` in `domain`, recursively (see
    `_reparented`); a `ValueError` if another neighbor dimension hangs from it, i.e. hung from
    `Pairs[R]` before the gather."""
    old, new = neighbor_of(relation.to_pairs), neighbor_of(relation)
    pairs = pairs_of(relation)
    hanging = [_reparented(dim, old, pairs).__name__ for dim in domain if old in _path(dim)[:-1]]
    if hanging:
        raise ValueError(f"Sum over {hanging} before unflattening '{pairs.__name__}'.")
    return {_reparented(dim, old, new): value for dim, value in domain.items()}


def _path(dim: Dimension) -> list[Dimension]:
    """The dimensions from the root of `dim` down to `dim`, e.g. `[Cell, Neighbor[C2E], ...]`."""
    return [*_path(dim.parent), dim] if issubclass(dim, Neighbor) else [dim]


def neighbor_indices(domain: Domain, axis: Dimension) -> tuple[Any, Any]:
    """Return the neighbor indices at the neighbor dimension `axis`, a `Neighbor[R, P]`, on
    `domain`, and whether they are valid: arrays with one axis per dimension of the path from
    the root of `axis` down to `axis`. A neighbor is valid if no position on that path is a skip
    position (a missing neighbor is valid: only a leaf can be one, see `_forward_step`)."""
    neighbor = cast("type[Neighbor[Any]]", axis)
    relation, parent = neighbor.relation, neighbor.parent
    if issubclass(parent, Neighbor):
        sources, valid = neighbor_indices(domain, parent)
    else:
        sources = np.asarray(domain[parent], dtype=int)  # int even if empty
        valid = np.ones(sources.shape, dtype=bool)
    table = current_mesh()[relation]
    own = table.domain[relation.source]
    rows = np.asarray(table.data)[_positions(own, np.where(valid, sources, own[0]))]
    neighbors = rows[..., np.asarray(domain[axis], dtype=int)]
    return neighbors, valid[..., None] & (neighbors != SKIP_VALUE)


# --- Local meshes -------------------------------------------------------------

_mesh_ids = itertools.count()


def decompose(
    tables: Mapping[type[Relation[Any, Any]], NeighborTable],
    extents: Mapping[Dimension, int],
    regions: Mapping[type[Region[Any]], range | IndexSet],
    owner: Mapping[Dimension, Sequence[int]],
    *,
    support: Sequence[type[Relation[Any, Any]]] = (),
    hops: int = 1,
    derived: Sequence[type[Relation[Any, Any]]] = (),
    halo: Sequence[Mapping[Dimension, Iterable[int]]] | None = None,
    mesh_id: object = None,
) -> list[LocalMesh]:
    """Return the `LocalMesh` of each rank of the partition `owner` of the global mesh `tables`,
    with `extents`, and its `regions` (global indices), e.g. for 2 ranks:

        decompose(tables, {Cell: 6, Edge: 6}, {}, {Cell: [0, 0, 0, 1, 1, 1], Edge: [...]},
                  support=(C2E, E2C), hops=2)

    `owner` gives the rank of each point of each distributed dimension. A rank's local points
    are the ones it owns, then its halo: the points reached from them through the relations of
    `support`, `hops` times (a hint: whether the halo is enough for a read is checked when
    computing, see `TRUNCATED`). Each in increasing global order: the local index of a point is
    its position there, and `global_index` gives its global index. The tables of the relations
    of `tables`, `support` and `derived` are checked (with their constructions), or derived for
    stencils, on the global mesh, then localized: the rows of the local points, in the same
    neighbor order, with their neighbors' local indices, or `TRUNCATED` outside the local points. So
    `derived` names stencils to derive globally, e.g. those with transposes, which a local mesh
    does not derive, or with `paths="unique"`, whose halo rows a local mesh only derives where
    they are computable (see `LocalMesh`). A region holds the local points whose global indices
    are in it. A `ValueError` if `owner` lacks a dimension of these relations or regions.

    `halo[rank]`, if given, adds global points to the local points of `rank`, after the `hops`
    closure (e.g. the footprints of `derive_distribution`). `mesh_id` is the `mesh_id` of the
    local meshes (by default a new `"mesh-<n>"`; `distribute` gives the logical mesh's id). A
    rank may own nothing: its local mesh is then empty (or only a halo).
    """
    the_mesh = Mesh(tables, extents)
    relations = list(dict.fromkeys(r.declared for r in [*tables, *support, *derived]))
    needed = [*(d for r in relations for d in (r.source, r.target)), *(r.dim for r in regions)]
    for dim in dict.fromkeys(needed):
        if dim not in owner:
            raise ValueError(f"`decompose` needs the owner of each point of '{dim.__name__}'.")
    rows = {r: np.asarray(the_mesh[r].data) for r in relations}  # derived globally if stencils
    if mesh_id is None:
        mesh_id = f"mesh-{next(_mesh_ids)}"
    local_meshes = []
    for rank in range(1 + max(max(ranks) for ranks in owner.values())):
        owned = {dim: [i for i, r in enumerate(ranks) if r == rank] for dim, ranks in owner.items()}
        reached = {dim: set(points) for dim, points in owned.items()}
        for _ in range(hops):  # each hop from the points reached before it
            before = {dim: sorted(points) for dim, points in reached.items()}
            for relation in support:
                targets = rows[relation.declared][before[relation.source]]
                reached[relation.target] |= set(targets[targets >= 0].tolist())
        for dim, points in (halo[rank] if halo is not None else {}).items():
            reached[dim] |= {int(point) for point in points}
        index = {dim: (*owned[dim], *sorted(reached[dim] - set(owned[dim]))) for dim in owner}
        local = {dim: {g: i for i, g in enumerate(points)} for dim, points in index.items()}
        local_tables = {
            r: neighbor_table(
                r,
                [
                    [local[r.target].get(t, TRUNCATED) if t >= 0 else t for t in row]
                    for row in rows[r][list(index[r.source])].tolist()
                ],
            )
            for r in relations
        }
        local_regions = {
            the_region: index_set(i for i, g in enumerate(index[the_region.dim]) if g in points)
            for the_region, points in ((r, set(p)) for r, p in regions.items())
        }
        local_meshes.append(
            LocalMesh(
                local_tables,
                extents={dim: len(points) for dim, points in index.items()},
                owned={dim: range(len(points)) for dim, points in owned.items()},
                global_index=index,
                regions=local_regions,
                mesh_id=mesh_id,
                rank=rank,
            )
        )
    return local_meshes


def scatter[T](
    field: Field[Any, T], local_mesh: LocalMesh, with_halo: bool = False
) -> NumPyArrayField[Any, T]:
    """Return the values of the global `field` on the points of `local_mesh` that it owns, or,
    `with_halo`, on all its points (as if exchanged), with local indices, along the dimensions
    of `field` it distributes; the others are unchanged. A `ValueError` if `field` does not
    have them all."""
    values = cast(NumPyArrayField[Any, T], field)
    domain, data = dict(values.domain), values.data
    for axis, dim in enumerate(values.domain):
        if dim not in local_mesh.global_index:
            if issubclass(dim, Location):  # e.g. `Pairs[R]`: its numbering is global
                raise ValueError(
                    f"'{dim.__name__}' is not distributed by the local mesh: it cannot be "
                    "scattered (nor are pairs, in Phase 1)."
                )
            continue
        points = range(local_mesh.extents[dim]) if with_halo else local_mesh.owned[dim]
        global_points = [local_mesh.global_index[dim][i] for i in points]
        values.subdomain({dim: index_set(global_points)})  # raises outside the domain
        data = np.take(data, _positions(values.domain[dim], global_points), axis=axis)
        domain[dim] = points
    return NumPyArrayField(domain, data)


def _scatter_valid[T](field: Field[Any, T], local_mesh: LocalMesh) -> NumPyArrayField[Any, T]:
    """As `scatter`, but on the owned points of `local_mesh` that are in the domain of `field`
    (`owned ∩ ι⁻¹(domain)`, an index set of local indices) along each distributed dimension,
    instead of all the owned points: for inputs defined on part of the mesh (the lowering)."""
    values = cast(NumPyArrayField[Any, T], field)
    domain, data = dict(values.domain), values.data
    for axis, dim in enumerate(values.domain):
        if dim not in local_mesh.global_index:
            if issubclass(dim, Location):  # e.g. `Pairs[R]`: its numbering is global
                raise ValueError(
                    f"'{dim.__name__}' is not distributed by the local mesh: it cannot be "
                    "scattered (nor are pairs, in Phase 1)."
                )
            continue
        own, global_index = set(values.domain[dim]), local_mesh.global_index[dim]
        points = index_set(i for i in local_mesh.owned[dim] if global_index[i] in own)
        global_points = [global_index[i] for i in points]
        data = np.take(data, _positions(values.domain[dim], global_points), axis=axis)
        domain[dim] = points
    return NumPyArrayField(domain, data)


def gather_owned[T](
    fields: Sequence[Field[Any, T]], local_meshes: Sequence[LocalMesh]
) -> NumPyArrayField[Any, T]:
    """The inverse of `scatter`: the global field of the values of `fields`, one per rank, at
    the points of their domains that their rank owns, with global indices, along their one
    distributed dimension (else a `ValueError`); the other dimensions are the same."""
    first = cast(NumPyArrayField[Any, T], fields[0])
    distributed = [dim for dim in first.domain if dim in local_meshes[0].global_index]
    if len(distributed) != 1:
        raise ValueError(f"{format_domain(first.domain)} must have one distributed dimension.")
    (dim,) = distributed
    axis = list(first.domain).index(dim)
    points, parts = [], []
    for field, local_mesh in zip(fields, local_meshes, strict=True):
        values, own = cast(NumPyArrayField[Any, T], field), set(local_mesh.owned[dim])
        owned = [i for i in values.domain[dim] if i in own]
        points += [local_mesh.global_index[dim][i] for i in owned]
        parts.append(np.take(values.data, _positions(values.domain[dim], owned), axis=axis))
    order = np.argsort(points)
    data = np.take(np.concatenate(parts, axis=axis), order, axis=axis)
    return NumPyArrayField({**first.domain, dim: index_set(points)}, data)


# --- Local programs -----------------------------------------------------------
# A local program is a Python function run once per rank, in its local mesh: the ranks are
# threads here, communicating through a shared board between barriers (a stand-in for MPI). Its
# fields are values, valid on their domains: results are new fields, never written into (see
# `NumPyArrayField.__setitem__`; nor `empty`), and the halo of a result on `Owned(Cell)` comes
# from the collective `exchange`. A read beyond the valid points is a containment error, as
# anywhere. Writing into the array of a field, `f.data[...] = ...`, is the escape hatch of plain
# Python code (not in DSL operators: see `DSLRules`), whose validity is not checked.


class LocalProgram[**P, R]:
    """A local program, from `@local_program`: its `function` is run once per rank by
    `run_local` (it is not called directly)."""

    def __init__(self, function: Callable[P, R]) -> None:
        self.function = function
        functools.update_wrapper(cast(Any, self), function)  # its name, not its call


def local_program[**P, R](function: Callable[P, R]) -> LocalProgram[P, R]:
    """Mark `function` as a local program, run on all the ranks by `run_local`. It may call the
    collectives `exchange` and `global_sum`, DSL operators on any backend, and Python code."""
    return LocalProgram(function)


@dataclasses.dataclass(frozen=True)
class _Rank:
    """A rank in a running local program: its `local_mesh`, and the `board`, `barrier` and
    `waits` shared by the ranks for collectives."""

    local_mesh: LocalMesh
    board: list[Any]
    barrier: threading.Barrier
    waits: list[int]  # the barrier waits begun by each rank

    def _wait(self) -> None:
        """Wait at the barrier. A rank released by it, but woken after another one aborted it,
        sees it broken: not an abort if all the ranks had arrived."""
        rank = self.local_mesh.rank
        self.waits[rank] += 1
        try:
            self.barrier.wait()
        except threading.BrokenBarrierError:
            if min(self.waits) < self.waits[rank]:  # not all arrived: a real abort
                raise

    def all_gather(self, value: Any, collective: str) -> list[Any]:
        """Return the `value`s of all the ranks, in rank order, once they have all given theirs;
        a `RuntimeError` if they called different collectives, e.g. one rank `exchange` while
        another one ends its program."""
        try:
            self.board[self.local_mesh.rank] = (collective, value)
            self._wait()
            called, values = zip(*self.board, strict=True)
            self._wait()  # all have read the board before the next collective writes
        except threading.BrokenBarrierError as error:
            raise threading.BrokenBarrierError(
                f"Rank {self.local_mesh.rank}: `{collective}` was aborted: another rank raised, or "
                "did not reach it within the timeout of `run_local`."
            ) from error
        if len(set(called)) != 1:
            raise RuntimeError(f"The ranks called different collectives: {list(called)}.")
        return list(values)


_running: contextvars.ContextVar[_Rank | None] = contextvars.ContextVar("rank", default=None)


def run_local[R](
    program: LocalProgram[..., R],
    local_meshes: Sequence[LocalMesh],
    args_per_rank: Sequence[Sequence[Any]],
    *,
    timeout: float = 10.0,
) -> list[R]:
    """Run `program` on each rank, `program(*args_per_rank[rank])` in
    `with local_meshes[rank]:`, all at once (threads), and return the results in rank order.

    The local meshes must be the ranks `0, 1, ...` (at least one) of the same mesh, with the same
    `mesh_id` (else a `ValueError`: the brand check), owning distinct global points. The end of
    the program is a collective too, so all the ranks call the same collectives, in the same
    order: else a `RuntimeError` on all of them. If a rank raises, the collectives the others
    wait in are aborted and `run_local` raises that error; a rank waiting in a collective for
    more than `timeout` seconds (e.g. while another computes) raises a `BrokenBarrierError`.
    """
    if not isinstance(program, LocalProgram):
        name = getattr(program, "__name__", program)
        raise TypeError(f"'{name}' is not a local program: decorate it with `@local_program`.")
    if not local_meshes:
        raise ValueError("`run_local` needs the local meshes of the ranks: none given.")
    if any(m.mesh_id != local_meshes[0].mesh_id for m in local_meshes):
        ids = sorted({str(m.mesh_id) for m in local_meshes})
        raise ValueError(f"The local meshes are parts of different meshes, {ids}.")
    ranks = [m.rank for m in local_meshes]
    if ranks != list(range(len(local_meshes))) or len(args_per_rank) != len(local_meshes):
        raise ValueError("Give the local meshes of the ranks 0, 1, ..., and the arguments of each.")
    for dim in local_meshes[0].owned:
        owned = [m.global_index[dim][i] for m in local_meshes for i in m.owned.get(dim, ())]
        if len(owned) != len(set(owned)):
            raise ValueError(f"The local meshes own some points of '{dim.__name__}' twice.")
    board: list[Any] = [None] * len(local_meshes)
    barrier = threading.Barrier(len(local_meshes), timeout=timeout)
    results: list[Any] = [None] * len(local_meshes)
    errors: list[BaseException | None] = [None] * len(local_meshes)
    waits = [0] * len(local_meshes)

    def run(rank: int) -> None:
        this = _Rank(local_meshes[rank], board, barrier, waits)
        _running.set(this)  # a thread has its own context
        try:
            with local_meshes[rank]:
                results[rank] = program.function(*args_per_rank[rank])
                this.all_gather(None, "the end of the program")  # no rank is left waiting
        except BaseException as error:  # noqa: BLE001  (raised again by `run_local`)
            errors[rank] = error
            barrier.abort()  # the others raise in their collectives: no rank waits forever

    threads = [threading.Thread(target=run, args=(rank,)) for rank in ranks]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    raised = [error for error in errors if error is not None]
    if raised:  # the first error that is not an abort, else the first abort (a timeout)
        first = [e for e in raised if not isinstance(e, threading.BrokenBarrierError)] or raised
        raise first[0]
    return results


def _rank(collective: str) -> _Rank:
    rank = _running.get()
    if rank is None:
        raise RuntimeError(
            f"`{collective}` is a collective: call it in a local program, run on all the ranks "
            "by `run_local`."
        )
    return rank


def _distributed(field: Field[Any, Any], local_mesh: LocalMesh, collective: str) -> Dimension:
    """The one distributed dimension of `field` in `local_mesh` (else a `ValueError`)."""
    distributed = [dim for dim in field.domain if dim in local_mesh.global_index]
    undistributed = [d for d in field.domain if issubclass(d, Location) and d not in distributed]
    if len(distributed) != 1 or undistributed:  # e.g. `Pairs[R]`: its numbering is global
        raise ValueError(
            f"`{collective}` needs one distributed dimension, and no other `Location` one (nor "
            f"pairs, in Phase 1), not {format_domain(field.domain)}."
        )
    return distributed[0]


def exchange[T](field: Field[Any, T]) -> NumPyArrayField[Any, T]:
    """Collective: return `field` on all the local points of its distributed dimension, the
    halo too, with the values of their owners. `field` must have one distributed dimension, and
    values on the owned points (else a `ValueError`), e.g. the result of an operator on
    `Owned(Cell)`; its other values are not used. Its other dimensions are kept whole: they must
    be the same on all the ranks. A `RuntimeError` outside a running local program. (The
    lowering of programs uses a lenient private path, `_exchange(..., strict=False)`.)"""
    return _exchange(field, _rank("exchange"), strict=True)


def _exchange[T](field: Field[Any, T], rank: _Rank, *, strict: bool) -> NumPyArrayField[Any, T]:
    """`exchange` (`strict`), or its lenient form for the lowering (D9): each rank sends the
    owned points that `field` has (`owned ∩ domain`, no check that they are all there), and
    gets the field on the local points whose owners sent a value (an index set of local
    indices). So values defined on part of the mesh, e.g. on its interior, are exchanged."""
    local_mesh = rank.local_mesh
    dim = _distributed(field, local_mesh, "exchange")
    owned = local_mesh.owned[dim]
    if strict and not _within(owned, field.domain[dim]):
        raise ValueError(
            f"`exchange` needs values on the owned points, {format_domain({dim: owned})}, "
            f"not only on {format_domain(field.domain)}."
        )
    sending = owned if strict else _intersect(owned, field.domain[dim])
    values = cast(NumPyArrayField[Any, T], field)._restricted({dim: sending})
    axis = list(values.domain).index(dim)
    others = {d: i for d, i in values.domain.items() if d is not dim}
    sent = ([local_mesh.global_index[dim][i] for i in sending], np.moveaxis(values.data, axis, 0))
    parts = rank.all_gather((others, sent), "exchange")
    if any(other != others for other, _ in parts):
        raise ValueError(f"`exchange` needs the same {format_domain(others)} on all the ranks.")
    owners = {g: k for k, g in enumerate(g for _, (points, _) in parts for g in points)}
    data = np.concatenate([part for _, (_, part) in parts])
    global_index = local_mesh.global_index[dim]
    if strict:
        local: range | IndexSet = range(local_mesh.extents[dim])
    else:
        local = index_set(i for i, g in enumerate(global_index) if g in owners)
    received = data[[owners[global_index[i]] for i in local]]
    domain = {**values.domain, dim: local}
    return NumPyArrayField(domain, np.moveaxis(received, 0, axis))


def global_sum(field: Field[Any, Any], /, *, domain: Domain) -> Scalar:
    """Collective: the sum of the values of `field` on `domain`, over all the ranks, the same on
    all of them. `domain` must be within the owned points of the distributed dimension of
    `field`, so that each point is counted once (else a `ValueError`), and within its domain.
    Bitwise the same for any decomposition: the values are added one by one as Python scalars
    (float64 for floats), in the order of their global indices, then of the other dimensions of
    the field, in its dimension order. A `RuntimeError` outside a running local program."""
    rank = _rank("global_sum")
    local_mesh = rank.local_mesh
    dim = _distributed(field, local_mesh, "global_sum")
    owned = local_mesh.owned[dim]
    if dim not in domain or not _within(domain[dim], owned):
        raise ValueError(
            f"`global_sum` sums on owned points, within {format_domain({dim: owned})}, not on "
            f"{format_domain(domain)}."
        )
    values = cast(NumPyArrayField[Any, Any], field)[domain]
    rows = np.moveaxis(values.data, list(values.domain).index(dim), 0)
    points = [local_mesh.global_index[dim][i] for i in values.domain[dim]]
    parts = rank.all_gather(list(zip(points, rows.tolist())), "global_sum")
    return _ordered_total(pair for part in parts for pair in part)


# --- Programs -----------------------------------------------------------------
# A program is a Python function with global semantics on a logical mesh (Proposal B), run
# directly here (the reference; the lowering to local programs comes later). Its body runs at
# *program level*: each DSL operator call is handed to the running program (`_ProgramRun.call`,
# from `DSLOperatorBase._call`), which runs the operator in an operator-level window
# (`_operator_level`), where no program is running: nested operator calls and the operator's own
# premaps and reductions are L1 code, and its result is materialized there (lazy backends too).
# At program level, fields are only combined point by point (the guard, `_not_in_program`), and
# summed over a `Location` (`sum_over`, a global reduction).

# The running program, while its body runs (not in the operator-level window).
_program_run: contextvars.ContextVar[_ProgramRun | None] = contextvars.ContextVar(
    "program_run", default=None
)
# Whether a program is running, the operator-level window too: programs do not call programs.
_in_program: contextvars.ContextVar[bool] = contextvars.ContextVar("in_program", default=False)


@contextlib.contextmanager
def _operator_level() -> Iterator[None]:
    """The operator-level window: no program is running in it (`_program_run` and
    `program_mesh` are `None`), so the guard is off; both are restored after it."""
    run_token, mesh_token = _program_run.set(None), program_mesh.set(None)
    try:
        yield
    finally:
        program_mesh.reset(mesh_token)
        _program_run.reset(run_token)


def _not_in_program(what: str) -> None:
    """The guard: a `TypeError` for `what` (e.g. "A premap") at program level. Lowered, a field at
    program level is a rank's local field: premaps, subscripts and neighbor reductions there
    would mix local and global indices, so programs only combine fields point by point."""
    if _program_run.get() is not None:
        raise TypeError(
            f"{what} in a program: compute with fields in field operators (programs call "
            "operators, `sum_over` over a `Location`, and pointwise arithmetic)."
        )


def _op_name(op: Any) -> str:
    """The name of a DSL operator: the name of its definition."""
    return str(getattr(getattr(op, "__gt_definition__", None), "__name__", op))


def _ordered_total(pairs: Iterable[tuple[int, Any]]) -> Scalar:
    """The reproducible sum of the values of `pairs`, `(global index, row of values)`: in the
    order of the global indices, each row in order, added one by one as Python scalars (float64
    for floats) from 0. Bitwise the same for any decomposition (see `global_sum`)."""
    total: Scalar = 0
    for _, row in sorted(pairs, key=lambda pair: pair[0]):
        for value in np.ravel(row).tolist():
            total += value
    return total


@dataclasses.dataclass(frozen=True)
class _ProgramRun:
    """A running program, on its logical `mesh`: what DSL operator calls (`call`) and global sums
    (`total`) do at its program level. `known` holds the program's fields (by identity, kept
    alive): its field arguments and the fields computed in the run. Any other field reaching an
    operator call, a global sum or pointwise arithmetic is a `TypeError` (`check`), in both runs:
    lowered, its global indices would be paired with local ones."""

    program: Program[..., Any]
    mesh: LogicalMesh
    known: dict[int, Any] = dataclasses.field(default_factory=dict, kw_only=True)

    def remember[F](self, value: F) -> F:
        """Record `value`, if a field, as one of the program's fields; return it."""
        if isinstance(value, Field):
            self.known[id(value)] = value
        return value

    def check(self, value: Any, what: str) -> None:
        """A `TypeError` if `value` is, or holds (see `_fields_in`), a field that is not one of
        the program's (an argument or a result computed in the run), except a plain field
        without a `Location` dimension (e.g. on `K`: the same on all ranks)."""
        for field in _fields_in(value):
            if self.known.get(id(field)) is field:
                continue
            if not isinstance(field, NumPyMeshField) and not any(
                issubclass(dim, Location) for dim in field.domain
            ):
                continue
            raise TypeError(
                "Programs get their fields as arguments: the field on "
                f"{format_domain(field.domain)}, {what}, is not an argument of the program or a "
                "result computed in it."
            )

    def check_arguments(self, op_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        """`check` every argument of a call of the operator `op_name`; then a `TypeError` for
        fields in a container argument (a tuple, list, dict or dataclass), even the program's:
        the lowering exchanges only fields given directly (`_separately`)."""
        for value in (*args, *kwargs.values()):
            self.check(value, f"given to '{op_name}'")
        for value in (*args, *kwargs.values()):
            _separately(value, f"'{op_name}', called by a program,")

    def call(
        self,
        op: DSLOperatorBase[..., Any],
        domain: DomainSpec | None,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Field[Any, Any]:
        raise NotImplementedError

    def total(self, field: Field[Any, Any], axis: Dimension, domain: Domain | None) -> Scalar:
        raise NotImplementedError


class _DirectRun(_ProgramRun):
    """The direct run: global semantics on the logical mesh, one process."""

    def call(
        self,
        op: DSLOperatorBase[..., Any],
        domain: DomainSpec | None,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Field[Any, Any]:
        """Run `op` (on the program's backend) in the operator-level window, restricted to
        `domain` if given, and return its result materialized there, as a field of the mesh."""
        config = getattr(op, "__gt_scan_config__", None)
        if config is not None and issubclass(config.axis, Location):
            raise NotImplementedError(
                f"A scan along '{config.axis.__name__}', a distributed dimension, is not "
                "supported in programs."
            )
        self.check_arguments(_op_name(op), args, kwargs)
        operator = self.program._operator(op, self.mesh)
        with _operator_level():
            result = operator(*args, **kwargs)
            if not isinstance(result, NDArrayField):
                raise NotImplementedError(
                    "A program calls operators returning one field (Phase 2)."
                )
            if domain is not None:
                result = restrict(result, domain)
            return self.remember(mesh_field(self.mesh, result.domain, np.asarray(result.data)))

    def total(self, field: Field[Any, Any], axis: Dimension, domain: Domain | None) -> Scalar:
        """`sum_over(field, axis=axis, domain=domain)` at program level: the ordered sum of the
        values of `field` (on `domain`), which has the one dimension `axis` (Phase 2)."""
        self.check(field, f"summed over '{axis.__name__}'")
        if tuple(field.domain) != (axis,):
            raise NotImplementedError(
                "A program sums fields on one dimension, its `axis` (Phase 2), not "
                f"{format_domain(field.domain)}."
            )
        with _operator_level():
            values = cast(NumPyArrayField[Any, Any], field[domain] if domain else field)
            return _ordered_total(zip(values.domain[axis], values.data.tolist()))


@dataclasses.dataclass(frozen=True)
class _LoweredRun(_ProgramRun):
    """The lowered run of a program on one `rank` (D8-D11): its fields are local fields, with
    local indices; it logs its actions in `log` (exchanges, computations, global sums). With
    `exchanges` (more than one rank), each argument an operator reads through a relation is
    replaced by its exchange, unless it is fresh: an exchange result, or exchanged before
    (`fresh`, by the identity of the value, holding it so that ids are not reused). So a helper
    writing into `f.data` after `f` was exchanged is not seen (writes into `.data` are unchecked,
    see "Local programs"): the stale exchange is reused. `fresh` keeps the exchanged values alive
    until the run ends. Errors of the local computations name the rank: their indices are
    local."""

    rank: _Rank
    log: list[str]
    exchanges: bool
    fresh: dict[int, tuple[Field[Any, Any], Field[Any, Any]]]

    def call(
        self,
        op: DSLOperatorBase[..., Any],
        domain: DomainSpec | None,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Field[Any, Any]:
        """Exchange the arguments read through relations (unless fresh), check the local
        coverage of the reads, and compute `op` on the owned points (within `domain`, global,
        if given), materialized in the operator-level window."""
        config = getattr(op, "__gt_scan_config__", None)
        if config is not None and issubclass(config.axis, Location):
            raise NotImplementedError(
                f"A scan along '{config.axis.__name__}', a distributed dimension, is not "
                "supported in programs."
            )
        dsl_op = cast(DSLOperator[..., Any], op)
        extents = dsl_op.__gt_extents__
        op_name = _op_name(op)
        self.check_arguments(op_name, args, kwargs)
        given = {**dict(zip(extents, args)), **kwargs}
        local_mesh = self.rank.local_mesh
        dims = _result_dims(result_annotation(dsl_op.__gt_definition__))
        request = {d: local_mesh.owned[d] for d in dims if d in local_mesh.global_index}
        for name, arg in given.items():
            reads = extents.get(name, [])
            if not isinstance(arg, Field) or not any(_has_gather(read) for read in reads):
                continue
            with _operator_level(), self._local_errors():
                if self.exchanges:
                    given[name] = arg = self._fresh(arg, name, op_name)
                if request:  # the local coverage: a `ValueError` beyond the local support
                    backward_domain(request, [_tolerant(read) for read in reads])
        positional = [given[name] for name in list(extents)[: len(args)]]
        keywords = {name: given[name] for name in kwargs}
        with _operator_level(), self._local_errors():
            result = self.program._operator(op, self.mesh)(*positional, **keywords)
            if not isinstance(result, NDArrayField):
                raise NotImplementedError(
                    "A program calls operators returning one field (Phase 2)."
                )
            distributed = [d for d in result.domain if d in local_mesh.global_index]
            if len(distributed) > 1:
                raise NotImplementedError(
                    "A program calls operators returning fields with one distributed dimension "
                    f"(Phase 2), not {format_domain(result.domain)}."
                )
            wanted: Domain | None = None if domain is None else dict(cast(Domain, domain))
            if distributed:
                (dim,) = distributed
                if wanted is None or dim not in wanted:
                    points = result.domain[dim]
                else:  # the local indices of the requested global points
                    requested = set(wanted[dim])
                    global_index = local_mesh.global_index[dim]
                    points = index_set(i for i, g in enumerate(global_index) if g in requested)
                owned = local_mesh.owned[dim]
                wanted = {**(wanted or {}), dim: _intersect(owned, points)}
            if wanted is not None:
                result = restrict(result, wanted)
            self.log.append(f"compute '{op_name}' on the owned points")
            return self.remember(NumPyArrayField(result.domain, np.asarray(result.data)))

    @contextlib.contextmanager
    def _local_errors(self) -> Iterator[None]:
        """`ValueError`s of this rank's local computations, prefixed with the rank: their
        indices are local."""
        try:
            yield
        except ValueError as error:
            rank = self.rank.local_mesh.rank
            raise ValueError(f"Rank {rank}, local indices: {error}") from error

    def _fresh(self, arg: Field[Any, Any], name: str, op_name: str) -> Field[Any, Any]:
        """The exchange of `arg`: reused if `arg` is fresh, else computed and remembered."""
        known = self.fresh.get(id(arg))
        if known is not None and known[0] is arg:
            return known[1]
        exchanged = _exchange(arg, self.rank, strict=False)
        self.fresh[id(arg)] = (arg, exchanged)
        self.fresh[id(exchanged)] = (exchanged, exchanged)
        self.log.append(f"exchange '{name}' for '{op_name}'")
        return exchanged

    def total(self, field: Field[Any, Any], axis: Dimension, domain: Domain | None) -> Scalar:
        """`sum_over(field, axis=axis, domain=domain)`: the `global_sum` of `field` on the owned
        points (within the global `domain`, if given), the same on all ranks."""
        self.check(field, f"summed over '{axis.__name__}'")
        if tuple(field.domain) != (axis,):
            raise NotImplementedError(
                "A program sums fields on one dimension, its `axis` (Phase 2), not "
                f"{format_domain(field.domain)}."
            )
        local_mesh = self.rank.local_mesh
        owned = local_mesh.owned[axis]
        if domain is None:
            points = _intersect(owned, field.domain[axis])
        else:
            requested = set(domain[axis])
            global_index = local_mesh.global_index[axis]
            points = index_set(i for i in owned if global_index[i] in requested)
        self.log.append(f"global_sum over '{axis.__name__}'")
        with _operator_level(), self._local_errors():
            return global_sum(field, domain={axis: points})


class Program[**P, R]:
    """A program, from `@program(backend=...)`: its `function`, run with global semantics on
    the logical mesh of its field arguments (`MeshField`s, all of one mesh), its DSL operators
    run on `backend` (each built once per mesh signature, see `_operator`).

    The compile key is `(operator, mesh.signature)` (see `LogicalMesh.signature`): the static
    types of the tables and the region layout, never the mesh instance, its `mesh_id`, extents
    or values. Meshes with the same signature share the operators built for them, e.g.
    `compiled` compiles each operator once for all of them.

    Calling it runs the function directly (see "Programs"): `region(R)` is the global domain of
    `R`, `sum_over(f, axis=Cell)` a global sum, and the guard applies at program level. Programs
    get their fields as arguments, in the direct and the lowered run alike (lowered, other
    fields would pair global indices with local ones):
    - each field is a separate argument, of the program and of the operators it calls: a field
      inside a tuple, list, dict or dataclass argument is a `TypeError` (containers without
      fields are fine);
    - a field from a global or a closure (e.g. a geometry field, `AREA = grid.field(...)`),
      given to an operator (also inside a container), summed, combined at program level, or
      returned, is a `TypeError`: pass it as an argument. Plain fields without a `Location`
      dimension (e.g. on `K`) are allowed;
    - it returns a field, a scalar, or a (nested) tuple of them (else a `TypeError`).
    Programs are called from Python, not from programs or local programs (a `RuntimeError`).
    `program.lower(distribution)` gives the program run on the ranks of a `Distribution`, with
    the same results (see `LoweredProgram`).
    """

    def __init__(self, function: Callable[P, R], backend: Backend | None) -> None:
        self.function = function
        self.backend = backend
        # the operators of `backend`, by `(id(op), mesh signature)`: `(op, backend(op))`, the
        # given operator kept with it so that its identity stays unique
        self._operators: dict[Any, Any] = {}
        functools.update_wrapper(cast(Any, self), function)  # its name, not its call

    @property
    def operators(self) -> tuple[DSLOperator[..., Any], ...]:
        """The DSL operators the body calls by name, `op(...)` or `op[domain](...)`, unique, in
        the order of first use in the source. Not found: operators called from plain helper
        functions, or through variables of the body."""
        body = cast(ast.FunctionDef, ast.parse(getsource(self.function)).body[0]).body
        parameters = inspect.signature(self.function).parameters
        namespace = _namespace(self.function)
        calls = sorted(
            (
                node
                for statement in body
                for node in ast.walk(statement)
                if isinstance(node, ast.Call)
            ),
            key=lambda node: (node.lineno, node.col_offset),
        )
        found: dict[int, Any] = {}  # by identity: compiled operators are unhashable
        for call in calls:
            func = call.func.value if isinstance(call.func, ast.Subscript) else call.func
            if isinstance(func, ast.Name) and func.id not in parameters:
                value = namespace.get(func.id)
                if hasattr(value, "__gt_operator_kind__"):
                    found.setdefault(id(value), value)
        return tuple(found.values())

    def _operator(self, op: DSLOperatorBase[..., Any], mesh: LogicalMesh) -> Any:
        """The operator run for `op` on `mesh`: `op` itself without a backend, else
        `backend(op)`, built once per compile key, `(op, mesh.signature)` (by the identity of
        `op`: compiled operators are unhashable), and cached."""
        if self.backend is None:
            return op
        key = (id(op), mesh.signature)
        if key not in self._operators:
            self._operators.setdefault(key, (op, self.backend(cast(Any, op))))
        return self._operators[key][1]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """The direct run (see `Program`)."""
        if _in_program.get() or _running.get() is not None:
            raise RuntimeError(
                "A program is called from Python, not from a program or a local program."
            )
        mesh = _program_mesh_of((*args, *kwargs.values()))
        tokens = (_in_program.set(True), program_mesh.set(mesh))
        run = _DirectRun(self, mesh)
        for value in (*args, *kwargs.values()):
            run.remember(value)  # the program's field arguments
        run_token = _program_run.set(run)
        try:
            with mesh:
                return _returned(run, self.function(*args, **kwargs))
        finally:
            _program_run.reset(run_token)
            program_mesh.reset(tokens[1])
            _in_program.reset(tokens[0])

    def lower(self, distribution: Distribution) -> LoweredProgram[P, R]:
        """The program lowered onto `distribution` (see `LoweredProgram`): a `TypeError` if it is
        not a `Distribution`, a `ValueError` if it does not cover the reads of the operators
        (`check_coverage`, before any execution). The operators are built for the mesh's
        signature here, and compiled if they can be, in the calling thread (not in the ranks)."""
        if not isinstance(distribution, Distribution):
            raise TypeError(
                f"A program is lowered onto a `Distribution`, e.g. `distribute(...)`, not "
                f"{distribution!r}."
            )
        check_coverage(self.operators, distribution)
        for op in self.operators:
            key = (id(op), distribution.mesh.signature)
            if self.backend is not None and key not in self._operators:
                operator = self._operator(cast(DSLOperatorBase[..., Any], op), distribution.mesh)
                if hasattr(operator, "compile"):
                    operator.compile()
        return LoweredProgram(self, distribution)


class LoweredProgram[**P, R]:
    """A program lowered onto a `distribution`, from `Program.lower`: calling it runs the program
    on every rank of the distribution (`run_local`, ranks as threads), each on its local fields,
    and returns the results of the direct run (the simulation condition): fields gathered from
    the owned points and branded, scalars of rank 0 (the same on all ranks).

    Per rank (see `_LoweredRun`): the field inputs are given on the owned points they have
    (`_scatter_valid`); before each operator call, the arguments read through a relation are
    exchanged unless fresh; results are computed on the owned points; `sum_over` over a
    `Location` is a `global_sum`. `log` holds rank 0's actions in the last call.
    """

    def __init__(self, program: Program[P, R], distribution: Distribution) -> None:
        self.program = program
        self.distribution = distribution
        self.log: list[str] = []
        self.local_program: LocalProgram[..., Any] = LocalProgram(self._on_rank)

    def _on_rank(self, logs: dict[int, list[str]], *args: Any) -> Any:
        """The local program: the program's function on this rank, as a `_LoweredRun`; its log
        goes to `logs`, a dict of this call (calls from several threads do not mix)."""
        local_mesh = cast(LocalMesh, current_mesh())
        rank = _running.get()
        assert rank is not None  # in `run_local`
        log: list[str] = []
        logs[local_mesh.rank] = log
        mesh = self.distribution.mesh
        run = _LoweredRun(
            self.program, mesh, rank, log, len(self.distribution.local_meshes) > 1, {}
        )
        for value in args:
            run.remember(value)  # the program's field arguments, local
        tokens = (_in_program.set(True), program_mesh.set(mesh), _program_run.set(run))
        try:
            return _returned(run, cast(Callable[..., Any], self.program.function)(*args))
        finally:
            _program_run.reset(tokens[2])
            program_mesh.reset(tokens[1])
            _in_program.reset(tokens[0])

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Run the program on all the ranks and gather its results (see `LoweredProgram`)."""
        if _in_program.get() or _running.get() is not None:
            raise RuntimeError(
                "A program is called from Python, not from a program or a local program."
            )
        mesh = _program_mesh_of((*args, *kwargs.values()))
        if mesh.mesh_id != self.distribution.mesh.mesh_id:
            raise ValueError(
                f"The fields are on '{mesh.mesh_id}', the distribution is of "
                f"'{self.distribution.mesh.mesh_id}'."
            )
        names = list(inspect.signature(self.program.function).parameters)
        values = [*args, *(kwargs[name] for name in names[len(args) :])]
        local_meshes = self.distribution.local_meshes
        logs: dict[int, list[str]] = {}  # per call
        args_per_rank = [
            [logs, *(_scatter_valid(v, lm) if isinstance(v, Field) else v for v in values)]
            for lm in local_meshes
        ]
        results = run_local(self.local_program, local_meshes, args_per_rank)
        self.log = logs[0]
        first = results[0]
        if isinstance(first, tuple):
            parts = [[r[i] for r in results] for i in range(len(first))]
            gathered = [self._gathered(part, mesh) for part in parts]
            # the type of the direct result: a namedtuple keeps its type (`_make`)
            make = getattr(type(first), "_make", tuple)
            return cast(R, make(gathered))
        return cast(R, self._gathered(results, mesh))

    def _gathered(self, per_rank: list[Any], mesh: LogicalMesh) -> Any:
        """One result of the ranks: a field with a distributed dimension gathered from their
        owned points, a field without one rank 0's, both branded; anything else rank 0's."""
        first = per_rank[0]
        if not isinstance(first, NDArrayField):
            return first
        local_meshes = self.distribution.local_meshes
        if any(dim in local_meshes[0].global_index for dim in first.domain):
            gathered = gather_owned(per_rank, local_meshes)
            return mesh_field(mesh, gathered.domain, gathered.data)
        return mesh_field(mesh, first.domain, np.asarray(first.data))


def _program_mesh_of(values: Iterable[Any]) -> LogicalMesh:
    """The logical mesh of the field arguments of a program: a `TypeError` if one is not a
    `MeshField` or there are none, a `ValueError` if they are on different logical meshes. A
    field inside another argument (a tuple, list, dict or dataclass) is a `TypeError` too:
    fields are given as separate arguments (`_separately`)."""
    values = list(values)
    for value in values:
        _separately(value, "A program")
    fields = [value for value in values if isinstance(value, Field)]
    if not fields or not all(isinstance(f, NumPyMeshField) for f in fields):
        given = "no fields" if not fields else "a field without a mesh"
        raise TypeError(
            "A program finds its logical mesh from its field arguments: give it fields of a "
            f"mesh, e.g. `grid.field(...)`, not {given}."
        )
    meshes = {f.mesh.mesh_id: f.mesh for f in cast(list[NumPyMeshField[Any, Any]], fields)}
    if len(meshes) > 1:
        raise ValueError(
            "The fields of a program are on one logical mesh (Phase 2), not on "
            f"{sorted(str(mesh_id) for mesh_id in meshes)}."
        )
    return next(iter(meshes.values()))


def _fields_in(value: Any) -> Iterator[Field[Any, Any]]:
    """Every field reachable in `value`: `value` itself if it is a field, else through tuples,
    lists, dicts (their values) and dataclass instances (their fields), recursively."""
    if isinstance(value, Field):
        yield value
    elif isinstance(value, (tuple, list)):
        for element in value:
            yield from _fields_in(element)
    elif isinstance(value, dict):
        for element in value.values():
            yield from _fields_in(element)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for member in dataclasses.fields(value):
            yield from _fields_in(getattr(value, member.name))


def _separately(value: Any, who: str) -> None:
    """A `TypeError` if `value` is not a field but holds one (see `_fields_in`): in programs,
    fields are given as separate arguments (the lowering scatters and exchanges only those).
    Containers without fields are fine."""
    if not isinstance(value, Field) and next(_fields_in(value), None) is not None:
        raise TypeError(
            f"{who} gets its fields as separate arguments: pass fields as separate arguments, "
            f"not in a {type(value).__name__}."
        )


def _returned[V](run: _ProgramRun, result: V) -> V:
    """Check a program's `result`, in both runs: a field, a scalar, or a (nested) tuple of
    them (else a `TypeError`), each field one of the program's (see `_ProgramRun.check`)."""
    if isinstance(result, tuple):
        for element in result:
            _returned(run, element)
    elif isinstance(result, Field):
        run.check(result, "returned")
    elif not isinstance(result, (int, float, complex, np.generic)):
        raise TypeError(
            f"A program returns fields, scalars or tuples of them, not a {type(result).__name__}."
        )
    return result


def program[**P, R](*, backend: Backend | None = None) -> Callable[[Callable[P, R]], Program[P, R]]:
    """Check that the decorated function is a valid program definition (`PROGRAM_RULES`), and
    wrap it into a `Program` whose operators run on `backend` (embedded by default)."""

    def decorator(function: Callable[P, R]) -> Program[P, R]:
        reason = PROGRAM_RULES.violation(function) if is_inspectable(function) else "no source"
        if reason is not None:
            raise TypeError(f"'{function.__name__}' is not a valid program definition: {reason}.")
        return Program(function, backend)

    return decorator


# --- Distributions ------------------------------------------------------------
# How a logical mesh is split into ranks (D12): given, `distribute` (a halo of `hops` over some
# relations, as `decompose`), or derived, `derive_distribution` (each rank's halo is exactly the
# footprint of its owned points through the reads of some programs' operators). Either way the
# local meshes keep the logical mesh's `mesh_id`. `check_coverage` checks that a distribution
# covers the reads of some operators: without it, an insufficient halo would silently shrink
# results.


def _has_gather(read: Read) -> bool:
    """Whether some step of `read` is a gather (a relation or a `Boundary`)."""
    return any(_is_gather(step) for step in read)


def _tolerant(read: Read) -> Read:
    """`read` with each plain step through a partial relation read with a boundary value,
    `Boundary(relation)`: its footprint holds the neighbors that are present, without failing on
    the missing ones."""
    return tuple(Boundary(step) if is_relation(step) and step.partial else step for step in read)


def footprints(
    operators: Sequence[DSLOperator[..., Any]],
    mesh: Mesh,
    owned: Sequence[Mapping[Dimension, IndexSet | range]],
) -> list[dict[Dimension, set[int]]]:
    """Per rank, the global points of each dimension of `owned[rank]` (the rank's owned points)
    that the reads of `operators` need to compute their results on the owned points: the owned
    points, plus, for each field operator (scans are skipped) with `Field` results (one, or a
    tuple of them), each read with a gather step of each argument, propagated back step by step
    (`_backward_step`, with `_tolerant` reads) from the owned points of the results' dimensions,
    on `mesh`. Every intermediate domain is kept too: the points a chain such as `t(L2R)(L2R)`
    or `t(E2C)(C2E)` passes through must be local, since their table rows lead to the next."""
    result: list[dict[Dimension, set[int]]] = []
    for points in owned:
        footprint = {dim: {int(i) for i in indices} for dim, indices in points.items()}
        for op in operators:
            if op.__gt_operator_kind__ is not DSLOperatorKind.FIELD_OPERATOR:
                continue
            dims = _result_dims(result_annotation(op.__gt_definition__))
            request = {dim: points[dim] for dim in dims if dim in points}
            if not request:
                continue
            for reads in op.__gt_extents__.values():
                for read in reads:
                    if not _has_gather(read):
                        continue
                    domain: Domain = dict(request)
                    with mesh:
                        for step in reversed(_tolerant(read)):  # the intermediate domains too
                            domain = _backward_step(domain, step)
                            for dim in footprint:
                                footprint[dim] |= {int(i) for i in domain.get(dim, ())}
        result.append(footprint)
    return result


def _result_dims(annotation: Any) -> list[Dimension]:
    """The dimensions of the `Field`s of a result annotation: of a `Field`, or of every `Field`
    of a (nested) tuple, in order, each once; none for scalars."""
    annotation = _resolve(annotation)
    if get_origin(annotation) is Field:
        return list(dims_of(annotation))
    if get_origin(annotation) is tuple:
        found = [dim for element in get_args(annotation) for dim in _result_dims(element)]
        return list(dict.fromkeys(found))
    return []


def distribute(
    mesh: LogicalMesh,
    owner: Mapping[Dimension, Sequence[int]],
    *,
    support: Sequence[type[Relation[Any, Any]]] = (),
    hops: int = 1,
    derived: Sequence[type[Relation[Any, Any]]] = (),
) -> Distribution:
    """The given mode: the `Distribution` of `mesh` for the partition `owner`, with a halo of
    `hops` over the relations of `support` (see `decompose`), the local meshes keeping the
    `mesh_id` of `mesh`."""
    local_meshes = decompose(
        dict(mesh),
        mesh.extents,
        mesh.regions,
        owner,
        support=support,
        hops=hops,
        derived=derived,
        mesh_id=mesh.mesh_id,
    )
    return Distribution(mesh, tuple(local_meshes))


def _operators_of(programs: Sequence[Program[..., Any]]) -> list[DSLOperator[..., Any]]:
    """The operators of `programs` (see `Program.operators`), unique by identity, in order."""
    found = {id(op): op for p in programs for op in p.operators}
    return list(found.values())


def derive_distribution(
    mesh: LogicalMesh,
    owner: Mapping[Dimension, Sequence[int]],
    programs: Sequence[Program[..., Any]],
) -> Distribution:
    """The derived mode: the `Distribution` of `mesh` for the partition `owner` whose halo, per
    rank, is exactly the footprint of its owned points through the reads of the operators of
    `programs` (`footprints`; operators called from helpers are not found, see
    `Program.operators`). The relations with a `stencil` that the reads gather through are
    derived on the global mesh, so the local meshes get their exact tables."""
    dims = [d for r in mesh for d in (r.source, r.target)] + [r.dim for r in mesh.regions]
    for dim in dict.fromkeys(dims):
        if dim not in owner:
            raise ValueError(
                f"`derive_distribution` needs the owner of each point of '{dim.__name__}'."
            )
    operators = _operators_of(programs)
    ranks = range(1 + max(max(r) for r in owner.values()))
    owned = [
        {dim: tuple(i for i, r in enumerate(by) if r == rank) for dim, by in owner.items()}
        for rank in ranks
    ]
    needed = footprints(operators, mesh, owned)
    halo = [
        {dim: sorted(points - set(own[dim])) for dim, points in footprint.items()}
        for footprint, own in zip(needed, owned, strict=True)
    ]
    stencils = {
        _gathered(step).declared: None
        for op in operators
        for reads in op.__gt_extents__.values()
        for read in reads
        for step in read
        if _is_gather(step) and _gathered(step).declared.stencil is not None
    }
    local_meshes = decompose(
        dict(mesh),
        mesh.extents,
        mesh.regions,
        owner,
        hops=0,
        halo=halo,
        derived=list(stencils),
        mesh_id=mesh.mesh_id,
    )
    return Distribution(mesh, tuple(local_meshes))


def check_coverage(operators: Sequence[DSLOperator[..., Any]], distribution: Distribution) -> None:
    """Raise a `ValueError` if a rank of `distribution` lacks points that the reads of one of
    `operators` need for its owned points (their footprint, see `footprints`): the halo is too
    small, e.g. for two hops on a distribution with `hops=1`."""
    local_meshes = distribution.local_meshes
    owned = [
        {dim: tuple(lm.global_index[dim][i] for i in points) for dim, points in lm.owned.items()}
        for lm in local_meshes
    ]
    for op in operators:
        for rank, footprint in enumerate(footprints([op], distribution.mesh, owned)):
            local = local_meshes[rank].global_index
            for dim, points in footprint.items():
                missing = points - set(local.get(dim, ()))
                if missing:
                    name = getattr(op.__gt_definition__, "__name__", op)
                    raise ValueError(
                        f"The distribution of '{distribution.mesh.mesh_id}' does not cover the "
                        f"reads of '{name}': rank {rank} needs "
                        f"{format_domain({dim: index_set(missing)})} (distribute with more "
                        "`hops`, or derive the distribution)."
                    )


# --- Allocation ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class NumPyFieldAllocator:
    """An `NDArrayFieldAllocator` creating `NumPyArrayField`s with numpy (`device` is ignored).

    The values are laid out in memory in `order`: "C", row-major, where the last dimension
    varies fastest (numpy's default), or "F", column-major, where the first one does.
    """

    order: Literal["C", "F"] = "C"

    def empty(
        self, domain: Domain, *, dtype: DType = None, device: Device = None
    ) -> NumPyArrayField:
        """A field with unspecified values; a `RuntimeError` in a running local program, where
        a field's domain is where its values are valid (see `run_local`)."""
        if _running.get() is not None:
            raise RuntimeError(
                "No `empty` fields in a local program: their values would not be valid on their "
                f"domain, {format_domain(domain)}. Use `zeros` or `full`."
            )
        return NumPyArrayField(domain, np.empty(_shape(domain), dtype=dtype, order=self.order))

    def empty_like(
        self, x: NDArrayField, /, *, dtype: DType = None, device: Device = None
    ) -> NumPyArrayField:
        return self.empty(x.domain, dtype=dtype or x.data.dtype)  # as `empty`

    def zeros(
        self, domain: Domain, *, dtype: DType = None, device: Device = None
    ) -> NumPyArrayField:
        return NumPyArrayField(domain, np.zeros(_shape(domain), dtype=dtype, order=self.order))

    def zeros_like(
        self, x: NDArrayField, /, *, dtype: DType = None, device: Device = None
    ) -> NumPyArrayField:
        return self.zeros(x.domain, dtype=dtype or x.data.dtype)

    def ones(
        self, domain: Domain, *, dtype: DType = None, device: Device = None
    ) -> NumPyArrayField:
        return NumPyArrayField(domain, np.ones(_shape(domain), dtype=dtype, order=self.order))

    def ones_like(
        self, x: NDArrayField, /, *, dtype: DType = None, device: Device = None
    ) -> NumPyArrayField:
        return self.ones(x.domain, dtype=dtype or x.data.dtype)

    def full(
        self, domain: Domain, fill_value: Scalar, *, dtype: DType = None, device: Device = None
    ) -> NumPyArrayField:
        values = np.full(_shape(domain), fill_value, dtype=dtype, order=self.order)
        return NumPyArrayField(domain, values)

    def full_like(
        self, x: NDArrayField, /, fill_value: Scalar, *, dtype: DType = None, device: Device = None
    ) -> NumPyArrayField:
        return self.full(x.domain, fill_value, dtype=dtype or x.data.dtype)


def _shape(domain: Domain) -> tuple[int, ...]:
    return tuple(len(r) for r in domain.values())


def store[T](result: T, allocator: NDArrayFieldAllocator) -> T:
    """Copy every field of `result`, a field or a (nested) tuple of fields, into a new field
    allocated by `allocator`, e.g. with another memory layout."""
    if isinstance(result, tuple):
        return cast(T, tuple(store(r, allocator) for r in result))
    stored = allocator.zeros_like(cast(NDArrayField, result))  # not `empty`: in local programs too
    _write(cast(NumPyArrayField, stored), ..., cast(NDArrayField, result))
    return cast(T, stored)


numpy_field_allocator = NumPyFieldAllocator()
column_major_field_allocator = NumPyFieldAllocator(order="F")


# --- DSL operators ------------------------------------------------------------

# What a lazy backend produces: computes the values of the operator on `domain`.
type Computation[**P] = Callable[Concatenate[Domain, P], Field]


class DSLOperatorBase[**P, R]:
    """Base of the DSL operators below: `op[domain](*args)` and `op(*args, domain=domain)`
    restrict the result of `op(*args)`, computed by `_compute`, to `domain`.

    Called with `NumPyMeshField`s, an operator runs in their mesh (without a `with` block) and
    its result is a field of that mesh (every field of a tuple), except a `DeferredField` (lazy
    backends: branding would compute it); mesh fields of different logical meshes are a
    `ValueError`, before computing.
    """

    def _compute(self, *args: P.args, **kwargs: P.kwargs) -> R:
        raise NotImplementedError

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        domain = cast(DomainSpec | None, kwargs.pop("domain", None))  # shorthand for `self[domain]`
        return self._call(domain, args, kwargs)

    def __getitem__(self, domain: DomainSpec) -> Callable[P, R]:
        def restricted(*args: P.args, **kwargs: P.kwargs) -> R:
            return self._call(domain, args, kwargs)

        return restricted

    def _call(self, domain: DomainSpec | None, args: tuple[Any, ...], kwargs: dict[str, Any]) -> R:
        """Compute `self(*args, **kwargs)` restricted to `domain`, if given: in the logical mesh
        of the mesh field arguments, if any (the brand), with the result branded. In a running
        program, the call is handed to it (see `Program`)."""
        run = _program_run.get()
        if run is not None:
            return cast(R, run.call(self, domain, args, kwargs))
        mesh = _mesh_of((*args, *kwargs.values()))
        with mesh if mesh is not None and current_mesh() is not mesh else contextlib.nullcontext():
            result = cast(Callable[..., R], self._compute)(*args, **kwargs)
        if mesh is not None:
            result = _branded(result, mesh)  # tuples too, element by element
        return result if domain is None else restrict(result, domain)


class FieldOperatorWrapper[**P, R](DSLOperatorBase[P, R]):
    """The `FieldOperator` returned by `@field_operator()`: a wrapper of its definition, which it
    runs in Python when called (embedded execution), computing all the values with forward
    domains. A requested `domain` only restricts the result."""

    __gt_operator_kind__: ClassVar[Literal[DSLOperatorKind.FIELD_OPERATOR]] = (
        DSLOperatorKind.FIELD_OPERATOR
    )

    def __init__(self, definition: FieldOperatorDefinition[P, R]) -> None:
        self.__gt_definition__ = definition
        functools.update_wrapper(self, definition)

    @property
    def __gt_extents__(self) -> Extents:
        return read_extents(self.__gt_definition__)

    def _compute(self, *args: P.args, **kwargs: P.kwargs) -> R:
        result = self.__gt_definition__(*args, **kwargs)
        return as_annotated(result, result_annotation(self.__gt_definition__))


class ScanOperatorWrapper(DSLOperatorBase[..., NumPyArrayField]):
    """The `ScanOperator` returned by `@scan_operator(...)`: a wrapper of its definition, which it
    runs in Python level by level when called (embedded execution).

    The definition is applied to all the columns at once, with numpy arrays as the values on
    one level (simplified). The domain of the result is computed forward.
    """

    __gt_operator_kind__: ClassVar[Literal[DSLOperatorKind.SCAN_OPERATOR]] = (
        DSLOperatorKind.SCAN_OPERATOR
    )

    def __init__(self, definition: ScanOperatorDefinition[..., Any], config: ScanConfig) -> None:
        self.__gt_definition__ = definition
        self.__gt_scan_config__ = config
        functools.update_wrapper(self, definition)

    @property
    def __gt_extents__(self) -> Extents:
        return scan_extents(self.__gt_definition__, self.__gt_scan_config__.axis)

    def _compute(self, *fields: Any, **kwargs: Any) -> NumPyArrayField:
        return scan(self.__gt_definition__, self.__gt_scan_config__, fields)


def field_operator[**P, R]() -> Callable[[Callable[P, R]], FieldOperatorWrapper[P, R]]:
    """Check that the decorated function is a valid field operator definition, and wrap it into
    a `FieldOperator`."""

    def decorator(function: Callable[P, R]) -> FieldOperatorWrapper[P, R]:
        if not is_field_operator_definition(function):
            reason = FIELD_OPERATOR_RULES.violation(function) if is_inspectable(function) else None
            raise TypeError(
                f"'{function.__name__}' is not a valid field operator definition"
                + (f": {reason}." if reason else ".")
            )
        return FieldOperatorWrapper(function)

    return decorator


def scan_operator(
    *, axis: Dimension, forward: bool = True, init: Scalar = 0.0
) -> Callable[[Callable[..., Any]], ScanOperatorWrapper]:
    """Check that the decorated function is a valid scan operator definition, and wrap it into
    a `ScanOperator` scanning as configured.

    The definition takes the carry and the values on one level, and returns the new carry,
    which is also the value of the result on that level.
    """

    def decorator(function: Callable[..., Any]) -> ScanOperatorWrapper:
        if not is_scan_operator_definition(function):
            reason = SCAN_OPERATOR_RULES.violation(function) if is_inspectable(function) else None
            raise TypeError(
                f"'{function.__name__}' is not a valid scan operator definition"
                + (f": {reason}." if reason else ".")
            )
        return ScanOperatorWrapper(function, ScanConfig(axis, forward, init))

    return decorator


def scan_extents(definition: ScanOperatorDefinition[..., Any], axis: Dimension) -> Extents:
    """The extents of a scan operator along `axis`: each argument but the carry is read with
    `(ScanAlong(axis),)`, since a level needs all the previous ones (simplified: no offsets)."""
    _, *names = inspect.signature(definition).parameters
    return {name: [(ScanAlong(axis),)] for name in names}


def scan(
    level: Callable[..., Any], config: ScanConfig, fields: tuple[NDArrayField, ...]
) -> NumPyArrayField:
    """Compute `level(carry, *values)` level by level along `config.axis`, for all the columns
    at once; the domain of the result is computed forward. The fields are broadcast, as in
    arithmetic (see `Field`): they may have their dimensions in any order, or fewer of them."""
    domain = functools.reduce(intersect, (f.domain for f in fields))
    dims, shape = list(domain), _shape(domain)
    axis = dims.index(config.axis)
    levels = [
        np.moveaxis(np.broadcast_to(_aligned(f[domain].data, list(f.domain), dims), shape), axis, 0)
        for f in fields
    ]  # broadcast, in the order of `domain`
    values = np.empty_like(levels[0])
    carry = np.full(values.shape[1:], config.init)
    for k in range(len(values)) if config.forward else reversed(range(len(values))):
        carry = values[k] = level(carry, *(lv[k] for lv in levels))
    return NumPyArrayField(domain, np.moveaxis(values, 0, axis))


class EagerFieldOperator[**P, R](FieldOperatorWrapper[P, R]):
    """A `FieldOperator` that computes all the values of its result as soon as it is called."""

    def _compute(self, *args: P.args, **kwargs: P.kwargs) -> R:
        result = super()._compute(*args, **kwargs)
        print(f"[eager] computed {result}")
        return result


class EagerScanOperator(ScanOperatorWrapper):
    """A `ScanOperator` computing all the values of its result when called, level by level."""

    def _compute(self, *fields: Any, **kwargs: Any) -> NumPyArrayField:
        result = super()._compute(*fields)
        print(f"[scan] computed {result}")
        return result


class AllocatedFieldOperator[**P, R](FieldOperatorWrapper[P, R]):
    """A `FieldOperator` computing its result when called, in fields allocated by `allocator`
    (simplified: the values are computed first, then copied into the allocated fields)."""

    def __init__(
        self, definition: FieldOperatorDefinition[P, R], allocator: NDArrayFieldAllocator
    ) -> None:
        super().__init__(definition)
        self.allocator = allocator

    def _compute(self, *args: P.args, **kwargs: P.kwargs) -> R:
        result = store(super()._compute(*args, **kwargs), self.allocator)
        print(f"[allocated] computed {result}")
        return result


class AllocatedScanOperator(ScanOperatorWrapper):
    """A `ScanOperator` computing its result when called, in a field allocated by `allocator`
    (simplified: the values are computed first, then copied into the allocated field)."""

    def __init__(
        self,
        definition: ScanOperatorDefinition[..., Any],
        config: ScanConfig,
        allocator: NDArrayFieldAllocator,
    ) -> None:
        super().__init__(definition, config)
        self.allocator = allocator

    def _compute(self, *fields: Any, **kwargs: Any) -> NumPyArrayField:
        result = cast(NumPyArrayField, store(super()._compute(*fields), self.allocator))
        print(f"[allocated] computed {result}")
        return result


@dataclasses.dataclass(frozen=True)
class LazyFieldOperator[**P, R](DSLOperatorBase[P, R]):
    """A `FieldOperator` whose result values are only computed when first used.

    A requested `domain` narrows the `DeferredField` result, like `[...]`, before any value is
    computed.
    """

    __gt_operator_kind__: ClassVar[Literal[DSLOperatorKind.FIELD_OPERATOR]] = (
        DSLOperatorKind.FIELD_OPERATOR
    )
    __gt_definition__: FieldOperatorDefinition[P, R]
    computation: Computation[P]

    @property
    def __gt_extents__(self) -> Extents:
        return read_extents(self.__gt_definition__)

    def _compute(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Compute the default domain of the result forward; no values are computed yet.

        They are computed later in the mesh of this call.
        """
        names = inspect.signature(self.__gt_definition__).parameters
        extents = self.__gt_extents__
        given = [*zip(names, args), *kwargs.items()]  # positional and keyword arguments
        domains = [forward_domain(a.domain, extents[n]) for n, a in given if isinstance(a, Field)]
        domain = _result_domain(self.__gt_definition__, functools.reduce(intersect, domains, {}))
        the_mesh = current_mesh()

        def compute(domain: Domain) -> Field:
            with the_mesh:
                return self.computation(domain, *args, **kwargs)

        return cast(R, DeferredField(compute, domain))  # happy path: a single field


# A value in the symbolic evaluation of `read_extents`: the reads it depends on, as
# `(argument, read, dims)`, with `dims` the dimensions after the read (a dict as an ordered
# set), or a tuple of such values.
type _Reads = dict[tuple[str, Read, tuple[Dimension, ...]], None]
type _Value = _Reads | tuple[_Value, ...]


def _namespace(function: Callable[..., Any]) -> dict[str, Any]:
    """The values the names in `function` refer to: its globals and its closure."""
    return {**function.__globals__, **inspect.getclosurevars(function).nonlocals}


def read_extents(definition: FieldOperatorDefinition[..., Any]) -> Extents:
    """Find how each argument is read in the source of `definition` (see `Extents`).

    A small symbolic evaluation of the definition, whose values are the reads of the arguments
    they depend on, with their dimensions (from the annotations of the fields). Reads flow
    through local variables (tuple unpacking, the branches of an `if` merged), premaps,
    reductions (`REDUCTIONS`), `flatten` and `unflatten`, calls to other DSL operators
    (composed with their extents), arithmetic and other calls (unions). Steps only apply to
    some reads (see `_applies`).
    Premaps and axes are evaluated in the globals and closure of the definition (`_namespace`):
    a local variable there raises a `NotImplementedError`, e.g. a boundary premap held in a
    local variable, `b = C2C.boundary(v); c(b)` (write `c(C2C.boundary(v))`).

    The extents over-approximate: both branches of an `if` or a conditional expression, and
    all the elements of an unpacked tuple result, contribute all their reads. So the default
    domains of the lazy backends may be smaller than those of the others: a requested domain
    that `eager` accepts may raise on `lazy`. The values are always right.
    """
    function = call_operator(definition)
    namespace = _namespace(function)
    hints = {name: _resolve(hint) for name, hint in get_type_hints(function).items()}
    names = inspect.signature(definition).parameters
    env: dict[str, _Value] = {
        name: {(name, (), dims_of(hints[name])): None} if get_origin(hints[name]) is Field else {}
        for name in names
    }
    body = cast(ast.FunctionDef, ast.parse(getsource(definition)).body[0]).body
    extents: Extents = {name: [] for name in names}
    for name, read, _ in _execute(body, env, namespace):
        extents[name].append(read)
    return extents


def _execute(body: list[ast.stmt], env: dict[str, _Value], namespace: dict[str, Any]) -> _Reads:
    """Run the statements `body` symbolically, assigning the local variables in `env`; return the
    reads of the returned values."""
    returned: _Reads = {}
    for statement in body:
        if isinstance(statement, (ast.Assign, ast.AnnAssign)) and statement.value is not None:
            value = _evaluate(statement.value, env, namespace)
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            for target in targets:
                _assign(target, value, env)
        elif isinstance(statement, ast.Return) and statement.value is not None:
            returned |= _flat(_evaluate(statement.value, env, namespace))
        elif isinstance(statement, ast.If):  # both branches, merged
            then, orelse = dict(env), dict(env)
            returned |= _execute(statement.body, then, namespace)
            returned |= _execute(statement.orelse, orelse, namespace)
            for name in dict.fromkeys([*then, *orelse]):
                env[name] = _merge(then.get(name, {}), orelse.get(name, {}))
    return returned


def _assign(target: ast.expr, value: _Value, env: dict[str, _Value]) -> None:
    """Assign `value` to the assignment target `target`, a name or a tuple of them (unpacking)."""
    if isinstance(target, ast.Tuple):
        values = value if isinstance(value, tuple) else (value,) * len(target.elts)
        for element, element_value in zip(target.elts, values, strict=True):
            _assign(element, element_value, env)
    elif isinstance(target, ast.Name):
        env[target.id] = value


def _flat(value: _Value) -> _Reads:
    """Return the reads of `value`, of all its elements if it is a tuple."""
    if isinstance(value, tuple):
        return functools.reduce(operator.or_, map(_flat, value), {})
    return value


def _merge(a: _Value, b: _Value) -> _Value:
    """Return the union of two values, element by element for tuples of the same length."""
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b):
        return tuple(_merge(x, y) for x, y in zip(a, b))
    return _flat(a) | _flat(b)


def _evaluate(node: ast.expr, env: dict[str, _Value], namespace: dict[str, Any]) -> _Value:
    """Return the reads of the value of the expression `node` (see `read_extents`)."""
    if isinstance(node, ast.Name):
        return env.get(node.id, {})  # a global or a constant reads no argument
    if isinstance(node, ast.Tuple):
        return tuple(_evaluate(element, env, namespace) for element in node.elts)
    if isinstance(node, ast.Call):
        return _call(node, env, namespace)
    children = ast.iter_child_nodes(node)  # arithmetic, attributes, ...: the union
    return _flat(tuple(_evaluate(c, env, namespace) for c in children if isinstance(c, ast.expr)))


def _call(node: ast.Call, env: dict[str, _Value], namespace: dict[str, Any]) -> _Reads:
    """Return the reads of a call: a premap, a reduction, a DSL operator or another function."""

    def resolve(expression: ast.expr) -> Any:  # e.g. `I + 1` -> `(I, 1)`
        text = ast.unparse(expression)
        names = {n.id for n in ast.walk(expression) if isinstance(n, ast.Name)}
        try:
            if not names & env.keys():  # no local variable or argument
                return eval(text, namespace)
        except NameError:
            pass
        raise NotImplementedError(
            f"Cannot resolve `{text}` in the definition: premap arguments, axes and called "
            "functions must come from its globals or closure, not its local variables."
        )

    def premap(expression: ast.expr) -> Step:  # `R.boundary(x)` -> `Boundary(R)`, without `x`
        method = expression.func if isinstance(expression, ast.Call) else None
        if isinstance(method, ast.Attribute) and method.attr == "boundary":
            return Boundary(resolve(method.value))
        resolved = _check_premap(resolve(expression))  # a `Boundary` of the globals: no value
        return Boundary(resolved.relation) if isinstance(resolved, Boundary) else resolved

    root = node.func
    while isinstance(root, ast.Attribute):
        root = root.value
    if not isinstance(root, ast.Name) or root.id in env:  # a premap, e.g. `f(I + 1, C2E)`
        reads = _flat(_evaluate(node.func, env, namespace))
        return functools.reduce(_then, [premap(arg) for arg in node.args], reads)
    callee = resolve(node.func)
    values = [_evaluate(arg, env, namespace) for arg in node.args]
    keywords = {k.arg: k.value for k in node.keywords if k.arg is not None}
    if callee in REDUCTIONS:  # e.g. `sum_over(f, axis=Neighbor[C2E])`
        return _then(_flat(values[0]), Reduction(resolve(keywords["axis"])))
    if callee in (flatten, unflatten):  # e.g. `flatten(f, axis=Neighbor[C2E])`: `Flatten(C2E)`
        step = (Flatten if callee is flatten else Unflatten)(resolve(keywords["axis"]).relation)
        return _then(_flat(values[0]), step)
    named = {name: _evaluate(value, env, namespace) for name, value in keywords.items()}
    if not hasattr(callee, "__gt_extents__"):  # another function: the union of its arguments
        return _flat((*values, *named.values()))
    extents: Extents = callee.__gt_extents__  # a DSL operator: compose with its reads
    passed = {**dict(zip(extents, values)), **named}
    return {
        read: None
        for name, value in passed.items()
        for callee_read in extents.get(name, [])
        for read in functools.reduce(_then, callee_read, _flat(value))
    }


def _then(reads: _Reads, step: Step) -> _Reads:
    """Add `step` to the `reads` it applies to; it is a no-op for the others (see `_applies`)."""
    return {
        (name, (*read, step), _step_dims(dims, step))
        if _applies(step, dims)
        else (name, read, dims): None
        for name, read, dims in reads
    }


def _check_premap[P](premap: P) -> P:
    """Return `premap`; a `TypeError` if it is a staged expression that is not a relation, e.g.
    a transpose, `C2V.T`, only used in declarations (see "Footprint algebra" in `concepts`)."""
    if isinstance(premap, (Compose, Transpose)):
        raise TypeError(
            f"`{premap}` has no table of a fixed width: declare a relation with it, "
            f"`stencil=Stencil({premap})`, and premap by that relation."
        )
    return premap


def _is_gather(step: Step) -> TypeIs[type[Relation[Any, Any]] | Boundary]:
    """Whether `step` is a gather: a relation, or a `Boundary`."""
    return is_relation(step) or isinstance(step, Boundary)


def _gathered(step: Step) -> type[Relation[Any, Any]]:
    """The relation of a gather step: a relation itself, or the one of a `Boundary`."""
    return step.relation if isinstance(step, Boundary) else cast(type[Relation[Any, Any]], step)


def _applies(step: Step, dims: tuple[Dimension, ...]) -> bool:
    """Whether `step` applies to a read with the dimensions `dims`: a gather needs the target of
    its relation, a reduction or a scan its axis, a `Flatten(R)` `Neighbor[R]`, an `Unflatten(R)`
    `Pairs[R]`, an offset its dimension, a half offset the staggered partner of its dimension
    (see `Staggered`). Not for a sibling or a broadcast operand, e.g. `k` in `(e * k)(C2E)`,
    with `k: Field[Dims[K], float]`."""
    if _is_gather(step):
        return _gathered(step).target in dims
    if _is_half(step):
        return flip_staggered(step[0]) in dims
    if isinstance(step, (Flatten, Unflatten)):
        pairs = isinstance(step, Unflatten)
        return (pairs_of(step.relation) if pairs else neighbor_of(step.relation)) in dims
    return (step.axis if isinstance(step, (Reduction, ScanAlong)) else step[0]) in dims


def _step_dims(dims: tuple[Dimension, ...], step: Step) -> tuple[Dimension, ...]:
    """The dimensions after `step`: a gather from `R.target` gives `(R.source, Neighbor[R],
    *rest)`, reparented (see `Field.__call__`), or `(R.source, *rest)` if `R` is functional, a
    reduction removes its axis, `Flatten(R)` gives `(Pairs[R], *rest)`, and `Unflatten(R)`
    `(R.source, Neighbor[R], *rest)`. A half offset renames the staggered partner of its
    dimension in place (see `Staggered`)."""
    if _is_gather(step):
        relation = _gathered(step)
        neighbor = neighbor_of(relation)
        rest = (_reparented(d, relation.target, neighbor) for d in dims if d is not relation.target)
        return (relation.source, *(() if relation.functional else (neighbor,)), *rest)
    if _is_half(step):
        dim, _ = step
        return tuple(dim if d is flip_staggered(dim) else d for d in dims)
    if isinstance(step, Reduction):
        return tuple(dim for dim in dims if dim is not step.axis)
    if isinstance(step, Flatten):
        flattened = (step.relation.source, neighbor_of(step.relation))
        return (pairs_of(step.relation), *(dim for dim in dims if dim not in flattened))
    if isinstance(step, Unflatten):
        gathered = _step_dims(dims, step.relation.to_pairs)
        return tuple(_unflattened(dict.fromkeys(gathered), step.relation))
    return dims


def _forward_step(domain: Domain, step: Step) -> Domain:
    """Forward propagation through one step: an offset shifts the domain back, a gather gives
    the sources whose neighbors are all in it (`sources_within`; for a `Boundary`, all but the
    missing ones) with all their neighbors (none if functional), a reduction removes its
    neighbor dimension, and a scan changes nothing. `Flatten(R)` gives the pairs of the sources
    and neighbors (the table of `R.to_pairs`), `Unflatten(R)` is the gather through
    `R.to_pairs`, renamed. A half offset renames the staggered partner of its dimension in
    place, shifted by `-c` (see `_half_domain`). A `Boundary` or a functional relation on a
    domain with neighbor dimensions raises a `NotImplementedError` (Phase 1)."""
    if _is_gather(step):
        relation, boundary = _gathered(step), isinstance(step, Boundary)
        if (boundary or relation.functional) and any(issubclass(dim, Neighbor) for dim in domain):
            read = f"`{relation.__name__}.boundary(...)`" if boundary else f"`{relation.__name__}`"
            raise NotImplementedError(
                f"{read} reads fields without neighbor dimensions only, not "
                f"{format_domain(domain)}."
            )
        sources = sources_within(relation, domain[relation.target], boundary)
        rest = [indices for dim, indices in domain.items() if dim is not relation.target]
        added = [] if relation.functional else [range(relation.max_neighbors)]
        return dict(zip(_step_dims(tuple(domain), step), [sources, *added, *rest], strict=True))
    if isinstance(step, (Reduction, ScanAlong)):
        return {dim: domain[dim] for dim in _step_dims(tuple(domain), step)}
    if isinstance(step, Flatten):
        relation = step.relation
        ranks = _pair_ranks(relation, domain)
        flattened = (relation.source, neighbor_of(relation))
        return {
            pairs_of(relation): index_set(ranks[ranks >= 0]),
            **{d: i for d, i in domain.items() if d not in flattened},
        }
    if isinstance(step, Unflatten):
        return _unflattened(_forward_step(domain, step.relation.to_pairs), step.relation)
    if _is_half(step):
        return _half_domain(domain, step)
    dim, offset = step
    return {**domain, dim: shift(domain[dim], -offset)}


def _backward_step(domain: Domain, step: Step) -> Domain:
    """Backward propagation through one step, the inverse of `_forward_step`: an offset shifts
    the domain, a gather needs all the neighbors of its sources (`neighbors`: a `ValueError`
    for missing ones, unless it is a `Boundary`, and for truncated ones) and un-reparents the
    neighbor dimensions, a reduction needs all the neighbors of the argument (it does not
    restrict its axis: the argument keeps all its neighbors), and a scan its whole axis.
    `Flatten(R)` needs the sources of its pairs (`neighbors` through `R.pair_source`), on all
    the neighbors of the argument (it does not restrict `Neighbor[R]`), and `Unflatten(R)` is
    the gather through `R.to_pairs`. A half offset renames `dim` back to its staggered
    partner, shifted by `+c` (see `_half_domain`). The dimensions not in `domain` are not
    restricted."""
    if _is_gather(step):
        relation = _gathered(step)
        neighbor = neighbor_of(relation)
        rest = {
            _reparented(dim, neighbor, relation.target): indices
            for dim, indices in domain.items()
            if dim not in (relation.source, neighbor)
        }
        if relation.source not in domain:
            return rest
        sources = domain[relation.source]
        return {relation.target: neighbors(relation, sources, isinstance(step, Boundary)), **rest}
    if isinstance(step, Flatten):  # the sources of the pairs, all their neighbors (unrestricted)
        relation, pairs = step.relation, pairs_of(step.relation)
        rest = {dim: indices for dim, indices in domain.items() if dim is not pairs}
        if pairs not in domain:
            return rest
        return {relation.source: neighbors(relation.pair_source, domain[pairs]), **rest}
    if isinstance(step, Unflatten):
        to_pairs = step.relation.to_pairs
        renamed = {
            _reparented(d, neighbor_of(step.relation), neighbor_of(to_pairs)): i
            for d, i in domain.items()
        }
        return _backward_step(renamed, to_pairs)
    if isinstance(step, Reduction):  # all the neighbors of the argument (unrestricted)
        return domain
    if isinstance(step, ScanAlong):
        return {dim: indices for dim, indices in domain.items() if dim is not step.axis}
    if _is_half(step):
        half_dim, half_offset = step
        if half_dim in domain:
            partner = flip_staggered(half_dim)
            indices = domain[half_dim]
            return {
                d if d is not half_dim else partner: (
                    shift(indices, _half_shift(half_dim, half_offset)) if d is half_dim else i
                )
                for d, i in domain.items()
            }
        return domain
    dim, offset = step
    return {**domain, dim: shift(domain[dim], offset)} if dim in domain else domain


def forward_domain(input_domain: Domain, reads: list[Read]) -> Domain:
    """Forward propagation: the largest domain on which all the `reads` stay in `input_domain`.

    Each read is propagated step by step (`_forward_step`), then the results are intersected:
    `{}` without reads, since an argument that is not read does not constrain the result.
    """
    domains = (functools.reduce(_forward_step, read, dict(input_domain)) for read in reads)
    return functools.reduce(intersect, domains, {})


def backward_domain(output_domain: Domain, reads: list[Read]) -> Domain:
    """Backward propagation: the smallest input domain needed to compute `output_domain`.

    Each read is propagated back step by step (`_backward_step`), then the results are united,
    along the dimensions all of them restrict: the others, e.g. the axis of a scan, or all of
    them without reads, are not restricted.
    """
    domains = [functools.reduce(_backward_step, reversed(read), output_domain) for read in reads]
    common = [dim for dim in (domains[0] if domains else {}) if all(dim in d for d in domains)]
    return {dim: union([d[dim] for d in domains]) for dim in common}


def _resolve(annotation: Any) -> Any:
    """Return the value of a type alias, e.g. of `type IJ = Field[Dims[I, J], float]`."""
    return annotation.__value__ if isinstance(annotation, TypeAliasType) else annotation


def result_annotation(definition: Callable[..., Any]) -> Any:
    """Return the result annotation of `definition`, resolved (`None` if there is none)."""
    return _resolve(get_type_hints(call_operator(definition)).get("return"))


def _result_domain(definition: FieldOperatorDefinition[..., Any], domain: Domain) -> Domain:
    """Return `domain` along the dimensions of the result annotated in `definition`, in order.

    A neighbor dimension the reads of the arguments do not give has all its neighbors. The other
    dimensions must be given by the extents of the arguments (see `Extents`).
    """
    result = result_annotation(definition)
    if get_origin(result) is not Field:
        return domain
    missing = [
        dim for dim in dims_of(result) if dim not in domain and not issubclass(dim, Neighbor)
    ]
    if missing:
        name = getattr(definition, "__name__", definition)
        raise NotImplementedError(
            f"The domain of '{name}' along {missing} cannot be computed from the "
            "extents of its arguments: no read of an argument has these dimensions."
        )
    return {
        dim: domain[dim]
        if dim in domain
        else range(cast("type[Neighbor[Any]]", dim).relation.max_neighbors)
        for dim in dims_of(result)
    }


# --- Compilation --------------------------------------------------------------
# Definition -> GTIR (frontend transform) -> generated Python code (backend transform).


def to_gtir(node: Any) -> Any:
    """Convert a Python `ast` node, with its children, into `GTIR` nodes."""
    if isinstance(node, ast.AST):
        return GTIR(type(node).__name__, {name: to_gtir(v) for name, v in ast.iter_fields(node)})
    if isinstance(node, list):
        return [to_gtir(v) for v in node]
    return node


def from_gtir(node: Any) -> Any:
    """Convert `GTIR` nodes back into Python `ast` nodes."""
    if isinstance(node, GTIR):
        return getattr(ast, node.kind)(**{name: from_gtir(v) for name, v in node.fields.items()})
    if isinstance(node, list):
        return [from_gtir(v) for v in node]
    return node


def python_frontend(operator: DSLOperator[..., Any]) -> GTIR:
    """Frontend transform: parse the source of the definition of `operator` into GTIR.

    The kind of the root node is the kind of the operator. It keeps the parsed `function`,
    the `definition` itself (simplified), the `scan_config` of a scan operator, and the global
    and closure values the definition uses (dimensions, types, ...) as `closure_vars`, since the
    generated code runs outside the definition's module (see `_namespace`).
    """
    definition = definition_of(operator)
    function_def = cast(ast.FunctionDef, ast.parse(getsource(definition)).body[0])
    function_def.decorator_list = []
    module_vars = _namespace(call_operator(definition))
    parameters = inspect.signature(definition).parameters
    names = {
        node.id
        for node in ast.walk(function_def)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id not in parameters
    }
    return GTIR(
        operator.__gt_operator_kind__.value,
        {
            "definition": definition,
            "function": to_gtir(function_def),
            "scan_config": getattr(operator, "__gt_scan_config__", None),
            "closure_vars": {name: module_vars[name] for name in names if name in module_vars},
        },
    )


def python_backend(ir: GTIR) -> CompiledDSLOperator[..., Any]:
    """Backend transform: generate Python source from GTIR, as a compiled DSL operator of the
    kind of the root node (simplified: no optimizations)."""
    function = ir.fields["function"]
    source = ast.unparse(ast.fix_missing_locations(from_gtir(function)))
    is_scan = ir.kind == DSLOperatorKind.SCAN_OPERATOR.value
    kind = GeneratedScanOperator if is_scan else GeneratedFieldOperator
    return kind(
        ir.fields["definition"],
        function.fields["name"],
        source,
        ir.fields["closure_vars"],
        ir.fields["scan_config"],
    )


@dataclasses.dataclass
class GeneratedOperator(DSLOperatorBase[..., Any]):
    """Base of the compiled DSL operators below, running Python source generated from GTIR.

    They are compiled by `.compile(...)` ahead of time, or else on the first call (just in
    time). The arguments of `.compile(...)` are ignored (simplified: no specialization).
    """

    __gt_definition__: DSLOperatorDefinition[..., Any]
    name: str
    source: str
    closure_vars: dict[str, Any]
    scan_config: ScanConfig | None = None
    compiled: Callable[..., Any] | None = None

    def compile(self, *args: Any, **kwargs: Any) -> Self:
        print(f"[compiled] compiling {self.name}")
        namespace = dict(self.closure_vars)
        exec(self.source, namespace)
        self.compiled = namespace[self.name]
        return self

    def _code(self, *args: Any, **kwargs: Any) -> Callable[..., Any]:
        if self.compiled is None:
            self.compile(*args, **kwargs)
        assert self.compiled is not None
        return self.compiled


class GeneratedFieldOperator(GeneratedOperator):
    """A `CompiledFieldOperator`: values computed when called, with forward domains, like
    `EagerFieldOperator`."""

    __gt_operator_kind__: ClassVar[Literal[DSLOperatorKind.FIELD_OPERATOR]] = (
        DSLOperatorKind.FIELD_OPERATOR
    )

    @property
    def __gt_extents__(self) -> Extents:
        return read_extents(self.__gt_definition__)

    def _compute(self, *args: Any, **kwargs: Any) -> Any:
        result = self._code(*args, **kwargs)(*args, **kwargs)
        result = as_annotated(result, result_annotation(self.__gt_definition__))
        print(f"[compiled] computed {result}")
        return result


class GeneratedScanOperator(GeneratedOperator):
    """A `CompiledScanOperator`: the compiled code computes one level, like the definition, and
    the levels are computed when called, like `EagerScanOperator`."""

    __gt_operator_kind__: ClassVar[Literal[DSLOperatorKind.SCAN_OPERATOR]] = (
        DSLOperatorKind.SCAN_OPERATOR
    )

    @property
    def __gt_scan_config__(self) -> ScanConfig:
        assert self.scan_config is not None
        return self.scan_config

    @property
    def __gt_extents__(self) -> Extents:
        return scan_extents(self.__gt_definition__, self.__gt_scan_config__.axis)

    def _compute(self, *fields: Any, **kwargs: Any) -> NumPyArrayField:
        result = scan(self._code(*fields), self.__gt_scan_config__, fields)
        print(f"[compiled] computed {result}")
        return result


# --- Backends -----------------------------------------------------------------
# They take DSL operators, e.g. those returned by the decorators, and return their own.


class BaseBackend(Backend):
    """Base of the backends below, which inherit `__call__` from `Backend`: fields are allocated
    with `allocator`, `numpy_field_allocator` unless overridden, and scan operators are computed
    eagerly (unless overridden)."""

    name: ClassVar[str]
    allocator: ClassVar[NumPyFieldAllocator] = numpy_field_allocator

    def __repr__(self) -> str:
        return self.name

    @property
    def __gt_allocator__(self) -> NumPyFieldAllocator:
        return self.allocator

    def scan_operator(self, operator: ScanOperator[..., Any]) -> ScanOperator[..., Any]:
        return EagerScanOperator(operator.__gt_definition__, operator.__gt_scan_config__)


class EagerInterpreter(BaseBackend):
    """Forward domains; values computed when called. A requested domain only slices the result."""

    name = "eager"

    def field_operator[**P, R](self, operator: FieldOperator[P, R]) -> EagerFieldOperator[P, R]:
        return EagerFieldOperator(operator.__gt_definition__)


class LazyInterpreter(BaseBackend):
    """Forward domains: when the result is first used, compute its values on the largest
    domain the arguments allow, then slice the requested domain."""

    name = "lazy"

    def field_operator[**P, R](self, operator: FieldOperator[P, R]) -> LazyFieldOperator[P, R]:
        definition = operator.__gt_definition__
        function = cast(Callable[..., Field], definition)  # happy path: returns a single field

        def computation(domain: Domain, *args: P.args, **kwargs: P.kwargs) -> Field:
            result = function(*args, **kwargs)
            print(f"[lazy] computed {format_domain(result.domain)}, kept {format_domain(domain)}")
            return result[domain]

        return LazyFieldOperator(definition, computation)


class LazyBackwardInterpreter(BaseBackend):
    """Backward domains: when the result is first used, propagate the requested domain back.

    Each argument is restricted to the domain its reads need for the requested one, step by
    step (`backward_domain`): grown by the extent along offsets, the neighbors through
    relations, whole along a scan axis. So values are only computed on the requested domain;
    through relations, a few more points may be computed: those whose neighbors are all among
    the needed ones. If an argument's domain does not hold the points needed, this restriction
    raises a `ValueError`, when the values are computed.
    """

    name = "lazy_backward"

    def field_operator[**P, R](self, operator: FieldOperator[P, R]) -> LazyFieldOperator[P, R]:
        definition = operator.__gt_definition__
        # Happy path: returns a single field.
        executable = cast(Callable[..., Field], definition)
        names = inspect.signature(definition).parameters
        extents = read_extents(definition)

        def computation(domain: Domain, *args: P.args, **kwargs: P.kwargs) -> Field:
            print(f"[lazy_backward] computed {format_domain(domain)}")

            def needed(name: str, a: Any) -> Any:  # the argument on the domain its reads need
                return a[backward_domain(domain, extents[name])] if isinstance(a, Field) else a

            inputs = [needed(n, a) for n, a in zip(names, args)]
            keywords = {n: needed(n, a) for n, a in kwargs.items()}
            return executable(*inputs, **keywords)[domain]

        return LazyFieldOperator(definition, computation)


class ColumnMajorInterpreter(BaseBackend):
    """Like `eager`, but with column-major fields: its allocator lays out the values in memory
    with the first dimension varying fastest (Fortran order), and its operators return their
    results in fields from that allocator. The values are the same as with `eager`."""

    name = "column_major"
    allocator = column_major_field_allocator

    def field_operator[**P, R](self, operator: FieldOperator[P, R]) -> AllocatedFieldOperator[P, R]:
        return AllocatedFieldOperator(operator.__gt_definition__, self.__gt_allocator__)

    def scan_operator(self, operator: ScanOperator[..., Any]) -> AllocatedScanOperator:
        definition, config = operator.__gt_definition__, operator.__gt_scan_config__
        return AllocatedScanOperator(definition, config, self.__gt_allocator__)


class PythonCompiler(GTIRCompiler, BaseBackend):
    """A `GTIRCompiler` generating Python code: operator -> GTIR -> compiled DSL operator.

    `field_operator` and `scan_operator` are inherited from `GTIRCompiler`. Both transforms
    can be replaced, e.g. `PythonCompiler(backend_transform=my_backend)`.
    """

    name = "compiled"

    def __init__(
        self,
        frontend_transform: Transform[DSLOperator[..., Any], GTIR] = python_frontend,
        backend_transform: Transform[GTIR, CompiledDSLOperator[..., Any]] = python_backend,
    ) -> None:
        self.frontend_transform = frontend_transform
        self.backend_transform = backend_transform


eager = EagerInterpreter()
lazy = LazyInterpreter()
lazy_backward = LazyBackwardInterpreter()
column_major = ColumnMajorInterpreter()
compiled = PythonCompiler()
