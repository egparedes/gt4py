---
tags: []
---

# Connectivities as Types

- **Status**: proposed
- **Authors**: Enrique González Paredes (@egparedes)
- **Created**: 2026-09-21
- **Updated**: 2026-10-29

A neighbor connectivity is declared as a **class**, and its local dimension as a
class **nested** in it:

```python
class V2E(gtx.NeighborConnectivity[Vertex, Edge], max_neighbors=6, min_neighbors=5):
    class Local(gtx.LocalDimensionIndex): ...


@gtx.field_operator
def f(a: Field[Dims[Edge], float]) -> Field[Dims[Vertex], float]:
    return neighbor_sum(a(V2E), axis=V2E.Local) + a(V2E[0])
```

The declaration is written in DSL code, owns its local dimension, and states the
constraints a neighbor table bound to it has to satisfy. It holds no data. It builds on
[ADR 0029](0029-Dimensions_As_Nominal_Types.md): the connectivity, like a dimension, is
identified by its type at the Python edge and by a value, its `ConnectivityRef`, below it,
and `V2E.Local` is an ordinary dimension class with the tag `<module>:V2E.Local`.

## Context

An unstructured connectivity used to be spelled by four independently authored
names that had to agree, none of them checked against the others: the
`FieldOffset` tag, the Python variable it was bound to, the local dimension's
name and the offset-provider key. The `V2EDim = Dimension("V2E")` convention made
all four equal, which hid which one each execution path actually used; the
regression tests in `test_offset_dimensions_names.py` break the convention one
name at a time. Nothing tied a local dimension to the table it indexes, so the
backends recovered that link by string equality, and the table's shape, codomain
and skip values were never checked against the `FieldOffset` declaration.

## Decision

### The declaration

- `NeighborConnectivity[Domain, Codomain]` is a PEP 695 generic whose subclasses
  are declarations: for each `Domain` element, a list of `Codomain` neighbors. Its
  metaclass, `ConnectivityMeta`, forbids instantiation. The two dimensions are
  the class attributes `V2E.domain` and `V2E.codomain`.
- The local dimension is the nested class `Local`, a subclass of
  `LocalDimensionIndex`. Declaring it is required, and `NeighborConnectivity`
  sets `Local.owner` to the connectivity when the class is created. A local
  dimension can have at most one owner; a declaration redefined under the same
  name (a re-run notebook cell) takes ownership over again, and for a local
  dimension adopted rather than nested, the first declaration wins.
- A local dimension with no table, such as the coefficient axis of a fixed-size
  stencil, is declared on its own: `class LsqCoeff(LocalDimensionIndex, size=3)`.
  Its `owner` is `None`. A declaration can also *adopt* such a module-level local
  dimension, written `Local: TypeAlias = LsqCoeff`, which then keeps its own tag.
  A local dimension that belongs to no connectivity by construction says so with
  the class variable `adoptable = False`; `iterator.builtins.ConstList`, the local
  dimension of `make_const_list` results, does, and `common` reads the marker
  without importing the module that declares it.
- A connectivity can *share* another one's local dimension,
  `Local: TypeAlias = C2E.Local`.
  This is the flattened sparse pattern, e.g. cell-to-cell-edge (`C2CE: Cell -> CellEdge`) indexing the same neighbor axis as `C2E`, so that its results
  combine with `C2E`-shaped sparse fields. The owner stays `C2E`, and the
  neighbor counts and skip-value structure are the owner's. A sharer must have
  the owner's domain; its codomain is free. The local dimension records its
  sharers (`Local.sharers`) as it records its owner.
- `max_neighbors` and `min_neighbors` are optional class keywords, not type
  parameters: Python has no integer type parameters, and nothing static needs
  the count. A declared count is a constraint on the bound table; an undeclared
  one is taken from the table. `min_neighbors < max_neighbors` means that the
  table must use skip values.
- `common.check_neighbor_table(V2E, table)` checks a table, or just its type
  (which is all an ahead-of-time compilation has), against the declaration, and
  returns the table's `NeighborTableType` (below): the domain is
  `(Domain, V2E.Local)`, the codomain is `Codomain`, the dtype is integral, and
  the neighbor counts and skip values agree. Skip values are checked on the
  table's type: a table with a `skip_value` counts as having skip values whether
  or not an entry uses it.

`Domain` and `Codomain` name the two index spaces the declaration maps between.
A bound table is a field over `(Domain, Local)` with values in `Codomain`: the
table's domain is the declaration's domain extended by the local axis, which is
the same use of the word as `Connectivity.domain` and
`CartesianConnectivity.domain_dim`. "Origin" would have been the other natural
name for the first dimension, but gt4py already uses it for the start of a
buffer (`__gt_origin__`).

### The type of a bound table

Transformations and code generation see types, never tables (ADR 0019). The type
of a table bound to a declaration is a `common.NeighborTableType`:
`connectivity` (the declaration's `ConnectivityRef`, below), `dtype`, `skip_value`
and `max_neighbors`. Its `domain` and `codomain` are derived from the ref,
`(connectivity.domain, connectivity.local)` and `connectivity.codomain`, so they
cannot disagree with it. The mapping from
offset-provider keys to these records is `common.TableTypes`, and it can be
given instead of the tables for ahead-of-time compilation.

A table cannot tell which declaration it is bound to: the table of a sharer
(`C2CE`) has the same domain as its owner's (`C2E`), with another codomain. So a
`NeighborTableType` is built where a table is bound, from its offset-provider key:
`check_neighbor_table(C2CE, table)`, or `offset_provider_to_type`, which finds
the declaration whose `offset_tag` is the key among the owner and the sharers of
the table's local dimension. `NeighborTable.__gt_type__()` returns only what the
table knows, the structural `common.ConnectivityType` (domain, codomain, dtype,
skip value).

A table bound under a key that no declaration answers to -- hand-written IR
names its offsets by plain strings -- has no declaration. Its
`NeighborTableType` then has the table's structural `ConnectivityType` as its
`connectivity`, and `domain` and `codomain` are read from that. This keeps the
IR level, which does not know declarations, working unchanged.

A `NeighborTableType` is fingerprinted through its fields, so the declaration
takes part in the fingerprint of everything compiled for it: the owner's and a
sharer's tables, identical as tables, produce different artifact keys.

### Classes at the Python edge, `ConnectivityRef` below it

The two levels of ADR 0029 apply to declarations too. `common.connectivity_ref(V2E)`
converts a declaration class into a `ConnectivityRef`: an immutable, interned value
holding the declaration's module and qualname (its `tag`), its `offset_tag`, its
domain, codomain and local dimension as `DimensionRef`s, and the neighbor counts
of its local dimension (the owner's, for a sharer; `None` where the table decides).
It follows `DimensionRef`: equal refs are the identical object, the cached hash is
neither pickled nor fingerprinted, its `repr` differs from the class's, and it
never compares equal to the class (raising under
`config.STRICT_DIMENSION_REF_COMPARISON`). The conversion is not memoized on the
class, since an owner redefined later rewrites the counts of an adopted local
dimension.

Below the edge only the ref appears: `NeighborTableType.connectivity` (a class
passed to it is converted), the `ts.ShiftType` of `V2E.__gt_type__()`, which is
built from the ref, and the table types `offset_provider_to_type` produces. A
declaration is fingerprinted through its ref, so a fingerprint no longer requires
the declaration to be importable. The runtime keeps the classes where it acts on
user objects: embedded execution (`a(V2E)`, `V2E[i]`), `check_neighbor_table`
called with a class, and the owner and sharers bookkeeping on the local dimension
that finds the declaration a table is bound to (`common.bound_declaration`).

A re-run notebook cell gives an equal ref for a new class, so the call-time
identity check of ADR 0029 covers declarations: `connectivity_ref` records the
class like `dimension_ref` does, on the definition side for the declarations among
its closure variables and on the argument side for the declaration each provided
table is bound to. A program written with a previous `C2CE`, called with a table
bound to the current one, is rejected, even though no dimension class changed.

### `NeighborConnectivity` is not a `Connectivity`

`common.Connectivity` is a *data* protocol (`ndarray`, `domain`, `asnumpy`); a
declaration holds no data. The neighbor table stays a `Connectivity`
implementation, and the declaration is only the type the table is checked
against. `Field.premap` and `Field.__call__` accept either, as they already
accepted a `FieldOffset`, which is not a `Connectivity` either.

### `LocalDimensionIndex` subclasses `DimensionIndex`

A separate root would force every `type[DimensionIndex]` annotation in the tree
(`ts.FieldType.dims`, `Domain`, `ConnectivityType.domain`, ...) to widen, and
would then accept local dimensions wherever a primary one is meant anyway. The
tree tells local dimensions apart at runtime, and constructors whose parameter
must be a primary dimension (`NeighborConnectivity[Domain, Codomain]`) check it;
`Staggered[D]` rejects a local dimension statically too, through its bound on a
declared Cartesian axis (ADR 0029).

### Localness is the class, and `DimensionKind.LOCAL` is removed

A dimension is local if and only if it subclasses `LocalDimensionIndex`
(`common.is_local_dimension(dim)`, which also takes a ref: the ref's
`DimensionCategory.LOCAL` is derived from that subclass), so `DimensionKind.LOCAL` is removed and
`DimensionKind` is `HORIZONTAL | VERTICAL`. A local dimension's `kind` is `None`,
and declaring one with `kind=` is a `TypeError`; on a `DimensionRef` too, `kind`
is `None` exactly for the `LOCAL` category. `None` rather than `HORIZONTAL`
keeps every `kind == HORIZONTAL` / `kind != VERTICAL` comparison in the backends
meaning what it meant: with `HORIZONTAL`, a sparse field would silently count its
local axis as horizontal. Two consequences:

- `None` does not order against the enum, so `order_dimensions` sorts by an
  explicit rank — horizontal, then local, then vertical — the order `kind` used to
  encode. It must not move, or the memory layout of sparse fields changes with it.
- Displays derive the label from the class: `str(V2E.Local)` is still
  `Local[local]`, the IR pretty printer still marks a local axis with `ₗ`, and DaCe
  map variables of a local dimension keep their `_gtx_localdim` suffix.

What remains of `kind` is the layout sort key and the scan axis.

### `Local` is not annotated anywhere

Neither `NeighborConnectivity` nor `ConnectivityMeta` annotates `Local`, and
that is load-bearing: an annotation makes a declaration's `Local` a *variable*
for the checkers, so `Field[Dims[Vertex, V2E.Local], float]` is rejected by
pyright ("Variable not allowed in type expression") for a nested `Local`, and by
mypy ("not valid as a type") for an adopted or shared one. A real nested `Local`
on the base is not an option either: pyright reports an incompatible override in
every declaration. With no annotation, all three spellings are types for both
checkers, which `typing_tests/pyright_probes.py` pins for pyright and
`typing_tests/test_next.yaml` for mypy.

The cost is that `conn.Local` is not an attribute the checkers know for a
*generic* `conn`. Library code reads it through `common.local_dimension_of(conn)`
instead, and code that has to name a local dimension generically uses a
`TypeVar` bound to `LocalDimensionIndex`. Writing an adopted or shared local as
`Local: TypeAlias = ...` (rather than a plain assignment) is what keeps mypy
treating it as a type.

### Frontend integration

A declaration is typed like the `FieldOffset` it replaces: `V2E.__gt_type__()`
is a `ts.ShiftType`, which takes a field over the codomain to one over the
domain, `Shift[<tag>: Edge -> (Vertex, V2E.Local)]`. `V2E[i]` has the domain
`(Vertex,)`, and so does a Cartesian shift `KDim + 1`, over `KDim` and without a
tag. The tag is the connectivity's `offset_tag`:

- **the local dimension's tag**, `V2E.Local.tag`, for the connectivity that
  declares it. This is the single string that shifts, neighbor reductions and
  sparse arguments already use to find the table in the offset provider, so
  existing backends need no change.
- **its own tag**, `C2CE.tag`, for a connectivity that shares another one's local
  dimension, since the local dimension's tag already names the owner's table.
  Shifts find the table by that tag. Reductions and sparse arguments know only
  the local dimension, and take its neighbor count and skip values from a table
  over it (`common.connectivity_key_over`): the owner's if bound, else the
  sharer with the smallest tag. Connectivities sharing a local dimension must
  therefore have the same neighbor *structure* — the same count, and a skip value
  at the same positions — which is what sharing a neighbor axis means.

`V2E.Local` inside DSL code types as that local dimension, and
`FieldOffset.Local` names the same thing on a legacy offset, so the spelling
works for both. The other frontend touch points treat the class like the
`FieldOffset` it derives: grid-type deduction (`transform_utils`, `past_to_itir`)
counts it as unstructured, and embedded `premap` accepts it. `V2E[i]` subscripts the
metaclass, which forwards type-parameter subscription (`NeighborConnectivity[V, E]`) to `__class_getitem__`, since a metaclass `__getitem__` shadows it.

## Consequences

- An unstructured connectivity is spelled once. The provider key, the offset tag
  and the local dimension are all derived from the declaration.
- A table bound to a connectivity can be checked against its declaration.
- `V2E.Local` in DSL code is resolved from the shift type, because the type of
  `V2E` is not the class.
- Code generation sees which declaration a table is bound to, by its
  `ConnectivityRef`, not only its shape; a table without a declaration is typed by
  its structure. Nothing below the Python edge holds a declaration class.
- A declaration is fingerprinted through its ref, i.e. by its name *and* its
  declared dimensions and counts, so redefining it under the same name (e.g.
  re-running a notebook cell) does not reuse artifacts compiled for the old
  declaration, and a call mixing the old and the new declaration is rejected.
- `FieldOffset` remains during migration; a `FieldOffset` and a
  `NeighborConnectivity` sharing a local dimension are interchangeable.

## Alternatives considered

- **The local dimension generated by the metaclass**, e.g. `V2E.Local` created
  from `V2E`'s name. Type checkers cannot see a generated class, so it could not
  be used in `Field[Dims[Vertex, V2E.Local], ...]`.
- **Neighbor counts as type parameters.** Python has no integer type parameters,
  and a `Literal[6]` argument would add a type parameter nothing statically uses.
- **`NeighborConnectivity` as a `Connectivity` subclass.** Mixes the
  declaration with the data protocol; see above.
