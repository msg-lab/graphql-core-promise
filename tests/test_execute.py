import threading
from concurrent.futures import Future
from unittest.mock import MagicMock, call

from graphql.execution.execute import execute
from graphql.language import parse
from graphql.type import (
    GraphQLArgument,
    GraphQLField,
    GraphQLID,
    GraphQLNonNull,
    GraphQLObjectType,
    GraphQLSchema,
    GraphQLString,
)
from graphql.type.definition import GraphQLList
from promise import Promise
from promise.dataloader import DataLoader
from promise.promise import async_instance

from graphql_core_promise.execute import PromiseExecutionContext

author_load_stub = MagicMock()
name_loader_stub = MagicMock()
book_loader_stub = MagicMock()


class AuthorNode:
    pass


class BookNode:
    pass


class AuthorLoader(DataLoader):
    def batch_load_fn(self, keys):
        author_load_stub(keys)
        return Promise.resolve([type(key, (AuthorNode,), {"id": key}) for key in keys])


class NameLoader(DataLoader):
    def batch_load_fn(self, keys):
        name_loader_stub(keys)
        return Promise.resolve([key + " name" for key in keys])


class BooksLoader(DataLoader):
    def batch_load_fn(self, keys):
        book_loader_stub(keys)
        return Promise.resolve([type(key, (BookNode,), {"id": key}) for key in keys])


author_loader = AuthorLoader()
name_loader = NameLoader()
books_loader = BooksLoader()

Book = GraphQLObjectType(
    "Book",
    lambda: {
        "id": GraphQLField(GraphQLString, resolve=lambda obj, _info: obj.id),
        "name": GraphQLField(
            GraphQLString, resolve=lambda obj, _info: name_loader.load(obj.id)
        ),
        "author": GraphQLField(
            Author,
            resolve=lambda obj, _info: author_loader.load(obj.id.split(" ")[0]),
        ),
    },
)

Author = GraphQLObjectType(
    "Author",
    lambda: {
        "id": GraphQLField(GraphQLString, resolve=lambda obj, _info: obj.id),
        "name": GraphQLField(
            GraphQLString, resolve=lambda obj, _info: name_loader.load(obj.id)
        ),
        "books": GraphQLField(
            GraphQLList(Book),
            resolve=lambda obj, _info: books_loader.load_many([
                obj.id + f" book_{i}" for i in range(1, 3)
            ]),
        ),
    },
)

TestQuery = GraphQLObjectType(
    "Query",
    {
        "author": GraphQLField(
            Author,
            args={"id": GraphQLArgument(GraphQLID)},
            resolve=lambda _obj, _info, id: author_loader.load(id),
        ),
        "books": GraphQLField(
            GraphQLList(Book),
            args={"ids": GraphQLArgument(GraphQLList(GraphQLID))},
            resolve=lambda _, __, *, ids: [books_loader.load(id) for id in ids],
        ),
    },
)
TestMutation = GraphQLObjectType(
    "Mutation",
    {
        "newAuthor": GraphQLField(
            Author,
            args={"id": GraphQLArgument(GraphQLString)},
            resolve=lambda obj, info, id: author_loader.load(id),
        )
    },
)

TestSchema = GraphQLSchema(TestQuery, TestMutation)


def test_executes_simple_query(benchmark):
    foo_query = parse(
        """
        query Foo {
            a: author(id: "1") {
            id,
            name,
            }
            b: author(id: "2") {
            id,
            name,
            }
            c: author(id: "2") {
            id,
            name,
            }
        }
        """
    )

    def t():
        assert execute(
            schema=TestSchema,
            document=foo_query,
            execution_context_class=PromiseExecutionContext,
        ) == (
            {
                "a": {"id": "1", "name": "1 name"},
                "b": {"id": "2", "name": "2 name"},
                "c": {"id": "2", "name": "2 name"},
            },
            None,
        )

    benchmark(t)

    author_load_stub.assert_called_once_with(["1", "2"])
    name_loader_stub.assert_called_once_with(["1", "2"])


def test_execute_mutation():
    author_load_stub.reset_mock()
    name_loader_stub.reset_mock()

    author_loader.clear_all()
    books_loader.clear_all()
    name_loader.clear_all()

    foo_mutation = parse(
        """
        mutation Foo {
            a: newAuthor(id: "3") {
                id
                name
            }
            b: newAuthor(id: "4") {
                id
                name
            }
        }
    """
    )

    assert execute(
        schema=TestSchema,
        document=foo_mutation,
        execution_context_class=PromiseExecutionContext,
    ) == (
        {"a": {"id": "3", "name": "3 name"}, "b": {"id": "4", "name": "4 name"}},
        None,
    )

    # Mutation fields run serially: ``b`` is resolved only after ``a`` and its
    # sub-selections have completed, so the loaders cannot batch across them.
    assert author_load_stub.call_args_list == [call(["3"]), call(["4"])]
    assert name_loader_stub.call_args_list == [call(["3"]), call(["4"])]


def test_handles_rejected_promise_field_error():
    schema = GraphQLSchema(
        GraphQLObjectType(
            "Query",
            {
                "error": GraphQLField(
                    GraphQLString,
                    resolve=lambda _obj, _info: Promise.reject(
                        RuntimeError("Promise failed")
                    ),
                )
            },
        )
    )

    result = execute(
        schema=schema,
        document=parse("{ error }"),
        execution_context_class=PromiseExecutionContext,
    )

    assert result.data == {"error": None}
    assert result.errors is not None
    assert len(result.errors) == 1
    assert result.errors[0].message == "Promise failed"
    assert result.errors[0].path == ["error"]


def test_execute_list_results(benchmark):
    author_load_stub.reset_mock()
    name_loader_stub.reset_mock()
    book_loader_stub.reset_mock()

    author_loader.clear_all()
    books_loader.clear_all()
    name_loader.clear_all()

    foo_query = parse(
        """
        query Foo {
            a: author(id: "1") {
            id,
            name,
            books {
                id,
                name,
                author {
                    id
                    name
                    books {
                        id,
                        name,
                    }
                }
            }
            }
            b: author(id: "2") {
                id,
                name,
                books {
                    id,
                    name,
                    author {
                        name
                    }
                }
            }
            books(ids: ["1 book_1", "2 book_1"]) {
                id
                name
            }
        }
        """
    )

    def t():
        assert execute(
            schema=TestSchema,
            document=foo_query,
            execution_context_class=PromiseExecutionContext,
        ) == (
            {
                "a": {
                    "id": "1",
                    "name": "1 name",
                    "books": [
                        {
                            "id": "1 book_1",
                            "name": "1 book_1 name",
                            "author": {
                                "id": "1",
                                "name": "1 name",
                                "books": [
                                    {"id": "1 book_1", "name": "1 book_1 name"},
                                    {"id": "1 book_2", "name": "1 book_2 name"},
                                ],
                            },
                        },
                        {
                            "id": "1 book_2",
                            "name": "1 book_2 name",
                            "author": {
                                "id": "1",
                                "name": "1 name",
                                "books": [
                                    {"id": "1 book_1", "name": "1 book_1 name"},
                                    {"id": "1 book_2", "name": "1 book_2 name"},
                                ],
                            },
                        },
                    ],
                },
                "b": {
                    "id": "2",
                    "name": "2 name",
                    "books": [
                        {
                            "id": "2 book_1",
                            "name": "2 book_1 name",
                            "author": {"name": "2 name"},
                        },
                        {
                            "id": "2 book_2",
                            "name": "2 book_2 name",
                            "author": {"name": "2 name"},
                        },
                    ],
                },
                "books": [
                    {"id": "1 book_1", "name": "1 book_1 name"},
                    {"id": "2 book_1", "name": "2 book_1 name"},
                ],
            },
            None,
        )

    benchmark(t)

    author_load_stub.assert_called_once_with(["1", "2"])
    name_loader_stub.assert_has_calls(
        [call(["1", "2", "1 book_1", "2 book_1"]), call(["1 book_2", "2 book_2"])],
        any_order=True,
    )
    book_loader_stub.assert_has_calls(
        [call(["1 book_1", "2 book_1"]), call(["1 book_2", "2 book_2"])],
        any_order=True,
    )


def _execute(schema, query):
    result = execute(
        schema=schema,
        document=parse(query),
        execution_context_class=PromiseExecutionContext,
    )
    assert not Promise.is_thenable(result), "promise execution must finish sync"
    return result


def _errors(result):
    return sorted(
        ((error.message, error.path) for error in result.errors or []),
        key=lambda error: error[0],
    )


def _reject(message):
    return lambda *_args, **_kwargs: Promise.reject(RuntimeError(message))


def _resolve(value):
    return lambda *_args, **_kwargs: Promise.resolve(value)


# The expected values below are what plain graphql-core produces for the same
# schemas with coroutines in place of promises (the spec's error-propagation
# rules): an error nulls the nearest nullable ancestor and is located where it
# occurred, never at the list or the root.


def test_non_null_child_failure_under_promise_resolved_object_nulls_the_object():
    child = GraphQLObjectType(
        "Child",
        {
            "good": GraphQLField(GraphQLString, resolve=_resolve("g")),
            "required": GraphQLField(
                GraphQLNonNull(GraphQLString), resolve=_reject("required failed")
            ),
        },
    )
    schema = GraphQLSchema(
        GraphQLObjectType(
            "Query",
            {
                "child": GraphQLField(child, resolve=_resolve({})),
                "other": GraphQLField(GraphQLString, resolve=_resolve("o")),
            },
        )
    )

    result = _execute(schema, "{ child { good required } other }")

    assert result.data == {"child": None, "other": "o"}
    assert _errors(result) == [("required failed", ["child", "required"])]


def test_sibling_errors_under_a_nulled_object_are_still_collected():
    # Matches plain graphql-core in both field orders. Note that on graphql-core
    # >= 3.2.10 ``CollectedErrors.add`` drops an error whose ancestor position is
    # already nulled, so the ``optional`` error survives only because its own
    # ``.catch`` runs before the one chained after ``Promise.all`` on ``child``;
    # an extra link in either chain could flip this.
    child = GraphQLObjectType(
        "Child",
        {
            "required": GraphQLField(
                GraphQLNonNull(GraphQLString), resolve=_reject("required failed")
            ),
            "optional": GraphQLField(GraphQLString, resolve=_reject("optional failed")),
        },
    )
    schema = GraphQLSchema(
        GraphQLObjectType("Query", {"child": GraphQLField(child, resolve=_resolve({}))})
    )

    result = _execute(schema, "{ child { required optional } }")

    assert result.data == {"child": None}
    assert _errors(result) == [
        ("optional failed", ["child", "optional"]),
        ("required failed", ["child", "required"]),
    ]


def test_rejected_item_in_list_nulls_only_that_item():
    schema = GraphQLSchema(
        GraphQLObjectType(
            "Query",
            {
                "items": GraphQLField(
                    GraphQLList(GraphQLString),
                    resolve=lambda *_: [
                        Promise.resolve("x"),
                        Promise.reject(RuntimeError("item failed")),
                        "z",
                    ],
                ),
                "other": GraphQLField(GraphQLString, resolve=_resolve("o")),
            },
        )
    )

    result = _execute(schema, "{ items other }")

    assert result.data == {"items": ["x", None, "z"], "other": "o"}
    assert _errors(result) == [("item failed", ["items", 1])]


def test_rejected_dataloader_item_in_list_nulls_only_that_item():
    class ItemLoader(DataLoader):
        def batch_load_fn(self, keys):
            return Promise.resolve([
                RuntimeError(f"no item {key}") if key == "2" else f"item {key}"
                for key in keys
            ])

    loader = ItemLoader()
    schema = GraphQLSchema(
        GraphQLObjectType(
            "Query",
            {
                "items": GraphQLField(
                    GraphQLList(GraphQLString),
                    resolve=lambda *_: [loader.load(key) for key in ("1", "2", "3")],
                )
            },
        )
    )

    result = _execute(schema, "{ items }")

    assert result.data == {"items": ["item 1", None, "item 3"]}
    assert _errors(result) == [("no item 2", ["items", 1])]


def test_thenable_item_without_then_nulls_only_that_item():
    # ``promise.is_thenable`` accepts future-like objects that have no ``then``;
    # the ``AttributeError`` belongs to the item, as ``await item`` would in
    # graphql-core, not to the list.
    future = Future()
    future.set_result("y")
    schema = GraphQLSchema(
        GraphQLObjectType(
            "Query",
            {
                "items": GraphQLField(
                    GraphQLList(GraphQLString), resolve=lambda *_: ["x", future]
                ),
                "other": GraphQLField(GraphQLString, resolve=_resolve("o")),
            },
        )
    )

    result = _execute(schema, "{ items other }")

    assert result.data == {"items": ["x", None], "other": "o"}
    assert [error.path for error in result.errors] == [["items", 1]]
    assert isinstance(result.errors[0].original_error, AttributeError)


def test_rejected_non_null_item_nulls_the_list_and_keeps_the_item_path():
    schema = GraphQLSchema(
        GraphQLObjectType(
            "Query",
            {
                "items": GraphQLField(
                    GraphQLList(GraphQLNonNull(GraphQLString)),
                    resolve=lambda *_: [
                        Promise.resolve("x"),
                        Promise.reject(RuntimeError("item failed")),
                    ],
                ),
                "other": GraphQLField(GraphQLString, resolve=_resolve("o")),
            },
        )
    )

    result = _execute(schema, "{ items other }")

    assert result.data == {"items": None, "other": "o"}
    assert _errors(result) == [("item failed", ["items", 1])]


def test_completion_error_on_a_resolved_promise_is_located_at_the_field():
    schema = GraphQLSchema(
        GraphQLObjectType(
            "Query",
            {
                "items": GraphQLField(GraphQLList(GraphQLString), resolve=_resolve(42)),
                "other": GraphQLField(GraphQLString, resolve=_resolve("o")),
            },
        )
    )

    result = _execute(schema, "{ items other }")

    assert result.data == {"items": None, "other": "o"}
    assert _errors(result) == [
        (
            "Expected Iterable, but did not find one for field 'Query.items'.",
            ["items"],
        )
    ]


# Mutation root fields: the expected values are again what plain graphql-core
# produces with coroutines, plus graphql-js's guarantee that a field's resolver
# is not called before the previous field has completed.


def _mutation_schema(fields):
    return GraphQLSchema(
        GraphQLObjectType("Query", {"unused": GraphQLField(GraphQLString)}),
        GraphQLObjectType("Mutation", fields),
    )


def _recorded(order, name, resolve):
    def resolver(*args, **kwargs):
        order.append(("resolve", name))
        return resolve(*args, **kwargs)

    return resolver


def test_mutation_fields_resolve_one_at_a_time():
    order = []

    class RecordingLoader(DataLoader):
        def batch_load_fn(self, keys):
            order.append(("batch", tuple(keys)))
            return Promise.resolve(list(keys))

    loader = RecordingLoader()
    schema = _mutation_schema({
        "a": GraphQLField(GraphQLString, resolve=_recorded(order, "a", _resolve("a"))),
        "b": GraphQLField(
            GraphQLString, resolve=_recorded(order, "b", lambda *_: loader.load("b"))
        ),
        "c": GraphQLField(
            GraphQLString, resolve=_recorded(order, "c", lambda *_: loader.load("c"))
        ),
    })

    result = _execute(schema, "mutation { a b c }")

    assert result.data == {"a": "a", "b": "b", "c": "c"}
    assert result.errors is None
    assert list(result.data) == ["a", "b", "c"]
    assert order == [
        ("resolve", "a"),
        ("resolve", "b"),
        ("batch", ("b",)),
        ("resolve", "c"),
        ("batch", ("c",)),
    ]


def test_rejected_non_null_mutation_field_after_a_promise_nulls_the_data():
    order = []
    schema = _mutation_schema({
        "a": GraphQLField(GraphQLString, resolve=_recorded(order, "a", _resolve("a"))),
        "b": GraphQLField(GraphQLString, resolve=_recorded(order, "b", _resolve("b"))),
        "d": GraphQLField(
            GraphQLNonNull(GraphQLString),
            resolve=_recorded(order, "d", _reject("d failed")),
        ),
        "e": GraphQLField(GraphQLString, resolve=_recorded(order, "e", _resolve("e"))),
    })

    result = _execute(schema, "mutation { a b d e }")

    assert result.data is None
    assert _errors(result) == [("d failed", ["d"])]
    # A failed non-null mutation field stops the later ones from running.
    assert order == [("resolve", "a"), ("resolve", "b"), ("resolve", "d")]


def test_raising_non_null_mutation_field_after_a_promise_nulls_the_data():
    def fail(*_args):
        msg = "d raised"
        raise RuntimeError(msg)

    schema = _mutation_schema({
        "a": GraphQLField(GraphQLString, resolve=_resolve("a")),
        "d": GraphQLField(GraphQLNonNull(GraphQLString), resolve=fail),
    })

    result = _execute(schema, "mutation { a d }")

    assert result.data is None
    assert _errors(result) == [("d raised", ["d"])]


def test_rejected_non_null_first_mutation_field_skips_the_rest():
    order = []
    schema = _mutation_schema({
        "a": GraphQLField(
            GraphQLNonNull(GraphQLString),
            resolve=_recorded(order, "a", _reject("a failed")),
        ),
        "b": GraphQLField(GraphQLString, resolve=_recorded(order, "b", _resolve("b"))),
    })

    result = _execute(schema, "mutation { a b }")

    assert result.data is None
    assert _errors(result) == [("a failed", ["a"])]
    assert order == [("resolve", "a")]


def test_rejected_nullable_mutation_field_after_a_promise_is_located_at_the_field():
    schema = _mutation_schema({
        "a": GraphQLField(GraphQLString, resolve=_resolve("a")),
        "b": GraphQLField(GraphQLString, resolve=_reject("b failed")),
        "c": GraphQLField(GraphQLString, resolve=_resolve("c")),
    })

    result = _execute(schema, "mutation { a b c }")

    assert result.data == {"a": "a", "b": None, "c": "c"}
    assert _errors(result) == [("b failed", ["b"])]


def test_non_null_child_failure_in_a_later_mutation_payload_nulls_that_payload():
    payload = GraphQLObjectType(
        "Payload",
        {
            "required": GraphQLField(
                GraphQLNonNull(GraphQLString), resolve=_reject("required failed")
            ),
        },
    )
    schema = _mutation_schema({
        "a": GraphQLField(GraphQLString, resolve=_resolve("a")),
        "b": GraphQLField(payload, resolve=_resolve({})),
        "c": GraphQLField(GraphQLString, resolve=_resolve("c")),
    })

    result = _execute(schema, "mutation { a b { required } c }")

    assert result.data == {"a": "a", "b": None, "c": "c"}
    assert _errors(result) == [("required failed", ["b", "required"])]


def test_non_null_child_failure_in_a_later_non_null_payload_nulls_the_data():
    order = []
    payload = GraphQLObjectType(
        "Payload",
        {
            "required": GraphQLField(
                GraphQLNonNull(GraphQLString), resolve=_reject("required failed")
            ),
        },
    )
    schema = _mutation_schema({
        "a": GraphQLField(GraphQLString, resolve=_recorded(order, "a", _resolve("a"))),
        "b": GraphQLField(
            GraphQLNonNull(payload), resolve=_recorded(order, "b", _resolve({}))
        ),
        "c": GraphQLField(GraphQLString, resolve=_recorded(order, "c", _resolve("c"))),
    })

    result = _execute(schema, "mutation { a b { required } c }")

    assert result.data is None
    assert _errors(result) == [("required failed", ["b", "required"])]
    assert order == [("resolve", "a"), ("resolve", "b")]


# A payload whose sub-fields are two loader-backed promises completes through
# ``Promise.all``, whose promises settle their callbacks synchronously and pass
# that on down the chain. Followed by many root fields that complete without a
# promise, a chain built up front nested each of them inside the previous one
# and overflowed the stack after about 160 fields, and the swallowed
# ``RecursionError`` left the response pending for good.

LONG_ROOT_COUNT = 1000


def _execute_bounded(schema, query, timeout=30, around=None):
    """Run ``_execute`` in a thread so a response left pending fails the test.

    ``around``, if given, is called in that thread with the execution as a
    zero-argument callable and returns its result.
    """
    outcome = {}

    def execution():
        return _execute(schema, query)

    def run():
        try:
            outcome["result"] = execution() if around is None else around(execution)
        except BaseException as error:
            outcome["error"] = error

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), "the mutation never returned"
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]


def _batched_payload_schema(extra_fields):
    class EchoLoader(DataLoader):
        def batch_load_fn(self, keys):
            return Promise.resolve(list(keys))

    loader = EchoLoader()
    payload = GraphQLObjectType(
        "Payload",
        {
            "x": GraphQLField(GraphQLString, resolve=lambda *_: loader.load("x")),
            "y": GraphQLField(GraphQLString, resolve=lambda *_: loader.load("y")),
        },
    )
    return _mutation_schema({
        "a": GraphQLField(payload, resolve=lambda *_: object()),
        **extra_fields,
    })


def _aliases(prefix, selection, count=LONG_ROOT_COUNT):
    return " ".join(f"{prefix}{i}: {selection}" for i in range(count))


def test_many_plain_root_fields_after_a_batched_payload_all_resolve():
    schema = _batched_payload_schema({
        "plain": GraphQLField(GraphQLString, resolve=lambda *_: "plain"),
    })
    query = (
        "mutation { a { x y } "
        + _aliases("t", "__typename")
        + " "
        + _aliases("p", "plain")
        + " }"
    )

    result = _execute_bounded(schema, query)

    assert result.errors is None
    expected_keys = (
        ["a"]
        + [f"t{i}" for i in range(LONG_ROOT_COUNT)]
        + [f"p{i}" for i in range(LONG_ROOT_COUNT)]
    )
    assert list(result.data) == expected_keys
    assert result.data["a"] == {"x": "x", "y": "y"}
    assert result.data[f"t{LONG_ROOT_COUNT - 1}"] == "Mutation"
    assert result.data[f"p{LONG_ROOT_COUNT - 1}"] == "plain"


def test_many_raising_nullable_root_fields_after_a_batched_payload_are_located():
    def fail(*_args):
        msg = "nullable failed"
        raise RuntimeError(msg)

    schema = _batched_payload_schema({
        "fail": GraphQLField(GraphQLString, resolve=fail),
        "last": GraphQLField(GraphQLString, resolve=lambda *_: "last"),
    })
    query = "mutation { a { x y } " + _aliases("f", "fail") + " last }"

    result = _execute_bounded(schema, query)

    assert len(result.data) == LONG_ROOT_COUNT + 2
    assert result.data["last"] == "last"
    assert result.data["f0"] is None
    assert len(result.errors) == LONG_ROOT_COUNT
    assert sorted(error.path[0] for error in result.errors) == sorted(
        f"f{i}" for i in range(LONG_ROOT_COUNT)
    )


def test_raising_non_null_root_after_many_plain_root_fields_nulls_the_data():
    order = []

    def fail(*_args):
        msg = "required failed"
        raise RuntimeError(msg)

    schema = _batched_payload_schema({
        "required": GraphQLField(GraphQLNonNull(GraphQLString), resolve=fail),
        "after": GraphQLField(
            GraphQLString, resolve=_recorded(order, "after", lambda *_: "after")
        ),
    })
    query = "mutation { a { x y } " + _aliases("t", "__typename") + " required after }"

    result = _execute_bounded(schema, query)

    assert result.data is None
    assert _errors(result) == [("required failed", ["required"])]
    assert order == []


def test_many_batched_payload_root_fields_resolve_in_document_order():
    order = []

    class RecordingLoader(DataLoader):
        def batch_load_fn(self, keys):
            order.append(tuple(keys))
            return Promise.resolve(list(keys))

    loader = RecordingLoader()
    payload = GraphQLObjectType(
        "Payload",
        {
            "x": GraphQLField(
                GraphQLString, resolve=lambda obj, _info: loader.load(f"{obj}.x")
            ),
            "y": GraphQLField(
                GraphQLString, resolve=lambda obj, _info: loader.load(f"{obj}.y")
            ),
        },
    )
    schema = _mutation_schema({
        "a": GraphQLField(
            payload,
            args={"n": GraphQLArgument(GraphQLString)},
            resolve=lambda _obj, _info, n: Promise.resolve(n),
        ),
    })
    query = (
        "mutation { "
        + " ".join(f'a{i}: a(n: "{i}") {{ x y }}' for i in range(LONG_ROOT_COUNT))
        + " }"
    )

    result = _execute_bounded(schema, query)

    assert result.errors is None
    assert list(result.data) == [f"a{i}" for i in range(LONG_ROOT_COUNT)]
    assert result.data[f"a{LONG_ROOT_COUNT - 1}"] == {
        "x": f"{LONG_ROOT_COUNT - 1}.x",
        "y": f"{LONG_ROOT_COUNT - 1}.y",
    }
    # One batch per root field, in document order: no field ran early.
    assert order == [(f"{i}.x", f"{i}.y") for i in range(LONG_ROOT_COUNT)]


def test_many_promise_root_fields_inside_a_running_tick_resolve_in_linear_time():
    # Executed from a promise callback, the walk runs inside a trampoline tick
    # that is already in use, and ``get`` drains the queue one callback at a
    # time, resolving the returned promise before each. A walk that returned a
    # promise per pass built a chain of promises following one another, walked
    # before every callback: quadratic, about 12 s for 5,000 roots and about
    # four times that for the 10,000 here, against well under a second now.
    schema = _mutation_schema({
        "settled": GraphQLField(GraphQLString, resolve=_resolve("settled")),
    })
    count = 10 * LONG_ROOT_COUNT
    query = "mutation { " + _aliases("s", "settled", count) + " }"

    result = _execute_bounded(
        schema,
        query,
        timeout=10,
        around=lambda execution: (
            Promise.resolve(None).then(lambda _: execution()).get()
        ),
    )

    assert result.errors is None
    assert list(result.data) == [f"s{i}" for i in range(count)]


def test_many_loader_root_fields_with_the_trampoline_off_all_resolve():
    # With the trampoline off every callback runs at once, so a field's
    # promise has often settled before ``execute_field`` returns it. Continuing
    # the walk from that promise's callback nested one pass per field.
    class EchoLoader(DataLoader):
        def batch_load_fn(self, keys):
            return Promise.resolve(list(keys))

    loader = EchoLoader()
    schema = _mutation_schema({
        "loaded": GraphQLField(GraphQLString, resolve=lambda *_: loader.load("loaded")),
    })
    query = "mutation { " + _aliases("l", "loaded") + " }"

    def without_trampoline(execution):
        # ``async_instance`` is thread-local: this only affects the test thread.
        async_instance.disable_trampoline()
        try:
            return execution()
        finally:
            async_instance.enable_trampoline()

    result = _execute_bounded(schema, query, around=without_trampoline)

    assert result.errors is None
    assert list(result.data) == [f"l{i}" for i in range(LONG_ROOT_COUNT)]
