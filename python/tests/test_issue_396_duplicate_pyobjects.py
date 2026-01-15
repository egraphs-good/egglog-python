"""
Test for issue #396: Duplicate Python Values

This test demonstrates that cloudpickle can produce different byte sequences
for the same object when pickled by-value, particularly for classes defined
in __main__, which results in duplicate egglog PyObject nodes.
"""

from __future__ import annotations

import cloudpickle
import pytest

from egglog import *


def test_duplicate_pyobjects_same_class_instance():
    """
    Test that demonstrates the duplicate PyObject issue.

    When a class is defined in __main__ (or in this test module), cloudpickle
    serializes it by value rather than by reference. According to issue #396,
    this can produce different byte sequences for the same object across
    different pickle calls, leading to duplicate nodes in the e-graph.

    EXPECTED BEHAVIOR: The same Python object should create equal PyObject nodes
    ACTUAL BEHAVIOR: May create duplicate nodes if cloudpickle produces different bytes

    Note: This test may pass in simple cases where cloudpickle is deterministic,
    but the issue can still occur in real-world scenarios where:
    - Classes are defined in __main__
    - Objects are pickled at different times/contexts
    - There are closures or complex nested structures
    """

    # Define a simple class in the test module (acts like __main__)
    class MyClass:
        def __init__(self, value):
            self.value = value

        def __eq__(self, other):
            return isinstance(other, MyClass) and self.value == other.value

    egraph = EGraph()

    # Create the SAME Python object
    obj = MyClass(42)

    # Pickle the same object multiple times to check for determinism
    pickled1 = cloudpickle.dumps(obj)
    pickled2 = cloudpickle.dumps(obj)

    # Report on pickle determinism
    print(f"\nCloudpickle determinism check:")
    print(f"  First pickle length: {len(pickled1)}")
    print(f"  Second pickle length: {len(pickled2)}")
    print(f"  Bytes are identical: {pickled1 == pickled2}")

    if pickled1 != pickled2:
        print(f"  ⚠ Cloudpickle produced DIFFERENT bytes for the same object!")
    else:
        print(f"  ✓ Cloudpickle is deterministic in this case")

    # Create PyObject expressions from the same object instance multiple times
    # In an ideal world, these should be recognized as equal
    py_obj1 = egraph.let("obj1", PyObject(obj))
    py_obj2 = egraph.let("obj2", PyObject(obj))

    egraph.run(10)

    # Check if the e-graph recognizes them as equal
    # This should pass, but might fail if cloudpickle produced different bytes
    try:
        egraph.check(eq(py_obj1).to(py_obj2))
        print("  ✓ E-graph correctly identifies objects as equal")
    except Exception as e:
        print(f"  ✗ E-graph treats objects as DIFFERENT - duplicates detected!")
        print(f"  Error: {e}")
        pytest.fail(
            f"Duplicate PyObject nodes detected for the same Python object.\n"
            f"This means cloudpickle produced different bytes for identical objects.\n"
            f"Error: {e}"
        )


def test_duplicate_pyobjects_nested_class():
    """
    Test with nested class definitions which are even more likely
    to be serialized by-value and produce different bytes.
    """

    class OuterClass:
        class InnerClass:
            def __init__(self, x):
                self.x = x

    egraph = EGraph()

    obj = OuterClass.InnerClass(100)

    # Create the same object in the egraph multiple times
    py_obj1 = egraph.let("nested1", PyObject(obj))
    py_obj2 = egraph.let("nested2", PyObject(obj))

    # Check if they're equal
    egraph.run(10)

    try:
        egraph.check(eq(py_obj1).to(py_obj2))
        print("✓ Nested objects are equal - no duplicates")
    except Exception as e:
        print(f"✗ Nested objects are NOT equal - duplicates detected!")
        pytest.fail(f"Duplicate PyObject nodes for nested class: {e}")


def test_pyobject_with_closure():
    """
    Test that functions with closures (which must be serialized by-value)
    can create duplicate PyObject nodes.
    """

    def make_adder(x):
        def adder(y):
            return x + y
        return adder

    egraph = EGraph()

    add_five = make_adder(5)

    # Create the same closure in the egraph multiple times
    py_fn1 = egraph.let("fn1", PyObject(add_five))
    py_fn2 = egraph.let("fn2", PyObject(add_five))

    egraph.run(10)

    try:
        egraph.check(eq(py_fn1).to(py_fn2))
        print("✓ Closures are equal - no duplicates")
    except Exception as e:
        print(f"✗ Closures are NOT equal - duplicates detected!")
        pytest.fail(f"Duplicate PyObject nodes for closure: {e}")


if __name__ == "__main__":
    # Run the tests manually to see output
    print("Testing duplicate PyObjects with same class instance...")
    test_duplicate_pyobjects_same_class_instance()

    print("\nTesting duplicate PyObjects with nested class...")
    test_duplicate_pyobjects_nested_class()

    print("\nTesting duplicate PyObjects with closure...")
    test_pyobject_with_closure()
