from fedlex_bench.types import Usage


def test_usage_adds_all_fields():
    a = Usage(1, 2, 3, 0.5, 1.0, 1)
    b = Usage(10, 20, 30, 0.25, 2.0, 2)
    assert a + b == Usage(11, 22, 33, 0.75, 3.0, 3)
