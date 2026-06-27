"""Smoke test: the package imports and exposes a version."""

def test_package_imports():
    import ecoforecast
    assert ecoforecast.__version__
