"""Shared pytest configuration: registers the 'slow' marker (deselect with -m "not slow")."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: long-running reproduction test on cached real matches (deselect with -m \"not slow\")")
