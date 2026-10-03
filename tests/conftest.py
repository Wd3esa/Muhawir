"""Shared test setup.

Most tests use fake models that only know the understanding and answer steps. The second
reading (check_support) is accepted by default here; tests marked `real_check` exercise it.
"""
import pytest

from muhawir.generate import ModelGenerator


def pytest_configure(config):
    config.addinivalue_line("markers", "real_check: run ModelGenerator.check_support for real")


@pytest.fixture(autouse=True)
def _accept_support_check(request, monkeypatch):
    if "real_check" not in request.keywords:
        monkeypatch.setattr(ModelGenerator, "check_support", lambda self, claims, passages: [True] * len(claims))
