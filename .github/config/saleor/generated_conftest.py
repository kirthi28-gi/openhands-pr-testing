"""Trusted fixture bridge for OpenHands-generated Saleor tests.

The repository root ``conftest.py`` prepares Django settings, test databases
and Saleor's core fixtures. This trusted conftest makes the same fixture
modules available to tests generated under ``tests/generated/``. Generated
tests may request these fixtures, but can never modify this file in the
sandbox.
"""

import pytest  # noqa: F401

pytest_plugins = [
    "saleor.tests.fixtures",
    "saleor.product.tests.fixtures",
    "saleor.graphql.tests.fixtures",
]
