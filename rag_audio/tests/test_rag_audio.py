"""Tests for the main module."""

from rag_audio import __version__


def test_version():
    """Check that the version is acceptable."""
    assert isinstance(__version__, str)
