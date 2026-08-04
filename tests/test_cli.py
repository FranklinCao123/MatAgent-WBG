"""Tests for portable command-line output."""

import unittest
from unittest.mock import patch

from matagent.cli import _configure_utf8_output


class RecordingStream:
    def __init__(self) -> None:
        self.encoding_requested = None

    def reconfigure(self, *, encoding: str) -> None:
        self.encoding_requested = encoding


class CLIOutputTests(unittest.TestCase):
    def test_scientific_unicode_uses_utf8_streams(self) -> None:
        stdout = RecordingStream()
        stderr = RecordingStream()

        with patch("matagent.cli.sys.stdout", stdout), patch(
            "matagent.cli.sys.stderr", stderr
        ):
            _configure_utf8_output()

        self.assertEqual(stdout.encoding_requested, "utf-8")
        self.assertEqual(stderr.encoding_requested, "utf-8")


if __name__ == "__main__":
    unittest.main()
