"""Unit tests for the visit database, the orchestrator's readiness wait, and its marker.

The rest of run_training.py is a thin shell around the docker CLI, covered by actually
running it rather than by mocks.
"""

import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "orchestrator"), str(ROOT / "provider")]

import filters  # noqa: E402
import provider  # noqa: E402
import run_training  # noqa: E402


def docker_said(stdout):
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


class VisitDatabase(unittest.TestCase):
    """What the consumer validates against: derived from the dataset, never checked in."""

    def test_holds_exactly_the_selected_visits_keyed_by_id(self):
        selected = [
            filters.Visit("aaa", Path("aaa.npy"), {"visit_id": "aaa", "age": 30}),
            filters.Visit("bbb", Path("bbb.npy"), {"visit_id": "bbb", "age": 70}),
        ]

        database = run_training.visit_db(selected)

        self.assertEqual(sorted(database), ["aaa", "bbb"])  # nothing the job will not stream
        self.assertEqual(database["bbb"], {"visit_id": "bbb", "age": 70})

    def test_is_empty_rather_than_stale_when_nothing_is_selected(self):
        self.assertEqual(run_training.visit_db([]), {})


class ReadyMarker(unittest.TestCase):
    def test_orchestrator_waits_for_the_line_the_provider_actually_logs(self):
        """The string is spelled out in both modules so the orchestrator needs no numpy."""
        self.assertEqual(run_training.PROVIDER_READY, provider.READY_MARKER)


class WaitForProvider(unittest.TestCase):
    """The consumer connects once and never retries, so this wait must be trustworthy."""

    def test_returns_once_the_marker_appears(self):
        logs = f"starting up\n{run_training.PROVIDER_READY}\n"
        with mock.patch.object(run_training, "provider_logs", return_value=logs):
            run_training.wait_for_provider(timeout=1)  # must not raise

    def test_fails_immediately_when_the_container_died(self):
        with mock.patch.object(run_training, "provider_logs", return_value="Traceback: boom"), \
             mock.patch.object(run_training.subprocess, "run", return_value=docker_said("exited\n")):
            with self.assertRaises(RuntimeError) as raised:
                run_training.wait_for_provider(timeout=30)  # must not wait those 30s

        self.assertIn("exited", str(raised.exception))
        self.assertIn("boom", str(raised.exception))  # the provider's own logs are surfaced

    def test_gives_up_if_a_running_container_never_announces_itself(self):
        with mock.patch.object(run_training, "provider_logs", return_value=""), \
             mock.patch.object(run_training.subprocess, "run", return_value=docker_said("running\n")):
            with self.assertRaises(RuntimeError) as raised:
                run_training.wait_for_provider(timeout=0.3)

        self.assertIn("not ready", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
