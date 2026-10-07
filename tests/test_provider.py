"""Unit tests for the chunk plan and for the wire format the consumer depends on."""

import logging
import queue
import random
import socket
import sys
import threading
import unittest
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "provider"))

import provider  # noqa: E402

CHANNELS = 4
CHUNK_BYTES = provider.CHUNK_ROWS * CHANNELS * 4  # 10,000 samples x 4 channels x float32


def setUpModule():
    logging.disable(logging.CRITICAL)  # the provider logs per connection; keep test output readable


def tearDownModule():
    logging.disable(logging.NOTSET)


class ProviderTestCase(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.data = Path(temporary.name)
        provider._eeg.cache_clear()  # recordings are memo-ised by path
        self.addCleanup(provider._eeg.cache_clear)

    def write_recording(self, visit_id, rows):
        """A recording whose values encode their own position, so a chunk identifies itself."""
        eeg = np.arange(rows * CHANNELS, dtype=np.float32).reshape(rows, CHANNELS)
        np.save(self.data / f"{visit_id}.npy", eeg)
        return eeg


class ChunkPlan(ProviderTestCase):
    def test_covers_every_chunk_of_every_visit_exactly_once(self):
        visit_ids = [str(uuid.uuid4()) for _ in range(3)]
        for visit_id in visit_ids:
            self.write_recording(visit_id, provider.CHUNK_ROWS * 4)

        plan = provider.build_chunk_plan(self.data, visit_ids)

        expected = [
            (visit_id, start)
            for visit_id in visit_ids
            for start in range(0, 4 * provider.CHUNK_ROWS, provider.CHUNK_ROWS)
        ]
        self.assertEqual(len(plan), 12)
        self.assertEqual(sorted(plan), sorted(expected))  # no duplicates, nothing missing

    def test_drops_a_trailing_partial_chunk(self):
        """The consumer reads fixed 10s blocks, so a leftover tail cannot be sent."""
        visit_id = str(uuid.uuid4())
        self.write_recording(visit_id, provider.CHUNK_ROWS * 2 + 1234)

        plan = provider.build_chunk_plan(self.data, [visit_id])

        self.assertEqual(sorted(start for _, start in plan), [0, provider.CHUNK_ROWS])

    def test_order_is_shuffled(self):
        random.seed(1234)  # seeded, so this cannot fail on an unlucky shuffle
        visit_ids = [str(uuid.uuid4()) for _ in range(2)]
        for visit_id in visit_ids:
            self.write_recording(visit_id, provider.CHUNK_ROWS * 20)

        plan = provider.build_chunk_plan(self.data, visit_ids)

        self.assertNotEqual(plan, sorted(plan))


class WireFormat(ProviderTestCase):
    def test_chunk_is_the_visit_uuid_followed_by_its_float32_block(self):
        visit_id = str(uuid.uuid4())
        eeg = self.write_recording(visit_id, provider.CHUNK_ROWS * 2)
        chunks = queue.Queue()
        for start in (provider.CHUNK_ROWS, 0):  # fixed order: this asserts content, not shuffling
            chunks.put((visit_id, start))

        listener = socket.create_server(("127.0.0.1", 0))
        self.addCleanup(listener.close)
        server = threading.Thread(target=provider.serve, args=(listener, chunks, self.data), daemon=True)
        server.start()
        client = socket.create_connection(listener.getsockname())
        self.addCleanup(client.close)

        received = [self.read_chunk(client) for _ in range(2)]
        server.join(timeout=10)

        self.assertEqual([visit for visit, _ in received], [visit_id, visit_id])
        np.testing.assert_array_equal(received[0][1], eeg[provider.CHUNK_ROWS:])
        np.testing.assert_array_equal(received[1][1], eeg[:provider.CHUNK_ROWS])
        self.assertTrue(chunks.empty())
        self.assertEqual(client.recv(1), b"")  # the provider hangs up once the plan is exhausted
        self.assertFalse(server.is_alive())

    def read_chunk(self, sock):
        raw = self.read_exactly(sock, 16 + CHUNK_BYTES)
        visit_id = str(uuid.UUID(bytes=raw[:16]))
        return visit_id, np.frombuffer(raw[16:], dtype=np.float32).reshape(-1, CHANNELS)

    def read_exactly(self, sock, size):
        buffer = b""
        while len(buffer) < size:
            part = sock.recv(size - len(buffer))
            self.assertTrue(part, "connection closed before the chunk was complete")
            buffer += part
        return buffer


if __name__ == "__main__":
    unittest.main()
