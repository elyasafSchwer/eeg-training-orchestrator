#!/usr/bin/env python3
"""Data provider: serves the EEG data of the selected visits over TCP as random 10s chunks.

One chunk on the wire is the 16-byte visit UUID followed by 40,000 float32 values
(10,000 samples x 4 channels). Every chunk of every selected visit is sent exactly
once, in random order, spread over the ports; each port serves one consumer.
"""

import argparse
import logging
import queue
import random
import socket
import threading
import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np

SAMPLE_RATE = 1000
CHUNK_SECONDS = 10
CHUNK_ROWS = SAMPLE_RATE * CHUNK_SECONDS

# The orchestrator waits for this line before starting the consumer, which does not retry.
READY_MARKER = "provider ready"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@lru_cache(maxsize=None)
def _eeg(path):
    """Memory-mapped EEG recording, so chunks are read straight off disk."""
    return np.load(path, mmap_mode="r")


def build_chunk_plan(data_dir, visit_ids):
    """Every (visit_id, start_row) chunk of the selected visits, in random order."""
    plan = []
    for visit_id in visit_ids:
        rows = len(_eeg(str(Path(data_dir) / f"{visit_id}.npy")))
        plan += [(visit_id, start) for start in range(0, rows - CHUNK_ROWS + 1, CHUNK_ROWS)]
    random.shuffle(plan)
    return plan


def serve(listener, chunks, data_dir):
    """Accept one consumer and send chunks until the shared plan is exhausted."""
    port = listener.getsockname()[1]
    conn, address = listener.accept()
    logging.info(f"port {port}: consumer connected from {address[0]}:{address[1]}")
    sent = 0
    with conn:
        while True:
            try:
                visit_id, start = chunks.get_nowait()
            except queue.Empty:
                break
            eeg = _eeg(str(Path(data_dir) / f"{visit_id}.npy"))
            chunk = eeg[start:start + CHUNK_ROWS]
            try:
                conn.sendall(uuid.UUID(visit_id).bytes + chunk.tobytes())
            except (BrokenPipeError, ConnectionResetError):
                logging.warning(f"port {port}: consumer went away after {sent} chunks")
                return
            sent += 1
    logging.info(f"port {port}: sent {sent} chunks, closing")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ports", required=True, help="comma-separated TCP ports to serve")
    parser.add_argument("--visits", required=True, help="comma-separated visit ids to stream")
    parser.add_argument("--data-dir", default="/data", help="directory holding <visit_id>.npy (default: /data)")
    args = parser.parse_args()

    ports = [int(port) for port in args.ports.split(",")]
    visit_ids = args.visits.split(",")

    chunks = queue.Queue()
    for chunk in build_chunk_plan(args.data_dir, visit_ids):
        chunks.put(chunk)
    logging.info(
        f"{chunks.qsize()} chunks of {CHUNK_SECONDS}s from {len(visit_ids)} visits, serving on ports {ports}"
    )

    # Bind every port before announcing readiness.
    listeners = [socket.create_server(("", port)) for port in ports]
    logging.info(READY_MARKER)

    threads = [threading.Thread(target=serve, args=(listener, chunks, args.data_dir)) for listener in listeners]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    unsent = chunks.qsize()
    logging.info(f"{unsent} chunks left unsent" if unsent else "all data sent")


if __name__ == "__main__":
    main()
