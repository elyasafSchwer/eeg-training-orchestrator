# consumer.py
"""
Consumer process for AI Platform Engineering assignment.
Connects to a list of TCP ports, receives numpy float32 arrays (10,000 samples),
and prints stats on the incoming data rate periodically.
"""
import socket
import threading
import numpy as np
import time
import os
import sys
import logging
import json
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

CHANNELS = 4
SECONDS = 10  # default, but not used for reading
SAMPLE_RATE = 1000
DTYPE = np.float32
UUID_SIZE = 16  # bytes
ARRAY_SHAPE = (SAMPLE_RATE * CHANNELS * SECONDS, )  # unknown, but we will infer from data size
STATS_INTERVAL = 10  # seconds

# Metadata fields expected in visit_db.json
REQUIRED_FIELDS = [
    "person_id",
    "visit_id",
    "person_name",
    "age",
    "gender",
    "wears_glasses",
    "date_of_visit",
    "dominant_hand"
]

VISIT_DB = None

def read_from_port(port, stats):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.connect(('localhost', port))
    logging.info(f"[Thread-{port}] Connected to port {port}")
    # arrays_received = 0
    global VISIT_DB
    while True:
        # First, read UUID (16 bytes)
        uuid_bytes = b''
        while len(uuid_bytes) < UUID_SIZE:
            chunk = s.recv(UUID_SIZE - len(uuid_bytes))
            if not chunk:
                logging.info(f"Connection closed on port {port}")
                s.close()
                return
            uuid_bytes += chunk
        visit_id = uuid_bytes.hex()

        # Now read the rest of the data (unknown size, but let's assume 4 channels, 1000Hz, 10s = 40,000 float32)
        expected_data_bytes = SAMPLE_RATE * CHANNELS * SECONDS * np.dtype(DTYPE).itemsize
        data = b''
        while len(data) < expected_data_bytes:
            chunk = s.recv(expected_data_bytes - len(data))
            if not chunk:
                logging.info(f"Connection closed on port {port}")
                s.close()
                return
            data += chunk
        arr = np.frombuffer(data, dtype=DTYPE)
        try:
            arr.reshape((SAMPLE_RATE * SECONDS, CHANNELS))
        except Exception as e:
            logging.warning(f"Could not reshape data for visit {visit_id}: {e}")

        # --- Validation ---
        if VISIT_DB is not None:
            if visit_id not in VISIT_DB:
                logging.warning(f"Visit ID {visit_id} not found in visit_db.json!")
            else:
                meta = VISIT_DB[visit_id]
                missing = [f for f in REQUIRED_FIELDS if f not in meta]
                if missing:
                    logging.warning(f"Visit ID {visit_id} metadata missing fields: {missing}")

        logging.info(f"Received visit_id {visit_id} with shape {arr.shape} on port {port}")
        stats[port]['count'] += 1
        stats[port]['last_time'] = time.time()
        # simulate some processing
        time.sleep(0.1)


def stats_printer(stats, ports):
    while True:
        time.sleep(STATS_INTERVAL)
        logging.info("--- Data Rate Stats ---")
        for port in ports:
            count = stats[port]['count']
            last_time = stats[port]['last_time']
            logging.info(f"Port {port}: {count} arrays received in last {STATS_INTERVAL}s (last at {last_time:.2f})")
            stats[port]['count'] = 0
        logging.info("----------------------")


def main():
    # Get ports from env or args
    ports_env = os.getenv('CONSUMER_PORTS')
    if ports_env:
        ports = [int(p) for p in ports_env.split(',')]
    else:
        if len(sys.argv) < 2:
            print("Usage: python consumer.py <port1> [<port2> ...]")
            sys.exit(1)
        ports = [int(p) for p in sys.argv[1:]]

    global VISIT_DB
    # Try to load visit_db.json from current directory or parent
    db_path = os.getenv('VISIT_DB_PATH', 'visit_db.json')
    if not os.path.exists(db_path):
        logging.warning("visit_db.json not found. Validation will be skipped.")
    else:
        with open(db_path, 'r') as f:
            VISIT_DB = json.load(f)
        logging.info(f"Loaded visit_db.json with {len(VISIT_DB)} visits from {db_path}")
    
    logging.info(f"[Main] Starting consumer. Will connect to ports: {ports}")
    stats = {port: {'count': 0, 'last_time': 0} for port in ports}
    threads = []
    for port in ports:
        logging.info(f"[Main] Initializing thread for port {port}")
        t = threading.Thread(target=read_from_port, args=(port, stats), daemon=True)
        t.start()
        threads.append(t)

    logging.info("[Main] Starting stats printer thread.")
    stats_thread = threading.Thread(target=stats_printer, args=(stats, ports), daemon=True)
    stats_thread.start()

    # Wait for threads to finish
    for t in threads:
        t.join()

if __name__ == "__main__":
    main()
