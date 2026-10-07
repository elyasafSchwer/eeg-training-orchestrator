#!/usr/bin/env python3
"""Orchestrator: launch a training job on the slice of EEG data you choose.

Selects the visits matching the metadata filters, starts the data provider
container on the requested ports, and runs the training (consumer) container
against it. Example:

    python run_training.py --filter gender=female --filter 'age>20' --ports 5001,5002
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import filters

HERE = Path(__file__).resolve().parent
PROVIDER_IMAGE = PROVIDER_CONTAINER = "eeg-provider"
CONSUMER_IMAGE = CONSUMER_CONTAINER = "ml-consumer"
PROVIDER_READY = "provider ready"  # logged by provider.py once all ports are listening


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="EXPR",
        help="metadata filter, repeatable: gender=female, 'age>20', wears_glasses=false, "
             "'date_of_visit>=2024-06-01'. Operators: = != > >= < <=",
    )
    parser.add_argument("--ports", default="5001,5002", help="ports the provider serves (default: 5001,5002)")
    parser.add_argument("--data-dir", type=Path, default=HERE.parent / "data", help="EEG data directory")
    parser.add_argument("--provider-dir", type=Path, default=HERE.parent / "provider", help="provider source directory")
    parser.add_argument("--consumer-dir", type=Path, default=HERE.parent / "consumer", help="consumer source directory")
    parser.add_argument("--dry-run", action="store_true", help="show the matching visits and exit")
    return parser.parse_args()


def say(message):
    """Our own progress lines, flushed so they stay in order with the container output."""
    print(message, flush=True)


def docker(*args, **kwargs):
    return subprocess.run(["docker", *args], check=True, text=True, **kwargs)


def visit_db(visits):
    """The lookup the consumer validates against, derived from the visits we are about to send."""
    return {visit.visit_id: visit.metadata for visit in visits}


def build_images(provider_dir, consumer_dir, visits):
    say("==> building images")
    docker("build", "-t", PROVIDER_IMAGE, str(provider_dir))
    with tempfile.TemporaryDirectory() as temporary:
        # The consumer's Dockerfile copies in a visit_db.json. Rather than trust a checked-in
        # snapshot of one, build the image from a context holding a freshly generated database.
        context = Path(temporary) / "consumer"
        shutil.copytree(consumer_dir, context)
        (context / "visit_db.json").write_text(json.dumps(visit_db(visits), indent=2))
        docker("build", "-t", CONSUMER_IMAGE, str(context))


def start_provider(ports, visit_ids, data_dir):
    say(f"==> starting provider on ports {','.join(ports)}")
    subprocess.run(["docker", "rm", "-f", PROVIDER_CONTAINER],  # a leftover from an earlier run
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    published = [arg for port in ports for arg in ("-p", f"{port}:{port}")]
    docker(
        "run", "-d", "--name", PROVIDER_CONTAINER,
        *published,
        "-v", f"{data_dir.resolve()}:/data:ro",
        PROVIDER_IMAGE,
        "--ports", ",".join(ports),
        "--visits", ",".join(visit_ids),
        stdout=subprocess.DEVNULL,
    )


def provider_logs():
    done = subprocess.run(["docker", "logs", PROVIDER_CONTAINER], capture_output=True, text=True)
    return done.stdout + done.stderr


def wait_for_provider(timeout=90):
    """The consumer connects once and does not retry, so the ports must be listening first."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        logs = provider_logs()
        if PROVIDER_READY in logs:
            return
        state = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", PROVIDER_CONTAINER],
            capture_output=True, text=True,
        ).stdout.strip()
        if state != "running":
            raise RuntimeError(f"provider is {state or 'gone'} before it was ready:\n{logs}")
        time.sleep(0.2)
    raise RuntimeError(f"provider was not ready within {timeout}s:\n{provider_logs()}")


def stream(command, prefix):
    """Run a command, mirroring its output to our stdout with a prefix."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def pump():
        for line in process.stdout:
            print(f"[{prefix}] {line}", end="", flush=True)

    threading.Thread(target=pump, daemon=True).start()
    return process


def main():
    args = parse_args()
    try:
        visit_filters = [filters.parse_filter(expr) for expr in args.filter]
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    if not args.data_dir.is_dir():
        print(f"no dataset at {args.data_dir} — it is too large for the repository, so copy "
              f"it there or point --data-dir at it", file=sys.stderr)
        return 2
    visits = filters.select_visits(args.data_dir, visit_filters)

    criteria = " and ".join(str(f) for f in visit_filters) or "no filter"
    say(f"==> {len(visits)} visits match {criteria}")
    if args.dry_run:
        for visit in visits:
            say(f"    {visit.visit_id}  {visit.metadata}")
        return 0
    if not visits:
        print("nothing to train on", file=sys.stderr)
        return 1

    ports = args.ports.split(",")
    try:
        build_images(args.provider_dir, args.consumer_dir, visits)
        start_provider(ports, [visit.visit_id for visit in visits], args.data_dir)
    except subprocess.CalledProcessError as error:
        print(f"docker failed: {' '.join(error.cmd)} — see its output above", file=sys.stderr)
        return 1
    try:
        wait_for_provider()
        stream(["docker", "logs", "-f", PROVIDER_CONTAINER], "provider")
        say("==> starting training job")
        # The consumer connects to localhost, so it shares the provider's network namespace.
        consumer = stream(
            ["docker", "run", "--rm", "--name", CONSUMER_CONTAINER,
             "--network", f"container:{PROVIDER_CONTAINER}",
             "-e", f"CONSUMER_PORTS={','.join(ports)}",
             CONSUMER_IMAGE],
            "consumer",
        )
        return consumer.wait()
    except KeyboardInterrupt:
        say("\n==> interrupted")
        return 130
    finally:
        say("==> cleaning up")
        subprocess.run(["docker", "rm", "-f", PROVIDER_CONTAINER, CONSUMER_CONTAINER],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    sys.exit(main())
