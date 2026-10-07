# EEG Training Orchestrator

Launches a deep-learning training job on a chosen slice of the EEG data factory.

You give it a metadata filter; it selects the matching visits, starts a **data provider**
that streams their EEG data over TCP as random 10-second chunks, and runs the **training
job** (the provided consumer) against it.

```
eeg-training-orchestrator/
├── orchestrator/
│   ├── run_training.py       # the orchestrator CLI — start here
│   └── filters.py            # metadata filter expressions -> list of visits
├── provider/                 # everything that goes into the provider image
│   ├── provider.py           # TCP data provider
│   ├── Dockerfile
│   └── requirements.txt      # numpy (provider only; the orchestrator is pure stdlib)
├── tests/                    # unit tests — no containers, no dataset needed
├── consumer/                 # given: the training job it feeds
└── data/                     # the EEG dataset — not in git, see below
```

The orchestrator finds its three siblings relative to its own location, so it runs from a
fresh clone with no setup; `--data-dir`, `--provider-dir` and `--consumer-dir` override them.

## Requirements

- Docker or compatible runtime (tested with Rancher Desktop / `docker` CLI)
- Python 3.9+ to run the orchestrator itself — standard library only
- **The dataset.** The 99 recordings are ~1.9 GB, too large for the repository, so `data/`
  is git-ignored. Unpack the given dataset into `data/` next to the code, or point
  `--data-dir` wherever you keep it. It is the single source of truth: every visit, every
  filter decision and the consumer's `visit_db.json` are derived from it, so nothing else
  needs to be supplied.

## Run a training job

```bash
cd orchestrator
python3 run_training.py --filter gender=female --filter age>20
```

That builds both images, starts the provider on ports 5001 and 5002, waits until it is
listening, starts the training job, and streams both log streams until all the data has
been sent (abridged output of a run over the 8 visits with `age>74`):

```
==> 8 visits match age>74
==> building images
==> starting provider on ports 5001,5002
==> starting training job
[provider] [2026-10-02 00:24:11] INFO 960 chunks of 10s from 8 visits, serving on ports [5001, 5002]
[provider] [2026-10-02 00:24:11] INFO provider ready
[provider] [2026-10-02 00:24:12] INFO port 5001: consumer connected from 127.0.0.1:56576
[consumer] [2026-10-02 00:24:13] INFO Received visit_id 6939a610db574d98b7eb0fba7fa7ecbd with shape (40000,) on port 5001
[consumer] [2026-10-02 00:24:22] INFO --- Data Rate Stats ---
[consumer] [2026-10-02 00:24:22] INFO Port 5001: 99 arrays received in last 10s (last at 1790900662.16)
[consumer] [2026-10-02 00:24:22] INFO Port 5002: 99 arrays received in last 10s (last at 1790900662.16)
[consumer] [2026-10-02 00:24:22] INFO ----------------------
[provider] [2026-10-02 00:24:56] INFO port 5002: sent 479 chunks, closing
[provider] [2026-10-02 00:24:57] INFO port 5001: sent 481 chunks, closing
[provider] [2026-10-02 00:24:57] INFO all data sent
==> cleaning up
```

`Ctrl-C` stops the job early (exit code 130); either way both containers are removed on exit.

### Options

| Option | Meaning |
| --- | --- |
| `--filter EXPR` | Metadata filter, repeatable — all filters must match (AND) |
| `--ports` | Ports the provider serves, one consumer per port (default `5001,5002`) |
| `--dry-run` | Print the matching visits and their metadata, then exit |
| `--data-dir`, `--provider-dir`, `--consumer-dir` | Override the default paths |

### Filter syntax

`<metadata field><operator><value>`, with operators `=`, `!=`, `>`, `>=`, `<`, `<=`:

```bash
--filter gender=female            # any field from the visit JSON
--filter age>=20 --filter age<40
--filter wears_glasses=false      # true/false
--filter dominant_hand!=right
--filter date_of_visit>=2024-06-01   # ISO dates compare correctly as strings
```

Quote expressions containing `>` or `<` so the shell does not read them as redirection.
Values are typed to match the metadata (bool, number, string). A visit whose metadata
lacks the field never matches — a few visits in the dataset have no `age`/`gender`.
No filter at all selects everything.

Check a selection before spending GPU time on it:

```bash
python3 run_training.py --dry-run --filter age>70 --filter gender=female
```

## How it works

**Chunking.** Each recording is `(1,200,000, 4)` float32 — 1200 s at 1 kHz over 4
channels, i.e. 120 chunks of 10 s. The provider builds the list of all
`(visit_id, start_row)` chunks across the selected visits, shuffles it, and serves it
from one queue shared by all ports. Every chunk is therefore sent exactly once, in random
order, and the ports divide the work between them as fast as each consumer drains it.
Recordings are memory-mapped, so a job over the whole dataset does not load 1.9 GB into RAM.

**Wire format,** per chunk, exactly what the consumer reads:

| Bytes | Content |
| --- | --- |
| 16 | visit ID — UUIDv4, big-endian 128 bits |
| 160,000 | 40,000 float32: 10,000 samples × 4 channels, row-major |

**Lifecycle.** The consumer connects once and does not retry, so the orchestrator waits
for the provider's `provider ready` line (logged only after every port is bound) before
starting it. The consumer also dials `localhost`, so its container joins the provider's
network namespace (`--network container:eeg-provider`) instead of using a bridge network.
The provider's ports are also published to the host, which is what lets you run a
consumer outside the container too. When the chunk plan is exhausted the provider closes
the connections, the consumer's threads exit, and the orchestrator cleans up both containers.

## Running the provider without containers

Useful for development — the provider is a normal script (needs `numpy`):

```bash
# terminal 1, from orchestrator/
python3 ../provider/provider.py --ports 5001 --visits <visit_id>,<visit_id> --data-dir ../data
# terminal 2
cd ../consumer && python3 consumer.py 5001
```

`python3 run_training.py --dry-run --filter ... ` prints the visit IDs to pass to `--visits`.

## Tests

Unit tests over the filter language, the chunk plan and the wire format. Standard library
only — no containers, no dataset, well under a second:

```bash
python3 -m unittest discover -s tests   # from the repo root
```

They cover the filter operators and value typing, the AND semantics, visits whose metadata
lacks the filtered field, both orphan-file cases the dataset actually contains, chunk
coverage (every chunk exactly once), the dropped partial tail, the shuffle, the
`uuid + float32` bytes a consumer reads off the socket, and the readiness wait the
orchestrator depends on.

Each test was then re-run against a deliberately broken copy of the code — wrong operator
precedence, a missing field treated as a match, a partial chunk included, the UUID prefix
dropped, the shuffle removed, the ready marker renamed in one module only — and each
failed as it should, so none of them is green by accident.

## Notes on the given consumer

- **Its `visit_db.json` is generated, not shipped.** The consumer's Dockerfile copies that
  file in, and the given copy is a hand-maintained duplicate of metadata that already lives
  in `data/` — nothing keeps the two in sync. So the orchestrator builds the consumer image
  from a temporary context holding a database it generates from the visits being streamed,
  and the checked-out copy is ignored. The dataset stays the only source of truth.
- It logs `Visit ID <hex> not found in visit_db.json!` for every chunk regardless, and this
  is the consumer's own bug, not a stale database: the chunks carry the correct 128-bit
  UUID, which the consumer hex-encodes without dashes (`1077dc01fd89...`) before looking it
  up in a database keyed with dashes (`1077dc01-fd89-...`), so the lookup can never hit.
  Left alone because the consumer stands in for the deep-learning team's code.
- It sleeps 100 ms per chunk, which caps a feed at ~10 chunks/s per port. The provider is
  not the bottleneck; add ports to go faster.
- The dataset ships one `.json` with no matching `.npy` (and one `.npy` with no `.json`),
  so 98 of the 99 visits are usable. Orphans are skipped with a line on stderr.

## Scaling to a cluster (design, not implemented)

Nothing above assumes one host except two things: `localhost` between the pair, and the
chunk queue living in the provider's memory. A cluster version would change these:

**Work partitioning.** Keep the chunk plan owned by the orchestrator, not the providers:
it resolves the filter to a visit list, builds the shuffled plan from a seed derived from
`(job_id, epoch)`, and hands provider replica *i* its deterministic shard. No runtime
coordination, no shared queue, and a restarted replica replays exactly its own shard. A
shared queue (Redis/NATS) only becomes worth its operational cost if shards need
rebalancing while a job runs.

**Discovery.** Replace `localhost` with endpoints: one provider pod per feed behind a
headless Service, so each training worker gets stable DNS names. The consumer's
hardcoded `localhost` can be preserved without touching its code by running a thin TCP
forwarder sidecar in the training pod (`localhost:5001` → `provider-0:5001`) — the same
trick as the shared network namespace here, one layer up.

**Data access.** Visits must be readable from any node: object storage (S3/GCS) with a
node-local cache, or a ReadWriteMany volume. Filtering should not stat 100k files, so the
per-visit JSONs become a metadata index (Parquet or a small catalog service) that the
orchestrator queries for the visit list. Schedule provider pods with affinity for the
nodes that already cache their shard to keep EEG bytes off the cross-zone network.

**Orchestration.** `run_training.py` becomes a submitter: it creates the provider
Deployment (replicas = number of feeds) and the training Job from the resolved plan, and
waits on readiness probes instead of grepping logs. `docker logs -f` becomes the cluster's
log pipeline, plus metrics per provider (chunks/s, shard progress) for the training team.

**Failure semantics.** Chunk delivery is at-least-once: a restarted worker gets its shard
from the start, which is acceptable for training on shuffled data. Exactly-once would
require checkpointing the consumed offset per shard, which is only worth it if the
training loop tracks sample counts precisely.
