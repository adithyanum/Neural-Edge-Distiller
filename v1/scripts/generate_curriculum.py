from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler, make_repetition_penalty


SYSTEM_PROMPT = """You are a distributed systems architect with deep expertise in failure modes, scalability, and real-world trade-offs.

Respond ONLY in this exact format with no preamble, greetings, filler, or advice:

QUESTION: <restate the question exactly>
THOUGHT: <Follow this exact reasoning chain — do not skip any step:
1. FAILURE MODE: What is the precise root cause or bottleneck? Name it specifically.
2. NAIVE FIXES: What are two obvious solutions engineers reach for first, and why does each one fail or make things worse?
3. MECHANISM: What is the precise solution that actually works, and how does it address the root cause specifically?
4. TRADE-OFF: What does this solution cost — latency, consistency, complexity, operational burden?>
ARCHITECTURE: <component1> -> <component2> -> <component3> -> ... (must reflect THOUGHT's conclusion exactly)
END_OF_ARCH"""


FEW_SHOT_EXAMPLES = """Below are four examples showing varied problem domains and architecture patterns. Study the reasoning depth and structure.

---

QUESTION: A distributed job scheduler is causing duplicate task execution across worker nodes during network partitions.
THOUGHT:
1. FAILURE MODE: Split-brain task assignment — during a network partition, multiple scheduler nodes independently believe they are the leader and assign the same job to different workers, causing duplicate execution with no idempotency guarantee.
2. NAIVE FIXES: Adding a heartbeat timeout to detect the partition doesn't help because the window between timeout and leader re-election is exactly when duplicates happen. Using a shared database lock fails because the partition prevents the secondary from reaching the DB, so it proceeds independently assuming the lock is free.
3. MECHANISM: Fencing tokens via a distributed consensus system like etcd or ZooKeeper. Every leader election produces a monotonically increasing token. Workers reject any task assignment carrying a token lower than the highest they've seen — this makes stale leader commands from the partitioned node automatically invalid without requiring the nodes to communicate with each other.
4. TRADE-OFF: Consensus adds a round-trip to etcd on every leader election, increasing failover latency. The system also becomes unavailable if etcd itself loses quorum, trading duplicate execution risk for hard unavailability during quorum loss.
ARCHITECTURE: Scheduler Cluster -> etcd (Consensus + Fencing Tokens) -> Worker Pool -> Task Execution Engine -> Result Store
END_OF_ARCH

---

QUESTION: ML feature pipeline is producing training/serving skew causing model accuracy degradation in production.
THOUGHT:
1. FAILURE MODE: Feature computation divergence — features computed at training time using batch aggregations (e.g., 30-day rolling averages over historical data) differ fundamentally from features computed at serving time using real-time approximations, so the model is evaluated on a distribution it was never trained on.
2. NAIVE FIXES: Retraining the model more frequently doesn't fix skew because the root cause is the computation difference, not data staleness — the skew persists even with fresh training. Logging serving features for analysis helps diagnose but doesn't eliminate the divergence during live inference.
3. MECHANISM: A unified feature store with point-in-time correct retrieval. Features are computed once using a shared transformation layer and stored with timestamps. Training jobs retrieve features as they existed at a specific historical timestamp, and the serving layer retrieves from the same store using the same transformation code — eliminating two separate code paths as the source of divergence.
4. TRADE-OFF: The feature store becomes a critical dependency for both training and serving, meaning its availability directly impacts model serving uptime. Storage costs increase significantly since point-in-time snapshots must be retained for training reproducibility.
ARCHITECTURE: Raw Event Stream -> Unified Transform Layer -> Feature Store (Point-in-Time) -> Training Pipeline | Serving API -> Model Inference
END_OF_ARCH

---

QUESTION: A multi-tenant SaaS API is experiencing latency spikes where one customer's heavy workload degrades performance for all others.
THOUGHT:
1. FAILURE MODE: Noisy neighbor resource contention — all tenants share the same worker pool and database connection pool, so a single tenant running expensive queries or bursting requests monopolizes connections and CPU, causing queue buildup and latency spikes for unrelated tenants.
2. NAIVE FIXES: Rate limiting at the API gateway per tenant caps request volume but doesn't prevent a single allowed expensive request from consuming disproportionate resources downstream. Horizontal scaling adds capacity but the noisy tenant scales proportionally with it, so the contention ratio stays the same.
3. MECHANISM: Tenant-aware weighted fair queuing with resource isolation. Each tenant gets a dedicated request queue with a weight proportional to their tier. A central dispatcher pulls from queues using weighted round-robin, guaranteeing each tenant a minimum share of worker capacity regardless of what others are doing. High-tier tenants get larger weights, but no tenant can starve others entirely.
4. TRADE-OFF: Per-tenant queuing adds memory overhead proportional to tenant count and introduces dispatcher logic that becomes a potential bottleneck at very high tenant counts. Queue depth monitoring per tenant is required to detect and alert on tier violations operationally.
ARCHITECTURE: API Gateway -> Per-Tenant Queue (Weighted Fair) -> Central Dispatcher -> Shared Worker Pool -> DB Connection Pool (Isolated per Tier)
END_OF_ARCH

---

QUESTION: A search indexing pipeline is falling behind real-time updates causing users to see stale search results after writes.
THOUGHT:
1. FAILURE MODE: Index propagation lag — writes commit to the primary database and return success to the user, but the search index is updated asynchronously via a batch job that runs on a fixed interval, creating a visible staleness window that grows under write load.
2. NAIVE FIXES: Reducing the batch interval improves latency but increases indexing infrastructure load and doesn't eliminate the window — it just shrinks it. Writing to the search index synchronously in the write path eliminates the window but couples write latency to indexing latency and creates a dual-write consistency problem if one fails.
3. MECHANISM: Change data capture on the database write-ahead log. A CDC consumer tails the DB transaction log and emits change events to a queue in commit order. The indexing service consumes from this queue and applies updates to the search index in near-real-time. Because CDC reads from the WAL after commit, it's decoupled from the write path, preserves ordering, and catches every change including those made outside the application layer.
4. TRADE-OFF: CDC introduces operational complexity — the consumer must handle log sequence number tracking, schema changes in the source DB require CDC pipeline updates, and the queue depth must be monitored to detect indexing lag under burst writes.
ARCHITECTURE: Write API -> Primary DB (WAL) -> CDC Consumer -> Change Event Queue -> Index Worker -> Search Index -> Read API
END_OF_ARCH"""


def generate_architecture(model, tokenizer, title, description):
    """
    Generates structured CoT architectural reasoning using instruct chat format.
    Teacher model: Meta-Llama-3-8B-Instruct-4bit via MLX.
    """

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"{FEW_SHOT_EXAMPLES}\n\n---\n\nNow reason through this problem with the same depth:\nQUESTION: {description}"
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    sampler = make_sampler(temp=0.4, top_p=0.9, min_p=0.05)
    rep_penalty = make_repetition_penalty(penalty=1.2, context_size=20)

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=600,
        sampler=sampler,
        logits_processors=[rep_penalty]
    )

    response = response.strip()

    # Enforce hard stop at END_OF_ARCH
    if "END_OF_ARCH" in response:
        response = response.split("END_OF_ARCH")[0].strip() + "\nEND_OF_ARCH"

    return response