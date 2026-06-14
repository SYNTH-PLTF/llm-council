"""Load test: mixed query classes against /v1/chat.

Bring the stack up (docker compose up), then run interactively:
    uv run locust -f tests/load/locustfile.py --host http://localhost:8000
or headless with a latency/error budget:
    uv run locust -f tests/load/locustfile.py --host http://localhost:8000 \
        --headless -u 20 -r 5 -t 1m --csv load
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task

_QUERIES = [
    {"query": "What is the capital of France?", "force_council": False},
    {
        "query": "Should we migrate our monolith to microservices? Weigh the tradeoffs.",
        "force_council": True,
    },
    {"query": "What is 17 times 23? Answer with just the number.", "force_council": False},
]


class CouncilUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task
    def chat(self) -> None:
        self.client.post("/v1/chat", json=random.choice(_QUERIES), name="/v1/chat")
