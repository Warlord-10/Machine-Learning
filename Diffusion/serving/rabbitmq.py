"""Job queue on RabbitMQ. The API publishes {job_id}; the worker consumes.

Run the worker with:  python rabbitmq.py

Messages that fail (malformed or handler raised) are dead-lettered to
`job.created.dlq` instead of being requeued in a hot loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable

import aio_pika
from aio_pika.abc import (AbstractIncomingMessage, AbstractRobustChannel,
                          AbstractRobustConnection, AbstractRobustQueue)

from config import BATCH_MAX_IMAGES, RABBITMQ_URL

log = logging.getLogger(__name__)

# Enough unacked jobs in hand to fill the running batch and the next one
# (a job has at least one image). More would just hoard work from other workers.
PREFETCH = 2 * BATCH_MAX_IMAGES


class JobQueue:
    name = "job.created"
    dlx = "job.dlx"
    dlq = "job.created.dlq"

    def __init__(self, url: str = RABBITMQ_URL):
        self._url = url
        self._conn: AbstractRobustConnection | None = None
        self._channel: AbstractRobustChannel | None = None
        self._queue: AbstractRobustQueue | None = None
        self._lock = asyncio.Lock()  # concurrent first publishes must share one connection

    async def _open(self, prefetch: int | None = None) -> AbstractRobustQueue:
        """Connect and declare once. Robust connections re-declare on reconnect."""
        async with self._lock:
            if self._queue is None:
                self._conn = await aio_pika.connect_robust(self._url)
                self._channel = await self._conn.channel()  # publisher confirms on by default
                if prefetch:
                    await self._channel.set_qos(prefetch_count=prefetch)
                self._queue = await self.declare(self._channel)
        return self._queue

    async def declare(self, channel: AbstractRobustChannel) -> AbstractRobustQueue:
        """Idempotent. The arguments must match the live queue exactly, or the
        broker closes the channel with PRECONDITION_FAILED."""
        dlx = await channel.declare_exchange(
            self.dlx, aio_pika.ExchangeType.DIRECT, durable=True
        )
        dlq = await channel.declare_queue(self.dlq, durable=True)
        await dlq.bind(dlx, routing_key=self.name)
        return await channel.declare_queue(
            self.name,
            durable=True,
            arguments={
                "x-dead-letter-exchange": self.dlx,
                "x-dead-letter-routing-key": self.name,
            },
        )

    async def publish(self, payload: dict) -> None:
        await self._open()
        message = aio_pika.Message(
            body=json.dumps(payload).encode(),
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,  # survives a broker restart
        )
        # mandatory: an unroutable message raises instead of vanishing
        await self._channel.default_exchange.publish(
            message, routing_key=self.name, mandatory=True
        )

    async def subscribe(
        self,
        handler: Callable[[dict], Awaitable[Any]],
        prefetch: int = PREFETCH,
    ) -> None:
        queue = await self._open(prefetch)

        async def on_message(message: AbstractIncomingMessage) -> None:
            try:
                # Ack on success; reject (-> DLQ) if the body or handler raises.
                async with message.process(requeue=False):
                    await handler(json.loads(message.body))
            except json.JSONDecodeError:
                log.error("malformed message on %s, dead-lettered: %.300r", self.name, message.body)
            except Exception:
                log.exception("failed on %s, dead-lettered", self.name)

        await queue.consume(on_message)

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
        self._conn = self._channel = self._queue = None


job_queue = JobQueue()


async def _run_worker() -> None:
    # Imported here so the API can import `job_queue` without loading the ONNX stack.
    from prometheus_client import start_http_server

    from config import METRICS_PORT
    from db import engine, init_db
    from tasks import batch_loop, run_image_pipeline

    start_http_server(METRICS_PORT)
    await init_db()
    await job_queue.subscribe(run_image_pipeline)
    try:
        await batch_loop()  # runs until Ctrl-C; on SIGTERM unacked jobs are simply redelivered
    finally:
        await job_queue.close()
        await engine.dispose()  # pooled DB connections otherwise hang interpreter exit


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_run_worker())
