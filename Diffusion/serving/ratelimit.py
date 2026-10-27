from __future__ import annotations

import time
import uuid

import redis.asyncio as aioredis

from config import REDIS_URL

r = aioredis.Redis.from_url(REDIS_URL)


async def cache_ping() -> None:
    """Raise if Redis is unreachable. For the API's /readyz probe."""
    await r.ping()


# System level rate limitter
async def system_rate_limit() -> bool:
    SYS_LIM = 50    # Req per minute for the system.
    SYS_LIM_WINDOW_S = 60   # 60 second window

    redis_key = "sys_rate_limit"
    current_time = time.time()
    member = uuid.uuid4().hex

    # Create an async pipeline
    async with r.pipeline(transaction=True) as pipe:
        pipe.zremrangebyscore(redis_key, 0, current_time - SYS_LIM_WINDOW_S)
        pipe.zadd(redis_key, {member: current_time})
        pipe.zcard(redis_key)
        pipe.expire(redis_key, SYS_LIM_WINDOW_S)
        _, _, count, _ = await pipe.execute()

    if count > SYS_LIM:
        await r.zrem(redis_key, member)
        return False
    return True

# User level rate limitter
async def user_rate_limit() -> bool:
    pass
