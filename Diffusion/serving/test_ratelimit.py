"""Self-check for the sliding-window algorithm, with a minimal in-memory fake
Redis (so it runs with no server). Proves: admits up to `limit` in a window,
rejects the next, and admits again once entries age out.

Run: python3 test_ratelimit.py
"""


class FakeZSet:
    def __init__(self):
        self.d = {}  # member -> score

    def zremrangebyscore(self, lo, hi):
        for m in [m for m, s in self.d.items() if lo <= s <= hi]:
            del self.d[m]

    def zadd(self, mapping):
        self.d.update(mapping)

    def zcard(self):
        return len(self.d)

    def zrem(self, member):
        self.d.pop(member, None)


class FakePipe:
    def __init__(self, z):
        self.z, self.ops = z, []

    def zremrangebyscore(self, key, lo, hi):
        self.ops.append(lambda: self.z.zremrangebyscore(lo, hi))

    def zadd(self, key, mapping):
        self.ops.append(lambda: self.z.zadd(mapping))

    def zcard(self, key):
        self.ops.append(lambda: self.z.zcard())

    def pexpire(self, key, ms):
        self.ops.append(lambda: None)

    def execute(self):
        return [op() for op in self.ops]


class FakeRedis:
    def __init__(self):
        self.z = FakeZSet()

    def pipeline(self):
        return FakePipe(self.z)

    def zrem(self, key, member):
        self.z.zrem(member)


def _allow(r, now_ms, limit=3, window_ms=60_000, member="m"):
    """Mirror of ratelimit.allow_request, parameterized on time for testing."""
    pipe = r.pipeline()
    pipe.zremrangebyscore("k", 0, now_ms - window_ms)
    pipe.zadd("k", {member: now_ms})
    pipe.zcard("k")
    pipe.pexpire("k", window_ms)
    _, _, count, _ = pipe.execute()
    if count > limit:
        r.zrem("k", member)
        return False
    return True


def test_window():
    r = FakeRedis()
    t = 1_000_000
    # 3 admitted inside the window
    assert _allow(r, t + 0, member="a")
    assert _allow(r, t + 1000, member="b")
    assert _allow(r, t + 2000, member="c")
    # 4th within the window -> rejected, and it must NOT occupy a slot
    assert not _allow(r, t + 3000, member="d")
    assert r.z.zcard() == 3, "rejected request must be rolled back"
    # after the window slides past the first entries, a new one is admitted
    assert _allow(r, t + 61_000, member="e")


if __name__ == "__main__":
    test_window()
    print("rate-limit self-check passed")
