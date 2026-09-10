def get_cached_value(cache, key, now):
    """Read a cached entry unless its expiry timestamp has passed."""
    entry = cache.get(key)
    if entry is None:
        return None
    if entry["expires_at"] <= now:
        del cache[key]
        return None
    return entry["value"]


def put_cached_value(cache, key, value, ttl, now):
    """Store a value with a time-to-live measured in seconds."""
    if ttl <= 0:
        raise ValueError("ttl must be positive")
    cache[key] = {"value": value, "expires_at": now + ttl}
    return value


def purge_expired_entries(cache, now):
    """Remove all expired entries and return the number removed."""
    expired = [key for key, entry in cache.items() if entry["expires_at"] <= now]
    for key in expired:
        del cache[key]
    return len(expired)
