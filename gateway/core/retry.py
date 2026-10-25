import random


def backoff_seconds(base, retry_number):
    """Return exponential backoff with bounded positive jitter."""
    return base * (2**retry_number) * random.uniform(1, 1.25)
