#!/usr/bin/env python3
"""Run the gateway test suite and optional live provider smoke checks."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_tests() -> int:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-W", "error::RuntimeWarning"],
        cwd=ROOT,
    )
    return result.returncode


async def run_live_checks(timeout: float) -> int:
    from gateway.main import configured_gateway
    from gateway.models import Message
    from gateway.core.errors import normalize

    gateway = configured_gateway()
    failures = 0
    try:
        for alias, model in gateway.models.items():
            if model.provider is None:
                print(json.dumps({"check": "live", "model": alias, "status": "credentials_missing"}))
                failures += 1
                continue
            try:
                result = await asyncio.wait_for(
                    model.provider.complete(
                        model.upstream,
                        [Message(role="user", content="Reply with exactly OK.")],
                        None,
                        timeout,
                    ),
                    timeout + 5,
                )
                print(json.dumps({
                    "check": "live",
                    "model": alias,
                    "status": "success",
                    "has_content": bool(result.content),
                    "usage": result.usage.model_dump(),
                }))
            except Exception as exc:  # provider-specific SDK exceptions are normalized
                error = normalize(exc)
                print(json.dumps({
                    "check": "live",
                    "model": alias,
                    "status": "failed",
                    "code": error.code,
                    "http_status": getattr(exc, "status_code", None),
                }))
                failures += 1
    finally:
        for model in gateway.models.values():
            if model.provider:
                await model.provider.close()
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-live", action="store_true", help="Run mocked tests without provider calls")
    parser.add_argument("--timeout", type=float, default=25, help="Per-provider live call timeout in seconds")
    args = parser.parse_args()

    tests_status = run_tests()
    if tests_status:
        return tests_status
    if args.skip_live:
        print("Live provider checks skipped.")
        return 0
    return 0 if asyncio.run(run_live_checks(args.timeout)) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
