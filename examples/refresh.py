#!/usr/bin/env python3
"""Explicitly refresh a protected SIWC credential store once.

Usage: refresh.py --store PATH

Issues one refresh POST through ``SiwcCredentialStore.refresh`` and prints only
the safe summary (plan-use flag, expiry, scope count). An expired access token
or disabled plan-use permission does not prevent renewal. Failures print the
safe error code and exit 1; nothing is cleared, reauthorized or retried.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from chatgpt_plan_sdk import SiwcAuthError, SiwcCredentialStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh a SIWC credential store once.")
    parser.add_argument("--store", required=True, help="path of the protected credential file")
    args = parser.parse_args(argv)
    try:
        credentials = SiwcCredentialStore(Path(args.store)).refresh()
    except SiwcAuthError as exc:
        print(f"refresh failed: {exc.code}", file=sys.stderr)
        return 1
    except Exception as exc:  # type name only: messages may carry raw provider text
        print(f"refresh failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    summary = credentials.redacted()
    print(
        "refreshed: "
        f"has_plan_use={summary['has_plan_use']} "
        f"expires_at={summary['expires_at']} "
        f"scopes={len(summary['scopes'])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
