#!/usr/bin/env python3
"""Scrape a public LinkedIn job description into a plain-text file."""

from __future__ import annotations

import argparse
from pathlib import Path

from jobpilot.tools.linkedin_jd import (
    default_output_path,
    fetch_linkedin_job,
    write_job_description,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape title, company, and About-the-job details from a public LinkedIn JD."
    )
    parser.add_argument("url", help="Public LinkedIn job URL.")
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        help="Output .txt path. Defaults to output/jds/<company>_<title>_<job-id>.txt.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="HTTP timeout in seconds.",
    )
    args = parser.parse_args()

    job = fetch_linkedin_job(args.url, timeout=args.timeout)
    output_path = args.out or default_output_path(job)
    write_job_description(job, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
