"""Bootstrap `data/bullet_pool.yaml` from your CV template.

ONE-TIME helper. After running:

  1. Rename the auto-generated `entry-NNN` IDs to meaningful slugs
     (e.g. `impact-cost-reduction`, `leadership-mentoring`).
  2. Add alternative framings of the same accomplishments (impact-focused,
     AI-focused, leadership-focused) so the tailor has more variety per JD.

From this point on the pool is hand-maintained — human review before adding
or rewording bullets, since the tailor will pick from whatever is here.

Usage:
    uv run python scripts/generate_bullet_pool.py
    uv run python scripts/generate_bullet_pool.py --template data/my_cv.docx
    uv run python scripts/generate_bullet_pool.py --anchor "ACME Corp, City"
    uv run python scripts/generate_bullet_pool.py --force   # overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jobpilot.config import get_settings
from jobpilot.tools.bullet_pool import BulletEntry, extract_current_role_bullets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--template",
        type=Path,
        help="CV template .docx (default: settings.cv_template_path).",
    )
    parser.add_argument(
        "--anchor",
        type=str,
        help="Role anchor line text (default: settings.current_role_anchor from .env).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output YAML path (default: settings.bullet_pool_path).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing pool file (you usually want to edit by hand instead).",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    template = args.template or settings.cv_template_path
    anchor = args.anchor or settings.current_role_anchor
    output = args.output or settings.bullet_pool_path

    if not anchor:
        print(
            "error: no role anchor. Set CURRENT_ROLE_ANCHOR in .env or pass --anchor.",
            file=sys.stderr,
        )
        return 2
    if not template.is_file():
        print(f"error: template not found: {template}", file=sys.stderr)
        return 2
    if output.exists() and not args.force:
        print(
            f"error: {output} already exists. Edit it by hand, or pass --force to overwrite.",
            file=sys.stderr,
        )
        return 2

    try:
        entries = extract_current_role_bullets(template, anchor)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not entries:
        print(
            f"error: no `List Paragraph` bullets found under anchor {anchor!r}.",
            file=sys.stderr,
        )
        return 2

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render_yaml(entries), encoding="utf-8")

    print(f"✓ Wrote {len(entries)} bullets to {output}")
    print("Next steps:")
    print("  1. Rename `entry-NNN` IDs to meaningful slugs.")
    print("  2. Add alternative framings of the same accomplishments.")
    print("  3. Review every bullet — the tailor cannot reject what you put here.")
    return 0


def _render_yaml(entries: list[BulletEntry]) -> str:
    """Hand-render YAML so output matches data/bullet_pool.example.yaml's style."""
    lines = [
        "# Bullet pool — bootstrapped by scripts/generate_bullet_pool.py.",
        "# Edit by hand from here:",
        "#   1. Rename `entry-NNN` IDs to meaningful slugs.",
        "#   2. Add alternative framings of the same accomplishments.",
        "",
        "current_role:",
    ]
    for entry in entries:
        text = entry.text.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f"  - id: {entry.id}")
        lines.append(f'    text: "{text}"')
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
