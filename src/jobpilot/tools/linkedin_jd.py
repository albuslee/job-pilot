"""Extract LinkedIn public job descriptions into JobPilot-friendly text."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup, Tag

_DEFAULT_TIMEOUT = 20
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)
_NOISE_LINES = {
    "show more",
    "show less",
    "apply",
    "save",
}


@dataclass(frozen=True)
class LinkedInJob:
    title: str
    company: str
    about_job: str
    source_url: str | None = None


def _clean_text(value: str) -> str:
    lines = []
    for raw_line in value.replace("\xa0", " ").splitlines():
        line = " ".join(raw_line.split())
        if not line or line.lower() in _NOISE_LINES:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _text_from_first(soup: BeautifulSoup, selectors: tuple[str, ...]) -> str:
    for selector in selectors:
        node = soup.select_one(selector)
        if node is not None:
            text = _clean_text(node.get_text("\n"))
            if text:
                return text
    return ""


def _find_about_section(soup: BeautifulSoup) -> Tag | None:
    for selector in (
        ".show-more-less-html__markup",
        ".description__text",
        "section.description",
        "[data-test-id='job-details']",
    ):
        node = soup.select_one(selector)
        if isinstance(node, Tag):
            return node

    for heading in soup.find_all(["h2", "h3"]):
        if _clean_text(heading.get_text(" ")).lower() == "about the job":
            section = heading.find_parent("section")
            if isinstance(section, Tag):
                return section
            sibling = heading.find_next_sibling()
            if isinstance(sibling, Tag):
                return sibling
    return None


def parse_linkedin_job(html: str, *, source_url: str | None = None) -> LinkedInJob:
    """Parse title, company, and About-the-job details from public LinkedIn HTML."""
    soup = BeautifulSoup(html, "html.parser")
    title = _text_from_first(
        soup,
        (
            "h1.top-card-layout__title",
            "h1",
            "[data-test-job-title]",
        ),
    )
    company = _text_from_first(
        soup,
        (
            "a.topcard__org-name-link",
            ".topcard__flavor--black-link",
            ".top-card-layout__second-subline a",
            "[data-test-company-name]",
        ),
    )
    about_node = _find_about_section(soup)
    about_job = _clean_text(about_node.get_text("\n")) if about_node else ""
    if about_job.lower().startswith("about the job\n"):
        about_job = about_job.split("\n", 1)[1].strip()

    return LinkedInJob(
        title=title or "Unknown title",
        company=company or "Unknown company",
        about_job=about_job or "No About the job section found.",
        source_url=source_url,
    )


def format_job_description(job: LinkedInJob) -> str:
    """Render a LinkedInJob as plain text for JobPilot eval/run inputs."""
    parts = [
        f"Title: {job.title}",
        f"Company: {job.company}",
    ]
    if job.source_url:
        parts.append(f"Source: {job.source_url}")
    parts.extend(["", "About the job:", job.about_job])
    return "\n".join(parts).rstrip() + "\n"


def fetch_linkedin_job(url: str, *, timeout: int = _DEFAULT_TIMEOUT) -> LinkedInJob:
    """Fetch and parse a public LinkedIn job URL.

    This intentionally does not handle authenticated sessions, CAPTCHAs, or private pages.
    """
    request = Request(
        url,
        headers={"User-Agent": _USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
    )
    with urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        html = response.read().decode(charset, errors="replace")
    return parse_linkedin_job(html, source_url=url)


def default_output_path(job: LinkedInJob, *, output_dir: Path = Path("output/jds")) -> Path:
    job_id = ""
    if job.source_url:
        path_parts = [p for p in urlparse(job.source_url).path.split("/") if p]
        if path_parts:
            job_id = path_parts[-1]
    safe = "_".join(
        part
        for part in (
            _clean_filename(job.company),
            _clean_filename(job.title),
            _clean_filename(job_id),
        )
        if part
    )
    return output_dir / f"{safe or 'linkedin_job'}.txt"


def _clean_filename(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in value.lower())
    return "_".join(part for part in cleaned.split("_") if part)


def write_job_description(job: LinkedInJob, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_job_description(job), encoding="utf-8")
    return output_path
