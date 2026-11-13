#!/usr/bin/env python3
"""Scrape Danish sermons from dansketaler.dk / danskepraedikener.dk.

The collection is freely readable.  This scraper is intended for local research
and for building a domain corpus with source metadata.  Keep the `source_url`,
`source_note`, and `year` fields with any derived dataset.  For an open-source
release, prefer public-domain sermons (the default year ceiling is 1900) or
texts whose publisher has given permission.

Example
-------
    python experiments/sermon/scrape_sermons.py \
        --years 1850-1860 \
        --output /tmp/sermon/sermons.jsonl \
        --cache-dir /tmp/sermon/cache
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import lxml.html

BASE = "https://www.dansketaler.dk"
USER_AGENT = "danglish-sermon-research/0.1 (+local research; contact: project maintainer)"


def parse_years(value: str) -> list[int]:
    if "-" in value:
        start, end = value.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(part) for part in value.split(",") if part.strip()]


def fetch(url: str, cache_dir: Path, delay: float) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", url.replace(BASE, ""))[:180]
    path = cache_dir / (safe or "index")
    if path.is_file():
        return path.read_text(encoding="utf-8", errors="replace")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                html = response.read().decode("utf-8", errors="replace")
            break
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 3:
                raise
            wait = 60 * (attempt + 1)
            print(f"429 {url}; sleeping {wait}s", file=sys.stderr)
            time.sleep(wait)
    else:
        raise RuntimeError(f"rate limited: {url}")
    path.write_text(html, encoding="utf-8")
    time.sleep(delay)
    return html


def month_fragments(year: int, cache_dir: Path, delay: float) -> list[str]:
    url = f"{BASE}/praedikener/tale?year={year}"
    html = fetch(url, cache_dir, delay)
    tree = lxml.html.fromstring(html)
    title = " ".join(tree.xpath("//title/text()")).strip()
    if title != str(year):
        # Unsupported years fall back to the current year.
        return []
    fragments = tree.xpath('//div[contains(@class,"current-speeches-container")]/@data-url')
    return [BASE + fragment for fragment in fragments]


def sermon_links(year: int, cache_dir: Path, delay: float) -> list[str]:
    links: list[str] = []
    seen: set[str] = set()
    for fragment in month_fragments(year, cache_dir, delay):
        html = fetch(fragment, cache_dir, delay)
        tree = lxml.html.fromstring(html)
        for href in tree.xpath('//a[contains(@class,"current-speech")]/@href'):
            url = href if href.startswith("http") else BASE + href
            if "/soeg" in url:
                continue
            if "/tale/" not in url:
                continue
            if url not in seen:
                seen.add(url)
                links.append(url)
    return links


def parse_sermon(url: str, html: str) -> dict[str, object] | None:
    tree = lxml.html.fromstring(html)
    title_nodes = tree.xpath('//h1[contains(@class,"article-title")]//text()')
    title = " ".join(" ".join(title_nodes).split()) if title_nodes else ""
    if not title:
        return None

    date_nodes = tree.xpath("//time/@datetime")
    date = date_nodes[0] if date_nodes else ""
    speaker_nodes = tree.xpath('//div[contains(@class,"speech-speaker")]//text()')
    speaker = " ".join(" ".join(speaker_nodes).split()) if speaker_nodes else ""
    location_nodes = tree.xpath('//div[contains(@class,"speech-location")]//text()')
    location = " ".join(" ".join(location_nodes).split()) if location_nodes else ""
    source_nodes = tree.xpath('//div[contains(@class,"speech-source")]//text()')
    source_note = " ".join(" ".join(source_nodes).split()) if source_nodes else ""

    body_nodes = tree.xpath('//div[@id="speech"]//div[contains(@class,"speech-article-content")]')
    if not body_nodes:
        return None
    body = body_nodes[0]
    paragraphs = [" ".join(paragraph.text_content().split()) for paragraph in body.xpath(".//p")]
    paragraphs = [paragraph for paragraph in paragraphs if paragraph]
    text = "\n\n".join(paragraphs) if paragraphs else " ".join(body.text_content().split())
    if len(text) < 500:
        return None

    year_match = re.search(r"(1[5-9]\d{2}|20\d{2})", date)
    year = int(year_match.group(1)) if year_match else 0
    return {
        "id": url.rstrip("/").rsplit("/", 1)[-1],
        "url": url,
        "title": title,
        "date": date,
        "year": year,
        "speaker": speaker,
        "location": location,
        "source_note": source_note,
        "license": "public_domain" if year and year < 1900 else "copyright_unknown",
        "text": text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", default="1850-1860")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache-dir", default=Path("/tmp/sermon/cache"), type=Path)
    parser.add_argument("--delay", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--links-only",
        action="store_true",
        help="write the sermon link list instead of fetching sermon pages",
    )
    parser.add_argument(
        "--links",
        type=Path,
        help="read a JSONL link list (`url`, `year`) instead of crawling year pages",
    )
    args = parser.parse_args()

    all_links: list[tuple[str, int]] = []
    if args.links:
        for line in args.links.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            all_links.append((str(row["url"]), int(row.get("year", 0))))
    else:
        years = parse_years(args.years)
        seen: set[str] = set()
        for year in years:
            try:
                links = sermon_links(year, args.cache_dir, args.delay)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                print(f"{year}: failed to list: {exc}", file=sys.stderr)
                continue
            new = [link for link in links if link not in seen]
            seen.update(new)
            all_links.extend((link, year) for link in new)
            if new:
                print(f"{year}: {len(new)} sermons", file=sys.stderr)
            if args.limit and len(all_links) >= args.limit:
                break

    if args.limit:
        all_links = all_links[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.links_only:
        with args.output.open("w", encoding="utf-8") as handle:
            for url, year in all_links:
                handle.write(json.dumps({"url": url, "year": year}, ensure_ascii=False) + "\n")
        print(f"wrote {len(all_links)} links to {args.output}", file=sys.stderr)
        return

    done: set[str] = set()
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                done.add(str(json.loads(line)["url"]))
            except (KeyError, json.JSONDecodeError):
                continue

    written = 0
    mode = "a" if done else "w"
    with args.output.open(mode, encoding="utf-8") as handle:
        for index, (url, year) in enumerate(all_links, 1):
            if url in done:
                continue
            try:
                html = fetch(url, args.cache_dir, args.delay)
                row = parse_sermon(url, html)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                print(f"{url}: failed: {exc}", file=sys.stderr)
                continue
            if row is None:
                print(f"{url}: no body", file=sys.stderr)
                continue
            if not row.get("year"):
                row["year"] = year
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            print(f"{index}/{len(all_links)} {row['year']} {row['title'][:70]}", file=sys.stderr)
            if written % 50 == 0:
                handle.flush()
    print(f"wrote {written} sermons to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
