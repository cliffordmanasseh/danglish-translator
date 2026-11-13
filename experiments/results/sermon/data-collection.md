# Sermon corpus collection, 2026-11-13

## Full corpus

| Metric | Value |
|---|---:|
| Sermons | **1,305** |
| Words | **2,203,932** |
| Characters | 12,183,554 |
| Train sentences | **72,004** |
| Held-out test sentences | **7,548** |
| Earliest sermon | 1539 |
| Latest sermon | 2026 |

The sentence split holds out every 10th sermon, so the test set measures
generalization to unseen preachers and styles. Sentences outside 4–45 words or
15–320 characters were skipped (15,390), as were 55 duplicate sermons.

## Public-domain subset

| Metric | Value |
|---|---:|
| Sermons (pre-1900) | **163** |
| Words | **599,509** |
| Train sentences | **8,771** |
| Test sentences | **2,356** |

This subset uses `--test-every 5` and is the part that can be released under a
public-domain claim. It is large enough for a clean open model, but much
smaller than the full research corpus.

## Source and licensing

- Source: [Danske Taler / Danske Prædikener](https://www.dansketaler.dk/praedikener),
  a curated collection of 1,000+ digitized Danish sermons from the Reformation
  to today.
- `robots.txt` allows the sermon and year-archive pages; search pagination is
  disallowed and was not used.
- Each record keeps `url`, `title`, `year`, `speaker`, `location`,
  `source_note`, and `license`.
- **163 sermons are pre-1900 and marked `public_domain`.**
- **1,142 sermons are modern and marked `copyright_unknown`.** They are for
  local research and domain adaptation only; do not redistribute the text or a
  derived dataset without permission.
- The 1940s block (112 sermons) uses older orthography (`Nytaar`, `saadan`) and
  is still copyrighted. Keep it for research; do not ship it without a rights
  review.

## Decade distribution

| Decade | Sermons |
|---|---:|
| 1530s–1790s | 32 |
| 1800s | 1 |
| 1810s | 6 |
| 1820s | 9 |
| 1840s | 50 |
| 1850s | 13 |
| 1860s | 44 |
| 1870s | 1 |
| 1880s | 2 |
| 1890s | 5 |
| 1900s–1930s | 11 |
| 1940s | 112 |
| 1950s–1990s | 72 |
| 2000s | 53 |
| 2010s | 166 |
| 2020s | 728 |

The 2020s dominate, which is appropriate for contemporary spoken style. The
1800s and 1940s blocks add historical and orthographic variety.

## Pipeline

```bash
# 1. Collect links from the year archive (robots-compliant, 429 backoff)
python experiments/sermon/scrape_sermons.py \
  --years 1900-2026 --links-only \
  --output data/sermons/dansketaler_links_modern.jsonl \
  --cache-dir data/sermons/cache --delay 3.0

python experiments/sermon/scrape_sermons.py \
  --years 1777-1899 --links-only \
  --output data/sermons/dansketaler_links_1800s.jsonl \
  --cache-dir data/sermons/cache --delay 3.0

# 2. Fetch the texts (resumable)
python experiments/sermon/scrape_sermons.py \
  --links data/sermons/dansketaler_links_all.jsonl \
  --output data/sermons/dansketaler_sermons_all.jsonl \
  --cache-dir data/sermons/cache --delay 1.2

python experiments/sermon/scrape_sermons.py \
  --links data/sermons/dansketaler_links_1800s.jsonl \
  --output data/sermons/dansketaler_sermons_1800s.jsonl \
  --cache-dir data/sermons/cache --delay 1.2

# 3. Normalize, deduplicate, and license-tag
python experiments/sermon/build_sermon_corpus.py \
  data/sermons/dansketaler_sermons_all.jsonl \
  data/sermons/dansketaler_sermons_1800s.jsonl \
  --output data/sermons/corpus.jsonl

# 4. Sentence-level splits
python experiments/sermon/prepare_sermon_data.py \
  --input data/sermons/corpus.jsonl \
  --output-dir data/sermons/split --test-every 10

python experiments/sermon/prepare_sermon_data.py \
  --input data/sermons/corpus_public_domain.jsonl \
  --output-dir data/sermons/split_public_domain --test-every 5
```

## Files

- `data/sermons/corpus.jsonl` — 1,305 normalized sermons with license tags.
- `data/sermons/corpus.jsonl.manifest.json` — counts by license and decade.
- `data/sermons/corpus_public_domain.jsonl` — 163 public-domain sermons.
- `data/sermons/split/sermon_train_sources.jsonl` — 72,004 train sentences.
- `data/sermons/split/sermon_test_sources.jsonl` — 7,548 held-out sentences.
- `data/sermons/split_public_domain/sermon_train_sources.jsonl` — 8,771.
- `data/sermons/split_public_domain/sermon_test_sources.jsonl` — 2,356.
- `data/sermons/cache/` — raw HTML cache; safe to delete after the corpus is
  built.

## Gaps and next steps

- The year archive yielded 1,311 links; the corpus has 1,305 sermons after
  deduplication. The site reports ~1,335, so roughly 30 sermons are not exposed
  by the year archive or duplicate an existing URL.
- Teacher generation is the next expensive step: 72k sentences at ~1 pair/s is
  about 20 hours on the local Qwen server. Start with a 5k–10k sentence subset
  to validate the larger corpus before committing to the full run.
- The public-domain subset (600k words / 8.8k train sentences) is the safe
  open-release path. For more public-domain data, consider Archive.org Danish
  sermon books or Wikisource; OCR quality and sermon-boundary detection are the
  main issues.
- Test references must be generated only after the split is frozen.
