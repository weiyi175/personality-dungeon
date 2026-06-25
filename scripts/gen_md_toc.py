#!/usr/bin/env python3
"""Generate / regenerate a linked Markdown TOC in place (one-click).

- Code-fence aware: ``` / ~~~ fenced blocks are skipped, so bash comments like
  "# Server ..." are NOT mistaken for headings.
- Anchors follow github-slugger (works on GitHub web + modern VSCode preview):
  lowercase, strip punctuation/symbols/emoji (keep letters incl. CJK, digits,
  underscore, space, hyphen), spaces -> '-', duplicate slugs get -1/-2 suffixes.
- Idempotent via markers; the depth is stored in the start marker so a bare
  re-run after editing the doc regenerates the same TOC:

    <!-- TOC START depth=2 -->
    ## <title>

    - [Heading](#heading)
    <!-- TOC END -->

Usage:
  # first insert (no markers yet):
  gen_md_toc.py FILE --before "## Some Heading" --title "📑 目錄" --depth 2
  # regenerate after editing (markers present; depth/title read from the block):
  gen_md_toc.py FILE
  # ...with overrides:
  gen_md_toc.py FILE --depth 3 --title "目錄"

TOC lists headings from level 2 down to --depth (the document H1 title is
skipped and remaining levels are de-indented so level 2 sits flush-left).
"""
import argparse
import re
import sys

START_RE = re.compile(r"<!--\s*TOC START(?:\s+depth=(\d+))?\s*-->")
END = "<!-- TOC END -->"
# Match only column-0 fences. Indented ``` are treated as content (they occur
# inside col-0 code blocks / list items); a real section heading is always at
# column 0, so it can never live inside an indented fence — this keeps fence
# parity in sync and never drops real headings nor picks up bash comments.
FENCE_RE = re.compile(r"^(```|~~~)")
HEAD_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def make_slugger():
    seen = {}

    def slug(text):
        s = re.sub(r"\s", "-", re.sub(r"[^\w\s\-]", "", text.strip().lower(), flags=re.UNICODE))
        if s not in seen:
            seen[s] = 0
            return s
        seen[s] += 1
        new = f"{s}-{seen[s]}"
        while new in seen:
            seen[s] += 1
            new = f"{s}-{seen[s]}"
        seen[new] = 0
        return new

    return slug


def build_toc(lines, depth):
    """TOC lines for headings up to `depth`. Every heading is slugged (to keep
    dedup counters aligned with the renderer); headings inside an existing TOC
    block and out-of-range levels are not emitted, and the document title (the
    first level-1 heading) is skipped. Remaining levels are de-indented so the
    shallowest emitted level sits flush-left."""
    slug = make_slugger()
    items = []  # (level, text, anchor)
    in_fence = in_toc = title_skipped = False
    for line in lines:
        if START_RE.search(line):
            in_toc = True
            continue
        if END in line:
            in_toc = False
            continue
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = HEAD_RE.match(line)
        if not m:
            continue
        text = m.group(2).strip()
        if not text:
            continue
        level = len(m.group(1))
        anchor = slug(text)  # always advance dedup, in document order
        if in_toc or level > depth:
            continue
        if not title_skipped and level == 1:
            title_skipped = True
            continue
        items.append((level, text, anchor))
    if not items:
        return []
    min_level = min(lv for lv, _, _ in items)
    return ["  " * (lv - min_level) + f"- [{t}](#{a})" for lv, t, a in items]


def main():
    ap = argparse.ArgumentParser(description="Generate/regenerate a linked Markdown TOC in place.")
    ap.add_argument("file")
    ap.add_argument("--depth", type=int, help="deepest heading level to include (default 3, or read from marker)")
    ap.add_argument("--title", help="TOC section heading text (default: keep existing, or '目錄')")
    ap.add_argument("--before", help="text to insert the new TOC before (first insert only)")
    args = ap.parse_args()

    text = open(args.file, encoding="utf-8").read()
    lines = text.splitlines()
    m = START_RE.search(text)

    if m:  # regenerate in place
        depth = args.depth or (int(m.group(1)) if m.group(1) else 3)
        start_idx = text.index(m.group(0))
        end_idx = text.index(END, start_idx)
        title = args.title
        if not title:
            tm = re.search(r"^\s*#{1,6}\s+(.*?)\s*$", text[start_idx:end_idx], re.M)
            title = tm.group(1).strip() if tm else "目錄"
        toc = build_toc(lines, depth)
        block = f"<!-- TOC START depth={depth} -->\n## {title}\n\n" + "\n".join(toc) + f"\n{END}"
        new = text[:start_idx] + block + text[end_idx + len(END):]
    else:  # first insert
        if not args.before or not args.title:
            sys.exit("error: first insert needs --before and --title")
        depth = args.depth or 3
        toc = build_toc(lines, depth)
        block = f"<!-- TOC START depth={depth} -->\n## {args.title}\n\n" + "\n".join(toc) + f"\n{END}\n\n"
        if args.before not in text:
            sys.exit(f"error: --before anchor not found: {args.before!r}")
        idx = text.index(args.before)
        new = text[:idx] + block + text[idx:]

    open(args.file, "w", encoding="utf-8").write(new)
    print(f"{args.file}: TOC written ({len(toc)} entries, depth={depth})")


if __name__ == "__main__":
    main()
