#!/usr/bin/env python3
"""Version bump tool for the argocd-apps app-of-apps values.yaml.

Two different things in charts/argocd-apps/values.yaml count as a "version":

  chart  -- spec.source.targetRevision, a real YAML scalar.
  image  -- an `image.tag:` somewhere inside the `helm.values: |-` *string
            block*. That block is not addressable by yq, so every edit here
            is a surgical line rewrite that preserves the original
            indentation, quote style and trailing comment.

Subcommands:
  list                          every app, its enabled flag and version fields
  show   <app>                  one app's version fields, in detail
  check  [--app X]              query upstream for newer versions
  set    <app> --to V [--kind]  rewrite one version field in place

`set` only touches values.yaml. Branching, the PR and the ArgoCD sync are
handled by .github/workflows/version-bump.yml and the in-cluster
argocd-sync-watcher respectively -- see tools/README.md.
"""

import argparse
import json
import os
import re
import sys
import urllib.request

VALUES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "charts", "argocd-apps", "values.yaml",
)

# Image repositories that are NOT written in values.yaml because the app
# relies on its chart's default value. `check` cannot guess these, so they
# are declared here -- same spirit as OVERRIDES in manifests/backstage-catalog.
KNOWN_IMAGES = {
    "technitium": "technitium/dns-server",
}

# Per-app, per-kind upstream overrides, for the cases automatic resolution
# gets wrong. Everything not listed here resolves on its own: a chart from
# its helm repo's index.yaml, an image from Docker Hub or GHCR.
#
#   ("github-releases", "<owner>/<repo>")  -- read tags from GitHub releases
#   ("none", "<reason shown in check output>")
#
# immich publishes thousands of commit-/pr- tags to GHCR and its registry
# path (immich-server) differs from its repo name, so releases are both
# faster and correct.
UPSTREAM = {
    "immich": {"image": ("github-releases", "immich-app/immich")},
    "immich-power-tools": {
        "chart": ("none", "chart version does not map to upstream releases")},
}

# targetRevision values that mean "track a branch", not "pin a version".
FLOATING = {"HEAD", "main", "master", "trunk"}

PRERELEASE = re.compile(r"(rc|alpha|beta|dev|pre|snapshot|nightly|canary)", re.I)
VERSION_RE = re.compile(r"^v?(\d+(?:\.\d+)*)$")
SCALAR_RE = re.compile(r"^((?:\"[^\"]*\"|'[^']*'|[^#\"'])*?)(\s+#.*)?$")

UA = {"User-Agent": "homelab-charts-bump/1.0"}


# --------------------------------------------------------------------------
# values.yaml parsing
# --------------------------------------------------------------------------

class Field:
    """One editable version field: which line it is, and how to rewrite it."""

    def __init__(self, kind, path, line_no, value, raw):
        self.kind = kind        # "chart" or "image"
        self.path = path        # e.g. "targetRevision" or "image.tag"
        self.line_no = line_no  # 0-based index into the file's lines
        self.value = value      # current value, unquoted
        self.raw = raw          # the original line, verbatim


class App:
    def __init__(self, name, start):
        self.name = name
        self.start = start
        self.end = None
        self.enabled = None
        self.repo_url = None
        self.chart = None
        self.fields = []

    def field(self, kind=None, path=None):
        """Pick exactly one field, erroring clearly when the choice is ambiguous.

        Without an exact `path`, only version-bearing fields are candidates --
        an app's `image.repository` is parsed (so `check` knows where to look)
        but is never what `--kind image` means.
        """
        found = [f for f in self.fields
                 if (kind is None or f.kind == kind)
                 and (path == f.path if path else
                      f.kind == "chart" or f.path.endswith("tag"))]
        if not found:
            avail = ", ".join("%s:%s" % (f.kind, f.path) for f in self.fields)
            raise SystemExit("%s: no %s field found. Available: %s"
                             % (self.name, path or kind or "version", avail or "none"))
        if len(found) > 1:
            raise SystemExit("%s: %d candidate fields, pass --path to choose: %s"
                             % (self.name, len(found),
                                ", ".join(f.path for f in found)))
        return found[0]

    @property
    def image_repo(self):
        """Fully qualified image repository, e.g. ghcr.io/paimonsoror/backstage.

        Charts split this two ways: immich writes the registry into
        `repository` itself, backstage keeps a separate `registry` key. Both
        have to end up qualified or the lookup goes to the wrong registry.
        """
        repo = registry = None
        for f in self.fields:
            if f.kind != "image":
                continue
            if f.path.endswith("repository"):
                repo = f.value
            elif f.path.endswith("registry"):
                registry = f.value
        if repo and registry and not repo.startswith(registry + "/"):
            repo = registry.rstrip("/") + "/" + repo
        return repo or KNOWN_IMAGES.get(self.name)


def split_scalar(text):
    """Split a scalar into (value, quote_char, trailing_comment)."""
    comment = ""
    m = SCALAR_RE.match(text)
    if m:
        text, comment = m.group(1), m.group(2) or ""
    text = text.strip()
    quote = ""
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        quote, text = text[0], text[1:-1]
    return text, quote, comment


def parse(path=VALUES):
    lines = open(path).read().splitlines()
    apps = []
    cur = None
    in_helm_values = False
    block_indent = 14
    stack = []  # (indent, key) pairs, for dotted paths inside the block scalar

    for i, line in enumerate(lines):
        m = re.match(r"^    - name: (\S+)\s*$", line)
        if m:
            if cur:
                cur.end = i
            cur = App(m.group(1), i)
            apps.append(cur)
            in_helm_values = False
            stack = []
            continue
        if cur is None:
            continue

        if in_helm_values:
            if line.strip() and not line.startswith(" " * block_indent):
                in_helm_values = False   # dedented back out of the block scalar
            else:
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    km = re.match(r"^\s*-?\s*([\w.\-/]+):\s*(.*)$", line)
                    if km:
                        while stack and stack[-1][0] >= indent:
                            stack.pop()
                        key, rest = km.group(1), km.group(2)
                        dotted = ".".join([s[1] for s in stack] + [key])
                        if key in ("tag", "repository", "registry"):
                            val = split_scalar(rest)[0]
                            cur.fields.append(Field("image", dotted, i, val, line))
                        if not rest.strip():
                            stack.append((indent, key))
                continue

        em = re.match(r"^      enabled:\s*(.*)$", line)
        if em:
            cur.enabled = split_scalar(em.group(1))[0]
            continue
        sm = re.match(r"^          (repoURL|targetRevision|chart):\s*(.*)$", line)
        if sm:
            key, val = sm.group(1), split_scalar(sm.group(2))[0]
            if key == "repoURL":
                cur.repo_url = val
            elif key == "chart":
                cur.chart = val
            else:
                cur.fields.append(Field("chart", "targetRevision", i, val, line))
            continue
        if re.match(r"^            values:\s*\|-?\s*$", line):
            in_helm_values = True
            stack = []
            continue

    if cur:
        cur.end = len(lines)
    return lines, apps


def find_app(apps, name):
    for a in apps:
        if a.name == name:
            return a
    raise SystemExit("no app named %r. Try: bump.py list" % name)


def is_enabled(app):
    return app.enabled not in ("false", "False", None, "")


# --------------------------------------------------------------------------
# upstream version lookup
# --------------------------------------------------------------------------

def http_get(url, headers=None, timeout=25):
    hdrs = dict(UA)
    hdrs.update(headers or {})
    req = urllib.request.Request(url, headers=hdrs)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def parse_version(tag):
    m = VERSION_RE.match(tag or "")
    if not m or PRERELEASE.search(tag):
        return None
    return tuple(int(p) for p in m.group(1).split("."))


def newest(tags):
    best, best_key = None, None
    for t in tags:
        key = parse_version(t)
        if key and (best_key is None or key > best_key):
            best, best_key = t, key
    return best


def helm_chart_versions(repo_url, chart):
    import yaml
    index = yaml.safe_load(http_get(repo_url.rstrip("/") + "/index.yaml"))
    entries = (index or {}).get("entries", {}).get(chart, [])
    return [e["version"] for e in entries if "version" in e]


def dockerhub_tags(repo):
    if "/" not in repo:
        repo = "library/" + repo
    url = ("https://hub.docker.com/v2/repositories/%s/tags"
           "?page_size=100&ordering=last_updated" % repo)
    return [t["name"] for t in json.loads(http_get(url)).get("results", [])]


def ghcr_tags(path, max_pages=10):
    """GHCR pages its tag list; follow the Link header, but not forever."""
    tok = json.loads(http_get(
        "https://ghcr.io/token?scope=repository:%s:pull" % path))["token"]
    url = "https://ghcr.io/v2/%s/tags/list?n=1000" % path
    auth = {"Authorization": "Bearer " + tok}
    tags = []
    for _ in range(max_pages):
        req = urllib.request.Request(url, headers=dict(UA, **auth))
        with urllib.request.urlopen(req, timeout=25) as r:
            tags.extend(json.loads(r.read()).get("tags", []))
            link = r.headers.get("Link", "")
        m = re.search(r"<([^>]+)>;\s*rel=\"next\"", link)
        if not m:
            break
        url = "https://ghcr.io" + m.group(1)
    return tags


def github_release_tags(slug):
    url = "https://api.github.com/repos/%s/releases?per_page=100" % slug
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")  # lifts the 60/hr anonymous limit
    if token:
        headers["Authorization"] = "Bearer " + token
    return [r["tag_name"] for r in json.loads(http_get(url, headers))
            if not r.get("prerelease") and not r.get("draft")]


def image_tags(repo):
    """Dispatch on the registry embedded in the repository string."""
    if repo.startswith("ghcr.io/"):
        return ghcr_tags(repo[len("ghcr.io/"):])
    if repo.startswith(("docker.io/", "index.docker.io/")):
        repo = repo.split("/", 1)[1]
    if re.match(r"^[^/]+\.[^/]+/", repo):
        raise RuntimeError("unsupported registry for %s" % repo)
    return dockerhub_tags(repo)


def github_slug(repo_url):
    m = re.match(r"^https?://github\.com/([^/]+/[^/]+?)(?:\.git)?/?$", repo_url or "")
    return m.group(1) if m else None


def resolve_source(app, field):
    """Decide where a field's upstream versions come from.

    -> (callable returning a tag list, note) or (None, reason it can't be done)
    """
    override = UPSTREAM.get(app.name, {}).get(field.kind)
    if override:
        how, arg = override
        if how == "none":
            return None, arg
        if how == "github-releases":
            return (lambda: github_release_tags(arg)), ""
        raise SystemExit("unknown UPSTREAM source %r" % how)

    if field.kind == "chart":
        if field.value in FLOATING:
            return None, "tracks %s, not a pinned version" % field.value
        slug = github_slug(app.repo_url)
        if slug:
            return (lambda: github_release_tags(slug)), ""
        if app.chart and (app.repo_url or "").startswith("http"):
            return (lambda: helm_chart_versions(app.repo_url, app.chart)), ""
        return None, "not a helm repo"

    repo = app.image_repo
    if not repo:
        return None, "no repository in values.yaml; add it to KNOWN_IMAGES"
    if not parse_version(field.value):
        return None, "not a pinned version"
    return (lambda: image_tags(repo)), ""


def check_app(app):
    """-> list of (kind, path, current, latest, note) tuples."""
    out = []
    for f in app.fields:
        if f.kind != "chart" and not f.path.endswith("tag"):
            continue
        fetch, note = resolve_source(app, f)
        if fetch is None:
            out.append((f.kind, f.path, f.value, None, note))
            continue
        try:
            out.append((f.kind, f.path, f.value, newest(fetch()), note))
        except Exception as e:
            out.append((f.kind, f.path, f.value, None, "lookup failed: %s" % e))
    return out


def is_newer(current, latest):
    c, l = parse_version(current), parse_version(latest)
    return bool(c and l and l > c)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_list(args):
    _lines, apps = parse(args.values)
    for a in apps:
        flag = "on " if is_enabled(a) else "off"
        vers = ", ".join("%s=%s" % (f.path, f.value) for f in a.fields
                         if f.kind == "chart" or f.path.endswith("tag")) or "-"
        print("%s  %-26s %s" % (flag, a.name, vers))


def cmd_show(args):
    _lines, apps = parse(args.values)
    a = find_app(apps, args.app)
    print("app         %s" % a.name)
    print("enabled     %s" % a.enabled)
    print("repoURL     %s" % a.repo_url)
    print("chart       %s" % (a.chart or "-"))
    print("image repo  %s" % (a.image_repo or "-"))
    print("fields:")
    for f in a.fields:
        print("  %-6s %-28s %-14s (values.yaml:%d)"
              % (f.kind, f.path, f.value, f.line_no + 1))


def cmd_check(args):
    _lines, apps = parse(args.values)
    if args.app:
        apps = [find_app(apps, args.app)]
    elif not args.all:
        apps = [a for a in apps if is_enabled(a)]

    rows = []
    for a in apps:
        for kind, path, cur, latest, note in check_app(a):
            rows.append({
                "app": a.name, "kind": kind, "path": path, "current": cur,
                "latest": latest, "outdated": is_newer(cur, latest), "note": note,
            })
    if args.only_outdated:
        rows = [r for r in rows if r["outdated"]]

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    print("%-26s %-6s %-14s %-14s NOTE" % ("APP", "KIND", "CURRENT", "LATEST"))
    for r in rows:
        mark = " *" if r["outdated"] else "  "
        print("%-26s %-6s %-14s %-14s%s%s"
              % (r["app"], r["kind"], r["current"], r["latest"] or "-",
                 mark, r["note"]))
    n = sum(1 for r in rows if r["outdated"])
    sys.stderr.write("\n%d upgrade(s) available (marked *)\n" % n)
    return 0


def cmd_set(args):
    lines, apps = parse(args.values)
    a = find_app(apps, args.app)
    f = a.field(kind=args.kind, path=args.path)

    if f.value == args.to:
        print("%s: %s already %s, nothing to do" % (a.name, f.path, args.to))
        return 0

    # Rewrite only the scalar, keeping indentation, quote style and any
    # trailing comment exactly as they were.
    key, _sep, rest = f.raw.partition(":")
    _val, quote, comment = split_scalar(rest)
    new = "%s: %s%s%s%s" % (key, quote, args.to, quote, comment)

    if args.dry_run:
        print("--- %s\n+++ %s" % (f.raw, new))
        return 0

    lines[f.line_no] = new
    with open(args.values, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("%s: %s %s -> %s (values.yaml:%d)"
          % (a.name, f.path, f.value, args.to, f.line_no + 1))

    # Consumed by version-bump.yml to build the commit/PR, and -- via the
    # Argocd-Sync commit trailer it writes -- to tell the in-cluster
    # argocd-sync-watcher which child Application to sync after merge.
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a") as fh:
            fh.write("app=%s\nfield=%s\nfrom=%s\nto=%s\n"
                     % (a.name, f.path, f.value, args.to))
    return 0


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--values", default=VALUES, help="path to values.yaml")
    sub = p.add_subparsers(dest="cmd")
    sub.required = True

    sub.add_parser("list").set_defaults(func=cmd_list)

    s = sub.add_parser("show")
    s.add_argument("app")
    s.set_defaults(func=cmd_show)

    c = sub.add_parser("check")
    c.add_argument("--app")
    c.add_argument("--all", action="store_true", help="include disabled apps")
    c.add_argument("--only-outdated", action="store_true")
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_check)

    s = sub.add_parser("set")
    s.add_argument("app")
    s.add_argument("--to", required=True)
    s.add_argument("--kind", choices=["chart", "image"])
    s.add_argument("--path", help="exact field path, e.g. image.tag")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_set)

    args = p.parse_args()
    sys.exit(args.func(args) or 0)


if __name__ == "__main__":
    main()
