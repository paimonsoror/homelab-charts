# Version upgrades

One path for changing the version of anything in the app-of-apps:

```
Backstage "Upgrade an app"  ──dispatch──▶  .github/workflows/version-bump.yml
                                                    │ runs tools/bump.py set
                                                    │ helm template (guard)
                                                    ▼
                                           pull request  ──review + merge──┐
                                                                           │
  argocd-sync-watcher CronJob (in cluster, every 2 min)  ◀──────────────────┘
      hard-refresh argocd-apps → wait for Synced → sync the named child
```

Nothing outside the cluster can reach ArgoCD (`letsencrypt-prod` uses DNS-01
via Route53, nothing is exposed inbound), so the last hop polls rather than
being pushed. That is the only reason the watcher exists.

## tools/bump.py

Versions live in two different shapes in `charts/argocd-apps/values.yaml`:

| shape | where | example |
|---|---|---|
| chart | `spec.source.targetRevision`, real YAML | `authentik` → `2026.8.2` |
| image | an `image.tag:` inside the `helm.values: \|-` **string block** | `technitium` → `15.4.0` |

The second is not addressable by `yq` — it is a string as far as YAML is
concerned. So every edit is a surgical line rewrite that preserves the
original indentation, quote style and trailing comment. (`vaultwarden`'s tag
carries a `# pin instead of "latest"` comment; it survives a bump.)

```bash
python3 tools/bump.py list                       # every app and its version fields
python3 tools/bump.py show technitium            # one app, with line numbers
python3 tools/bump.py check                      # what is out of date upstream
python3 tools/bump.py check --only-outdated --json
python3 tools/bump.py set technitium --to 15.5.0 --kind image --dry-run
```

`check` resolves each field on its own: a chart from its helm repo's
`index.yaml`, a chart whose `repoURL` is a GitHub repo from that repo's
releases, an image from Docker Hub or GHCR. Two tables at the top of the
script handle what cannot be resolved automatically:

- **`KNOWN_IMAGES`** — apps that rely on their chart's default image and so
  have no `repository:` in values.yaml. `technitium` is the only one today.
  Add to it if `check` reports *"no repository in values.yaml"*.
- **`UPSTREAM`** — per-app, per-kind overrides. `immich` is there because it
  publishes thousands of `commit-`/`pr-` tags to GHCR and its registry path
  (`immich-server`) differs from its repo name, so GitHub releases is both
  faster and correct.

Apps pinned to `HEAD`/`main` report *"tracks HEAD, not a pinned version"* —
that is accurate, not a gap.

`check` uses `GITHUB_TOKEN` when set, which lifts the anonymous 60 req/hour
limit on the releases API.

## .github/workflows/version-bump.yml

`workflow_dispatch` with inputs `app`, `version`, `kind`, `path`, `reason`,
`requested_by`. Runs `bump.py set`, fails if nothing changed, renders
`helm template charts/argocd-apps` as a guard, then commits and opens a PR.

The commit follows this repo's existing convention —
`<app>: upgrade X -> Y` — and carries the trailer the watcher reads:

```
Argocd-Sync: technitium
```

Dispatch it from Backstage, from the Actions tab, or:

```bash
gh workflow run version-bump.yml -R paimonsoror/homelab-charts \
  -f app=technitium -f version=15.5.0 -f kind=image -f reason="match the pi"
```

**Approval** is the PR review. To make it mandatory rather than conventional,
turn on a branch protection rule on `main` requiring one approving review —
the workflow does not enforce it on its own.

## manifests/argocd-sync-watcher

A CronJob in the `argocd` namespace, every two minutes:

1. Annotates `argocd-apps` with `argocd.argoproj.io/refresh=hard` and waits
   for ArgoCD to clear the annotation — the parent must re-read git before
   any child's spec is current.
2. Compares `.status.sync.revision` against the last one it recorded in the
   `argocd-sync-watcher-state` ConfigMap. Unchanged → exits.
3. Reads the new commit's `Argocd-Sync:` trailer from the GitHub API and syncs
   exactly those Applications, waiting for each operation to reach a terminal
   phase and logging `phase=` and `health=`.

Precision is the point of step 3. Syncing everything `OutOfSync` would also
pick up apps deliberately left unsynced, so with no trailer it syncs nothing
and says so. Set `FALLBACK=outofsync` on the CronJob to opt into the broad
behaviour.

It uses a `Role`, not a `ClusterRole`: `get`/`list`/`patch` on Applications
plus its own state ConfigMap, all in `argocd`. It never needs access to what
it syncs, and it needs no ArgoCD API token — it drives the CRD directly.

Hand-edited commits pushed from the box work too, as long as you add the
trailer:

```bash
git commit -m "pihole: upgrade 2.35.0 -> 2.38.0" -m "Argocd-Sync: pihole"
```

Without it, apps with `syncPolicy.automated` still pick the change up on
ArgoCD's own poll; apps without it stay OutOfSync until synced by hand.

**Logs:**

```bash
kubectl -n argocd get cronjob argocd-sync-watcher
kubectl -n argocd logs -l batch.kubernetes.io/job-name --tail=50 --prefix
kubectl -n argocd get cm argocd-sync-watcher-state -o jsonpath='{.data.revision}'
```

On its very first run it records the current revision as a baseline and syncs
nothing — otherwise it would sync whatever the last commit happened to be.

## Backstage template

`backstage/templates/app-upgrade/template.yaml` in this repo. The scaffolder
cannot do a surgical edit inside a block scalar, so the template dispatches
the workflow instead of writing the file itself — one implementation of the
edit logic, living in the repo it edits.

Registering it needs a location entry in the Backstage repo's
`app-config.production.yaml`, the same shape Courier's template uses:

```yaml
    - type: url
      target: https://github.com/paimonsoror/homelab-charts/blob/main/backstage/templates/app-upgrade/template.yaml
      rules:
        - allow: [Template]
```

`rules` is required — the global `catalog.rules` do not allow `Template`, so
without it the entity is silently rejected. Config is baked into the image, so
this needs a Backstage rebuild + redeploy (~3-5 min); it cannot be picked up
by a restart alone.
