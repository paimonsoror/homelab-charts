# backstage-catalog

`cluster-apps.yaml` is generated, not hand-authored. It creates a Backstage
Component entity for every ArgoCD Application in this cluster that doesn't
have its own repo to hold a `catalog-info.yaml` (i.e. every third-party Helm
chart app -- cert-manager, authentik, prometheus, etc). Each entity is
annotated with `argocd/app-name` and a `backstage.io/kubernetes-label-selector`
(`app.kubernetes.io/instance=<name>`, the label every Helm chart in this
cluster has been confirmed to set) so the Backstage Kubernetes and ArgoCD
plugin tabs show real version/sync data for it.

**Regenerate after adding/removing/renaming an app** in
`charts/argocd-apps/values.yaml`:

```bash
python3 generate.py /path/to/cluster-apps.yaml
```

Requires `kubectl` access to the cluster (reads live ArgoCD Application
objects, not the values.yaml directly, so the namespace/name pairing is
always accurate). Apps to exclude, and any per-app description/type
overrides, are hardcoded at the top of `generate.py` -- update `SKIP` if a
new app gets its own repo (and thus its own catalog-info.yaml) later.
