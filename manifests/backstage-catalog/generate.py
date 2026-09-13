import subprocess
import json
import sys

# Apps to skip: the app-of-apps meta application, infra-only manifests (no
# single "service" workload), and apps that already have their own
# hand-authored catalog-info.yaml in their own repo.
SKIP = {"argocd-apps", "backstage-infra", "backstage", "mila-spell-tumble", "technitium"}

# type/description overrides for apps where "service" + a generic description
# isn't quite right or could use more context. Everything else defaults to
# type: service with a generic description.
OVERRIDES = {
    "cert-manager": {"description": "Issues and renews TLS certificates cluster-wide via Let's Encrypt."},
    "cloudnative-pg-operator": {"description": "CloudNativePG operator -- manages the Postgres Clusters used by immich, authentik, and backstage."},
    "vector": {"description": "Log pipeline: ingests UniFi syslog + fluent-bit forwarded logs, ships to Loki."},
    "fluentbit": {"description": "Collects container logs cluster-wide, forwards to vector."},
    "loki": {"description": "Log storage/query backend (loki-distributed chart -- deprecated, pending migration to the unified loki chart)."},
    "prometheus": {"description": "kube-prometheus-stack: Prometheus, Alertmanager, and Grafana."},
    "prom-push-gateway": {"description": "Prometheus Pushgateway, for batch/cron job metrics."},
    "pihole": {"description": "Legacy DNS filtering -- superseded by Technitium, kept running but not actively maintained."},
    "technitium-exporter": {"description": "Prometheus exporter for Technitium DNS stats."},
}

result = subprocess.run(
    ["kubectl", "-n", "argocd", "get", "applications.argoproj.io", "-o", "json"],
    capture_output=True, text=True, check=True,
)
apps = json.loads(result.stdout)["items"]

entities = []
for app in apps:
    name = app["metadata"]["name"]
    if name in SKIP:
        continue
    namespace = app["spec"]["destination"]["namespace"]
    override = OVERRIDES.get(name, {})
    description = override.get("description", f"Deployed via ArgoCD in the {namespace} namespace.")
    entities.append({
        "apiVersion": "backstage.io/v1alpha1",
        "kind": "Component",
        "metadata": {
            "name": name,
            "description": description,
            "annotations": {
                "argocd/app-name": name,
                "backstage.io/kubernetes-label-selector": f"app.kubernetes.io/instance={name}",
            },
        },
        "spec": {
            "type": override.get("type", "service"),
            "owner": "group:sororlab",
            "lifecycle": "production",
        },
    })

entities.sort(key=lambda e: e["metadata"]["name"])

out = sys.argv[1] if len(sys.argv) > 1 else "/dev/stdout"
with open(out, "w") as f:
    for e in entities:
        f.write("---\n")
        # Minimal, deterministic YAML writer -- avoids a PyYAML dependency
        # and keeps formatting stable across regenerations for clean diffs.
        f.write(f"apiVersion: {e['apiVersion']}\n")
        f.write(f"kind: {e['kind']}\n")
        f.write("metadata:\n")
        f.write(f"  name: {e['metadata']['name']}\n")
        f.write(f"  description: {json.dumps(e['metadata']['description'])}\n")
        f.write("  annotations:\n")
        for k, v in e["metadata"]["annotations"].items():
            f.write(f"    {k}: {v}\n")
        f.write("spec:\n")
        f.write(f"  type: {e['spec']['type']}\n")
        f.write(f"  owner: {e['spec']['owner']}\n")
        f.write(f"  lifecycle: {e['spec']['lifecycle']}\n")

print(f"Generated {len(entities)} entities -> {out}", file=sys.stderr)
