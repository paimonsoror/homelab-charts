{{- define "telltaledns.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "telltaledns.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{- define "telltaledns.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{ include "telltaledns.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "telltaledns.selectorLabels" -}}
app.kubernetes.io/name: {{ include "telltaledns.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/* Resolver pods: their own name, so the primary's Deployment never selects them. */}}
{{- define "telltaledns.resolverSelectorLabels" -}}
app.kubernetes.io/name: {{ include "telltaledns.name" . }}-resolver
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: resolver
{{- end }}

{{/* Every pod that answers DNS (the primary and the resolver pods): the DNS Service's selector. */}}
{{- define "telltaledns.dnsSelectorLabels" -}}
app.kubernetes.io/instance: {{ .Release.Name }}
telltaledns.sororlab.dev/serves-dns: "true"
{{- end }}
