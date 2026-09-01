#!/usr/bin/env bash
set -euo pipefail

# Run this script as root only after a fresh SSH connection has been validated.
# It intentionally leaves the graphical target unchanged; headless boot is a
# separate, reversible operator decision.
project_dir=${1:-/opt/sensetrace/source}
reset_config=${2:-0}
install -d -m 0750 -o worker-03 -g worker-03 /opt/sensetrace /etc/sensetrace /var/lib/sensetrace
install -d -m 0750 -o worker-03 -g worker-03 /var/lib/sensetrace/data /var/lib/sensetrace/runs /var/lib/sensetrace/state
install -m 0644 "$project_dir/infra/worker03/sensetrace.service" /etc/systemd/system/sensetrace.service
install -m 0644 "$project_dir/infra/worker03/sensetrace.target" /etc/systemd/system/sensetrace.target
install -m 0644 "$project_dir/infra/worker03/90-sensetrace.conf" /etc/sysctl.d/90-sensetrace.conf
install -m 0644 "$project_dir/infra/worker03/tmpfiles.conf" /etc/tmpfiles.d/sensetrace.conf
if [[ ! -e /etc/sensetrace/worker03.yaml || "$reset_config" == 1 ]]; then
  install -m 0640 -o worker-03 -g worker-03 "$project_dir/configs/worker03.example.yaml" /etc/sensetrace/worker03.yaml
  config_action=initialized
else
  config_action=preserved
fi
systemd-tmpfiles --create /etc/tmpfiles.d/sensetrace.conf
sysctl --system
systemctl daemon-reload
systemctl enable --now sensetrace.service
printf '{"event":"service_migration","mode":"system","config_action":"%s","timestamp_utc":"%s"}\n' "$config_action" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> /var/lib/sensetrace/state/service-events.jsonl
systemctl --no-pager --full status sensetrace.service
