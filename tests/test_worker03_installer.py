from pathlib import Path


def test_root_installer_creates_a_runner_writable_service_journal_before_starting():
    script = Path("infra/worker03/install-root.sh").read_text()

    journal_setup = script.index("touch /var/lib/sensetrace/state/service-events.jsonl")
    service_start = script.index("systemctl enable --now sensetrace.service")
    assert journal_setup < service_start
    assert "chown worker-03:worker-03 /var/lib/sensetrace/state/service-events.jsonl" in script
    assert "chmod 0640 /var/lib/sensetrace/state/service-events.jsonl" in script
