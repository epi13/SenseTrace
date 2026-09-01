from pathlib import Path


def test_reboot_uses_a_transient_systemd_timer_not_shell_backgrounding():
    client = Path("src/sensetrace/host/client.py").read_text()
    reboot = client[client.index("    def reboot(self)") :]

    assert "systemd-run --quiet --collect --on-active=5s" in reboot
    assert "nohup sh -c 'sleep 1" not in reboot
