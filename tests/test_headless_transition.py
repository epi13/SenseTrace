from types import SimpleNamespace

from sensetrace.host.client import RemoteHost


def test_headless_transition_accepts_an_absent_display_manager_alias(monkeypatch):
    host = RemoteHost.__new__(RemoteHost)
    host.alias = "worker-03"
    host.project_root = "."
    profiles = iter(
        [
            {"default_target": "graphical.target", "display_manager_active": "inactive"},
            {"default_target": "multi-user.target", "display_manager_active": "inactive"},
        ]
    )
    commands: list[str] = []
    host.sudo_available = lambda: True  # type: ignore[method-assign]
    host.boot_profile = lambda: next(profiles)  # type: ignore[method-assign]
    host.run = lambda command, **kwargs: commands.append(command) or SimpleNamespace(ok=False)  # type: ignore[method-assign]

    class FreshConnection:
        connection = SimpleNamespace(close=lambda: None)

        def run(self, command, **kwargs):
            return SimpleNamespace(ok=True)

    monkeypatch.setattr("sensetrace.host.client.RemoteHost", lambda *args, **kwargs: FreshConnection())

    result = host.set_headless()

    assert result["after"]["default_target"] == "multi-user.target"
    assert "sudo -n systemctl stop display-manager.service" in commands
