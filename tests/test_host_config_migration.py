from types import SimpleNamespace

from sensetrace.host.client import RemoteHost


def test_first_system_install_copies_live_fallback_config_without_reset():
    host = RemoteHost.__new__(RemoteHost)
    commands: list[str] = []
    hashes = iter(["missing", "fallback-sha256"])
    host._remote_hash = lambda path: next(hashes)  # type: ignore[method-assign]

    def run(command: str) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(ok=True)

    host.run = run  # type: ignore[method-assign, assignment]

    result = host._preserve_fallback_config(
        fallback_config="/home/worker-03/.config/sensetrace/worker03.yaml",
        system_config="/etc/sensetrace/worker03.yaml",
        reset_config=False,
    )

    assert result == {
        "before": "fallback-sha256",
        "source": "user-fallback",
        "system_before": "missing",
    }
    assert commands == [
        "sudo -n install -m 0640 -o worker-03 -g worker-03 "
        "/home/worker-03/.config/sensetrace/worker03.yaml /etc/sensetrace/worker03.yaml"
    ]


def test_explicit_reset_does_not_copy_fallback_config():
    host = RemoteHost.__new__(RemoteHost)
    commands: list[str] = []
    hashes = iter(["missing", "fallback-sha256"])
    host._remote_hash = lambda path: next(hashes)  # type: ignore[method-assign]

    def run(command: str) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(ok=True)

    host.run = run  # type: ignore[method-assign, assignment]

    result = host._preserve_fallback_config(
        fallback_config="/home/worker-03/.config/sensetrace/worker03.yaml",
        system_config="/etc/sensetrace/worker03.yaml",
        reset_config=True,
    )

    assert result == {"before": "missing", "source": "missing", "system_before": "missing"}
    assert commands == []
