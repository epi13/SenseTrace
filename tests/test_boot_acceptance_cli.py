from sensetrace.cli import build_parser


def test_reboot_acceptance_defaults_to_the_multi_user_baseline():
    args = build_parser().parse_args(["host", "reboot-acceptance", "worker-03"])

    assert args.require_appliance is False


def test_reboot_acceptance_can_require_the_dedicated_appliance_target():
    args = build_parser().parse_args(
        ["host", "reboot-acceptance", "worker-03", "--require-appliance"]
    )

    assert args.require_appliance is True
