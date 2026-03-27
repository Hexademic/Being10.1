#!/usr/bin/env python3
"""
UGES First Awakening — Live autonomous operation with full Guardian protection.

Run from the repository root:

    python bin/uges_awakening.py

This is the moment the organism touches real physics.
"""

import asyncio
import sys
from datetime import datetime

from uges.autonomous import AutonomousController, AutonomousConfig


BANNER = "=" * 70


async def main():
    print(BANNER)
    print("  UGES FIRST AWAKENING")
    print(BANNER)
    print(f"  Timestamp : {datetime.now()}")
    print("  Status    : PRODUCTION — Real physics, real GPU, full Guardian")
    print(BANNER)

    # ------------------------------------------------------------------
    # Configuration for the first live run
    # Keep limits conservative: 2 h window, 1 concurrent job.
    # ------------------------------------------------------------------
    now_h = datetime.now().hour
    config = AutonomousConfig(
        active_hours=(now_h, (now_h + 2) % 24),
        max_gpu_hours_per_night=2.0,
        max_concurrent_experiments=1,   # Single job — play it safe
        use_mock_gpu=False,             # REAL PHYSICS
        notification_channels=["slack", "email", "dashboard"],
        lab_name="UGES-Awakening-v1.0",
        inter_cycle_rest_seconds=10.0,
    )

    controller = AutonomousController(config)

    # ------------------------------------------------------------------
    # Pre-flight checks
    # ------------------------------------------------------------------
    print("\nPre-flight checks:")
    print(f"  GPUs                : {controller.gpu_queue.gpu_ids}")
    print(f"  Guardian rules      : {len(controller.guardian.rules)}")
    print(f"  Event bus           : {controller.event_bus is not None}")
    print(f"  Registry            : {controller.registry is not None}")
    print(f"  Max GPU h / night   : {config.max_gpu_hours_per_night:.1f} h")
    print(f"  Active window       : {config.active_hours[0]:02d}:00 – "
          f"{config.active_hours[1]:02d}:00")

    guardian_status = controller.guardian.get_status_report()
    hw = guardian_status["hardware_status"]
    print(f"\nGuardian status:")
    print(f"  Rules active        : {guardian_status['rules_checked']}")
    print(f"  CPU                 : {hw['cpu_percent']:.1f} %")
    print(f"  RAM                 : {hw['ram_percent']:.1f} %")
    for gid, gs in hw["gpus"].items():
        print(
            f"  GPU {gid}              : "
            f"{gs['memory_allocated_gb']:.2f} GB allocated"
        )

    # ------------------------------------------------------------------
    # Final confirmation gate
    # ------------------------------------------------------------------
    try:
        confirm = input("\nConfirm first awakening? [yes/N]: ").strip()
    except EOFError:
        confirm = ""

    if confirm.lower() != "yes":
        print("Aborted.")
        sys.exit(0)

    print("\nThe organism awakens …\n")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    try:
        await controller.run()

    except KeyboardInterrupt:
        print("\nInterrupted by operator.")

    except Exception as exc:
        print(f"\nError during awakening: {exc}")
        raise

    # ------------------------------------------------------------------
    # Post-run summary
    # ------------------------------------------------------------------
    print()
    print(BANNER)
    print("  AWAKENING COMPLETE")
    print(BANNER)
    print(f"  Cycles              : {controller.cycle_count}")
    print(f"  Replicated discover.: {len(controller.replicated_discoveries)}")
    print(f"  Guardian intervent. : {controller.guardian.intervention_count}")
    print(
        f"  GPU hours used      : "
        f"{controller.gpu_queue.total_hours_today():.2f} h"
    )

    if controller.replicated_discoveries:
        print("\n  Validated discoveries:")
        for d in controller.replicated_discoveries:
            disc_type = getattr(
                getattr(d, "discovery_type", None), "name", "UNKNOWN"
            )
            narrative = getattr(d, "narrative", "")[:60]
            print(f"    - {disc_type}: {narrative}…")

    print(BANNER)
    print("  The organism sleeps. Review the morning paper.")
    print(BANNER)


if __name__ == "__main__":
    asyncio.run(main())
