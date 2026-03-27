"""
UGES Autonomous Controller
Orchestrates self-directed experiment cycles with full Guardian protection.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import List, Optional, Tuple

from uges.guardian import UGESGuardian

logger = logging.getLogger("uges.autonomous")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AutonomousConfig:
    """All tuneable parameters for an autonomous session."""

    # Operating window: (start_hour, end_hour) in 24-h clock
    active_hours: Tuple[int, int] = (22, 6)

    # Resource caps
    max_gpu_hours_per_night: float = 8.0
    max_concurrent_experiments: int = 2

    # Safety / testing flag
    use_mock_gpu: bool = True  # Set False for real hardware

    # Notification sinks
    notification_channels: List[str] = field(
        default_factory=lambda: ["dashboard"]
    )

    # Human-readable label stamped on every run
    lab_name: str = "UGES-Lab"

    # Seconds to pause between cycles
    inter_cycle_rest_seconds: float = 30.0


# ---------------------------------------------------------------------------
# Minimal stubs for subsystems not yet implemented
# (These will be replaced by real implementations; the Guardian contracts
#  against their public interfaces.)
# ---------------------------------------------------------------------------

class _StubRegistry:
    """Minimal run registry used until the real one is wired in."""

    def load(self, run_id):
        return None

    def query(self, **kwargs):
        return []


class _StubGPUQueue:
    """Minimal GPU queue stub."""

    def __init__(self, use_mock: bool):
        self.gpu_ids: List[int] = [0] if not use_mock else []
        self._hours_today: float = 0.0

    def total_hours_today(self) -> float:
        return self._hours_today

    def stop(self):
        logger.info("GPU queue stopped")


class _StubNotify:
    """Minimal notification stub — logs to console until real channels wired."""

    async def send_alert(
        self,
        priority: str,
        subject: str,
        body: str,
    ):
        logger.warning("[%s] %s\n%s", priority.upper(), subject, body)


class _StubEventBus:
    """Placeholder event bus."""

    async def emit(self, event: str, payload=None):
        logger.debug("Event: %s payload=%s", event, payload)


class _StubDiscoveryEngine:
    """Placeholder discovery engine."""

    def __init__(self):
        self.discoveries: list = []


# ---------------------------------------------------------------------------
# Autonomous controller
# ---------------------------------------------------------------------------

class AutonomousController:
    """
    Orchestrates self-directed experiment cycles.

    Lifecycle
    ---------
    1. ``run()`` is called; Guardian starts.
    2. Each cycle: check Guardian pause flag → ``_run_cycle()`` → rest.
    3. Guardian monitors hardware and scientific integrity throughout.
    4. On clean exit *or* exception the Guardian is stopped gracefully.
    """

    def __init__(self, config: AutonomousConfig):
        self.config = config

        # --- Subsystems (stubs until full implementation) ---------------
        self.registry = _StubRegistry()
        self.gpu_queue = _StubGPUQueue(use_mock=config.use_mock_gpu)
        self.notify = _StubNotify()
        self.event_bus = _StubEventBus()
        self.discovery = _StubDiscoveryEngine()

        # --- Discovery bookkeeping --------------------------------------
        self.unverified_discoveries: list = []
        self.replicated_discoveries: list = []

        # --- Runtime state ----------------------------------------------
        self.cycle_count: int = 0
        self.emergency_stop: bool = False
        self._guardian_pause: bool = False
        self._stop_current_cycle: bool = False

        # --- Guardian (wired after subsystems are ready) ----------------
        self.guardian = UGESGuardian(self, self.gpu_queue.gpu_ids)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self):
        """
        Main autonomous loop, protected by the Guardian at all times.
        Runs until the active window closes, the daily budget is spent,
        or the Guardian triggers an emergency stop.
        """
        await self.guardian.start()
        logger.info("Autonomous controller started — lab: %s", self.config.lab_name)

        try:
            while self._should_continue():
                # Respect guardian pause
                if self._guardian_pause:
                    logger.info("Guardian pause active — waiting 60 s …")
                    await asyncio.sleep(60)
                    continue

                await self._run_cycle()
                await self._rest()

        except asyncio.CancelledError:
            logger.info("Controller cancelled — shutting down cleanly")
        except Exception:
            logger.exception("Unhandled exception in autonomous controller")
            raise
        finally:
            self.guardian.stop()
            logger.info(
                "Autonomous controller finished — cycles: %d, "
                "guardian interventions: %d",
                self.cycle_count,
                self.guardian.intervention_count,
            )

    # ------------------------------------------------------------------
    # Single cycle
    # ------------------------------------------------------------------

    async def _run_cycle(self):
        """
        Execute one experiment cycle with pre/post Guardian oversight.
        """
        self._stop_current_cycle = False
        self.cycle_count += 1
        logger.info("--- Cycle %d start ---", self.cycle_count)

        # Pre-cycle: surface any recent critical alerts
        status = self.guardian.get_status_report()
        recent = status.get("recent_alerts", [])
        if recent and recent[-1]["level"] == "CRITICAL":
            logger.warning(
                "Entering cycle %d with a recent CRITICAL alert: %s",
                self.cycle_count,
                recent[-1]["message"],
            )

        await self.event_bus.emit("cycle_start", {"cycle": self.cycle_count})

        # --- Placeholder for real experiment execution ------------------
        # Replace with: schedule → run → collect → analyse pipeline.
        logger.info("Cycle %d: running experiments (stub)", self.cycle_count)
        await asyncio.sleep(1)  # Simulates work

        if self._stop_current_cycle:
            logger.warning("Cycle %d aborted by Guardian", self.cycle_count)
            return

        # Post-cycle: annotate any integrity-flagged discoveries
        for disc in list(self.discovery.discoveries):
            validity_issue = self.guardian.integrity.check_discovery_validity(disc)
            if validity_issue:
                disc.narrative = (
                    getattr(disc, "narrative", "")
                    + f"\n\n[Guardian note: {validity_issue}]"
                )
                disc.requires_human_review = True

        await self.event_bus.emit("cycle_end", {"cycle": self.cycle_count})
        logger.info("--- Cycle %d end ---", self.cycle_count)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _should_continue(self) -> bool:
        """Return False when any termination condition is met."""
        if self.emergency_stop:
            logger.critical("Emergency stop flag set — halting")
            return False

        if self.gpu_queue.total_hours_today() >= self.config.max_gpu_hours_per_night:
            logger.info("Daily GPU budget exhausted — stopping")
            return False

        if not self._in_active_window():
            logger.info("Outside active window — stopping")
            return False

        return True

    def _in_active_window(self) -> bool:
        """Return True if the current hour falls within the configured window."""
        start_h, end_h = self.config.active_hours
        now_h = datetime.now().hour

        if start_h <= end_h:
            # Same-day window, e.g. 09:00–17:00
            return start_h <= now_h < end_h
        else:
            # Overnight window, e.g. 22:00–06:00
            return now_h >= start_h or now_h < end_h

    async def _rest(self):
        """Pause between cycles to avoid busy-looping."""
        await asyncio.sleep(self.config.inter_cycle_rest_seconds)
