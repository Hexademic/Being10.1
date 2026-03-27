"""
UGES Guardian: Safety, ethics, and integrity enforcement.

The Guardian is the immune system that keeps the organism healthy.
It operates at three levels:
  1. Hardware Protection  — GPU temperature, memory, power draw
  2. Scientific Integrity — Prevents self-deception, enforces reproducibility
  3. Resource Ethics      — Respects shared compute, prevents runaway consumption
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Callable, Dict, List, Optional

import psutil
import torch

logger = logging.getLogger("uges.guardian")


# ---------------------------------------------------------------------------
# Alert levels
# ---------------------------------------------------------------------------

class GuardianAlertLevel(Enum):
    """Severity levels for guardian interventions."""
    INFO = auto()       # Logged, no action
    WARNING = auto()    # Notification, soft throttle
    CRITICAL = auto()   # Hard stop, human required
    EMERGENCY = auto()  # Immediate shutdown


# ---------------------------------------------------------------------------
# Rule definition
# ---------------------------------------------------------------------------

@dataclass
class GuardianRule:
    """A safety or integrity constraint."""
    name: str
    check: Callable[[], bool]      # Returns True if violation detected
    level: GuardianAlertLevel
    message: str
    auto_action: Optional[str] = None  # "throttle" | "pause" | "stop"


# ---------------------------------------------------------------------------
# Hardware monitor
# ---------------------------------------------------------------------------

class HardwareMonitor:
    """Real-time GPU and system health monitoring."""

    def __init__(self, gpu_ids: List[int]):
        self.gpu_ids = gpu_ids
        self.temperature_history: Dict[int, List[tuple]] = {
            gpu: [] for gpu in gpu_ids
        }
        self.peak_memory: Dict[int, float] = {gpu: 0.0 for gpu in gpu_ids}

    async def start_monitoring(self, interval_seconds: float = 5.0):
        """Background monitoring loop — runs until cancelled."""
        while True:
            for gpu_id in self.gpu_ids:
                if torch.cuda.is_available():
                    # --- Temperature (requires nvidia-ml-py or similar) ----
                    try:
                        temp = torch.cuda.temperature(gpu_id)
                        self.temperature_history[gpu_id].append(
                            (datetime.now(), temp)
                        )
                        # Retain last hour only
                        cutoff = datetime.now() - timedelta(hours=1)
                        self.temperature_history[gpu_id] = [
                            (t, v)
                            for t, v in self.temperature_history[gpu_id]
                            if t > cutoff
                        ]
                    except Exception:
                        pass  # Temperature API not available on all platforms

                    # --- Memory -----------------------------------------
                    mem_info = torch.cuda.memory_stats(gpu_id)
                    allocated = mem_info.get("allocated_bytes.all.current", 0)
                    self.peak_memory[gpu_id] = max(
                        self.peak_memory[gpu_id],
                        allocated / 1e9,  # bytes → GB
                    )

            await asyncio.sleep(interval_seconds)

    def get_current_stats(self) -> Dict:
        """Return a snapshot of the current hardware state."""
        stats: Dict = {
            "cpu_percent": psutil.cpu_percent(),
            "ram_percent": psutil.virtual_memory().percent,
            "gpus": {},
        }

        for gpu_id in self.gpu_ids:
            gpu_stats: Dict = {
                "memory_allocated_gb": 0.0,
                "temperature_c": None,
                "utilization_percent": None,
            }

            if torch.cuda.is_available():
                gpu_stats["memory_allocated_gb"] = (
                    torch.cuda.memory_allocated(gpu_id) / 1e9
                )
                # Full temperature / utilisation requires nvidia-ml-py;
                # pull from cached history when available.
                hist = self.temperature_history.get(gpu_id, [])
                if hist:
                    gpu_stats["temperature_c"] = hist[-1][1]

            stats["gpus"][gpu_id] = gpu_stats

        return stats


# ---------------------------------------------------------------------------
# Scientific integrity checker
# ---------------------------------------------------------------------------

class ScientificIntegrityChecker:
    """
    Ensures UGES doesn't deceive itself.

    Detects: p-hacking, cherry-picking, insufficient replication,
             metric manipulation, and publication bias in miniature.
    """

    def __init__(self, registry):
        self.registry = registry
        self.discovery_attempts: Dict[str, List[datetime]] = {}

    def check_discovery_validity(self, discovery) -> Optional[str]:
        """
        Validate that a claimed discovery meets scientific standards.

        Returns None if valid, or a human-readable error string if not.
        """
        issues: List[str] = []

        # 1. Sufficient replication
        rep_ids = getattr(discovery, "replication_run_ids", [])
        if len(rep_ids) < 2:
            issues.append(
                f"Only {len(rep_ids)} replication(s) "
                "(minimum 3 recommended for validation)"
            )

        # 2. Multiple-comparison correction (Bonferroni heuristic)
        similar_runs = self._count_similar_experiments(discovery)
        confidence = getattr(discovery, "confidence", 1.0)
        if similar_runs > 20 and confidence < 0.99:
            issues.append(
                f"Discovery in field of {similar_runs} similar experiments "
                f"with confidence {confidence:.2f} may not survive "
                "multiple-comparison correction"
            )

        # 3. Effect size
        evidence = getattr(discovery, "empirical_evidence", None) or {}
        signal = evidence.get("speedup_ratio", 1.0)
        if signal < 1.5:
            issues.append(
                f"Effect size ({signal:.2f}x) may be too small for "
                "practical significance"
            )

        # 4. File-drawer problem — are there many null results nearby?
        nulls_nearby = self._count_null_results_near(discovery)
        if nulls_nearby > 5:
            issues.append(
                f"{nulls_nearby} null results in nearby parameter space "
                "suggest a possible false positive"
            )

        return "; ".join(issues) if issues else None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _count_similar_experiments(self, discovery) -> int:
        """Count experiments with a similar configuration to this discovery."""
        base_run = self.registry.load(getattr(discovery, "run_id", None))
        if not base_run:
            return 0

        similar = self.registry.query(
            phase=base_run.config.get("phase"),
            since=datetime.now() - timedelta(days=7),
        )

        count = 0
        for run in similar:
            e_diff = abs(run.config.get("E", 0) - base_run.config.get("E", 0))
            dt_diff = abs(
                run.config.get("dt", 0) - base_run.config.get("dt", 0)
            )
            if e_diff < 5_000 and dt_diff < 0.02:
                count += 1

        return count

    def _count_null_results_near(self, discovery) -> int:
        """Count completed runs with no discoveries in the parameter neighbourhood."""
        base_run = self.registry.load(getattr(discovery, "run_id", None))
        if not base_run:
            return 0

        nearby = self.registry.query(
            since=datetime.now() - timedelta(days=7),
            status="completed",
        )

        nulls = 0
        for run in nearby:
            if not getattr(run, "discoveries", None) and not getattr(
                run, "anomalies", None
            ):
                e_diff = abs(
                    run.config.get("E", 0) - base_run.config.get("E", 0)
                )
                if e_diff < 10_000:
                    nulls += 1

        return nulls


# ---------------------------------------------------------------------------
# Central guardian
# ---------------------------------------------------------------------------

class UGESGuardian:
    """
    Central safety and integrity enforcement for UGES.
    The watchful presence that keeps the organism healthy.
    """

    # Safety thresholds -------------------------------------------------
    GPU_TEMP_WARNING: int = 80       # °C — soft throttle
    GPU_TEMP_CRITICAL: int = 90      # °C — hard stop
    GPU_MEMORY_WARNING: float = 0.80  # 80 % of VRAM
    GPU_MEMORY_CRITICAL: float = 0.95  # 95 % of VRAM
    GPU_MEMORY_CRITICAL_GB: float = 14.0  # Absolute cap (assumes 16 GB cards)
    MAX_DAILY_GPU_HOURS: float = 12.0
    MAX_CONSECUTIVE_FAILURES: int = 5

    def __init__(self, controller, gpu_ids: List[int]):
        self.controller = controller
        self.hardware = HardwareMonitor(gpu_ids)
        self.integrity = ScientificIntegrityChecker(controller.registry)

        self.rules: List[GuardianRule] = []
        self._setup_default_rules()

        self.alert_history: List[Dict] = []
        self.intervention_count: int = 0
        self._running: bool = False

    # ------------------------------------------------------------------
    # Rule setup
    # ------------------------------------------------------------------

    def _setup_default_rules(self):
        """Register all default safety and integrity rules."""

        # ---- Hardware rules ------------------------------------------
        self.rules.append(GuardianRule(
            name="gpu_temperature",
            check=self._check_gpu_temperature,
            level=GuardianAlertLevel.WARNING,
            message="GPU temperature elevated",
            auto_action="throttle",
        ))

        self.rules.append(GuardianRule(
            name="gpu_memory",
            check=self._check_gpu_memory,
            level=GuardianAlertLevel.CRITICAL,
            message="GPU memory nearly exhausted",
            auto_action="pause",
        ))

        self.rules.append(GuardianRule(
            name="daily_budget",
            check=self._check_daily_budget,
            level=GuardianAlertLevel.WARNING,
            message="Approaching daily GPU-hour limit",
            auto_action="throttle",
        ))

        # ---- Scientific integrity rules ------------------------------
        self.rules.append(GuardianRule(
            name="discovery_validity",
            check=self._check_discovery_validity,
            level=GuardianAlertLevel.WARNING,
            message="Discovery may not meet scientific standards",
            auto_action=None,  # Requires human review
        ))

        self.rules.append(GuardianRule(
            name="failure_rate",
            check=self._check_failure_rate,
            level=GuardianAlertLevel.CRITICAL,
            message="High experiment failure rate suggests a systemic issue",
            auto_action="pause",
        ))

        self.rules.append(GuardianRule(
            name="replication_urgency",
            check=self._check_replication_urgency,
            level=GuardianAlertLevel.INFO,
            message="Unverified discoveries awaiting replication",
            auto_action=None,
        ))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        """Begin guardian monitoring (non-blocking — schedules background tasks)."""
        self._running = True
        asyncio.create_task(self.hardware.start_monitoring())
        asyncio.create_task(self._monitoring_loop())
        logger.info("Guardian started — %d rules active", len(self.rules))

    def stop(self):
        """Signal the monitoring loop to exit on its next iteration."""
        self._running = False
        logger.info("Guardian stopped")

    # ------------------------------------------------------------------
    # Monitoring loop
    # ------------------------------------------------------------------

    async def _monitoring_loop(self, interval_seconds: float = 10.0):
        """Evaluate every rule on a fixed interval."""
        while self._running:
            for rule in self.rules:
                try:
                    if rule.check():
                        await self._handle_violation(rule)
                except Exception as exc:
                    logger.error(
                        "Guardian rule '%s' raised an exception: %s",
                        rule.name, exc,
                    )
            await asyncio.sleep(interval_seconds)

    # ------------------------------------------------------------------
    # Violation handling
    # ------------------------------------------------------------------

    async def _handle_violation(self, rule: GuardianRule):
        """Respond to a triggered safety or integrity rule."""
        alert: Dict = {
            "timestamp": datetime.now(),
            "rule": rule.name,
            "level": rule.level.name,
            "message": rule.message,
            "action_taken": None,
        }

        if rule.level == GuardianAlertLevel.EMERGENCY:
            await self._emergency_stop(rule.message)
            alert["action_taken"] = "emergency_stop"

        elif rule.level == GuardianAlertLevel.CRITICAL:
            if rule.auto_action == "pause":
                await self._pause_autonomous(rule.message)
                alert["action_taken"] = "pause"
            elif rule.auto_action == "stop":
                await self._stop_current_cycle(rule.message)
                alert["action_taken"] = "stop_cycle"

        elif rule.level == GuardianAlertLevel.WARNING:
            if rule.auto_action == "throttle":
                await self._throttle_execution(rule.message)
                alert["action_taken"] = "throttle"

        # Always attempt human notification for CRITICAL / EMERGENCY
        await self._notify_humans(alert)

        self.alert_history.append(alert)
        self.intervention_count += 1

    # ------------------------------------------------------------------
    # Rule check implementations
    # ------------------------------------------------------------------

    def _check_gpu_temperature(self) -> bool:
        """Return True if any GPU recently exceeded the warning threshold."""
        for gpu_id, history in self.hardware.temperature_history.items():
            if not history:
                continue
            # Sample last ~60 s worth of readings (12 × 5 s intervals)
            recent_temps = [temp for _, temp in history[-12:]]
            if recent_temps and max(recent_temps) > self.GPU_TEMP_WARNING:
                return True
        return False

    def _check_gpu_memory(self) -> bool:
        """Return True if any GPU is critically close to its memory limit."""
        stats = self.hardware.get_current_stats()
        for gpu_stats in stats["gpus"].values():
            if gpu_stats["memory_allocated_gb"] > self.GPU_MEMORY_CRITICAL_GB:
                return True
        return False

    def _check_daily_budget(self) -> bool:
        """Return True if today's GPU usage is approaching the daily cap."""
        hours = self.controller.gpu_queue.total_hours_today()
        return hours > self.MAX_DAILY_GPU_HOURS * 0.8

    def _check_discovery_validity(self) -> bool:
        """Return True if any recent unverified discovery has integrity issues."""
        cutoff = datetime.now() - timedelta(hours=1)
        discovery_engine = getattr(self.controller, "discovery", None)
        all_discoveries = getattr(discovery_engine, "discoveries", []) if discovery_engine else []
        recent_discoveries = [
            d
            for d in all_discoveries
            if getattr(d, "timestamp", datetime.min) > cutoff
            and getattr(d, "status", "") == "unverified"
        ]

        for disc in recent_discoveries:
            issue = self.integrity.check_discovery_validity(disc)
            if issue:
                logger.warning("Discovery validity issue: %s", issue)
                return True
        return False

    def _check_failure_rate(self) -> bool:
        """Return True if too many experiments have failed in the last 2 hours."""
        recent = self.controller.registry.query(
            since=datetime.now() - timedelta(hours=2)
        )
        if len(recent) < 5:
            return False
        failures = sum(1 for r in recent if getattr(r, "status", "") == "failed")
        return failures > self.MAX_CONSECUTIVE_FAILURES

    def _check_replication_urgency(self) -> bool:
        """Informational: flag when too many discoveries are awaiting replication."""
        unverified = len(getattr(self.controller, "unverified_discoveries", []))
        if unverified > 3:
            logger.info("%d discoveries awaiting replication", unverified)
            return True
        return False

    # ------------------------------------------------------------------
    # Intervention actions
    # ------------------------------------------------------------------

    async def _emergency_stop(self, reason: str):
        """Immediate full shutdown — hardware at risk."""
        logger.critical("EMERGENCY STOP: %s", reason)
        self.controller.emergency_stop = True
        self.controller.gpu_queue.stop()

        await self.controller.notify.send_alert(
            priority="emergency",
            subject="UGES EMERGENCY STOP",
            body=(
                f"The guardian has triggered an emergency stop:\n\n{reason}\n\n"
                "All GPU workers halted. Human intervention required."
            ),
        )

    async def _pause_autonomous(self, reason: str):
        """Pause autonomous mode while preserving all state."""
        logger.warning("Autonomous pause: %s", reason)
        self.controller._guardian_pause = True

        await self.controller.notify.send_alert(
            priority="urgent",
            subject="UGES Autonomous Pause",
            body=(
                f"The guardian has paused autonomous execution:\n\n{reason}\n\n"
                "Current state preserved. Resume when conditions improve."
            ),
        )

    async def _stop_current_cycle(self, reason: str):
        """Abort the running cycle without a full shutdown."""
        logger.warning("Stopping current cycle: %s", reason)
        self.controller._stop_current_cycle = True

    async def _throttle_execution(self, reason: str):
        """Reduce the number of concurrent experiments by one."""
        logger.info("Throttling execution: %s", reason)
        cfg = self.controller.config
        old_max = cfg.max_concurrent_experiments
        cfg.max_concurrent_experiments = max(1, old_max - 1)
        logger.info(
            "Concurrent experiments reduced: %d → %d",
            old_max, cfg.max_concurrent_experiments,
        )

    async def _notify_humans(self, alert: Dict):
        """Route alert to human channels for CRITICAL and EMERGENCY levels."""
        if alert["level"] in ("CRITICAL", "EMERGENCY"):
            notify = getattr(self.controller, "notify", None)
            if notify is None:
                return
            await notify.send_alert(
                priority=alert["level"].lower(),
                subject=f"UGES Guardian: {alert['rule']}",
                body=(
                    f"Level:   {alert['level']}\n"
                    f"Rule:    {alert['rule']}\n"
                    f"Message: {alert['message']}\n"
                    f"Action:  {alert['action_taken'] or 'None'}"
                ),
            )

    # ------------------------------------------------------------------
    # Status report (for dashboard / logging)
    # ------------------------------------------------------------------

    def get_status_report(self) -> Dict:
        """Return current guardian status as a plain dictionary."""
        return {
            "active": self._running,
            "rules_checked": len(self.rules),
            "interventions": self.intervention_count,
            "recent_alerts": self.alert_history[-5:],
            "hardware_status": self.hardware.get_current_stats(),
            "integrity_checks": {
                "unverified_discoveries": len(
                    getattr(self.controller, "unverified_discoveries", [])
                ),
                "suspicious_discoveries": sum(
                    1
                    for a in self.alert_history
                    if a["rule"] == "discovery_validity"
                ),
            },
        }
