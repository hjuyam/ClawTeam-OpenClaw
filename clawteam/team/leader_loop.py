"""Leader monitor loop.

This is the missing "leader closes the loop" layer:
- Drain leader inbox and surface messages
- Ping owners of pending tasks (start work / ACK)
- Nudge owners of long-running in_progress tasks (PROGRESS/BLOCKED)
- Detect dead agents (via TaskWaiter) and optionally auto-respawn

Designed to be safe-by-default: it does not force-update tasks unless asked.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from clawteam.team.models import TaskItem, TaskStatus, TeamMessage


@dataclass
class LeaderLoopConfig:
    poll_interval: float = 5.0
    ping_after: float = 30.0
    nudge_after: float = 180.0
    timeout: float | None = None
    stalled_after: float = 600.0


@dataclass
class LeaderLoopState:
    last_ping: dict[str, float] = field(default_factory=dict)   # task_id -> ts
    last_nudge: dict[str, float] = field(default_factory=dict)  # task_id -> ts
    last_stalled: dict[str, bool] = field(default_factory=dict)  # task_id -> stalled event already emitted for current in_progress span
    last_status: dict[str, TaskStatus] = field(default_factory=dict)  # task_id -> last seen status


class LeaderLoop:
    def __init__(
        self,
        team_name: str,
        leader_inbox: str,
        mailbox,
        task_store,
        spawner=None,
        cfg: LeaderLoopConfig | None = None,
    ):
        self.team_name = team_name
        self.leader_inbox = leader_inbox
        self.mailbox = mailbox
        self.task_store = task_store
        self.spawner = spawner
        self.cfg = cfg or LeaderLoopConfig()
        self.state = LeaderLoopState()
        self._start = time.monotonic()
        self._running = False

    def run(self, on_message=None, on_progress=None, on_agent_dead=None,
            on_status_change=None, on_stalled=None) -> None:
        self._running = True
        last_summary = ""
        first_iteration = True
        # Use stateful last_status / last_stalled so we can dedupe across loop iterations

        while self._running:
            # Always run at least once
            if first_iteration:
                first_iteration = False
            else:
                # Timeout
                if self.cfg.timeout is not None and (time.monotonic() - self._start) >= self.cfg.timeout:
                    return

            # 1) Drain inbox
            msgs = self.mailbox.receive(self.leader_inbox, limit=50)
            for msg in msgs:
                if on_message:
                    on_message(msg)

            # 2) Task snapshot
            tasks = self.task_store.list_tasks()

            # 3) Progress callback (dedupe)
            completed = sum(1 for t in tasks if t.status == TaskStatus.completed)
            total = len(tasks)
            in_prog = sum(1 for t in tasks if t.status == TaskStatus.in_progress)
            pending = sum(1 for t in tasks if t.status == TaskStatus.pending)
            blocked = sum(1 for t in tasks if t.status == TaskStatus.blocked)
            summary = f"{completed}/{total}/{in_prog}/{pending}/{blocked}"
            if summary != last_summary:
                last_summary = summary
                if on_progress:
                    on_progress(completed, total, in_prog, pending, blocked)

            # 4) Status change and stalled detection
            now = time.time()
            for t in tasks:
                if not t.owner:
                    continue

                # Detect status changes
                old_status = self.state.last_status.get(t.id)
                if old_status is None:
                    # First time seeing this task; record baseline status (no event).
                    self.state.last_status[t.id] = t.status
                elif old_status != t.status:
                    # Status changed - emit event and reset stalled dedupe
                    if on_status_change:
                        from clawteam.team.tasks import _tasks_root
                        task_file = _tasks_root(self.team_name) / f"task-{t.id}.json"
                        on_status_change(
                            task_id=t.id,
                            from_status=old_status,
                            to_status=t.status,
                            owner=t.owner,
                            subject=t.subject,
                            task_file=task_file,
                        )
                    self.state.last_status[t.id] = t.status
                    self.state.last_stalled.pop(t.id, None)

                # Detect stalled tasks
                if t.status == TaskStatus.in_progress:
                    age = _age_seconds(t, started=True)
                    if age >= self.cfg.stalled_after and not self.state.last_stalled.get(t.id, False):
                        if on_stalled:
                            from clawteam.team.tasks import _tasks_root
                            task_file = _tasks_root(self.team_name) / f"task-{t.id}.json"
                            on_stalled(
                                task_id=t.id,
                                owner=t.owner,
                                subject=t.subject,
                                stalled_after=self.cfg.stalled_after,
                                reason=f"Task has been running for {int(age)}s without status change",
                                suggested_action="Update task status or reply with PROGRESS to avoid getting stuck",
                                task_file=task_file,
                            )
                        # Dedup until status changes
                        self.state.last_stalled[t.id] = True
                else:
                    # Reset dedupe once task leaves in_progress
                    self.state.last_stalled.pop(t.id, None)

            # 5) Ping/nudge logic
            for t in tasks:
                if not t.owner:
                    continue

                if t.status == TaskStatus.pending:
                    # ping after ping_after, but avoid spamming
                    age = _age_seconds(t)
                    if age >= self.cfg.ping_after:
                        last = self.state.last_ping.get(t.id, 0)
                        if now - last >= self.cfg.ping_after:
                            self.mailbox.send(
                                from_agent="leader",
                                to=t.owner,
                                content=(
                                    f"PING: start task {t.id} ({t.subject}). "
                                    f"First action: inbox send leader 'ACK {t.id}'. "
                                    f"If blocked, inbox send 'BLOCKED {t.id}: <reason>'."
                                ),
                            )
                            self.state.last_ping[t.id] = now

                if t.status == TaskStatus.in_progress:
                    age = _age_seconds(t, started=True)
                    if age >= self.cfg.nudge_after:
                        last = self.state.last_nudge.get(t.id, 0)
                        if now - last >= self.cfg.nudge_after:
                            self.mailbox.send(
                                from_agent="leader",
                                to=t.owner,
                                content=(
                                    f"NUDGE: task {t.id} running {int(age)}s. "
                                    f"Reply with 'PROGRESS {t.id}: ...' or 'BLOCKED {t.id}: ...'."
                                ),
                            )
                            self.state.last_nudge[t.id] = now

            # 6) Sleep
            time.sleep(self.cfg.poll_interval)

    def stop(self):
        self._running = False


def _age_seconds(t: TaskItem, started: bool = False) -> float:
    """Best-effort age calc."""
    ts = None
    if started and getattr(t, "started_at", None):
        ts = t.started_at
    if not ts and getattr(t, "created_at", None):
        ts = t.created_at
    if not ts:
        return 0.0
    try:
        # Handle both datetime objects and ISO format strings
        if isinstance(ts, str):
            from datetime import datetime
            ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        return time.time() - ts.timestamp()
    except Exception:
        return 0.0
