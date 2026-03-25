import time

import pytest

from clawteam.team.leader_loop import LeaderLoop, LeaderLoopConfig
from clawteam.team.models import TaskItem, TaskStatus


class FakeMailbox:
    def __init__(self):
        self.sent = []
        self._msgs = []

    def receive(self, inbox: str, limit: int = 50):
        out = self._msgs[:limit]
        self._msgs = self._msgs[limit:]
        return out

    def send(self, from_agent: str, to: str, content: str):
        self.sent.append({"from": from_agent, "to": to, "content": content})


class FakeTaskStore:
    def __init__(self, tasks):
        self._tasks = list(tasks)

    def list_tasks(self):
        return self._tasks

    def update_task_status(self, task_id, status):
        """Update a task's status (for testing status changes)."""
        for t in self._tasks:
            if t.id == task_id:
                t.status = status
                return t
        return None


@pytest.fixture
def fixed_time(monkeypatch):
    # Make time deterministic
    t = {"now": 1_700_000_000.0, "mono": 0.0}

    def fake_time():
        return t["now"]

    def fake_monotonic():
        return t["mono"]

    def advance(seconds: float):
        t["now"] += seconds
        t["mono"] += seconds

    monkeypatch.setattr(time, "time", fake_time)
    monkeypatch.setattr(time, "monotonic", fake_monotonic)
    return advance


def _task_pending(task_id="t1", owner="a1", created_at=None):
    t = TaskItem(subject="S", owner=owner)
    t.id = task_id
    t.status = TaskStatus.pending
    if created_at:
        t.created_at = created_at
    return t


def _task_in_progress(task_id="t2", owner="a2", started_at=None, created_at=None):
    t = TaskItem(subject="S2", owner=owner)
    t.id = task_id
    t.status = TaskStatus.in_progress
    if started_at:
        t.started_at = started_at
    if created_at:
        t.created_at = created_at
    return t


class FakeCallback:
    def __init__(self):
        self.events = []

    def __call__(self, **kwargs):
        self.events.append(kwargs)


def test_status_change_event(fixed_time, monkeypatch):
    """Test that status change events are emitted when task status changes."""
    created_iso = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(time.time() - 10))
    task1 = _task_pending(task_id="t1", created_at=created_iso)

    mb = FakeMailbox()
    store = FakeTaskStore([task1])

    # Allow a couple iterations; we will update status during the loop.
    cfg = LeaderLoopConfig(poll_interval=0.0, ping_after=9999, nudge_after=9999, timeout=2.0)
    loop = LeaderLoop(team_name="t", leader_inbox="leader", mailbox=mb, task_store=store, cfg=cfg)

    callback = FakeCallback()

    updated = {"done": False}

    def fake_sleep(_):
        # On first sleep, flip the task status so next iteration sees a change.
        if not updated["done"]:
            store.update_task_status("t1", TaskStatus.in_progress)
            updated["done"] = True
        fixed_time(1.0)

    monkeypatch.setattr(time, "sleep", fake_sleep)
    loop.run(on_status_change=callback)

    assert len(callback.events) == 1
    e = callback.events[0]
    assert e["task_id"] == "t1"
    assert e["from_status"] == TaskStatus.pending
    assert e["to_status"] == TaskStatus.in_progress
    assert e["owner"] == "a1"


def test_stalled_event(fixed_time, monkeypatch):
    """Test that stalled events are emitted when in_progress task exceeds stalled_after."""
    started_iso = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(time.time() - 700))
    task = _task_in_progress(task_id="t1", started_at=started_iso)

    mb = FakeMailbox()
    store = FakeTaskStore([task])

    cfg = LeaderLoopConfig(poll_interval=0.0, ping_after=9999, nudge_after=9999, stalled_after=600, timeout=2.0)
    loop = LeaderLoop(team_name="t", leader_inbox="leader", mailbox=mb, task_store=store, cfg=cfg)

    callback = FakeCallback()

    def fake_sleep(_):
        fixed_time(1.0)

    monkeypatch.setattr(time, "sleep", fake_sleep)
    loop.run(on_stalled=callback)

    assert len(callback.events) == 1
    event = callback.events[0]
    assert event["task_id"] == "t1"
    assert event["owner"] == "a2"
    assert event["stalled_after"] == 600
    assert "running for 700s" in event["reason"]
    assert "Update task status" in event["suggested_action"]


def test_stalled_event_dedupe(fixed_time, monkeypatch):
    """Test that stalled events are deduped - only emitted once until status changes."""
    started_iso = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(time.time() - 700))
    task = _task_in_progress(task_id="t1", started_at=started_iso)

    mb = FakeMailbox()
    store = FakeTaskStore([task])

    cfg = LeaderLoopConfig(poll_interval=0.0, ping_after=9999, nudge_after=9999, stalled_after=600, timeout=2.0)
    loop = LeaderLoop(team_name="t", leader_inbox="leader", mailbox=mb, task_store=store, cfg=cfg)

    callback = FakeCallback()

    def fake_sleep(_):
        fixed_time(1.0)

    monkeypatch.setattr(time, "sleep", fake_sleep)
    loop.run(on_stalled=callback)

    assert len(callback.events) == 1


def test_stalled_event_resets_on_status_change(fixed_time, monkeypatch):
    """After leaving in_progress, stalled dedupe resets so a new in_progress span can emit again."""
    started_iso = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(time.time() - 700))
    task = _task_in_progress(task_id="t1", started_at=started_iso)

    mb = FakeMailbox()
    store = FakeTaskStore([task])

    cfg = LeaderLoopConfig(poll_interval=0.0, ping_after=9999, nudge_after=9999, stalled_after=600, timeout=3.0)
    loop = LeaderLoop(team_name="t", leader_inbox="leader", mailbox=mb, task_store=store, cfg=cfg)

    callback = FakeCallback()

    step = {"i": 0}

    def fake_sleep(_):
        step["i"] += 1
        # iteration 1 done -> change to completed
        if step["i"] == 1:
            store.update_task_status("t1", TaskStatus.completed)
        # iteration 2 done -> change back to in_progress and make it old enough
        elif step["i"] == 2:
            store.update_task_status("t1", TaskStatus.in_progress)
            # keep started_at old so it stalls again
        fixed_time(1.0)

    monkeypatch.setattr(time, "sleep", fake_sleep)
    loop.run(on_stalled=callback)

    # Expect two stalled events: one for first in_progress span, one after reset when re-entering in_progress
    assert len(callback.events) == 2
    assert all(e["task_id"] == "t1" for e in callback.events)


def test_ping_triggers_after_ping_after(fixed_time, monkeypatch):
    # created 40s ago -> should ping when ping_after=30
    created_iso = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(time.time() - 40))
    task = _task_pending(created_at=created_iso)

    mb = FakeMailbox()
    store = FakeTaskStore([task])

    cfg = LeaderLoopConfig(poll_interval=0.0, ping_after=30.0, nudge_after=9999, timeout=0.0)
    loop = LeaderLoop(team_name="t", leader_inbox="leader", mailbox=mb, task_store=store, cfg=cfg)

    # Prevent sleep from blocking
    monkeypatch.setattr(time, "sleep", lambda *_: None)

    loop.run()

    assert len(mb.sent) == 1
    assert mb.sent[0]["to"] == "a1"
    assert "PING" in mb.sent[0]["content"]


def test_nudge_triggers_after_nudge_after(fixed_time, monkeypatch):
    started_iso = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(time.time() - 400))
    task = _task_in_progress(started_at=started_iso)

    mb = FakeMailbox()
    store = FakeTaskStore([task])

    cfg = LeaderLoopConfig(poll_interval=0.0, ping_after=9999, nudge_after=180.0, timeout=0.0)
    loop = LeaderLoop(team_name="t", leader_inbox="leader", mailbox=mb, task_store=store, cfg=cfg)

    monkeypatch.setattr(time, "sleep", lambda *_: None)
    loop.run()

    assert len(mb.sent) == 1
    assert mb.sent[0]["to"] == "a2"
    assert "NUDGE" in mb.sent[0]["content"]


def test_timeout_exits_loop(fixed_time, monkeypatch):
    # no tasks; just ensure returns when timeout hits
    mb = FakeMailbox()
    store = FakeTaskStore([])

    # timeout=0 means immediate exit on first check
    cfg = LeaderLoopConfig(poll_interval=9999, timeout=0.0)
    loop = LeaderLoop(team_name="t", leader_inbox="leader", mailbox=mb, task_store=store, cfg=cfg)

    monkeypatch.setattr(time, "sleep", lambda *_: None)
    loop.run()

    assert mb.sent == []


def test_dedup_prevents_spam(fixed_time, monkeypatch):
    created_iso = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(time.time() - 60))
    task = _task_pending(created_at=created_iso)

    mb = FakeMailbox()
    store = FakeTaskStore([task])

    # Allow multiple iterations: stop after 2 seconds monotonic
    cfg = LeaderLoopConfig(poll_interval=0.0, ping_after=30.0, nudge_after=9999, timeout=1.0)
    loop = LeaderLoop(team_name="t", leader_inbox="leader", mailbox=mb, task_store=store, cfg=cfg)

    # Each loop sleep advances time a bit (simulate polling)
    def fake_sleep(_):
        fixed_time(1.0)

    monkeypatch.setattr(time, "sleep", fake_sleep)
    loop.run()

    # With dedup, even though we loop twice, we should not send more than 1 ping
    assert len([m for m in mb.sent if "PING" in m["content"]]) == 1
