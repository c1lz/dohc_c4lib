"""Device-timestamp matching for synchronized multi-camera frame groups."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dohc2_c4lib.aprilgrid_common import CAMERA_IDS


def group_skew_us(group) -> float:
    """Return the maximum device-timestamp spread within one frame group."""
    timestamps = [sample.timestamp_ns for sample in group.values()]
    return (max(timestamps) - min(timestamps)) / 1_000.0


class TimestampGroupMatcher:
    """Match non-reused camera frames by nearest device timestamp.

    ``states`` must map each camera ID to an object exposing an ordered
    ``pending`` mapping. Samples in that mapping must expose ``timestamp_ns``.
    cam0 is the reference stream. A reference is only declared unmatched after
    every camera has advanced beyond its association window, so a temporarily
    late host queue cannot be mistaken for a one-frame hardware-sync offset.
    """

    def __init__(self, pairing_window_us: float, camera_ids=CAMERA_IDS):
        if pairing_window_us <= 0:
            raise ValueError("pairing_window_us must be positive")
        self.camera_ids = tuple(camera_ids)
        if "cam0" not in self.camera_ids:
            raise ValueError("camera_ids must contain cam0 as the reference")
        self.pairing_window_ns = int(pairing_window_us * 1_000)
        self.last_reference_timestamp_ns = -1
        self.unmatched_references = 0
        self.last_unmatched_skew_us: float | None = None

    def match(self, states):
        if any(not states[name].pending for name in self.camera_ids):
            return None
        while True:
            references = [
                sample
                for sample in states["cam0"].pending.values()
                if sample.timestamp_ns > self.last_reference_timestamp_ns
            ]
            if not references:
                return None
            reference = references[0]
            group = {"cam0": reference}
            for name in self.camera_ids:
                if name == "cam0":
                    continue
                group[name] = min(
                    states[name].pending.values(),
                    key=lambda sample: abs(
                        sample.timestamp_ns - reference.timestamp_ns
                    ),
                )
            skew_us = group_skew_us(group)
            if skew_us <= self.pairing_window_ns / 1_000.0:
                self._consume_group(states, group)
                self.last_reference_timestamp_ns = reference.timestamp_ns
                return group

            latest_common_timestamp = min(
                max(
                    sample.timestamp_ns
                    for sample in states[name].pending.values()
                )
                for name in self.camera_ids
            )
            if (
                latest_common_timestamp
                < reference.timestamp_ns + self.pairing_window_ns
            ):
                return None
            self.unmatched_references += 1
            self.last_unmatched_skew_us = skew_us
            self.last_reference_timestamp_ns = reference.timestamp_ns
            self._discard_unmatched_reference(states, reference)

    @staticmethod
    def _consume_group(states, group) -> None:
        for name, state in states.items():
            consumed_timestamp = group[name].timestamp_ns
            for sequence, sample in tuple(state.pending.items()):
                if sample.timestamp_ns <= consumed_timestamp:
                    del state.pending[sequence]

    def _discard_unmatched_reference(self, states, reference) -> None:
        for sequence, sample in tuple(states["cam0"].pending.items()):
            if sample is reference:
                del states["cam0"].pending[sequence]
                break
        oldest_useful = reference.timestamp_ns - self.pairing_window_ns
        for name in self.camera_ids:
            if name == "cam0":
                continue
            state = states[name]
            for sequence, sample in tuple(state.pending.items()):
                if sample.timestamp_ns < oldest_useful:
                    del state.pending[sequence]
