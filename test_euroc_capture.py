"""Offline tests: run with .venv/bin/python -m unittest test_euroc_capture -v."""
import csv
import hashlib
import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dohc2_c4lib import euroc_capture as ec
from dohc2_c4lib.record_oak4p import DatasetWriter, ImageTask


class OfflineTests(unittest.TestCase):
    def test_imu_interpolation_epoch_and_boundaries(self):
        epoch = 1788839123456000000
        a = [[epoch, 0, 2, 4], [epoch+2500000, 2, 4, 6]]
        g = [[epoch-1, 0, 0, 0], [epoch, 1, 2, 3],
             [epoch+1250000, 4, 5, 6], [epoch+2500001, 0, 0, 0]]
        result, info = ec.align_imu(a, g, 400)
        self.assertEqual(result[1], [epoch+1250000, 4, 5, 6, 1, 3, 5])
        self.assertEqual(info["boundary_omitted"], 2)
        self.assertEqual(result[0][0], epoch)

    def test_imu_gap_duplicates_nonfinite(self):
        result, info = ec.align_imu([[0, 0, 0, 0], [10000000, 1, 1, 1]],
                                    [[5000000, 0, 0, 0]], 400)
        self.assertEqual(result, [])
        self.assertEqual(info["gap_omitted"], 1)
        for bad in ([[0, 0, 0, 0], [0, 1, 1, 1]],
                    [[1, 0, 0, 0], [0, 1, 1, 1]],
                    [[0, float("nan"), 0, 0]]):
            with self.assertRaises(ValueError):
                ec.align_imu(bad, [[0, 0, 0, 0]], 400)

    def test_sync_no_reuse_and_missing(self):
        rows = {"cam0": [[0], [33333000], [66666000]],
                "cam1": [[52000], [66718000]]}
        stats = ec.sync_stats(rows, 1000)
        self.assertEqual(stats["matched_groups"], 2)
        self.assertEqual(stats["unmatched_reference_frames"], 1)
        self.assertEqual(stats["max_us"], 52)

    def test_writer_color_roundtrip_and_failure(self):
        for fail in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "cam0/data").mkdir(parents=True)
                (root / "imu0").mkdir()
                image = np.zeros((800, 1280, 3), np.uint8)
                image[:, :, 0] = 31
                image[:, :, 1] = 73
                image[:, :, 2] = 191
                writer = DatasetWriter(root, "png", 101, 4, camera_ids=("cam0",), color=True)
                if fail:
                    with patch("cv2.imwrite", return_value=False):
                        writer.start()
                        writer.submit(ImageTask("cam0", 123, 1, 2000, 400, image))
                        with self.assertRaises(RuntimeError):
                            writer.close()
                else:
                    writer.start()
                    writer.submit(ImageTask("cam0", 123, 1, 2000, 400, image))
                    writer.close()
                    actual = cv2.imread(str(root / "cam0/data/123.png"), cv2.IMREAD_UNCHANGED)
                    np.testing.assert_array_equal(actual, image)
                    self.assertNotIn("cam1", writer.written)

    def test_writer_overflow_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = DatasetWriter(Path(tmp), "png", 101, 4, camera_ids=("cam0",))
            for i in range(4):
                self.assertTrue(writer.submit(ImageTask("cam0", i, i, 2000, 400, None)))
            self.assertFalse(writer.submit(ImageTask("cam0", 4, 4, 2000, 400, None)))
            self.assertEqual(writer.dropped["cam0"], 1)

    def test_mocap_import_and_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meta").mkdir()
            (root / "mav0").mkdir()
            ec.write_json(root / "meta/manifest.yaml", {"qa": "RAW"})
            source = root / "source.csv"
            ec.write_csv(source, ["timestamp_ns", "px", "py", "pz", "qw", "qx", "qy", "qz", "tracking_valid"],
                         [[100, 0, 0, 0, 1, 0, 0, 0, 1], [200, 0, 0, 0, 0, 0, 0, 0, 0]])
            ec.import_mocap(source, root, "mocap_pc")
            self.assertEqual((root / "meta/mocap_source.csv").read_bytes(), source.read_bytes())
            manifest = json.loads((root / "meta/manifest.yaml").read_text())
            self.assertEqual(manifest["mocap"]["tracking_invalid"], 1)
            self.assertFalse((root / "mav0/state_groundtruth_estimate0").exists())
            for line in (root / "meta/checksums.sha256").read_text().splitlines():
                digest, filename = line.split("  ", 1)
                self.assertEqual(digest, hashlib.sha256((root / filename).read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                ec.import_mocap(source, root, "mocap_pc")

    def test_invalid_mocap_does_not_create_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ec.write_csv(root / "bad.csv",
                         ["timestamp_ns", "px", "py", "pz", "qw", "qx", "qy", "qz"],
                         [[1, 0, 0, 0, 2, 0, 0, 0]])
            with self.assertRaises(ValueError):
                ec.import_mocap(root / "bad.csv", root, "server")
            self.assertFalse((root / "mav0").exists())

    def run_fake_capture(self, camera_ids, interrupted=False, disk_failure=False, duplicate=False):
        import depthai as dai
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = ec.parse_args(camera_ids, ["--duration", ".01", "--warmup-seconds", "0",
                                              "--output-dir", tmp, "--queue-size", "2048"])
            image = np.zeros((800, 1280, 3), np.uint8)
            image[:] = (30, 80, 180)
            device = MagicMock()
            device.getConnectedCameras.return_value = [SimpleNamespace(name="CAM_"+c) for c in "ABCD"]
            device.getUsbSpeed.return_value = "SUPER"
            device.getDeviceId.return_value = "SIMULATED"
            device.getConnectedIMU.return_value = "BNO086"
            device.__enter__.return_value = device
            pipeline = MagicMock()
            pipeline.__enter__.return_value = pipeline
            pipeline.isRunning.return_value = True
            camera_nodes = []

            def output_queue(messages):
                queue = MagicMock()
                pending = [messages]
                queue.tryGetAll.side_effect = lambda: pending.pop() if pending else []
                return queue

            def create(node_type):
                node = MagicMock()
                if node_type == dai.node.Camera:
                    frames = []
                    for index in range(30):
                        frame = MagicMock()
                        frame.getSequenceNum.return_value = index
                        stamp_us = 33333*index if not duplicate else 0
                        frame.getTimestampDevice.return_value = timedelta(microseconds=stamp_us)
                        frame.getExposureTime.return_value = timedelta(microseconds=2000)
                        frame.getSensitivity.return_value = 400
                        frame.getCvFrame.return_value = image
                        frames.append(frame)
                    node.requestFullResolutionOutput.return_value.createOutputQueue.return_value = output_queue(frames)
                    camera_nodes.append(node)
                else:
                    packets = []
                    for index in range(400):
                        report = MagicMock()
                        report.getTimestampDevice.return_value = timedelta(microseconds=index*2500)
                        report.x, report.y, report.z = 0., 0., 9.8
                        packets.append(SimpleNamespace(acceleroMeter=report, gyroscope=report))
                    message = MagicMock()
                    message.getSequenceNum.return_value = 1
                    message.packets = packets
                    node.out.createOutputQueue.return_value = output_queue([message])
                return node

            pipeline.create.side_effect = create
            if interrupted:
                pipeline.isRunning.side_effect = KeyboardInterrupt
            pipeline_ctx = patch.object(dai, "Pipeline", return_value=pipeline)
            device_ctx = patch.object(dai, "Device", return_value=device)
            original_imwrite = cv2.imwrite
            with pipeline_ctx, device_ctx, patch("cv2.imwrite", side_effect=(
                    lambda *a: False) if disk_failure else original_imwrite):
                code = ec.capture(args, camera_ids)
            self.assertEqual(len(camera_nodes), len(camera_ids))
            for node, name in zip(camera_nodes, camera_ids):
                self.assertEqual(node.build.call_args.args[0].name, "CAM_"+"ABCD"[int(name[-1])])
            session = next(root.iterdir())
            self.assertEqual({p.name for p in (session / "mav0").iterdir()}, {*camera_ids, "imu0"})
            report = json.loads((session / "meta/recorder_stats.json").read_text())
            self.assertEqual(code, 2 if disk_failure or duplicate else 0, report["errors"])
            self.assertEqual(report["passed"], not (disk_failure or duplicate))
            self.assertFalse(report["hardware_performance_verified"])
            if interrupted:
                manifest = json.loads((session / "meta/manifest.yaml").read_text())
                self.assertEqual(manifest["status"], "interrupted")
            if not disk_failure and not duplicate:
                rows = ec.read_rows(session / "mav0/imu0/data.csv")
                self.assertEqual(len(rows), 400)
                self.assertTrue(all(len(r) == 7 for r in rows))
                for name in camera_ids:
                    rows = ec.read_rows(session / "mav0" / name / "data.csv")
                    self.assertTrue(all(len(r) == 2 for r in rows))
                self.assertTrue(ec.finalize(session)["passed"])

    def test_fake_single_camera_capture(self):
        self.run_fake_capture(("cam0",))

    def test_fake_four_camera_capture(self):
        self.run_fake_capture(ec.CAMERAS)

    def test_fake_ctrl_c(self):
        self.run_fake_capture(("cam0",), interrupted=True)

    def test_fake_disk_failure(self):
        self.run_fake_capture(("cam0",), disk_failure=True)

    def test_duplicate_camera_stops_without_overwrite(self):
        self.run_fake_capture(("cam0",), duplicate=True)


if __name__ == "__main__":
    unittest.main()
