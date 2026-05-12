"""Unit tests for srt/configs/device_config.py"""

import unittest

from sglang.srt.configs.device_config import SUPPORTED_DEVICES, DeviceConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")


class TestDeviceConfig(CustomTestCase):
    def test_default_initialization(self):
        """Test DeviceConfig with default parameters."""
        config = DeviceConfig()
        self.assertEqual(config.device_type, "cuda")
        self.assertEqual(config.device.type, "cuda")
        self.assertEqual(config.gpu_id, -1)

    def test_cuda_device(self):
        """Test CUDA device initialization."""
        config = DeviceConfig(device="cuda", gpu_id=0)
        self.assertEqual(config.device_type, "cuda")
        self.assertEqual(config.device.type, "cuda")
        self.assertEqual(config.gpu_id, 0)

    def test_cpu_device(self):
        """Test CPU device initialization."""
        config = DeviceConfig(device="cpu", gpu_id=-1)
        self.assertEqual(config.device_type, "cpu")
        self.assertEqual(config.device.type, "cpu")
        self.assertEqual(config.gpu_id, -1)

    def test_all_supported_devices(self):
        """Test all supported device types can be initialized."""
        for device_type in SUPPORTED_DEVICES:
            config = DeviceConfig(device=device_type)
            self.assertEqual(config.device_type, device_type)
            self.assertEqual(config.device.type, device_type)

    def test_unsupported_device_raises_error(self):
        """Test that unsupported device type raises RuntimeError."""
        with self.assertRaises(RuntimeError) as context:
            DeviceConfig(device="unsupported_device")
        self.assertIn("Not supported device type", str(context.exception))

    def test_gpu_id_assignment(self):
        """Test gpu_id is correctly assigned."""
        config = DeviceConfig(device="cuda", gpu_id=3)
        self.assertEqual(config.gpu_id, 3)

    def test_negative_gpu_id(self):
        """Test negative gpu_id is allowed (default behavior)."""
        config = DeviceConfig(device="cuda", gpu_id=-1)
        self.assertEqual(config.gpu_id, -1)


if __name__ == "__main__":
    unittest.main()
