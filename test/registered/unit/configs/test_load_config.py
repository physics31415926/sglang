"""Unit tests for srt/configs/load_config.py"""

import unittest
from unittest.mock import patch

from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestLoadFormat(CustomTestCase):
    def test_load_format_enum_values(self):
        """Test all LoadFormat enum values are accessible."""
        self.assertEqual(LoadFormat.AUTO, "auto")
        self.assertEqual(LoadFormat.PT, "pt")
        self.assertEqual(LoadFormat.SAFETENSORS, "safetensors")
        self.assertEqual(LoadFormat.NPCACHE, "npcache")
        self.assertEqual(LoadFormat.DUMMY, "dummy")
        self.assertEqual(LoadFormat.SHARDED_STATE, "sharded_state")
        self.assertEqual(LoadFormat.GGUF, "gguf")
        self.assertEqual(LoadFormat.BITSANDBYTES, "bitsandbytes")
        self.assertEqual(LoadFormat.MISTRAL, "mistral")
        self.assertEqual(LoadFormat.LAYERED, "layered")
        self.assertEqual(LoadFormat.FLASH_RL, "flash_rl")
        self.assertEqual(LoadFormat.JAX, "jax")
        self.assertEqual(LoadFormat.REMOTE, "remote")
        self.assertEqual(LoadFormat.REMOTE_INSTANCE, "remote_instance")
        self.assertEqual(LoadFormat.RDMA, "rdma")
        self.assertEqual(LoadFormat.LOCAL_CACHED, "local_cached")
        self.assertEqual(LoadFormat.FASTSAFETENSORS, "fastsafetensors")
        self.assertEqual(LoadFormat.PRIVATE, "private")
        self.assertEqual(LoadFormat.RUNAI_STREAMER, "runai_streamer")

    def test_load_format_from_string(self):
        """Test LoadFormat can be created from string."""
        self.assertEqual(LoadFormat("auto"), LoadFormat.AUTO)
        self.assertEqual(LoadFormat("safetensors"), LoadFormat.SAFETENSORS)


class TestLoadConfig(CustomTestCase):
    def test_default_initialization(self):
        """Test LoadConfig with default parameters."""
        config = LoadConfig()
        self.assertEqual(config.load_format, LoadFormat.AUTO)
        self.assertIsNone(config.download_dir)
        self.assertEqual(config.model_loader_extra_config, {})
        self.assertEqual(config.ignore_patterns, ["original/**/*"])
        self.assertIsNone(config.decryption_key_file)
        self.assertEqual(config.decrypt_max_concurrency, -1)

    def test_load_format_string_conversion(self):
        """Test load_format string is converted to LoadFormat enum."""
        config = LoadConfig(load_format="safetensors")
        self.assertEqual(config.load_format, LoadFormat.SAFETENSORS)

    def test_load_format_case_insensitive(self):
        """Test load_format string conversion is case-insensitive."""
        config = LoadConfig(load_format="SAFETENSORS")
        self.assertEqual(config.load_format, LoadFormat.SAFETENSORS)

    def test_load_format_enum_direct(self):
        """Test load_format can be set directly as enum."""
        config = LoadConfig(load_format=LoadFormat.PT)
        self.assertEqual(config.load_format, LoadFormat.PT)

    def test_ignore_patterns_default(self):
        """Test ignore_patterns defaults to original/**/*."""
        config = LoadConfig()
        self.assertEqual(config.ignore_patterns, ["original/**/*"])

    def test_ignore_patterns_custom(self):
        """Test custom ignore_patterns are preserved."""
        patterns = ["*.tmp", "cache/**/*"]
        config = LoadConfig(ignore_patterns=patterns)
        self.assertEqual(config.ignore_patterns, patterns)

    def test_ignore_patterns_empty_list_uses_default(self):
        """Test empty ignore_patterns list uses default."""
        config = LoadConfig(ignore_patterns=[])
        self.assertEqual(config.ignore_patterns, ["original/**/*"])

    def test_model_loader_extra_config_dict(self):
        """Test model_loader_extra_config as dict."""
        extra = {"key": "value"}
        config = LoadConfig(model_loader_extra_config=extra)
        self.assertEqual(config.model_loader_extra_config, extra)

    def test_model_loader_extra_config_json_string(self):
        """Test model_loader_extra_config as JSON string."""
        config = LoadConfig(model_loader_extra_config='{"key": "value"}')
        self.assertEqual(config.model_loader_extra_config, {"key": "value"})

    def test_download_dir(self):
        """Test download_dir is set correctly."""
        config = LoadConfig(download_dir="/tmp/models")
        self.assertEqual(config.download_dir, "/tmp/models")

    def test_decryption_options(self):
        """Test decryption-related options."""
        config = LoadConfig(
            decryption_key_file="/path/to/key", decrypt_max_concurrency=4
        )
        self.assertEqual(config.decryption_key_file, "/path/to/key")
        self.assertEqual(config.decrypt_max_concurrency, 4)

    def test_modelopt_config_creation(self):
        """Test ModelOptConfig is created in __post_init__."""
        config = LoadConfig(
            modelopt_checkpoint_restore_path="/restore",
            modelopt_checkpoint_save_path="/save",
            modelopt_export_path="/export",
        )
        self.assertIsNotNone(config.modelopt_config)
        self.assertEqual(
            config.modelopt_config.checkpoint_restore_path, "/restore"
        )
        self.assertEqual(config.modelopt_config.checkpoint_save_path, "/save")
        self.assertEqual(config.modelopt_config.export_path, "/export")

    def test_remote_instance_options(self):
        """Test remote instance weight loader options."""
        config = LoadConfig(
            remote_instance_weight_loader_seed_instance_ip="192.168.1.1",
            remote_instance_weight_loader_seed_instance_service_port=8080,
            remote_instance_weight_loader_send_weights_group_ports=[9000, 9001],
            remote_instance_weight_loader_backend="nccl",
        )
        self.assertEqual(
            config.remote_instance_weight_loader_seed_instance_ip, "192.168.1.1"
        )
        self.assertEqual(
            config.remote_instance_weight_loader_seed_instance_service_port, 8080
        )
        self.assertEqual(
            config.remote_instance_weight_loader_send_weights_group_ports,
            [9000, 9001],
        )
        self.assertEqual(config.remote_instance_weight_loader_backend, "nccl")

    def test_rl_quant_profile(self):
        """Test RL quantization profile option."""
        config = LoadConfig(rl_quant_profile="/root/profile.7b.pt")
        self.assertEqual(config.rl_quant_profile, "/root/profile.7b.pt")

    def test_draft_model_idx(self):
        """Test draft_model_idx for multi-layer MTP."""
        config = LoadConfig(draft_model_idx=2)
        self.assertEqual(config.draft_model_idx, 2)

    @patch("sglang.srt.utils.is_hip")
    def test_rocm_unsupported_format_validation(self, mock_is_hip):
        """Test ROCm unsupported format validation (currently empty list)."""
        mock_is_hip.return_value = True
        # Currently rocm_not_supported_load_format is empty, so all formats work
        config = LoadConfig(load_format="safetensors")
        self.assertEqual(config.load_format, LoadFormat.SAFETENSORS)


if __name__ == "__main__":
    unittest.main()
