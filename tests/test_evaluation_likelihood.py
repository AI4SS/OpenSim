"""Adapter base-model overrides are checked without importing real model runtimes."""
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

from opensim.evaluation.likelihood import PolicyBundle


class EvaluationLikelihoodTests(unittest.TestCase):
    def load_bundle(self, override):
        metadata = SimpleNamespace(base_model_name_or_path="/previous/host/base")
        peft_config = SimpleNamespace(from_pretrained=Mock(return_value=metadata))
        peft_model = SimpleNamespace(from_pretrained=Mock(return_value=Mock()))
        model_loader = SimpleNamespace(from_pretrained=Mock(return_value=Mock()))
        tokenizer_loader = SimpleNamespace(from_pretrained=Mock(return_value=SimpleNamespace(pad_token=None, eos_token="eos")))
        torch = SimpleNamespace(device=lambda value: SimpleNamespace(type="cpu"), float16="float16", float32="float32",
                                cuda=SimpleNamespace(is_available=lambda: False))
        modules = {"torch": torch, "peft": SimpleNamespace(PeftConfig=peft_config, PeftModel=peft_model),
                   "transformers": SimpleNamespace(AutoModelForCausalLM=model_loader, AutoTokenizer=tokenizer_loader)}
        with patch.dict(sys.modules, modules):
            PolicyBundle("adapter", device="cpu", base_model=override)
        return model_loader.from_pretrained.call_args.args[0], metadata

    def test_explicit_override_takes_precedence_without_mutating_metadata(self):
        loaded, metadata = self.load_bundle("/current/host/base")
        self.assertEqual(loaded, "/current/host/base")
        self.assertEqual(metadata.base_model_name_or_path, "/previous/host/base")

    def test_adapter_metadata_remains_default(self):
        loaded, _ = self.load_bundle(None)
        self.assertEqual(loaded, "/previous/host/base")


if __name__ == "__main__":
    unittest.main()
