"""Contract tests: real parsing/rules/CLI and local HTTP with a fixture model server.

The HTTP server and lightweight Qwen doubles test transport and token handling,
not model quality or checkpoint loading. Tests require only Python's standard library.
"""

from __future__ import annotations

import base64
import copy
from contextlib import nullcontext
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest

from vlm_director import Director
from vlm_director.cli import save_result
from vlm_director.evaluation import summarize_reports
from vlm_director.models.vlm_wrapper import (
    CompatibleAPIWrapper, GenerationSettings, ModelError, QwenVLMWrapper, ReplayWrapper,
)
from vlm_director.parser import parse_response
from vlm_director.schemas.scenario_schema import Scenario, SchemaError

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples/director"
# A genuine 1x1 PNG is enough to verify byte transport without importing Pillow.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a/1sAAAAASUVORK5CYII=")


class StaticWrapper:
    """A named fixture wrapper exposes precisely what the Director sent."""

    supports_vision = True
    metadata = {"backend": "test-double", "model_id": "fixture", "is_mock": True}

    def __init__(self, raw: str) -> None:
        """Store a response and initialize call tracing."""
        self.raw, self.calls = raw, []

    def generate(self, prompt, image_path, settings):
        """Record inputs before returning fixed text."""
        self.calls.append((prompt, image_path, settings))
        return self.raw


class DirectorContracts(unittest.TestCase):
    """Validate schema/feasibility behavior at the public Director boundary."""

    def setUp(self):
        """Reload independent fixture state for every test."""
        self.full = Scenario.from_file(EXAMPLES / "scenarios/urban_full.json")
        self.minimal = Scenario.from_file(EXAMPLES / "scenarios/urban.json")
        self.raw = (EXAMPLES / "responses/cut_in.json").read_text(encoding="utf-8")

    def test_units_are_normalized_without_input_mutation(self):
        """The supplied 40 km/h example must not become 40 m/s."""
        data = json.loads((EXAMPLES / "scenarios/urban.json").read_text())
        original = copy.deepcopy(data)
        normalized = Scenario.from_dict(data)
        self.assertEqual(data, original)
        self.assertAlmostEqual(normalized.ego["speed"], 40 / 3.6)
        self.assertEqual(normalized.data["speed_unit"], "m/s")
        self.assertEqual(Scenario.from_dict(normalized.data).data, normalized.data)

    def test_input_duplicates_invalid_units_and_unknown_lanes(self):
        """Bad actor identities, units, and map IDs fail before generation."""
        for mutate in (
            lambda d: d["actors"].append(copy.deepcopy(d["actors"][0])),
            lambda d: d.update(speed_unit="mph"),
            lambda d: d["ego"].update(speed=True),
            lambda d: d["ego"].update(lane=999),
        ):
            data = copy.deepcopy(self.full.data)
            mutate(data)
            with self.assertRaises(SchemaError):
                Scenario.from_dict(data)

    def test_export_is_not_a_snapshot(self):
        """Existing pipeline output requires an explicit adapter, not guessed state."""
        with self.assertRaisesRegex(SchemaError, "snapshot adapter"):
            Scenario.from_file(ROOT / "output/scenario.json")

    def test_all_three_rule_events(self):
        """Each requested family uses a suitable existing actor and passes checks."""
        for kind, actor in (("cut_in", "car_1"), ("sudden_braking", "lead_1"),
                            ("pedestrian_crossing", "ped_1")):
            result = Director(0, rule_event=kind).run(self.full)
            self.assertEqual(result.prediction["actor"], actor)
            self.assertEqual(result.prediction["event"], kind)
            self.assertEqual(result.report["feasibility"]["status"], "pass")
            self.assertFalse(result.report["feasibility"]["simulation_verified"])

    def test_seed_reproducibility_and_unknown_geometry(self):
        """A repeatable choice is separate from a claim of known geometry."""
        director = Director(0, settings=GenerationSettings(seed=7))
        self.assertEqual(director.run(self.full).prediction, director.run(self.full).prediction)
        result = director.run(self.minimal)
        self.assertTrue(result.success)
        self.assertEqual(result.report["feasibility"]["status"], "unknown")

    def test_no_actor_is_invented(self):
        """Empty input and an unavailable event fail without synthesizing IDs."""
        data = copy.deepcopy(self.full.data)
        data["actors"] = []
        with self.assertRaisesRegex(SchemaError, "No eligible actor"):
            Director(0).run(data)
        with self.assertRaises(SchemaError):
            Director(0, rule_event="pedestrian_crossing").run(self.minimal)

    def test_fences_and_legacy_are_explicit(self):
        """Recovered JSON and lossless legacy aliases are distinguishable in reports."""
        parsed = parse_response("```json\n" + self.raw + "\n```")
        self.assertFalse(parsed.strict_json_valid)
        self.assertTrue(parsed.recovered)
        obj = json.loads(self.raw)
        obj["corner_case_type"] = obj.pop("event")
        obj["target_actor"] = obj.pop("actor")
        obj["parameters"] = "{}"
        parsed = parse_response(json.dumps(obj))
        self.assertTrue(parsed.legacy_normalized)
        self.assertEqual(parsed.event["event"], "cut_in")
        obj["trigger"] = "when close"
        with self.assertRaises(SchemaError):
            parse_response(json.dumps(obj))

    def test_unsafe_or_ambiguous_json_is_rejected(self):
        """Never guess a JSON object out of prose or execute a Python expression."""
        for raw in ('{"event":"cut_in","event":"sudden_braking"}',
                    '{"x":NaN}', '{"x":Infinity}', '[]', '{} {}',
                    'Here is the result: ' + self.raw, '__import__("os")'):
            with self.subTest(raw=raw), self.assertRaises(SchemaError):
                parse_response(raw)

    def test_unknown_actor_is_withheld_but_schema_counted(self):
        """A syntactically correct hallucinated actor must not reach an executor."""
        obj = json.loads(self.raw)
        obj["actor"] = "invented"
        result = Director(1, StaticWrapper(json.dumps(obj))).run(self.full)
        self.assertIsNone(result.prediction)
        self.assertTrue(result.report["schema_valid"])
        self.assertEqual(result.report["feasibility"]["status"], "fail")

    def test_explicit_topology_overrides_integer_adjacency(self):
        """Consecutive lane numbers cannot override known map topology."""
        data = copy.deepcopy(self.full.data)
        data["map"]["adjacent_lanes"] = [[1, 2]]
        result = Director(1, StaticWrapper(self.raw)).run(data)
        self.assertFalse(result.success)
        self.assertIn("adjacent", result.report["error"])

    def test_lateral_acceleration_and_braking_bounds(self):
        """Known physical contradictions are distinct from syntax/schema failures."""
        obj = json.loads(self.raw)
        obj["behavior"]["duration_s"] = 2
        result = Director(1, StaticWrapper(json.dumps(obj))).run(self.full)
        self.assertTrue(result.report["schema_valid"])
        self.assertIn("lateral acceleration", result.report["error"])
        obj.update(event="sudden_braking", actor="lead_1", behavior={
            "type": "brake", "deceleration_mps2": -8, "duration_s": 4})
        result = Director(1, StaticWrapper(json.dumps(obj))).run(self.full)
        self.assertIn("negative final speed", result.report["error"])

    def test_opposite_direction_and_behind_actor_are_rejected(self):
        """An adjacent oncoming car or car behind ego cannot satisfy this baseline."""
        for changes in ({"direction": "opposite"}, {"s_m": 90}):
            data = copy.deepcopy(self.full.data)
            data["actors"][0].update(changes)
            self.assertFalse(Director(1, StaticWrapper(self.raw)).run(data).success)

    def test_unreachable_distance_trigger_is_uncertain(self):
        """Static placement alone cannot guarantee the event will ever trigger."""
        data = copy.deepcopy(self.full.data)
        data["actors"][0]["speed"] = data["ego"]["speed"] + 1
        result = Director(1, StaticWrapper(self.raw)).run(data)
        self.assertEqual(result.report["feasibility"]["status"], "unknown")
        self.assertIn("reachability", result.report["feasibility"]["warnings"][0])

    def test_misspelled_or_low_level_commands_are_rejected(self):
        """Unknown fields cannot smuggle steering commands through parameters."""
        obj = json.loads(self.raw)
        obj["parameters"] = {"throttle": 0.5}
        with self.assertRaises(SchemaError):
            parse_response(json.dumps(obj))
        obj["parameters"] = {}
        obj["behavior"]["duration"] = obj["behavior"].pop("duration_s")
        with self.assertRaises(SchemaError):
            parse_response(json.dumps(obj))

    def test_json_and_schema_failures_have_different_metrics(self):
        """Valid syntax is still counted if the response lacks a valid event schema."""
        result = Director(1, StaticWrapper('{"event":"unsupported"}')).run(self.full)
        self.assertTrue(result.report["strict_json_valid"])
        self.assertFalse(result.report["schema_valid"])
        result = Director(1, StaticWrapper('{broken')).run(self.full)
        self.assertFalse(result.report["json_object_valid"])

    def test_image_requirement_and_text_ablation(self):
        """Baseline 2 sends an image, while baseline 1 cannot accidentally consume it."""
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "bev.png"
            image.write_bytes(PNG)
            model = StaticWrapper(self.raw)
            result = Director(2, model).run(self.full, image)
            self.assertTrue(result.success)
            self.assertEqual(model.calls[0][1], image.resolve())
            self.assertIn('"speed_unit": "m/s"', model.calls[0][0])
            with self.assertRaises(ValueError):
                Director(2, model).run(self.full)
            with self.assertRaises(ValueError):
                Director(1, model).run(self.full, image)

    def test_replay_is_marked_and_failures_stay_in_metrics(self):
        """Fixtures cannot be confused with model performance or hide failures."""
        good = Director(1, ReplayWrapper(EXAMPLES / "responses/cut_in.json")).run(self.full)
        bad = Director(1, StaticWrapper("broken")).run(self.full)
        unknown = Director(0).run(self.minimal)
        summary = summarize_reports([good.report, bad.report, unknown.report])
        self.assertEqual(summary["attempts"], 3)
        self.assertTrue(summary["interface_test_only"])
        self.assertAlmostEqual(summary["schema_valid_rate"], 2 / 3)
        self.assertAlmostEqual(summary["static_feasibility_pass_rate"], 1 / 3)
        self.assertAlmostEqual(summary["static_feasibility_unknown_rate"], 1 / 3)

    def test_failure_removes_stale_prediction_and_protects_input(self):
        """Output replacement must not leave a previous success after a failed run."""
        good = Director(0, rule_event="cut_in").run(self.full)
        bad = Director(1, StaticWrapper("broken")).run(self.full)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save_result(good, output, "demo")
            self.assertTrue((output / "demo_prediction.json").exists())
            save_result(bad, output, "demo")
            self.assertFalse((output / "demo_prediction.json").exists())
            with self.assertRaises(ValueError):
                save_result(good, output, "demo", protected_paths=(output / "demo_report.json",))

    def test_cli_runs_from_another_directory_and_returns_failure_code(self):
        """Direct script execution works independently of cwd and writes exact events."""
        with tempfile.TemporaryDirectory() as directory:
            cmd = [sys.executable, str(ROOT / "scripts/run_director.py"), "--scenario",
                   str(EXAMPLES / "scenarios/urban_full.json"), "--model", "rule",
                   "--event", "cut_in", "--output-dir", directory]
            result = subprocess.run(cmd, cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            event = json.loads((Path(directory) / "urban_full_prediction.json").read_text())
            self.assertEqual(event["actor"], "car_1")
            cmd[cmd.index("--scenario") + 1] = str(Path(directory) / "missing.json")
            result = subprocess.run(cmd, cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)


class HTTPContracts(unittest.TestCase):
    """Exercise actual local HTTP requests, including multimodal JSON payloads."""

    def setUp(self):
        """Start a fixture endpoint on an ephemeral loopback port."""
        self.requests = []
        self.response = (EXAMPLES / "responses/cut_in.json").read_text()
        self.mode = "normal"
        owner = self

        class Handler(BaseHTTPRequestHandler):
            """Return controlled capability and transport failures for regression tests."""

            def log_message(self, *args):
                """Keep routine server access logs out of test output."""
                pass

            def do_POST(self):
                """Capture bytes and return one compatible completion envelope."""
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append(payload)
                status = 200
                if owner.mode == "no-json" and "response_format" in payload:
                    status, body = 400, {"error": "response_format is unsupported"}
                elif owner.mode == "bad-request":
                    status, body = 400, {"error": "unknown model"}
                else:
                    body = {"choices": [{"message": {"content": owner.response}}]}
                encoded = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.model = CompatibleAPIWrapper(f"http://127.0.0.1:{self.server.server_port}/v1", "fixture")

    def tearDown(self):
        """Release the socket and stop the background server after each test."""
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_text_request_and_bev_bytes(self):
        """Text-only and multimodal calls share model/settings but differ in content."""
        full = Scenario.from_file(EXAMPLES / "scenarios/urban_full.json")
        self.assertTrue(Director(1, self.model).run(full).success)
        self.assertIsInstance(self.requests[0]["messages"][1]["content"], str)
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "bev.png"
            image.write_bytes(PNG)
            self.assertTrue(Director(2, self.model).run(full, image).success)
        content = self.requests[1]["messages"][1]["content"]
        self.assertEqual(content[1]["type"], "image_url")
        self.assertEqual(base64.b64decode(content[1]["image_url"]["url"].split(",")[1]), PNG)
        self.assertEqual(self.requests[1]["seed"], 0)
        self.assertEqual(self.requests[1]["max_tokens"], 512)

    def test_only_json_capability_errors_are_retried(self):
        """Reuse the repository fallback while avoiding unrelated-request retries."""
        self.mode = "no-json"
        self.assertEqual(self.model.generate("prompt", None, GenerationSettings()), self.response)
        self.assertEqual(len(self.requests), 2)
        self.assertNotIn("response_format", self.requests[1])
        self.assertFalse(self.model.metadata["json_mode"])
        self.mode = "bad-request"
        with self.assertRaises(ModelError):
            self.model.generate("prompt", None, GenerationSettings())
        self.assertEqual(len(self.requests), 3)


class QwenTokenContract(unittest.TestCase):
    """Lightweight doubles verify the optional backend without downloading weights."""

    def test_template_image_and_completion_only_decoding(self):
        """Verify tokenizer arguments, image attachment, device move, and token slicing."""
        class Inputs(dict):
            """Small BatchFeature analogue with the attribute Qwen code consumes."""
            input_ids = [[101, 102]]

            def to(self, device):
                """Record the transfer target while returning the same feature batch."""
                self.device = device
                return self

        calls = []
        decoded = []
        inputs = Inputs(input_ids=Inputs.input_ids)

        def template(messages, **kwargs):
            """Capture all multimodal processor arguments."""
            calls.append((messages, kwargs))
            return inputs

        def decode(ids, **kwargs):
            """Capture sliced completion IDs rather than prompt IDs."""
            decoded.append(ids)
            return ['{"event":"cut_in"}']

        wrapper = QwenVLMWrapper(device="cpu")
        wrapper._model = SimpleNamespace(device="cpu", generate=lambda **kwargs: [[101, 102, 201, 202]])
        wrapper._processor = SimpleNamespace(apply_chat_template=template, batch_decode=decode)
        wrapper._torch = SimpleNamespace(
            random=SimpleNamespace(fork_rng=lambda **kwargs: nullcontext()),
            manual_seed=lambda seed: None, inference_mode=nullcontext)
        settings = GenerationSettings()
        wrapper.generate("text prompt", None, settings)
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "bev.png"
            image.write_bytes(PNG)
            wrapper.generate("vision prompt", image, replace(settings, temperature=0.3))
        self.assertEqual(decoded, [[[201, 202]], [[201, 202]]])
        self.assertEqual(calls[0][0][1]["content"][0]["type"], "text")
        self.assertEqual(calls[1][0][1]["content"][0]["type"], "image")
        self.assertTrue(calls[1][1]["tokenize"])
        self.assertEqual(inputs.device, "cpu")


if __name__ == "__main__":
    unittest.main()
