"""Cloud-only decisions: escalate rule, fallback, redaction, and tool surface.

No network and no Keychain: every call injects a transport, and keys are passed
through the environment so the tests run on any machine.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sys12  # noqa: E402


def cfg(**overrides) -> dict:
    base = {
        "cloud": {
            "base_url": "https://cloud.example",
            "decision_model": "clef",
            "chat_model": "deepseek",
            "api_key_env": "OLLAMA_API_KEY",
            "keychain_service": "unused-in-tests",
            "api_key_file": "",
        },
        "route_gate_threshold": 0.5,
        "decision_timeout": 60,
        "chat_timeout": 300,
        "omlx": {"base_url": "http://127.0.0.1:8000", "api_key_file": ""},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key].update(value)
        else:
            base[key] = value
    return base


def choice_answer(label="beta", confidence=0.8) -> dict:
    return {"answers": {"q": {"type": "choice", "choice": label,
                              "confidence": confidence,
                              "probabilities": {label: confidence}}},
            "usage": {"input_tokens": 10}}


def chat_reply(content: str, model: str = "deepseek") -> dict:
    return {"model": model,
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5}}


def env_key():
    return mock.patch.dict(os.environ, {"OLLAMA_API_KEY": "test-key"}, clear=False)


class DecideTests(unittest.TestCase):
    def test_cloud_succeeds_with_one_call_and_no_fallback(self) -> None:
        seen: list[str] = []

        def transport(url, body, token, timeout):
            seen.append(url)
            return choice_answer("beta", 0.8)

        with env_key():
            out = sys12.decide(cfg(), "Which label?",
                               {"q": {"type": "choice",
                                      "criteria": {"alpha": "a", "beta": "b"}}},
                               transport=transport)
        self.assertEqual(seen, ["https://cloud.example/v1/systemone"])
        self.assertEqual(out["backend"], "ollama-cloud")
        self.assertIsNone(out["fallback_reason"])
        self.assertEqual(out["answers"]["q"]["choice"], "beta")
        self.assertFalse(out["answers"]["q"]["escalate"])
        self.assertFalse(out["escalate"])

    def test_retryable_cloud_failure_falls_back_to_chat(self) -> None:
        seen: list[str] = []

        def transport(url, body, token, timeout):
            seen.append(url)
            if url.endswith("/v1/systemone"):
                raise sys12.UpstreamError(503, "HTTP 503: unavailable", True)
            return chat_reply('{"q": "beta"}')

        with env_key():
            out = sys12.decide(cfg(), "Which label?",
                               {"q": {"type": "choice",
                                      "criteria": {"alpha": "a", "beta": "b"}}},
                               transport=transport)
        self.assertEqual(seen, ["https://cloud.example/v1/systemone",
                                "https://cloud.example/v1/chat/completions"])
        self.assertEqual(out["backend"], "ollama-cloud-chat")
        self.assertIn("503", out["fallback_reason"])
        self.assertEqual(out["answers"]["q"]["choice"], "beta")
        # An emulated answer has no probabilities, so it can never stand.
        self.assertTrue(out["answers"]["q"]["escalate"])
        self.assertIsNone(out["answers"]["q"]["confidence"])

    def test_credential_error_does_not_fall_back(self) -> None:
        calls: list[str] = []

        def transport(url, body, token, timeout):
            calls.append(url)
            raise sys12.UpstreamError(401, "HTTP 401: token=test-key", False)

        with env_key():
            with self.assertRaises(sys12.Sys12Error) as caught:
                sys12.decide(cfg(), "Which?",
                             {"q": {"type": "choice",
                                    "criteria": {"a": "a", "b": "b"}}},
                             transport=transport)
        self.assertEqual(len(calls), 1, "401 must not be retried on chat")
        self.assertNotIn("test-key", str(caught.exception))

    def test_malformed_success_falls_back_to_chat(self) -> None:
        def transport(url, body, token, timeout):
            if url.endswith("/v1/systemone"):
                return {"usage": {}}  # 2xx with no answers
            return chat_reply('{"q": "A"}')

        with env_key():
            out = sys12.decide(cfg(), "Which?",
                               {"q": {"type": "choice",
                                      "criteria": {"A": "first", "B": "second"}}},
                               transport=transport)
        self.assertEqual(out["backend"], "ollama-cloud-chat")
        self.assertIn("no answers", out["fallback_reason"])
        self.assertEqual(out["answers"]["q"]["choice"], "A")

    def test_secret_is_redacted_when_both_backends_fail(self) -> None:
        def transport(url, body, token, timeout):
            raise sys12.UpstreamError(None, "boom secret-key-123", True)

        with mock.patch.dict(os.environ, {"OLLAMA_API_KEY": "secret-key-123"}):
            with self.assertRaises(sys12.Sys12Error) as caught:
                sys12.decide(cfg(), "Which?",
                             {"q": {"type": "choice",
                                    "criteria": {"a": "a", "b": "b"}}},
                             transport=transport)
        self.assertNotIn("secret-key-123", str(caught.exception))
        self.assertIn("[redacted]", str(caught.exception))


class EscalateTests(unittest.TestCase):
    def _decide_with(self, payload, questions=None):
        questions = questions or {"q": {"type": "choice",
                                        "criteria": {"alpha": "a", "beta": "b"}}}
        with env_key():
            return sys12.decide(cfg(), "Which?", questions,
                                transport=lambda *a: payload)

    def test_choice_confidence_below_threshold_escalates(self) -> None:
        out = self._decide_with(choice_answer("beta", 0.5499))
        self.assertTrue(out["answers"]["q"]["escalate"])
        self.assertIn("0.55", out["answers"]["q"]["escalate_reason"])
        self.assertTrue(out["escalate"])

    def test_choice_confidence_exactly_at_threshold_stands(self) -> None:
        out = self._decide_with(choice_answer("beta", 0.55))
        self.assertFalse(out["answers"]["q"]["escalate"])
        self.assertFalse(out["escalate"])

    def test_yes_no_inside_the_band_escalates_at_both_ends(self) -> None:
        for probability in (0.11, 0.5, 0.89):
            out = self._yes_no(probability)
            self.assertTrue(out["answers"]["q"]["escalate"],
                            f"p={probability} should escalate")

    def _yes_no(self, probability: float):
        payload = {"answers": {"q": {"type": "noul", "noul": probability}}}
        with env_key():
            return sys12.decide(
                cfg(), "Valid?",
                {"q": {"type": "noul", "criteria": {"true": "t", "false": "f"}}},
                transport=lambda *a: payload,
            )

    def test_yes_no_at_the_band_edges_stands(self) -> None:
        for probability in (0.10, 0.90, 0.02, 0.98):
            out = self._yes_no(probability)
            self.assertFalse(out["answers"]["q"]["escalate"],
                             f"p={probability} should stand")

    def test_yes_no_still_reports_the_flip_at_half(self) -> None:
        self.assertEqual(self._yes_no(0.49)["answers"]["q"]["answer"], "no")
        self.assertEqual(self._yes_no(0.51)["answers"]["q"]["answer"], "yes")

    def test_score_maps_to_its_label_and_escalates_on_low_confidence(self) -> None:
        payload = {"answers": {"q": {"type": "score", "score": 2.0,
                                     "confidence": 0.31,
                                     "probabilities": {"2": 0.31}}}}
        out = self._decide_with(
            payload, {"q": {"type": "score", "criteria": ["low", "mid", "high"]}}
        )
        self.assertEqual(out["answers"]["q"]["answer"], "high")
        self.assertTrue(out["answers"]["q"]["escalate"])


class QuestionShapeTests(unittest.TestCase):
    def test_options_beside_type_become_criteria(self) -> None:
        normalized = sys12.normalize_questions(
            {"pick": {"type": "choice", "A": "canary", "B": "full"}}
        )
        self.assertEqual(normalized["pick"]["criteria"], {"A": "canary", "B": "full"})

    def test_score_dict_becomes_an_ordered_list(self) -> None:
        normalized = sys12.normalize_questions(
            {"sev": {"type": "score", "criteria": {"a": "low", "b": "high"}}}
        )
        self.assertEqual(normalized["sev"]["criteria"], ["low", "high"])

    def test_choice_list_becomes_lettered_labels(self) -> None:
        normalized = sys12.normalize_questions(
            {"pick": {"type": "choice", "criteria": ["one", "two", "three"]}}
        )
        self.assertEqual(normalized["pick"]["criteria"],
                         {"A": "one", "B": "two", "C": "three"})

    def test_yes_no_alias_and_default_instructions(self) -> None:
        normalized = sys12.normalize_questions({"ok": {"type": "yes_no"}})
        self.assertEqual(normalized["ok"]["type"], "noul")
        self.assertTrue(normalized["ok"]["instructions"])

    def test_single_criteria_and_duplicate_score_levels_are_rejected(self) -> None:
        with self.assertRaises(sys12.Sys12Error):
            sys12.normalize_questions({"q": {"type": "score", "criteria": ["only"]}})
        with self.assertRaises(sys12.Sys12Error):
            sys12.normalize_questions(
                {"q": {"type": "score", "criteria": ["low", "low", "high"]}}
            )

    def test_unknown_type_is_rejected(self) -> None:
        with self.assertRaises(sys12.Sys12Error):
            sys12.normalize_questions({"q": {"type": "essay"}})

    def test_blank_state_falls_back_to_instructions(self) -> None:
        captured: dict = {}

        def transport(url, body, token, timeout):
            captured.update(body)
            return {"answers": {"q": {"type": "noul", "noul": 0.99}}}

        with env_key():
            sys12.decide(cfg(), "   ", {"q": {"type": "noul"}}, transport=transport)
        self.assertTrue(captured["state"].strip())

    def test_notes_apply_to_one_call_only(self) -> None:
        seen: list[dict] = []

        def transport(url, body, token, timeout):
            seen.append(body["questions"]["q"]["criteria"])
            return choice_answer("A")

        questions = {"q": {"type": "choice", "criteria": {"A": "one", "B": "two"}}}
        with env_key():
            sys12.decide(cfg(), "Which?", questions,
                         notes={"A": "the first option"}, transport=transport)
            sys12.decide(cfg(), "Which?", questions, transport=transport)
        self.assertEqual(seen[0]["A"], "the first option")
        self.assertEqual(seen[1]["A"], "one")


class ImageTests(unittest.TestCase):
    PNG = ("iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAFElEQVR4nGP8"
           "z8Dwn4GBgYGJAQ0AAB0eAQFq7h0cAAAAAElFTkSuQmCC")

    def test_bare_base64_is_wrapped_with_a_data_uri(self) -> None:
        self.assertTrue(sys12.to_data_uri(self.PNG).startswith("data:image/png;base64,"))

    def test_existing_data_uri_is_kept(self) -> None:
        uri = f"data:image/jpeg;base64,{self.PNG}"
        self.assertEqual(sys12.to_data_uri(uri), uri)

    def test_invalid_base64_is_rejected(self) -> None:
        with self.assertRaises(sys12.Sys12Error):
            sys12.to_data_uri("this is a file path, not base64 !!")

    def test_images_reach_the_cloud_body(self) -> None:
        captured: dict = {}

        def transport(url, body, token, timeout):
            captured.update(body)
            return choice_answer("A")

        with env_key():
            sys12.decide(cfg(), "What is shown?",
                         {"q": {"type": "choice",
                                "criteria": {"A": "red", "B": "blue"}}},
                         images=[self.PNG], transport=transport)
        self.assertEqual(len(captured["images"]), 1)
        self.assertTrue(captured["images"][0].startswith("data:image/png;base64,"))


class AskTests(unittest.TestCase):
    def test_chat_completion_shape(self) -> None:
        def transport(url, body, token, timeout):
            self.assertEqual(url, "https://cloud.example/v1/chat/completions")
            self.assertEqual(body["messages"][0]["role"], "system")
            return chat_reply("pong")

        with env_key():
            out = sys12.ask(cfg(), "ping", system="be terse", max_tokens=8,
                            transport=transport)
        self.assertEqual(out["content"], "pong")
        self.assertEqual(out["usage"]["completion_tokens"], 5)

    def test_blank_prompt_is_rejected_before_transport(self) -> None:
        with self.assertRaises(sys12.Sys12Error):
            sys12.ask(cfg(), "  ", transport=lambda *a: self.fail("no call"))


class RouteTests(unittest.TestCase):
    def _questions(self):
        return {"q": {"type": "choice", "criteria": {"A": "one", "B": "two"}}}

    def test_free_form_task_skips_the_gate(self) -> None:
        urls: list[str] = []

        def transport(url, body, token, timeout):
            urls.append(url)
            return chat_reply("here is a plan")

        with env_key():
            out = sys12.route(cfg(), "Design a backup plan", transport=transport)
        self.assertEqual(out["system"], "two")
        self.assertEqual(len(urls), 1, "the gate would only add latency")
        self.assertIsNone(out["gate"]["p_deep"])

    def test_gate_below_threshold_answers_with_system_one(self) -> None:
        def transport(url, body, token, timeout):
            questions = body.get("questions") or {}
            if "gate" in questions:
                return {"answers": {"gate": {"type": "noul", "noul": 0.2}}}
            return choice_answer("A", 0.9)

        with env_key():
            out = sys12.route(cfg(), "Pick a label", questions=self._questions(),
                              transport=transport)
        self.assertEqual(out["system"], "one")
        self.assertFalse(out["gate"]["escalate"])
        self.assertFalse(out["gate"]["degraded"])
        self.assertEqual(out["result"]["answers"]["q"]["choice"], "A")
        self.assertFalse(out["escalate"])

    def test_gate_at_or_above_threshold_escalates_to_system_two(self) -> None:
        def transport(url, body, token, timeout):
            questions = body.get("questions") or {}
            if "gate" in questions:
                return {"answers": {"gate": {"type": "noul", "noul": 0.5}}}
            self.assertEqual(url, "https://cloud.example/v1/chat/completions")
            return chat_reply("deep answer")

        with env_key():
            out = sys12.route(cfg(), "Pick a label", questions=self._questions(),
                              transport=transport)
        self.assertEqual(out["system"], "two")
        self.assertTrue(out["gate"]["escalate"])
        self.assertEqual(out["result"]["content"], "deep answer")

    def test_custom_threshold_is_honoured(self) -> None:
        def transport(url, body, token, timeout):
            questions = body.get("questions") or {}
            if "gate" in questions:
                return {"answers": {"gate": {"type": "noul", "noul": 0.6}}}
            return choice_answer("B", 0.9)

        with env_key():
            out = sys12.route(cfg(), "task", questions=self._questions(),
                              threshold=0.95, transport=transport)
        self.assertEqual(out["system"], "one", "0.6 is below a 0.95 threshold")
        self.assertEqual(out["result"]["answers"]["q"]["choice"], "B")

    def test_system_one_answer_carries_its_escalate_to_the_route(self) -> None:
        def transport(url, body, token, timeout):
            questions = body.get("questions") or {}
            if "gate" in questions:
                return {"answers": {"gate": {"type": "noul", "noul": 0.1}}}
            return choice_answer("A", 0.2)

        with env_key():
            out = sys12.route(cfg(), "task", questions=self._questions(),
                              transport=transport)
        self.assertEqual(out["system"], "one")
        self.assertTrue(out["escalate"])


class StatusTests(unittest.TestCase):
    def test_unreachable_omlx_is_reported_not_raised(self) -> None:
        def getter(url, token, timeout):
            raise sys12.UpstreamError(None, "connection refused", True)

        report = sys12.status(cfg(), getter=getter)
        self.assertFalse(report["omlx"]["reachable"])
        self.assertIn("refused", report["omlx"]["error"])
        self.assertEqual(report["cloud"]["decision_model"], "clef")

    def test_loaded_models_are_summarized(self) -> None:
        def getter(url, token, timeout):
            return {"loaded_count": 1, "model_count": 3, "current_model_memory": 2e10,
                    "final_ceiling": 48e9,
                    "models": [{"model_alias": "Qwen Pi", "id": "x", "loaded": True,
                                "pinned": False, "is_default": True,
                                "estimated_size": 2e10}]}

        report = sys12.status(cfg(), getter=getter)
        self.assertTrue(report["omlx"]["reachable"])
        self.assertEqual(report["omlx"]["loaded_count"], 1)
        self.assertEqual(report["omlx"]["models"][0]["alias"], "Qwen Pi")
        self.assertEqual(report["omlx"]["current_memory_gb"], 20.0)


class KeyResolutionTests(unittest.TestCase):
    def test_environment_key_wins(self) -> None:
        with mock.patch.dict(os.environ, {"OLLAMA_API_KEY": "env-key"}):
            self.assertEqual(sys12.cloud_key(cfg()), "env-key")

    def test_keychain_is_used_when_the_environment_is_empty(self) -> None:
        done = mock.Mock(returncode=0, stdout="keychain-key\n")
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch("subprocess.run", return_value=done):
            self.assertEqual(sys12.cloud_key(cfg()), "keychain-key")

    def test_key_file_is_the_last_resort(self) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as handle:
            handle.write("OLLAMA_API_KEY=file-key\n")
            path = handle.name
        failed = mock.Mock(returncode=44, stdout="")
        try:
            with mock.patch.dict(os.environ, {}, clear=True), \
                    mock.patch("subprocess.run", return_value=failed):
                self.assertEqual(
                    sys12.cloud_key(cfg(cloud={"api_key_file": path})), "file-key"
                )
        finally:
            Path(path).unlink(missing_ok=True)

    def test_missing_key_names_every_option(self) -> None:
        failed = mock.Mock(returncode=44, stdout="")
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch("subprocess.run", return_value=failed):
            with self.assertRaises(sys12.Sys12Error) as caught:
                sys12.cloud_key(cfg(cloud={"keychain_service": "gone"}))
        message = str(caught.exception)
        self.assertIn("OLLAMA_API_KEY", message)
        self.assertIn("gone", message)


class RegressionTests(unittest.TestCase):
    """Each test pins a defect found in the audit, so it cannot come back."""

    def test_unparsed_fallback_answer_errors_instead_of_guessing(self) -> None:
        # Was: a missing key became noul=0.0, escalate=False — a confident "no"
        # the model never gave.
        def transport(url, body, token, timeout):
            if url.endswith("/v1/systemone"):
                raise sys12.UpstreamError(503, "down", True)
            return chat_reply('{"a": "A"}')

        questions = {"a": {"type": "choice", "criteria": {"A": "x", "B": "y"}},
                     "b": {"type": "noul"}}
        with env_key():
            with self.assertRaises(sys12.Sys12Error) as caught:
                sys12.decide(cfg(), "s", questions, transport=transport)
        self.assertIn("could not parse", str(caught.exception))
        self.assertIn("b", str(caught.exception))

    def test_null_noul_raises_a_typed_error_not_typeerror(self) -> None:
        def transport(url, body, token, timeout):
            return {"answers": {"q": {"type": "noul"}}}

        with env_key():
            with self.assertRaises(sys12.Sys12Error) as caught:
                sys12.decide(cfg(), "x", {"q": {"type": "noul"}},
                             transport=transport)
        self.assertNotIsInstance(caught.exception, TypeError)
        self.assertIn("non-numeric", str(caught.exception))

    def test_non_retryable_upstream_error_keeps_its_status(self) -> None:
        def transport(url, body, token, timeout):
            raise sys12.UpstreamError(401, "bad key", False)

        with env_key():
            with self.assertRaises(sys12.UpstreamError) as caught:
                sys12.decide(cfg(), "x",
                             {"q": {"type": "choice",
                                    "criteria": {"A": "x", "B": "y"}}},
                             transport=transport)
        self.assertEqual(caught.exception.status, 401)
        self.assertFalse(caught.exception.retryable)

    def test_bad_threshold_env_is_a_typed_error(self) -> None:
        with mock.patch.dict(os.environ, {"SYS12_ROUTE_THRESHOLD": "abc"}):
            with self.assertRaises(sys12.Sys12Error):
                sys12.load_config()

    def test_out_of_range_threshold_is_rejected(self) -> None:
        with mock.patch.dict(os.environ, {"SYS12_ROUTE_THRESHOLD": "2.5"}):
            with self.assertRaises(sys12.Sys12Error):
                sys12.load_config()

    def test_choice_options_may_share_a_description(self) -> None:
        # Two options with the same text are legitimate; only keys must differ.
        normalized = sys12.normalize_questions(
            {"q": {"type": "choice", "criteria": {"A": "yes", "B": "yes"}}}
        )
        self.assertEqual(len(normalized["q"]["criteria"]), 2)

    def test_duplicate_cli_criteria_are_rejected_not_silently_truncated(self) -> None:
        with self.assertRaises(sys12.Sys12Error):
            sys12.parse_question_spec("q:choice:a|a|b")

    def test_cli_spec_over_the_cap_is_rejected(self) -> None:
        spec = "q:choice:" + "|".join(f"o{i}" for i in range(300))
        with self.assertRaises(sys12.Sys12Error):
            sys12.parse_question_spec(spec)

    def test_empty_chat_content_is_flagged(self) -> None:
        def transport(url, body, token, timeout):
            return {"model": "m", "choices": [{"message": {
                "content": "", "reasoning": "thought a lot"}, "finish_reason": "length"}]}

        with env_key():
            out = sys12.ask(cfg(), "hi", transport=transport)
        self.assertIn("empty_reason", out)
        self.assertIn("max_tokens", out["empty_reason"])

    def test_malformed_chat_response_raises_shape_error(self) -> None:
        with env_key():
            with self.assertRaises(sys12.ShapeError):
                sys12.ask(cfg(), "hi", transport=lambda *a: {"nope": True})

    def test_mcp_errors_are_typed(self) -> None:
        import sys12_mcp

        failed = mock.Mock(returncode=44, stdout="")
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch("subprocess.run", return_value=failed):
            out = sys12_mcp.sys12_decide("s", {"q": {"type": "noul"}})
        self.assertEqual(out["error_kind"], "input")

    def test_emulated_parser_handles_real_model_reply_shapes(self) -> None:
        qs = {"pick": {"type": "choice",
                       "criteria": {"A": "canary", "B": "full rollout"}},
              "ok": {"type": "noul"},
              "sev": {"type": "score",
                      "criteria": ["low", "medium", "high"]}}
        cases = [
            ('{"pick": "B", "ok": true, "sev": 2}',
             ("B", 1.0, 0.5)),
            ('{"pick": "(B)", "ok": "yes", "sev": "3"}',
             ("B", 1.0, 1.0)),
            ("```json\n{\"pick\": \"full rollout\", \"ok\": false, \"sev\": 1}\n```",
             ("B", 0.0, 0.0)),
            ('{"pick": "2", "ok": "no", "sev": 3}',
             ("B", 0.0, 1.0)),
        ]
        for text, (letter, yes, score) in cases:
            parsed = sys12._parse_emulated(qs, text)
            self.assertEqual(parsed["pick"]["choice"], letter, text)
            self.assertEqual(parsed["ok"]["noul"], yes, text)
            self.assertEqual(parsed["sev"]["score"], score, text)

    def test_emulated_parser_omits_unanswered_questions(self) -> None:
        parsed = sys12._parse_emulated(
            {"a": {"type": "choice", "criteria": {"A": "x", "B": "y"}},
             "b": {"type": "noul"}},
            '{"a": "A"}',
        )
        self.assertIn("a", parsed)
        self.assertNotIn("b", parsed)

    def test_emulated_parser_ignores_garbage(self) -> None:
        for text in ("", "I cannot answer that.", '{"pick": "Z"}', '{"pick": null}'):
            parsed = sys12._parse_emulated(
                {"pick": {"type": "choice", "criteria": {"A": "x", "B": "y"}}}, text
            )
            self.assertEqual(parsed, {}, text)

    def test_browser_stop_letters_cannot_collide_with_elements(self) -> None:
        import browser

        letters = set(browser._CANDIDATE_LETTERS)
        self.assertNotIn(browser.STOP_GOAL_LETTER, letters)
        self.assertNotIn(browser.STOP_STUCK_LETTER, letters)
        # The old Y/Z choice collided once the cap passed 24.
        self.assertLess(browser.MAX_CANDIDATES, len(browser._CANDIDATE_LETTERS) + 1)

    def test_browser_bounds_max_steps(self) -> None:
        import browser

        with mock.patch.object(browser, "_ensure_page") as page:
            page.return_value.goto.side_effect = RuntimeError("stop here")
            with self.assertRaises(RuntimeError):
                browser.browse(cfg(), "https://example.com", "goal", max_steps=0)
        self.assertTrue(page.called, "max_steps=0 must still reach the page")


class McpSurfaceTests(unittest.TestCase):
    def test_server_exposes_five_prefixed_tools(self) -> None:
        import asyncio

        import sys12_mcp

        tools = asyncio.run(sys12_mcp.mcp.list_tools())
        names = sorted(tool.name for tool in tools)
        self.assertEqual(
            names,
            ["sys12_ask", "sys12_browse", "sys12_decide", "sys12_route", "sys12_status"],
        )
        for tool in tools:
            self.assertTrue(tool.description, f"{tool.name} needs a description")

    def test_decide_reports_an_error_instead_of_raising(self) -> None:
        import sys12_mcp

        failed = mock.Mock(returncode=44, stdout="")
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch("subprocess.run", return_value=failed):
            out = sys12_mcp.sys12_decide("state", {"q": {"type": "noul"}})
        self.assertIn("error", out)

    def test_browse_is_disabled_without_the_opt_in(self) -> None:
        import sys12_mcp

        with mock.patch.dict(os.environ, {}, clear=True):
            out = sys12_mcp.sys12_browse("https://example.com", "find a price")
        self.assertIn("SYS12_ALLOW_BROWSE", out["error"])


if __name__ == "__main__":
    unittest.main()
