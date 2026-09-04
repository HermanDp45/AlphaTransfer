#!/usr/bin/env python3
"""Контрактные тесты извлечения и дневной агрегации P2P-курсов."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("extract_p2p_rates.py")
SPEC = importlib.util.spec_from_file_location("extract_p2p_rates", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ExtractionTest(unittest.TestCase):
    # Проверяет нормализацию прямого курса RUB→KZT.
    def test_currency_equality(self) -> None:
        # Arrange
        text = "Меняю рубли на тенге: 1 RUB = 5,42 KZT, перевод на Kaspi"

        # Act
        quotes = MODULE.extract_rate_quotes(MODULE.normalize_text(text))

        # Assert
        self.assertIn((5.42, "currency_equality", None), quotes)
        self.assertEqual(MODULE.classify_segment(MODULE.normalize_text(text)), "card_transfer")
        self.assertEqual(MODULE.infer_direction(MODULE.normalize_text(text)), "rub_to_kzt")

    # Проверяет обращение котировки RUB за один KZT.
    def test_inverse_currency_equality(self) -> None:
        # Arrange
        text = "1 тенге = 0,2 рубля, обмен наличными"

        # Act
        quotes = MODULE.extract_rate_quotes(MODULE.normalize_text(text))

        # Assert
        self.assertIn((5.0, "inverse_currency_equality", None), quotes)
        self.assertEqual(MODULE.classify_segment(MODULE.normalize_text(text)), "cash")

    # Проверяет расчёт курса из двух явно подписанных сумм.
    def test_amount_ratio(self) -> None:
        # Arrange
        text = "Отдам 100 000 рублей, нужно 540 000 тенге, переводом"

        # Act
        rate = MODULE.extract_amount_ratio(MODULE.normalize_text(text))

        # Assert
        self.assertAlmostEqual(rate, 5.4)
        self.assertEqual(MODULE.infer_direction(MODULE.normalize_text(text)), "rub_to_kzt")

    # Проверяет, что криптовалюта имеет приоритет над наличным и карточным признаками.
    def test_crypto_segment_precedence(self) -> None:
        # Arrange
        text = MODULE.normalize_text("USDT, RUB и KZT; расчёт наличными или переводом на карту")

        # Act
        segment = MODULE.classify_segment(text)

        # Assert
        self.assertEqual(segment, "crypto")

    # Проверяет, что наличная нога не попадает в чистый карточный сегмент.
    def test_cash_precedence_over_card(self) -> None:
        # Arrange
        text = MODULE.normalize_text("Отдам наличные рубли, получу перевод тенге на Kaspi")

        # Act
        segment = MODULE.classify_segment(text)

        # Assert
        self.assertEqual(segment, "cash")

    # Проверяет короткий числовой ответ на вопрос о курсе.
    def test_bare_reply(self) -> None:
        # Arrange
        text = "5,35"

        # Act
        quotes = MODULE.extract_rate_quotes(text, parent_used=True)

        # Assert
        self.assertEqual(quotes, [(5.35, "bare_reply", None)])

    # Проверяет удаление контактов и длинных идентификаторов из review-фрагмента.
    def test_redaction(self) -> None:
        # Arrange
        text = "Пишите @dealer, +7 999 123-45-67, карта 1234567890123456, https://example.com"

        # Act
        redacted = MODULE.redact_text(text)

        # Assert
        self.assertNotIn("@dealer", redacted)
        self.assertNotIn("999", redacted)
        self.assertNotIn("1234567890123456", redacted)
        self.assertNotIn("example.com", redacted)

    # Проверяет, что forward fill не маскируется под новое наблюдение.
    def test_forward_fill_has_explicit_age(self) -> None:
        # Arrange
        observation = MODULE.Observation(
            source_file="result0.json",
            source_chat="test",
            message_ref="abc",
            participant_key="p1",
            timestamp="2026-01-01T10:00:00",
            day="2026-01-01",
            segment="card_transfer",
            direction="rub_to_kzt",
            rate_kzt_per_rub=5.4,
            extraction_method="explicit_rate",
            pair_basis="direct_message",
            confidence_score=0.9,
            confidence="high",
            quality_status="accepted",
            official_rate_kzt_per_rub=5.3,
            deviation_from_official_pct=5.4 / 5.3 - 1,
            evidence_excerpt="",
        )

        # Act
        rows = MODULE.aggregate_daily([observation], "2026-01-01", "2026-01-03", {})
        target = [
            row for row in rows
            if row["segment"] == "card_transfer" and row["direction"] == "rub_to_kzt"
        ]

        # Assert
        self.assertTrue(target[0]["is_observed"])
        self.assertEqual(target[1]["fill_method"], "forward_fill")
        self.assertFalse(target[1]["is_observed"])
        self.assertEqual(target[1]["days_since_observed"], 1)
        self.assertEqual(target[2]["effective_rate_kzt_per_rub"], 5.4)


if __name__ == "__main__":
    unittest.main()
