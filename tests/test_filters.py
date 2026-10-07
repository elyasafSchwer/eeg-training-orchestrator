"""Unit tests for the metadata filtering that decides which visits a training job sees."""

import io
import json
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "orchestrator"))

import filters  # noqa: E402


class ParseFilter(unittest.TestCase):
    def test_every_operator(self):
        cases = [
            ("age>=20", ("age", ">=", 20)),
            ("age<=20", ("age", "<=", 20)),
            ("age>20", ("age", ">", 20)),
            ("age<20", ("age", "<", 20)),
            ("gender=female", ("gender", "=", "female")),
            ("gender!=female", ("gender", "!=", "female")),
        ]
        for expr, expected in cases:
            with self.subTest(expr=expr):
                parsed = filters.parse_filter(expr)
                self.assertEqual((parsed.key, parsed.op, parsed.value), expected)

    def test_two_character_operators_win(self):
        """'age>=20' must not parse as '>' with the value '=20'."""
        parsed = filters.parse_filter("age>=20")
        self.assertEqual((parsed.op, parsed.value), (">=", 20))

    def test_value_takes_the_type_the_metadata_uses(self):
        self.assertEqual(filters.parse_filter("age=20").value, 20)
        self.assertEqual(filters.parse_filter("age=20.5").value, 20.5)
        self.assertIs(filters.parse_filter("wears_glasses=true").value, True)
        self.assertIs(filters.parse_filter("wears_glasses=FALSE").value, False)
        # An ISO date must stay a string, not become a subtraction.
        self.assertEqual(filters.parse_filter("date_of_visit>=2024-06-01").value, "2024-06-01")

    def test_surrounding_whitespace_is_ignored(self):
        parsed = filters.parse_filter(" age > 20 ")
        self.assertEqual((parsed.key, parsed.op, parsed.value), ("age", ">", 20))

    def test_rejects_nonsense(self):
        for expr in ("nonsense", "", "=female", "age="):
            with self.subTest(expr=expr), self.assertRaises(ValueError):
                filters.parse_filter(expr)


class Matches(unittest.TestCase):
    METADATA = {"age": 74, "gender": "female", "wears_glasses": True, "date_of_visit": "2024-12-04"}

    def matches(self, *exprs):
        return filters.matches(self.METADATA, [filters.parse_filter(expr) for expr in exprs])

    def test_numbers(self):
        self.assertTrue(self.matches("age>70"))
        self.assertTrue(self.matches("age>=74"))
        self.assertFalse(self.matches("age>74"))
        self.assertTrue(self.matches("age<74.5"))  # int metadata against a float filter

    def test_strings_and_bools(self):
        self.assertTrue(self.matches("gender=female"))
        self.assertFalse(self.matches("gender!=female"))
        self.assertTrue(self.matches("wears_glasses=true"))
        self.assertFalse(self.matches("wears_glasses=false"))

    def test_iso_dates_compare_chronologically(self):
        self.assertTrue(self.matches("date_of_visit>=2024-01-01"))
        self.assertFalse(self.matches("date_of_visit>=2025-01-01"))

    def test_filters_are_anded(self):
        self.assertTrue(self.matches("gender=female", "age>70"))
        self.assertFalse(self.matches("gender=female", "age>80"))

    def test_missing_field_never_matches(self):
        """Some visits in the dataset carry no age or gender at all."""
        for expr in ("height>0", "height<0", "height=1", "height!=1"):
            with self.subTest(expr=expr):
                self.assertFalse(self.matches(expr))

    def test_no_filters_match_everything(self):
        self.assertTrue(filters.matches(self.METADATA, []))


class SelectVisits(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.data = Path(temporary.name)

    def write_visit(self, visit_id, eeg=True, **metadata):
        (self.data / f"{visit_id}.json").write_text(json.dumps({"visit_id": visit_id, **metadata}))
        if eeg:
            (self.data / f"{visit_id}.npy").touch()  # selection only checks that it exists

    def test_selects_matching_visits_with_their_metadata(self):
        self.write_visit("aaa", age=30)
        self.write_visit("bbb", age=70)
        selected = filters.select_visits(self.data, [filters.parse_filter("age>50")])
        self.assertEqual([visit.visit_id for visit in selected], ["bbb"])
        self.assertEqual(selected[0].eeg_path, self.data / "bbb.npy")
        self.assertEqual(selected[0].metadata["age"], 70)

    def test_skips_metadata_without_eeg_and_says_so(self):
        """The given dataset ships one such orphan."""
        self.write_visit("aaa", age=30)
        self.write_visit("orphan", eeg=False, age=30)
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            selected = filters.select_visits(self.data, [])
        self.assertEqual([visit.visit_id for visit in selected], ["aaa"])
        self.assertIn("orphan", stderr.getvalue())

    def test_ignores_eeg_without_metadata(self):
        """And one of these: data we cannot filter on is data we cannot select."""
        (self.data / "stray.npy").touch()
        self.write_visit("aaa", age=30)
        self.assertEqual([visit.visit_id for visit in filters.select_visits(self.data, [])], ["aaa"])

    def test_no_filters_selects_everything_in_a_stable_order(self):
        for visit_id in ("ccc", "aaa", "bbb"):
            self.write_visit(visit_id, age=30)
        selected = filters.select_visits(self.data, [])
        self.assertEqual([visit.visit_id for visit in selected], ["aaa", "bbb", "ccc"])


if __name__ == "__main__":
    unittest.main()
