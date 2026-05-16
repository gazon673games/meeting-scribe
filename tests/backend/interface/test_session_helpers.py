from __future__ import annotations

import unittest

from interface.session_controller_parts.helpers import safe_bool


class SessionHelperTests(unittest.TestCase):
    def test_safe_bool_parses_common_strings_and_defaults_unknown_values(self) -> None:
        self.assertTrue(safe_bool(True))
        self.assertTrue(safe_bool(" yes "))
        self.assertFalse(safe_bool(None))
        self.assertFalse(safe_bool("off"))
        self.assertTrue(safe_bool("maybe", default=True))


if __name__ == "__main__":
    unittest.main()
