import datetime
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import requests

import signal_generator as sg


class ReviewAndLearnSignalsTests(unittest.TestCase):
    def test_review_skips_rows_when_market_data_lookup_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            journal_path = tmp_path / "signal_journal.csv"
            learning_state_path = tmp_path / "learning_state.json"

            pd.DataFrame([
                {
                    "signal_date": str(datetime.date.today() - datetime.timedelta(days=1)),
                    "symbol": "BTCUSD",
                    "source": "binance",
                    "strategy": "momentum",
                    "direction": "BUY",
                    "raw_direction": "BUY",
                    "evaluated": False,
                }
            ]).to_csv(journal_path, index=False)
            learning_state_path.write_text(json.dumps({}), encoding="utf-8")

            with mock.patch.object(sg, "_get_next_bar_return", side_effect=requests.ConnectionError("boom")):
                summary = sg.review_and_learn_signals(
                    journal_path=str(journal_path),
                    learning_state_path=str(learning_state_path),
                )

            self.assertEqual(summary["reviewed"], 0)
            self.assertEqual(summary["right"], 0)
            self.assertEqual(summary["wrong"], 0)
            self.assertEqual(summary["pending"], 1)

            journal = pd.read_csv(journal_path)
            self.assertFalse(bool(journal.loc[0, "evaluated"]))


if __name__ == "__main__":
    unittest.main()
