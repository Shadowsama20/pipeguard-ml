import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pipeguard.cli import main


IMPORTS = """\
from sklearn.model_selection import train_test_split as split
from sklearn.preprocessing import StandardScaler as Scaler
"""


class CliTests(unittest.TestCase):
    def run_scan(self, name, source, *options):
        with TemporaryDirectory() as directory:
            path = Path(directory) / name
            path.write_text(source, encoding="utf-8")
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                status = main([*options, directory])
        return status, stdout.getvalue(), stderr.getvalue()

    def test_detects_pre_split_and_test_fit(self):
        status, output, errors = self.run_scan("training.py", IMPORTS + """\
X = df.fillna(df.median())
scaler = Scaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = split(X, y)
scaler.fit(X_test.copy())
""")
        self.assertEqual(status, 1)
        self.assertEqual(errors, "")
        self.assertIn("training.py:3:15: PG003", output)
        self.assertIn("training.py:5:5: PG001", output)
        self.assertIn("training.py:7:1: PG002", output)

    def test_clean_train_fit_and_unrelated_splits(self):
        status, output, errors = self.run_scan("training.py", IMPORTS + """\
Scaler().fit(unrelated)
X_train, X_test = split(X)
scaler = Scaler().fit(X_train)
X_test = scaler.transform(X_test)
""")
        self.assertEqual((status, output, errors), (0, "", ""))

    def test_notebook_state_and_physical_cell_number(self):
        source = json.dumps({"cells": [
            {"cell_type": "markdown", "source": "training notes"},
            {"cell_type": "code", "source": IMPORTS},
            {"cell_type": "code", "source": ["%matplotlib inline\n", "X = Scaler().fit_transform(X)"]},
            {"cell_type": "code", "source": "X_train, X_test = split(X)"},
        ]})
        status, output, errors = self.run_scan("training.ipynb", source)
        self.assertEqual(status, 1)
        self.assertIn("training.ipynb:cell 3:2:5: PG001", output)
        self.assertEqual(errors, "")

    def test_configured_index_split_follows_held_out_rows(self):
        source = IMPORTS + """\
from src.splitting import group_split as holdout
X = Scaler().fit_transform(X)
train, test = holdout(X, y, groups)
Scaler().fit(X.iloc[test])
"""
        status, output, errors = self.run_scan(
            "training.py", source, "--index-split", "src.splitting.group_split"
        )
        self.assertEqual(status, 1)
        self.assertIn("training.py:4:5: PG001", output)
        self.assertIn("training.py:6:1: PG002", output)
        self.assertEqual(errors, "")

    def test_configured_index_split_allows_train_only_fit(self):
        source = IMPORTS + """\
from src.splitting import group_split
train, test = group_split(X, y, groups)
scaler = Scaler().fit(X.iloc[train])
result = scaler.transform(X.iloc[test])
"""
        result = self.run_scan(
            "training.py", source, "--index-split", "src.splitting.group_split"
        )
        self.assertEqual(result, (0, "", ""))

    def test_fit_before_validation_split(self):
        result = self.run_scan("nested.py", IMPORTS + """
train, test = split(X)
scaled = Scaler().fit_transform(train)
inner_train, validation = split(scaled)
""")
        self.assertEqual(result[0], 1)
        self.assertIn("PG001", result[1])

    def test_rowwise_statistics(self):
        result = self.run_scan("rowwise.py", IMPORTS + """
X = df.sub(df.mean(axis=1), axis=0)
train, test = split(X)
""")
        self.assertEqual(result, (0, "", ""))

    def test_parse_failure(self):
        status, output, errors = self.run_scan("training.py", "if\n")
        self.assertEqual(status, 2)
        self.assertEqual(output, "")
        self.assertIn("training.py:1:", errors)


if __name__ == "__main__":
    unittest.main()
