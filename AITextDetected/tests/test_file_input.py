import io
import os
import sys
import unittest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.file_input import read_batch_csv, read_txt


class TestLegacyFileInput(unittest.TestCase):
    def test_read_txt(self):
        f = io.BytesIO("Hello world unicode café".encode("utf-8"))
        self.assertEqual(read_txt(f), "Hello world unicode café")

    def test_read_batch_csv_finds_text_column(self):
        csv_bytes = "id,text,label\n1,Hello there,0\n2,Another row,1\n".encode("utf-8")
        df = read_batch_csv(io.BytesIO(csv_bytes))
        self.assertEqual(list(df["text"]), ["Hello there", "Another row"])

    def test_read_batch_csv_drops_empty_rows(self):
        csv_bytes = "text\nHello\n\n   \nWorld\n".encode("utf-8")
        df = read_batch_csv(io.BytesIO(csv_bytes))
        self.assertEqual(len(df), 2)

    def test_read_batch_csv_missing_column_raises(self):
        csv_bytes = "id,content\n1,Hello\n".encode("utf-8")
        with self.assertRaises(ValueError):
            read_batch_csv(io.BytesIO(csv_bytes))


if __name__ == "__main__":
    unittest.main()
