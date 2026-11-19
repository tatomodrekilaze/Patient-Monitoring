"""Tamper-evident event log: any edit, insertion or deletion of a past line is found."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitor.alerts import EventLog, verify_chain


class AuditChainTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        log = EventLog(self.dir)
        for i in range(6):
            log.add("b1", "alert", alert="fall", severity="critical", n=i, msg="fall")
        self.path = log.path

    def lines(self):
        return open(self.path, encoding="utf-8").read().splitlines()

    def write(self, lines):
        open(self.path, "w", encoding="utf-8").write("\n".join(lines) + "\n")

    def test_intact(self):
        r = verify_chain(self.path)
        self.assertTrue(r["ok"])
        self.assertEqual(r["records"], 6)

    def test_edited_line(self):
        ls = self.lines()
        rec = json.loads(ls[2])
        rec["severity"] = "warning"                  # someone downgrades a past alarm
        ls[2] = json.dumps(rec, ensure_ascii=False)
        self.write(ls)
        r = verify_chain(self.path)
        self.assertFalse(r["ok"])
        self.assertEqual(r["first_bad_line"], 3)

    def test_deleted_line(self):
        ls = self.lines()
        del ls[3]
        self.write(ls)
        self.assertEqual(verify_chain(self.path)["first_bad_line"], 4)

    def test_chain_continues_after_restart(self):
        log = EventLog(self.dir)                     # reopened: continues from the last hash
        log.add("b1", "ack", alert_id=1)
        r = verify_chain(self.path)
        self.assertTrue(r["ok"])
        self.assertEqual(r["records"], 7)
        self.assertEqual(log.entries[-1]["kind"], "ack")
        self.assertNotIn("_h", log.entries[0])       # reports never see the hash fields


if __name__ == "__main__":
    unittest.main()
