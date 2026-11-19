"""Eulerian magnification: breathing motion and pulse colour come out bigger; nothing else changes."""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitor.magnify import Magnifier

H, W, FPS = 180, 320, 15.0
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)


def blob(cy):
    """Grey frame with a soft bright 'chest' whose centre is at row cy."""
    g = 100 + 80 * np.exp(-((XX - 160) ** 2 + (YY - cy) ** 2) / (2 * 22 ** 2))
    return np.repeat(g[..., None], 3, axis=2).astype(np.uint8)


def centroid_row(img):
    g = img[..., 0].astype(np.float32) - 100
    g[g < 0] = 0
    return float((g * YY).sum() / g.sum())


class MagnifyTests(unittest.TestCase):
    def test_breathing_motion_is_amplified(self):
        m = Magnifier()
        m.set_mode("breath")
        ins, outs = [], []
        for i in range(int(30 * FPS)):
            t = i / FPS
            cy = 90 + 0.6 * math.sin(2 * math.pi * 0.3 * t)        # 18 breaths/min, 0.6 px
            f = blob(cy)
            o = m.apply(t, f, box=(100, 40, 220, 140))
            if t > 15:
                ins.append(centroid_row(f))
                outs.append(centroid_row(o))
        gain = np.ptp(outs) / np.ptp(ins)
        self.assertGreater(gain, 5.0, f"motion gain {gain:.1f}")

    def test_pulse_colour_is_amplified_only_in_the_face(self):
        m = Magnifier()
        m.set_mode("pulse")
        face, out = [], []
        for i in range(int(20 * FPS)):
            t = i / FPS
            f = np.full((H, W, 3), 120, np.float32)
            f[40:120, 120:200, 2] += 0.8 * math.sin(2 * math.pi * 1.2 * t)   # red channel, 72 bpm
            f = np.clip(np.round(f), 0, 255).astype(np.uint8)
            o = m.apply(t, f, box=(120, 40, 200, 120))
            if t > 10:
                face.append(float(o[80, 160, 2]))
                out.append(float(o[170, 10, 2]))
        self.assertGreater(np.ptp(face), 20.0)       # ~1 grey level in -> tens out
        self.assertLessEqual(np.ptp(out), 1.0)       # far outside the face: unchanged

    def test_off_returns_the_frame(self):
        m = Magnifier()
        f = blob(90)
        self.assertIs(m.apply(0.0, f), f)
        m.set_mode("nonsense")
        self.assertIsNone(m.mode)

    def test_movement_fades_it_out(self):
        m = Magnifier()
        m.set_mode("breath")
        for i in range(int(10 * FPS)):
            m.apply(i / FPS, blob(90 + 3 * math.sin(i / 3)), still=True)
        for i in range(int(10 * FPS), int(13 * FPS)):
            f = blob(90 + 30 * math.sin(i / 3))
            o = m.apply(i / FPS, f, still=False)
        self.assertLess(np.abs(o.astype(int) - f.astype(int)).max(), 3)


if __name__ == "__main__":
    unittest.main()
