"""DSP checks: peak ceiling, gate timing, streaming continuity, inactive identity."""
from dataclasses import asdict
import unittest

import numpy as np

from audio_preprocessing import Preprocessor, Settings


class PreprocessingTests(unittest.TestCase):
    def test_disabled_preserves_samples_exactly(self):
        x = np.random.default_rng(4).normal(0,.2,5000).astype(np.float32)
        self.assertTrue(np.array_equal(Preprocessor(16000,Settings()).process(x),x))
        p = Preprocessor(16000,Settings(gate_open=-40,gate_close=-45,limiter_dbfs=-12))
        p.enabled = False
        self.assertTrue(np.array_equal(p.process(x),x))

    def test_limiter_ceiling_and_unlimited_low_signal(self):
        p = Preprocessor(48000,Settings(limiter_dbfs=-12))
        low = np.linspace(-.01,.01,960,dtype=np.float32)
        self.assertTrue(np.array_equal(p.process(low),low))
        x = np.random.default_rng(5).normal(0,.8,9600).astype(np.float32)
        x[101],x[3000] = 3.,-4.
        y = p.process(x)
        self.assertTrue(np.isfinite(y).all())
        self.assertLessEqual(float(np.max(np.abs(y))),10**(-12/20))

    def test_gate_quiet_speech_hold_and_release(self):
        p = Preprocessor(16000,Settings(gate_open=-40,gate_close=-45,
                                       gate_attack_ms=5,gate_hold_ms=150,gate_release_ms=100))
        noise = np.full(1600,.001,dtype=np.float32)
        self.assertEqual(float(np.max(np.abs(p.process(noise)))),0.)
        speech = (.1*np.sin(2*np.pi*250*np.arange(8000)/16000)).astype(np.float32)
        p.process(speech)
        self.assertTrue(p.gate_opened)
        self.assertGreater(p.gate_gain,.99)
        p.process(np.zeros(640,dtype=np.float32))  # 40 ms pause: do not chatter shut
        self.assertTrue(p.gate_opened)
        p.process(np.zeros(16000,dtype=np.float32))
        self.assertFalse(p.gate_opened)
        self.assertLess(p.gate_gain,.001)

    def test_chunk_boundaries_do_not_change_audio(self):
        x = np.random.default_rng(8).normal(0,.04,6000).astype(np.float32)
        x[1200:2400] = .0001
        s = Settings(gate_open=-35,gate_close=-40,limiter_dbfs=-24)
        expected = Preprocessor(16000,s).process(x)
        p = Preprocessor(16000,s)
        actual = np.concatenate([p.process(x[i:i+137]) for i in range(0,len(x),137)])
        self.assertTrue(np.array_equal(actual,expected))

    def test_reject_invalid_edit_without_changing_settings(self):
        p = Preprocessor(16000,Settings(gate_open=-40,gate_close=-45,limiter_dbfs=-6))
        before = asdict(p.settings)
        for command in ('gate -50 -40','gate nan','limit 5','limit -inf'):
            with self.assertRaises(ValueError):
                p.control(command)
            self.assertEqual(asdict(p.settings),before)
        p.control('gate -50 -55')
        self.assertEqual(p.settings.gate_close,-55)
        p.control('limit off')
        self.assertIsNone(p.settings.limiter_dbfs)


if __name__ == '__main__':
    unittest.main()
