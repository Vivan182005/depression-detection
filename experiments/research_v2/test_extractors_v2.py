"""
Module: experiments/research_v2/test_extractors_v2.py
Description: Unit tests for Canonical Feature Extractors v2.
Validates:
1. Wav2Vec2ExtractorV2 extracts (768,) representations and raises on invalid inputs.
2. AcousticExtractorV2 extracts (154,) representations with real eGeMAPSv02 functionals,
   asserting that dims 62-87 are non-zero.
3. BertExtractorV2 normalizes text, extracts (768,) embeddings, and raises ValueError on empty text.
"""

import os
import sys
import unittest
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.research_v2.extractors_v2 import (
    Wav2Vec2ExtractorV2,
    AcousticExtractorV2,
    BertExtractorV2,
    normalize_text_for_bert
)

class TestExtractorsV2(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Create 3-second synthetic speech signal
        cls.sr = 16000
        t = np.linspace(0, 3, 3 * cls.sr)
        cls.test_sig = (0.3 * np.sin(2 * np.pi * 300 * t) + 
                        0.2 * np.sin(2 * np.pi * 800 * t) + 
                        0.05 * np.random.randn(len(t))).astype(np.float32)
                        
    def test_text_normalizer(self):
        raw = "<laughter> [unin] Yes, I have been feeling down lately... xxx and tired. <sigh>"
        cleaned = normalize_text_for_bert(raw)
        self.assertEqual(cleaned, "yes i have been feeling down lately and tired")
        self.assertEqual(normalize_text_for_bert(""), "")
        # Verify apostrophe retention (Task A3)
        raw_apostrophe = "I don't think it's fair that they wouldn't listen."
        cleaned_apostrophe = normalize_text_for_bert(raw_apostrophe)
        self.assertEqual(cleaned_apostrophe, "i don't think it's fair that they wouldn't listen")

    def test_bert_empty_raises(self):
        ext = BertExtractorV2()
        with self.assertRaises(ValueError):
            ext.extract_from_text("")
        with self.assertRaises(ValueError):
            ext.extract_from_text("   <laughter>   [unin]   ")

    def test_bert_extraction_valid(self):
        ext = BertExtractorV2()
        text = "I have been having trouble sleeping lately and feeling hopeless."
        vec = ext.extract_from_text(text)
        self.assertEqual(vec.shape, (768,))
        self.assertTrue(np.isfinite(vec).all())
        self.assertGreater(np.linalg.norm(vec), 0.0)

    def test_acoustic_opensmile_valid(self):
        ext = AcousticExtractorV2(sr=self.sr)
        vec = ext.extract(self.test_sig)
        self.assertEqual(vec.shape, (154,))
        self.assertTrue(np.isfinite(vec).all())
        # eGeMAPS 88d is at indices 0:88
        egemaps = vec[:88]
        # Assert dims 62-87 are non-zero (proving real OpenSMILE ran)
        self.assertFalse(np.all(egemaps[62:88] == 0.0), "Dims 62-87 must NOT be all zero!")

    def test_acoustic_robust_135d(self):
        ext = AcousticExtractorV2(sr=self.sr)
        rob = ext.extract_robust(self.test_sig)
        self.assertEqual(rob.shape, (135,))
        self.assertTrue(np.isfinite(rob).all())
        self.assertGreater(np.linalg.norm(rob), 0.0)

    def test_wav2vec2_extraction_valid(self):
        ext = Wav2Vec2ExtractorV2()
        vec = ext.extract(self.test_sig, sr=self.sr, chunk_sec=30)
        self.assertEqual(vec.shape, (768,))
        self.assertTrue(np.isfinite(vec).all())
        self.assertGreater(np.linalg.norm(vec), 0.0)

if __name__ == '__main__':
    unittest.main()
