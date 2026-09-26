import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from diagnose_omniad_coverage import coverage_regions, regional_counts, audit_category


class CoverageAuditTests(unittest.TestCase):
    def test_regions_match_bilinear_support(self):
        canvas = torch.zeros(1, 1, 8, 8)
        canvas[:, :, 2:6, 2:6] = 1
        for shape in [(8, 8), (16, 16), (13, 29), (3, 4)]:
            regions = coverage_regions(shape, 8, (2, 2, 6, 6))
            mask = F.interpolate(canvas, size=shape, mode='bilinear', align_corners=False)[0, 0].numpy()
            np.testing.assert_array_equal(sum(r.astype(int) for r in regions.values()), 1)
            np.testing.assert_array_equal(regions['center'], mask >= 1-1e-6)
            np.testing.assert_array_equal(regions['border'], mask <= 1e-6)

    def test_regional_counts_add_to_global_counts(self):
        regions = coverage_regions((16, 16), 8, (2, 2, 6, 6))
        scores = np.random.default_rng(1).random((16, 16))
        truth = scores > .6
        result = regional_counts(scores, truth, regions, .4)
        self.assertEqual(sum(v['tp'] for v in result.values()), int(truth.sum()))
        self.assertEqual(sum(v['fp'] for v in result.values()), int(((scores >= .4) & ~truth).sum()))

    def test_unchanged_center_with_new_border_false_positives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image, mask = root / 'a.png', root / 'mask.png'
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image)
            truth = np.zeros((8, 8), dtype=np.uint8)
            truth[3, 3] = 255
            Image.fromarray(truth).save(mask)
            a = np.zeros((8, 8), dtype=np.float32)
            a[2:6, 2:6] = .2
            a[3, 3] = .8
            b = np.full_like(a, .9)
            b[2:6, 2:6] = a[2:6, 2:6]
            np.save(root/'base.npy', a)
            np.save(root/'new.npy', b)
            key = str(image)
            base = {key: ('sample', (image, root/'base.npy', mask))}
            candidate = {key: ('sample', (image, root/'new.npy', mask))}
            report = audit_category(base, candidate, [key], 8, (2, 2, 6, 6), root/'report', bins=64, top_k=1)
            self.assertEqual(report['center_check']['max_abs_change'], 0.)
            counts = report['region_counts']
            self.assertEqual(counts['baseline']['border']['fp'], 0)
            self.assertEqual(counts['candidate_at_baseline_threshold']['border']['fp'], 48)
            self.assertEqual(counts['candidate_at_baseline_threshold']['center']['fp'],
                             counts['baseline']['center']['fp'])
            self.assertTrue((root/'report/01_baseline.png').is_file())
            self.assertTrue((root/'report/01_candidate.png').is_file())
            self.assertTrue((root/'report/per_image.csv').is_file())


if __name__ == '__main__':
    unittest.main()
