import unittest

import torch

from omniad_losses import local_hard_loss


class LocalLossTests(unittest.TestCase):
    def test_exact_reconstruction(self):
        x = torch.randn(2, 8, 4, 4)
        self.assertLess(local_hard_loss([x], [x]).item(), 1e-6)

    def test_teacher_detached_and_student_gradient(self):
        teacher = torch.randn(2, 8, 4, 4, requires_grad=True)
        student = torch.randn(2, 8, 4, 4, requires_grad=True)
        loss = local_hard_loss([teacher], [student])
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertGreater(student.grad.abs().sum().item(), 0)

    def test_mining_is_per_image(self):
        a, b = torch.randn(2, 8, 4, 4), torch.randn(2, 8, 4, 4)
        together = local_hard_loss([a], [b])
        separate = sum(local_hard_loss([a[i:i+1]], [b[i:i+1]]) for i in range(2)) / 2
        torch.testing.assert_close(together, separate)

    def test_single_pixel_and_invalid_fraction(self):
        a, b = torch.randn(1, 4, 1, 1), torch.randn(1, 4, 1, 1)
        self.assertTrue(torch.isfinite(local_hard_loss([a], [b], 0.001)))
        with self.assertRaises(ValueError):
            local_hard_loss([a], [b], 0)


if __name__ == "__main__":
    unittest.main()
