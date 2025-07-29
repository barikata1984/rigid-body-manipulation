import unittest

import numpy as np


class TestRegression(unittest.TestCase):
    def setUp(self):
        """Set up a common regression problem for all test cases."""
        # Define true inertial parameters (phi_true)
        self.phi_true = np.array([1.5, 0.01, 0.02, -0.015, 0.01, 0.001, -0.002, 0.012, 0.003, 0.015])
        self.param_names = ["m", "mc_x", "mc_y", "mc_z", "I_xx", "I_xy", "I_xz", "I_yy", "I_yz", "I_zz"]

        # Simulation conditions
        num_samples = 200
        np.random.seed(0)

        # Generate ideal (noiseless) data
        A_true = np.random.randn(6 * num_samples, 10)
        w_true = A_true @ self.phi_true

        # Simulate measurement noise
        noise_level_A = 0.01
        noise_level_w = 0.01
        self.A_measured = A_true + np.random.normal(0, noise_level_A, A_true.shape)
        self.w_measured = w_true + np.random.normal(0, noise_level_w, w_true.shape)

    def test_ols_estimation(self):
        """Test parameter estimation using Ordinary Least Squares (OLS)."""
        phi_ols, _, _, _ = np.linalg.lstsq(self.A_measured, self.w_measured, rcond=None)
        error_ols = np.linalg.norm(self.phi_true - phi_ols)

        print("\n--- OLS Estimation --- ")
        print(f"OLS Estimation Error (Norm): {error_ols:.6f}")
        # This is a baseline test; we don't assert its accuracy vs. TLS here,
        # but we ensure it runs and produces a result.
        self.assertEqual(phi_ols.shape, (10,))

    def test_tls_estimation(self):
        """
        Test parameter estimation using Total Least Squares (TLS),
        which is suitable when the regressor matrix (A) is also noisy.
        """
        # 1. Create the augmented matrix C = [A_measured | w_measured]
        C_tls = np.hstack([self.A_measured, self.w_measured.reshape(-1, 1)])

        # 2. Perform Singular Value Decomposition (SVD)
        _, _, Vt = np.linalg.svd(C_tls)

        # 3. Extract the solution vector
        # The solution corresponds to the last row of Vt (the right singular vector
        # associated with the smallest singular value).
        v_min = Vt[-1, :]

        # 4. Normalize to get the parameter vector phi_tls
        # The vector v_min is proportional to [phi; -1].
        last_element = v_min[-1]
        phi_tls = -(1 / last_element) * v_min[:-1]
        error_tls = np.linalg.norm(self.phi_true - phi_tls)

        print("\n--- TLS Estimation --- ")
        print(f"TLS Estimation Error (Norm): {error_tls:.6f}")
        self.assertEqual(phi_tls.shape, (10,))

    def test_tls_is_more_accurate_than_ols_with_noisy_regressor(self):
        """
        Verify that TLS provides a more accurate estimate than OLS when both
        the observation vector (w) and the regressor matrix (A) are noisy.
        """
        # OLS estimation
        phi_ols, _, _, _ = np.linalg.lstsq(self.A_measured, self.w_measured, rcond=None)
        error_ols = np.linalg.norm(self.phi_true - phi_ols)

        # TLS estimation
        C_tls = np.hstack([self.A_measured, self.w_measured.reshape(-1, 1)])
        _, _, Vt = np.linalg.svd(C_tls)
        v_min = Vt[-1, :]
        phi_tls = -(1 / v_min[-1]) * v_min[:-1]
        error_tls = np.linalg.norm(self.phi_true - phi_tls)

        print("\n--- OLS vs. TLS Comparison --- ")
        print(f"OLS Error: {error_ols:.6f}, TLS Error: {error_tls:.6f}")
        self.assertLess(
            error_tls, error_ols, "TLS should be more accurate than OLS when the regressor matrix A is noisy."
        )


if __name__ == "__main__":
    unittest.main()
