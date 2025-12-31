import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA
from randomized_svd import rpca


class TestRandomizedPCA:

    def test_compare_with_sklearn_dense(self) -> None:
        """
        Verify rpca produces same singular values as sklearn PCA (on dense data).
        """
        m, n = 100, 50
        np.random.seed(42)
        X = np.random.randn(m, n) + 10  # Add shift to ensure mean is not zero
        t = 5

        # 1. Run our rPCA
        # Fix seeds internally? rSVD uses np.random. 
        np.random.seed(99)
        U, S, Vt = rpca(X, t=t, p=2)  # X is pure noise, let us fix singular value decay

        # 2. Run Sklearn PCA
        # Sklearn PCA uses centered SVD internally. 
        # Note: Sklearn returns 'components_' which is Vt.
        # Singular values are stored in 'singular_values_'.
        pca_sk = PCA(n_components=t, svd_solver='randomized', random_state=99)
        pca_sk.fit(X)

        # Compare Singular Values (Variance explained)
        # Allow small tolerance due to randomized nature
        np.testing.assert_allclose(S, pca_sk.singular_values_, rtol=0.1)

    def test_sparse_matrix_support(self) -> None:
        """
        Verify rpca works on sparse matrices (Virtual Centering).
        If implementation was naive (X - mu), this would likely fail or be slow 
        on huge matrices, but here we check functional correctness on a small one.
        """
        m, n = 100, 50
        density = 0.1
        # Generate sparse matrix
        X_sparse = sp.rand(m, n, density=density, format='csr', random_state=42)

        # X is sparse, but mathematically it has a non-zero mean.
        # rpca should handle this without converting X to dense.
        t = 5
        U, S, Vt = rpca(X_sparse, t=t)

        assert S.shape == (t,)
        assert Vt.shape == (t, n)

    def test_virtual_centering_logic(self) -> None:
        """
        Internal test for the CenteredMatrix class logic.
        Checks if A_virtual @ v == (A_dense - mu) @ v
        """
        from randomized_svd.pca import CenteredMatrix

        m, n = 20, 10
        X = np.random.randn(m, n)
        mean = X.mean(axis=0)

        # 1. Create Virtual Wrapper
        X_virtual = CenteredMatrix(X, mean)

        # 2. Create Explicit Centered Matrix
        X_explicit = X - mean

        # Test Matrix-Vector multiplication
        v = np.random.randn(n)
        res_virtual = X_virtual @ v
        res_explicit = X_explicit @ v
        np.testing.assert_allclose(res_virtual, res_explicit, atol=1e-10)

        # Test Matrix-Matrix multiplication (Crucial for rSVD)
        V = np.random.randn(n, 4) # Block of 4 vectors
        res_virtual_mat = X_virtual @ V
        res_explicit_mat = X_explicit @ V
        np.testing.assert_allclose(res_virtual_mat, res_explicit_mat, atol=1e-10)
