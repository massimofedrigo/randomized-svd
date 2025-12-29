# randomized-svd: Fast Randomized SVD implemented in Python

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)

**randomized-svd** is a lightweight, high-performance Python library for computing the **Randomized Singular Value Decomposition (rSVD)**. 

It is designed to handle massive matrices efficiently by decomposing them into a smaller, random subspace before computing the SVD. This approach is significantly faster than deterministic methods (like LAPACK's `dgesdd`) while maintaining high numerical accuracy for low-rank approximations.

> **Original Research:** This library is the engineering implementation of the thesis *"A Randomized Algorithm for SVD Calculation"* (M. Fedrigo). You can read the full theoretical background in the [docs/thesis.pdf](./docs/thesis.pdf).

---

## 🚀 Key Features

* **Smart Dispatching:** Automatically selects the optimal algorithm strategy for "Tall-and-Skinny" ($m \ge n$) vs "Short-and-Fat" ($m < n$) matrices to minimize memory footprint.
* **Automatic Denoising:** Includes an implementation of the **Gavish-Donoho** method for optimal hard thresholding.
* **Production Ready:** Fully type-hinted, unit-tested, and packaged with modern standards (`pyproject.toml`).
* **Zero-Bloat:** Core dependency is just **NumPy**. Visualization and testing tools are optional.

---

## 🛠 Installation

To avoid conflicts with other projects or system packages, it is recommended to install this library within a **virtual environment**.

### 1. Create and Activate a Virtual Environment

Within your project root folder, run the following commands:

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate

```

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\activate

```

### 2. Install the Library

Once the environment is active (you should see `(venv)` in your terminal), choose your installation mode:

**For Users (Standard Usage):**
Install directly from PyPI:

```bash
pip install randomized-svd

```

**For Developers (Testing & Contributing):**
Clone the repository and install in *editable mode* to reflect code changes immediately.

```bash
git clone [https://github.com/massimofedrigo/randomized-svd.git](https://github.com/massimofedrigo/randomized-svd.git)
cd randomized-svd
pip install -e ".[dev]"

```

---

## ⚡ Quick Start

### 1. Basic Decomposition

Compute the approximated SVD of a generic matrix.

```python
import numpy as np
from randomized_svd import rsvd

# Generate a large random matrix (1000 x 500)
X = np.random.randn(1000, 500)

# Compute rSVD with target rank k=10
U, S, Vt = rsvd(X, t=10)

print(f"U shape: {U.shape}")   # (1000, 10)
print(f"S shape: {S.shape}")   # (10, 10)
print(f"Vt shape: {Vt.shape}") # (10, 500)

```

### 2. Automatic Noise Reduction (Denoising)

Use the Gavish-Donoho optimal threshold to remove white noise from a signal.

```python
import numpy as np
from randomized_svd import rsvd, optimal_threshold

# Create a synthetic noisy signal
X_true = np.random.randn(1000, 10) @ np.random.randn(10, 500)
X_noisy = X_true + 0.5 * np.random.randn(1000, 500)

# Calculate optimal rank based on noise level (gamma)
target_rank = optimal_threshold(m=1000, n=500, gamma=0.5)

# Clean the matrix using the optimal rank
U, S, Vt = rsvd(X_noisy, t=target_rank)
X_clean = U @ S @ Vt

```

---

## 🏗 Project Structure

The project follows a modern `src`-layout to prevent import errors and ensure clean packaging.

```text
randomized-svd/
├── .github/workflows/    # CI/CD pipelines
├── docs/                 # Thesis PDF and extra documentation
├── examples/             # Jupyter Notebooks (Demos & Benchmarks)
├── src/                  # Source code
│   └── randomized_svd/   # Package source
│       ├── __init__.py
│       ├── core.py       # Main rSVD logic (Facade & Implementations)
│       └── utils.py      # Math helpers (Gavish-Donoho threshold)
├── tests/                # Pytest suite
├── Dockerfile            # Reproducible testing environment
├── pyproject.toml        # Dependencies and metadata (replaces setup.py)
└── README.md

```

---

## 🐳 Docker Support

To ensure reproducibility across different machines and operating systems, we provide a **Dockerfile**.

> **Note:** Docker is primarily used here for running the **test suite** in an isolated, clean environment. For using the library in your own projects, the standard `pip install` (above) is recommended.

**Build the image:**

```bash
docker build -t randomized-svd-test .

```

**Run the test suite:**

```bash
docker run randomized-svd-test

```

---

## 📈 Performance

*Benchmarks run on an Intel i7, 16GB RAM.*

| Matrix Size | Method | Time (s) | Speedup |
| --- | --- | --- | --- |
| 5000 x 5000 | **rSVD (k=50)** | **0.82s** | **~12x** |
| 5000 x 5000 | NumPy SVD | 9.94s | - |

*See `examples/2_benchmark_performance.ipynb` for the full reproduction script.*

---

## 🧪 Testing

We use **pytest** for unit testing, covering:

1. **Invariance:** Output dimensions match mathematical expectations.
2. **Accuracy:** Reconstruction error on low-rank matrices is negligible.
3. **Orthogonality:**  and  matrices are verified to be orthogonal.

Run tests locally (requires dev installation):

```bash
pytest -v

```

---

## 📚 References

1. **Fedrigo, M.** (2024). *A Randomized Algorithm for SVD Calculation*. [PDF Available](./docs/thesis.pdf).
2. **Halko, N., Martinsson, P. G., & Tropp, J. A.** (2011). *Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions*. *SIAM review*.
3. **Gavish, M., & Donoho, D. L.** (2014). *The optimal hard threshold for singular values is *.
4. **Brunton, S. L., & Kutz, N. J.** (2019). *Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control*.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://mit-license.org/) file for details.

**Author:** Massimo Fedrigo

**Portfolio & Research:** [massimofedrigo.com](https://massimofedrigo.com)

**Contact:** contact@massimofedrigo.com