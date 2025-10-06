# CP Decomposition (Python CLI Tool)

A command-line program in Python that performs **Canonical Polyadic (CP) tensor decomposition** using the **Alternating Least Squares (ALS)** algorithm. It supports **HOSVD-based initialization**, **rank auto-detection**, normalization of factors, and reconstruction error evaluation.

---

## Features
- **ALS optimization** for CP decomposition  
- **HOSVD initialization** for faster and more stable convergence  
- **Automatic rank selection** with error-based stopping criteria  
- **Factor normalization** with scaling weights (lambdas)  
- **Tensor reconstruction** and relative error measurement  
- Works with **real or complex tensors** of arbitrary dimensions  
- Lightweight single-file CLI implementation

---

## Requirements
- Python 3.9+
- NumPy

---

## Build & Run
From the repository root:

```bash
python CPD2.py
```

You will be prompted to enter tensor dimensions and values through the CLI, or you can modify the script to load data programmatically.

---

## Project Structure

- `CPD2.py` — Main program (ALS-based CP decomposition)
- `README.md` — Project overview
- `LICENSE` — MIT License
- `.gitignore` — Ignore build artifacts

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
