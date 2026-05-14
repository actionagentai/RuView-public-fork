# Reference platforms for `expected_features.sha256`

The hash in `expected_features.sha256` was generated on a specific BLAS / FFT
backend. Numpy + scipy delegate FFT/linear-algebra to platform-native
libraries, and those libraries produce **bit-different output on identical
IEEE 754 inputs** depending on the backend. This is not a bug in the proof
pipeline — it is a property of the underlying numerical libraries. (See
issue #560.)

## Platforms where the hash is expected to MATCH

| Platform | BLAS backend | Status |
|---|---|---|
| `linux-x86_64-gnu` (Python 3.11.x, numpy 1.26.4 from PyPI wheels, scipy 1.14.1) | OpenBLAS | ✅ Reference |
| `windows-x86_64-msvc` (Python 3.11.x / 3.13.x, numpy 1.26.4 from PyPI wheels, scipy 1.14.1) | OpenBLAS | ✅ Reference |

## Platforms where the hash is **expected to MISMATCH**

| Platform | BLAS backend | Why |
|---|---|---|
| `darwin-arm64` (macOS arm64, Apple Silicon) | Accelerate.framework | FFT + matrix kernels differ in last-bit positions; the SHA-256 will differ even with pinned `numpy 1.26.4` / `scipy 1.14.1`. |
| Any environment with MKL installed | Intel MKL | Same root cause as Accelerate: different vectorized FFT path. |

## What to do if you get MISMATCH on a non-reference platform

The pipeline is still correct on your platform — the *output* is bit-different
because the *backend* is bit-different, not because the proof code has a bug.
Three workable responses:

1. **Run the proof on a reference platform** (Linux x86_64 or Windows x86_64
   with the PyPI OpenBLAS wheels). This is what CI does.

2. **Generate a new local-reference hash** for your platform and check it
   against the same hash on a teammate's machine with the *same* backend:

   ```bash
   # Regenerate from your platform
   python archive/v1/data/proof/verify.py --generate-hash

   # Commit the new hash to a side file (do NOT overwrite expected_features.sha256
   # unless you are publishing a new cross-platform reference)
   ```

3. **Compare numerical output, not the hash.** A relaxed-tolerance comparison
   on the feature vectors (e.g. `np.allclose(features, reference, atol=1e-10)`)
   will pass across backends. This is on the roadmap (see issue #560).

## The `verify.py` runtime environment block

Every run of `verify.py` now prints a `RUNTIME ENVIRONMENT` block before the
pipeline runs. Include that block in any issue report — it identifies the
platform + numpy version + BLAS backend in one place.
