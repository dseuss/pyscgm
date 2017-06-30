### Which objects & operations do we need?

- `(Ψ, W)` ... sketch of X defined in (15)
	- `Ψ` ... sketcher, l*n matrix where l = 4r + 3 (or similar)
	- `W = ΨX`
	- reconstruction of X by Eq. (17)
	- needs to support updates of the form `X ← X + β uu^*`

- `A` ... the measurement matrix
	- since we usually need around `m ~ n * r` measurements to guarantee reconstruction, each row `A_i` of `A` can at most use `log(n)` memory
	- this forbids standard models such as Gaussian measurements, but is good for e.g. Pauli measurements or coded diffraction patterns
	- also needs to support efficient implementation of the following operations:
```
A(UV^*), U^*(A^* z), UA^* z) V^*
```
where U, V are `n * k` dimensional matrices (for k ≪ n) and `z` denotes an `m` dimensional vector

- an effficient largest singular vectors/eigen vectors algorithm of an `n*n` matrix `Z`
	- since we should not actually compute `Z`, we should use a random sketch of `Z`
	- in our application `Z = A^* z`, so the last two operations demanded above combined with the randomized SVD/EigVec algorithm from [2] will give us that

### Contracts

- SVD/Eig module

```python
class MatrixSketch[m, n]:
	"""Sketch of a m * n matrix `Z`"""
	def __lmul__(U: Matrix[k, m]) -> Matrix[k, n]:
		"""Compute U * Z"""
	def __rmul__(V: Matrix[n, k]) -> Matrix[m, k]:
		"""Compute Z * V"""

class HermitianMatrixSketch[n]:
	"""Sketch of a n * n hermitian matrix `Z`"""
	def __lmul__(U: Matrix[k, n]) -> Matrix[k, n]:
		"""Compute U * Z"""

def svd(Z: MatrixSketch[m ,n], k: int, tol: float)
	-> Matrix[m, k], Vector[k], Matrix[k, n]:
	"""Compute the k largest singular values and the
	   corresponding singular vectors of Z with tolerance
	   tol.
	"""

def eigh(Z: HermitianMatrixSketch[n], k: int, tol: float)
	-> Vector[k], Matrix[k, n]:
	"""Compute the k largest (in magnitude) eigenvalues and
	   corresponding eigenvectors of Z with tolerance tol.
	"""
```

- SketchyCGM module

```python
class CGMSketchH[n, r](HermitianMatrixSketch[n]):
	"""Sketch of hermitian, pos. semidef. matrix of rank < r
	using the techniques from [1]
	"""
	def cgm_update(u: Vector[n], eta: float) -> CGMSketchH:
		"""Performs the CGM update Z <- (1 - eta) Z + eta uu^*"""

class CGMMeasurements[m, n]:
	def __call__(sketch: CGMSketch) -> Vector[m]:
		"""Computes AX"""
    def adj(z: Vector[m]) -> HermitianMatrixSketch[n]:
    	"""Computes A^*(z)"""
```
