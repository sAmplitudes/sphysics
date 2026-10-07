# coding: utf-8
'''
:Author: Godo Kurten
:Description: Invariant K-Matrix Formalism in the P-Vector approach [taken from Chung et al., Ann. Physik 4 (1995)], Created on Friday 10.04.26
'''

from __future__ import absolute_import, print_function, division, annotations

import tensorflow as tf
import numpy as np
from iminuit import Minuit

from ..math import _tfnp as tfnp
from ..math import _math as math
from ..pwa.barrierFactors._bfQuiggHippel import barrierFactor_QuiggHippleComplexContinuation
from ..kinematics._kinematics import twobodyBreakupmomentumSquared, twobodyPhasespaceComplexContinuation


class KMatrixFormalismParams:
	'''
	Static (s-independent) parameters of the K-matrix formalism.

	Stores pole parameters, decay/production couplings, and optional polynomial
	backgrounds. Call ``evaluate(s)`` or ``getKMatrixFormalism(s)`` to obtain a
	``KMatrixFormalism`` instance evaluated at a given invariant mass squared.

	The parametrization follows Chung et al., Ann. Physik 4 (1995).
	Couplings may be supplied either as direct decay/production couplings
	(``decayCouplings``, ``productionCouplings``) or derived from ``partialWidthsPoles``.
	'''

	def __init__(
		self,
		nChannels: int,
		nPoles: int,
		bareMassesPoles: np.ndarray,
		daughterParticlesProperties: np.ndarray,
		partialWidthsPoles: np.ndarray = None,
		decayCouplings: np.ndarray = None,
		productionCouplings: np.ndarray = None,
		productionVectorPoly: list|np.ndarray = None,
		kMatrixPoly: list|np.ndarray = None,
	):
		'''
		Parameters
		----------
		nChannels : int
		    Number of decay channels.
		nPoles : int
		    Number of resonance poles.
		bareMassesPoles : np.ndarray, shape (nPoles,)
		    Bare masses of the poles before dressing by channel coupling.
		daughterParticlesProperties : np.ndarray, shape (nChannels, 3)
		    Properties of the two decay daughters for each channel. Each row
		    contains [orbital angular momentum L, mass of daughter 1, mass of
		    daughter 2].
		partialWidthsPoles : np.ndarray, shape (nPoles, nChannels), optional
		    Partial widths of each pole in each channel. Mutually exclusive with
		    ``decayCouplings`` and ``productionCouplings``; when provided, decay
		    and production couplings are derived from these values.
		decayCouplings : np.ndarray, shape (nPoles, nChannels), optional
		    Decay couplings g_{ai} of pole a to channel i. Must be provided
		    together with ``productionCouplings`` when ``partialWidthsPoles`` is
		    absent.
		    If ``partialWidthsPoles`` is provided, decay couplings are set as:
		    g_{ai} = sqrt(m_a * sum_i Gamma_{ai} / rho_i(m_a^2)), where m_a is the
		    bare mass of pole a, Gamma_{ai} is the partial width of pole a in channel i,
		    and rho_i(m_a^2) is the phase space factor for channel i at s = m_a^2.
		productionCouplings : np.ndarray, shape (nPoles,), optional
		    Production couplings beta_a for each pole. Must be provided together
		    with ``decayCouplings`` when ``partialWidthsPoles`` is absent.
		    If ``partialWidthsPoles`` is provided, production couplings are set as:
		    beta_a = sqrt(m_a * sum_i Gamma_{ai} / rho_i(m_a^2)), where m_a is the
		    bare mass of pole a, Gamma_{ai} is the partial width of pole a in channel i,
		    and rho_i(m_a^2) is the phase space factor for channel i at s = m_a^2.
		productionVectorPoly : list or np.ndarray, optional
		    Polynomial coefficients for a smooth background term added to the
		    P-vector. Defaults to zero.
		kMatrixPoly : list or np.ndarray, optional
		    Polynomial coefficients for a smooth background term added to the
		    K-matrix. Defaults to zero.
		'''
		self.nChannels = nChannels
		self.nPoles = nPoles
		# Construct the particles that the resonance decays into for each channel
		# First check that the shape of the decay particle corresponds to (number of channels, number of decay particles = 2)
		if not daughterParticlesProperties.shape == (self.nChannels, 3):
			raise ValueError(f"Number of possible decay channels must be equal to {self.nChannels} but {daughterParticlesProperties.shape[0]} were given! ")
		self.daughterParticlesProperties = daughterParticlesProperties

		if not bareMassesPoles.shape[0] == self.nPoles:
			raise ValueError("Number of bare masses for the poles must be equal to the number of poles.")
		self.bareMassesPoles = bareMassesPoles
		# Construct decay couplings and production couplings
		if partialWidthsPoles is not None:
			if decayCouplings is not None or productionCouplings is not None:
				raise ValueError("Provide either 'partialWidthsPoles' or both 'decayCouplings' and 'productionCouplings' but not both!")
			if not partialWidthsPoles.shape == (self.nPoles, self.nChannels):
				raise ValueError(f"Shape of partial widths for the poles must be equal to ({self.nPoles}, {self.nChannels}) but {partialWidthsPoles.shape} was provided.")
			# Construct the width of the poles as the sum of the partial widths for each channel
			phaseSpaceFactorsPoles = []
			for daughterChannel in self.daughterParticlesProperties:
				phaseSpaceFactorsPoles.append(twobodyPhasespaceComplexContinuation(self.bareMassesPoles**2, daughterChannel[1], daughterChannel[2]))
			phaseSpaceFactorsPoles = np.array(phaseSpaceFactorsPoles).T
			bareWidthsPoles = tfnp.sum(partialWidthsPoles / phaseSpaceFactorsPoles, axis=1)
			# Construct decay couplings and production couplings from the bare widths of the poles
			self.decayCouplings = tfnp.castToComplex(np.ones((self.nPoles, self.nChannels))) * math.sqrt(self.bareMassesPoles * bareWidthsPoles)[:, np.newaxis]
			self.productionCouplings = tfnp.castToComplex(np.ones(self.nPoles)) * math.sqrt(self.bareMassesPoles * bareWidthsPoles)

		else:
			if decayCouplings is None or productionCouplings is None:
				raise ValueError("Provide either 'partialWidthsPoles' or both 'decayCouplings' and 'productionCouplings' but not both!")
			if not decayCouplings.shape == (self.nPoles, self.nChannels):
				raise ValueError(f"Shape of decay couplings must be {(self.nPoles, self.nChannels)} but {decayCouplings.shape} was given!")
			if not productionCouplings.shape == (self.nPoles,):
				raise ValueError(f"Shape of production couplings must be {(self.nPoles)} but {productionCouplings.shape} was given!")
			if not productionCouplings.dtype in [np.complex128, np.complex64, tf.complex64, tf.complex128, complex]:
				raise ValueError(f"Production couplings must be complex but dtype {productionCouplings.dtype} was given!")
			if not decayCouplings.dtype in [np.float64, np.float32, tf.float64, tf.float32, float]:
				raise ValueError(f"Decay couplings must be float but dtype {decayCouplings.dtype} was given!")
			self.decayCouplings = tfnp.complex(decayCouplings, tfnp.zeros_like(decayCouplings))
			self.productionCouplings = productionCouplings


		# Construct polynomials for the production vector and the K-matrix if given
		if productionVectorPoly is not None:
			self.productionVectorPoly = tfnp.Polynomial(productionVectorPoly, castComplex=True)
		else:
			self.productionVectorPoly = tfnp.Polynomial([0.], castComplex=True)
		if kMatrixPoly is not None:
			self.kMatrixPoly = tfnp.Polynomial(kMatrixPoly, castComplex=True)
		else:
			self.kMatrixPoly = tfnp.Polynomial([0.], castComplex=True)

	def evaluate(self, s: tf.Tensor|np.ndarray, channelNum: int, riemannSheet: list = None) -> np.ndarray|tf.Tensor:
		'''
		Evaluate the scattering amplitude F-vector for a single channel at s.

		Parameters
		----------
		s : tf.Tensor or np.ndarray
		    Invariant mass squared values.
		channelNum : int
		    Channel number (1-based).
		riemannSheet : list of int, optional
		    Sign (+1 or -1) per channel for analytic continuation of the phase
		    space factor. Defaults to all +1 (physical sheet).

		Returns
		-------
		tf.Tensor or np.ndarray
		    F-vector element for ``channelNum``, shape (len(s),).
		'''
		kMatrixForm = KMatrixFormalism(self, s, riemannSheet=riemannSheet)
		return kMatrixForm(channelNum)

	def getAbsDetOneMatrixMinusJKRho(self, sReal: tf.Tensor|np.ndarray, sImag: tf.Tensor|np.ndarray, riemannSheet: list = None) -> np.ndarray|tf.Tensor:
		'''
		Compute ``|det(1 - i K rho)|^2`` at a complex point s = sReal + i*sImag.

		Used to locate T-matrix poles as zeros of the determinant on a chosen
		Riemann sheet.

		Parameters
		----------
		sReal : tf.Tensor or np.ndarray
		    Real part of the invariant mass squared.
		sImag : tf.Tensor or np.ndarray
		    Imaginary part of the invariant mass squared.
		riemannSheet : list of int, optional
		    Sign per channel for analytic continuation. Defaults to all +1.

		Returns
		-------
		tf.Tensor or np.ndarray
		    ``|det(1 - i K rho)|^2``, real and non-negative.
		'''
		kMatrixForm = KMatrixFormalism(self, sReal + 1j*sImag, riemannSheet=riemannSheet)
		return tfnp.abs(tfnp.det(tfnp.swapaxes(kMatrixForm.oneMatrixMinusJKRho, -1, 0)))**2

	def getFitFncForPoleDet(self, riemannSheet: list = None):
		'''
		Return a callable suitable for passing to ``iminuit.Minuit`` for pole finding.

		The returned function evaluates ``|det(1 - i K rho)|^2`` at complex s, which
		is zero at a T-matrix pole.

		Parameters
		----------
		riemannSheet : list of int, optional
		    Sign per channel for analytic continuation.

		Returns
		-------
		callable
		    Function ``f(sReal, sImag)`` returning ``|det(1 - i K rho)|^2``.
		'''
		def fitFnc(sReal: tf.Tensor|np.ndarray, sImag: tf.Tensor|np.ndarray):
			return  self.getAbsDetOneMatrixMinusJKRho(sReal, sImag, riemannSheet=riemannSheet)
		return fitFnc

	def getKMatrixFormalism(self, s: tf.Tensor|np.ndarray, riemannSheet: list = None) -> 'KMatrixFormalism':
		'''
		Construct and return a ``KMatrixFormalism`` instance evaluated at s.

		Parameters
		----------
		s : tf.Tensor or np.ndarray
		    Invariant mass squared values.
		riemannSheet : list of int, optional
		    Sign per channel for analytic continuation. Defaults to all +1.

		Returns
		-------
		KMatrixFormalism
		'''
		return KMatrixFormalism(self, s, riemannSheet=riemannSheet)

	def findPoles(
		self, realRange: tuple, imagRange: tuple, riemannSheet: list = None,
		limReal: tuple = None, limImag: tuple = None,
		numFitAttempts: int = 10, tol: float = 1e-5, atol: float = 1e-5,
	):
		'''
		Locate a T-matrix pole on the specified Riemann sheet using Minuit.

		Runs ``numFitAttempts`` independent minimizations of ``|det(1 - i K rho)|^2``
		from random starting points within ``realRange`` x ``imagRange``.
		Convergence requires the two best results to agree within ``atol``.

		Parameters
		----------
		realRange : tuple of float
		    (min, max) range for the random initial guess of Re(s).
		imagRange : tuple of float
		    (min, max) range for the random initial guess of Im(s).
		riemannSheet : list of int, optional
		    Sign (+1 or -1) per channel selecting the Riemann sheet. Defaults
		    to all +1 (physical sheet, no poles expected there).
		limReal : tuple of float, optional
		    Hard bounds (min, max) on Re(s) during minimization.
		limImag : tuple of float, optional
		    Hard bounds (min, max) on Im(s) during minimization.
		numFitAttempts : int, optional
		    Number of independent minimizations. Default 10.
		tol : float, optional
		    Step size tolerance passed to Minuit. Default 1e-5.
		atol : float, optional
		    Absolute tolerance for the agreement check between the two best
		    results. Default 1e-5.

		Returns
		-------
		dict
		    Keys: 'sReal', 'sImag', 'fval', 'cov', 'valid', 'accurate'.

		Raises
		------
		RuntimeError
		    If no two attempts agree within ``atol``, or if the best result is
		    not valid or not accurate according to Minuit.
		'''
		resultList = []
		for _ in range(numFitAttempts):
			initialGuessReal = np.random.uniform(*realRange)
			initialGuessImag = np.random.uniform(*imagRange)
			fitFnc = self.getFitFncForPoleDet(riemannSheet=riemannSheet)
			m = Minuit(fitFnc, sReal=initialGuessReal, sImag=initialGuessImag)
			if limReal is not None:
				m.limits['sReal'] = limReal
			if limImag is not None:
				m.limits['sImag'] = limImag
			m.errors['sReal'] = tol
			m.errors['sImag'] = tol
			m.tol = tol
			m.migrad(iterate=30)
			resultList.append({'fval': m.fval, 'sReal': m.values['sReal'], 'sImag': m.values['sImag'], 'cov': m.covariance, 'valid': m.valid, 'accurate': m.accurate})
		resultsSorted = sorted(resultList, key=lambda resultList: resultList['fval'])
		# Check that at least twice the same result was obtained
		if not np.isclose(resultsSorted[0]['fval'], resultsSorted[1]['fval'], atol=atol):
			raise RuntimeError("Pole search did not converge! Function value of two best results do not match.")
		if not np.isclose(resultsSorted[0]['sReal'], resultsSorted[1]['sReal'], atol=atol):
			raise RuntimeError("Pole search did not converge! Real part of the two best results do not match within tolerance.")
		if not np.isclose(resultsSorted[0]['sImag'], resultsSorted[1]['sImag'], atol=atol):
			raise RuntimeError("Pole search did not converge! Imaginary part of the two best results do not match within tolerance.")
		if not resultsSorted[0]['valid']:
			raise RuntimeError("Pole search did not converge! The best fit result was not valid.")
		if not resultsSorted[0]['accurate']:
			raise RuntimeError("Pole search did not converge! The best fit result was not accurate.")
		return resultsSorted[0]


class KMatrixFormalism:
	'''
	K-matrix formalism evaluated at a specific invariant mass squared s.

	Implements the P-vector parametrization for n-channel production amplitudes
	following Chung et al., Ann. Physik 4 (1995). All s-dependent quantities
	(K-matrix, P-vector, F-vector, etc.) are computed lazily and cached.

	Obtain instances via ``KMatrixFormalismParams.evaluate(s)`` or
	``KMatrixFormalismParams.getKMatrixFormalism(s)``.
	'''

	def __init__(self, params: KMatrixFormalismParams, s: tf.Tensor|np.ndarray, riemannSheet: list = None):
		'''
		Parameters
		----------
		params : KMatrixFormalismParams
		    Static (s-independent) parameters of the K-matrix formalism.
		s : tf.Tensor or np.ndarray
		    Invariant mass squared at which to evaluate the formalism. A scalar
		    is promoted to a length-1 array.
		riemannSheet : list of int, optional
		    Sign (+1 or -1) for each channel, selecting the branch of the square
		    root in the two-body phase space factor. Used for analytic continuation
		    into the complex s-plane to access unphysical Riemann sheets where
		    resonance poles reside. Defaults to all +1 (physical sheet).
		'''
		if not isinstance(s, (np.ndarray, tf.Tensor)):
			s = np.array([s])
		self.s = s
		self.params = params

		# Unpack static parameters so all existing build methods work unchanged
		self.nChannels = params.nChannels
		self.nPoles = params.nPoles
		self.bareMassesPoles = params.bareMassesPoles
		self.daughterParticlesProperties = params.daughterParticlesProperties
		self.decayCouplings = params.decayCouplings
		self.productionCouplings = params.productionCouplings
		self.productionVectorPoly = params.productionVectorPoly
		self.kMatrixPoly = params.kMatrixPoly

		if riemannSheet is None:
			self.riemannSheet = tfnp.eye(self.nChannels, self.s, dtype=self.s.dtype)
		else:
			if not len(riemannSheet) == self.nChannels:
				raise ValueError(f"Riemann sheet must be specified for each channel! Number of channels is {self.nChannels} but {len(riemannSheet)} were given!")
			self.riemannSheet = tfnp.diag(riemannSheet, refTensor=self.s)
		self.riemannSheet = tfnp.complex(self.riemannSheet, tfnp.zeros_like(self.riemannSheet))

		# Construct the phase space factors for each channel (depends on s, computed eagerly)
		phaseSpaceFactors = []
		for daughterChannel in self.daughterParticlesProperties:
			phaseSpaceFactors.append(twobodyPhasespaceComplexContinuation(self.s, daughterChannel[1], daughterChannel[2]))
		self._phaseSpaceMatrix = tfnp.transpose(self.riemannSheet * tfnp.stack(phaseSpaceFactors, axis=1)[:,:,None], [1, 2, 0])

		# Lazy caches for s-dependent quantities
		self._kMatrix = None
		self._centrifugalBarrierFactorsRunning = None
		self._ratioCentrifugalBarrierFactors = None
		self._pVector = None
		self._oneMatrixMinusJKRho = None
		self._invOneMatrixMinusJKRho = None
		self._detInvOfOneMatrixMinusJKRho = None
		self._fVector = None

	def __call__(self, channelNum: int):
		'''
		Return the F-vector element for the given channel at all values of s.

		Parameters
		----------
		channelNum : int
		    Channel number using 1-based indexing.

		Returns
		-------
		tf.Tensor or np.ndarray
		    F-vector for ``channelNum``, shape (len(s),).
		'''
		if channelNum <= 0:
			raise ValueError("Channel number must be larger than 0! We use 1-based counting for the channels.")
		if channelNum > self.nChannels:
			raise ValueError(f"Channel number must be smaller or equal to the number of channels {self.nChannels} but {channelNum} was given!")
		return self.fVector[channelNum-1, :]

	@property
	def phaseSpaceMatrix(self):
		'''
		Diagonal phase space matrix rho(s), shape (nChannels, nChannels, len(s)).

		Each diagonal entry is the Lorentz-invariant two-body phase space factor
		rho_i(s) for channel i, multiplied by the Riemann sheet sign. Off-diagonal
		entries are zero.
		'''
		return self._phaseSpaceMatrix

	@property
	def centrifugalBarrierFactorRunning(self):
		'''
		Centrifugal barrier factors F^l_i(q(s)) at the running mass sqrt(s),
		shape (nChannels, len(s)).
		'''
		if self._centrifugalBarrierFactorsRunning is None:
			self._buildRatiosCentrifugalBarrierFactors()
		return self._centrifugalBarrierFactorsRunning

	@property
	def ratioCentrifugalBarrierFactors(self):
		'''
		Ratio F^l_i(q(s)) / F^l_i(q(m_a)) of running to on-shell barrier factors,
		shape (nPoles, nChannels, len(s)).

		Encodes the threshold behaviour of each channel relative to each pole
		mass m_a, as required by the K-matrix parametrization.
		'''
		if self._ratioCentrifugalBarrierFactors is None:
			self._buildRatiosCentrifugalBarrierFactors()
		return self._ratioCentrifugalBarrierFactors

	@property
	def kMatrix(self):
		'''
		K-matrix K_{ij}(s), shape (nChannels, nChannels, len(s)).

		K_{ij}(s) = sum_a g_{ai} g_{aj} [F^l_i/F^l_i^a][F^l_j/F^l_j^a] / (m_a^2 - s) + poly(s),
		where g_{ai} are decay couplings and m_a are bare pole masses.

		See Eq. (74) in Chung et al., Ann. Physik 4 (1995).
		'''
		if self._kMatrix is None:
			self._buildKMatrix()
		return self._kMatrix

	@property
	def pVector(self):
		'''
		Production P-vector P_i(s), shape (nChannels, len(s)).

		P_i(s) = sum_a beta_a g_{ai} [F^l_i/F^l_i^a] / (m_a^2 - s) + poly(s),
		where beta_a are the production couplings.

		See Eq. (119) in Chung et al., Ann. Physik 4 (1995).
		'''
		if self._pVector is None:
			self._buildPVector()
		return self._pVector

	@property
	def oneMatrixMinusJKRho(self):
		'''
		Denominator matrix (1 - i K rho)(s), shape (nChannels, nChannels, len(s)).

		Its inverse maps the P-vector to the F-vector. Its determinant vanishes
		at T-matrix poles.
		'''
		if self._oneMatrixMinusJKRho is None:
			self._buildIdentityMinusJKRhoMatrix()
		return self._oneMatrixMinusJKRho

	@property
	def invOneMatrixMinusJKRho(self):
		'''
		Inverse of (1 - i K rho)(s), shape (nChannels, nChannels, len(s)).
		'''
		if self._invOneMatrixMinusJKRho is None:
			self._buildInvOfOneMatrixMinusJKRho()
		return self._invOneMatrixMinusJKRho

	@property
	def detInvOneMatrixMinusJKRho(self):
		'''
		Determinant of inverse of (1 - i K rho)(s), shape (nChannels, len(s)).
		'''
		if self._detInvOfOneMatrixMinusJKRho is None:
			self._buildDetInvOfOneMatrixMinusJKRho()
		return self._detInvOfOneMatrixMinusJKRho

	@property
	def fVector(self):
		'''
		Scattering amplitude F-vector F_i(s), shape (nChannels, len(s)).

		F = (1 - i K rho)^{-1} P / F^l_i(s), where dividing by the running centrifugal
		factor F^l_i(s) removes the centrifugal factor included in the K-matrix
		parametrization and yields the bare amplitude.

		See Eq. (115) in Chung et al., Ann. Physik 4 (1995).
		'''
		if self._fVector is None:
			self._buildFVector()
		return self._fVector

	def _buildRatiosCentrifugalBarrierFactors(self):
		'''Compute and cache ``centrifugalBarrierFactorsRunning`` and ``ratioCentrifugalBarrierFactors``.'''
		# Calculate the running breakup momenta for each channel and each mass
		# Create an extended array of the decay particles masses for each invariant mass
		extendedDecayParticlesMass = tfnp.stack([self.daughterParticlesProperties[:,1:]] * self.s.shape[0])
		# Calculate the squared two Body Breakup Momenta
		twoBodyBreakupMomentaSquaredRunning = twobodyBreakupmomentumSquared(self.s, extendedDecayParticlesMass[:,:,0].T, extendedDecayParticlesMass[:,:,1].T)

		# Do the same for the Resonance Breakup Momenta for each pole and channel
		extendedDecayParticlesMass = tfnp.stack([self.daughterParticlesProperties[:,1:]] * self.nPoles)
		twoBodyBreakupMomentaSquaredResonances = twobodyBreakupmomentumSquared(self.bareMassesPoles**2, extendedDecayParticlesMass[:,:,0].T, extendedDecayParticlesMass[:,:,1].T)

		# Calculate the centrifugal barrier factors for the running and resonance breakup momenta
		runningList = []
		resonancesList = []
		for channelNumber in range(self.nChannels):
			runningList.append(barrierFactor_QuiggHippleComplexContinuation(self.daughterParticlesProperties[channelNumber,0], twoBodyBreakupMomentaSquaredRunning[channelNumber, :]))
			resonancesList.append(barrierFactor_QuiggHippleComplexContinuation(self.daughterParticlesProperties[channelNumber,0], twoBodyBreakupMomentaSquaredResonances[channelNumber, :]))
		centrifugalBarrierFactorsRunning = tfnp.stack(runningList)
		self._centrifugalBarrierFactorsRunning = centrifugalBarrierFactorsRunning
		centrifugalBarrierFactorsResonances = tfnp.stack(resonancesList)

		# Build ratio of the running and resonance centrifugal barrier factors
		self._ratioCentrifugalBarrierFactors = centrifugalBarrierFactorsRunning[np.newaxis, :, :] / tfnp.transpose(centrifugalBarrierFactorsResonances, axes=(1,0))[:, :, np.newaxis]
		if not self._ratioCentrifugalBarrierFactors.dtype in [np.complex64, np.complex128, tf.complex64, tf.complex128, complex]:
			self._ratioCentrifugalBarrierFactors = tfnp.complex(self._ratioCentrifugalBarrierFactors, np.array([0], dtype=self.s.dtype))

	def _buildKMatrix(self):
		'''Compute and cache the K-matrix.'''
		denominator = tfnp.complex((self.bareMassesPoles**2)[:, np.newaxis] - self.s[np.newaxis, :],
							 tfnp.zeros_like((self.bareMassesPoles**2)[:, np.newaxis] - self.s[np.newaxis, :]))
		self._kMatrix = tfnp.einsum(
			'ai, aj, aim, ajm, am -> ijm',
			self.decayCouplings, self.decayCouplings,
			self.ratioCentrifugalBarrierFactors, self.ratioCentrifugalBarrierFactors,
			1/denominator,
		)
		self._kMatrix += self.kMatrixPoly(self.s)

	def _buildPVector(self):
		'''Compute and cache the P-vector.'''
		denominator = tfnp.complex((self.bareMassesPoles**2)[:, np.newaxis] - self.s[np.newaxis, :],
							 tfnp.zeros_like((self.bareMassesPoles**2)[:, np.newaxis] - self.s[np.newaxis, :]))
		self._pVector = tfnp.einsum('a, ai, aim, am -> im', self.productionCouplings, self.decayCouplings, self.ratioCentrifugalBarrierFactors, 1/denominator)
		self._pVector += self.productionVectorPoly(self.s)

	def _buildIdentityMinusJKRhoMatrix(self):
		'''Compute and cache the denominator matrix (1 - i K rho).'''
		#identityMatrices = tfnp.castToComplex(tfnp.eye(self.nChannels, self.s)[:,:,None])
		identityMatrices = tfnp.complex(tfnp.eye(self.nChannels, self.s, dtype=self.s.dtype)[:,:,None],
								  tfnp.zeros_like(tfnp.eye(self.nChannels, self.s, dtype=self.s.dtype)[:,:,None]))
		self._oneMatrixMinusJKRho = identityMatrices - 1j*tfnp.einsum('ikm, kjm -> ijm', self.kMatrix, self.phaseSpaceMatrix)

	def _buildInvOfOneMatrixMinusJKRho(self):
		'''Compute and cache the matrix inverse of (1 - i K rho) at each s.'''
		self._invOneMatrixMinusJKRho = tfnp.swapaxes(tfnp.invert(tfnp.swapaxes(self.oneMatrixMinusJKRho, -1, 0)), 0, -1)

	def _buildDetInvOfOneMatrixMinusJKRho(self):
		'''Compute and cache the determinant of (1-iKrho)^{-1} at each s.'''
		self._detInvOfOneMatrixMinusJKRho = tfnp.det(tfnp.swapaxes(self.invOneMatrixMinusJKRho, -1, 0))

	def _buildFVector(self):
		'''Compute and cache the F-vector.'''
		fVectorWithRatioOfBarrierFactors = tfnp.einsum('ijm, jm -> im', self.invOneMatrixMinusJKRho, self.pVector)
		self._fVector = fVectorWithRatioOfBarrierFactors / self.centrifugalBarrierFactorRunning
