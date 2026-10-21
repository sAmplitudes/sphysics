#!/usr/bin/env python
# coding: utf-8
'''
Tests of the three-body hadronic tensors and of the tau -> 3pi partial-wave model.

The hadronic tensors are compared with the form factors of
Krinner and Paul, Eur. Phys. J. C 81 (2021) 1073, and with the tensor identities of
Anisovich et al., J. Phys. G 28 (2002) 15.

@author: Stefan Wallner
'''

from __future__ import absolute_import, print_function, division

import pickle
import unittest

import numpy as np

import sphysics
from sphysics import lorentz
from sphysics import Constants as C
from sphysics.pwa.threebody import HadronicTensors
from sphysics.pwa.threebody._hadronicTensors import Wave
from sphysics.pwa.twobody._hadronicTensor import Wave as TwoBodyWave
from sphysics.pwa.threebody import tau as tau3
from sphysics.pwa.threebody.tau._utils import decomposeHermitianMatrix2
from sphysics.amplitudes import resonance, LambdaF
from sphysics.pwa import barrierFactors


ET = lorentz.et
REFLECT = np.array([1., -1., -1., -1.])[:, None]  # reverses the spatial components of a four-vector


def _threeBodyEvents(nEvents, M, m1, m2, m3, rng):
	'''Three-body decay events with arbitrary masses, boosted with random velocities.'''
	p1, p2, p3 = sphysics.generators.NBodyGenerator(M, [m1, m2, m3]).gen(nEvents)
	beta = rng.normal(size=(3, nEvents))
	beta *= rng.uniform(0., 0.8, nEvents)/np.linalg.norm(beta, axis=0)
	gamma = 1./np.sqrt(1. - np.sum(beta**2, axis=0))
	pX = np.vstack((gamma*M, gamma*M*beta))
	boost = lorentz.getBoostFromRestFrame(pX)
	return tuple(lorentz.applyBoost(boost, p) for p in (p1, p2, p3))


def _axialFormFactors(t, p2perp, p3perp):
	'''Decompose t = p2perp f2 + p3perp f3 (eq. (28) of Krinner and Paul) by least squares.

	Returns f2, f3 and the relative residual of the decomposition.'''
	f2 = np.empty(t.shape[-1])
	f3 = np.empty(t.shape[-1])
	for e in range(t.shape[-1]):
		(f2[e], f3[e]), *_ = np.linalg.lstsq(np.stack([p2perp[:, e], p3perp[:, e]], axis=1), t[:, e], rcond=None)
	return f2, f3, np.max(np.abs(t - p2perp*f2 - p3perp*f3))/np.max(np.abs(t))


def _relDev(a, b):
	'''Maximum absolute deviation between a and b relative to the maximum absolute value of b.'''
	return np.max(np.abs(a - b))/np.max(np.abs(b))


class HadronicTensorsTest(unittest.TestCase):
	'''Tests of sphysics.pwa.threebody.HadronicTensors against Krinner and Paul and Anisovich et al.'''

	@classmethod
	def setUpClass(cls):
		sphysics.random.generators.setSeed(1234)
		rng = np.random.default_rng(1234)
		mpi = C.M.pi
		cls.samples = [_threeBodyEvents(500, 1.4, mpi, mpi, mpi, rng) + ((mpi, mpi, mpi),),
		               _threeBodyEvents(500, 1.8, 0.4937, 0.1396, 0.1396, rng) + ((0.4937, 0.1396, 0.1396),),
		               _threeBodyEvents(500, 2.2, 0.3, 0.5, 0.7, rng) + ((0.3, 0.5, 0.7),)]
		cls.waves1 = ['wave_1p_0p_P', 'wave_1p_1m_S', 'wave_1p_1m_D', 'wave_1p_2p_P', 'wave_1p_2p_F', 'wave_1p_3m_D', 'wave_1m_1m_P']
		cls.waves0 = ['wave_0m_0p_S', 'wave_0m_1m_P', 'wave_0m_2p_D', 'wave_0m_3m_F']


	def test_tensorProperties(self):
		'''Transversality, symmetry and tracelessness of the tensors and the contraction identities eqs. (8) and (10) of Anisovich et al.'''
		for p1, p2, p3, _ in self.samples:
			H = HadronicTensors(p1, p2, p3)
			pX = H.p123
			self.assertAlmostEqual(np.max(np.abs(lorentz.lp(H.k123, pX))), 0., 12)
			self.assertAlmostEqual(np.max(np.abs(lorentz.lp(H.k12, H.p12))), 0., 12)
			self.assertAlmostEqual(np.max(np.abs(lorentz.lp(H.k123, H.k123) + sphysics.kinematics.twobodyBreakupmomentumSquared(H.s123, H.m12, H.m3))), 0., 12)
			self.assertAlmostEqual(np.max(np.abs(lorentz.lp(H.k12, H.k12) + sphysics.kinematics.twobodyBreakupmomentumSquared(H.s12, H.m1, H.m2))), 0., 12)
			for k, TD, TF, p in ((H.k123, H.tensor_123_Dwave, H.tensor_123_Fwave, pX), (H.k12, H.tensor_12_Dwave, H.tensor_12_Fwave, H.p12)):
				k2 = lorentz.lp(k, k)
				self.assertAlmostEqual(np.max(np.abs(TD - np.transpose(TD, (1, 0, 2)))), 0., 12)
				self.assertAlmostEqual(np.max(np.abs(np.einsum('mme,m->e', TD, ET))), 0., 12)
				self.assertAlmostEqual(np.max(np.abs(np.einsum('mne,n,ne->me', TD, ET, p))), 0., 12)
				for perm in [(1, 0, 2, 3), (0, 2, 1, 3), (2, 1, 0, 3)]:
					self.assertAlmostEqual(np.max(np.abs(TF - np.transpose(TF, perm))), 0., 12)
				self.assertAlmostEqual(np.max(np.abs(np.einsum('mmre,m->re', TF, ET))), 0., 12)
				self.assertAlmostEqual(np.max(np.abs(np.einsum('mnre,r,re->mne', TF, ET, p))), 0., 12)
				# eq. (8): X^(L) k = k^2 X^(L-1); eq. (10): X^(L).X^(L) = (2L-1)!!/L! k^(2L)
				self.assertAlmostEqual(_relDev(np.einsum('mne,n,ne->me', TD, ET, k), k2*k), 0., 12)
				self.assertAlmostEqual(_relDev(np.einsum('mnre,r,re->mne', TF, ET, k), k2*TD), 0., 12)
				self.assertAlmostEqual(_relDev(np.einsum('mne,mne,m,n->e', TD, TD, ET, ET), 1.5*k2**2), 0., 12)
				self.assertAlmostEqual(_relDev(np.einsum('mnre,mnre,m,n,r->e', TF, TF, ET, ET, ET), 2.5*k2**3), 0., 12)
			for wave in self.waves1:
				t = H[wave]
				self.assertAlmostEqual(np.max(np.abs(lorentz.lp(pX, t)))/np.max(np.abs(t)), 0., 12, msg=wave)
			for wave in self.waves0:
				t = H[wave]
				self.assertAlmostEqual(_relDev(t, pX*t[0]/pX[0]), 0., 12, msg=wave)


	def test_formFactors(self):
		'''Compare the hadronic tensors with the form factors of Krinner and Paul for arbitrary final-state masses.'''
		for p1, p2, p3, (m1, m2, m3) in self.samples:
			H = HadronicTensors(p1, p2, p3)
			sX, s12, s13, s23 = H.s123, H.s12, H.s13, H.s23
			m12, m22, m32 = m1**2, m2**2, m3**2
			p2perp = np.einsum('mne,n,ne->me', H.g123, ET, p2)
			p3perp = np.einsum('mne,n,ne->me', H.g123, ET, p3)
			# eqs. (53)-(57)
			A = sX - m32 + s12
			FI = -sX**2 - (m32 - s12)**2 + 2*sX*(s12 + m32)
			FII = (sX + s12 - m32)*(m22*m32 + s12*(m22 + m32) - s12**2 + sX*(s12 - m22) + m12*(sX - m32 + s12) - 2*s12*s13)
			FIII = sX*s12*(m12 - m22)**2 - sX*s12**2*(2*m12 + 2*m22 - s12)
			FIV = m12**2 + (s12 - m22)**2 - 2*m12*(s12 + m22)
			FV = (sX - m32)*(m12 - m22) - s12*(s13 - s23)
			D = s12 - m12 + m22

			f2, f3, res = _axialFormFactors(H.wave_1p_0p_P, p2perp, p3perp)                     # eq. (29)
			self.assertAlmostEqual(res, 0., 10)
			self.assertAlmostEqual(np.max(np.abs(f2)), 0., 10)
			self.assertAlmostEqual(np.max(np.abs(f3 + 1.)), 0., 10)

			f2, f3, res = _axialFormFactors(H.wave_1p_1m_S, p2perp, p3perp)                     # eq. (52)
			self.assertAlmostEqual(res, 0., 10)
			self.assertAlmostEqual(np.max(np.abs(f2 + 1.)), 0., 10)
			self.assertAlmostEqual(np.max(np.abs(f3 + 0.5*(1. - (m12 - m22)/s12))), 0., 10)

			f2, f3, res = _axialFormFactors(H.wave_1p_1m_D, p2perp, p3perp)                     # eqs. (31), (58)
			self.assertAlmostEqual(res, 0., 10)
			self.assertAlmostEqual(_relDev(f2, (2*s12 - sX + 2*m32 - (s12 - m32)**2/sX)/8.), 0., 10)
			self.assertAlmostEqual(_relDev(f3, (D*FI - 3*A*FV)/(16*sX*s12)), 0., 10)

			f2, f3, res = _axialFormFactors(H.wave_1p_2p_P, p2perp, p3perp)                     # eqs. (59), (60)
			self.assertAlmostEqual(res, 0., 10)
			f2Exp = -3*(sX + s12 - m32)*FV/(16*sX*s12)
			self.assertAlmostEqual(_relDev(f2, f2Exp), 0., 10)
			self.assertAlmostEqual(_relDev(f3, D/(2*s12)*f2Exp + FIII/(32*sX**2*s12**3)*(sX*(3*s12 - sX + m32) - 4*sX*s12 + (m32 - s12)*A)), 0., 10)

			f2, f3, res = _axialFormFactors(H.wave_1p_2p_F, p2perp, p3perp)                     # eqs. (61), (62)
			self.assertAlmostEqual(res, 0., 10)
			self.assertAlmostEqual(_relDev(f2, 3*FI*FII/(64*sX**2*s12)), 0., 10)
			self.assertAlmostEqual(_relDev(f3, (-3*FI*(4*FIII + FV**2 - 2*D*FII) + 2*FIV*FI**2 - 15*FII**2)/(256*sX**2*s12**2)), 0., 10)

			f2, f3, res = _axialFormFactors(H.wave_1p_3m_D, p2perp, p3perp)                     # eqs. (63), (64)
			self.assertAlmostEqual(res, 0., 10)
			# eq. (63) as printed misses the factor FI in the last term, which is dimensionally required
			self.assertAlmostEqual(_relDev(f2, (-3*A**2*FIV*FI - 15*FII**2 + FI*(-20*FIII - 5*FV**2) + FIV*FI*(16*sX*s12 + A**2))/(256*sX**2*s12**2)), 0., 10)
			self.assertAlmostEqual(_relDev(f3, -((10*FV**2*D*sX*s12 + 4*FIII*FV*A)*((sX + s12 - m32)**2 + 2*sX*s12) + 3*FI*FIII*D*A**2
			                                     + FIII*D*((sX + m32 - s12)**2 - 4*sX*m32)**2)/(512*sX**3*s12**4)), 0., 10)

			fs = H.wave_0m_1m_P[0]/H.p123[0]                                                    # eq. (65)
			self.assertAlmostEqual(_relDev(fs, -2./3.*f2Exp), 0., 10)
			fs = H.wave_0m_2p_D[0]/H.p123[0]                                                    # eq. (66)
			self.assertAlmostEqual(_relDev(fs, (-(sX**2 - 2*sX*(m32 - 5*s12) + (m32 - s12)**2 - 3*A**2)*FIV*FI + 9*A**2*FV**2 + 3*FI*(4*FIII + FV**2))/(256*sX**2*s12**2)), 0., 10)
			fs = H.wave_0m_3m_F[0]/H.p123[0]                                                    # eq. (67)
			# the prefactor of eq. (67) as printed is 1/200; the tensor contraction gives 1/2048
			self.assertAlmostEqual(_relDev(fs, FII/(2048*sX**3*s12**3)*(15*A**2*FIV*FI - 3*FIV*FI*(-3*FI + 20*sX*s12) + 25*FII**2 + 15*FI*(4*FIII + FV**2))), 0., 10)

			if m1 == m2 == m3:                                                                  # eqs. (32), (37)
				f2, f3, _ = _axialFormFactors(H.wave_1p_2p_P, p2perp, p3perp)
				f2Exp = 3./16.*(s13 - s23)*(1. + (s12 - m32)/sX)
				self.assertAlmostEqual(_relDev(f2, f2Exp), 0., 10)
				self.assertAlmostEqual(_relDev(f3, (4*m32 - s12)/(32*sX*s12)*(sX + s12 - m32)**2 + f2Exp/2.), 0., 10)
				fs = H.wave_0m_1m_P[0]/H.p123[0]
				self.assertAlmostEqual(_relDev(fs, (s23 - s13)*(1. + (s12 - m32)/sX)/8.), 0., 10)


	def test_vectorWave(self):
		'''The 1- current is proportional to eps(p1, p2, p3), with opposite sign for the Bose-symmetrization partner (eq. (35)).'''
		for p1, p2, p3, _ in self.samples:
			H = HadronicTensors(p1, p2, p3)
			pEps = np.einsum('mnls,n,ne,l,le,s,se->me', lorentz.epsilon, ET, p1, ET, p2, ET, p3)
			self.assertAlmostEqual(_relDev(H.wave_1m_1m_P, pEps), 0., 10)
			self.assertAlmostEqual(_relDev(HadronicTensors(p1, p3, p2).wave_1m_1m_P, -pEps), 0., 10)


	def test_parityOfCurrents(self):
		'''Axial-vector and pseudo-scalar currents are true four-vectors, the vector current is a pseudo-vector.'''
		p1, p2, p3, _ = self.samples[0]
		H = HadronicTensors(p1, p2, p3)
		HR = HadronicTensors(REFLECT*p1, REFLECT*p2, REFLECT*p3)
		for wave in self.waves1 + self.waves0:
			sign = -1. if wave == 'wave_1m_1m_P' else 1.
			self.assertAlmostEqual(_relDev(HR[wave], sign*REFLECT*H[wave]), 0., 12, msg=wave)


	def test_waveFromString(self):
		'''Parsing of wave labels.'''
		for label in ['wave_1p_1m_S', '1p_1m_S', 'wave_1p_rho(770)[1m]_S', '1p_rho(770)[1m]_S']:
			wave = Wave.fromString(label)
			self.assertEqual((wave.J_X, wave.P_X, wave.J_isobar, wave.P_isobar, wave.L), (1, 1, 1, -1, 0), msg=label)
			self.assertEqual(wave.isobarLabel, 'rho(770)' if 'rho' in label else None, msg=label)
		wave = Wave.fromString('wave_0m_f2(1270)[2p]_D')
		self.assertEqual((wave.J_X, wave.P_X, wave.J_isobar, wave.P_isobar, wave.L), (0, -1, 2, 1, 2))
		self.assertEqual(str(wave), '0m_f2(1270)[2p]_D')
		self.assertEqual(str(Wave.fromString('wave_1m_1m_P')), '1m_1m_P')
		for label in ['wave_1p_rho(770)_S', 'wave_1p_a1_S', '1p_1m', 'wave_1p_(pipi)_S[0p]_P', 'wave_1x_1m_S', 'wave_1p_rho(770)[1]_S',
		              'wave_1p_rho(770)[1m_S', 'wave_1p_rho(770)[1m]]_S', 'wave_1p_[1m]_S', 'wave_1p_1m_X', 'wave_1p_1m_s']:
			with self.assertRaises(ValueError, msg=label):
				Wave.fromString(label)


	def test_waveFromStringTwoBody(self):
		'''Parsing of two-body wave labels.'''
		for label in ['wave_1m', '1m']:
			wave = TwoBodyWave.fromString(label)
			self.assertEqual((wave.J_X, wave.P_X), (1, -1), msg=label)
		self.assertEqual(str(TwoBodyWave.fromString('wave_0p')), '0p')
		for label in ['wave_1', 'wave_1m_S', '1x', 'xm', 'wave__1m', 'wavewave_1m', '']:
			with self.assertRaises(ValueError, msg=label):
				TwoBodyWave.fromString(label)


	def test_pickle(self):
		'''Pickling a kinematics object does not reset the object itself, and the pickle contains no event data.'''
		p1, p2, p3, _ = self.samples[0]
		H = HadronicTensors(p1, p2, p3)
		current = H.wave_1p_1m_S.copy()
		restored = pickle.loads(pickle.dumps(H))
		self.assertIsNone(restored._p1)  # pylint: disable=protected-access
		self.assertAlmostEqual(np.max(np.abs(H.wave_1p_1m_S - current)), 0., 14)


class Tau3PiModelTest(unittest.TestCase):
	'''Tests of sphysics.pwa.threebody.tau.Tau2ThreeCharedPi.'''

	@classmethod
	def setUpClass(cls):
		sphysics.random.generators.setSeed(1234)
		cls.nEvents = 2000
		cls.ECMS = C.M.upsilon4S
		mpi = C.M.pi
		pTau, pNu, p1, p2, p3 = sphysics.generators.tauToThreePi(cls.nEvents, cls.ECMS**2, (1.0, 1.5))
		cls.momenta = (p1, p2, p3, pTau, pNu)
		cls.isParticle = np.arange(cls.nEvents) % 2 == 0
		cls.events = cls._makeEvents(p1, p2, p3, pTau, pNu, cls.isParticle)

		def bwRho(m):
			return resonance.breitWigner(m, mpi, mpi, 1, 0.775, 0.149)
		def bwF2(m):
			return resonance.breitWigner(m, mpi, mpi, 2, 1.2755, 0.1867)
		def bwF0(m):
			return resonance.constantWidthBreitWigner(m, 0.99, 0.055)
		def one(m):
			return np.ones_like(m, dtype=complex)
		barrierX = LambdaF(barrierFactors.compensationFactor_unnormalizedM, m2=mpi)
		barrierIsobar = LambdaF(barrierFactors.compensationFactor_unnormalizedM, m1=mpi, m2=mpi)
		cls.model = tau3.Tau2ThreeCharedPi('test')
		for name, tensorName, isobar in [('1p_rho_S', 'wave_1p_1m_S', bwRho), ('1p_f0_P', 'wave_1p_0p_P', bwF0), ('1p_f2_P', 'wave_1p_2p_P', bwF2),
		                                 ('0m_rho_P', 'wave_0m_1m_P', bwRho), ('1m_rho_P', 'wave_1m_1m_P', bwRho)]:
			cls.model.addWave(name, tensorName, one, isobar, barrierX, barrierIsobar)
		cls.couplings = {'1p_rho_S': 1.0, '1p_f0_P': 0.4+0.3j, '1p_f2_P': -0.3+0.6j, '0m_rho_P': 0.2-0.5j, '1m_rho_P': 0.7+0.2j}
		cls.couplingVector = np.array([cls.couplings[name] for name in cls.model.waveNames])


	@classmethod
	def _makeEvents(cls, p1, p2, p3, pTau, pNu, isParticle):
		events = tau3.EventsTau2ThreeChargedPi(cls.ECMS/2., C.M.tau)
		events.pi1PM, events.pi2MP, events.pi3MP, events.tauMP, events.nu = p1, p2, p3, pTau, pNu
		events.isParticle = isParticle
		events.check()
		return events


	def _intensity(self, events, couplings):
		'''Intensity from the alpha-integrated decay amplitudes.'''
		amplitudes = self.model.calcDecayAmplitudesIntegrated(events, normalized=False)
		couplingVector = np.array([couplings[name] for name in self.model.waveNames])
		return np.sum(np.abs(np.einsum('a,aie->ie', couplingVector, amplitudes))**2, axis=0)


	def test_boseSymmetry(self):
		'''The symmetrized currents are invariant under the exchange of the two same-charge pions.'''
		p1, p2, p3, pTau, pNu = self.momenta
		currents = self.model.calcHadronicCurrents(self.events)
		currentsSwapped = self.model.calcHadronicCurrents(self._makeEvents(p1, p3, p2, pTau, pNu, self.isParticle))
		for name, current, currentSwapped in zip(self.model.waveNames, currents, currentsSwapped):
			self.assertAlmostEqual(_relDev(currentSwapped, current), 0., 12, msg=name)
			if name.startswith('1'):
				self.assertAlmostEqual(np.max(np.abs(lorentz.lp(p1+p2+p3, current)))/np.max(np.abs(current)), 0., 10, msg=name)


	def test_leptonicTensor(self):
		'''The leptonic tensor from the trace formula equals 1/2 sum_h l l* from the explicit spinors for tau- and tau+.'''
		_, _, _, pTau, pNu = self.momenta
		tensor = self.model.calcLeptonicTensorUnpolarized(pTau, pNu, self.isParticle)
		up = self.model.calcLeptonicCurrent(pTau, pNu, True, self.isParticle)
		down = self.model.calcLeptonicCurrent(pTau, pNu, False, self.isParticle)
		tensorSpinors = 0.5*(lorentz.tensorproduct(up, np.conj(up)) + lorentz.tensorproduct(down, np.conj(down)))
		self.assertAlmostEqual(_relDev(tensor, tensorSpinors), 0., 12)
		# scalar flags (Python and numpy bool) are equivalent to the corresponding arrays
		dirac = self.model._dirac  # pylint: disable=protected-access
		for flag in (False, np.False_):
			self.assertAlmostEqual(_relDev(dirac.weakLepTensor_unpolarizedDecay(pTau, pNu, flag),
			                               dirac.weakLepTensor_unpolarizedDecay(pTau, pNu, np.zeros(self.nEvents, dtype=bool))), 0., 12)
		for flag in (True, np.True_):
			self.assertAlmostEqual(_relDev(dirac.weakLepTensor_unpolarizedDecay(pTau, pNu, flag),
			                               dirac.weakLepTensor_unpolarizedDecay(pTau, pNu, np.ones(self.nEvents, dtype=bool))), 0., 12)
		# non-boolean flags, e.g. charges, are rejected instead of being silently cast to True
		for flag in (np.where(self.isParticle, 1, -1), 1, 0.):
			with self.assertRaises(ValueError, msg=repr(flag)):
				dirac.weakLepTensor_unpolarizedDecay(pTau, pNu, flag)


	def test_decayAmplitudes(self):
		'''The decay amplitudes from the eigen-decomposition of the leptonic tensor reproduce the intensity from the explicit leptonic currents.'''
		intensity = self.model.calcIntensityUnpolarized(self.events, self.couplings, normalized=False)
		amplitudes = self.model.calcDecayAmplitudes(self.events, normalized=False)
		self.assertEqual(amplitudes.shape, (self.model.nWaves, 4, self.nEvents))
		intensityAmplitudes = np.sum(np.abs(np.einsum('a,aie->ie', self.couplingVector, amplitudes))**2, axis=0)
		self.assertAlmostEqual(_relDev(intensityAmplitudes, intensity), 0., 12)


	def test_alphaIntegration(self):
		'''The alpha-integrated leptonic tensor equals the numerical integral over the unknown tau direction.'''
		p1, p2, p3, pTau, pNu = self.momenta
		p123 = p1+p2+p3
		parallel, ortho1, ortho2 = tau3.getTauMomentumComponentsCMS(p123, self.ECMS/2., C.M.tau)
		nAlpha = 32
		tensorNumerical = 0.
		for alpha in (np.arange(nAlpha)+0.5)*2*np.pi/nAlpha:
			pTauAlpha = parallel + np.cos(alpha)*ortho1 + np.sin(alpha)*ortho2
			self.assertAlmostEqual(np.max(np.abs(lorentz.lp(pTauAlpha, pTauAlpha) - C.M.tau**2)), 0., 10)
			self.assertAlmostEqual(np.max(np.abs(lorentz.lp(pTauAlpha-p123, pTauAlpha-p123))), 0., 10)
			tensorNumerical = tensorNumerical + self.model.calcLeptonicTensorUnpolarized(pTauAlpha, pTauAlpha-p123, self.isParticle)*(2*np.pi/nAlpha)
		tensorIntegrated = self.model.calcLeptonicTensorIntegratedUnpolarized(self.events)
		self.assertAlmostEqual(_relDev(tensorIntegrated, tensorNumerical), 0., 12)
		# the true tau momentum is one of the pTau(alpha)
		cosAlpha = lorentz.lp(pTau-parallel, ortho1)/lorentz.lp(ortho1, ortho1)
		sinAlpha = lorentz.lp(pTau-parallel, ortho2)/lorentz.lp(ortho2, ortho2)
		self.assertAlmostEqual(np.max(np.abs(pTau - (parallel + cosAlpha*ortho1 + sinAlpha*ortho2))), 0., 10)
		# the integrated decay amplitudes reproduce J L_int J*
		current = self.model.calcTotalHadronicCurrent(self.events, self.couplings, normalized=False)
		intensityIntegrated = np.einsum('me,mne,ne,m,n->e', current, tensorIntegrated, np.conj(current), ET, ET).real
		self.assertAlmostEqual(_relDev(self._intensity(self.events, self.couplings), intensityIntegrated), 0., 12)


	def test_cpMirrorWithoutVectorWave(self):
		'''For axial-vector and pseudo-scalar waves, the tau+ intensity at space-reflected momenta equals the tau- intensity.'''
		p1, p2, p3, pTau, pNu = self.momenta
		eventsMinus = self._makeEvents(p1, p2, p3, pTau, pNu, np.ones(self.nEvents, dtype=bool))
		eventsPlusMirror = self._makeEvents(REFLECT*p1, REFLECT*p2, REFLECT*p3, REFLECT*pTau, REFLECT*pNu, np.zeros(self.nEvents, dtype=bool))
		couplings = dict(self.couplings, **{'1m_rho_P': 0.})
		self.assertAlmostEqual(_relDev(self.model.calcIntensityUnpolarized(eventsPlusMirror, couplings, normalized=False),
		                               self.model.calcIntensityUnpolarized(eventsMinus, couplings, normalized=False)), 0., 12)
		self.assertAlmostEqual(_relDev(self._intensity(eventsPlusMirror, couplings), self._intensity(eventsMinus, couplings)), 0., 12)


	def test_cpMirrorWithVectorWave(self):
		'''For the same couplings, the tau+ intensity at space-reflected momenta equals the tau- intensity (CP invariance),
		also with a vector-current (1-) wave interfering with the axial-vector and pseudo-scalar waves
		(cf. Finkemeier and Mirkes, AIP Conf. Proc. 349 (1996) 119, arXiv:hep-ph/9508312, eq. (42)).'''
		p1, p2, p3, pTau, pNu = self.momenta
		eventsMinus = self._makeEvents(p1, p2, p3, pTau, pNu, np.ones(self.nEvents, dtype=bool))
		eventsPlusMirror = self._makeEvents(REFLECT*p1, REFLECT*p2, REFLECT*p3, REFLECT*pTau, REFLECT*pNu, np.zeros(self.nEvents, dtype=bool))
		self.assertAlmostEqual(_relDev(self.model.calcIntensityUnpolarized(eventsPlusMirror, self.couplings, normalized=False),
		                               self.model.calcIntensityUnpolarized(eventsMinus, self.couplings, normalized=False)), 0., 10)
		self.assertAlmostEqual(_relDev(self._intensity(eventsPlusMirror, self.couplings), self._intensity(eventsMinus, self.couplings)), 0., 10)


	def test_antiparticleCurrents(self):
		'''The tau+ current is the CP transform of the tau- current, J_+(p) = Lambda_P J_-(R p), for every wave:
		only the vector-current wave changes sign relative to the tau- formula. The charge of every event must be given as a boolean flag.'''
		p1, p2, p3, _, _ = self.momenta
		currentsMinus = self.model.calcHadronicCurrentsUnsymmetrized(p1, p2, p3)
		currentsPlus = self.model.calcHadronicCurrentsUnsymmetrized(p1, p2, p3, isParticle=False)
		currentsMinusReflected = self.model.calcHadronicCurrentsUnsymmetrized(REFLECT*p1, REFLECT*p2, REFLECT*p3)
		currentsMixed = self.model.calcHadronicCurrentsUnsymmetrized(p1, p2, p3, isParticle=self.isParticle)
		for name, minus, plus, minusReflected, mixed in zip(self.model.waveNames, currentsMinus, currentsPlus, currentsMinusReflected, currentsMixed):
			self.assertAlmostEqual(_relDev(plus, REFLECT*minusReflected), 0., 12, msg=name)
			self.assertAlmostEqual(_relDev(plus, (-1. if name.startswith('1m') else 1.)*minus), 0., 12, msg=name)
			self.assertAlmostEqual(_relDev(mixed[:, self.isParticle], minus[:, self.isParticle]), 0., 12, msg=name)
			self.assertAlmostEqual(_relDev(mixed[:, ~self.isParticle], plus[:, ~self.isParticle]), 0., 12, msg=name)
		events = tau3.EventsTau2ThreeChargedPi(self.ECMS/2., C.M.tau)
		events.pi1PM, events.pi2MP, events.pi3MP = p1, p2, p3
		with self.assertRaises(ValueError):
			self.model.calcHadronicCurrents(events)
		# charges instead of boolean flags are rejected instead of being silently cast to True
		events.isParticle = np.where(self.isParticle, 1, -1)
		with self.assertRaises(ValueError):
			self.model.calcHadronicCurrents(events)
		with self.assertRaises(ValueError):
			self.model.calcHadronicCurrentsUnsymmetrized(p1, p2, p3, isParticle=events.isParticle)


	def test_decomposeHermitianMatrix(self):
		'''Decomposition of positive semi-definite hermitian matrices; significantly negative eigenvalues raise.'''
		_, _, _, pTau, pNu = self.momenta
		tensor = self.model.calcLeptonicTensorUnpolarized(pTau, pNu, self.isParticle)
		vectors = decomposeHermitianMatrix2(tensor)
		self.assertAlmostEqual(_relDev(np.einsum('ime,ine->mne', vectors, np.conj(vectors)), tensor), 0., 12)
		# match the message, since the subsequent reconstruction check would raise a ValueError as well
		with self.assertRaisesRegex(ValueError, 'negative eigenvalues'):
			decomposeHermitianMatrix2(np.diag([1., -0.5, 2., 3.]).astype(complex)[:, :, None])
		matrix = np.diag([1., -1e-13, 2., 3.]).astype(complex)[:, :, None]  # numerically zero eigenvalue is tolerated
		vectors = decomposeHermitianMatrix2(matrix)
		self.assertAlmostEqual(np.max(np.abs(np.einsum('ime,ine->mne', vectors, np.conj(vectors)) - matrix)), 0., 12)


	def test_tauMomentumComponents(self):
		'''Reconstruction of the tau momentum components; events outside the kinematic limits and degenerate reference directions raise.'''
		tauE = self.ECMS/2.
		pHadron = np.array([[3.5, 3.5], [1.0, -0.5], [2.0, 1.5], [2.4, -2.9]])  # physical events
		parallel, ortho1, ortho2 = tau3.getTauMomentumComponentsCMS(pHadron, tauE, C.M.tau)
		for p in (ortho1, ortho2):
			self.assertAlmostEqual(np.max(np.abs(lorentz.lp(p, pHadron))), 0., 10)
			self.assertAlmostEqual(np.max(np.abs(lorentz.lp(p, parallel))), 0., 10)
		self.assertAlmostEqual(np.max(np.abs(lorentz.lp(ortho1, ortho2))), 0., 10)
		self.assertAlmostEqual(np.max(np.abs(lorentz.lp(ortho1, ortho1) - lorentz.lp(ortho2, ortho2))), 0., 10)
		pTau = parallel + np.cos(1.)*ortho1 + np.sin(1.)*ortho2
		self.assertAlmostEqual(np.max(np.abs(lorentz.lp(pTau, pTau) - C.M.tau**2)), 0., 10)
		self.assertAlmostEqual(np.max(np.abs(pTau[0] - tauE)), 0., 10)
		self.assertAlmostEqual(np.max(np.abs(lorentz.lp(pTau - pHadron, pTau - pHadron))), 0., 10)
		# event outside the kinematic limits of the tau decay (|cos(theta)| > 1)
		with self.assertRaises(ValueError):
			tau3.getTauMomentumComponentsCMS(np.array([[3.5, 1.5], [1.0, 0.3], [2.0, -0.2], [2.4, 0.8]]), tauE, C.M.tau)
		# hadron momentum along the reference direction
		with self.assertRaises(ValueError):
			tau3.getTauMomentumComponentsCMS(np.array([[3.5], [0.], [0.], [3.288]]), tauE, C.M.tau)
		parallel, ortho1, ortho2 = tau3.getTauMomentumComponentsCMS(np.array([[3.5], [0.], [0.], [3.288]]), tauE, C.M.tau, orthoDirection=[1., 0., 0.])
		self.assertTrue(np.all(np.isfinite(ortho1)) and np.all(np.isfinite(ortho2)))


	def test_tauMomentumComponentsAtKinematicLimit(self):
		'''Events with |cos(theta)| above 1 within the rounding tolerance are clipped instead of giving NaN.'''
		tauE = self.ECMS/2.
		tauAbsP = np.sqrt(tauE**2 - C.M.tau**2)
		eNu = np.array([0.8, 1.3, 2.1, 2.9])
		pHadron = np.vstack((tauE - eNu, np.outer([0.36, -0.48, 0.8], tauAbsP - eNu)))  # collinear with the tau: cos(theta) = 1
		pHadron[0] += 2e-12  # cos(theta) - 1 = 8e-14 ... 6e-13, inside the rounding tolerance of 1e-12
		parallel, ortho1, ortho2 = tau3.getTauMomentumComponentsCMS(pHadron, tauE, C.M.tau)
		self.assertTrue(all(np.all(np.isfinite(p)) for p in (parallel, ortho1, ortho2)))
		self.assertAlmostEqual(np.max(np.abs(ortho1)), 0., 5)
		self.assertAlmostEqual(np.max(np.abs(ortho2)), 0., 5)
		self.assertAlmostEqual(np.max(np.abs(lorentz.lp(parallel, parallel) - C.M.tau**2)), 0., 5)


if __name__ == '__main__':
	sphysics.random.generators.setSeed(1234)
	unittest.main()
