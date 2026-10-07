import unittest
import numpy as np
import tensorflow as tf

import sphysics
from sphysics.amplitudes.resonance import KMatrixFormalismParams, KMatrixFormalism, breitWignerS, constantWidthBreitWignerS, flatteS
from sphysics._constants import Constants as C
from sphysics.kinematics._kinematics import twobodyBreakupmomentumSquared, twobodyPhasespace, twobodyBreakupmomentum, twobodyPhasespaceComplexContinuation
from sphysics.pwa.barrierFactors._bfQuiggHippel import barrierFactor_QuiggHipple, barrierFactor_QuiggHippleComplexContinuation

# Helper Functions

def doubleBW(s, m0, g0, m1, g1, L, m01, m02):
            breakupMom00 = twobodyBreakupmomentumSquared(m0**2, m01, m02)
            breakupMom0s = twobodyBreakupmomentumSquared(s, m01, m02)
            breakupMom10 = twobodyBreakupmomentumSquared(m1**2, m01, m02)
            barrier0s = barrierFactor_QuiggHipple(L, breakupMom0s)
            barrier00 = barrierFactor_QuiggHipple(L, breakupMom00)
            barrier10 = barrierFactor_QuiggHipple(L, breakupMom10)
            B0 = barrier0s/barrier00
            B1 = barrier0s/barrier10
            phaseSpace = twobodyPhasespace(s, m01, m02)
            phaseSpace0 = twobodyPhasespace(m0**2, m01, m02)
            phaseSpace1 = twobodyPhasespace(m1**2, m01, m02)
            g0 = g0/phaseSpace0
            g1 = g1/phaseSpace1
            firstSummand = (m0*g0/barrier00)/(m0**2-s-1j*phaseSpace*(B0**2*m0*g0+B1**2*m1*g1*((m0**2-s)/(m1**2-s))))
            secondSummand = (m1*g1/barrier10)/(m1**2-s-1j*phaseSpace*(B1**2*m1*g1+B0**2*m0*g0*((m1**2-s)/(m0**2-s))))
            return firstSummand + secondSummand


# Test Class

class KMatrixFormalismTest(unittest.TestCase):
    """Numerical regression tests for the K-matrix formalism."""

    # Numerical backend tests

    def test_tensorflow_kmatrix_matches_numpy_reference(self):

        # -------------------------------------------------
        # 1. Define the same mass range
        # -------------------------------------------------
        lowerMass = 0.8
        upperMass = 1.7
        nPoints = 100000

        mass_np = np.linspace(lowerMass, upperMass, nPoints)
        s_np = mass_np**2

        mass_tf = tf.linspace(
            tf.constant(lowerMass, dtype=tf.float64),
            tf.constant(upperMass, dtype=tf.float64),
            nPoints
        )
        s_tf = mass_tf**2

        # -------------------------------------------------
        # 2. Define K-matrix parameters
        #    Same setup as "Test Tensorflow KMatrix"
        # -------------------------------------------------
        nChannels = 1
        nPoles = 2

        bareMassesPoles = np.array([
            C.M['a1(1260)-'],
            1.66
        ])

        bareWidthsPoles = np.array([
            [C.G['a1(1260)-']],
            [0.425]
        ])

        decayParticlesMass = np.array([
            [0, C.M['rho(770)0'], C.M['pi0']]
        ])

        # -------------------------------------------------
        # 3. Calculate using NumPy input
        # -------------------------------------------------
        kMatrixParams_np = KMatrixFormalismParams(
            nChannels,
            nPoles,
            bareMassesPoles,
            decayParticlesMass,
            partialWidthsPoles=bareWidthsPoles
        )

        kMatrix_np = kMatrixParams_np.getKMatrixFormalism(s_np)

        amplitude_np = kMatrix_np(1)

        # -------------------------------------------------
        # 4. Calculate using TensorFlow input
        # -------------------------------------------------
        kMatrixParams_tf = KMatrixFormalismParams(
            nChannels,
            nPoles,
            bareMassesPoles,
            decayParticlesMass,
            partialWidthsPoles=bareWidthsPoles
        )

        kMatrix_tf = kMatrixParams_tf.getKMatrixFormalism(s_tf)

        amplitude_tf = kMatrix_tf(1)

        # -------------------------------------------------
        # 5. Convert TensorFlow result to NumPy
        # -------------------------------------------------
        if tf.is_tensor(amplitude_tf):
            amplitude_tf = amplitude_tf.numpy()
        else:
            amplitude_tf = np.asarray(amplitude_tf)

        amplitude_np = np.asarray(amplitude_np)

        # -------------------------------------------------
        # 6. Calculate differences
        # -------------------------------------------------
        diff_real = amplitude_tf.real - amplitude_np.real
        diff_imag = amplitude_tf.imag - amplitude_np.imag

        max_diff_real = np.max(np.abs(diff_real))
        max_diff_imag = np.max(np.abs(diff_imag))

        # -------------------------------------------------
        # 7. Require maximum difference <= 1e-13
        # -------------------------------------------------
        self.assertLessEqual(
            max_diff_real,
            1e-13,
        )

        self.assertLessEqual(
            max_diff_imag,
            1e-13,
        )


    # Physics/formalism tests

    def test_single_pole_single_channel_L0(self):
        # -------------------------------------------------
        # 1. Define parameters
        # -------------------------------------------------
        mass = np.linspace(0.4,0.9, 100)
        s = mass**2
        nChannels = 1
        nPoles = 1
        bareMassesPoles = np.array([C.M['rho(770)0']])
        bareWidthsPoles = np.array([[C.G['rho(770)0']]])
        decayParticlesMass = np.array([[0, C.M['pi+'], C.M['pi-']]])

        # -------------------------------------------------
        # 2. Compute K-matrix amplitude
        # -------------------------------------------------
        kMatrix = KMatrixFormalismParams(nChannels, nPoles, bareMassesPoles, decayParticlesMass, partialWidthsPoles=bareWidthsPoles)
        kMatrix = kMatrix.getKMatrixFormalism(s)


        # -------------------------------------------------
        # 3. Compute Breit-Wigner
        # -------------------------------------------------
        bWAmp = breitWignerS(s, C.M['pi+'], C.M['pi-'], 0, C.M['rho(770)0'], C.G['rho(770)0'])
        bWAmp = bWAmp*np.max(np.abs(kMatrix(1)))/np.max(np.abs(bWAmp))
        # -------------------------------------------------
        # 4. Compare real parts
        # -------------------------------------------------
        np.testing.assert_allclose(
            kMatrix(1).real,
            bWAmp.real,
            rtol=0.0,
            atol=1e-13,
        )

        # -------------------------------------------------
        # 5. Compare imaginary parts
        # -------------------------------------------------
        np.testing.assert_allclose(
            kMatrix(1).imag,
            bWAmp.imag,
            rtol=0.0,
            atol=1e-13,
        )


    def test_single_pole_single_channel_L1(self):
                # -------------------------------------------------
        # 1. Define parameters
        # -------------------------------------------------
        mass = np.linspace(0.5, 0.9,70)
        s = mass**2
        nChannels = 1
        nPoles = 1
        bareMassesPoles = np.array([C.M['rho(770)0']])
        bareWidthsPoles = np.array([[C.G['rho(770)0']]])
        decayParticlesMass = np.array([[1, C.M['pi+'], C.M['pi-']]])

        # -------------------------------------------------
        # 2. Compute K-matrix amplitude
        # -------------------------------------------------
        kMatrixParams = KMatrixFormalismParams(nChannels, nPoles, bareMassesPoles, decayParticlesMass, partialWidthsPoles=bareWidthsPoles)
        kMatrix = kMatrixParams.getKMatrixFormalism(s)

        kMatrix._buildFVector()
        kMatrix._fVector[0,:] /= np.max(np.abs(kMatrix._fVector[0,:]))

        # -------------------------------------------------
        # 3. Compute Breit-Wigner
        # -------------------------------------------------
        bwAmp = breitWignerS(s, C.M['pi+'], C.M['pi-'], 1, C.M['rho(770)0'], C.G['rho(770)0'])
        bwAmp = bwAmp/np.max(np.abs(bwAmp))

        # -------------------------------------------------
        # 4. Compare real parts
        # -------------------------------------------------
        np.testing.assert_allclose(
            kMatrix(1).real,
            bwAmp.real,
            rtol=0.0,
            atol=1e-13,
        )

        # -------------------------------------------------
        # 5. Compare imaginary parts
        # -------------------------------------------------
        np.testing.assert_allclose(
            kMatrix(1).imag,
            bwAmp.imag,
            rtol=0.0,
            atol=1e-13,
        )


    def test_double_pole_single_channel(self):
        # -------------------------------------------------
        # 1. Define parameters
        # -------------------------------------------------
        mass = np.linspace(0.5, 1.7, 200)
        s = mass**2
        nChannels = 1
        nPoles = 2
        bareMassesPoles = np.array([C.M['rho(770)0'], C.M['rho(1450)0']])
        bareWidthsPoles = np.array([[C.G['rho(770)0']],[C.G['rho(1450)0']]])
        decayParticlesMass = np.array([[1, C.M['pi+'], C.M['pi-']]])

        # -------------------------------------------------
        # 2. Compute equivalent K-matrix amplitude
        # -------------------------------------------------
        kMatrixParams = KMatrixFormalismParams(nChannels, nPoles, bareMassesPoles, decayParticlesMass, partialWidthsPoles=bareWidthsPoles)
        kMatrix = kMatrixParams.getKMatrixFormalism(s)

        # -------------------------------------------------
        # 3. Compute Breit-Wigner
        # -------------------------------------------------
        doubleBWAmp = doubleBW(s, C.M['rho(770)0'], C.G['rho(770)0'], C.M['rho(1450)0'], C.G['rho(1450)0'], 1, C.M['pi+'], C.M['pi-'])

        # -------------------------------------------------
        # 4. Compare real parts
        # -------------------------------------------------
        np.testing.assert_allclose(
            kMatrix(1).real,
            doubleBWAmp.real,
            rtol=0.0,
            atol=1e-13,
        )

        # -------------------------------------------------
        # 5. Compare imaginary parts
        # -------------------------------------------------
        np.testing.assert_allclose(
            kMatrix(1).imag,
            doubleBWAmp.imag,
            rtol=0.0,
            atol=1e-13,
        )


    def test_single_pole_two_channel_flatte(self):
        # -------------------------------------------------
        # 1. Define parameters
        # -------------------------------------------------
        mass = np.linspace(0.8,1.3, 200)
        s = mass**2
        nChannels = 2
        nPoles = 1
        bareMassesPoles = np.array([C.M['f0(980)0']])
        #bareWidthsPoles = np.array([[C.G['f0(980)0'],C.G['f0(980)0']]])
        bareWidthsPoles = np.array([C.G['f0(980)0']])
        decayParticlesMass = np.array([[0, C.M['pi+'], C.M['pi-']],[0, C.M['K-'], C.M['K+']]])
        decayCouplings=np.ones((nPoles,nChannels))*np.sqrt(bareWidthsPoles*bareMassesPoles)
        productionCouplings=np.ones((nPoles))*np.sqrt(bareWidthsPoles*bareMassesPoles) + 1j*np.array([0], dtype=np.float64)


        # -------------------------------------------------
        # 3. Compute equivalent K-matrix amplitude
        # -------------------------------------------------
        kMatrixParams = KMatrixFormalismParams(nChannels, nPoles, bareMassesPoles, decayParticlesMass, decayCouplings=decayCouplings, productionCouplings=productionCouplings)
        kMatrix = kMatrixParams.getKMatrixFormalism(s)

        kMatrix._buildFVector()
        kMatrix._fVector[0,:] /= max(np.abs(kMatrix._fVector[0,:]))
        kMatrix._fVector[1,:] /= max(np.abs(kMatrix._fVector[1,:]))

        # -------------------------------------------------
        # 2. Compute Flatte
        # -------------------------------------------------
        flatte = flatteS(s, M0=C.M['f0(980)0'], g1=C.M['f0(980)0']*C.G['f0(980)0'], g2g1=1)
        flatte = flatte*C.M['f0(980)0']*C.G['f0(980)0']
        flatte = flatte/max(np.abs(flatte))

        # -------------------------------------------------
        # 4. Compare real parts
        # -------------------------------------------------
        np.testing.assert_allclose(
            kMatrix(1).real,
            flatte.real,
            rtol=0.0,
            atol=1e-13,
        )

        # -------------------------------------------------
        # 5. Compare imaginary parts
        # -------------------------------------------------
        np.testing.assert_allclose(
            kMatrix(1).imag,
            flatte.imag,
            rtol=0.0,
            atol=1e-13,
        )




if __name__ == '__main__':
    unittest.main()
