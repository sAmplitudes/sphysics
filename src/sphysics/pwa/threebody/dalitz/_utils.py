# coding: utf-8
'''
Created on Monday 30 01 2023
Author: Stefan Wallner
Description: Utility functions and classes for Dalitz Plot analyses
'''

from __future__ import absolute_import, print_function, division, annotations

import gc
import numpy as np

from ...._constants import Constants as C
from .... import generators
from .... import math
from .... import eventselection
from ....utils import Logger
from ._model import PTo3P, B0ToKpipi0, BpToKSpipi0
from ._generatorWeights import calculateGeneratorWeightsB0ToKpipi0

log = Logger("dalitz")


def calcDecayAmplitudesIntegrals(onlyDiagonal: bool,
                                 normalizedDecayAmplplitudes: bool,
                                 model: PTo3P,
                                 nMCevents: int,
                                 nEventsPerBatch: int = 1000000,
                                 motherMass: float = None,
                                 fsMasses: np.array = None) -> np.ndarray:
	"""Calculate the integral matrox of the decay amplitudes

	Args:
		onlyDiagonal (bool): Calculate only diagonal elements, i.e. normalization integrals
		normalizedDecayAmplplitudes (bool): Calculate integrals of normalized decay amplitudes
		model (PTo3P): Amplitude model
		nMCevents (int): Number of phase-events used to calculate the integrals
		nEventsPerBatch (int, optional): Integrals are calculated in batches for cache efficiency. Defaults to 1000000.
		motherMass (float, optional): Mass of mother particle. If not given, derived from `model`. Defaults to None.
		fsMasses (np.array, optional): Masses of 3 final-state particle. If not given, derived from `model`. Defaults to None.

	Returns:
		np.ndarray: Array of shape [<nWaves>] if onlyDiagonal else [<nWaves>,<nWaves>]
	"""
	gc.collect()

	if motherMass is None:
		if isinstance(model, B0ToKpipi0):
			motherMass = C.M.B0
		elif isinstance(model, BpToKSpipi0):
			motherMass = C.M.B
		else:
			motherMass = model.m123
	if fsMasses is None:
		if isinstance(model, B0ToKpipi0):
			fsMasses = [C.M.K, C.M.pi, C.M.pi0]
		elif isinstance(model, BpToKSpipi0):
			fsMasses = [C.M.K0, C.M.pi, C.M.pi0]
		else:
			fsMasses = [model.m1, model.m2, model.m3]

	nEventsPerBatch = min(nEventsPerBatch, nMCevents)

	integrals = None
	totalEvents = 0
	nBatches = int(nMCevents // nEventsPerBatch)
	q = nMCevents // nBatches
	m = nMCevents % nBatches
	log.initProgress(nBatches)
	for iBatch in range(nBatches):
		nEvents = q + (1 if iBatch < m else 0)
		psGenerator = generators.NBodyGenerator(motherMass, fsMasses)
		p1, p2, p3 = psGenerator.gen(nEvents)

		decayAmplitudes = model.calcDecayAmplitudes(
		    p1, p2, p3, normalized=normalizedDecayAmplplitudes)

		if onlyDiagonal:
			batchIntegrals = math.sum(math.abs2(decayAmplitudes), axis=1)
		else:
			batchIntegrals = math.einsum('ae,be->ab', decayAmplitudes,
			                             math.conjugate(decayAmplitudes))
		if integrals is None:
			integrals = batchIntegrals
		else:
			integrals += batchIntegrals
		totalEvents += nEvents
		log.updateProgress()
	log.finishProgress()

	integrals /= totalEvents

	del p1, p2, p3, decayAmplitudes, psGenerator
	gc.collect()

	return integrals


def calcDecayAmplitudesIntegratedNormIntegrals(
        model: PTo3P,
        nMCevents: int,
        nEventsPerBatch: int = 1000000,
        motherMass: float = None,
        fsMasses: np.array = None) -> np.ndarray:
	"""Calculate the normalization integrals of the decay amplitudes

	Args:
		model (PTo3P): Amplitude model
		nMCevents (int): Number of phase-events used to calculate the integrals
		nEventsPerBatch (int, optional): Integrals are calculated in batches for cache efficiency. Defaults to 1000000.
		motherMass (float, optional): Mass of mother particle. If not given, derived from `model`. Defaults to None.
		fsMasses (np.array, optional): Masses of 3 final-state particle. If not given, derived from `model`. Defaults to None.

	Returns:
		np.ndarray: Array of shape [<nWaves>]
	"""
	return calcDecayAmplitudesIntegrals(onlyDiagonal=True,
	                                    normalizedDecayAmplplitudes=False,
	                                    model=model,
	                                    nMCevents=nMCevents,
	                                    nEventsPerBatch=nEventsPerBatch,
	                                    motherMass=motherMass,
	                                    fsMasses=fsMasses)


def calcDecayAmplitudesIntegralMatrix(model: PTo3P,
                                      nMCevents: int,
                                      nEventsPerBatch: int = 1000000,
                                      motherMass: float = None,
                                      fsMasses: np.array = None) -> np.ndarray:
	"""Calculate the integral matrix of the decay amplitudes

	Args:
		model (PTo3P): Amplitude model
		nMCevents (int): Number of phase-events used to calculate the integrals
		nEventsPerBatch (int, optional): Integrals are calculated in batches for cache efficiency. Defaults to 1000000.
		motherMass (float, optional): Mass of mother particle. If not given, derived from `model`. Defaults to None.
		fsMasses (np.array, optional): Masses of 3 final-state particle. If not given, derived from `model`. Defaults to None.

	Returns:
		np.ndarray: Array of shape [<nWaves>,<nWaves>]
	"""
	return calcDecayAmplitudesIntegrals(onlyDiagonal=False,
	                                    normalizedDecayAmplplitudes=True,
	                                    model=model,
	                                    nMCevents=nMCevents,
	                                    nEventsPerBatch=nEventsPerBatch,
	                                    motherMass=motherMass,
	                                    fsMasses=fsMasses)


def calcModelWeights(inputfiles: list,
			             model: PTo3P,
						 couplingAntiparticle: np.ndarray,
						 couplingParticle: np.ndarray) -> eventselection.Variables:
	"""Load the Monte Carlo files and calculate the weights for the files according to given model

	Args:
	    inputfiles: list of input files
		ourmodel: Amplitude model
		couplingAntiparticle: array of antiparticle coupling amplitude which correspond to the model
		couplingParticle: array of particle coupling amplitude which correspond to the model

	Returns:
	    sphysics.eventselection._variables.Variables
	"""
	variables = eventselection.Variables()
	variables.addVariable("experiment", "__experiment__")
	variables.addVariable("run", "__run__")
	variables.addVariable("event", "__event__")
	variables.addVariable("production", "__production__")
	variables.addVariable("nParticlesInBtoKpipi0_MCList")
	for p in ['K', 'pi', 'pi0']:
		variables.addVariable(f'gen_{p}_p', f'{p}_mcP', is4Momentum=True, momentumVariables=['mcE', 'mcPX', 'mcPY', 'mcPZ'])
		variables.addVariable(f'gen_{p}_PDG', f'{p}_mcPDG')
	log.initProgress(len(inputfiles))
	variables.loadFromRootFiles(inputfiles, 'merged','all')
	mask =  variables.nParticlesInBtoKpipi0_MCList == 1
	signal = variables.filter(mask, copy=False)
	del variables
	signal.isSignal = np.ones(signal.nEvents, dtype=int)
	signal.generatorWeights = calculateGeneratorWeightsB0ToKpipi0(
                                 signal.gen_K_p, signal.gen_pi_p, signal.gen_pi0_p
                                 )
	decayAmplitudes = model.calcDecayAmplitudes(signal.gen_K_p, signal.gen_pi_p, signal.gen_pi0_p)
	totalAmplitudeAntiParticle = np.dot(couplingAntiparticle,decayAmplitudes)
	totalAmplitudeParticle = np.dot(couplingParticle,decayAmplitudes)
	signal.modelWeights = np.zeros_like(signal.generatorWeights)
	signal.modelWeights[signal.gen_K_PDG== 321] = math.abs2(totalAmplitudeParticle[    signal.gen_K_PDG== 321])
	signal.modelWeights[signal.gen_K_PDG==-321] = math.abs2(totalAmplitudeAntiParticle[signal.gen_K_PDG==-321])
	signal.reweights = signal.modelWeights/(signal.generatorWeights*np.sum(1./signal.generatorWeights))
	signal.reweights *= signal.nEvents/sum(signal.reweights)
	return signal
