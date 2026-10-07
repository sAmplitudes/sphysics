'''
Created on Tuesday 18 07 2023
Author: Stefan Wallner
Description: Helper functions for kinematics, angles, ....
'''

from __future__ import absolute_import, print_function, division, annotations

from typing import Sequence
import numpy as np

from ... import math
from ... import lorentz

def calcHelicityAngles(p12, p1, pRefPlane):
	''' Calculates the helicity angles of the decay :math:`(12) \\to (1) (2)`

	The angles are calculated in the (12) rest frame.

	The z-axis is defined by the direction of :math:`p_{12}` in the frame in which the momenta are given.
	Therefore, the frame in which :math:`p_{12}`, :math:`p_{1}`, and pRefPlane are given is important and must be different
	from the (12) rest frame.

	The plane w/r/t which the :math:`\\phi` angle is measured is the plane of the z-axis and the pRefPlane direction,

	i.e. y-axis = cross(pRefPlane, z-axis)

	The momentum of (1), :math:`p_1`, is used as spin analyzer.

	:param p12: 4-momentum of subsystem (12), in the shape (4, nEvents), or (4,)
	:type p12: array
	:param p1: 4-momentum of particle 1, in the shape (4, nEvents), or (4,)
	:type p1: array
	:param pRefPlane: 4-momentum defining the reference plane, in the shape (4, nEvents), or (4,)
	:type pRefPlane: array
	:return: (cosTheta, phi), (boost12, pi1_12RF, z12RF); Tuple containing helicity angles and boosted momenta
	'''
	return calcAngles(p12, p1, p12, pRefPlane)


def calcAngles(p12, p1, pZDirection, pRefPlane):
	''' Calculates the decay angles of the decay :math:`(12) \\to (1) (2)`

	The angles are calculated in the (12) rest frame.

	The z-axis is defined by the direction of `pZDirection`, WITHOUT any boost

	The plane w/r/t which the :math:`\\phi` angle is measured is the plane of the z-axis and the pRefPlane direction, WITHOUT any boost,

	i.e. y-axis = cross(pRefPlane, z-axis)

	The momentum of (1), :math:`p_1`, is used as spin analyzer.

	:param p12: 4-momentum of subsystem (12), in the shape (4, nEvents), or (4,)
	:type p12: array
	:param p1: 4-momentum of particle 1, in the shape (4, nEvents), or (4,)
	:type p1: array
	:param pZDirection: 4-momentum which defines the z - direction, in the shape (4, nEvents), or (4,)
	:type pZDirection: array
	:param pRefPlane: 4-momentum defining the reference plane, in the shape (4, nEvents), or (4,)
	:type pRefPlane: array
	:return: (cosTheta, phi), (boost12, pi1_12RF, z12RF)
	'''
	boost12 = lorentz.getBoostToRestFrame(p12)

	pZDirection = pZDirection[1:] if pZDirection.shape[0] == 4 else pZDirection
	z12RF = pZDirection/math.sqrt(math.sum(pZDirection**2, axis=0))
	pRefPlane = pRefPlane[1:] if pRefPlane.shape[0] == 4 else pRefPlane
	y12RF = math.cross(pRefPlane/math.sqrt(math.sum(pRefPlane**2, axis=0)), z12RF, axis=0)
	x12RF = math.cross(y12RF, z12RF, axis=0)

	p1_12RF = math.einsum('ij...,j...->i...', boost12, p1)

	absP1_12RF=math.sqrt(math.sum(p1_12RF[1:]**2, axis=0))

	cosTheta = math.sum(z12RF*p1_12RF[1:], axis=0)/absP1_12RF

	phi = math.arctan2( math.sum(y12RF*p1_12RF[1:], axis=0)/absP1_12RF,
		   math.sum(x12RF*p1_12RF[1:], axis=0)/absP1_12RF)

	return (cosTheta, phi), (boost12, p1_12RF, z12RF)



def calculateIsobarTreeHelicityAngles(
	momenta: Sequence[np.ndarray],
	zRefPlaneOrig: np.ndarray = np.array([0., 0., 0., 1.]),
	momentaInCMS: bool = False
) -> dict[np.ndarray]:
	"""Calculate helicity angles in the isobar decay tree given by the four momenta.

	The four-momenta `momenta` are given in the order :math:`[p_1, p_2, p_3, ... ]`.

	The decay topology is chosen such that in each decay, the n-body system decays to a final state and the (n-1)-body system.
	   - 3-body: :math:`(123)  \\to [ (12) \\to (1) (2) ] (3)`
	   - 4-body: :math:`(1234) \\to [ (123) \\to [ (12) \\to (1) (2) ] (3) ] (4)`

	As spin-analyzer, the remaining (n-1) body system is chosen, e.g.
	   - System :math:`(123) \\to (12) (3): (12)` is the spin analyzer
	   - System :math:`(12) \\to (1) (2): (1)` is the spin analyzer

	The output variables in the dict are named 'cosTheta_<i>__<j>' and 'phi_<i>__<j>', where
		- `<i>` is the index of the spin analyzer
		- `<j>` is the index of the decaying system, i.e. in which rest frame the angles are defined

	The reference plane for the definition of the :math:`\\phi` angle of the top-most decay is given by the momentum of
	the total (123...) system in the frame in which the momenta are given and by `zRefPlaneOrig`.
	`zRefPlaneOrig` could be the quantization axis of the production process of the total (123...) system.

	The definition of the z-axis for the top-most decay is the direction momentum of the total (123...) system in the
	original reference frame. Hence, the reference frame in which the four-momenta in `momenta` are given defines (together with `zRefPlaneOrig`)
	the helicity angles of the top-most decay.
	Accordingly, the four-momenta in `momenta` must not be given in the total (123...) reference frame, otherwise these angles
	cannot be calculated.

	Args:
		momenta (Sequence[np.ndarray]): List of 4-momenta of all final-state particles (shape=(4,nEvents), (E,px,py,pz))
		zRefPlaneOrig (np.ndarray, optional): Direction that defines the reference plane. Defaults to np.array([0., 0., 0., 1.]).
		momentaInCMS (bool, optional):  True if the momenta are given in their CMS. In this case, the helicity angles of the
										decay of the full (1,2,...) system cannot be calculated and `zRefPlanOrig` is used as
										reference plane of the subsystem decays.

	Returns:
		dict[np.ndarray]: Helicilty angles, named as described above.
	"""

	helicityAngles = {}
	momenta = list(momenta)

	pXsystem = math.copy(momenta[0])
	for mom in momenta[1:]:
		pXsystem += mom
	zRefPlane = zRefPlaneOrig



	for i in range(len(momenta) - 1):
		pSpinAnalyzer = math.copy(momenta[0])
		for mom in momenta[1:-1]:
			pSpinAnalyzer += mom
		if i==0 and momentaInCMS: # if momenta are given in CMS, cannot calculate angles because z-direction is not defined
			zXframe = zRefPlane
			boostXframe = lorentz.getBoostToRestFrame(pXsystem)
			pSpinAnalyzer_Xframe = lorentz.applyBoost(boostXframe, pSpinAnalyzer)
		else:
			(cosTheta_Xframe, phi_Xframe), (boostXframe, pSpinAnalyzer_Xframe, zXframe) = calcAngles(pXsystem, pSpinAnalyzer, pXsystem, zRefPlane)
			indexLabel = '_{0}__{1}'.format(
				''.join([str(i + 1) for i, _ in enumerate(momenta[:-1])]),
				''.join([str(i + 1) for i, _ in enumerate(momenta)]))
			helicityAngles[f'cosTheta{indexLabel}'] = cosTheta_Xframe
			helicityAngles[f'phi{indexLabel}'] = phi_Xframe
		momenta.pop()
		if len(momenta) > 1:
			pXsystem = pSpinAnalyzer_Xframe
			zRefPlane = zXframe
			momenta = [
				math.einsum('ij...,j...->i...', boostXframe, mom) for mom in momenta
			]
	return helicityAngles


def calculateIsobarGroupHelicityAngles(
	groupI: tuple[int,...],
	momenta: Sequence[np.ndarray],
	zRefPlaneOrig: np.ndarray = np.array([0., 0., 0., 1.]),
	momentaInCMS: bool = False,
) -> dict[np.ndarray]:
	"""Calculate helicity angles in the isobar decay tree given by the four momenta.

	The four-momenta `momenta` are given in the order [ p1, p2, p3, ... ].

	The  decay products are grouped according to `groupI` , i.e. all momenta corresponding to indices in `groupI` form
	one isobar system, the remaining momentum the other one.

	For example if we have four momenta and `groupI` is (1,3)
	   - 4-body: :math:`(1234) \\to [ (13) ->  (1) (3) ] [ (24) \\to (2) (4) ]`

	The angles in both isobar subsystem decays are calculated using `calculateIsobarTreeHelicityAngles`.

	The output variables in the dict are named 'cosTheta_<i>__<j>' and  'phi_<i>__<j>', where
		- `<i>` is the index of the spin analyzer
		- `<j>` is the index of the decaying system, i.e. in which rest frame the angles are defined

	The reference plane for the definition of the :math:`\\phi` angle of the top-most decay is given by the momentum of
	the total (123...) system in the frame in which the momenta are given and by `zRefPlaneOrig`.
	`zRefPlaneOrig` could be the quantization axis of the production process of the total (123...) system.

	The definition of the z-axis for the top-most decay is the direction momentum of the total (123...) system in the
	original reference frame. Hence, reference frame in which the four-momenta in `momenta` are given defines (together with `zRefPlaneOrig`)
	the helicity angles of the top-most decay.
	Accordingly, the four-momenta in `momenta` must not be given in the total (123...) reference frame, otherwise, this
	angles cannot be calculated.
	If the `momentaInCMS` flag is not set, it is tested if the momenta are given in the CMS to avoid returning wrong values for these angles.

	Some additional remarks for 4-body decays:
	Let's assume we want a decay into two groups (12) and (34).
	In general, as explained above the naming scheme is 'angle_<i>__<j>', where the `angle` is either `phi` or `cosTheta`,
	`i` is the spin analyzer, `j` the restframe in which we calculate the angles.

	E.g., when we choose groupI = (12) the following angles are produced:
		- cosTheta_12__1234 and phi_12__1234; since we need the direction of the momentum of the (1234) system, these are only calculated if the momenta are not given in the CMS.
		- cosTheta_1__12 and phi_1__12; after boosting into the (12) rest frame, we take (1) as spin analyzer
		- cosTheta_3__34 and phi_3__34; since we know that the other group (groupII) has to be (34), we can also calculate the angles of spin analyzer (3) in the (34) rest frame.

	With this it becomes clear that the last two pairs of angles are calculated for a given groupI = (12) and also for a given groupI = (34).
	However they are not the same since for the calculations in groupII we take the reference plane of groupI.

	Only the difference of the :math:`\\phi` angles in groupI and groupII is the same for both choices (i.e. (12) or (34) as groupI),
	where we take :math:`\\phi` of groupI - :math:`\\phi` of groupII,
	e.g. phi_1__12 - phi_3__34 for groupI = (12) or phi_3__34 - phi_1__12 for groupI = (34).

	We provide this in an additional variable; for groupI = (12), this is phi_12_34__1234,
	and for groupI = (34) phi_34_12__1234. Both are the same and are in the range (-pi,pi)

	Args:
		groupI (tuple[int,...]): Indices of momenta belonging to the first isobar group, starting at 1!
		momenta (Sequence[np.ndarray]): List of 4-momenta of all final-state particles (shape=(4,nEvents), (E,px,py,pz))
		zRefPlaneOrig (np.ndarray, optional): Direction that define the reference plane. Defaults to np.array([0., 0., 0., 1.]).
		momentaInCMS (bool, optional):  True if the momenta are given in their CMS. In this case, the helicity angles of the
										decay of the full (1,2,...) system cannot be calculated and `zRefPlanOrig` is used as
										reference plane of the subsystem decays.

	Returns:
		dict[np.ndarray]: Helicilty angles, named as described above.
	"""
	if len(momenta) > 10:
		raise Exception("Cannot handle more than 10 momenta at the moment due to index matching")

	helicityAngles = {}

	# transform to list indices starting at 0
	groupI = tuple(i-1 for i in groupI)

	groupII = []
	for i in range(len(momenta)):
		if i not in groupI:
			groupII.append(i)
	groupII = tuple(groupII)


	pXsystem = math.copy(momenta[0])
	for mom in momenta[1:]:
		pXsystem += mom
	pSpinAnalyzerI= math.copy(momenta[groupI[0]])
	for i in groupI[1:]:
		pSpinAnalyzerI += momenta[i]
	zRefPlane = zRefPlaneOrig


	if not momentaInCMS and max(np.concatenate(np.sum(momenta,axis=0)[1:])) < 1e-10: # check whether the momenta are really not in CMS
		raise Exception("The given momenta appear to be in CMS, please set the momentaInCMS flag. This is crucial for correct calculations.")

	if not momentaInCMS: # angles of X decay are only well define with the momenta are not given in their CMS, i.e. in the X rest frame
		(cosTheta_Xframe, phi_Xframe), (boostXframe, _, zXframe) = calcAngles(pXsystem, pSpinAnalyzerI, pXsystem, zRefPlane)
		indexLabel = '_{0}__{1}'.format(
			''.join([str(i + 1) for i in groupI]),
			''.join([str(i + 1) for i, _ in enumerate(momenta)]))
		helicityAngles[f'cosTheta{indexLabel}'] = cosTheta_Xframe
		helicityAngles[f'phi{indexLabel}'] = phi_Xframe
		zRefPlane = zXframe
	else:
		boostXframe = lorentz.getBoostToRestFrame(pXsystem)



	if len(groupI) > 1:
		momentaI_Xsystem = [
			math.einsum('ij...,j...->i...', boostXframe, momenta[i]) for i in groupI
		]
		subanglesI = calculateIsobarTreeHelicityAngles(momentaI_Xsystem, zRefPlane)
		mapI = {i+1: j+1 for i, j in enumerate(groupI)}
		for labelSubangles in subanglesI:
			label = "".join([str(mapI[int(l)]) if l.isdigit() else l for l in labelSubangles])
			if label in helicityAngles:
				raise Exception(f"Label {label} already in angles")
			helicityAngles[label] = subanglesI[labelSubangles]

	if len(groupII) > 1:
		momentaII_Xsystem = [
			math.einsum('ij...,j...->i...', boostXframe, momenta[i]) for i in groupII
		]
		# In principle we have two options to determine the phi angle between the groupI and groupII decay planes
		# We either invert the 3-momenta of all particles in system II and then take phi as the difference of the individual phi angles
		# or we do not invert the momenta and take phi as the sum of the individual phi angles. Both options are equivalent and give the same result.
		# We use the second option here, i.e. we do not invert the momenta of system II. This is also the choice made in most LHCb analyses.
		# Both choices leave the relative angles in this subsystem unchanged.

		subanglesII = calculateIsobarTreeHelicityAngles(momentaII_Xsystem, zRefPlane)
		mapII = {i+1: j+1 for i, j in enumerate(groupII)}
		for labelSubangles in subanglesII:
			label = "".join([str(mapII[int(l)]) if l.isdigit() else l for l in labelSubangles])
			if label in helicityAngles:
				raise Exception(f"Label {label} already in angles")
			helicityAngles[label] = subanglesII[labelSubangles]

	if len(groupI) > 1 and len(groupII) > 1: # Add here the angle between the two decay planes
		phiTotLabel = 'phi_{0}_{1}__{2}'.format(
			''.join([str(i + 1) for i in groupI]),
			''.join([str(i + 1) for i in groupII]),
			''.join([str(i + 1) for i, _ in enumerate(momenta)]))
		firstTopLabel = 'phi_{0}__{1}'.format(
			''.join([str(i + 1) for i in groupI[:-1]]),
			''.join([str(i + 1) for i in groupI]))
		secondTopLabel = 'phi_{0}__{1}'.format(
			''.join([str(i + 1) for i in groupII[:-1]]),
			''.join([str(i + 1) for i in groupII]))

		# Here, we take the sum as explained above, since the phi angles for both subsystems have inverted y-axes, i.e. the phi angles are defined in opposite directions.
		# Therefore, we need to add them to get the angle between the two decay planes.
		phiTot = helicityAngles[firstTopLabel]+helicityAngles[secondTopLabel]
		phiTot[phiTot<-np.pi] += 2*np.pi
		phiTot[phiTot>np.pi] -= 2*np.pi
		helicityAngles[phiTotLabel] = phiTot

	return helicityAngles
def calculateHelicityAngles2x2Topology(
	momenta: Sequence[np.ndarray],
	zRefPlaneOrig: np.ndarray = np.array([0., 0., 0., 1.]),
	momentaInCMS: bool = False,
) -> dict[np.ndarray]:
	"""Calculates the three helicity angles typically needed for modeling the 4-body decay of a spin 0 particle in the 2x2 topology,
	i.e. the phi angle between the two decay planes of groups (12) and (34) `phi_12_34__1234`
	and the cosTheta value of (1) in the system (12) `cosTheta_1__12` and of (3) in the system (34) `cosTheta_3__34`.
	Further information can be found in the description of `calculateIsobarGroupHelicityAngles`.

	Args:
		momenta (Sequence[np.ndarray]): List of 4-momenta of all final-state particles (shape=(4,nEvents), (E,px,py,pz))
		zRefPlaneOrig (np.ndarray, optional): Direction that define the reference plane. Defaults to np.array([0., 0., 0., 1.]).
		momentaInCMS (bool, optional):  True if the momenta are given in their CMS. In this case, the helicity angles of the
										decay of the full (1,2,...) system cannot be calculated and `zRefPlanOrig` is used as
										reference plane of the subsystem decays.

	Returns:
		dict[np.ndarray]: Helicilty angles, named as described above.
	"""

	helicityAngles2x2 = calculateIsobarGroupHelicityAngles((1,2),momenta,zRefPlaneOrig,momentaInCMS)
	if not momentaInCMS:
		helicityAngles2x2.pop('cosTheta_12__1234')
		helicityAngles2x2.pop('phi_12__1234')
	helicityAngles2x2.pop('phi_1__12')
	helicityAngles2x2.pop('phi_3__34')

	return helicityAngles2x2
