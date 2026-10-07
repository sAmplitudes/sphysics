
/**
 * @file amplitudes.h
 * @author Stefan Wallner (swallner@mpp.mpg.de)
 * @brief Special constants for amplitudes
 * @date 2022-10-31
 * 
 * @copyright Copyright (c) 2022 the sphysics authors (GPL-3.0-or-later, see LICENSE)
 * 
 */
#ifndef SPHYSICS_CONSTANTS_AMPLITUDES_H
#define SPHYSICS_CONSTANTS_AMPLITUDES_H

#include "specialConstants.h"

namespace sphysics {
	namespace constants {
		namespace amplitudes {

			class LASS: public SpecialConstantsBase<LASS> {
			friend class SpecialConstantsBase<LASS>;
			protected:
				LASS(){
					_setName("LASS");
					_setDescription(
R"(Parameters for LASS [Kpi]_S wave parameterization
taken from Dunwoodie (2013) https://www.slac.stanford.edu/~wmd/kpi_swave/kpi_swave_fit.note.
These parameters are the solution of the fit to 34 data poitns of LASS data, which should be
used according to the homepage:
> The solid curves in the first two plots correspond to the 34 point
> fit, with maximum K pi mass 1.52 GeV; the dashed curves represent the
> 37 point fit, with maximum mass 1.60 GeV. The imposition of elastic
> unitarity up to the latter mass value, which is well above K eta'
> threshold, is probably not reasonable, and so the 34 point fit should
> be considered more reliable.
>
> These results should replace those quoted in the original LASS
> publication { D.Aston et al, NPB 296 (1988) 493 }.

These parameters wer used, e.g., in an analysis by LHCb of D-> KKpi
Phys.Rev.D (2016) 052018.)");
					_set("m0", 1.435, "Mass of K_0^*(1430) in units of GeV/c^2");
					_set("g0", 0.279, "Width of K_0^*(1430) in units of GeV/c^2");
					_set("a", 1.95,   "Scattering length for kappa");
					_set("r", 1.76,   "Effective range for kappa");
				}

			};


			class FlatteF0: public SpecialConstantsBase<FlatteF0> {
			friend class SpecialConstantsBase<FlatteF0>;
			protected:
				FlatteF0(){
					_setName("FlatteF0");
					_setDescription(
R"(Parameters for the Flatte parameterization of the f_0(980) resonance.
The parameters are taken from the BESII analysis of J/ψ -> ϕππ and ϕKK decays 
[PLB 607 (2005) 243] (https://doi.org/10.1016/j.physletb.2004.12.041).

The parameters were used in various COMPASS analyses (3pi, Kpipi, ...).)");
					_set("m0", 0.965,  "Mass of f_0(980) in units of GeV/c^2");
					_set("g1", 0.165,  "Coupling constant of f_0(980) to the ππ channel in units of GeV/c^2");
					_set("g2g1", 4.21, "Ratio of coupling to KK to coupling to ππ");
				}

			};
		}
	}
}


#endif