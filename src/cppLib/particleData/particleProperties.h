///////////////////////////////////////////////////////////////////////////
//
//    Copyright 2010
//
//    This file is part of rootpwa
//
//    rootpwa is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    rootpwa is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with rootpwa. If not, see <http://www.gnu.org/licenses/>.
//
///////////////////////////////////////////////////////////////////////////
//-------------------------------------------------------------------------
//
// Description:
//      container class for particle properties
//
//      !NOTE! all potentially half-integer quantum numbers (I, I_z, J)
//             are in units of hbar / 2 so that they are always integer
//
//
// Author List:
//      Boris Grube          TUM            (original author)
//
//
//-------------------------------------------------------------------------
//
// Modifications in sphysics: namespace rpwa replaced by sphysics::constants,
// required helper functions of ROOTPWA's mathUtils.hpp included, ROOTPWA's
// reporting macros replaced by std::cout and std::cerr (2023-05-02), class
// renamed to ParticleProperties (2023-05-04).


#ifndef PARTICLEPROPERTIES_H
#define PARTICLEPROPERTIES_H


#include <string>
#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <sstream>

namespace sphysics {
namespace constants{
	inline
	int
	powMinusOne(const int exponent)  ///< optimized function for (-1)^n
	{
		if (exponent & 0x1)  // exponent is odd
			return -1;
		else                 // exponent is even
			return +1;
	}


	template <typename T>
	inline
	bool
	isOdd(const T val)  ///< returns whether val is an odd number (assuming T is integer type)
	{
		return val & 0x1;
	}


	template <typename T>
	inline
	bool
	isEven(const T val)  ///< returns whether val is an even number (assuming T is integer type)
	{
		return not isOdd(val);
	}
	template<typename T>
	inline
	T signum(const T& val)  ///< extracts sign from value
	{
		if (val < 0)
			return -1;
		if (val > 0)
			return +1;
		return 0;
	}

	inline
	bool
	igjpIsExotic(const int _I,
	             const int G,
	             const int _J,
	             const int P)  ///< checks whether IGJP combination is exotic
	{
		// for baryons and flavoured mesons G is not defined
		if ((P == 0) or (G == 0) or (_I < 0) or isOdd(_I) or (_J < 0) or isOdd(_J))
			return false;
		const int I = _I / 2;  // isospin in units of hbar
		const int J = _J / 2;  // spin in units of hbar
		// quark model restrictions for mesons: P = (-1)^(L + 1), G = (-1)^(I + L + S)
		// for S = 1 is P == (-1)^I * G; each J can be made with odd and even L,
		// except for J = 0, which requires L = 1 -> 0-0- and 1+0- are forbidden
		// for S = 0 is J = L so that P = (-1)^(J + 1) and G = (-1)^(I + 1) * P for all J
		if (J == 0) {
			if ((P == -1) and (G == powMinusOne(I) * P))
				return true;
		}
		if (G == powMinusOne(I + 1) * P and P != powMinusOne(J + 1))
			return true;
		return false;
	}

	inline
	double
	spinQn(const int qn)  ///< convert internal spin quantum number into real quantum number in units of hbar
	{
		return 0.5 * qn;
	}

	inline
	int
	sign(const char s)  ///< converts sign character into int
	{
		if (s == '+')
			return +1;
		if (s == '-')
			return -1;
		return 0;
	}

	template<typename T>
	inline
	std::string
	sign(const T& val)  ///< extracts sign string from a value
	{
		if (val < 0)
			return "-";
		if (val > 0)
			return "+";
		return "0";
	}

	inline
	std::string
	parityQn(const int qn)  ///< convert parity quantum number into string (+, -, or empty string)
	{
		return (qn != 0) ? sign(qn) : "";
	}

	inline
	std::string yesNo(const bool val)  ///< converts bool to "yes"/"no" string
	{
		if (val)
			return "yes";
		else
			return "no";
	}

	class ParticleProperties {

	public:

		// helper class that holds information about particle decay
		struct decayMode {

			decayMode(const std::multiset<std::string>& daughters = std::multiset<std::string>(),
			          const int                         L         = -1,
			          const int                         S         = -1);
			virtual ~decayMode();

			bool operator ==(const decayMode& rhsDecay) const;

			virtual std::ostream& print(std::ostream& out) const;  ///< prints decay mode informayion in human-readable form

			std::multiset<std::string> _daughters;         ///< names of daughter particles
			int                        _L;                 ///< L for two-body decay; -1 means undefined
			int                        _S;                 ///< total spin; -1 means undefined

			std::vector<std::string>   _massDependencies;  ///< list of mass dependencies for this decay (empty list = default mass dependence)

		};

		ParticleProperties();
		ParticleProperties(const ParticleProperties& partProp);
		ParticleProperties(const std::string&        partName,
		                   const int                 isospin,
		                   const int                 G,
		                   const int                 J,
		                   const int                 P,
		                   const int                 C);

		virtual ~ParticleProperties();

		ParticleProperties& operator =(const ParticleProperties& partProp);
		// comparison operators that check equality of all fields
		bool operator ==(const ParticleProperties& rhsProp) const { return this->isEqualTo(rhsProp); }
		bool operator !=(const ParticleProperties& rhsProp) const { return not (*this == rhsProp);   }
		// comparison operators that check equality of fields selectable via string
		friend bool operator ==(const ParticleProperties&                         lhsProp,
		                        const std::pair<ParticleProperties, std::string>& rhsPropSel);
		friend bool operator !=(const ParticleProperties&                         lhsProp,
		                        const std::pair<ParticleProperties, std::string>& rhsPropSel)
		{ return not (lhsProp == rhsPropSel); }

		std::string  name            () const { return nameWithCharge(_name, _charge);          }  ///< returns particle name including charge
		std::string  bareName        () const { return _name;                                   }  ///< returns particle name w/o charge
		std::string  antiPartName    () const { return nameWithCharge(_antiPartName, -_charge); }  ///< returns antiparticle name including charge
		std::string  antiPartBareName() const { return _antiPartName;                           }  ///< returns antiparticle name w/o charge
		int          charge          () const { return _charge;                                 }  ///< returns particle's charge
		double       mass            () const { return _mass;                                   }  ///< returns particle mass
		double       mass2           () const { return _mass2;                                  }  ///< returns particle mass squared
		double       width           () const { return _width;                                  }  ///< returns particle width
		int          baryonNmb       () const { return _baryonNmb;                              }  ///< returns particle's baryon number
		int          isospin         () const { return _isospin;                                }  ///< returns particle's isospin [hbar/2]
		int          isospinProj     () const { return 2 * _charge - (baryonNmb() + strangeness() + charm() + beauty()); }  ///< returns z-component of isospin using Gell-Mann-Nishijima formula (see PDG 2008 eq. 14.1)
		int          strangeness     () const { return _strangeness;                            }  ///< returns particle's strangeness
		int          charm           () const { return _charm;                                  }  ///< returns particle's charm
		int          beauty          () const { return _beauty;                                 }  ///< returns particle's beauty
		int          G               () const { return _G;                                      }  ///< returns particle's G-parity
		int          J               () const { return _J;                                      }  ///< returns particle's spin [hbar/2]
		int          P               () const { return _P;                                      }  ///< returns particle's parity
		int          C               () const { return _C;                                      }  ///< returns particle's C-parity
		unsigned int geantId         () const; ///< returns particle's Geant ID.

		bool isXParticle() const;  ///< returns whether particle's name is either of "X{,-,0,+}"

		bool isMeson () const;  ///< returns whether particle is a meson
		bool isBaryon() const;  ///< returns whether particle is a baryon
		bool isLepton() const;  ///< returns whether particle is a lepton
		bool isPhoton() const;  ///< returns whether particle is a lepton

		bool isItsOwnAntiPart() const;  ///< returns whether particle is its own antiparticle

		bool isSpinExotic() const;  ///< returns whether particle is spin-exotic

		const std::vector<decayMode>& decayModes() const { return _decayModes; }                 ///< returns decay modes defined for this particle
		unsigned int nmbDecays       () const { return _decayModes.size(); }                     ///< returns number of decay modes defined for this particle
		bool         hasDecay        (const decayMode& decay) const;                             ///< returns whether given decay mode is in list of decays
		void         addDecayMode    (const decayMode& decay) { _decayModes.push_back(decay); }  ///< adds decay channel into list of allowed decay modes
		void         deleteDecayModes()                       { _decayModes.clear();          }  ///< deletes all decay modes

		bool isStable() const; ///< returns true if the particle can be considered stable (only the case for pions at this point)

		bool fillFromDataTable(const std::string& name,
		                       const bool         warnIfNotExistent = true);  ///< sets particle properties from entry in particle data table

		void setName        (const std::string& name       );                                                ///< sets particle name and charge (if given in name)
		void setAntiPartName(const std::string& name       ) { _antiPartName = stripChargeFromName(name); }  ///< sets antiparticle name (charge in name is ignored)
		void setCharge      (const int          charge     );                                                ///< sets particle's charge (limited to |q| <= 9)
		void setMass        (const double       mass       );                                                ///< sets particle mass
		void setWidth       (const double       width      ) { _width        = width;                     }  ///< sets particle width
		void setBaryonNmb   (const int          baryonNmb  ) { _baryonNmb    = baryonNmb;                 }  ///< sets particle's baryon number
		void setIsospin     (const int          isospin    ) { _isospin      = abs(isospin);              }  ///< sets particle's isospin [hbar/2]
		void setIsospinProj (const int          isospinProj) { _charge       = (isospinProj + baryonNmb() + strangeness() + charm() + beauty()) / 2; }  ///< sets particle's z component of the isospin [hbar/2]
		void setStrangeness (const int          strangeness) { _strangeness  = strangeness;               }  ///< sets particle's strangeness
		void setCharm       (const int          charm      ) { _charm        = charm;                     }  ///< sets particle's charm
		void setBeauty      (const int          beauty     ) { _beauty       = beauty;                    }  ///< sets particle's beauty
		void setG           (const int          G          ) { _G            = signum(G);                 }  ///< sets particle's G-parity
		void setJ           (const int          J          ) { _J            = abs(J);                    }  ///< sets particle's spin in [hbar/2]
		void setP           (const int          P          ) { _P            = signum(P);                 }  ///< sets particle's parity
		void setC           (const int          C          ) { _C            = signum(C);                 }  ///< sets particle's C-parity

		void setSCB  (const int strangeness,
		              const int charm,
		              const int beauty);  ///< sets particle's strangeness, charm, and beauty
		void setIGJPC(const int isospin,
		              const int G,
		              const int J,
		              const int P,
		              const int C);  ///< sets particle's isospin, G-parity, spin, parity, and C-parity

		ParticleProperties antiPartProperties(const bool convertDecaysModes = false) const;  ///< constructs antiparticle properties from particle

		virtual std::string qnSummary() const;  ///< returns particle's quantum number summary in form name[IG(JPC)]

		// std::string         nameLaTeX     () const;  ///< returns particle name including charge in LaTeX markup
		std::string         bareNameLaTeX () const;  ///< returns particle name w/o charge in LaTeX markup
		// virtual std::string qnSummaryLaTeX() const;  ///< returns particle's quantum number summary in form name[IG(JPC)] in LaTeX markup

		virtual std::ostream& print(std::ostream& out) const;  ///< prints particle data in human-readable form
		virtual std::ostream& dump (std::ostream& out) const;  ///< dumps particle properties into one text line as in data file

		bool read(std::istringstream& line);  ///< reads whitespace separated properties from single line

		static std::string nameWithCharge(const std::string& bareName,
		                                  const int          charge);  ///< appends charge information to bare particle name

		static std::string chargeFromName(const std::string& name,
		                                  int&               charge);  ///< checks whether last characters of particle name contain charge information and sets charge accordingly; returns name with charge stripped off
		static std::string stripChargeFromName(const std::string& name);  ///< checks whether last characters of particle name contain charge information and returns name with charge stripped off

		static bool debug() { return _debug; }                             ///< returns debug flag
		static void setDebug(const bool debug = true) { _debug = debug; }  ///< sets debug flag


	protected:

		virtual bool isEqualTo(const ParticleProperties& partProp) const;  ///< returns whether partProp is equal to this by checking equality of all member variables


	private:

		std::string _name;          ///< full PDG name of particle w/o charge
		std::string _antiPartName;  ///< full PDG name of antiparticle w/o charge
		int         _charge;        ///< charge
		double      _mass;          ///< mass [GeV/c^2]
		double      _mass2;         ///< mass^2 [(GeV/c^2)^2]
		double      _width;         ///< total width [GeV/c^2]
		int         _baryonNmb;     ///< baryon number
		int         _isospin;       ///< isospin [hbar/2]
		int         _strangeness;   ///< strangeness
		int         _charm;         ///< charm
		int         _beauty;        ///< beauty
		int         _G;             ///< G-parity (0 = undefined)
		int         _J;             ///< spin [hbar/2]
		int         _P;             ///< parity (0 = undefined)
		int         _C;             ///< C-parity (0 = undefined)

		std::vector<decayMode> _decayModes; ///< allowed decay modes

		static bool _debug;  ///< if set to true, debug messages are printed

	};  // ParticleProperties


	inline
	std::ostream&
	operator <<(std::ostream&                        out,
	            const ParticleProperties::decayMode& mode)
	{
		return mode.print(out);
	}


	inline
	bool
	ParticleProperties::isXParticle() const
	{
		return ((_name == "X") or (_name == "X-") or (_name == "X0") or (_name == "X+"));
	}


	inline
	bool
	ParticleProperties::isMeson() const
	{
		return (    (abs(charge()) <= 1) and (baryonNmb() == 0)
		        and (isospin() <= 2) and (abs(strangeness()) <= 1) and (abs(charm()) <= 1)
		        and (abs(beauty()) <= 1) and isEven(J()) and (abs(P()) == 1) and (abs(C()) <= 1)
		        and ((abs(charge()) != 0) or (G() == powMinusOne(isospin() / 2) * C())));
	}


	inline
	bool
	ParticleProperties::isBaryon() const
	{
		return (    (abs(charge()) <= 2) and (baryonNmb() == 1)
		        and (isospin() <= 3) and (abs(strangeness()) <= 3) and (abs(charm()) <= 3)
		        and (abs(beauty()) <= 3) and isOdd(J()) and (abs(P()) == 1)
		        and (C() == 0) and (G() == 0));
	}


	inline
	bool
	ParticleProperties::isLepton() const
	{
		return (    (abs(charge()) <= 1) and (baryonNmb() == 0) and (isospin() == 0)
		        and (strangeness() == 0) and (charm() == 0) and (beauty() == 0)
		        and (J() == 1) and (abs(P()) == 1) and (C() == 0) and (G() == 0));
	}


	inline
	bool
	ParticleProperties::isPhoton() const
	{
		return (    (charge() == 0) and (baryonNmb() == 0) and (isospin() == 0)
            and (strangeness() == 0) and (charm() == 0) and (beauty() == 0)
            and (J() == 2) and (P() == -1) and (C() == -1) and (G() == 0));
	}


	inline
	bool
	ParticleProperties::isItsOwnAntiPart() const
	{
		return (    (isMeson() or isPhoton()) and (name() == antiPartName())
		        and (charge() == 0) and (baryonNmb() == 0) and (isospinProj() == 0)
		        and (strangeness() == 0) and (charm() == 0) and (beauty() == 0));
	}


	inline
	void
	ParticleProperties::setName(const std::string& partName)
	{
		int charge;
		_name = chargeFromName(partName, charge);
		setCharge(charge);
	}


	inline
	void
	ParticleProperties::setCharge(const int charge)
	{
		if (abs(charge) < 10)
			_charge = charge;
		else {
			std::cerr << "absolute value of charge " << charge << " is larger than 9. "
			         << "Aborting..." << std::endl;
			throw;
		}
	}


	inline
	void
	ParticleProperties::setMass(const double mass)
	{
		_mass  = mass;
		_mass2 = mass * mass;
	}


	inline
	std::ostream&
	operator <<(std::ostream&             out,
	            const ParticleProperties& partProp)
	{
		return partProp.print(out);
	}


	inline
	std::istream&
	operator >>(std::istream&       in,
	            ParticleProperties& partProp)
	{
		std::string line;
		if (getline(in, line)) {
			// skip comments and empty lines
			while ((line == "") or (line[0] == '#')) {
				if (partProp.debug())
					std::cout << "ignoring line '" << line << "'" << std::endl;
				if (not getline(in, line)) {
					std::cout << "could not find valid particle entry before end of file" << std::endl;
					return in;
				}
			}
			std::istringstream lineStream(line);
			partProp.read(lineStream);
		}
		return in;
	}


}
} // namespace sphysics


#endif  // PARTICLEPROPERTIES_H
