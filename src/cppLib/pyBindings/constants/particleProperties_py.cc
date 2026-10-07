///////////////////////////////////////////////////////////////////////////
//
//    Based on the boost.python bindings of ROOTPWA
//    (pyInterface/bindings/particleData/particleProperties_py.cc, mainly
//    written by Karl Bicker), which is distributed under the GNU General Public
//    License version 3.
//
///////////////////////////////////////////////////////////////////////////
//
// Modifications: ported to pybind11 for sphysics (2023-05-04).
//



#include <stdexcept>
#include <pybind11/pybind11.h>

#include "../../particleData/particleProperties.h"
#include "utilities/pyhelper.hpp"

namespace py = pybind11;

namespace
{

	bool ParticleProperties_equal(const sphysics::constants::ParticleProperties &self, const py::object &rhsObj)
	{
		if (py::isinstance<sphysics::constants::ParticleProperties>(rhsObj))
		{
			return (self == py::cast<sphysics::constants::ParticleProperties>(rhsObj));
		}
		try
		{
			const auto rhsPair = py::cast<std::pair<sphysics::constants::ParticleProperties, std::string>>(rhsObj);
			return (self == rhsPair);
		}
		catch (py::cast_error &)
		{
			throw std::invalid_argument("Got invalid input when executing sphysics::constants::ParticleProperties::operator==()");
		}
	}

	bool ParticleProperties_nequal(const sphysics::constants::ParticleProperties &self, const py::object &rhsObj)
	{
		return not ParticleProperties_equal(self, rhsObj);
	}

	bool
	ParticleProperties_hasDecay(const sphysics::constants::ParticleProperties &self, py::object &pyDaughters)
	{
		std::multiset<std::string> daughters;
		if (not sphysics::py::convertPyObjectToMultiSet<std::string>(pyDaughters, daughters))
		{
			throw std::invalid_argument("Got invalid input for daughters when executing sphysics::constants::ParticleProperties::hasDecay()");
		}
		return self.hasDecay(daughters);
	}

	void ParticleProperties_addDecayMode(sphysics::constants::ParticleProperties &self, py::object &pyDaughters)
	{
		std::multiset<std::string> daughters;
		if (not sphysics::py::convertPyObjectToMultiSet<std::string>(pyDaughters, daughters))
		{
			throw std::invalid_argument("Got invalid input for daughters when executing sphysics::constants::ParticleProperties::addDecayMode()");
		}
		self.addDecayMode(daughters);
	}

	bool ParticleProperties_read(sphysics::constants::ParticleProperties &self, py::object &pyLine)
	{
		std::string strLine = py::cast<std::string>(pyLine);
		std::istringstream sstrLine(strLine, std::istringstream::in);
		return self.read(sstrLine);
	}

	py::tuple ParticleProperties_chargeFromName(const std::string &name)
	{
		int charge;
		std::string newName = sphysics::constants::ParticleProperties::chargeFromName(name, charge);
		return py::make_tuple(newName, charge);
	}

}

namespace sphysics
{
	namespace py
	{
		void constantsModuleParticleProperties(pybind11::module_ &m)
		{
			pybind11::class_<sphysics::constants::ParticleProperties>(m, "ParticleProperties")

				.def(pybind11::init<const sphysics::constants::ParticleProperties &>())
				.def(pybind11::init<std::string, int, int, int, int, int>())

				.def("__eq__", &ParticleProperties_equal)
				.def("__neq__", &ParticleProperties_nequal)

				.def_property("name", &sphysics::constants::ParticleProperties::name, &sphysics::constants::ParticleProperties::setName)
				.def_property("antiPartName", &sphysics::constants::ParticleProperties::antiPartName, &sphysics::constants::ParticleProperties::setAntiPartName)
				.def_property("charge", &sphysics::constants::ParticleProperties::charge, &sphysics::constants::ParticleProperties::setCharge)
				.def_property("mass", &sphysics::constants::ParticleProperties::mass, &sphysics::constants::ParticleProperties::setMass)
				.def_property("width", &sphysics::constants::ParticleProperties::width, &sphysics::constants::ParticleProperties::setWidth)
				.def_property("baryonNmb", &sphysics::constants::ParticleProperties::baryonNmb, &sphysics::constants::ParticleProperties::setBaryonNmb)
				.def_property("isospin", &sphysics::constants::ParticleProperties::isospin, &sphysics::constants::ParticleProperties::setIsospin)
				.def_property("isospinProj", &sphysics::constants::ParticleProperties::isospinProj, &sphysics::constants::ParticleProperties::setIsospinProj)
				.def_property("strangeness", &sphysics::constants::ParticleProperties::strangeness, &sphysics::constants::ParticleProperties::setStrangeness)
				.def_property("charm", &sphysics::constants::ParticleProperties::charm, &sphysics::constants::ParticleProperties::setCharm)
				.def_property("beauty", &sphysics::constants::ParticleProperties::beauty, &sphysics::constants::ParticleProperties::setBeauty)
				.def_property("G", &sphysics::constants::ParticleProperties::G, &sphysics::constants::ParticleProperties::setG)
				.def_property("J", &sphysics::constants::ParticleProperties::J, &sphysics::constants::ParticleProperties::setJ)
				.def_property("P", &sphysics::constants::ParticleProperties::P, &sphysics::constants::ParticleProperties::setP)
				.def_property("C", &sphysics::constants::ParticleProperties::C, &sphysics::constants::ParticleProperties::setC)
				.def_property_readonly("bareName", &sphysics::constants::ParticleProperties::bareName)
				.def_property_readonly("antiPartBareName", &sphysics::constants::ParticleProperties::antiPartBareName)
				.def_property_readonly("mass2", &sphysics::constants::ParticleProperties::mass2)
				.def_property_readonly("isXParticle", &sphysics::constants::ParticleProperties::isXParticle)
				.def_property_readonly("isMeson", &sphysics::constants::ParticleProperties::isMeson)
				.def_property_readonly("isBaryon", &sphysics::constants::ParticleProperties::isBaryon)
				.def_property_readonly("isLepton", &sphysics::constants::ParticleProperties::isLepton)
				.def_property_readonly("isPhoton", &sphysics::constants::ParticleProperties::isPhoton)
				.def_property_readonly("isItsOwnAntiPart", &sphysics::constants::ParticleProperties::isItsOwnAntiPart)
				.def_property_readonly("isSpinExotic", &sphysics::constants::ParticleProperties::isSpinExotic)
				.def_property_readonly("isStable", &sphysics::constants::ParticleProperties::isStable)
				.def_property_readonly("nmbDecays", &sphysics::constants::ParticleProperties::nmbDecays)

				.def(
					"fillFromDataTable", &sphysics::constants::ParticleProperties::fillFromDataTable, pybind11::arg("name"), pybind11::arg("warnIfNotExistent") = true)

				.def("hasDecay", &ParticleProperties_hasDecay)
				.def("addDecayMode", &ParticleProperties_addDecayMode)

				.def("setSCB", &sphysics::constants::ParticleProperties::setSCB)
				.def("setIGJPC", &sphysics::constants::ParticleProperties::setIGJPC)

				.def_property_readonly("antiPartProperties", &sphysics::constants::ParticleProperties::antiPartProperties)
				.def("qnSummary", &sphysics::constants::ParticleProperties::qnSummary)
				.def_property_readonly("bareNameLaTeX", &sphysics::constants::ParticleProperties::bareNameLaTeX)
				.def("read", &ParticleProperties_read)

				.def_static("nameWithCharge", &sphysics::constants::ParticleProperties::nameWithCharge)
				.def_static("chargeFromName", &ParticleProperties_chargeFromName)
				.def_static("stripChargeFromName", &sphysics::constants::ParticleProperties::stripChargeFromName)

				.def_property_static("debugParticleProperties", &sphysics::constants::ParticleProperties::debug, &sphysics::constants::ParticleProperties::setDebug);
		}
	}
}
