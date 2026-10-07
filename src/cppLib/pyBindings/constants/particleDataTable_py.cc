///////////////////////////////////////////////////////////////////////////
//
//    Based on the boost.python bindings of ROOTPWA
//    (pyInterface/bindings/particleData/particleDataTable_py.cc, mainly written
//    by Karl Bicker), which is distributed under the GNU General Public License
//    version 3.
//
///////////////////////////////////////////////////////////////////////////
//
// Modifications: ported to pybind11 for sphysics (2023-05-04), binding of
// getNames() added (2023-05-12).
//

#include <set>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../../particleData/particleDataTable.h"
#include "utilities/pyhelper.hpp"

namespace py = pybind11;

namespace
{

	bool ParticleDataTable_read(py::object pyLine)
	{
		std::string strLine = py::cast<std::string>(pyLine);
		std::istringstream sstrLine(strLine, std::istringstream::in);
		return sphysics::constants::ParticleDataTable::read(sstrLine);
	}

	py::list ParticleDataTable_entriesMatching(const sphysics::constants::ParticleProperties &prototype,
											   const std::string &sel,
											   const double minMass = 0.,
											   const double minMassWidthFactor = 0.,
											   const std::vector<std::string> &whiteList = {},
											   const std::vector<std::string> &blackList = {},
											   const py::object &pyDecayProducts = py::list(),
											   const bool &forceDecayCheck = true)
	{

		std::multiset<std::string> decayProducts;
		if (not sphysics::py::convertPyObjectToMultiSet<std::string>(pyDecayProducts, decayProducts))
		{
			throw std::invalid_argument("Got invalid input for decayProducts when executing sphysics::constants::ParticleDataTable::entriesMatching()");
		}

		std::vector<const sphysics::constants::ParticleProperties *> retPtrVec = sphysics::constants::ParticleDataTable::entriesMatching(prototype,
																																		 sel,
																																		 minMass,
																																		 minMassWidthFactor,
																																		 whiteList,
																																		 blackList,
																																		 decayProducts,
																																		 forceDecayCheck);
		py::list retVec;
		// std::vector<sphysics::constants::ParticleProperties> retVec(retPtrVec.size());
		for (unsigned int i = 0; i < retPtrVec.size(); ++i)
		{
			// retVec[i] = *(retPtrVec[i]);
			retVec.append(*retPtrVec[i]);
		}

		return retVec;
	}

	py::tuple ParticleDataTable_geantIdAndChargeFromParticleName(const std::string &name)
	{
		int id;
		int charge;
		sphysics::constants::ParticleDataTable::geantIdAndChargeFromParticleName(name, id, charge);
		return py::make_tuple(id, charge);
	}

}

namespace sphysics
{
	namespace py
	{
		void constantsModuleParticleDataTable(pybind11::module_ &m)
		{

			pybind11::class_<sphysics::constants::ParticleDataTable, std::unique_ptr<sphysics::constants::ParticleDataTable, pybind11::nodelete>>(m, "ParticleDataTable")

				.def_property_readonly_static(
					"instance", [](pybind11::object /*self*/)
					{ return &sphysics::constants::ParticleDataTable::instance(); },
					pybind11::return_value_policy::reference_internal)

				.def_static("isInTable", &sphysics::constants::ParticleDataTable::isInTable)

				.def_static(
					"entry", &sphysics::constants::ParticleDataTable::entry, pybind11::arg("partName"), pybind11::arg("warnIfNotExistent") = true, pybind11::return_value_policy::reference_internal)

				.def_static("addEntry", &sphysics::constants::ParticleDataTable::addEntry)

				.def_static(
					"entriesMatching", &ParticleDataTable_entriesMatching, pybind11::arg("prototype"), pybind11::arg("sel"), pybind11::arg("minMass") = 0., pybind11::arg("minMassWidthFactor") = 0., pybind11::arg("whiteList") = pybind11::list(), pybind11::arg("blackList") = pybind11::list(), pybind11::arg("decayProducts") = pybind11::list(), pybind11::arg("forceDecayCheck") = true)

				.def_static("nmbEntries", &sphysics::constants::ParticleDataTable::nmbEntries)

				.def_static(
					"items", []()
					{ return pybind11::make_iterator(sphysics::constants::ParticleDataTable::instance().begin(), sphysics::constants::ParticleDataTable::instance().end()); })

				.def_static(
					"readFile", &sphysics::constants::ParticleDataTable::readFile, pybind11::arg("fileName") = "./ParticleDataTable.txt")
				.def_static("read", &ParticleDataTable_read)

				.def_static("clear", &sphysics::constants::ParticleDataTable::clear)

				.def_static("particleNameFromGeantId", &sphysics::constants::ParticleDataTable::particleNameFromGeantId)

				.def_static("geantIdFromParticleName", &sphysics::constants::ParticleDataTable::geantIdFromParticleName)

				.def_static("geantIdAndChargeFromParticleName", &ParticleDataTable_geantIdAndChargeFromParticleName, pybind11::arg("name"))

				.def_property_static("debugParticleDataTable", [](pybind11::object /*cls*/){return sphysics::constants::ParticleDataTable::debug();},
				                                               [](pybind11::object /*cls*/, const bool debug){return sphysics::constants::ParticleDataTable::setDebug(debug);})
				.def_property_readonly_static("names", [](pybind11::object /*cls*/){return sphysics::constants::ParticleDataTable::getNames();})
				.def_static("keys", &sphysics::constants::ParticleDataTable::getNames)
				.def_static("print", [](){sphysics::constants::ParticleDataTable::print();})
				;
		}
	}
}
