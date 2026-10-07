
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../../utilities/environment.h"

namespace sphysics
{
	namespace py
	{

		pybind11::tuple
		environment_getstate(const sphysics::Environment &self)
		{
			pybind11::tuple state = pybind11::make_tuple(self.getVersion(), self.getGitHash(), self.getGitStatus(), self.getLibraryVersions());
			return state;
		}

		sphysics::Environment
		environment_setstate(const pybind11::tuple &state)
		{
			sphysics::Environment self;
			if (pybind11::len(state) == 3){
				self.set("unknown",
						 pybind11::cast<std::string>(state[0]),
						 pybind11::cast<std::vector<std::string>>(state[1]),
						 pybind11::cast<std::map<std::string, std::string>>(state[2]));
			} else {
				self.set(pybind11::cast<std::string>(state[0]),
						 pybind11::cast<std::string>(state[1]),
						 pybind11::cast<std::vector<std::string>>(state[2]),
						 pybind11::cast<std::map<std::string, std::string>>(state[3]));
			}
			return self;
		}
	}
}

namespace sphysics {
	namespace py {
		void utilitiesModule(pybind11::module_& m)
		{

			pybind11::class_<sphysics::Environment, std::shared_ptr<sphysics::Environment>>(m, "Environment")
				.def(pybind11::init<>())
				.def("getVersion", &sphysics::Environment::getVersion)
				.def("getGitHash", &sphysics::Environment::getGitHash)
				.def("isGitStatusClean", &sphysics::Environment::isGitStatusClean)
				.def("getGitStatus", &sphysics::Environment::getGitStatus)
				.def("getLibraryVersions", &sphysics::Environment::getLibraryVersions)
				.def("getSummary", &sphysics::Environment::getSummary)
				.def(pybind11::pickle(&sphysics::py::environment_getstate, &sphysics::py::environment_setstate));
		}
	}
}
