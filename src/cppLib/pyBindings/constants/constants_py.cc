

#include <algorithm>

#include "../../constants/specialConstants.h"
#include "../../constants/amplitudes.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;


namespace {


template<typename T>
py::object
get(const T&, const std::string& key)
{
	if (T::template hasKeyOfType<double>(key)) return py::cast(T::template get<double>(key));
	if (T::template hasKeyOfType<int>(key)) return py::cast(T::template get<int>(key));
	if (T::template hasKeyOfType<std::string>(key)) return py::cast(T::template get<std::string>(key));
	throw std::invalid_argument("Key " + key + " not in constants!");
}


template<typename T>
std::string
getConstantDescription(const T&, const std::string& key)
{
	return T::getConstantDescription(key);
}

template<typename T>
std::string
getDescription(const T&)
{
	return T::getDescription();
}


template <typename T>
py::dict
getConstants(const T& self)
{
	py::dict allConstantsPy;
	for(const auto& key: T::keys()){
		allConstantsPy[py::str(key)] = get<T>(self, key);
	}
	return allConstantsPy;
}

template <typename T>
std::string
toString(const T& )
{
	return T::toString();
}

template<typename T>
const T*
instance()
{
	return &T::getInstance();
}

template<typename T, typename T2>
void
exportWrapper(T2 m, const char* name)
{
	py::class_<T>(m, name)
	.def_static("getInstance", &instance<T>, py::return_value_policy::reference)
	.def("get", &get<T>)
	.def("__getattr__", &get<T>)
	.def("getConstants", &getConstants<T>)
	.def("getConstantDescription", &getConstantDescription<T>)
	.def("getDescription", &getDescription<T>)
	.def("toString", &toString<T>)
	.def("__str__", &toString<T>)
	.def("__repr__", &toString<T>)
	;
}

}


namespace sphysics {
	namespace py {
		void constantsModuleParticleProperties(pybind11::module_& m);
		void constantsModuleParticleDataTable(pybind11::module_& m);


		void constantsModule(pybind11::module_& m){
			{
				auto subModule = m.def_submodule("amplitudes", "Amplitude constants");
				exportWrapper<sphysics::constants::amplitudes::LASS>(subModule, "LASS");
				exportWrapper<sphysics::constants::amplitudes::FlatteF0>(subModule, "FlatteF0");

				constantsModuleParticleProperties(m);
				constantsModuleParticleDataTable(m);
			}
		}
	}
}
// PYBIND11_MODULE(_sphysics_constants, m) { }
