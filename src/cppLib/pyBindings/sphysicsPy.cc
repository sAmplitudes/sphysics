
#include <pybind11/pybind11.h>

namespace py = pybind11;



namespace sphysics {
	namespace py {
		void constantsModule(pybind11::module_& m);
		void pwaModule(pybind11::module_& m);
		void utilitiesModule(pybind11::module_& m);
	}
}


PYBIND11_MODULE(_core, m) {
		{
			auto mSub = m.def_submodule("constants", "Constants collection");
			sphysics::py::constantsModule(mSub);
		}
		{
			auto mSub = m.def_submodule("pwa", "PWA utilities");
			sphysics::py::pwaModule(mSub);
		}
		{
			auto mSub = m.def_submodule("utilities", "Collection of various utilities");
			sphysics::py::utilitiesModule(mSub);
		}
}


// namespace sphysics {
// 	namespace py {
// 		void
// 		initialize()
// 		{
// 			boost::python::numpy::initialize();
// 		}
// 	}
// }


// BOOST_PYTHON_MODULE(libSphysicsPy){

// 	sphysics::py::initialize();
// 	sphysics::py::exportEnvironment();
// 	sphysics::py::exportNumpy();
// 	sphysics::py::exportPwaFitresult();
// 	sphysics::py::exportConstants();
// }
