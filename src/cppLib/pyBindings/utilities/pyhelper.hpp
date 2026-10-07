/**
 * @file numpy.hpp
 * @author Stefan Wallner (swallner@mpp.mpg.de)
 * @brief  Helper functions for numpy
 * @date 2022-05-02
 * 
 * @copyright Copyright (c) 2022 the sphysics authors (GPL-3.0-or-later, see LICENSE)
 * 
 */

#include <set>
#include <pybind11/pybind11.h>

namespace sphysics {
	namespace py {

		template<typename T>
		bool convertPyObjectToMultiSet(const pybind11::object& pyList, std::multiset<T>& multiset) {
			if(not pybind11::isinstance<pybind11::list>(pyList)) {
				std::cout<<"cannot convert pybind11::object to list."<<std::endl;
				return false;
			}
			pybind11::list pyListList = pybind11::cast<pybind11::list>(pyList);
			for(size_t i = 0; i < pybind11::len(pyListList); ++i) {
				try {
					const auto item = pybind11::cast<T>(pyListList[i]);
					multiset.insert(item);
				} catch (pybind11::cast_error&){
					std::cout<<"cannot convert list item " << i << " ."<<std::endl;
					return false;
				}
			}
			return true;
		}

	}
}