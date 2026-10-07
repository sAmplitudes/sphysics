/**
 * @file numpy.hpp
 * @author Stefan Wallner (swallner@mpp.mpg.de)
 * @brief  Helper functions for numpy
 * @date 2022-05-02
 * 
 * @copyright Copyright (c) 2022 the sphysics authors (GPL-3.0-or-later, see LICENSE)
 * 
 */

#include <vector>
#include <stdexcept>
#include <algorithm>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace sphysics{
	namespace py {

		/**
		 * @brief Convert STL vector to linear numpy array
		 * 
		 * @tparam T element type
		 * @param vec  input array
		 * @return boost::python::numpybind11::ndarray Output array (flat)
		 */
		template <typename T>
		pybind11::array_t<T>	
		vector2ndarray(const std::vector<T>& vec)
		{
			pybind11::array_t<T> vecPy(vec.size());
			auto vecPyAccess = vecPy.mutable_unchecked();
			for(size_t i=0; i < vec.size(); ++i) vecPyAccess(i) = vec[i];
			return vecPy;
		}


		/**
		 * @brief Convert flat numpy array to STL vector with checks
		 * 
		 * @tparam T element type
		 * @param vecPy  input numpy array
		 * @return std::vector<T> 
		 */
		template <typename T>
		std::vector<T>
		ndarray2vector(const pybind11::array_t<T>& vecPy)
		{
			if ( (vecPy.ndim() != 1) ) {
				throw std::invalid_argument("`nparray2vector` can convert only flat arrays!");
			}
			auto vecPyAccess = vecPy.unchecked();
			std::vector<T> vec;
			vec.reserve(vecPy.shape(0));
			for(int i=0; i < vecPy.shape(0); ++i){
				vec.push_back(vecPyAccess(i));
			}
			return vec;
		}


		// template <typename T>
		// boost::python::numpybind11::ndarray
		// boostmatrix2ndarray(const boost::numeric::ublas::matrix<T, boost::numeric::ublas::row_major>& matrix)
		// {
		// 	boost::python::numpybind11::dtype dt = boost::python::numpybind11::dtype::get_builtin<T>();
		// 	const auto shape = boost::python::make_tuple(matrix.size1(), matrix.size2());
		// 	const auto stride = boost::python::make_tuple(sizeof(T)*matrix.size2(), sizeof(T));
		// 	boost::python::object own;
		// 	boost::python::numpybind11::ndarray matrixPy = boost::python::numpybind11::from_data(&matrix(0,0), dt, shape, stride, own);
		// 	return matrixPy;
		// }

		// template <typename T>
		// boost::numeric::ublas::matrix<T, boost::numeric::ublas::row_major>
		// ndarray2boostmatrix(const boost::python::numpybind11::ndarray& matrixPy)
		// {
		// 	if ( (matrixPy.get_nd() != 2) ) {
		// 		throw std::invalid_argument("`ndarray2boostmatrix` can convert only 2D arrays!");
		// 	}
		// 	boost::numeric::ublas::matrix<T, boost::numeric::ublas::row_major> matrix(matrixPy.shape(0), matrixPy.shape(1));
		// 	for(size_t i=0; i < matrix.size1(); ++i){
		// 		for(size_t j=0; j < matrix.size2(); ++j){
		// 			matrix(i,j) = boost::python::extract<T>(matrixPy[boost::python::make_tuple(i,j)])();
		// 		}
		// 	}
		// 	return matrix;
		// }

	}
}
