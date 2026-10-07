
#include <complex>
#include <algorithm>
#include <iostream>

#include "../../pwa/fitresult.h"

#include "utilities/numpy.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

namespace {

//////////////////////////////////////////////////////////////////////////////////////////////////
////// FitResult /////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
	/**
	 * @brief Constructor
	 * 
	 * @param couplingsPy 
	 * @param covMatrixPy 
	 * @param covMatrixCouplingIndicesPy 
	 * @param negLogLikelihood 
	 * @param fitConverged 
	 * @param covMatrixValid 
	 * @param covMatrixMadePosDef 
	 * @param resultObjPy 
	 * @param auxiliaryParameters
	 * @param auxiliaryParameterNames
	 * @return boost::shared_ptr<sphysics::pwa::FitResult> 
	 */
	std::shared_ptr<sphysics::pwa::FitResult>
	fitresult_constructor(const py::array_t<std::complex<double>>& couplingsPy,
	                      const py::array_t<double>& covMatrixPy,
						  const double negLogLikelihood,
						  const bool fitConverged,
						  const bool covMatrixValid,
						  const bool covMatrixMadePosDef,
						  const py::bytes& resultObjPy,
						  const std::vector<double>& auxiliaryParameters,
						  const std::vector<std::string>& auxiliaryParameterNames
						  )
	{
		const auto couplings = sphysics::py::ndarray2vector<std::complex<double>>(couplingsPy);

		sphysics::pwa::CovarianceMatrix covMatrix(static_cast<size_t>(covMatrixPy.shape(0)), static_cast<size_t>(covMatrixPy.shape(1)));
		auto covMatrixPyAccess = covMatrixPy.unchecked<2>();
		for(int i=0; i < covMatrixPy.shape(0); ++i){
			for(int j=0; j < covMatrixPy.shape(1); ++j){
				covMatrix(i,j) = covMatrixPyAccess(i,j);
			}
		}

		const auto resultObjStr = py::cast<std::string>(resultObjPy);
		std::vector<char> resultObj;
		for(size_t i=0; i < resultObjStr.size(); ++i) resultObj.push_back(resultObjStr[i]);
		auto result = std::shared_ptr<sphysics::pwa::FitResult>(
			new sphysics::pwa::FitResult(couplings, covMatrix, auxiliaryParameters, auxiliaryParameterNames, negLogLikelihood, fitConverged, covMatrixValid, covMatrixMadePosDef, resultObj)
		);
		return result;
	}

	py::array_t<std::complex<double>>
	fitresult_getcouplings(const sphysics::pwa::FitResult& fitResult){
		return sphysics::py::vector2ndarray(fitResult.getCouplings());
	}

	py::array_t<double>
	fitresult_getcovmatrix(const sphysics::pwa::FitResult& fitResult){
		py::array_t<double> covMatrixPy({fitResult.getCovarianceMatrix().rows(), fitResult.getCovarianceMatrix().cols()});
		for(int i=0; i < covMatrixPy.shape(0); ++i){
			for(int j=0; j < covMatrixPy.shape(1); ++j){
				covMatrixPy.mutable_at(i,j) = fitResult.getCovarianceMatrix()(i,j);
			}
		}
		return covMatrixPy;
	}

	py::bytes
	fitresult_getresultobj(const sphysics::pwa::FitResult& fitResult){
		const auto& resultObj = fitResult.getResultObj();
		std::string resultObjStr(resultObj.size(), '0');
		for (size_t i=0; i < resultObj.size(); ++i) resultObjStr[i] = resultObj[i];
		return py::bytes(resultObjStr);
	}


	py::tuple
	fitresult_getstate(const sphysics::pwa::FitResult& self)
	{
		return py::make_tuple(fitresult_getcouplings(self),
								fitresult_getcovmatrix(self),
								self.getNegLogLikelihood(),
								self.getFitConverged(),
								self.getCovarianceMatrixValid(),
								self.getCovarianceMatrixMadePosDef(),
								fitresult_getresultobj(self),
								self.getauxiliaryParameters(),
								self.getauxiliaryParameterNames()
								);
	}

	std::shared_ptr<sphysics::pwa::FitResult>
	fitresult_setstate(const py::tuple& t)
	{
		// handle old version (aux in 3rd/4th place)
		if (t.size() == 9 && !py::isinstance<py::float_>(t[2])) {

			// old layout: (couplings, covMatrix, aux, auxNames, nll, conv, valid, posdef, resultObj)
			return fitresult_constructor(
				py::cast<py::array_t<std::complex<double>>>(t[0]),
				py::cast<py::array_t<double>>(t[1]),
				py::cast<double>(t[4]),
				py::cast<bool>(t[5]),
				py::cast<bool>(t[6]),
				py::cast<bool>(t[7]),
				py::cast<py::bytes>(t[8]),
				py::cast<std::vector<double>>(t[2]),
				py::cast<std::vector<std::string>>(t[3])
			);
		}

			// default layout: (couplings, covMatrix, nll, conv, valid, posdef, resultObj, aux, auxNames)
			return fitresult_constructor(
			py::cast<py::array_t<std::complex<double>>>(t[0]),
			py::cast<py::array_t<double>>(t[1]),
			py::cast<double>(t[2]),
			py::cast<bool>(t[3]),
			py::cast<bool>(t[4]),
			py::cast<bool>(t[5]),
			t[6],
			(t.size() > 7) ? py::cast<std::vector<double>>(t[7]) : std::vector<double>{},
			(t.size() > 8) ? py::cast<std::vector<std::string>>(t[8]) : std::vector<std::string>{}
		);
	}

//////////////////////////////////////////////////////////////////////////////////////////////////
////// ResultCollection //////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
	std::shared_ptr<sphysics::pwa::ResultCollection>
	resultcollection_constructor(
	                              const std::string& modelName,
								  const std::string& modelDescription,
								  const py::bytes& modelObjPy,
								  const sphysics::pwa::IntegralMatrix& integralMatrixGen,
								  const sphysics::pwa::IntegralMatrix& integralMatrixReco,
								  const sphysics::pwa::CouplingIndices& indicesCouplings,
								  const sphysics::pwa::CovarianceMatrixCouplingIndices& covarianceMatrixCouplingIndices,
								  const py::list& fitResultsPy,
								  const std::string& label,
								  const sphysics::Environment& environment
						  )
	{

		const auto modelObjStr = py::cast<std::string>(modelObjPy);
		std::vector<char> modelObj;
		for(size_t i=0; i < modelObjStr.size(); ++i) modelObj.push_back(modelObjStr[i]);

		// sphysics::pwa::CouplingIndices indicesCouplings;
		// {
		// 	const auto keys = py::list(indicesCouplingsPy.keys());
		// 	const size_t nKeys = py::len(keys);
		// 	for(size_t i=0; i < nKeys; ++i){
		// 		for(size_t j=0; j < py::len(indicesCouplingsPy[keys[i]]); ++j){
		// 			indicesCouplings[py::cast<std::string>(keys[i])].push_back(py::cast<size_t>(indicesCouplingsPy[keys[i]][j]));
		// 		}
		// 	}
		// }

		// sphysics::pwa::CovarianceMatrixCouplingIndices covarianceMatrixCouplingIndices;
		// covarianceMatrixCouplingIndices.reserve(py::len(covarianceMatrixCouplingIndicesPy));
		// for (size_t i=0; i < py::len(covarianceMatrixCouplingIndicesPy); ++i){
		// 	covarianceMatrixCouplingIndices.push_back(std::make_pair(py::cast<size_t>(covarianceMatrixCouplingIndicesPy[i][0]), py::cast<size_t>(covarianceMatrixCouplingIndicesPy[i][1])));
		// }

		std::vector<sphysics::pwa::FitResult> fitResults;
		fitResults.reserve(py::len(fitResultsPy));
		for (size_t i=0; i < py::len(fitResultsPy); ++i){
			fitResults.push_back(*py::cast<sphysics::pwa::FitResult*>(fitResultsPy[i]));
		}

		return std::shared_ptr<sphysics::pwa::ResultCollection>(
			new sphysics::pwa::ResultCollection(modelName, modelDescription, modelObj, integralMatrixGen, integralMatrixReco, indicesCouplings, covarianceMatrixCouplingIndices, fitResults, label, environment)
		);
	}

	py::array_t<std::complex<double>>
	resultcollection_getintegralmatrixgen(const sphysics::pwa::ResultCollection& self){
		py::array_t<std::complex<double>> intMatrixPy({self.getIntegralMatrixGen().rows(), self.getIntegralMatrixGen().cols()});
		for(int i=0; i < intMatrixPy.shape(0); ++i){
			for(int j=0; j < intMatrixPy.shape(1); ++j){
				intMatrixPy.mutable_at(i,j) = self.getIntegralMatrixGen()(i,j);
			}
		}
		return intMatrixPy;
	}

	py::array_t<std::complex<double>>
	resultcollection_getintegralmatrixreco(const sphysics::pwa::ResultCollection& self){
		py::array_t<std::complex<double>> intMatrixPy({self.getIntegralMatrixReco().rows(), self.getIntegralMatrixReco().cols()});
		for(int i=0; i < intMatrixPy.shape(0); ++i){
			for(int j=0; j < intMatrixPy.shape(1); ++j){
				intMatrixPy.mutable_at(i,j) = self.getIntegralMatrixReco()(i,j);
			}
		}
		return intMatrixPy;
	}

	py::dict
	resultcollection_getindicescouplings(const sphysics::pwa::ResultCollection& self){
		py::dict indicesCouplingsPy;
		for(const auto& nameAndIndices: self.getIndicesCouplings()){
			py::list indicesPy;
			for(const auto& index: nameAndIndices.second){
				indicesPy.append(index);
			}
			indicesCouplingsPy[py::str(nameAndIndices.first)] = indicesPy;
		}
		return indicesCouplingsPy;
	}	

	py::list
	resultcollection_getcovariancematrixcouplingindices(const sphysics::pwa::ResultCollection& self){
		py::list covMatIndPy;
		for(const auto& indices: self.getCovarianceMatrixCouplingIndices()){
			covMatIndPy.append(py::make_tuple(indices.first, indices.second));
		}
		return covMatIndPy;
	}
	

	py::list
	resultcollection_getfitresults(const sphysics::pwa::ResultCollection& self){
		py::list fitResultsPy;
		for(const auto& fitResult: self.getFitResults()){
			fitResultsPy.append(sphysics::pwa::FitResult(fitResult));
		}
		return fitResultsPy;
	}

	void
	resultcollection_insert(sphysics::pwa::ResultCollection& self, const sphysics::pwa::FitResult& fitResult)
	{
		self.insert(sphysics::pwa::FitResult(fitResult));
	}

	py::bytes
	resultcollection_getmodelobj(const sphysics::pwa::ResultCollection& self){
		const auto& modelObj = self.getModelObj();
		std::string modelObjStr(modelObj.size(), '0');
		for(size_t i=0; i < modelObj.size(); ++i) modelObjStr[i] = modelObj[i];
		return py::bytes(modelObjStr);
	}

	py::str
	resultcollection_getlabel(const sphysics::pwa::ResultCollection& self){
		return py::str(self.getLabel());
	}

	py::str
	resultcollection_getmodelname(const sphysics::pwa::ResultCollection& self){
		return py::str(self.getModelName());
	}

	py::str
	resultcollection_getmodeldescription(const sphysics::pwa::ResultCollection& self){
		return py::str(self.getModelDescription());
	}


	const sphysics::pwa::FitResult*
	resultcollection_getbest(const sphysics::pwa::ResultCollection& self)
	{
		if (self.hasBest()) return &self.getBest();
		else return nullptr;
	}


	const sphysics::pwa::FitResult*
	resultcollection_getsmallestlossresult(const sphysics::pwa::ResultCollection& self){
		if (self.getFitResults().size()>0) return &self.getSmallestLossResult();
		else return nullptr;
	}


	double
	resultcollection_getintensity(const sphysics::pwa::ResultCollection& self, const std::string& waveName, std::shared_ptr<sphysics::pwa::FitResult> result)
	{
		return self.getIntensity(waveName, result.get());
	}

	double
	resultcollection_getintensityunc(const sphysics::pwa::ResultCollection& self, const std::string& waveName, std::shared_ptr<sphysics::pwa::FitResult> result)
	{
		// sphysics::pwa::FitResult const * result = nullptr;
		// if (not resultPy.is_none()){
		// 	result = &py::cast<sphysics::pwa::FitResult>(resultPy);
		// }
		return self.getIntensityUnc(waveName, result.get());
	}

	py::tuple
	resultcollection_getstate(const sphysics::pwa::ResultCollection& self)
	{
		return py::make_tuple(self.getModelName(),
								self.getModelDescription(),
								resultcollection_getmodelobj(self),
								resultcollection_getintegralmatrixgen(self),
								resultcollection_getintegralmatrixreco(self),
								resultcollection_getindicescouplings(self),
								resultcollection_getcovariancematrixcouplingindices(self),
								resultcollection_getfitresults(self),
								self.getLabel(),
								self.getEnvironment());
	}

	std::shared_ptr<sphysics::pwa::ResultCollection>
	resultcollection_setstate(const py::tuple& t)
	{
		// old constructor for [modelName, modelDescription, modelObj, covarianceMatrix, CouplingIndices, fitResults, label, environment]
		if (t.size() == 8){
		return resultcollection_constructor(
			py::cast<std::string>(t[0]),
			py::cast<std::string>(t[1]),
			t[2],
			Eigen::Matrix<double,1,1>(1),
			Eigen::Matrix<double,1,1>(1),
			py::cast<sphysics::pwa::CouplingIndices>(t[3]),
			py::cast<sphysics::pwa::CovarianceMatrixCouplingIndices>(t[4]),
			t[5],
			py::cast<std::string>(t[6]),
			py::cast<sphysics::Environment>(t[7])
		);
		}

		return resultcollection_constructor(
			py::cast<std::string>(t[0]),
			py::cast<std::string>(t[1]),
			t[2],
			py::cast<sphysics::pwa::IntegralMatrix>(t[3]),
			py::cast<sphysics::pwa::IntegralMatrix>(t[4]),
			py::cast<sphysics::pwa::CouplingIndices>(t[5]),
			py::cast<sphysics::pwa::CovarianceMatrixCouplingIndices>(t[6]),
			t[7],
			py::cast<std::string>(t[8]),
			py::cast<sphysics::Environment>(t[9])
		);

	}

}


namespace sphysics {
	namespace py {
		void pwaModule(pybind11::module_& m){

			pybind11::class_<sphysics::pwa::FitResult, std::shared_ptr<sphysics::pwa::FitResult>>(m, "FitResult")
			.def(pybind11::init(&::fitresult_constructor))
			.def(pybind11::pickle(&fitresult_getstate, &fitresult_setstate))
			.def("__str__", &sphysics::pwa::FitResult::str)
			.def_property_readonly("negLogLikelihood", &sphysics::pwa::FitResult::getNegLogLikelihood)
			.def_property_readonly("fitConverged", &sphysics::pwa::FitResult::getFitConverged)
			.def_property_readonly("covarianceMatrixValid", &sphysics::pwa::FitResult::getCovarianceMatrixValid)
			.def_property_readonly("covarianceMatrixMadePosDef", &sphysics::pwa::FitResult::getCovarianceMatrixMadePosDef)
			.def_property_readonly("couplings", &::fitresult_getcouplings)
			.def_property_readonly("covarianceMatrix", &::fitresult_getcovmatrix)
			.def("getResultObjDump", &::fitresult_getresultobj)
			.def("delResultObjDump", &sphysics::pwa::FitResult::delResultObj)
			.def(pybind11::self <  pybind11::self)
			.def_property_readonly("auxiliaryParameters", &sphysics::pwa::FitResult::getauxiliaryParameters)
			.def_property_readonly("auxiliaryParameterNames", &sphysics::pwa::FitResult::getauxiliaryParameterNames)
			;

			pybind11::class_<sphysics::pwa::ResultCollection, std::shared_ptr<sphysics::pwa::ResultCollection>>(m, "ResultCollection")
			.def(pybind11::init(&::resultcollection_constructor))
			.def(pybind11::pickle(&resultcollection_getstate, &resultcollection_setstate))
			//.def("__getitem__", [](const sphysics::pwa::ResultCollection& self, const int i) {return self[i];}, pybind11::return_value_policy::reference_internal)
			.def("__getitem__",[](const sphysics::pwa::ResultCollection& self, const int i) -> decltype(auto) {return self[i];},pybind11::return_value_policy::reference_internal)
			.def_property_readonly("label", &::resultcollection_getlabel)
			.def_property_readonly("mdoelName", &::resultcollection_getmodelname)
			.def_property_readonly("mdoelDescription", &::resultcollection_getmodeldescription)
			.def_property_readonly("integralMatrixGen", &::resultcollection_getintegralmatrixgen)
			.def_property_readonly("integralMatrixReco", &::resultcollection_getintegralmatrixreco)
			.def_property_readonly("indicesCouplings", &::resultcollection_getindicescouplings)
			.def_property_readonly("fitResults", &::resultcollection_getfitresults)
			.def_property_readonly("ordered", &::resultcollection_getfitresults)
			.def_property_readonly("best", &::resultcollection_getbest, pybind11::return_value_policy::reference_internal)
			.def_property_readonly("smallestLossResult", &::resultcollection_getsmallestlossresult, pybind11::return_value_policy::reference_internal)
			.def("getModelObjDump", &::resultcollection_getmodelobj)
			.def("getNBestResults", &sphysics::pwa::ResultCollection::getNBestResults)
			.def("hasBest", &sphysics::pwa::ResultCollection::hasBest)
			.def("insert", &resultcollection_insert)
			.def("check", &sphysics::pwa::ResultCollection::check)
			.def("getIntensity", &::resultcollection_getintensity, pybind11::arg("waveName"), pybind11::arg("result")=static_cast<sphysics::pwa::FitResult*>(nullptr))
			.def("getIntensityUnc", &::resultcollection_getintensityunc, pybind11::arg("waveName"), pybind11::arg("result")=static_cast<sphysics::pwa::FitResult*>(nullptr))
			.def("getPhase", &sphysics::pwa::ResultCollection::getPhase, pybind11::arg("waveNameA"), pybind11::arg("waveNameB"), pybind11::arg("result")=static_cast<sphysics::pwa::FitResult*>(nullptr))
			.def("getPhaseUnc", &sphysics::pwa::ResultCollection::getPhaseUnc, pybind11::arg("waveNameA"), pybind11::arg("waveNameB"), pybind11::arg("result")=static_cast<sphysics::pwa::FitResult*>(nullptr))
			.def("getNumSigEventsGen", &sphysics::pwa::ResultCollection::getNumSigEventsGen, pybind11::arg("result")=static_cast<sphysics::pwa::FitResult*>(nullptr))
			.def("getNumSigEventsReco", &sphysics::pwa::ResultCollection::getNumSigEventsReco, pybind11::arg("result")=static_cast<sphysics::pwa::FitResult*>(nullptr))
			.def("getNumSigEventsUncGen", &sphysics::pwa::ResultCollection::getNumSigEventsUncGen, pybind11::arg("result")=static_cast<sphysics::pwa::FitResult*>(nullptr))
			.def("getFitFraction", &sphysics::pwa::ResultCollection::getFitFraction, pybind11::arg("waveName"), pybind11::arg("result")=static_cast<sphysics::pwa::FitResult*>(nullptr))
			.def("getFitFractionUnc", &sphysics::pwa::ResultCollection::getFitFractionUnc, pybind11::arg("waveName"), pybind11::arg("result")=static_cast<sphysics::pwa::FitResult*>(nullptr))
			.def("getAllFitFraction", &sphysics::pwa::ResultCollection::getAllFitFraction, pybind11::arg("result")=static_cast<sphysics::pwa::FitResult*>(nullptr))
			.def("getAllFitFractionUnc", &sphysics::pwa::ResultCollection::getAllFitFractionUnc, pybind11::arg("result")=static_cast<sphysics::pwa::FitResult*>(nullptr))
			.def("getCouplingSpinDensityMatrix", &sphysics::pwa::ResultCollection::getCouplingSpinDensityMatrix, pybind11::arg("waveNameA"), pybind11::arg("waveNameB"), pybind11::arg("result")=static_cast<sphysics::pwa::FitResult*>(nullptr))
			.def("getSpinDensityMatrixUnc", &sphysics::pwa::ResultCollection::getSpinDensityMatrixUnc, pybind11::arg("waveNameA"), pybind11::arg("waveNameB"), pybind11::arg("result")=static_cast<sphysics::pwa::FitResult*>(nullptr))
			.def("getSpinDensityCovarianceMatrix", &sphysics::pwa::ResultCollection::getSpinDensityCovarianceMatrix, pybind11::arg("excludeWavesVec")=std::vector<std::string>{}, pybind11::arg("result")=static_cast<sphysics::pwa::FitResult*>(nullptr))
			.def("getIncludedSpinDensityCovarianceMatrix", &sphysics::pwa::ResultCollection::getIncludedSpinDensityCovarianceMatrix, pybind11::arg("includeWavesVec"), pybind11::arg("result")=static_cast<sphysics::pwa::FitResult*>(nullptr))
			;

			m.def("equal", &sphysics::pwa::equal, pybind11::arg("fitResult1"), pybind11::arg("fitResult2"), pybind11::arg("resultCollection"), pybind11::arg("verbose")=false);
		}
	}
}
