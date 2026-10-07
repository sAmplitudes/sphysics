/**
 * @file fitresult.cpp
 * @author Stefan Wallner (swallner@mpp.mpg.de)
 * @brief C++ implementaiton of FitResult Class
 * @date 2022-05-02
 * 
 * @copyright Copyright (c) 2022 the sphysics authors (GPL-3.0-or-later, see LICENSE)
 * 
 */

#include "fitresult.h"

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <set>
#include <functional>
#include <stdexcept>
#include <tuple>

namespace {
	template<typename T>
	void
	hash_combine(size_t& hash, const T& data){
		const size_t dataHash = std::hash<T>{}(data);
		hash = hash ^ (dataHash<<1);
	}
}

sphysics::pwa::FitResult::FitResult (const std::vector<std::complex<double>>&         couplings,
                                     const sphysics::pwa::CovarianceMatrix&                covarianceMatrix,
									 const std::vector<double>&                   		   auxiliaryParameters,
									 const std::vector<std::string>&              		   auxiliaryParameterNames,
                                     const double                                          negLogLikelihood,
                                     const bool                                            fitConverged,
                                     const bool                                            covarianceMatrixValid,
                                     const bool                                            covarianceMatrixMadePosDef,
                                     const std::vector<char>&                              resultObj
                                    ):
									couplings(couplings),
									covarianceMatrix(covarianceMatrix),
									auxiliaryParameters(auxiliaryParameters),
									auxiliaryParameterNames(auxiliaryParameterNames),
									negLogLikelihood(negLogLikelihood),
									fitConverged(fitConverged),
									covarianceMatrixValid(covarianceMatrixValid),
									covarianceMatrixMadePosDef(covarianceMatrixMadePosDef),
									resultObj(resultObj)
{

}


// sphysics::pwa::FitResult::FitResult(const sphysics::pwa::FitResult& other)
// {
// 	std::cout << "copy" << std::endl;
// 	std::cout << "\tcouplings" << std::endl;
// 	couplings = other.couplings;
// 	std::cout << "\tcov" << std::endl;
// 	covarianceMatrix = other.covarianceMatrix;
// 	std::cout << "\tll" << std::endl;
// 	negLogLikelihood = other.negLogLikelihood;
// 	std::cout << "\tfc" << std::endl;
// 	fitConverged = other.fitConverged;
// 	std::cout << "\tcv" << std::endl;
// 	covarianceMatrixValid = other.covarianceMatrixValid;
// 	std::cout << "\tcovMPD" << std::endl;
// 	covarianceMatrixMadePosDef = other.covarianceMatrixMadePosDef;
// 	std::cout << "\tro" << std::endl;
// 	resultObj = other.resultObj;
// 	std::cout << "\tend" << std::endl;
// }

// sphysics::pwa::FitResult::FitResult(sphysics::pwa::FitResult&& other)
// {
// 	std::cout << "move to " << this << " from " << &other << std::endl;
// 	std::cout << "\tcouplings" << std::endl;
// 	std::cout << "\t" << couplings.size() << std::endl;
// 	std::cout << "\t" << other.couplings.size() << std::endl;
// 	couplings = other.couplings;
// 	// std::cout << "\tcov" << std::endl;
// 	covarianceMatrix = other.covarianceMatrix;
// 	// std::cout << "\tll" << std::endl;
// 	negLogLikelihood = other.negLogLikelihood;
// 	// std::cout << "\tfc" << std::endl;
// 	fitConverged = other.fitConverged;
// 	// std::cout << "\tcv" << std::endl;
// 	covarianceMatrixValid = other.covarianceMatrixValid;
// 	// std::cout << "\tcovMPD" << std::endl;
// 	covarianceMatrixMadePosDef = other.covarianceMatrixMadePosDef;
// 	// std::cout << "\tro" << std::endl;
// 	resultObj = other.resultObj;
// 	// std::cout << "\tend" << std::endl;
// }

// sphysics::pwa::FitResult&
// sphysics::pwa::FitResult::operator= (sphysics::pwa::FitResult&& other)
// {
// 	std::cout << "assign to " << this << " from " << &other << std::endl;
// 	std::cout << "\tcouplings" << std::endl;
// 	std::cout << "\t" << couplings.size() << std::endl;
// 	std::cout << "\t" << other.couplings.size() << std::endl;
// 	couplings = other.couplings;
// 	// std::cout << "\tcov" << std::endl;
// 	covarianceMatrix = other.covarianceMatrix;
// 	// std::cout << "\tll" << std::endl;
// 	negLogLikelihood = other.negLogLikelihood;
// 	// std::cout << "\tfc" << std::endl;
// 	fitConverged = other.fitConverged;
// 	// std::cout << "\tcv" << std::endl;
// 	covarianceMatrixValid = other.covarianceMatrixValid;
// 	// std::cout << "\tcovMPD" << std::endl;
// 	covarianceMatrixMadePosDef = other.covarianceMatrixMadePosDef;
// 	// std::cout << "\tro" << std::endl;
// 	resultObj = other.resultObj;
// 	// std::cout << "\tend" << std::endl;
// 	return *this;
// }

std::string
sphysics::pwa::FitResult::str() const
{
	std::stringstream ss;
	ss << "-log(L) = " << std::setw(12) << std::fixed << std::setprecision(2) << negLogLikelihood;
	ss << "; " << (fitConverged? "    converged": "not converged");
	ss << "; " << (covarianceMatrixValid? "Cov   valid": "Cov invalid");
	return ss.str();
}



bool sphysics::pwa::equal (const sphysics::pwa::FitResult& fitResult1, const sphysics::pwa::FitResult& fitResult2,
                           const sphysics::pwa::ResultCollection& resultCollection, const bool verbose)
{

	const double deltaNegLL  = 1e-5;
	const double deltaRelC   = 10e-2;
	const double deltaCorr   = 20e-2;

	if (fabs(fitResult1.negLogLikelihood - fitResult2.negLogLikelihood) > deltaNegLL) return false;

	if (fitResult1.fitConverged != fitResult2.fitConverged){
		if (verbose) std::cout << "Convergence of fit resutls differs." << std::endl;	
		return false;
	}
	if (fitResult1.covarianceMatrixValid != fitResult2.covarianceMatrixValid){
		if (verbose) std::cout << "Covariance matrix validity of fit resutls differs." << std::endl;	
		return false;
	}
	if (fitResult1.covarianceMatrixMadePosDef != fitResult2.covarianceMatrixMadePosDef){
		if (verbose) std::cout << "Covariance matrix made-pos-def of fit resutls differs." << std::endl;	
		return false;
	}

	if (fitResult1.couplings.size() != fitResult2.couplings.size()){
		if (verbose) std::cout << "Couplings size fit resutls differs." << std::endl;	
		return false;
	}
	if (fitResult1.covarianceMatrix.rows() != fitResult2.covarianceMatrix.rows()){
		if (verbose) std::cout << "Covariance matrix size fit resutls differs." << std::endl;	
		return false;
	}
	if (fitResult1.covarianceMatrix.cols() != fitResult2.covarianceMatrix.cols()){
		if (verbose) std::cout << "Covariance matrix size fit resutls differs." << std::endl;	
		return false;
	}

	// check couplings
	for (size_t i = 0; i < fitResult1.couplings.size(); ++i){
		const auto& covIndices = resultCollection.covarianceMatrixCouplingIndices[i];
		if ( fabs(fitResult1.couplings[i].real()-fitResult2.couplings[i].real())/std::sqrt(fitResult1.covarianceMatrix(covIndices.first,  covIndices.first )) > deltaRelC ){
			if (verbose) std::cout << "Real part of coupling (" << i << ") differs." << std::endl;	
			return false;
		}
		if ( fabs(fitResult1.couplings[i].imag()-fitResult2.couplings[i].imag())/std::sqrt(fitResult1.covarianceMatrix(covIndices.second, covIndices.second)) > deltaRelC ){
			if (verbose) std::cout << "Imaginary part of coupling (" << i << ") differs." << std::endl;	
			return false;
		}
	}
	// check covariance-matrix elements
	// Acutally, the correlation matrix from fitResult1 is chacked agains the cov(i,j) from fitResult2 divided by sqrt(cov(i,i)*cov(j,j)) from fitResult1.
	// This is something like the difference between the correlation matrixes, which gives a usefule absolute scale for the difference.
	for (Eigen::Index i = 0; i < fitResult1.covarianceMatrix.rows(); ++i){
		for (Eigen::Index j = 0; j < fitResult1.covarianceMatrix.cols(); ++j){
			const double norm = sqrt(fitResult1.covarianceMatrix(i,i)*fitResult1.covarianceMatrix(j,j));
			if (fabs(fitResult1.covarianceMatrix(i,j)-fitResult2.covarianceMatrix(i,j))/norm > deltaCorr){
				if (verbose) std::cout << "Covariance matrix element (" << i << ", " << j << ") differs." << std::endl;	
				return false;
			}
		}
	}
	return true;
}


// sphysics::pwa::ResultCollection::ResultCollection(
//                                                   const std::string& modelName,
//                                                   const std::string& modelDescription,
//                                                   const std::vector<char>& modelObj,
//                                                   const std::map<std::string, size_t>& indicesCouplings,
//                                                   const CovarianceMatrixCouplingIndices& covarianceMatrixCouplingIndices
//                                )                  :
// 							   fitResults({}),
// 							   label(""),
// 							   modelName(modelName),
// 							   modelDescription(modelDescription),
// 							   environment(Environment()),
// 							   indicesCouplings(indicesCouplings),
// 							   covarianceMatrixCouplingIndices(covarianceMatrixCouplingIndices),
// 							   modelObj(modelObj)
// {
// }

sphysics::pwa::ResultCollection::ResultCollection(
                                                  const std::string& modelName,
                                                  const std::string& modelDescription,
                                                  const std::vector<char>& modelObj,
												  const sphysics::pwa::IntegralMatrix& integralMatrixGen,
												  const sphysics::pwa::IntegralMatrix& integralMatrixReco,
                                                  const sphysics::pwa::CouplingIndices& indicesCouplings,
                                                  const CovarianceMatrixCouplingIndices& covarianceMatrixCouplingIndices,
												  const std::vector<sphysics::pwa::FitResult>& fitResults,
												  const std::string& label,
                                                  const sphysics::Environment& environment
                               )                  :
							   fitResults(fitResults),
							   label(label),
							   modelName(modelName),
							   modelDescription(modelDescription),
							   environment(environment),
							   integralMatrixGen(integralMatrixGen),
							   integralMatrixReco(integralMatrixReco),
							   indicesCouplings(indicesCouplings),
							   covarianceMatrixCouplingIndices(covarianceMatrixCouplingIndices),
							   modelObj(modelObj)
{
	if (label.size() == 0){ // empty fit result
		if (fitResults.size() != 0) throw std::invalid_argument("ResultCollection initialization must have be initialized with fitResults!");
		buildLabel();
	}
	sort();
	
}

void
sphysics::pwa::ResultCollection::sort()
{
	std::sort(fitResults.begin(), fitResults.end());

	// return;

	// // to keep object resultPtr2Sort is pointing to
	// const std::vector<sphysics::pwa::FitResult> oldResults(fitResults);
	// std::vector<const sphysics::pwa::FitResult*> resultPtr2Sort;
	// for(auto& fitResult: oldResults) resultPtr2Sort.push_back(&fitResult);

	// std::cout << "Start sort" << std::endl;
	// std::sort(resultPtr2Sort.begin(), resultPtr2Sort.end(),
	//        [](const sphysics::pwa::FitResult* aPtr, const sphysics::pwa::FitResult* bPtr) -> bool { return *aPtr < *bPtr; });
	// std::cout << "End sort" << std::endl;


	// std::cout << "Reset fitResult" << std::endl;
	// fitResults = std::vector<sphysics::pwa::FitResult>();
	// fitResults.reserve(oldResults.size());
	// std::cout << "Copy fitresults" << std::endl;
	// for(const auto fitResultPtr: resultPtr2Sort) fitResults.push_back(*fitResultPtr);
	// std::cout << "Copy done" << std::endl;
}


const sphysics::pwa::FitResult&
sphysics::pwa::ResultCollection::getBest() const
{
	if (hasBest()) {
		return fitResults[0];
	}
	throw std::out_of_range("ResultCollection does not have a best solution!");
}



const sphysics::pwa::FitResult&
sphysics::pwa::ResultCollection::getSmallestLossResult() const
{
	if (fitResults.size() == 0){
		throw std::out_of_range("ResultCollection does not have any result!");
	}
	std::vector<const sphysics::pwa::FitResult*> nLLOrderedFitResults;
	nLLOrderedFitResults.reserve(fitResults.size());
	for(const auto& fitResult: fitResults) nLLOrderedFitResults.push_back(&fitResult);

	std::sort(nLLOrderedFitResults.begin(), nLLOrderedFitResults.end(),
	[] (const sphysics::pwa::FitResult* a, const sphysics::pwa::FitResult* b) -> bool {
		return a->getNegLogLikelihood() < b->getNegLogLikelihood();
	});
	return *nLLOrderedFitResults[0];
}


void
sphysics::pwa::ResultCollection::buildLabel()
{
	size_t hash = 0;
	::hash_combine(hash, modelName);
	::hash_combine(hash, modelDescription);
	std::string modelDump(modelObj.data());
	::hash_combine(hash, modelDump);
	::hash_combine(hash, environment.getGitHash());
	for(const auto& [name, version]: environment.getLibraryVersions()){
		::hash_combine(hash, name);
		::hash_combine(hash, version);
	}

	std::stringstream ss;
	ss << std::hex << hash;
	label = ss.str().substr(0, 7);
}


size_t
sphysics::pwa::ResultCollection::getNBestResults() const
{
	if (!hasBest()) return 0;

	size_t n = 0;
	const auto& best = getBest();
	for(const auto& fitResult: fitResults){
		if (equal(fitResult, best, *this)) ++n;
	}
	return n;
}


bool
sphysics::pwa::ResultCollection::check() const
{
	double fractionOfAttemptsFoundBest = 0.1;

	bool checksOK=true;

	// Check coupling index matching
	std::set<size_t> allCouplingIndices;
	size_t nCouplingsPerWave = 0;
	bool first=true;
	for (const auto& waveAndIndices: indicesCouplings){
		if (first) { nCouplingsPerWave = waveAndIndices.second.size(); first=false; }
		if (waveAndIndices.second.size() != nCouplingsPerWave) {
			std::cerr << "Number of coupling indices for wave '" << waveAndIndices.first << "' is " << waveAndIndices.second.size() << " but shoud be " << nCouplingsPerWave << std::endl;
			checksOK=false;
		}
		for (const auto index: waveAndIndices.second){
			if (allCouplingIndices.count(index)>0){
				std::cerr << "The coupling index " << index << "is used for wave '" << waveAndIndices.first << "', but was used before." << std::endl;
				checksOK=false;
			}
		}
		allCouplingIndices.insert(waveAndIndices.second.begin(), waveAndIndices.second.end());
	}
	const size_t nCouplings = allCouplingIndices.size();

	// Check covariance matrix index matching
	if (covarianceMatrixCouplingIndices.size() != nCouplings){
		std::cerr << "Number of covariance-matrix coupling indices (" << covarianceMatrixCouplingIndices.size() << ") is different form the number of couplings (" << nCouplings << ")!" << std::endl;
		checksOK=false;
	}
	std::set<size_t> covMatIndices;
	for (const auto& covIndices: covarianceMatrixCouplingIndices){
		covMatIndices.insert(covIndices.first);
		covMatIndices.insert(covIndices.second);
	}
	const size_t maxCovMatIndex = *std::max_element(covMatIndices.begin(), covMatIndices.end());

	// check data in fitResults
	for(const auto& fitResult: fitResults) {
		if (fitResult.getCouplings().size() != nCouplings){
			std::cerr << "Number of couplings in a FitResult should be " << nCouplings << " but is " << fitResult.getCouplings().size() << std::endl;
			checksOK = false;
		}
		if (static_cast<size_t>(fitResult.getCovarianceMatrix().rows()) < maxCovMatIndex+1){
			std::cerr << "Expect at least " << (maxCovMatIndex+1) << " entries in covariance matrix, but it has only " << fitResult.getCovarianceMatrix().rows() << " entries" << std::endl;
			checksOK = false;
		}
		if (static_cast<size_t>(fitResult.getCovarianceMatrix().cols()) < maxCovMatIndex+1){
			std::cerr << "Expect at least " << (maxCovMatIndex+1) << " entries in covariance matrix, but it has only " << fitResult.getCovarianceMatrix().cols() << " entries" << std::endl;
			checksOK = false;
		}
	}

	if (!hasBest()){
		std::cerr << "ResultCollection has no best result!" << std::endl;
		checksOK = false;
	}

	if (getNBestResults() == 1){
		std::cerr << "Best fitResult found only once!" << std::endl;
		checksOK = false;
	} else if (getNBestResults() <= getNResults()*fractionOfAttemptsFoundBest){
		std::cerr << "Best fitResult found only " << getNBestResults() << " times, but sould be found " << getNResults()*fractionOfAttemptsFoundBest << " (" << 100*fractionOfAttemptsFoundBest<< " %) times!" << std::endl;
		checksOK = false;
	}

	if (hasBest() and !equal(getBest(), getSmallestLossResult(), *this)){
		std::cerr << "There is a result with a smaller neg. log-likelihood (" << getSmallestLossResult().getNegLogLikelihood() << "), which is not the best result (" << getBest().getNegLogLikelihood() << ")!" << std::endl;
		checksOK = false;
	}

	return checksOK;
}


std::complex<double>
sphysics::pwa::ResultCollection::getCouplingSpinDensityMatrix(const std::string& waveNameA, const std::string& waveNameB, const sphysics::pwa::FitResult* result) const
{

	if (result == nullptr) result = &getBest();

	const auto& indicesA = indicesCouplings.at(waveNameA);
	const auto& indicesB = indicesCouplings.at(waveNameB);

	std::complex<double> sdme;
	for(size_t i=0; i < indicesA.size(); ++i){
		sdme += result->getCouplings()[indicesA[i]]*std::conj(result->getCouplings()[indicesB[i]]);
	}
	return sdme;
}


double
sphysics::pwa::ResultCollection::getIntensityUnc(const std::string& waveName, const sphysics::pwa::FitResult* result) const
{
	if (result == nullptr) result = &getBest();


	const auto& indicesC = indicesCouplings.at(waveName);
	const size_t nCouplings = indicesC.size();
	std::vector<std::pair<size_t, size_t>> indicesCov;
	indicesCov.reserve(nCouplings);
	for (size_t i=0; i < nCouplings; ++i){
		indicesCov.push_back(covarianceMatrixCouplingIndices[indicesC[i]]);
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1> jacobian(int(nCouplings*2));
	for (size_t i=0; i < nCouplings; ++i){
		jacobian(2*i  ) = 2*result->getCouplings()[indicesC[i]].real();
		jacobian(2*i+1) = 2*result->getCouplings()[indicesC[i]].imag();
	}


	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> cov(int(nCouplings * 2), int(nCouplings * 2));
	for (size_t i=0; i < nCouplings; ++i){
		const size_t iRe = i*2;
		const size_t iIm = i*2+1;
		for (size_t j=0; j < nCouplings; ++j){
			const size_t jRe = j*2;
			const size_t jIm = j*2+1;
			cov(iRe, jRe) = result->getCovarianceMatrix(indicesCov[i].first, indicesCov[j].first);
			cov(iRe, jIm) = result->getCovarianceMatrix(indicesCov[i].first, indicesCov[j].second);
			cov(iIm, jRe) = result->getCovarianceMatrix(indicesCov[i].second, indicesCov[j].first);
			cov(iIm, jIm) = result->getCovarianceMatrix(indicesCov[i].second, indicesCov[j].second);
		}
	}

	return sqrt(jacobian.transpose()*cov*jacobian);	
}

Eigen::VectorXd
sphysics::pwa::ResultCollection::getNumSigEventsGradient(Eigen::VectorXcd couplings, sphysics::pwa::IntegralMatrix integralMatrix) const
{
	// Calculate uncertainty without background parameters
	const size_t nWaves = couplings.size();
    Eigen::VectorXd vectorDerivatives(2*nWaves);
	double realDerivative = 0.0;
	double imagDerivative = 0.0;
    for (size_t i=0; i < nWaves; i++){
		realDerivative = 2*couplings[i].real() * integralMatrix(i,i).real();
        imagDerivative = 2*couplings[i].imag() * integralMatrix(i,i).real();
        for (size_t j=0; j < nWaves; j++){
            realDerivative+=2*couplings[j].real() * integralMatrix(i,j).real();
            realDerivative+=2*couplings[j].imag() * integralMatrix(i,j).imag();
            imagDerivative+=2*couplings[j].imag() * integralMatrix(i,j).real();
            imagDerivative-=2*couplings[j].real() * integralMatrix(i,j).imag();
		}
        vectorDerivatives[2*i] = realDerivative;
        vectorDerivatives[2*i+1] = imagDerivative;

	}
    
    return vectorDerivatives;
	
}

double
sphysics::pwa::ResultCollection::getNumSigEventsUncGen(const sphysics::pwa::FitResult* result) const
{
	//get result and convert vector into Eigen::vector
	if (result == nullptr) result = &getBest();
	Eigen::VectorXcd couplings = Eigen::Map<const Eigen::VectorXcd>(
    result->getCouplings().data(), static_cast<Eigen::Index>(result->getCouplings().size()));
	//get gradient
	Eigen::VectorXd gradient = getNumSigEventsGradient(couplings,getIntegralMatrixGen());
	//get covariance matrix for all couplings of the signal model
	std::cout << "getIntegralMatrixGen().rows()" << std::endl; 
	std::cout << getIntegralMatrixGen().rows() << std::endl; 
	int numCoupsSigModel = 2*getIntegralMatrixGen().rows();
	Eigen::VectorXi indices = Eigen::ArrayXi::LinSpaced(numCoupsSigModel, 0, numCoupsSigModel-1);
	sphysics::pwa::CovarianceMatrix covMatSigModel = result->getCovarianceMatrix()(indices,indices);
	// Calculation
	return sqrt(gradient.transpose()*(covMatSigModel*gradient));
}

double
sphysics::pwa::ResultCollection::getNumSigEventsGen(const sphysics::pwa::FitResult* result) const
{
	//get result and convert vector into Eigen::vector
	if (result == nullptr) result = &getBest();
	Eigen::VectorXcd couplings = Eigen::Map<const Eigen::VectorXcd>(
    result->getCouplings().data(), static_cast<Eigen::Index>(result->getCouplings().size()));

	// Calculation
	return (couplings.transpose()*(getIntegralMatrixGen()*couplings.conjugate())).value().real();
}

double
sphysics::pwa::ResultCollection::getNumSigEventsReco(const sphysics::pwa::FitResult* result) const
{
	//get result and convert vector into Eigen::vector
	if (result == nullptr) result = &getBest();
	Eigen::VectorXcd couplings = Eigen::Map<const Eigen::VectorXcd>(
    result->getCouplings().data(), static_cast<Eigen::Index>(result->getCouplings().size()));

	// Calculation
	return (couplings.transpose()*(getIntegralMatrixReco()*couplings.conjugate())).value().real();
}

double
sphysics::pwa::ResultCollection::getFitFraction(const std::string& waveName, const sphysics::pwa::FitResult* result) const
{
	if (result == nullptr) result = &getBest();

	return sphysics::pwa::ResultCollection::getIntensity(waveName,result)/sphysics::pwa::ResultCollection::getNumSigEventsGen(result);
}

double
sphysics::pwa::ResultCollection::getFitFractionUnc(const std::string& waveName, const sphysics::pwa::FitResult* result) const
{
	if (result == nullptr) result = &getBest();

	return sphysics::pwa::ResultCollection::getIntensityUnc(waveName,result)/sphysics::pwa::ResultCollection::getNumSigEventsGen(result);
}

std::map<std::string, double>
sphysics::pwa::ResultCollection::getAllFitFraction(const sphysics::pwa::FitResult* result) const
{
	if (result == nullptr) result = &getBest();
	std::map<std::string, double> fitFractions;
	for (const auto& waveName: getIndicesCouplings()){
		fitFractions.emplace(waveName.first,getFitFraction(waveName.first));
	}
	return fitFractions;
}


std::map<std::string, double>
sphysics::pwa::ResultCollection::getAllFitFractionUnc(const sphysics::pwa::FitResult* result) const
{
	if (result == nullptr) result = &getBest();
	std::map<std::string, double> fitFractionUnc;
	for (const auto& waveName: getIndicesCouplings()){
		fitFractionUnc.emplace(waveName.first,getFitFractionUnc(waveName.first));
	}
	return fitFractionUnc;
}




double
sphysics::pwa::ResultCollection::getPhaseUnc(const std::string& waveNameA, const std::string& waveNameB, const sphysics::pwa::FitResult* result) const 
{

	if (result == nullptr) result = &getBest(); //if nothing specified as result, result is set to getBest() automatically.

	if (waveNameA == waveNameB) {
		throw std::invalid_argument("Uncertainty of the phase difference can only be calculated for two different waves!");
	}

	Eigen::Vector2d jacobian;
	// Get the spin density matrix element for the two waves and calculate the real and imaginary part as well as the denominator.
	const auto& sdme = getCouplingSpinDensityMatrix(waveNameA, waveNameB, result);
	const auto& sdmeReal = sdme.real();
	const auto& sdmeImag = sdme.imag();
	const auto& denom = sdmeReal*sdmeReal + sdmeImag*sdmeImag;

	// Create the jacobian
	jacobian(0) = -sdmeImag/denom;
	jacobian(1) = sdmeReal/denom;

	auto cov = getIncludedSpinDensityCovarianceMatrix(std::vector<std::string>{waveNameA, waveNameB}, result);
	// Get the block of the covariance matrix the conatian only the rho_{ab} and rho_{ba} elements
	auto covMatrix = cov.block<2,2>(1, 1);
	// These elements of the spin density covariance matrix have to be flipped due to the definition of the matrix elements
	// On the off diagonal elements, we need Cov(Re(rho_{ab}), Im(rho_{ab})), but from our covariance definition we have Cov(Re(rho_{ab}), Im(rho_{ba}))
	// Since rho_{ab} = rho_{ba}^*, we have to flip the sign of these covariance matrix elements
	covMatrix(1,0) = -covMatrix(1,0);
	covMatrix(0,1) = -covMatrix(0,1);

	return sqrt(jacobian.transpose()*covMatrix*jacobian);
}


std::tuple<double, double> 
sphysics::pwa::ResultCollection::getSpinDensityMatrixUnc(const std::string& waveNameA, const std::string& waveNameB, const sphysics::pwa::FitResult* result) const 
{
	
	if (result == nullptr) result = &getBest(); //if nothing specified as result, result is set to getBest() automatically.


	const auto& indicesCA = indicesCouplings.at(waveNameA);
	const auto& indicesCB = indicesCouplings.at(waveNameB);
	std::vector<size_t> indicesCAB;
	std::copy(indicesCA.begin(), indicesCA.end(), std::back_inserter(indicesCAB));
	std::copy(indicesCB.begin(), indicesCB.end(), std::back_inserter(indicesCAB));
	const size_t nCouplings = indicesCAB.size();
	std::vector<std::pair<size_t, size_t>> indicesCov;
	indicesCov.reserve(nCouplings);
	for (size_t i=0; i<nCouplings; ++i) {
		indicesCov.push_back(covarianceMatrixCouplingIndices[indicesCAB[i]]);
	}


	Eigen::Matrix<double, Eigen::Dynamic, 1> jacobianRe(int(nCouplings*2));
	for (size_t i=0; i< indicesCA.size(); ++i){
		jacobianRe(4*i) = result->getCouplings()[indicesCB[i]].real();
		jacobianRe(4*i+1)=result->getCouplings()[indicesCB[i]].imag();
		jacobianRe(4*i+2)=result->getCouplings()[indicesCA[i]].real();
		jacobianRe(4*i+3)=result->getCouplings()[indicesCA[i]].imag();
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1> jacobianIm(int(nCouplings*2));
	for (size_t i=0; i< indicesCA.size(); ++i){
		jacobianIm(4*i) = -1*result->getCouplings()[indicesCB[i]].imag();
		jacobianIm(4*i+1)= result->getCouplings()[indicesCB[i]].real();
		jacobianIm(4*i+2)=result->getCouplings()[indicesCA[i]].imag();
		jacobianIm(4*i+3)=-1*result->getCouplings()[indicesCA[i]].real();
	}


	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> cov(int(nCouplings * 2), int(nCouplings * 2));
	for (size_t i=0; i < nCouplings; ++i){
		const size_t iRe = i*2;
		const size_t iIm = i*2+1;
		for (size_t j=0; j < nCouplings; ++j){
			const size_t jRe = j*2;
			const size_t jIm = j*2+1;
			cov(iRe, jRe) = result->getCovarianceMatrix(indicesCov[i].first, indicesCov[j].first);
			cov(iRe, jIm) = result->getCovarianceMatrix(indicesCov[i].first, indicesCov[j].second);
			cov(iIm, jRe) = result->getCovarianceMatrix(indicesCov[i].second, indicesCov[j].first);
			cov(iIm, jIm) = result->getCovarianceMatrix(indicesCov[i].second, indicesCov[j].second);
		}
	}

	const double VarRe= jacobianRe.transpose()*cov*jacobianRe;
	const double VarIm= jacobianIm.transpose()*cov*jacobianIm;
	return std::make_tuple(sqrt(VarRe), sqrt(VarIm));

}

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
sphysics::pwa::ResultCollection::getSpinDensityCovarianceMatrix(const std::vector<std::string> excludeWavesVec, const sphysics::pwa::FitResult* result) const
{
	if (result == nullptr) result = &getBest(); //if nothing specified as result, result is set to getBest() automatically.
	// Define the rank of the spin density matrix, 
	const auto rank = int(indicesCouplings.begin()->second.size());
	if (rank > 1) {
		throw std::logic_error("We have not tested on an actual fit result whether this works for rank > 1. Technically it should work, but please cross check the result!");
	}
	// Define the number of waves used for the error propagation
	size_t nWaves;
	nWaves = indicesCouplings.size() - excludeWavesVec.size();
	// Get couplings and couplings covariance matrix
	auto couplings = result->getCouplings();
	auto fullCouplingsCov = result->getCovarianceMatrix();

	// Create sorted couplings covariance matrix indices
	std::vector<std::pair<size_t, size_t>> spinDensityIndices;
	spinDensityIndices.reserve(nWaves*nWaves);
	for (size_t i=0; i < nWaves; ++i){
		for (size_t j=0; j < nWaves; ++j){
			spinDensityIndices.push_back(std::make_pair(i,j));
		}
	}

	// Create a vector of couplings indices, which are not in the excludeWavesVec
	std::vector<double> couplingIndicesVec;
	couplingIndicesVec.reserve(nWaves*rank);
	for (const auto& waveName: getIndicesCouplings()){
		if (std::find(excludeWavesVec.begin(), excludeWavesVec.end(), waveName.first) == excludeWavesVec.end()) {
			const auto& indicesC = indicesCouplings.at(waveName.first);
			couplingIndicesVec.insert(couplingIndicesVec.end(), indicesC.begin(), indicesC.end());
		}
	}

	// Sort coupling indices vector (by default they are sorted the way CouplingsIndices are sorted)
	auto sortedIndices = couplingIndicesVec;
	std::sort(sortedIndices.begin(), sortedIndices.end());
	// Exclude the covariance matrix indices not excluded by excludeWavesVec
	std::vector<size_t> excludeCovIndicesVec;
	excludeCovIndicesVec.reserve(2*nWaves*rank);
	for (const auto& sortedIndex: sortedIndices){
		const auto& covIndex = covarianceMatrixCouplingIndices[sortedIndex];
		excludeCovIndicesVec.push_back(covIndex.first);
		excludeCovIndicesVec.push_back(covIndex.second);
	}

	// Create a smaller covariance matrix. Exclude the elements that are mentionned excludeCovIndicesVec
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> cov(int(2*nWaves*rank), int(2*nWaves*rank));
	for (size_t i=0; i<2*nWaves; ++i){
		for (size_t j=0; j<2*nWaves; ++j){
			cov(i,j) = fullCouplingsCov(excludeCovIndicesVec[i], excludeCovIndicesVec[j]);
		}
	}

	// Create a vector of couplings, which are not in excludeWavesVec
	std::vector<double> couplingsVec;
	couplingsVec.reserve(2*nWaves*rank);
	for (const auto& index: sortedIndices){
		couplingsVec.push_back(result->getCouplings()[index].real());
		couplingsVec.push_back(result->getCouplings()[index].imag());
	}

	// Create the jacobian matrix
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> jacobian(int(nWaves*nWaves), int(2*rank*nWaves));
	// Loop over the rows of the jacobian matrix
	for (size_t row=0; row<nWaves*nWaves; ++row){
		// Go in this statement if it corresponds to a diagonal spin density matrix element
		if (row%(nWaves+1)==0){
			const auto indexCA = spinDensityIndices[row].first;
			for (size_t col=0; col<2*nWaves*rank; ++col){
				if (col < 2*indexCA*rank || col >= 2*indexCA*rank + 2*rank){
					jacobian(row, col)=0;
				} else {
					jacobian(row, col) = 2*couplingsVec[col];
				}
			}
		// Go in this statement if it corresponds to a Real off diagonal spin density matrix element
		} else if (spinDensityIndices[row].first < spinDensityIndices[row].second) {
			const size_t indexCA = spinDensityIndices[row].first;
			const size_t indexCB = spinDensityIndices[row].second;
			const int indexCDifference = indexCA-indexCB;
			for (size_t col=0; col<2*nWaves*rank; ++col){
				// It applies for the spin densit matrix indices a,b: a<b
				if (col>=2*indexCA*rank && col<2*indexCA*rank+2*rank){
					jacobian(row, col) = couplingsVec[col-2*indexCDifference*rank];
				// It applies for the spin densit matrix indices a,b: a>b
				} else if (col>=2*indexCB*rank && col<2*indexCB*rank+2*rank){
					jacobian(row, col) = couplingsVec[col+2*indexCDifference*rank];
				} else {
					jacobian(row, col) = 0;
				}
			}
		// Go in this statement if it corresponds to an Imaginary off diagonal spin density matrix element
		} else {
			const auto indexCA = spinDensityIndices[row].first;
			const auto indexCB = spinDensityIndices[row].second;
			const int indexCDifference = indexCA-indexCB;

			for (size_t col=0; col<2*nWaves*rank; ++col){
				// The differentiations of Im(rho_{ab}) are given by:
				// dIm(rho_{ba})/dRe(c_a) =  Im(c_b), dIm(rho_{ba})/dIm(c_a) = -Re(c_b)
				// dIm(rho_{ba})/dRe(c_b) = -Im(c_a), dIm(rho_{ba})/dIm(c_b) =  Re(c_a)

				// It applies for the spin densit matrix indices a,b: a<b
				if (col>=2*indexCA*rank && col<2*indexCA*rank+2*rank){
					if (col%2==0) {
						// dIm(rho_{ba})/dRe(c_a) = Im(c_b)
						jacobian(row, col) = couplingsVec[col-2*indexCDifference*rank+1];
						// The index of couplingsVec switches from Re(c_a) to Im(c_b), similarly for the other cases
					} else {
						// dIm(rho_{ba})/dIm(c_a) = -Re(c_b)
						jacobian(row, col) = -1.0*couplingsVec[col-2*indexCDifference*rank-1];
					}
				// It applies for the spin densit matrix indices a,b: a>b
				} else if (col>=2*indexCB*rank && col<2*indexCB*rank+2*rank){
					if (col%2==0) {
						// dIm(rho_{ba})/dRe(c_b) = -Im(c_a)
						jacobian(row, col) = -1.0*couplingsVec[col+2*indexCDifference*rank+1];
					} else {
						// dIm(rho_{ba})/dIm(c_b) = Im(c_a)
						jacobian(row, col) = couplingsVec[col+2*indexCDifference*rank-1];
					}
				} else{
					jacobian(row, col) = 0;
				}
			}
		}
	}
	return jacobian*cov*jacobian.transpose();
}

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
sphysics::pwa::ResultCollection::getIncludedSpinDensityCovarianceMatrix(const std::vector<std::string> includeWavesVec, const sphysics::pwa::FitResult* result) const
{

	if (result == nullptr) result = &getBest(); //if nothing specified as result, result is set to getBest() automatically.

	// From the includeWavesVec, create the excludeWavesVec
	std::vector<std::string> excludeWavesVec;
	excludeWavesVec.reserve(getIndicesCouplings().size() - includeWavesVec.size());
	for (const auto& waveName: getIndicesCouplings()){
		if (std::find(includeWavesVec.begin(), includeWavesVec.end(), waveName.first) == includeWavesVec.end()) {
			excludeWavesVec.push_back(waveName.first);
		}
	}
	return getSpinDensityCovarianceMatrix(excludeWavesVec, result);
}
