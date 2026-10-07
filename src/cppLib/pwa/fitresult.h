/**
 * @file fitresult.h
 * @author Stefan Wallner (swallner@mpp.mpg.de)
 * @brief C++ implementaiton of FitResult class
 * @date 2022-05-02
 * 
 * @copyright Copyright (c) 2022 the sphysics authors (GPL-3.0-or-later, see LICENSE)
 * 
 */
#ifndef SPHYSICS_PWA_FITRESULT_H
#define SPHYSICS_PWA_FITRESULT_H

#include <vector>
#include <complex>
#include <cmath>
#include <string>
#include <stdexcept>
#include <Eigen/Dense>
#include "../utilities/environment.h"

namespace sphysics {
	namespace py {
		class FitResultPickleSuite;
	}
}

namespace sphysics {
	namespace pwa {

		// class CovarianceMatrix {
		// 	public:
		// 		CovarianceMatrix() {}
		// 		CovarianceMatrix(const size_t, const size_t) {throw std::runtime_error("CovarianceMatrix not implemented");}
		// 		size_t GetNrows() const {throw std::runtime_error("CovarianceMatrix not implemented"); return 0; }
		// 		size_t GetNcols() const {throw std::runtime_error("CovarianceMatrix not implemented"); return 0; }
		// 		double operator() (const size_t i, const size_t j) const {
		// 			(void)i;
		// 			(void)j;
		// 			return 0.0;
		// 		}
		// 		double&  operator() (const size_t i, const size_t j) {
		// 			(void)i;
		// 			(void)j;
		// 			return _dummyValue;
		// 		}
		// 	private:
		// 		double _dummyValue;
		// };
		typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> CovarianceMatrix;

		// typedef TMatrixD CovarianceMatrix;
		typedef std::vector<std::pair<int,int>> CovarianceMatrixCouplingIndices;
		typedef std::map<std::string, std::vector<size_t>> CouplingIndices;
		typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> IntegralMatrix;
		class ResultCollection;

		class FitResult {
		public:
			/**
			 * @brief Construct a new Fit Result object with data
			 * 
			 * @param couplings  Parameter sef of the fit
			 * @param covarianceMatrix Covariance amtrix
			 * @param numSigEvents  Number of signal events, calculated via the integralmatrix
			 * @param auxiliaryParameters
			 * @param auxiliaryParameterNames
			 * @param negLogLikelihood Optimized loss value
			 * @param fitConverged true if the fit attempt converged
			 * @param covarianceMatrixValid Covariance matrix of the fit is a valide minimum
			 * @param covarianceMatrixMadePosDef Covariance matrix is not valide, but was foced to be psotive definite
			 * @param resultObj Dump of result object of the minimizer
			 */
			FitResult (const std::vector<std::complex<double>>&     couplings,
			           const CovarianceMatrix&                      covarianceMatrix,
					   const std::vector<double>&                   auxiliaryParameters,
					   const std::vector<std::string>&              auxiliaryParameterNames,
			           const double                                 negLogLikelihood,
					   const bool                                   fitConverged,
					   const bool                                   covarianceMatrixValid,
					   const bool                                   covarianceMatrixMadePosDef,
					   const std::vector<char>&                     resultObj
					  );

			FitResult( const FitResult&  ) = default;
			FitResult(       FitResult&& ) = default;
			FitResult& operator = (const FitResult&  ) = default;
			FitResult& operator = (      FitResult&& ) = default;
			~FitResult() = default;


			const std::vector<std::complex<double>>& getCouplings() const { return couplings; }
			const CovarianceMatrix& getCovarianceMatrix() const { return covarianceMatrix; }

			double getNumSigEvents() const {return numSigEvents; }
			const std::vector<double>& getauxiliaryParameters() const {return auxiliaryParameters; }
			const std::vector<std::string>& getauxiliaryParameterNames() const {return auxiliaryParameterNames; }
			
			double getCovarianceMatrix(const size_t i, const size_t j) const { return covarianceMatrix(i,j); }

			double getNegLogLikelihood() const {return negLogLikelihood; }
			bool getFitConverged() const {return fitConverged; }
			bool getCovarianceMatrixValid() const {return covarianceMatrixValid; }
			bool getCovarianceMatrixMadePosDef() const {return covarianceMatrixMadePosDef; }
			const std::vector<char>& getResultObj() const {return resultObj; }
			void delResultObj() {resultObj.clear(); resultObj.shrink_to_fit();}

			std::string str() const;

		private:
			std::vector<std::complex<double>> couplings;
			CovarianceMatrix covarianceMatrix;
			double numSigEvents;
			std::vector<double> auxiliaryParameters;
			std::vector<std::string> auxiliaryParameterNames;

			double negLogLikelihood;
			bool fitConverged;
			bool covarianceMatrixValid;
			bool covarianceMatrixMadePosDef;
			std::vector<char> resultObj;

		public:
			friend class sphysics::py::FitResultPickleSuite;
			friend bool equal(const FitResult&, const FitResult&, const ResultCollection&, const bool);
			friend bool operator<  (const FitResult& fitResult1, const FitResult& fitResult2);
		};

		/**
		 * @brief Check agreement of two fit results
		 * 
		 * Two fitResults agree, iff
		 * - the difference in neg.- log-likelihood is smaller than diffLogLike
		 * - the convergence of the results are equal
		 * - the covariance matrix validity and `madePosDef` are equal
		 * - the couplings differ not more than 10% of their uncertainty
		 * - the "correlation matrix" elements differ not more than 20%
		 * 
		 * @param fitResult1 
		 * @param fitResult2 
		 * @return true if fitResult1 equals fitResult2
		 * @return false else
		 */
		bool equal (const FitResult& fitResult1, const FitResult& fitResult2, const ResultCollection& resultCollection, const bool verbose=false);

		/**
		 * @brief Compare two fit results
		 * 
		 * fitResult1 is smaller than fitResult2 if
		 *  - fitResult2 is not converged, or
		 *  - fitResult2 has not a valid covariance matrix, or
		 *  - fitResult2's covaraice matrix was made pos. def., or
		 *  - fitResult2 or fitResult1 do not have a likelihood value, or
		 *  - fitResult1 has a lower neg. log-likelihood than fitResult2
		 * 
		 * @param fitResult1 
		 * @param fitResult2 
		 * @return true 
		 * @return false 
		 */
		bool operator< (const FitResult& fitResult1, const FitResult& fitResult2)
		{
			if (fitResult1.fitConverged && !fitResult2.fitConverged) return true;
			if (!fitResult1.fitConverged) return false;
			if (fitResult1.covarianceMatrixValid && !fitResult2.covarianceMatrixValid) return true;
			if (!fitResult1.covarianceMatrixValid) return false;
			if (!fitResult1.covarianceMatrixMadePosDef && fitResult2.covarianceMatrixMadePosDef) return true;
			if (fitResult1.covarianceMatrixMadePosDef) return false;
			if (!std::isnan(fitResult1.negLogLikelihood) && std::isnan(fitResult2.negLogLikelihood)) return true;
			if (std::isnan(fitResult1.negLogLikelihood)) return false;
			return fitResult1.negLogLikelihood < fitResult2.negLogLikelihood;
		}



		/**
		 * @brief Class that stores multiple results form the same fit
		 * 
		 * The results are orders by their quality (see opterator < for FitResult)
		 */
		class ResultCollection {
		public:
			// ResultCollection(
			//                  const std::string& modelName,
			// 				 const std::string& modelDescription,
			// 		         const std::vector<char>& modelObj,
			//                  const std::map<std::string, size_t>& indicesCouplings,
			// 				 const CovarianceMatrixCouplingIndices& covarianceMatrixCouplingIndices
			// 				 );

			ResultCollection(
			                 const std::string& modelName,
							 const std::string& modelDescription,
					         const std::vector<char>& modelObj,
							 const IntegralMatrix& integralMatrixGen,
							 const IntegralMatrix& integralMatrixReco,
			                 const CouplingIndices& indicesCouplings,
							 const CovarianceMatrixCouplingIndices& covarianceMatrixCouplingIndices,
							 const std::vector<FitResult>& fitResults = {},
							 const std::string& label = "",
							 const sphysics::Environment& environment = sphysics::Environment()
							 );

			ResultCollection( const ResultCollection&  ) = default;
			ResultCollection(       ResultCollection&& ) = delete;
			ResultCollection& operator = (const ResultCollection&  ) = default;
			ResultCollection& operator = (      ResultCollection&& ) = delete;
			~ResultCollection() = default;

			void insert(const FitResult&& fitResult){insertOnly(std::move(fitResult)); sort();}
			void insert(const FitResult&  fitResult){insertOnly(fitResult); sort();}

			bool hasBest() const {return fitResults.size() > 0 && fitResults[0].getFitConverged();}
			const FitResult& getBest() const;
			const FitResult& getSmallestLossResult() const;
			const FitResult& operator[] (int i) const {if(i<0)i+=fitResults.size(); return fitResults[i];}
			size_t getNResults() const {return fitResults.size();}
			size_t getNBestResults() const;

			/**
			 * @brief Check consistency of collection and of fit results
			 * 
			 * @return true All resutls are consistent
			 */
			bool check() const;

			double getIntensity(const std::string& waveName, const FitResult* result=nullptr) const;
			double getIntensityUnc(const std::string& waveName, const FitResult* result=nullptr) const;
			double getPhase(const std::string& waveNameA, const std::string& waveNameB, const FitResult* result=nullptr) const;
			/**
			 * @brief This method calculates the uncertainty of the phase difference between two waves.
			 * The phase difference is calculated as: phi_{ab} = arg(rho_{ab}) = atan2(Im(rho_{ab}), Re(rho_{ab}))
			 * The uncertainty is then calcualted by the error propagation:
			 * The jacobian is given by: J = [-Im(rho_{ab})/denom, Re(rho_{ab})/denom], where denom = Re(rho_{ab})^2 + Im(rho_{ab})^2
			 * Then obtain a part of the spin density covariance matrix: ( Var(Re(rho_{ab})),               Cov(Re(rho_{ab}), Im(rho_{ab})) )
			 *                                                           ( Cov(Im(rho_{ab}), Re(rho_{ab})), Var(Im(rho_{ab}))               )
			 * The uncertainty of the phase difference is then given by: sqrt(J^T * Cov * J)
			 * 
			 * @param waveNameA: Name of the first wave
			 * @param waveNameB: Name of the second wave
			 * @param result: FitResult to use for the calculation, if not given, the best fit result is used
			 * 
			 * @return: The uncertainty of the phase difference between the two waves in radians.
			 * 
			 * @throws std::invalid_argument if waveNameA and waveNameB are equal, since the uncertainty of the phase difference is not defined for the same wave.
			 */
			double getPhaseUnc(const std::string& waveNameA, const std::string& waveNameB, const FitResult* result=nullptr) const;
			/**
			 * @brief This method calculates the total model intensity including interference effects without reoncstruction effects.
			 * 
			 * @param result: FitResult to use for the calculation, if not given, the best fit result is used
			 * 
			 * @return: The total model intensity without reconstruction effects.
			 */
			double getNumSigEventsGen(const FitResult* result=nullptr) const;
			/**
			 * @brief This method calculates the total model intensity including interference effects with reoncstruction effects.
			 * 
			 * @param result: FitResult to use for the calculation, if not given, the best fit result is used
			 * 
			 * @return: The total model intensity with reconstruction effects.
			 */
			double getNumSigEventsReco(const FitResult* result=nullptr) const;
			/**
			 * @brief This method calculates the fit fraction of a given wave, defined by the wave intenisty divided by the total model intensity including interference effects.
			 * 
			 * @param waveName: Name of the wave
			 * @param result: FitResult to use for the calculation, if not given, the best fit result is used
			 * 
			 * @return: The fit fraction of the given wave.
			 */
			double getFitFraction(const std::string& waveName, const FitResult* result=nullptr) const;
			/**
			 * @brief This method calculates the total model intensity uncertainty taking into account the complete covariance matrix.
			 * 
			 * @param result: FitResult to use for the calculation, if not given, the best fit result is used.
			 * 
			 * @return: The total model intensity uncertainty.
			 */
			double getNumSigEventsUncGen(const sphysics::pwa::FitResult* result=nullptr) const;
			/**
			 * @brief This method calculates the gradient used for the uncertainty calculation of the total model intensity uncertainty.
			 * 
			 * @param couplings: Model couplings
			 * @param integralMatrix: Integralmatrix used for the computation
			 * 
			 * @return: The gradient for the total model intensity uncertainty calculation.
			 */
			Eigen::VectorXd getNumSigEventsGradient(Eigen::VectorXcd couplings, sphysics::pwa::IntegralMatrix integralMatrix) const;
			/**
			 * @brief This method calculates the uncertainty of the fit fraction of a given wave.
			 * 
			 * @param waveName: Name of the wave
			 * @param result: FitResult to use for the calculation, if not given, the best fit result is used
			 * 
			 * @return: The fit fraction uncertainty of the given wave.
			 */
			double getFitFractionUnc(const std::string& waveName, const FitResult* result=nullptr) const;
			/**
			 * @brief This method calculates the fit fraction for all waves of a given FitResult.
			 * 
			 * @param result: FitResult to use for the calculation, if not given, the best fit result is used
			 * 
			 * @return: The fit fractions of the given fitResult as map.
			 */
			std::map<std::string, double> getAllFitFraction(const FitResult* result=nullptr) const;
			/**
			 * @brief This method calculates the uncertainties of all fit fractions for all waves of a given fitResult.
			 * 
			 * @param result: FitResult to use for the calculation, if not given, the best fit result is used
			 * 
			 * @return: The fit fraction uncertainties of the given fitResult as map.
			 */
			std::map<std::string, double> getAllFitFractionUnc(const FitResult* result=nullptr) const;
			std::complex<double> getCouplingSpinDensityMatrix(const std::string& waveNameA, const std::string& waveNameB, const FitResult* result=nullptr) const;
			std::tuple<double, double>  getSpinDensityMatrixUnc(const std::string& waveNameA, const std::string& waveNameB, const FitResult* result=nullptr) const;
			/**
			 * @brief This method calculates the error propagation from the covariance matrix obtained by the fit to the covariance matrix of the spin density matrix.
			 * The covariance matrix is calculated as follows:
			 * Create the lambda vector: lambda = [Re(rho_{ab}), if a<=b; Im(rho_{ba}), if a>b;]
			 * Then calculate the jacobian matrix to transform the couplings covariance matrix:
			 * J = d(lambda)/d(couplings)
			 * The the covariance matrix of the spin density matrix is given by:
			 * Cov(rho_{ab}) = J * Cov(couplings) * J^T
			 * 
			 * @param excludeWavesVec: Vector of wave names to exclude from the covariance matrix calculation, optional
			 *                         if no waves are given, all waves are included
			 * @param result: FitResult to use for the calculation, if not given, the best fit result is used
			 * 
			 * @return: One covariance matrix (i,j) of real and imaginary parts of the sde for each space point.
			 *          i=nWaves*a+b corresponds to the real part of the sde of wave a and b if b >= a
			 *          i=nWaves*a+b corresponds to the imaginary part of the sde of wave b and a if b < a
			 * 
			 * @throws std::invalid_argument if the rank of the spin density matrix is larger than 1. The calculation should technically work and was cross checked on a dummy result.
			 *         But we have not fully tested it on an actual fit result. Please use with care and cross check your results!
			 */
			Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> getSpinDensityCovarianceMatrix(const std::vector<std::string> excludeWavesVec={}, const FitResult* result=nullptr) const;
			/**
			 * @brief This method is the same as getSpinDensityCovarianceMatrix, but the covariance matrix is calculated only for waves given in includeWavesVec
			 * 
			 * @param includeWavesVec: Vector of wave names to include in the covariance matrix calculation, mandatory
			 * 
			 * @param result: FitResult to use for the calculation, if not given, the best fit result is used
			 * 
			 * @return: One covariance matrix (i,j) of real and imaginary parts of the sde for each space point.
			 *          i=nWaves*a+b corresponds to the real part of the sde of wave a and b if b >= a
			 *          i=nWaves*a+b corresponds to the imaginary part of the sde of wave b and a if b < a
			 */
			Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> getIncludedSpinDensityCovarianceMatrix(const std::vector<std::string> includeWavesVec, const FitResult* result=nullptr) const;

			const std::string& getLabel() const { return label; }
			const std::string& getModelName() const { return modelName; }
			const std::string& getModelDescription() const { return modelDescription; }
			const sphysics::Environment& getEnvironment() const { return environment; }
			const IntegralMatrix& getIntegralMatrixGen() const { return integralMatrixGen; }
			const IntegralMatrix& getIntegralMatrixReco() const { return integralMatrixReco; }
			const CouplingIndices& getIndicesCouplings() const { return indicesCouplings; }
			const CovarianceMatrixCouplingIndices& getCovarianceMatrixCouplingIndices() const { return covarianceMatrixCouplingIndices; }
			const std::vector<char>& getModelObj() const { return modelObj; }
			const std::vector<FitResult>& getFitResults() const { return fitResults; }

		private:
			std::vector<FitResult> fitResults;
			std::string label;
			std::string modelName;
			std::string modelDescription;
			sphysics::Environment environment;
			IntegralMatrix integralMatrixGen;
			IntegralMatrix integralMatrixReco;
			CouplingIndices indicesCouplings;
			CovarianceMatrixCouplingIndices covarianceMatrixCouplingIndices;
			std::vector<char> modelObj;

			void sort();
			/**
			 * @brief Only insets the fit result, does NOT sort
			 * 
			 * @param fitResult 
			 */
			void insertOnly(const FitResult&& fitResult){fitResults.push_back(std::move(fitResult));}
			void insertOnly(const FitResult&  fitResult){fitResults.push_back(          fitResult);}
			void buildLabel();

			friend bool equal(const FitResult&, const FitResult&, const ResultCollection&, const bool);
		};
	}
}

inline double
sphysics::pwa::ResultCollection::getIntensity(const std::string& waveName, const sphysics::pwa::FitResult* result) const
{
	return getCouplingSpinDensityMatrix(waveName, waveName, result).real();
}

inline double
sphysics::pwa::ResultCollection::getPhase(const std::string& waveNameA, const std::string& waveNameB, const sphysics::pwa::FitResult* result) const
{
	return std::arg(getCouplingSpinDensityMatrix(waveNameA, waveNameB, result));
}

#endif
