/*
 * environment.h
 *
 *  Created on: Apr 22, 2020
 *      Author: stefan
 */

#ifndef SPHYSICS_ENVIRONMENT_H
#define SPHYSICS_ENVIRONMENT_H

#include <string>
#include <map>
#include <vector>


namespace pybind11 {
	class tuple;
}
namespace sphysics {

	class Environment;

	namespace py {
		Environment environment_setstate(const pybind11::tuple& state);
	}


class Environment {
public:
	Environment();
	Environment(const Environment& other) = default;
	~Environment(){}
	const Environment& operator = (const Environment& other) = delete;

	const std::string getGitHash()const{ return _gitHash; }
	const std::string getVersion()const{ return _version; }
	const std::vector<std::string> getGitStatus()const{return _gitStatus;}
	/**
	* @return: True if the status of the working space is clean
	*/
	bool isGitStatusClean()const;
	const std::map<std::string,std::string>& getLibraryVersions()const{ return _libraryVersions; }

	std::string getSummary()const;

private:
	std::string _version;
	std::string _gitHash;
	std::vector<std::string> _gitStatus;
	std::map<std::string,std::string> _libraryVersions;

	void set(const std::string& version, const std::string& gitHash, const std::vector<std::string>& gitStatus, const std::map<std::string,std::string>& libraryVersions);

	friend Environment sphysics::py::environment_setstate(const pybind11::tuple& state);
};
}

#endif
