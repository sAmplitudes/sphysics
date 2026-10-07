/*
 * environment.cc
 *
 *  Created on: Apr 22, 2020
 *      Author: stefan
 */



#include <sstream>
#include <iomanip>

#include "environment.h"
#include "environmentStatus.h"


sphysics::Environment::Environment():
_version(sphysics_VERSION),
_gitHash(GIT_HASH)
{
	{
		const std::string gitStatusString = GIT_STATUS;
		const std::string sep = "; ";
		size_t i=0;
		while(i < gitStatusString.size()){
			size_t iNext = gitStatusString.find(sep, i);
			if (iNext == std::string::npos) iNext = gitStatusString.size();
			_gitStatus.push_back(gitStatusString.substr(i, iNext-i));
			i = iNext + sep.size();
		}
	}

	_libraryVersions["Tensorflow"] = Tensorflow_VERSION;

}

void
sphysics::Environment::set(const std::string& version, const std::string& gitHash, const std::vector<std::string>& gitStatus, const std::map<std::string,std::string>& libraryVersions)
{
	_version = version;
	_gitHash = gitHash;
	_gitStatus = gitStatus;
	_libraryVersions = libraryVersions;
}


bool sphysics::Environment::isGitStatusClean() const {
	return _gitStatus.size() == 0;
}

std::string
sphysics::Environment::getSummary() const
{
	std::stringstream out;
	out << "Compilation environment:" << std::endl;
	out << '\t' << "sphysics software version:" << std::endl;
	out << '\t' << '\t' << "Version:   " << getVersion() << "" << std::endl;
	out << '\t' << '\t' << "Git hash: '" << getGitHash() << "'" << std::endl;
	out << '\t' << '\t' << (isGitStatusClean()? "Workspace clean": "Uncommited changes!")<< std::endl;
	out << '\t' << "Library versions:" << std::endl;
	for(const auto& lib: _libraryVersions) out << "\t\t" << std::left << std::setw(15) << lib.first << ": " << lib.second << std::endl;
	return out.str();
}
