///////////////////////////////////////////////////////////////////////////
//
//    Copyright 2010
//
//    This file is part of rootpwa
//
//    rootpwa is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    rootpwa is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with rootpwa. If not, see <http://www.gnu.org/licenses/>.
//
///////////////////////////////////////////////////////////////////////////
//-------------------------------------------------------------------------
//
// Description:
//      singleton class that manages all particle data
//
//
// Author List:
//      Boris Grube          TUM            (original author)
//
//
//-------------------------------------------------------------------------
//
// Modifications in sphysics: namespace rpwa replaced by sphysics::constants,
// boost::bimap replaced by std::map, reading of decay-mode files (libconfig)
// removed, ROOTPWA's reporting macros replaced by std::cout and std::cerr
// (2023-05-02), classes renamed to ParticleDataTable and ParticleProperties
// (2023-05-04), getNames() added (2023-05-12), format of the log output changed
// (2024-03-22).


#include <fstream>
#include <iomanip>
#include <iterator>


#include "particleDataTable.h"


using namespace std;
using namespace sphysics::constants;


namespace {

	typedef map<string, int> nameGeantIdBimap;
	nameGeantIdBimap initNameGeantIdTranslator()
	{
		nameGeantIdBimap translator;
		translator["gamma0"]=       1;
		translator["e+"]=           2;
		translator["e-"]=           3;
		translator["mu+"]=          5;
		translator["mu-"]=          6;
		translator["pi0"]=          7;
		translator["pi+"]=          8;
		translator["pi-"]=          9;
		translator["K_L"]=         10;// K_S and K_L not in ParticleDataTable! Replacing with K0 or Kbar0 may cause problems for states containing both particle and antiparticle
		translator["K+"]=          11;
		translator["K-"]=          12;
		translator["n0"]=          13;
		translator["p+"]=          14;
		translator["pbar-"]=       15;
		translator["K_S"]=         16;// K_S and K_L not in ParticleDataTable! Replacing with K0 or Kbar0 may cause problems for states containing both particle and antiparticle
		translator["eta0"]=        17;
		translator["Lambda0"]=     18;
		translator["nbar0"]=       25;
		translator["Lambdabar0"]=  26;
		translator["rho(770)0"]=   57;
		translator["rho(770)+"]=   58;
		translator["rho(770)-"]=   59;
		translator["omega(782)0"]= 60;
		translator["eta'(982)0"]=  61;
		translator["phi(1020)0"]=  62;
		return translator;
	}

}


ParticleDataTable               ParticleDataTable::_instance;
map<string, ParticleProperties> ParticleDataTable::_dataTable;
bool                            ParticleDataTable::_debug = false;
nameGeantIdBimap                ParticleDataTable::_nameGeantIdMap = initNameGeantIdTranslator();


string ParticleDataTable::particleNameFromGeantId(const int id) {
	std::string nameFromId;
	for(const auto& [nameElement, idElement]: _nameGeantIdMap){
		if (idElement == id) {
			nameFromId = nameElement;
			break;
		}
	}
	if (nameFromId.size() == 0) {
		std::cerr << id << " is unknown GEANT particle ID. returning particle name 'unknown'." << endl;
		return "unknown";
	}
	// make sure charge is correctly put into particle name
	int charge;
	const string bareName = ParticleProperties::chargeFromName(nameFromId, charge);
	return ParticleProperties::nameWithCharge(bareName, charge);
}


void ParticleDataTable::geantIdAndChargeFromParticleName(const string& name,
                                                         int&          id,
                                                         int&          charge)
{
	id = 0;
	const string bareName = ParticleProperties::chargeFromName(name, charge);
	if (_nameGeantIdMap.count(name) > 0){
		id = _nameGeantIdMap.at(name);
	} else if (_nameGeantIdMap.count(bareName) > 0){
		id = _nameGeantIdMap.at(bareName);
	} else {
		std::cerr << "particle '" << name << "' is unknown. returning GEANT particle ID 0." << endl;
		return;
	}
}


unsigned int ParticleDataTable::geantIdFromParticleName(const std::string& name) {
	int retval;
	int tmp;
	geantIdAndChargeFromParticleName(name, retval, tmp);
	return retval;
}


bool
ParticleDataTable::isInTable(const string& partName)
{
	if (_dataTable.find(partName) == _dataTable.end()) {
		return false;
	} else
		return true;
}


const ParticleProperties*
ParticleDataTable::entry(const string& partName,
                         const bool    warnIfNotExistent)
{
	iterator i = _dataTable.find(partName);
	if (i == _dataTable.end()) {
		if (warnIfNotExistent)
			std::cout << "could not find entry for particle '" << partName << "'" << endl;
		return 0;
	} else
		return &(i->second);
}


vector<const ParticleProperties*>
ParticleDataTable::entriesMatching(const ParticleProperties&            prototype,
                                   const string&                        sel,
                                   const double                         minMass,
                                   const double                         minMassWidthFactor,
                                   const vector<string>&                whiteList,
                                   const vector<string>&                blackList,
                                   const ParticleProperties::decayMode& decay,
                                   const bool&                          forceDecayCheck)
{
	const pair<ParticleProperties, string> selector(prototype, sel);
	vector<const ParticleProperties*>      matchingEntries;
	for (iterator i = _dataTable.begin(); i != _dataTable.end(); ++i) {
		if (i->second != selector)
			continue;
		const ParticleProperties& partProp = i->second;
		// limit isobar mass, if minMass > 0
		if ((minMass > 0) and (partProp.mass() + minMassWidthFactor * partProp.width() < minMass))
		{
			if(_debug)std::cout << partProp.name() << " not in mass window " << flush;
			continue;
		}
		// apply white list
		bool whiteListMatch = (whiteList.size() == 0) ? true : false;
		for (size_t j = 0; j < whiteList.size(); ++j)
			if ((partProp.name() == whiteList[j]) or (partProp.bareName() == whiteList[j])) {
				whiteListMatch = true;
				break;
			}
		if (not whiteListMatch){
			if(_debug)std::cout << partProp.name() << " not in whitelist " << endl;
			continue;
		}
		// apply black list
		bool blackListMatch = false;
		for (size_t j = 0; j < blackList.size(); ++j)
			if ((partProp.name() == blackList[j]) or (partProp.bareName() == blackList[j])) {
				blackListMatch = true;
				break;
			}
		if (blackListMatch) {
			if(_debug)std::cout << partProp.name() << " on blacklist " << endl;
			continue;
		}
		// apply list of decays
		bool decayMatch = true;
		if ((decay._daughters.size() > 0) and partProp.nmbDecays() > 0) {
			if (not partProp.hasDecay(decay))
				decayMatch = false;
		} else if (forceDecayCheck)
			decayMatch = false;
		if (not decayMatch) {
			if (_debug)
				std::cout << partProp.name() << " does not have a decay into " << decay << endl;
			continue;
		}

		if (_debug) {
			std::cout << "found entry " << partProp.name() << " matching " << prototype
				<< " and '" << sel << "'" << flush;
			if (minMass > 0)
				cout << " with mass > " << minMass - minMassWidthFactor * partProp.width() << " GeV";
			if (whiteList.size() > 0)
				cout << " ; in white list";
			if (blackList.size() > 0)
				cout << " ; not in black list";
			if (decayMatch)
				cout << " ; with allowed decay into " << decay << endl;
		}
		matchingEntries.push_back(&partProp);
	}
	return matchingEntries;
}


bool
ParticleDataTable::addEntry(const ParticleProperties& partProp)
{
	const string name = partProp.name();
	iterator     i    = _dataTable.find(name);
	if (i != _dataTable.end()) {
		std::cout << "trying to add entry for particle '" << name << "' "
		          << "which already exists in table"     << endl
		          << "    existing entry: " << i->second << endl
		          << "    conflicts with: " << partProp  << endl
		          << "    entry was not added to table." << endl;
		return false;
	} else {
		_dataTable[name] = partProp;
		if (_debug)
			std::cout << "added entry for '" << name << "' into particle data table" << endl;
		return true;
	}
}



std::vector<std::string>
ParticleDataTable::getNames()
{
	std::vector<std::string> names;
	names.reserve(_dataTable.size());
	for( const auto& [name, _]: _dataTable){
		names.push_back(name);
	}
	return names;
}


ostream&
ParticleDataTable::print(ostream& out)
{
	unsigned int countEntries = 0;
	for (iterator i = begin(); i != end(); ++i) {
		++countEntries;
		out << "entry " << setw(3) << countEntries << ": " << i->second << endl;
	}
	return out;
}


ostream&
ParticleDataTable::dump(ostream& out)
{
	for (iterator i = begin(); i != end(); ++i) {
		i->second.dump(out);
		out << endl;
	}
	return out;
}


bool
ParticleDataTable::readFile(const string& fileNameList)
{
	string fileName = fileNameList;
	std::cout << "[INF: sphysics  ] ";
	std::cout << "Reading particle data from '" << fileName << "'" << endl;
	ifstream file(fileName.c_str());
	if (not file or not file.good()) {
		std::cout << "cannot open file '" << fileName << "'" << endl;
		return false;
	}
	bool success = read(file);
	return success;
}


bool
ParticleDataTable::read(istream& in)
{
	if (not in or not in.good()) {
		std::cout << "cannot read from input stream" << endl;
		return false;
	}
	if (_debug)
		std::cout << "data table has " << nmbEntries() << " entries (before reading)" << endl;
	unsigned int countEntries = 0;
	while (in.good()) {
		ParticleProperties partProp;
		if (in >> partProp) {
			if (addEntry(partProp))
				++countEntries;
			if (not partProp.isItsOwnAntiPart() and addEntry(partProp.antiPartProperties()))
				++countEntries;
		}
	}
	if (_debug) {
		std::cout << "read " << countEntries << " new entries into particle data table" << endl;
		cout << "    data table has " << nmbEntries() << " entries (after reading)" << endl;
	}
	return true;
}

