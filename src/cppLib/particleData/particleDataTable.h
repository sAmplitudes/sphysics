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
// boost::bimap replaced by std::map, reading of decay-mode files removed
// (2023-05-02), class renamed to ParticleDataTable (2023-05-04), getNames()
// added (2023-05-12).


#ifndef PARTICLEDATATABLE_H
#define PARTICLEDATATABLE_H


#include <string>
#include <vector>
#include <map>
#include <set>

#include "particleProperties.h"


namespace sphysics {
namespace constants{


	class ParticleDataTable {

	public:

		static ParticleDataTable& instance() { return _instance; }  ///< get singleton instance

		static bool isInTable(const std::string& partName);  ///< returns, whether particle has a table entry

		static const ParticleProperties* entry(const std::string& partName,
		                                       const bool         warnIfNotExistent = true);  ///< access properties by particle name

		static bool addEntry(const ParticleProperties& partProp);  ///< adds entry to particle data table

		static std::vector<const ParticleProperties*>
		entriesMatching(const ParticleProperties&            prototype,
		                const std::string&                   sel,
		                const double                         minMass            = 0,
		                const double                         minMassWidthFactor = 0,
		                const std::vector<std::string>&      whiteList          = std::vector<std::string>(),
		                const std::vector<std::string>&      blackList          = std::vector<std::string>(),
		                const ParticleProperties::decayMode& decay              = ParticleProperties::decayMode(),
		                const bool&                          forceDecayCheck    = true);  ///< returns entries that have the same quantum numbers as prototype property; quantum numbers to be compared are selected by sel string; if minMass > 0 the isobar mass is limited; checks for allowed decays if they are defined; decay checks can be forced, then particles which have no specified decays will be discarded

		static unsigned int nmbEntries() { return _dataTable.size(); }  ///< returns number of entries in particle data table

		typedef std::map<std::string, ParticleProperties>::const_iterator iterator;
		static iterator begin() { return _dataTable.begin(); }  ///< returns iterator pointing at first entry of particle data table
		static iterator end()   { return _dataTable.end();   }  ///< returns iterator pointing after last entry of particle data table
		static std::vector<std::string> getNames(); ///< return particle names

		static std::ostream& print(std::ostream& out);  ///< prints particle data in human-readable form
		static void print() { print(std::cout); }
		static std::ostream& dump (std::ostream& out);  ///< dumps particle properties in format of data file

		static bool readFile(const std::string& fileNameList = "./ParticleDataTable.txt");  ///< reads in particle data from file, an optional file name for particle decays can be added after ";"
		static bool read(std::istream& in);  ///< reads whitespace separated properties from stream

		static std::string  particleNameFromGeantId(const int id);
		static void         geantIdAndChargeFromParticleName(const std::string& name,
		                                                    int&               id,
		                                                    int&               charge);
		static unsigned int geantIdFromParticleName(const std::string& name);

		static void clear() { _dataTable.clear(); }  ///< deletes all entries in particle data table

		static bool debug() { return _debug; }                             ///< returns debug flag
		static void setDebug(const bool debug = true) { _debug = debug; }  ///< sets debug flag


	private:

		ParticleDataTable () { }
		~ParticleDataTable() { }
		ParticleDataTable (const ParticleDataTable&);
		ParticleDataTable& operator =(const ParticleDataTable&);

		static ParticleDataTable                         _instance;   ///< singleton instance
		static std::map<std::string, ParticleProperties> _dataTable;  ///< map with particle data

		static std::map<std::string, int> _nameGeantIdMap; ///< map with translation particle name <> GeantId

		static bool _debug;  ///< if set to true, debug messages are printed

	};


	inline
	std::ostream&
	operator <<(std::ostream&            out,
	            const ParticleDataTable& dataTable)
	{
		return dataTable.print(out);
	}


	inline
	std::istream&
	operator >>(std::istream&      in,
	            ParticleDataTable& dataTable)
	{
		dataTable.read(in);
		return in;
	}

}
} // namespace sphysics


#endif  // PARTICLEDATATABLE_H
