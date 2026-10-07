/**
 * @file constantsBase.h
 * @author Stefan Wallner (swallner@mpp.mpg.de)
 * @brief C++ implementaiton of base class for constants
 * @date 2022-10-31
 * 
 * @copyright Copyright (c) 2022 the sphysics authors (GPL-3.0-or-later, see LICENSE)
 * 
 */
#ifndef SPHYSICS_CONSTANTS_SPECIALCONSTANTS_H
#define SPHYSICS_CONSTANTS_SPECIALCONSTANTS_H


#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>


namespace sphysics {
	namespace constants {
		template <typename T> class SpecialConstantsBase;
	
		namespace helper {
			/**
			 * @brief Simple helper class that allows us to do partial template specialization needed for getData
			 */
			template <typename Tval, typename Tc>
			class HelperGetData{
				public: static std::map<std::string, Tval> getData(Tc*);
				public: static const std::map<std::string, Tval> getData(const Tc*);
			};
			template <typename Tc>
			class HelperGetData<double, Tc>{
				public: static std::map<std::string, double>& getData(Tc* constants) {return constants->_dataDouble;}
				public: static const std::map<std::string, double>& getData(const Tc* constants) {return constants->_dataDouble;}
			};
			template <typename Tc>
			class HelperGetData<int, Tc>{
				public: static std::map<std::string, int>& getData(Tc* constants) {return constants->_dataInt;}
				public: static const std::map<std::string, int>& getData(const Tc* constants) {return constants->_dataInt;}
			};
			template <typename Tc>
			class HelperGetData<std::string, Tc>{
				public: static std::map<std::string, std::string>& getData(Tc* constants) {return constants->_dataString;}
				public: static const std::map<std::string, std::string>& getData(const Tc* constants) {return constants->_dataString;}
			};

			template <typename Tval>
			void printData(std::stringstream& stream, const std::map<std::string, Tval>& values, const std::map<std::string,std::string>& descriptions);
		}


		template <typename T>
		class SpecialConstantsBase {
			template<typename,typename> friend class helper::HelperGetData;
		public:
			static const T& getInstance() {
				static const T data;
				return data;
			}

			/**
			 * @brief Get all keys stored in this SpecialConstants object
			 * 
			 * @return std::vector<std::string>  all keys
			 */
			static std::vector<std::string> keys() {return getInstance()._keys();}

			/**
			 * @brief Check key is stored in this SpecialConstants object
			 * 
			 * @param key  Key to check
			 * @return true  Key is stored
			 * @return false  Key is not stored
			 */
			static bool hasKey(const std::string& key){return getInstance()._hasKey(key);}

			/**
			 * @brief Check key is stored with the given type in this SpecialConstants object
			 * 
			 * @param key  Key to check
			 * @return true  Key is stored
			 * @return false  Key is not stored
			 */
			template <typename Tval>
			static bool hasKeyOfType(const std::string& key);

			/**
			 * @brief Get value of the constant `key` with type `Tval`
			 * 
			 * @tparam Tval  Type of the constant
			 * @param key  Name of the constant
			 * @return Tval Value of the constant
			 */
			template <typename Tval>
			static Tval get(const std::string& key){return getInstance().template _getData<Tval>().at(key);}

			static const std::string& getConstantDescription(const std::string& key){return getInstance()._descriptions.at(key);}

			static const std::string& getDescription(){return getInstance()._description;}

			static const std::string& getName(){return getInstance()._name;}

			/**
			 * @brief Get all constants of the given type
			 * 
			 * @tparam Tval 
			 * @return const std::map<std::string, Tval>& 
			 */
			template <typename Tval>
			static const std::map<std::string, Tval>& getConstantsOfType(){return getInstance().template _getData<Tval>();}


			static std::string toString();

			static void print(){std::cout << toString() << std::endl;}

			SpecialConstantsBase(const SpecialConstantsBase&) = delete;
			SpecialConstantsBase(const SpecialConstantsBase&&) = delete;
			SpecialConstantsBase& operator=(const SpecialConstantsBase&) = delete;
			SpecialConstantsBase& operator=(const SpecialConstantsBase&&) = delete;


		protected:
			SpecialConstantsBase(){}
			~SpecialConstantsBase(){}

			/**
			 * @brief Get the data object corresponding to the given type T2
			 * 
			 * @tparam T2 type of stored data
			 * @return std::map<std::string, T2>& 
			 */
			template <typename Tval>
			std::map<std::string, Tval>& _getData(){ return helper::HelperGetData<Tval, SpecialConstantsBase<T>>::getData(this);}
			template <typename Tval>
			const std::map<std::string, Tval>& _getData()const{ return helper::HelperGetData<Tval, SpecialConstantsBase<T>>::getData(this);}

			static const std::map<std::string, std::string>& getConstantDescriptions(){return getInstance()._descriptions;}


			/**
			 * @brief Set new constatn
			 * 
			 * @tparam Tval  Type of the contant value
			 * @param key  Name of the constant
			 * @param val  Value of the constant
			 * @param desc  Description of the constant
			 */
			template <typename Tval>
			void _set(const std::string& key, const Tval& val, const std::string& desc);

			void _setName(const std::string& name) { _name = name; }

			void _setDescription(const std::string& description) { _description = description; }


			/**
			 * @brief Get all keys stored in this SpecialConstants object
			 * 
			 * @return std::vector<std::string>  all keys
			 */
			std::vector<std::string> _keys() const;


			/**
			 * @brief Check key is stored in this SpecialConstants object
			 * 
			 * @param key  Key to check
			 * @return true  Key is stored
			 * @return false  Key is not stored
			 */
			bool _hasKey(const std::string& key) const;

		private:
			std::string _name;
			std::string _description;
			std::map<std::string,double> _dataDouble;
			std::map<std::string,int> _dataInt;
			std::map<std::string,std::string> _dataString;
			std::map<std::string,std::string> _descriptions;


		};
	}
}

template <typename T>
std::vector<std::string>
sphysics::constants::SpecialConstantsBase<T>::_keys() const
{
	std::vector<std::string> keys;
	for(const auto& key_value: _descriptions) keys.push_back(key_value.first);
	return keys;
}


template <typename T>
bool
sphysics::constants::SpecialConstantsBase<T>::_hasKey(const std::string& key) const
{
	return _descriptions.find(key) != _descriptions.end();
}


template <typename T>
template <typename Tval>
bool
sphysics::constants::SpecialConstantsBase<T>::hasKeyOfType(const std::string& key)
{
	const std::map<std::string, Tval>& data = getInstance().template _getData<Tval>();
	return data.find(key) != data.end();
}


template <typename T>
template <typename Tval>
void
sphysics::constants::SpecialConstantsBase<T>::_set(const std::string& key, const Tval& val, const std::string& desc)
{
	if (_hasKey(key))
		throw std::invalid_argument("Key already exists!");
	_getData<Tval>()[key] = val;
	_descriptions[key] = desc;
}


template <typename Tval>
void
sphysics::constants::helper::printData(std::stringstream& stream, const std::map<std::string, Tval>& values, const std::map<std::string,std::string>& descriptions)
{
	for(const auto& key_val: values){
		stream << '\n' << '\t' << key_val.first << '\t' << key_val.second << '\t' << descriptions.at(key_val.first);
	}

}


template <typename T>
std::string
sphysics::constants::SpecialConstantsBase<T>::toString()
{
	std::stringstream stream;
	stream << getName();
	const std::string& desc = getDescription();
	size_t pos = 0;
	while (pos < desc.size()){
		size_t next = desc.find('\n', pos);
		if (next == std::string::npos) next = desc.size();
		stream << "\n\t" << desc.substr(pos, next-pos );
		pos = next+1;
	}
	helper::printData(stream, getConstantsOfType<double>(), getConstantDescriptions());
	helper::printData(stream, getConstantsOfType<int>(), getConstantDescriptions());
	helper::printData(stream, getConstantsOfType<std::string>(), getConstantDescriptions());
	
	return stream.str();
}

#endif