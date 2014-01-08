#ifndef MISC_FUNCTIONS_H
#define MISC_FUNCTIONS_H

/*
 * file_functions.h
 *
 *  Created on: Aug 13, 2012
 *      Author: gh258
 */

#include <string>
#include <vector>
#include <iostream>
#include <ostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <cstdlib>

using namespace std;

// returns TRUE if file or path exists, else FALSE
bool fileExists(const string &str);

// returns TRUE if folder exists, else FALSE
bool folderExists(const string &str);

// return the number of space-delimited items in the line
int count_items_in_line(const string & line);

// for the file denoted by file_name, read the file to detect the number of rows and columns
// and return these values by reference
// if file does not exist, exit
void detect_file_dimensions(const string & file_name, int & n_rows, int &n_cols);

/*! /brief Give the dimensions of a matrix stored in a file, given a filename
 *
 * \param file file string a matrix in space or tab delimited text format
 * \param nrow return the # of rows in matrix
 * \param ncol return the # of columns in matrix
 */
void get_matrix_dimensions( const string &file, unsigned int *nrow, unsigned int *ncol);

/*! \brief Count the number of tokens in string str using the specified delimiter and the C function strtok()
 *
 * \param str The string to be tokenized
 * \param delimiters String of possible delimiters between tokens
 */
long count_tokens(const char *str_in, const char *delimiters);


// converts int to string
// opposite of atoi
string itoa(const int & val);

// returns the absolute value of largest pairwise difference between the doubles in the vector
double largest_pairwise_difference(const vector<double> & list);

int index_with_largest_value(const vector<double> &vect);

template <class myType>
void print_vector(const vector<myType> &vect, const string sep = " "){
	for(unsigned int i=0; i<vect.size(); i++){
		cout << vect[i] << sep;
	}
	cout << endl;
}

template <class myType1, class myType2>
void print_vector(const vector<pair<myType1,myType2> > &vect, const string sep = "\n"){
	for(unsigned int i=0; i<vect.size(); i++){
		cout << vect[i].first << " " << vect[i].second << sep;
	}
	cout << endl;
}

template <class myType>
void save_vector(const vector<myType> &vect, const string &file_name, const string sep = "\n"){

	ofstream file( file_name.c_str() );

	for(unsigned int i=0; i<vect.size(); i++){
		file << vect[i] << sep;
	}
	file.close();
}


// return the slope
double slope(const int &x_val_1, const int &x_val_2, const double &y_val_1, const double &y_val_2);

double newton_raphson_step(const vector<double> & list, const int & lag, const int & jump);

template <class myType>
bool is_in_vector(const vector<myType> &vect, myType value){
	bool found_value = false;
	for(unsigned int i=0; i<vect.size(); i++){
		if(vect[i] == value){
			found_value = true;
			break;
		}
	}
	return(found_value);
}

// loc is the location where value is found, or the last index if not found
template <class myType>
bool is_in_vector(const vector<myType> &vect, myType value, int &loc){
	bool found_value = false;
	for(loc=0; loc<(int) vect.size(); loc++){
		if(vect[loc] == value){
			found_value = true;
			break;
		}
	}
	return(found_value);
}

template <class myType1, class myType2>
bool is_in_vector(const vector<pair<myType1,myType2> > &vect, myType1 value){
	bool found_value = false;
	for(unsigned int i=0; i<vect.size(); i++){
		if(vect[i].first == value){
			found_value = true;
			break;
		}
	}
	return(found_value);
}

template <typename myType1, typename myType2>
bool is_in_array(myType1 *array, const int length, myType2 value){
	bool found_value = false;
	for(int i=0; i<length; i++){
		if(array[i] == (myType1) value){
			found_value = true;
			break;
		}
	}
	return(found_value);
}

// find the index of the vector whose entry has the greatest value
template<class myType>
int which_max(const vector<myType> &vect){
	int index = 0;
	double max_value = vect[0];
	for(unsigned int i=0; i<vect.size(); i++){
		if(vect[i] > max_value){
			max_value = vect[i];
			index = i;
		}
	}
	return(index);
}

// find the index of the vector whose entry has the smallest value
template<class myType>
int which_min(const vector<myType> &vect){
	int index = 0;
	double min_value = vect[0];
	for(unsigned int i=0; i<vect.size(); i++){
		if(vect[i] < min_value){
			min_value = vect[i];
			index = i;
		}
	}
	return(index);
}


template<class myType>
bool is_sorted(const vector<myType> &vect, const string &direction){

	if( direction == "increasing" ){
		for(unsigned int i=1; i<vect.size(); i++){
			if( vect[i] < vect[i-1] ){
				return false;
			}
		}
	}else if( direction == "decreasing" ){
		for(unsigned int i=1; i<vect.size(); i++){
			if( vect[i] > vect[i-1] ){
				return false;
			}
		}
	}else{
		cout << "Call to is_sorted(), has invalid direction: " << direction << endl;
		exit(1);
	}

	return true;
}

// convert a double into a string
// Limit length at nchr
template<typename myType>
inline string stringify(const myType& x, int nchr = 0) {
  ostringstream o;
  o << x;

  if(nchr == 0){
	  return o.str();
  }else{
	  return o.str().substr(0, nchr);
  }
}

// return a string with count number of tabs
string multitab(const int count);


// return the number of times target occurs in str
int count_occurrences(const string & str, const string & target);


// returns time interval in seconds
double timediff(struct timeval *starttime, struct timeval *finishtime);


template<typename myType>
myType *convert_vector_to_array(const vector<myType> vect){

	myType *array = (myType *) malloc(sizeof(myType) * vect.size());

	for(unsigned int i=0; i<vect.size(); i++){
		array[i] = vect[i];
	}
	return array;
}

// return sign(v)
template<typename myType>
int sign(myType v){
	if(v < 0){
		return -1;
	}else if(v > 0){
		return 1;
	}else{
		return 0;
	}
}

// return true iff str = '0'...'9'
bool isDigit( const string str);

/*!
 * \brief Parse a space or tab delimited string of integers and return a vector<int> with the integers as entries
 * \param str Space or tab delimited string of integers
 * \return vector<int> with the integers as entries
 */
vector<int> parse_string_to_vector(const string &str);

/*! \brief Determine if a string contails letter characters
 *
 * \param str The string to be parsed
 * \retrun true if the string contains a-z or A-Z
 */
bool contains_letters( const string & str);
bool is_column_header( const string & line);


/*! \brief Given a character string, returns true iff it is numeric
 * From http://rosettacode.org/wiki/Determine_if_a_string_is_numeric#C.2B.2B
 */
bool isNumeric( const char* pszInput );


/*! \brief Print progress bar like 5/10 [========-------] 50 %
 * \param count numerator
 * \param total denominator
 * \param length number of characters in progress bar
 * \param start_time time in seconds that process was started
 * \return string with progress bar
 */
string print_progress( int count, int total, int length, time_t start_time = 0);


void process_mem_usage(double& vm_usage, double& resident_set);

/*! brief Read in an STL vector of values as myType which is specified at the function call
 *
 */
template<typename myType>
vector<myType> read_vector(const string &file){

	myType entry;

	vector<myType> v;

	/////////////////////////
	// Read and store data //
	/////////////////////////

	ifstream file_stream(file.c_str());

	while(file_stream >> entry){
		v.push_back( entry );
	}

	file_stream.close();

	return v;
}

/*! brief Concatenate two vectors
 *
 */
template<typename myType>
vector<myType> vector_bind(const vector<myType> a, const vector<myType> b){

	vector<myType> result;

	for(unsigned int i=0; i<a.size(); i++){
		result.push_back( a[i] );
	}
	for(unsigned int i=0; i<b.size(); i++){
		result.push_back( b[i] );
	}

	return result;
}


/*!
 * \brief Return vector with unique entries, sorted
 */
template<typename myType>
vector<myType> unique( const vector<myType> &v){

	vector<myType> v_unique = v;

	// Sort entries in vector and remove duplicates
	sort(  v_unique.begin(), v_unique.end() );
	v_unique.erase( unique( v_unique.begin(), v_unique.end() ), v_unique.end() );

	return v_unique;
}

/*!
 * \brief Convert time in seconds to h, m, s format
 */
string timeToString( double time_elapsed_seconds );

/**
 * \brief Construct vector that is c(minValue, maxValue)[-indeces]
 * \param minValue inclusive
 * \param maxValue inclusive
 */
vector<int> vector_complement( const vector<int> & indeces, const int minValue, const int maxValue);


template<typename myType>
bool any_is_na( myType *array, int len){

	bool naFound = false;

	for(int i=0; i<len; i++){
		if( isnan(array[i]) && ! isfinite(array[i]) ){
			naFound = true;
			break;
		}
	}

	return naFound;
}


template <class myType, class myType2>
vector<int> which( const myType &A, const myType2 value ){

	vector<int> idx;

	for(int i=0; i<A.size(); i++){
		if( A[i] == value ){
			idx.push_back( i );
		}
	}
	return idx;
}

template <class myType>
vector<myType> get_values( const vector<myType> &dcmp_features, const vector<int> &idx){

	vector<myType> result;

	for(int i=0; i<idx.size(); i++){
		// if there is a negative entry, return an empty vector
		if( idx[i] < 0){ 
			return vector<myType>();
		} 
		result.push_back( dcmp_features[idx[i]] );
	}

	return result;
}


#endif
