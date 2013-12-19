/*
 * file_functions.h
 *
 *  Created on: Aug 13, 2012
 *      Author: gh258
 */

#include "misc_functions.h"

#include <vector>
#include <string>

#include <sys/stat.h>
#include <string>
#include <string.h>
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <limits>

using namespace std;

// returns TRUE if file or path exists, else FALSE
bool fileExists(const string &str){
	struct stat st;
	return(stat(str.c_str(),&st) == 0);
}


bool folderExists(const string &str){
	struct stat st;
	return(stat(str.c_str(),&st) == 0);
}

int count_items_in_line(const string & line){
	unsigned int count = 0, pos=0;

	unsigned int found_pos = line.find(" ", pos);

	while(found_pos != string::npos){
		pos = found_pos + 1;
		found_pos = line.find(" ", pos);
		count++;
	}

	//cout << line << "|"<< count <<endl;
	// n_items = n_spaces + 1
	return(count)+1;
}

void detect_file_dimensions(const string & file_name, int & n_rows, int &n_cols){
	if(!fileExists(file_name)){
		#ifdef PRINT_TO_CONSOLE
		cout<<"Does not exist: "<< file_name <<endl;
		exit(1);
		#endif
	}

	// initialize file stream
	ifstream file_stream;
	file_stream.open(file_name.c_str());

	string line;

	getline(file_stream, line);

	n_cols = count_items_in_line(line);
	n_rows = 1;
	while(getline(file_stream, line)){
		// if a line does not have the same number of items as the first line
		if(n_cols != count_items_in_line(line)){
			#ifdef PRINT_TO_CONSOLE
			cout<<"n_items in a line is inconsistent: "<< file_name <<endl;
			exit(1);
			#endif
		}
		n_rows++;
	}
}

void get_matrix_dimensions( const string &file, unsigned int *nrow, unsigned int *ncol){

	ifstream file_stream;

	if( fileExists( file ) ){
		// initialize file stream
		file_stream.open( file.c_str() );
	}else{

		#ifdef PRINT_TO_CONSOLE
		cout << "Does not exist: "<< file <<endl;
		exit(1);
		#endif
	}

	int n_tokens = 0;
	int n_lines = 1;

	const char *delimiters = " \t";
	string line;

	while( ! getline(file_stream, line).eof() ){
		if( n_tokens == 0 ){
			n_tokens = count_tokens( line.c_str(), delimiters);
		}else{
			if( n_tokens != count_tokens( line.c_str(), delimiters) ){

				#ifdef PRINT_TO_CONSOLE
				cout << "Error in file: " << file << endl;
				cout << "There are " << n_tokens << "values in line 1," << endl;
				cout << "but " << count_tokens( line.c_str(), delimiters) << "values in line " << n_lines + 1 << endl;
				exit(1);
				#endif
			}
		}
		n_lines++;
	}

	file_stream.close();

	// n_lines over counts by 1, but I am not sure why
	*nrow = n_lines - 1;
	*ncol = n_tokens;
}

long count_tokens(const char *str_in, const char *delimiters){

	//cout << "str_in: " << str_in << endl;

	// copy str_in
	size_t n = strlen( str_in );

	// need to add one to account for terminating \0 at end of character string
	char *str = (char *) malloc( sizeof(char) * n + 1);

	strcpy(str, str_in);

	// Delimiter is either space or tab
	char *pch = strtok( str, delimiters);

	long count = 0;

	// Tokenize until end is reached
	while ( pch != NULL )	{
		pch = strtok (NULL, delimiters);

		count++;
	}

	//cout << "n: " << n << endl;
	//cout << "count: " << count << endl;
	//cout << "strlen( str ): " << strlen( str ) << endl;

	free( str );

	return count;
}

// converts int to string
// opposite of atoi
string itoa(const int & val){
	char str[10];
	sprintf(str, "%d", val);
	return(string(str));

}

// returns the absolute value of largest pairwise difference between the doubles in the vector
double largest_pairwise_difference(const vector<double> & list){
	double max_diff = 0, curr_diff;

	for(unsigned int i=0; i<list.size(); i++){
		for(unsigned int j=i; j<list.size(); j++){
			curr_diff = fabs(list[i] - list[j]);
			if(curr_diff > max_diff){
				max_diff = curr_diff;
			}
		}
	}
	return(max_diff);
}

int index_with_largest_value(const vector<double> &vect){
	double max_value = -numeric_limits<double>::max();
	int max_index = -1;
	for(unsigned int i=0; i<vect.size(); i++){
		if(vect[i] > max_value){
			max_value = vect[i];
			max_index = i;
		}
	}
	return(max_index);
}

/* template <class myType>
void print_vector(const vector<myType> &vect){
	for(int i=0; i<vect.size(); i++){
		cout<<vect[i] << " ";
	}
	cout << endl;
}*/

// return the slope
double slope(const int &x_val_1, const int &x_val_2, const double &y_val_1, const double &y_val_2){
	return( (y_val_2 - y_val_1) / (x_val_2 - x_val_1) );
}


double newton_raphson_step(const vector<double> & list, const int & lag, const int & jump){
	int last_index = list.size() - 1;
	double m = slope(last_index - lag, last_index, list[last_index - lag],  list[last_index]);

	//cout << endl << last_index<< " " << m << " " << list[last_index - lag] << " " << list[last_index] <<endl;

	return( list[last_index] +  jump * m);

}

/*template <class myType>
bool is_in_vector(const vector<myType> &vect, myType value){
	bool found_value = false;
	for(int i=0; i<vect.size(); i++){
		if(vect[i] == value){
			found_value = true;
		}
	}
	return(found_value);

}
*/


string multitab(const int count){
	string str = "";

	for(int i=0; i<count; i++){
		str += "\t";
	}
	return(str);
}



/* string.find() return 2 when matching "abcd" vs "cf" since 'c' is found in index 2
 *	i.e. it returns the index of the last matching char even if it isn't a full match
 *	Therefore, check that the target and the string found are equal
 */
int count_occurrences(const string & str, const string & target){

	unsigned int count = 0;
	unsigned int pos = str.find_first_of(target, 0);

	while(pos != string::npos && str.substr(pos, target.size()) == target ){

		pos = str.find_first_of(target, pos+1 );
		count++;
	}
	return count;
}

double timediff(struct timeval *starttime, struct timeval *finishtime){
  long msec;
  msec = (finishtime->tv_sec-starttime->tv_sec)*1000;
  msec += (finishtime->tv_usec-starttime->tv_usec)/1000;

  return msec/(double) 1000;
}

bool isDigit( const string str){

	if( str == "0" || str == "1" || str == "2" || str == "3" || str == "4" || str == "5" || str == "6" || str == "7" || str == "8" || str == "9"){
		return true;
	}
	return false;
}


vector<int> parse_string_to_vector(const string &str){

	// append a space to the end of str so that the parsing code is simpler
	string str_local = str + " ";

	vector<int> return_vector;

	// initialize to -1
	int found_num = -1, found_space = -1;

	while(1){

		// search for the next digit starting at 1 position following the previous space
		found_num 		= str_local.find_first_of("0123456789",found_space + 1);

		// search for the next space starting at 1 position following the previous digit
		found_space 	= str_local.find_first_of(" \t", found_num + 1);

		//cout << found_num << " " << found_space - found_num << endl;

		// if the found_num index is valid (i.e. is not off the end of the string
		if(found_num >= 0){

			// push integer onto the vector
			return_vector.push_back( atoi(str_local.substr(found_num, found_space - found_num).c_str()) );
		}else{
			break;
		}
	}

	return return_vector;
}

bool contains_letters( const string & str){

	bool result = false;

	for(unsigned int i=0; i<str.size(); i++){
		if( (str[i] >= 65 && str[i] <= 90) ||  (str[i] >= 97 && str[i] <= 122)){

			#ifdef PRINT_TO_CONSOLE
			cout << "Is a letter: "<<  str[i]<<endl;
			#endif
			result = true;
			break;
		}
	}

	return result;
}

bool isNumeric( const char* pszInput ){

	istringstream iss( pszInput );

	double dTestSink;
	iss >> dTestSink;

	// was any input successfully consumed/converted?
	if ( ! iss ) return false;

	// was all the input successfully consumed/converted?
	return ( iss.rdbuf()->in_avail() == 0 );
}

bool is_column_header( const string & line){

	char *pch;
	const char *delimiters = " \t";

	// Initialize tokenizer
	// const_cast can be used because the string is not accessed again until a new line is read
	pch = strtok( const_cast<char *>(line.c_str()), delimiters);

	do{

		if( ! isNumeric( pch ) ) return true;

		pch = strtok (NULL, delimiters);

	}while ( pch != NULL );

	return false;

}


void print_progress( int count, int total, int length, time_t start_time){

	double fraction = count / (double) total;

	string bar_string = "";

	int i;

	for(i=0; i<length*fraction; i++){
		bar_string += "=";
	}

	for(int j=i; j<length; j++){
		bar_string += "-";
	}

	#ifdef PRINT_TO_CONSOLE
	cout << "\r" << count << "/" << total << " [" << bar_string << "] ";
	printf("%2.2f", fraction*100);
	cout << "%" << flush;
	#endif
	// if start_time is not the default value
	if( start_time != 0 && count > 0){

		// time_elapsed / fraction = total_time
		// time_remaining = total_time - time_elapsed = time_elapsed / fraction - time_elapsed

		time_t current_time;
		time (&current_time);

		double time_elapsed = difftime (current_time, start_time);
		double time_remaining =  time_elapsed / fraction - time_elapsed;

		double hours, minutes, seconds;

		// if operation is complete, print elapsed time
		if( count == total ){
			hours = floor( time_elapsed / 3600);
			minutes = floor( (time_elapsed - hours * 3600) / 60);
			seconds = floor( time_elapsed - hours * 3600 - minutes*60 );
		}else{
			hours = floor( time_remaining / 3600);
			minutes = floor( (time_remaining - hours * 3600) / 60);
			seconds = floor( time_remaining - hours * 3600 - minutes*60 );
		}
		#ifdef PRINT_TO_CONSOLE
		cout << "    " << hours << "h " << minutes << "m " << seconds << "s  ";
		#endif
		//cout << "      " << hours << "h " << minutes << "m           ";
	}


}

#include <unistd.h>
#include <ios>
#include <iostream>
#include <fstream>

void process_mem_usage(double& vm_usage, double& resident_set){
   using std::ios_base;
   using std::ifstream;
   using std::string;

   vm_usage     = 0.0;
   resident_set = 0.0;

   // 'file' stat seems to give the most reliable results
   //
   ifstream stat_stream("/proc/self/stat",ios_base::in);

   // dummy vars for leading entries in stat that we don't care about
   //
   string pid, comm, state, ppid, pgrp, session, tty_nr;
   string tpgid, flags, minflt, cminflt, majflt, cmajflt;
   string utime, stime, cutime, cstime, priority, nice;
   string O, itrealvalue, starttime;

   // the two fields we want
   //
   unsigned long vsize;
   long rss;

   stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
               >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
               >> utime >> stime >> cutime >> cstime >> priority >> nice
               >> O >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest

   stat_stream.close();

   long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
   vm_usage     = vsize / 1024.0;
   resident_set = rss * page_size_kb;

   vm_usage /= 1024.0;
   resident_set /= 1024.0;

}


string timeToString( double time_elapsed_seconds ){

	double hours = floor( time_elapsed_seconds / 3600);
	double minutes = floor( (time_elapsed_seconds - hours * 3600) / 60);
	double seconds = floor( time_elapsed_seconds - hours * 3600 - minutes*60 );

	stringstream str;
	str << "    " << hours << "h " << minutes << "m " << seconds << "s  ";

	return str.str();
}

vector<int> vector_complement( const vector<int> &indeces, const int minValue, const int maxValue){

	vector<int> result;

	for(int i=0; i<=maxValue; i++){
		if( ! is_in_vector( indeces, i) ){
			result.push_back( i );
		}
	}

	return result;
}

