
static Rcpp::Function asMatrix("as.matrix");

class Rexpress {

	private:
	string expression, query;
	Environment env;
	NumericMatrix X_data;
	NumericVector One;
	Formula form;

	public:
	
	Rexpress( const string &expression_, const NumericMatrix &X_data_, Environment &env_){
		
		expression 	= expression_;
		X_data 		= X_data_;
		env 		= env_;
		query = ".lrgpr_tmp";
		form = Formula(expression);

		// One = rep(1, nrow(X))
		One = NumericVector( X_data.nrow() );
		std::fill(One.begin(), One.end(), 1);
	}

	Rexpress( const string &expression_, const int n_indivs, Environment &env_){
		
		expression 	= expression_;
		env 		= env_;
		query = ".lrgpr_tmp";
		form = Formula(expression);

		// One = rep(1, nrow(X))
		One = NumericVector( n_indivs );
		std::fill(One.begin(), One.end(), 1);
	}

	/*RcppGSL::matrix<double> get_model_matrix( const int j ){

		// SNP = X[,j]
		env[".lrgpr_tmp"] = X_data(_,j);

		Language call( "model.matrix.default", form, env);	
		return call.eval( env );
	}*/

	RcppGSL::matrix<double> get_model_matrix_clean(){

		env[".lrgpr_tmp"] = One;

		Language call( "model.matrix.default", form, env);	
		return call.eval( env );
	}

	// Get the matrix for the null model
	RcppGSL::matrix<double> get_model_matrix_null(){

		// replace query SNP with nothing
		Language call1( ".mm_replace_query", form, query);	
		string expressionLocal = as<string>( call1.eval( env ) );

		// evaluate formula
		Formula form2( expressionLocal );
		Language call2( "model.matrix.default", form2, env);	
		return call2.eval( env );
	}

	RcppGSL::vector<double> get_response(){	

		env[".lrgpr_tmp"] = One;

		Language call( ".mm_get_response", form, env);
		return call.eval( env ); 
	}

	RcppGSL::matrix<double> get_response_m(){
		
		env[".lrgpr_tmp"] = One;
		
		Language call( ".mm_get_response", form, env);
		return call.eval( env ); 
	}

	vector<int> get_loop_terms(){		

		Language call( ".mm_get_terms", form, query, env);
		NumericVector loopIndex = call.eval( env ); 

		return as<vector<int> >(loopIndex);
	}

	string get_expression(){
		return expression;
	}
};
