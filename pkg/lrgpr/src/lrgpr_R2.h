

RcppExport SEXP R_getAlleleVariance2( SEXP data_, SEXP pBigMat_, SEXP nthreads_, SEXP quiet_){

BEGIN_RCPP

	int nthreads = as<int>( nthreads_ );
	bool quiet = as<int>( quiet_ );
	
	featureBatcher fbatch( data_, pBigMat_, 10000);

	// Set threads to 1
	omp_set_num_threads( nthreads );
	// Intel paralellism
	#ifdef INTEL
	mkl_set_num_threads( 1 );
	#endif
	// disable nested OpenMP parallelism
	omp_set_nested(0);

	long batch_size = MAX( 1, fbatch.getBatchSize()/100.0 ); 

	long tests_completed = 0;
	Progress p(0, false);

	time_t start_time;
	time(&start_time);

	vector<double> colVariance( fbatch.get_N_features() );

	for(int i_set=0; i_set<fbatch.get_N_features(); i_set+=fbatch.getBatchSize()){

		// Load data from binary matrix (or do noting if NumericMatrix is used)
		fbatch.loadNextChunk();

		#pragma omp parallel
		{		
			gsl_vector *marker_j = gsl_vector_alloc( fbatch.get_N_indivs() ); 

			#pragma omp for schedule(static, batch_size)			
			for(int j=0; j<fbatch.getBatchSize(); j++){			

				#pragma omp critical
				tests_completed++;

				fbatch.getFeautureInChunk( j, marker_j );	

				// Compute varaince for non-missing entries
				gsl_vector *v = gsl_vector_get_nonmissing( marker_j );

				if( v == NULL ) colVariance[j+i_set] = -1;
				else colVariance[j+i_set] = gsl_stats_variance( v->data, v->stride, v->size);

				gsl_vector_free(v);

			} // END for

			gsl_vector_free(marker_j);

		} // End parallel

		if( ! quiet )  Rcpp::Rcout << print_progress( tests_completed, fbatch.get_N_features(), 25, start_time);

		if( Progress::check_abort() ){
			return wrap(0);
		}
	} // End set loop

	if( ! quiet ) Rcpp::Rcout << endl;

	return( wrap(colVariance) );

END_RCPP
}