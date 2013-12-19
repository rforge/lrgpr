/*
 * informationCriteria.h
 *
 *  Created on: Aug 14, 2012
 *      Author: gh258
 */

#ifndef INFORMATIONCRITERIA_H_
#define INFORMATIONCRITERIA_H_

/* See /home/likewise-open/CB/gh258/common/docs/projects/new_information_criteria/informationCriteria.pdf
	 * For more information about these information criteria
	 */

#include <gsl/gsl_sf.h>

#include <math.h>
#include <vector>

using namespace std;

class modelScore {
public:
	double log_L, AIC, BIC, EBICe, EBIC, RIC, mRIC, Cauchy_BIC, mBIC, mBIC1, mBIC2, mBIC3;
	double df, c;
	int n, p;

	// constructor
	modelScore( double log_L_in, double df_in, int n_in, int p_in, double c_in){

		//log_L = -fabs(log_L2 - log_L1);
		//double k = fabs(df1 - df2);

		double k = df;

		df = df_in;
		n = n_in;
		p = p_in;
		c = c_in;
		log_L = log_L_in;

		// AIC = -2\log L + 2k
		AIC = -2*log_L + 2 * k;

		// BIC = -2\log L + k\log n
		BIC = -2*log_L + k*log(n);

		// EBIC = -2\log L + k\log n + 2\gamma\log \binom{p}{k} \quad     where  } \gamma > 1 - \frac{\log n}{2\log p}
		double gamma = 1 - log(n)/(2*log(p));
		EBICe = -2*log_L + k*log(n) + 2*gamma*gsl_sf_lnchoose(p, k);

		// EBIC with gamma = 1
		// this has a flat prior on model size
		EBIC = -2*log_L + k*log(n) + 2*gsl_sf_lnchoose(p, k);

		// RIC = -2\log L + 2k\log p
		RIC = -2*log_L + 2*k*log(p) ;

		// mRIC = -2\log L + 2k\log p - 2\log k!
		mRIC = -2*log_L + 2*k*log(p) - 2*gsl_sf_lnfact( k );

		// Cauchy-BIC = -2\log L + k\log n + \log \Big[ \sqrt{\pi} / \big[2^{(p-k)/2} \Gamma\big(\frac{p-k+1}{2}\big) \big] \Big]
		//Cauchy_BIC = 0; //-2*log_L + k*log(n) + 0.5 * log(M_PI) - (p-k)/2.0*log(2) - gsl_sf_lngamma( (p-k+1)/2 );

		// mBIC = -2\log L + k\log n + 2k\log p - 2k\log c
		mBIC = -2*log_L + k*log(n) + 2*k*log(p) - 2*k*log(c);

		// mBIC1 = -2\log L + k\log n + 2k\log p - 2k\log c - 2\log k!- \sum_{i=1}^k \log\log np^2/i^2
		double sum = 0;
		for(int i=1; i<=k; i++) sum += log(log(n*p*p/(i*i)));
		mBIC1 = -2*log_L + k*log(n) + 2*k*log(p) - 2*k*log(c) - 2*gsl_sf_lnfact( k ) - sum;

		// mBIC2 = -2\log L + k\log n + 2k\log p - 2k\log c - 2\log k!
		mBIC2 = -2*log_L + k*log(n) + 2*k*log(p) - 2*k*log(c) - 2*gsl_sf_lnfact( k );

		// mBIC3 = -2\log L + k\log n + 2k\log p - 2k\log c - 2k\log k
		mBIC3 = -2*log_L + k*log(n) + 2*k*log(p) - 2*k*log(c) - 2*k*log(k);
	}

	// default constructor
	modelScore(){};

	vector<double> get_criteria(){
		vector<double> scores;

		scores.push_back(log_L);
		//scores.push_back(AIC);
		//scores.push_back(BIC);
		//scores.push_back(EBICe);
		//scores.push_back(EBIC);
		//scores.push_back(RIC);
		//scores.push_back(mRIC);
		//scores.push_back(Cauchy_BIC);
		//scores.push_back(mBIC);
		//scores.push_back(mBIC1);
		//scores.push_back(mBIC2);
		//scores.push_back(mBIC3);

		return scores;
	}

	vector<string> get_criteria_names(){
		vector<string> names;

		names.push_back("logLik");
		//names.push_back("AIC");
		//names.push_back("BIC");
		//names.push_back("EBICe");
		//names.push_back("EBIC");
		//names.push_back("RIC");
		//names.push_back("mRIC");
		//names.push_back("Cauchy_BIC");
		//names.push_back("mBIC");
		//names.push_back("mBIC1");
		//names.push_back("mBIC2");
		//names.push_back("mBIC3");

		return names;
	}

};


#endif /* INFORMATIONCRITERIA_H_ */
