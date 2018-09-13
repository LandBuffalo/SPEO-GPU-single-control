#ifndef __CEC2014_H__
#define __CEC2014_H__

#include <math.h>
#include "config.h"
using namespace std;
#define INF 1.0e99
#define EPS 1.0e-14
#define E  2.7182818284590452353602874713526625
#define PI 3.1415926535897932384626433832795029
class  CEC2014
{
private:
	void sphere_func(double *, double *, int, double *, double *, int, int, double, double); /* Sphere */
	void ellips_func(double *, double *, int, double *, double *, int, int, double, double); /* Ellipsoidal */
	void bent_cigar_func(double *, double *, int, double *, double *, int, int, double, double); /* Discus */
	void discus_func(double *, double *, int, double *, double *, int, int, double, double);  /* Bent_Cigar */
	void dif_powers_func(double *, double *, int, double *, double *, int, int, double, double);  /* Different Powers */
	void rosenbrock_func(double *, double *, int, double *, double *, int, int, double, double); /* Rosenbrock's */
	void schaffer_F7_func(double *, double *, int, double *, double *, int, int, double, double); /* Schwefel's F7 */
	void ackley_func(double *, double *, int, double *, double *, int, int, double, double); /* Ackley's */
	void rastrigin_func(double *, double *, int, double *, double *, int, int, double, double); /* Rastrigin's  */
	void weierstrass_func(double *, double *, int, double *, double *, int, int, double, double); /* Weierstrass's  */
	void griewank_func(double *, double *, int, double *, double *, int, int, double, double); /* Griewank's  */
	void schwefel_func(double *, double *, int, double *, double *, int, int, double, double); /* Schwefel's */
	void katsuura_func(double *, double *, int, double *, double *, int, int, double, double); /* Katsuura */
	void bi_rastrigin_func(double *, double *, int, double *, double *, int, int, double, double); /* Lunacek Bi_rastrigin */
	void grie_rosen_func(double *, double *, int, double *, double *, int, int, double, double); /* Griewank-Rosenbrock  */
	void escaffer6_func(double *, double *, int, double *, double *, int, int, double, double); /* Expanded Scaffer??s F6  */
	void step_rastrigin_func(double *, double *, int, double *, double *, int, int, double, double); /* Noncontinuous Rastrigin's  */
	void happycat_func(double *, double *, int, double *, double *, int, int, double, double); /* HappyCat */
	void hgbat_func(double *, double *, int, double *, double *, int, int, double, double); /* HGBat  */

	void hf01(double *, double *, int, double *, double *, int *, int, int); /* Hybrid Function 1 */
	void hf02(double *, double *, int, double *, double *, int *, int, int); /* Hybrid Function 2 */
	void hf03(double *, double *, int, double *, double *, int *, int, int); /* Hybrid Function 3 */
	void hf04(double *, double *, int, double *, double *, int *, int, int); /* Hybrid Function 4 */
	void hf05(double *, double *, int, double *, double *, int *, int, int); /* Hybrid Function 5 */
	void hf06(double *, double *, int, double *, double *, int *, int, int); /* Hybrid Function 6 */

	void cf01(double *, double *, int, double *, double *, int); /* Composition Function 1 */
	void cf02(double *, double *, int, double *, double *, int); /* Composition Function 2 */
	void cf03(double *, double *, int, double *, double *, int); /* Composition Function 3 */
	void cf04(double *, double *, int, double *, double *, int); /* Composition Function 4 */
	void cf05(double *, double *, int, double *, double *, int); /* Composition Function 5 */
	void cf06(double *, double *, int, double *, double *, int); /* Composition Function 6 */
	void cf07(double *, double *, int, double *, double *, int *, int); /* Composition Function 7 */
	void cf08(double *, double *, int, double *, double *, int *, int); /* Composition Function 8 */

	void shiftfunc(double*, double*, int, double*);
	void rotatefunc(double*, double*, int, double*);
	void sr_func(double *, double *, int, double*, double*, double, int, int); /* shift and rotate */
	void asyfunc(double *, double *x, int, double);
	void oszfunc(double *, double *, int);
	void cf_cal(double *, double *, int, double *, double *, double *, double *, int);

	double 							*OShift;
	double							*Mdata;
	double							*ye;
	double							*ze;
	double							*x_bound;
	int 								*SS;

	int									func_id_;
	int            			dim_;
	int 								cf_func_num_;
	int                 cf_index_;

	double 							*pop_original_;
public:
	CEC2014();			//construction function of CEC2014, it sets member variables
	~CEC2014();
	int								Initilize(int func_id, int dim);
	int								Unitilize();
	double							EvaluateFitness(const vector<double> & elements);
	double 							EvaluateFitness(double * elements);
};

#endif
