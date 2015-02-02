/*
 * smd_kernel_functions.h
 *
 *  Created on: Jan 28, 2015
 *      Author: ganzenmueller
 */

#ifndef SMD_KERNEL_FUNCTIONS_H_
#define SMD_KERNEL_FUNCTIONS_H_

namespace SMD_Kernels {
static inline double Kernel_Wendland_Quintic_NotNormalized(const double r, const double h) {
	if (r < h) {
		double q = 2.0 * r / h;
		return pow(1.0 - 0.5 * q, 4) * (2.0 * q + 1.0);
	} else {
		return 0.0;
	}
}

static inline double Kernel_Barbara(const double r, const double h) {
	double arg = (1.570796327 * (r + h)) / h;
	double hsq = h * h;
	//wf = (1.680351548 * (cos(arg) + 1.)) / hsq;
	return -2.639490040 * sin(arg) / (hsq * h);
}

//static inline void spiky_kernel_and_derivative(const double h, const double r, const int dimension, double &wf, double &wfd) {
//
//	/*
//	 * Spiky kernel
//	 */
//
//	if (r > h) {
//		//printf("r=%f > h=%f in Spiky kernel\n", r, h);
//		wf = wfd = 0.0;
//		return;
//	}
//
//	double hr = h - r; // [m]
//	if (dimension == 2) {
//		double n = 0.3141592654e0 * h * h * h * h * h; // [m^5]
//		wfd = -3.0e0 * hr * hr / n; // [m*m/m^5] = [1/m^3] ==> correct for dW/dr in 2D
//		wf = -0.333333333333e0 * hr * wfd; // [m/m^3] ==> [1/m^2] correct for W in 2D
//	} else {
//		wfd = -14.0323944878e0 * hr * hr / (h * h * h * h * h * h); // [1/m^4] ==> correct for dW/dr in 3D
//		wf = -0.333333333333e0 * hr * wfd; // [m/m^4] ==> [1/m^3] correct for W in 3D
//	}
//
//}

static inline void barbara_kernel_and_derivative(const double h, const double r, const int dimension, double &wf, double &wfd) {

	/*
	 * Barbara kernel
	 */

	double arg = (1.570796327 * (r + h)) / h;
	double hsq = h * h;

	if (r > h) {
		printf("r = %f > h = %f in barbara kernel function\n", r, h);
		exit(1);
		//wf = wfd = 0.0;
		//return;
	}

	if (dimension == 2) {
		wf = (1.680351548 * (cos(arg) + 1.)) / hsq;
		wfd = -2.639490040 * sin(arg) / (hsq * h);
	} else {
		wf = 2.051578323 * (cos(arg) + 1.) / (hsq * h);
		wfd = -3.222611694 * sin(arg) / (hsq * hsq);
	}
}
}

#endif /* SMD_KERNEL_FUNCTIONS_H_ */
