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
		double q = 2.0 * r/h;
		return pow(1.0 - 0.5*q, 4) * (2.0 * q + 1.0);
	}

	static inline double Kernel_Barbara(const double r, const double h) {
		double arg = (1.570796327 * (r + h)) / h;
		double hsq = h * h;
		//wf = (1.680351548 * (cos(arg) + 1.)) / hsq;
		return -2.639490040 * sin(arg) / (hsq * h);
	}
}



#endif /* SMD_KERNEL_FUNCTIONS_H_ */
