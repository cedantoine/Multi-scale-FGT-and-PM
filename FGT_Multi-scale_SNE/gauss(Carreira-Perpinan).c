/*
 * N-Body Methods
 * Copyright (C) 2006 Dustin Lang, Mike Klass, Firas Hamze, Anthony Lee
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#include "gauss(Carreira-Perpinan).h"


// double* xat  								in		x coordinates of sources
// double* yat        							in		y coordinates of sources
// double* str                                  in		weights = a given column (=dimension) of X
// int natoms                                   in		number of sources
// double* xtarg 								in		x coordinates of targets
// double* ytarg								in		y coordinates of targets
// double* pot      		   				    out		results (=output)
// int ntarg                                    in		number of targets
// double delta                                 in		sigma^2 = 1/scale^2
// int nterms                                   in		p = number of expansion terms
// int kdis										in		number of boxes to truncate, default set to 4
// int nfmax									in		max number of source points in each box for expansion, default set to 5
// int nlmax									in		max number of ?target? points in each box for expansion, default set to 5
// double r										in      default set to 1/2
// int* typeCount								out		number of iterations for exact, local, far and far/far expansion (=output)
void gausst(double* xat, double* yat, double* str, int natoms, double* xtarg,
            double* ytarg, double* pot, int ntarg, double delta, int nterms,
            int kdis, int nfmax, int nlmax, double r, int* typeCount) {
	// note no workspace stuff here, all allocated in lenchk and in
	// gafexp as needed
	int j, ntbox, nside;
	int nmax; // maximum between number of targets and sources
	double boxdim; // size of the box
    
	// dynamically allocated in lenchk
	int* iloc, *itofst, *itradr, *ibox;
	double* center, ****locexp;
    
	if (natoms > ntarg)
		nmax = natoms;
	else
		nmax = ntarg;
    
	boxdim = sqrt(0.5 * delta) * r; // size of each box -> regarder page 970 paper sur code
	nside = (int) (1.0 / boxdim) + 1; // number of boxes per side?
	printf("Number of boxes per side : %d \n", nside);
    
	//double errConst = computeError(r, nterms);
	double errConst = 0;
	double errConst2 = 0;
	// allocate iloc, locexp, ibox, center, itofst, itradr and set up
	// iloc. the int ntbox is assigned there also
	lenchk(nside, xtarg, ytarg, ntarg, nterms, nmax, &iloc, &ibox, &itofst,
           &itradr, &locexp, &center, &ntbox);
    
	// zero the potentials (?=similarities?) //assigns zero to all output values
	for (j = 0; j < ntarg; j++)
		pot[j] = 0;
    
	// call gafexp
	gafexp(xat, yat, str, natoms, xtarg, ytarg, ntarg, delta, nside, nterms,
           iloc, ntbox, nfmax, nlmax, pot, kdis, ibox, itradr, itofst, center,
           locexp, typeCount, errConst, errConst2);
    
	gaeval(xtarg, ytarg, pot, ntarg, locexp, nterms, ibox, center, delta, iloc,
           nlmax, nside, itofst, itradr);
    
	/*
	 alloced_mem.valid = 0;
	 */
    
	// out with the trash
	freestuff(iloc, ibox, itofst, itradr, locexp, center, nterms, ntbox);
	return;
}

// Computes direct interaction between (xp,yp) and all point in the current analysed box -> (xar, yat) (=influence from all (xar,yar) on (xp,yp))
// double xp							in		x coordinate of a given target point
// double yp							in		y coordinate of a given target point
// double* xat							in		x coordinates of sources in a given box
// double* yat							in		y coordinates of sources in a given box
// double natoms						in		number of sources in the given box (box currently computed)
// double* str							in		weights
// double* dpot							out		potentials to update (=output)
// double delta							in		sigma^2 = 1/scale^2 si on utilise sigma=1
void gdir(double xp, double yp, double* xat, double* yat, int natoms,
          double* str, double* dpot, double delta) {
	double x, y, z;
	int j;
    
	for (j = 0; j < natoms; j++) {
		x = xp - xat[j];
		y = yp - yat[j];
		z = 0;
		*dpot = *dpot + exp(-(x * x + y * y + z * z) / delta) * str[j];
	}
}

// double*** ffexp
// int nterms
// double* cent
// double xpoint,
// double ypoint
// double zpoint
// double* pot
// double delta
void gamval(double*** ffexp, int nterms, double* cent, double xpoint,
            double ypoint, double* pot, double delta) {
	int i, j, k, i2;
	double facx, facy, facz, x, y, z, x2, y2, z2, dsq, potinc, hi, hj;
	double* hexpx, *hexpy, *hexpz;
    
	hexpx = (double*) malloc((nterms + 1) * sizeof(double));
	hexpy = (double*) malloc((nterms + 1) * sizeof(double));
	hexpz = (double*) malloc((nterms + 1) * sizeof(double));
    
	dsq = 1.0 / sqrt(delta);
	x = (xpoint - cent[0]) * dsq;
	y = (ypoint - cent[1]) * dsq;
	z = (0 - 0.08333333333333332871) * dsq;
    
	x2 = 2.0 * x;
	y2 = 2.0 * y;
	z2 = 2.0 * z;
    
	facx = exp(-x * x);
	facy = exp(-y * y);
	facz = exp(-z * z);
    
	// inintialize
	hexpx[0] = facx;
	hexpy[0] = facy;
	hexpz[0] = facz;
    
	hexpx[1] = x2 * facx;
	hexpy[1] = y2 * facy;
	hexpz[1] = z2 * facz;
    
	for (i = 1; i < nterms; i++) {
		i2 = 2 * i;
		hexpx[i + 1] = x2 * hexpx[i] - i2 * hexpx[i - 1];
		hexpy[i + 1] = y2 * hexpy[i] - i2 * hexpy[i - 1];
		hexpz[i + 1] = z2 * hexpz[i] - i2 * hexpz[i - 1];
	}
    
	potinc = 0.0;
	for (i = 0; i < (nterms + 1); i++) {
		hi = hexpx[i];
		for (j = 0; j < (nterms + 1); j++) {
			hj = hexpy[j];
			for (k = 0; k < (nterms + 1); k++) {
				potinc = potinc + hi * hj * hexpz[k] * ffexp[i][j][k];
			}
		}
	}
    
	*pot = *pot + potinc;
    
	free(hexpx);
	free(hexpy);
	free(hexpz);
    
}

// double* cent							in		centroid of the box
// double* xatoms, yatoms, zatoms		in		sources sorted according to the group belonging
// double* str							in		w of the points in the given box
// int ninbox							in		number of source points
// double*** ffexp					 	out		far field expansion
// int nterms							in		p
// double delta							in		sigma^2
void gamkxp(double* cent, double* xatoms, double* yatoms, double* str,
            int ninbox, double*** ffexp, int nterms, double delta) {
	int i, j, k, l;
	double dsq, x, y, z, xp;
	double* zp, *yp;
    
	zp = (double*) malloc((nterms + 1) * sizeof(double));
	yp = (double*) malloc((nterms + 1) * sizeof(double));
    
	for (i = 0; i < (nterms + 1); i++) {
		for (j = 0; j < (nterms + 1); j++) {
			for (k = 0; k < (nterms + 1); k++) {
				ffexp[i][j][k] = 0.0;
			}
		}
	}
    
	dsq = 1.0 / sqrt(delta); // 1/sigma
	for (i = 0; i < ninbox; i++) {
		x = (xatoms[i] - cent[0]) * dsq;
		y = (yatoms[i] - cent[1]) * dsq;
		z = (0 - 0.08333333333333332871) * dsq;
        
		zp[0] = 1.0;
		yp[0] = 1.0;
		for (k = 1; k < (nterms + 1); k++) {
			zp[k] = zp[k - 1] * z / k;
			yp[k] = yp[k - 1] * y / k;
		}
		xp = str[i];
		for (j = 0; j < (nterms + 1); j++) {
			for (k = 0; k < (nterms + 1); k++) {
				for (l = 0; l < (nterms + 1); l++) {
					ffexp[j][k][l] = ffexp[j][k][l] + xp * yp[k] * zp[l];
				}
			}
			xp = xp * x / (j + 1);
		}
	}
	free(zp);
	free(yp);
	return;
}

// ******************************************************************
void galval(double*** local, int nterms, double* cent, double xpoint,
            double ypoint, double* pot, double delta) {
    
	int i, j, k, l;
	double parsum1, parsum2, parsum3, x, y, z, potinc, dsq;
    
	dsq = 1.0 / (sqrt(delta));
	x = (xpoint - cent[0]) * dsq;
	y = (ypoint - cent[1]) * dsq;
	z = (0 - 0.08333333333333332871) * dsq;
    
	// using 3-d horner's rule
	potinc = 0.0;
	for (i = nterms; i > 0; i--) {
		parsum1 = 0.0;
		for (j = nterms; j > 0; j--) {
			parsum2 = 0.0;
			for (k = nterms; k > 0; k--) {
				parsum2 = (parsum2 + local[i][j][k]) * z;
			}
			parsum2 = parsum2 + local[i][j][0];
			parsum1 = (parsum1 + parsum2) * y;
		}
		parsum3 = 0.0;
		for (l = nterms; l > 0; l--) {
			parsum3 = (parsum3 + local[i][0][l]) * z;
		}
		parsum3 = parsum3 + local[i][0][0];
		potinc = (potinc + parsum3 + parsum1) * x;
	}
    
	parsum1 = 0.0;
	for (j = nterms; j > 0; j--) {
		parsum2 = 0.0;
		for (k = nterms; k > 0; k--) {
			parsum2 = (parsum2 + local[0][j][k]) * z;
		}
		parsum2 = parsum2 + local[0][j][0];
		parsum1 = (parsum1 + parsum2) * y;
	}
	potinc = potinc + parsum1;
    
	parsum1 = 0.0;
	for (k = nterms; k > 0; k--) {
		parsum1 = (parsum1 + local[0][0][k]) * z;
	}
	parsum1 = parsum1 + local[0][0][0];
	potinc = potinc + parsum1;
    
	*pot = potinc;
	return;
}


// 
// double* cent 		in		coordinates of the centroid
// double xp			in 		x coordinates of a given source point
// double yp			in		x coordinates of a given source point
// double charge		in		weight of the given source point
// double*** b			out		?
// int nterms			in		p (= number of expansion terms)
// double delta			in		sigma^2 = 1/scale^2 
void galkxp(double* cent, double xp, double yp, double charge, double*** b,
            int nterms, double delta) {
	int i, l, m, n, i2;
	double dsq, x, y, z, x2, y2, z2, facx, facy, facz, facs, hl, hm;
    
	double* hexpx, *hexpy, *hexpz;
    
	hexpx = (double*) malloc((nterms + 1) * sizeof(double));
	hexpy = (double*) malloc((nterms + 1) * sizeof(double));
	hexpz = (double*) malloc((nterms + 1) * sizeof(double));
    
	dsq = 1.0 / sqrt(delta);
    
	// MAYBE WRONG!! I reversed them- makes sense see paper
	x = (cent[0] - xp) * dsq;
	x2 = 2.0 * x;
	y = (cent[1] - yp) * dsq;
	y2 = 2.0 * y;
	z = (0.08333333333333332871 - 0) * dsq;
	z2 = 2.0 * z;
    
	facx = exp(-x * x) * charge;
	facy = exp(-y * y);
	facz = exp(-z * z);
    
	hexpx[0] = facx;
	hexpy[0] = facy;
	hexpz[0] = facz;
    
	hexpx[1] = x2 * facx;
	hexpy[1] = y2 * facy;
	hexpz[1] = z2 * facz;
    
	for (i = 1; i < nterms; i++) {
		i2 = 2 * i;
		hexpx[i + 1] = x2 * hexpx[i] - i2 * hexpx[i - 1];
		hexpy[i + 1] = y2 * hexpy[i] - i2 * hexpy[i - 1];
		hexpz[i + 1] = z2 * hexpz[i] - i2 * hexpz[i - 1];
	}
    
	facs = 1.0;
	for (l = 0; l < nterms + 1; l++) {
		hexpx[l] = hexpx[l] * facs;
		hexpy[l] = hexpy[l] * facs;
		hexpz[l] = hexpz[l] * facs;
		facs = -1 * facs / (l + 1.0);
	}
	for (l = 0; l < nterms + 1; l++) {
		hl = hexpx[l];
		for (m = 0; m < nterms + 1; m++) {
			hm = hexpy[m];
			for (n = 0; n < nterms + 1; n++)
				b[l][m][n] = hl * hm * hexpz[n];
		}
	}
	free(hexpx);
	free(hexpy);
	free(hexpz);
	return;
}

// ******************************************************************
void gaeval(double* xtarg, double* ytarg, double* pot, int ntarg,
            double**** locexp, int nterms, int* ibox, double* center, double delta,
            int* iloc, int nlmax, int nside, int* itofst, int* itradr) {
    
	int i, k, jt, nboxes, inoff, jadr, ninbox;
	double xp, yp;
    
	nboxes = nside * nside;
	for (i = 0; i < nboxes; i++) {
		inoff = itofst[i];
		ninbox = itofst[i + 1] - inoff;
		if (ninbox > nlmax) {
			jadr = iloc[i];
			for (k = 0; k < ninbox; k++) {
				jt = itradr[inoff + k];
				xp = xtarg[jt];
				yp = ytarg[jt];
				galval(locexp[jadr], nterms, &center[2 * i], xtarg[jt],
                       ytarg[jt], &pot[jt], delta);
			}
		}
	}
	return;
}

//double* xat 								   in		x coordinates of sources
//double* yat							       in		y coordinates of sources
//double* str                                  in		weights = a given column (=dimension) of X
//int natoms                                   in		number of sources
//double* xtarg								   in		x coordinates of targets	 			
//double* ytarg 							   in		y coordinates of targets
//int ntarg                                    in		number of targets
//double delta                                 in		sigma^2 = 1/scale^2
//int nside                                    in		number of boxes per side
//int nterms                                   in		p = number of expansion terms
//int* iloc                                    in		enumerator of the target box according to the input elements.
//int ntbox                                    in		number of occupied boxes
//int nfmax                                    in		max number of source points in each box for expansion
//int nlmax                                    in		min number of target points in the box for expansion
//double* pot                                  out		[1 x ntarg] results (=output)
//int kdis                                     in		number of boxes to truncate
//int* ibox                                    out		[1 x nmax] assignment of box for source points
//int* itradr                                  out		[1 x nmax] index of the target points in the sorted array according to group belonging
//int* itofst                                  out		[1 x nboxes+1] offset for target boxes (look in 'assign' for explanation)
//double* center                               out		coordinates of the centroids
//double**** locexp                            out		? ->  local expansions
//int* typeCount                  			   out		number of iterations for exact, local, far and far/far expansion
//double errConst							   ?        assigned to 0
//double errConst2							   ?        assigned to 0
void gafexp(double* xat, double* yat, double* str, int natoms, double* xtarg,
            double* ytarg, int ntarg, double delta, int nside, int nterms,
            int* iloc, int ntbox, int nfmax, int nlmax, double* pot, int kdis,
            int* ibox, int* itradr, int* itofst, double* center, double**** locexp,
            int* typeCount, double errConst, double errConst2) {

	double* xatoms, *yatoms, *zatoms, *charg2;
	double cent[2], error = 0;
	double*** ffexp, ***b;
	double xp, yp;
	int i, j, k, l, nboxes, maxpts, ioff, iadr, inoff, nnbors, ninnbr, jt,
    iadloc, ninbox;
	int*ioffst, *nbors;
	double*** temp1, ***temp2;

	int temp_bis = 0;
	int temp_bis_2 = 0;

	nboxes = nside * nside;
	if (natoms > ntarg) {
		maxpts = natoms;
	} else {
		maxpts = ntarg;
	}
    
	// dynamic allocations?
	ioffst = (int*) malloc((nboxes + 1) * sizeof(int));
	nbors = (int*) malloc(nboxes * sizeof(int));
	xatoms = (double*) malloc(natoms * sizeof(double));
	yatoms = (double*) malloc(natoms * sizeof(double));
	zatoms = (double*) malloc(natoms * sizeof(double));
	charg2 = (double*) malloc(natoms * sizeof(double));
    
	// locexp is allocated in lenchk
    
	// allocating the ffexp
	ffexp = (double***) malloc((nterms + 1) * sizeof(double**));
	for (i = 0; i < (nterms + 1); i++) {
		ffexp[i] = (double **) malloc((nterms + 1) * sizeof(double*));
		for (j = 0; j < (nterms + 1); j++)
			ffexp[i][j] = (double *) malloc((nterms + 1) * sizeof(double));
	}
    
	// allocating b, temp1, temp2
	b = (double***) malloc((nterms + 1) * sizeof(double**));
	temp1 = (double***) malloc((nterms + 1) * sizeof(double**));
	temp2 = (double***) malloc((nterms + 1) * sizeof(double**));
	for (i = 0; i < (nterms + 1); i++) {
		b[i] = (double**) malloc((nterms + 1) * sizeof(double*));
		temp1[i] = (double**) malloc((nterms + 1) * sizeof(double*));
		temp2[i] = (double**) malloc((nterms + 1) * sizeof(double*));
		for (j = 0; j < (nterms + 1); j++) {
			b[i][j] = (double*) malloc((nterms + 1) * sizeof(double));
			temp1[i][j] = (double*) malloc((nterms + 1) * sizeof(double));
			temp2[i][j] = (double*) malloc((nterms + 1) * sizeof(double));
		}
	}
    
	// call assign:
	assign(nside, xat, yat, str, xatoms, yatoms, charg2, ioffst, xtarg, ytarg,
           ntarg, ibox, natoms, center, itradr, itofst);
	// zero the local expansions
	for (i = 0; i < ntbox; i++)
		for (j = 0; j < (nterms + 1); j++)
			for (k = 0; k < (nterms + 1); k++)
				for (l = 0; l < (nterms + 1); l++)
					locexp[i][j][k][l] = 0;
    
	for (l = 0; l < 4; l++)
		typeCount[l] = 0;
    
	for (i = 0; i < nboxes; i++) { // for every box
		ioff = ioffst[i]; //= sum of the number of points in boxes j for i<j
		ninbox = ioffst[i + 1] - ioff;  //number of source points in the box i+1
		if ((ninbox <= nfmax) && (ninbox > 0)) { // if the number of source points is in (0, nfmax]
			// No far-field expansion
			mknbor(i, nbors, &nnbors, nside, kdis); // select target boxes around given source box
			for (j = 0; j < nnbors; j++) {
				iadr = nbors[j];    			// index of neighboring box
				inoff = itofst[iadr];
				ninnbr = itofst[iadr + 1] - inoff; // number of target point in the box iadr+1
                
				if (ninnbr <= nlmax) { // if the number of target points is smaller than nlmax
					temp_bis_2++;
					// Exact interaction
					typeCount[0] += ninbox * ninnbr;
					for (k = 0; k < ninnbr; k++) { // for all the target points
						jt = itradr[inoff + k]; // recover target point
						xp = xtarg[jt];
						yp = ytarg[jt];
						gdir(xp, yp, &xatoms[ioff], &yatoms[ioff], ninbox,
                             &charg2[ioff], &pot[jt], delta);
					}
				} else {
					typeCount[1] += ninbox * ninnbr;
					// Sources in neighbors shifted to taylor series
					cent[0] = center[2 * iadr];  //x coordinate of the center of the current analysed box
					cent[1] = center[2 * iadr + 1]; //y coordinate of the center of the current analysed box
					iadloc = iloc[iadr];  //unique value of the current box (iloc is assigned in lenchl -> +- random number)
					//sumWeights = 0;
					for (k = 0; k < ninbox; k++) { // for all the source points
						galkxp(cent, xatoms[ioff + k], yatoms[ioff + k],
                               charg2[ioff + k], b, nterms, delta);
						addexp(b, locexp[iadloc], nterms);
						//sumWeights += charg2[ioff + k];
					}
					//error += sumWeights * errConst;
				}
			}
		} else if ((ninbox > nfmax) && (ninbox > 0)) {
			temp_bis++;
			// create far-field expansion
			cent[0] = center[2 * i];
			cent[1] = center[2 * i + 1];
			gamkxp(cent, &xatoms[ioff], &yatoms[ioff], &charg2[ioff], ninbox,
                   ffexp, nterms, delta);
			//sumWeights = 0;
			//for (i = 0; i < ninbox; i++)
			//	sumWeights += charg2[ioff + k];
			mknbor(i, nbors, &nnbors, nside, kdis); // select target boxes around given source box
			//error += sumWeights * errConst;
			for (j = 0; j < nnbors; j++) {
				iadr = nbors[j];
				inoff = itofst[iadr];
				ninnbr = itofst[iadr + 1] - inoff;
				if (ninnbr <= nlmax) {
					// too few targets, shift to local
					typeCount[2] += ninbox * ninnbr;
					for (k = 0; k < ninnbr; k++) {
						jt = itradr[inoff + k];
						xp = xtarg[jt];
						yp = ytarg[jt];
						gamval(ffexp, nterms, cent, xp, yp, &pot[jt], delta);
					}
				} else {
					typeCount[3] += ninbox * ninnbr;
					gshift(ffexp, cent, &center[2 * iadr], b, nterms, temp1,
                           temp2, delta);
					iadloc = iloc[iadr];
					addexp(b, locexp[iadloc], nterms);
					//error += sumWeights * errConst2;
				}
			}
		}
        // printf("temp_bis = %d\n", temp_bis);
		// printf("temp_bis_2 = %d\n", temp_bis_2);
	}
	// free ffexp, b, temp1, temp2
	for (i = 0; i < (nterms + 1); i++) {
		for (j = 0; j < (nterms + 1); j++) {
			free(ffexp[i][j]);
			free(b[i][j]);
			free(temp1[i][j]);
			free(temp2[i][j]);
		}
		free(ffexp[i]);
		free(b[i]);
		free(temp1[i]);
		free(temp2[i]);
	}
	free(ffexp);
	free(b);
	free(temp1);
	free(temp2);
    
	free(xatoms);
	free(yatoms);
	free(zatoms);
	free(charg2);
	free(ioffst);
	free(nbors);
	return;
}

double sqrtFact(unsigned int n) {
	static const double table[] = { 1, 1, 1.414213562373095, 2.449489742783178,
        4.898979485566356, 10.954451150103322, 26.832815729997478,
        70.992957397195397, 2.007984063681781e+02, 6.023952191045344e+02,
        1.904940943966505e+03, 6.317974358922328e+03, 2.188610518114176e+04,
        7.891147445080469e+04, 2.952597012800765e+05, 1.143535905863913e+06,
        4.574143623455652e+06, 1.885967730625315e+07, 8.001483428544985e+07,
        3.487765766344294e+08, 1.559776268628498e+09 };
	return table[n];
}

double computeError(double r, int p) {
	int d = 2;
	double dchoosek[] = { 1.0, 3.0, 3.0 };
	int k;
	double rp = pow(r, p);
	double c = 0;
	for (k = 0; k < d - 1; k++)
		c += dchoosek[k] * pow(1 - rp, k) * pow(rp / sqrtFact(p), d - k);
	c = c / pow(1 - r, d);
	return c;
}

double computeErrorFar(double r, int p) {
	int d = 2;
	double dchoosek[] = { 1.0, 3.0, 3.0 };
	int k;
	double rp2 = pow(sqrt(2) * r, p);
	double c = 0;
	for (k = 0; k < d - 1; k++)
		c += dchoosek[k] * pow(1 - rp2, k) * pow(rp2 / sqrtFact(p), d - k);
	c = pow(c, 2) / pow(1 - sqrt(2) * r, 2 * d);
	return c;
}

// ******************************************************************
void addexp(double***b, double***a, int nterms) {
	int i, j, k;
	double sum = 0;
	for (i = 0; i < nterms + 1; i++)
		for (j = 0; j < nterms + 1; j++)
			for (k = 0; k < nterms + 1; k++) {
				a[i][j][k] = a[i][j][k] + b[i][j][k];
				sum += a[i][j][k];
			}
	return;
}

// Assigns all point to the box they belong to (ibox), sorts all points with respect to their box (sources -> xatoms, yatoms) and keeps track of where these points are in the sorted array (for targets), allso computes offsets of targets and sources 
// nside                   in		number of boxes per side
// xat 					   in		x coordinates of sources
// yat		       	   	   in		y coordinates of sources
// str                     in		weights = a given column (=dimension) of X
// xatoms			 	   out		x coordinates of sources sorted according to the group belonging
// yatoms                  out		y coordinates of sources sorted according to the group belonging
// charg2                  out		w (=str) sorted according to group belonging
// ioffst                  out		[1 x nboxes+1] offset for source boxes -> utilisé pour classer les sources celon leur box
// xtarg				   in		x coordinates of targets
// ytarg				   in		y coordinates of targets
// ntarg                   in 		number of targets
// ibox                    out 		[1 x nmax] assignment of box for each source points
// natoms                  in  		number of sources
// center,                 out 		coordinates of the centroids (=centers of boxes)
// itradr,                 out 		[1 x nmax] index of the target points in the sorted array according to group belonging
// itofst                  out 		[1 x nboxes+1] offset for target boxes -> utilisé pour classer les targets celon leur box
void assign(int nside, double* xat, double* yat, double* str, double* xatoms,
            double* yatoms, double* charg2, int* ioffst, double* xtarg,
            double* ytarg, int ntarg, int* ibox, int natoms, double* center,
            int* itradr, int* itofst) {
    // just to be clear, to avoid wierd 3d array effects, am leaving it
    // as a 3*nboxes length array.
	int j, nboxes, ixh, iyh, iadr, indx;
	double h, hh;
	int* icnt; // number points per box
	int* icnt2; // same
	int temp = 0;
    
	nboxes = nside * nside; //total number of boxes
	icnt = (int*) malloc(nboxes * sizeof(int));
	icnt2 = (int*) malloc(nboxes * sizeof(int));
    
	for (j = 0; j < nboxes; j++) {  /////////////////////Assigning sources to boxes and sorting them
		icnt2[j] = 0;
		icnt[j] = 0;
	}
	h = 1.0 / nside;
	for (j = 0; j < natoms; j++) {
		ixh = (int) floor(xat[j] / h);
		if (ixh > nside - 1) {
			ixh = nside - 1;
		}
		if (ixh < 0) {
			ixh = 0;
		}
		iyh = (int) floor(yat[j] / h);
		if (iyh > nside - 1) {
			iyh = nside - 1;
		}
		if (iyh < 0) {
			iyh = 0;
		}
		iadr = iyh * nside + ixh; // number of box (indice-> look lenchk for explenaiton of indices)
		icnt[iadr]++; //count of points in box iadr
		ibox[j] = iadr; //assigns source j to box iadr
	}

	for (j = 0; j < nboxes; j++) {  /////////////////////Assigning sources to boxes and sorting them
		if (icnt[j] > 5){
			temp++;
		}	
	}

	// printf("%d\n", temp);
    
	ioffst[0] = 0;
	for (j = 1; j < nboxes + 1; j++)
		ioffst[j] = ioffst[j - 1] + icnt[j - 1];  //ioffst[1] = nombres de points dans box 0, ioffset[2] = nombre de points dans (box 0 + box1), ioffset[3] = nombre de points dans (box 0 + box1 + box 2), etc
    
	for (j = 0; j < natoms; j++) { // sort points according to their box belonging
		iadr = ibox[j];
		indx = ioffst[iadr] + icnt2[iadr];  //ioffst[iadr] = somme du nombre de points de tt les box[i] oû i<iadr, icnt2[iadr] = nombre de points de la box iadr que on a déja classé
		xatoms[indx] = xat[j];
		yatoms[indx] = yat[j];
		charg2[indx] = str[j];
		icnt2[iadr]++;
	}
    
	for (j = 0; j < nboxes; j++) {  ////////////////////Same as befor but for targets 
		icnt2[j] = 0;
		icnt[j] = 0;
	}
    
	for (j = 0; j < ntarg; j++) {
		ixh = (int) floor(xtarg[j] / h);
		if (ixh > nside - 1) {
			ixh = nside - 1;
		}
		if (ixh < 0) {
			ixh = 0;
		}
		iyh = (int) floor(ytarg[j] / h);
		if (iyh > nside - 1) {
			iyh = nside - 1;
		}
		if (iyh < 0) {
			iyh = 0;
		}
		iadr = iyh * nside + ixh;
		icnt[iadr]++;
		ibox[j] = iadr;
	}
    
	itofst[0] = 0;
	for (j = 1; j < nboxes + 1; j++) {
		itofst[j] = itofst[j - 1] + icnt[j - 1];
	}
    
	for (j = 0; j < ntarg; j++) {
		iadr = ibox[j];
		indx = itofst[iadr] + icnt2[iadr];
		itradr[indx] = j; // index of the target points in the sorted array -> keep track of where points are in the sorted array (targets)
		icnt2[iadr]++;
	}
    
	hh = 0.5 * h;
	for (j = 0; j < 2 * nboxes; j = j + 2) {  ////////////////////Assigning center of boxes
		iyh = (j / 2) / nside;
		ixh = (j / 2) % nside;
		center[j] = ixh * h + hh;
		center[j + 1] = iyh * h + hh;
	}
	free(icnt);
	free(icnt2);
	return;
}

// ******************************************************************
void freestuff(int* iloc, int* ibox, int* itofst, int* itradr,
               double**** locexp, double* center, int nterms, int ntbox) {
    
    // frees all the stuff allocated dynamically in lenchk
	int i, j, k;
    
	free(iloc);
	free(ibox);
	free(itofst);
	free(itradr);
    
    // free locexp
	for (i = 0; i < ntbox; i++) {
		for (j = 0; j < (nterms + 1); j++) {
			for (k = 0; k < (nterms + 1); k++) {
				free(locexp[i][j][k]);
			}
			free(locexp[i][j]);
		}
		free(locexp[i]);
	}
	free(locexp);
    
    // center
	free(center);
	return;
}

// ******************************************************************
void gshift(double*** ffexp, double* cent1, double* cent2, double*** local,
            int nterms, double*** temp1, double*** temp2, double delta) {
    
	int i,/*j,k,*/len, i2, alpha1, alpha2, alpha3, beta1, beta2, beta3;
	double dsq, x, y, z, x2, y2, z2, sum, facx, facy, facz;
	double* hexpx, *hexpy, *hexpz, *fac;
    
	hexpx = (double*) malloc((2 * nterms + 1) * sizeof(double));
	hexpy = (double*) malloc((2 * nterms + 1) * sizeof(double));
	hexpz = (double*) malloc((2 * nterms + 1) * sizeof(double));
	fac = (double*) malloc((nterms + 1) * sizeof(double));
    
	dsq = 1.0 / sqrt(delta);
	x = (cent2[0] - cent1[0]) * dsq;
	x2 = 2 * x;
	y = (cent2[1] - cent1[1]) * dsq;
	y2 = 2 * y;
	z = 0;
	z2 = 2 * z;
    
	facx = exp(-x * x);
	facy = exp(-y * y);
	facz = exp(-z * z);
    
	hexpx[0] = facx;
	hexpy[0] = facy;
	hexpz[0] = facz;
    
	hexpx[1] = x2 * facx;
	hexpy[1] = y2 * facy;
	hexpz[1] = z2 * facz;
    
	for (i = 1; i < (2 * nterms); i++) {
		i2 = 2 * i;
		hexpx[i + 1] = x2 * hexpx[i] - i2 * hexpx[i - 1];
		hexpy[i + 1] = y2 * hexpy[i] - i2 * hexpy[i - 1];
		hexpz[i + 1] = z2 * hexpz[i] - i2 * hexpz[i - 1];
	}
	fac[0] = 1.0;
	for (i = 1; i < nterms + 1; i++) {
		fac[i] = -fac[i - 1] / i;
	}
    
	len = nterms + 1;
	for (alpha1 = 0; alpha1 < len; alpha1++) {
		for (alpha2 = 0; alpha2 < len; alpha2++) {
			for (beta3 = 0; beta3 < len; beta3++) {
				sum = 0.0;
				for (alpha3 = nterms; alpha3 >= 0; alpha3--) {
					sum = sum
                    + ffexp[alpha1][alpha2][alpha3]
                    * hexpz[alpha3 + beta3];
				}
				temp1[alpha1][alpha2][beta3] = sum;
			}
		}
	}
    
	for (alpha1 = 0; alpha1 < len; alpha1++) {
		for (beta2 = 0; beta2 < len; beta2++) {
			for (beta3 = 0; beta3 < len; beta3++) {
				sum = 0.0;
				for (alpha2 = nterms; alpha2 >= 0; alpha2--) {
					sum = sum
                    + temp1[alpha1][alpha2][beta3]
                    * hexpy[alpha2 + beta2];
				}
				temp2[alpha1][beta2][beta3] = sum;
			}
		}
	}
    
	for (beta1 = 0; beta1 < len; beta1++) {
		for (beta2 = 0; beta2 < len; beta2++) {
			for (beta3 = 0; beta3 < len; beta3++) {
				sum = 0.0;
				for (alpha1 = nterms; alpha1 >= 0; alpha1--) {
					sum = sum
                    + temp2[alpha1][beta2][beta3]
                    * hexpx[alpha1 + beta1];
				}
				local[beta1][beta2][beta3] = sum * fac[beta1] * fac[beta2]
                * fac[beta3];
			}
		}
	}
    
	free(hexpx);
	free(hexpy);
	free(hexpz);
	free(fac);
	return;
}

//Computes the boxes which have to be taken into account for the coputation of the potential of points of the box ibox
//int ibox					in		current box
//int* nbors				out		index of box to whom we compute the interaction
//int* nnbors				out		number of neighboring boxes to whom we compute the interaction
//int nside					in 		(1.0/sqrt(0.5*delta))+1 -> number of boxes per side
//double kdis				in		number of boxes to truncate (i.e. boxes further away than 4 are not tken into account)
void mknbor(int ibox, int* nbors, int* nnbors, int nside, double kdis) {
    
	int i, j, irow, icol, imin, imax, jmin, jmax;
	*nnbors = 0;
	irow = ibox / nside; // same as modulo  // this is the number between 0 and 1/nside ->row in which the box is
	icol = ibox - irow * nside; // ditto -> colum in which the box is
    // threshold the x dimension
	if ((icol - kdis) > 0)
		imin = (int) floor(icol - kdis);
	else
		imin = 0;
	if ((icol + kdis) < (nside - 1))
		imax = (int) floor(icol + kdis);
	else
		imax = nside - 1;
    
    // threshold y
	if ((irow - kdis) > 0)
		jmin = (int) floor(irow - kdis);
	else
		jmin = 0;
	if ((irow + kdis) < (nside - 1))
		jmax = (int) floor(irow + kdis);
	else
		jmax = nside - 1;
    
	for (i = imin; i < imax + 1; i++) {
		for (j = jmin; j < jmax + 1; j++) {
			nbors[*nnbors] = j * nside + i;
			(*nnbors)++;
		}
	}
	return;
}

// Defines ntbox and iloc and allocates space to ibox, itofst, itradr, locexp and center
// int nside						in		number of boxes per each side
// double* xtarg, ytarg				in		normalized targets
// int ntarg						in		number of targets
// int nterms						in		p (=number of expansion terms)
// int nmax							in		maximum between number of targets and sources
// int** iloc						out		[1 x nboxes (=total number of boxes)] identifier of the box. Assigns a unique number per each box (I'm not sure why do we need this) defined in lenchk
// int** ibox						out		[1 x nmax (= in our case = number of points in data)] allocate space
// int** itofst						out		[1 x (nboxes+1)] allocate space
// int** itradr						out		[1 x ntarg (=number of targers)] allocate space
// double***** locexp				out		[ntbox x (nterms+1) x (nterms+1) x (nterms + 1)] allocate space
// double** center					out		[1 x 2*nboxes] allocate space (for centers of each box (x and y of each center))
// int* ntbox						out		number of occupied boxes -> defined in lenchk
void lenchk(int nside, double* xtarg, double* ytarg, int ntarg, int nterms,
            int nmax, int** iloc, int** ibox, int** itofst, int** itradr,
            double***** locexp, double** center, int* ntbox) {
    
	int i, j, k, nboxes, ixh, iyh, iadr;
	double h;
    
	nboxes = nside * nside; //total number of boxes
	*ntbox = 0;
    // allocate the arrays
	*iloc = (int*) malloc(nboxes * sizeof(int));
	*itofst = (int*) malloc((nboxes + 1) * sizeof(int));
	*itradr = (int*) malloc(ntarg * sizeof(int));
	*ibox = (int*) malloc(nmax * sizeof(int));
    
    // allocating the locexp array AFTER calculating ntbox!
    
    // centers
	*center = (double*) malloc(2 * nboxes * sizeof(double));
    
	h = 1.0 / nside;
	for (i = 0; i < nboxes; i++)
		(*iloc)[i] = 0;
    
    // for each target point...
	for (j = 0; j < ntarg; j++) {
		// determine which box this point will go into.
		ixh = (int) floor(xtarg[j] / h);
		iyh = (int) floor(ytarg[j] / h);
		if (ixh > nside - 1)
			ixh = nside - 1;
		if (ixh < 0)
			ixh = 0;
		if (iyh > nside - 1)
			iyh = nside - 1;
		if (iyh < 0)
			iyh = 0;
		// iadr is the index of the box to which this point is assigned.
		iadr = iyh * nside + ixh;  //box indices go from left to right (e.g carré séparé en quatre boxes, premières ligne a box 0 et 1 et deuxième ligne a box 2 et 3)
		if ((*iloc)[iadr] == 0) {
			(*iloc)[iadr] = *ntbox;  //on assigne a chaque box une valeure (+- aléatoire) unique (jsp si c'est utile)
			// number of target boxes?
			(*ntbox) = (*ntbox) + 1;
		}
	}
    
    // allocating the locexp array
	*locexp = (double****) malloc(*ntbox * sizeof(double***));
	for (i = 0; i < *ntbox; i++) {
		(*locexp)[i] = (double***) malloc((nterms + 1) * sizeof(double**));
		for (j = 0; j < (nterms + 1); j++) {
			(*locexp)[i][j] = (double**) malloc((nterms + 1) * sizeof(double*));
			for (k = 0; k < (nterms + 1); k++)
				(*locexp)[i][j][k] = (double*) malloc(
                                                      (nterms + 1) * sizeof(double));
		}
	}
	return;
}

// Scales all coordinates into a cube of [0,1] and returns the scale (i.e. biggest distance on x or y axis between 2 points)
// double* sx		in		source x coordinate
// double* sy		in		source y coordinate
// int NX			in		number of sources
// double* tx		in		target x coordinate
// double* ty		in		target y coordinate
// int NY			in		number of targets
double scale_to_cube(double* sx, double* sy, int NX, double* tx, double* ty,
                     int NY) {
	int i;
	double maxx = -INF; // biggest x coordinate of both sources and targets
	double minx = INF;  // smallest x coordinate of both sources and targets
	double maxy = -INF; // etc.
	double miny = INF;
	double dx, dy; // sizes of x,y,z
	double scale;      // max size of dx dy dz
	for (i = 0; i < NX; i++) {
		maxx = dmax(maxx, sx[i]);
		minx = dmin(minx, sx[i]);
		maxy = dmax(maxy, sy[i]);
		miny = dmin(miny, sy[i]);
	}
	for (i = 0; i < NY; i++) {
		maxx = dmax(maxx, tx[i]);
		minx = dmin(minx, tx[i]);
		maxy = dmax(maxy, ty[i]);
		miny = dmin(miny, ty[i]);
	}
	dx = maxx - minx; // max distance between points in x coordinate
	dy = maxy - miny; // max distance between points in y coordinate
	scale = dmax(dx, dy);
    
	// normalize the coordinate of the points to be in [0,1] box
	for (i = 0; i < NX; i++) {
		sx[i] = (sx[i] - minx) / scale;
		sy[i] = (sy[i] - miny) / scale;
	}
	for (i = 0; i < NY; i++) {
		tx[i] = (tx[i] - minx) / scale;
		ty[i] = (ty[i] - miny) / scale;
	}
	return scale;
}

// Preprocesses input variables and calls gausst(l.32)
// double *X			in		sources written in a 1D array
// double *Y			in		targets written in a 1D array
// double *w			in		weights = a given column (=dimension) of X
// double sigma			in		sigma = 1  ->bandwith used to compute box size , in multiscale != 0
// int NX				in		number of X
// int NY				in		number of Y
// int D				in		dimension => max 2
// double* bests		out		output
// int p				in		number of terms in the expansion
// int kdis				in      number of boxes to truncate, default set to 4 -> look paper on code p.970 == K (fixed parameter)
// int nfmax			in      max number of source points in each box for expansion, default set to 5 -> look paper on code p.970 == M0 (fixed parameter)
// int nlmax			in		max number of ?target? points in each box for expansion, default set to 5 -> look paper on code p.970 == M0 (fixed parameter)
// double r				in      default ret to 1/2 (?how to divide a box in next leaf?)
// int* typeCount		out		number of iterations for exact, local, far and far/far expansion

void gaussth(const double *X, const double *Y, double *w, double sigma, int NX, int NY,
             int D, double* bests, int p, int kdis, int nfmax, int nlmax, double r,
             int* typeCount) {
	double* sx;
	double* sy;
	double* tx;
	double* ty;
	double h;
	double scale;
	int i;
    
	if (D > 2) {
		printf("fgt doesn't work for D > 2\n");
		return;
	}
	h = sigma; 		// sigma
	sx = (double*) malloc(NX * sizeof(double));
	sy = (double*) malloc(NX * sizeof(double));
	tx = (double*) malloc(NY * sizeof(double));
	ty = (double*) malloc(NY * sizeof(double));
	for (i = 0; i < NX; i++) { //we separate x and y coordinates of sources
		sx[i] = X[i * D];
		if (D >= 2) { //thus if D == 2 as further we stop if D>2
			sy[i] = X[i * D + 1];
		} else {
			sy[i] = 0;
		}
	}
	for (i = 0; i < NY; i++) { //we separate x and y coordinates of targets
		tx[i] = Y[i * D]; 
		if (D >= 2) { //thus if D == 2 as further we stop if D>2
			ty[i] = Y[i * D + 1];
		} else {
			ty[i] = 0;
		}
	}

	// printf("X0_bis2 = %f\n",sx[0]);
    // printf("Y0_bis2 = %f\n",sy[0]);
	// printf("X1_bis2 = %f\n",sx[1]);
    // printf("Y1_bis2 = %f\n",sy[1]);
	scale = scale_to_cube(sx, sy, NX, tx, ty, NY); // scale is biggest distance between the points along x,y or z
	gausst(sx, sy, w, NX, tx, ty, bests, NY, h * h / (scale * scale), p, kdis,
           nfmax, nlmax, r, typeCount);
	free(sx);
	free(sy);
	free(tx);
	free(ty);
}

double dmax(double a, double b) {
	return (a > b) ? a : b;
}

double dmin(double a, double b) {
	return (a < b) ? a : b;
}
