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

#ifndef GAUSS_H_
#define GAUSS_H_

#define INF 1e100
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>

void gausst(double* xat, double* yat, double* str, int natoms, double* xtarg,
		double* ytarg, double* pot, int ntarg, double delta, int nterms,
		int kdis, int nfmax, int nlmax, double r, int* typeCount);

void gdir(double xp, double yp, double* xat, double* yat, int natoms,
		double* str, double* dpot, double delta);

void gamval(double*** ffexp, int nterms, double* cent, double xpoint,
		double ypoint, double* pot, double delta);

void gamkxp(double* cent, double* xatoms, double* yatoms, double* str,
		int ninbox, double*** ffexp, int nterms, double delta);

void galval(double*** local, int nterms, double* cent, double xpoint,
		double ypoint, double* pot, double delta);

void galkxp(double* cent, double xp, double yp, double charge, double*** b,
		int nterms, double delta);

void gaeval(double* xtarg, double* ytarg, double* pot, int ntarg,
		double**** locexp, int nterms, int* ibox, double* center, double delta,
		int* iloc, int nlmax, int nside, int* itofst, int* itradr);

void gafexp(double* xat, double* yat, double* str, int natoms, double* xtarg,
		double* ytarg, int ntarg, double delta, int nside, int nterms,
		int* iloc, int ntbox, int nfmax, int nlmax, double* pot, int kdis,
		int* ibox, int* itradr, int* itofst, double* center, double**** locexp,
		int* typeCount, double errConst, double errConst2);

void addexp(double***b, double***a, int nterms);

void assign(int nside, double* xat, double* yat, double* str, double* xatoms,
		double* yatoms, double* charg2, int* ioffst, double* xtarg,
		double* ytarg, int ntarg, int* ibox, int natoms, double* center,
		int* itradr, int* itofst);

void freestuff(int* iloc, int* ibox, int* itofst, int* itradr,
		double**** locexp, double* center, int nterms, int ntbox);

void gshift(double*** ffexp, double* cent1, double* cent2, double*** local,
		int nterms, double*** temp1, double*** temp2, double delta);

void mknbor(int ibox, int* nbors, int* nnbors, int nside, double kdis);

void lenchk(int nside, double* xtarg, double* ytarg, int ntarg, int nterms,
		int nmax, int** iloc, int** ibox, int** itofst, int** itradr,
		double***** locexp, double** center, int* ntbox);

void gaussth(double *X, double *Y, double *w, double sigma, int NX, int NY,
		int D, double* bests, int p, int kdis, int nfmax, int nlmax, double r,
		int* typeCount);

double computeError(double r, int p);

double computeErrorFar(double r, int p);

double dmax(double a, double b);

double dmin(double a, double b);

#endif /*GAUSS_H_*/
