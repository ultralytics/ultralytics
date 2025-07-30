/* statlib.c -- Statistical functions for testing the randomness of
   number sequences. */

/*
Copyright 1999, 2000 Free Software Foundation, Inc.

This file is part of the GNU MP Library test suite.

The GNU MP Library test suite is free software; you can redistribute it
and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 3 of the License,
or (at your option) any later version.

The GNU MP Library test suite is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
the GNU MP Library test suite.  If not, see https://www.gnu.org/licenses/.  */

/* The theories for these functions are taken from D. Knuth's "The Art
of Computer Programming: Volume 2, Seminumerical Algorithms", Third
Edition, Addison Wesley, 1998. */

/* Implementation notes.

The Kolmogorov-Smirnov test.

Eq. (13) in Knuth, p. 50, says that if X1, X2, ..., Xn are independent
observations arranged into ascending order

	Kp = sqr(n) * max(j/n - F(Xj))		for all 1<=j<=n
	Km = sqr(n) * max(F(Xj) - (j-1)/n))	for all 1<=j<=n

where F(x) = Pr(X <= x) = probability that (X <= x), which for a
uniformly distributed random real number between zero and one is
exactly the number itself (x).


The answer to exercise 23 gives the following implementation, which
doesn't need the observations to be sorted in ascending order:

for (k = 0; k < m; k++)
	a[k] = 1.0
	b[k] = 0.0
	c[k] = 0

for (each observation Xj)
	Y = F(Xj)
	k = floor (m * Y)
	a[k] = min (a[k], Y)
	b[k] = max (b[k], Y)
	c[k] += 1

	j = 0
	rp = rm = 0
	for (k = 0; k < m; k++)
		if (c[k] > 0)
			rm = max (rm, a[k] - j/n)
			j += c[k]
			rp = max (rp, j/n - b[k])

Kp = sqr (n) * rp
Km = sqr (n) * rm

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "gmp.h"
#include "gmpstat.h"

/* ks (Kp, Km, X, P, n) -- Perform a Kolmogorov-Smirnov test on the N
   real numbers between zero and one in vector X.  P is the
   distribution function, called for each entry in X, which should
   calculate the probability of X being greater than or equal to any
   number in the sequence.  (For a uniformly distributed sequence of
   real numbers between zero and one, this is simply equal to X.)  The
   result is put in Kp and Km.  */

void
ks (mpf_t Kp,
    mpf_t Km,
    mpf_t X[],
    void (P) (mpf_t, mpf_t),
    unsigned long int n)
{
  mpf_t Kt;			/* temp */
  mpf_t f_x;
  mpf_t f_j;			/* j */
  mpf_t f_jnq;			/* j/n or (j-1)/n */
  unsigned long int j;

  /* Sort the vector in ascending order. */
  qsort (X, n, sizeof (__mpf_struct), mpf_cmp);

  /* K-S test. */
  /*	Kp = sqr(n) * max(j/n - F(Xj))		for all 1<=j<=n
	Km = sqr(n) * max(F(Xj) - (j-1)/n))	for all 1<=j<=n
  */

  mpf_init (Kt); mpf_init (f_x); mpf_init (f_j); mpf_init (f_jnq);
  mpf_set_ui (Kp, 0);  mpf_set_ui (Km, 0);
  for (j = 1; j <= n; j++)
    {
      P (f_x, X[j-1]);
      mpf_set_ui (f_j, j);

      mpf_div_ui (f_jnq, f_j, n);
      mpf_sub (Kt, f_jnq, f_x);
      if (mpf_cmp (Kt, Kp) > 0)
	mpf_set (Kp, Kt);
      if (g_debug > DEBUG_2)
	{
	  printf ("j=%lu ", j);
	  printf ("P()="); mpf_out_str (stdout, 10, 2, f_x); printf ("\t");

	  printf ("jnq="); mpf_out_str (stdout, 10, 2, f_jnq); printf (" ");
	  printf ("diff="); mpf_out_str (stdout, 10, 2, Kt); printf (" ");
	  printf ("Kp="); mpf_out_str (stdout, 10, 2, Kp); printf ("\t");
	}
      mpf_sub_ui (f_j, f_j, 1);
      mpf_div_ui (f_jnq, f_j, n);
      mpf_sub (Kt, f_x, f_jnq);
      if (mpf_cmp (Kt, Km) > 0)
	mpf_set (Km, Kt);

      if (g_debug > DEBUG_2)
	{
	  printf ("jnq="); mpf_out_str (stdout, 10, 2, f_jnq); printf (" ");
	  printf ("diff="); mpf_out_str (stdout, 10, 2, Kt); printf (" ");
	  printf ("Km="); mpf_out_str (stdout, 10, 2, Km); printf (" ");
	  printf ("\n");
	}
    }
  mpf_sqrt_ui (Kt, n);
  mpf_mul (Kp, Kp, Kt);
  mpf_mul (Km, Km, Kt);

  mpf_clear (Kt); mpf_clear (f_x); mpf_clear (f_j); mpf_clear (f_jnq);
}

/* ks_table(val, n) -- calculate probability for Kp/Km less than or
   equal to VAL with N observations.  See [Knuth section 3.3.1] */

void
ks_table (mpf_t p, mpf_t val, const unsigned int n)
{
  /* We use Eq. (27), Knuth p.58, skipping O(1/n) for simplicity.
     This shortcut will result in too high probabilities, especially
     when n is small.

     Pr(Kp(n) <= s) = 1 - pow(e, -2*s^2) * (1 - 2/3*s/sqrt(n) + O(1/n)) */

  /* We have 's' in variable VAL and store the result in P. */

  mpf_t t1, t2;

  mpf_init (t1); mpf_init (t2);

  /* t1 = 1 - 2/3 * s/sqrt(n) */
  mpf_sqrt_ui (t1, n);
  mpf_div (t1, val, t1);
  mpf_mul_ui (t1, t1, 2);
  mpf_div_ui (t1, t1, 3);
  mpf_ui_sub (t1, 1, t1);

  /* t2 = pow(e, -2*s^2) */
#ifndef OLDGMP
  mpf_pow_ui (t2, val, 2);	/* t2 = s^2 */
  mpf_set_d (t2, exp (-(2.0 * mpf_get_d (t2))));
#else
  /* hmmm, gmp doesn't have pow() for floats.  use doubles. */
  mpf_set_d (t2, pow (M_E, -(2 * pow (mpf_get_d (val), 2))));
#endif

  /* p = 1 - t1 * t2 */
  mpf_mul (t1, t1, t2);
  mpf_ui_sub (p, 1, t1);

  mpf_clear (t1); mpf_clear (t2);
}

static double x2_table_X[][7] = {
  { -2.33, -1.64, -.674, 0.0, 0.674, 1.64, 2.33 }, /* x */
  { 5.4289, 2.6896, .454276, 0.0, .454276, 2.6896, 5.4289} /* x^2 */
};

#define _2D3 ((double) .6666666666)

/* x2_table (t, v, n) -- return chi-square table row for V in T[]. */
void
x2_table (double t[],
	  unsigned int v)
{
  int f;


  /* FIXME: Do a table lookup for v <= 30 since the following formula
     [Knuth, vol 2, 3.3.1] is only good for v > 30. */

  /* value = v + sqrt(2*v) * X[p] + (2/3) * X[p]^2 - 2/3 + O(1/sqrt(t) */
  /* NOTE: The O() term is ignored for simplicity. */

  for (f = 0; f < 7; f++)
      t[f] =
	v +
	sqrt (2 * v) * x2_table_X[0][f] +
	_2D3 * x2_table_X[1][f] - _2D3;
}


/* P(p, x) -- Distribution function.  Calculate the probability of X
being greater than or equal to any number in the sequence.  For a
random real number between zero and one given by a uniformly
distributed random number generator, this is simply equal to X. */

static void
P (mpf_t p, mpf_t x)
{
  mpf_set (p, x);
}

/* mpf_freqt() -- Frequency test using KS on N real numbers between zero
   and one.  See [Knuth vol 2, p.61]. */
void
mpf_freqt (mpf_t Kp,
	   mpf_t Km,
	   mpf_t X[],
	   const unsigned long int n)
{
  ks (Kp, Km, X, P, n);
}


/* The Chi-square test.  Eq. (8) in Knuth vol. 2 says that if Y[]
   holds the observations and p[] is the probability for.. (to be
   continued!)

   V = 1/n * sum((s=1 to k) Y[s]^2 / p[s]) - n */

void
x2 (mpf_t V,			/* result */
    unsigned long int X[],	/* data */
    unsigned int k,		/* #of categories */
    void (P) (mpf_t, unsigned long int, void *), /* probability func */
    void *x,			/* extra user data passed to P() */
    unsigned long int n)	/* #of samples */
{
  unsigned int f;
  mpf_t f_t, f_t2;		/* temp floats */

  mpf_init (f_t); mpf_init (f_t2);


  mpf_set_ui (V, 0);
  for (f = 0; f < k; f++)
    {
      if (g_debug > DEBUG_2)
	fprintf (stderr, "%u: P()=", f);
      mpf_set_ui (f_t, X[f]);
      mpf_mul (f_t, f_t, f_t);	/* f_t = X[f]^2 */
      P (f_t2, f, x);		/* f_t2 = Pr(f) */
      if (g_debug > DEBUG_2)
	mpf_out_str (stderr, 10, 2, f_t2);
      mpf_div (f_t, f_t, f_t2);
      mpf_add (V, V, f_t);
      if (g_debug > DEBUG_2)
	{
	  fprintf (stderr, "\tV=");
	  mpf_out_str (stderr, 10, 2, V);
	  fprintf (stderr, "\t");
	}
    }
  if (g_debug > DEBUG_2)
    fprintf (stderr, "\n");
  mpf_div_ui (V, V, n);
  mpf_sub_ui (V, V, n);

  mpf_clear (f_t); mpf_clear (f_t2);
}

/* Pzf(p, s, x) -- Probability for category S in mpz_freqt().  It's
   1/d for all S.  X is a pointer to an unsigned int holding 'd'. */
static void
Pzf (mpf_t p, unsigned long int s, void *x)
{
  mpf_set_ui (p, 1);
  mpf_div_ui (p, p, *((unsigned int *) x));
}

/* mpz_freqt(V, X, imax, n) -- Frequency test on integers.  [Knuth,
   vol 2, 3.3.2].  Keep IMAX low on this one, since we loop from 0 to
   IMAX.  128 or 256 could be nice.

   X[] must not contain numbers outside the range 0 <= X <= IMAX.

   Return value is number of observations actually used, after
   discarding entries out of range.

   Since X[] contains integers between zero and IMAX, inclusive, we
   have IMAX+1 categories.

   Note that N should be at least 5*IMAX.  Result is put in V and can
   be compared to output from x2_table (v=IMAX). */

unsigned long int
mpz_freqt (mpf_t V,
	   mpz_t X[],
	   unsigned int imax,
	   const unsigned long int n)
{
  unsigned long int *v;		/* result */
  unsigned int f;
  unsigned int d;		/* number of categories = imax+1 */
  unsigned int uitemp;
  unsigned long int usedn;


  d = imax + 1;

  v = (unsigned long int *) calloc (imax + 1, sizeof (unsigned long int));
  if (NULL == v)
    {
      fprintf (stderr, "mpz_freqt(): out of memory\n");
      exit (1);
    }

  /* count */
  usedn = n;			/* actual number of observations */
  for (f = 0; f < n; f++)
    {
      uitemp = mpz_get_ui(X[f]);
      if (uitemp > imax)	/* sanity check */
	{
	  if (g_debug)
	    fprintf (stderr, "mpz_freqt(): warning: input insanity: %u, "\
		     "ignored.\n", uitemp);
	  usedn--;
	  continue;
	}
      v[uitemp]++;
    }

  if (g_debug > DEBUG_2)
    {
      fprintf (stderr, "counts:\n");
      for (f = 0; f <= imax; f++)
	fprintf (stderr, "%u:\t%lu\n", f, v[f]);
    }

  /* chi-square with k=imax+1 and P(x)=1/(imax+1) for all x.*/
  x2 (V, v, d, Pzf, (void *) &d, usedn);

  free (v);
  return (usedn);
}

/* debug dummy to drag in dump funcs */
void
foo_debug ()
{
  if (0)
    {
      mpf_dump (0);
#ifndef OLDGMP
      mpz_dump (0);
#endif
    }
}

/* merit (rop, t, v, m) -- calculate merit for spectral test result in
   dimension T, see Knuth p. 105.  BUGS: Only valid for 2 <= T <=
   6. */
void
merit (mpf_t rop, unsigned int t, mpf_t v, mpz_t m)
{
  int f;
  mpf_t f_m, f_const, f_pi;

  mpf_init (f_m);
  mpf_set_z (f_m, m);
  mpf_init_set_d (f_const, M_PI);
  mpf_init_set_d (f_pi, M_PI);

  switch (t)
    {
    case 2:			/* PI */
      break;
    case 3:			/* PI * 4/3 */
      mpf_mul_ui (f_const, f_const, 4);
      mpf_div_ui (f_const, f_const, 3);
      break;
    case 4:			/* PI^2 * 1/2 */
      mpf_mul (f_const, f_const, f_pi);
      mpf_div_ui (f_const, f_const, 2);
      break;
    case 5:			/* PI^2 * 8/15 */
      mpf_mul (f_const, f_const, f_pi);
      mpf_mul_ui (f_const, f_const, 8);
      mpf_div_ui (f_const, f_const, 15);
      break;
    case 6:			/* PI^3 * 1/6 */
      mpf_mul (f_const, f_const, f_pi);
      mpf_mul (f_const, f_const, f_pi);
      mpf_div_ui (f_const, f_const, 6);
      break;
    default:
      fprintf (stderr,
	       "spect (merit): can't calculate merit for dimensions > 6\n");
      mpf_set_ui (f_const, 0);
      break;
    }

  /* rop = v^t */
  mpf_set (rop, v);
  for (f = 1; f < t; f++)
    mpf_mul (rop, rop, v);
  mpf_mul (rop, rop, f_const);
  mpf_div (rop, rop, f_m);

  mpf_clear (f_m);
  mpf_clear (f_const);
  mpf_clear (f_pi);
}

double
merit_u (unsigned int t, mpf_t v, mpz_t m)
{
  mpf_t rop;
  double res;

  mpf_init (rop);
  merit (rop, t, v, m);
  res = mpf_get_d (rop);
  mpf_clear (rop);
  return res;
}

/* f_floor (rop, op) -- Set rop = floor (op). */
void
f_floor (mpf_t rop, mpf_t op)
{
  mpz_t z;

  mpz_init (z);

  /* No mpf_floor().  Convert to mpz and back. */
  mpz_set_f (z, op);
  mpf_set_z (rop, z);

  mpz_clear (z);
}


/* vz_dot (rop, v1, v2, nelem) -- compute dot product of z-vectors V1,
   V2.  N is number of elements in vectors V1 and V2. */

void
vz_dot (mpz_t rop, mpz_t V1[], mpz_t V2[], unsigned int n)
{
  mpz_t t;

  mpz_init (t);
  mpz_set_ui (rop, 0);
  while (n--)
    {
      mpz_mul (t, V1[n], V2[n]);
      mpz_add (rop, rop, t);
    }

  mpz_clear (t);
}

void
spectral_test (mpf_t rop[], unsigned int T, mpz_t a, mpz_t m)
{
  /* Knuth "Seminumerical Algorithms, Third Edition", section 3.3.4
     (pp. 101-103). */

  /* v[t] = min { sqrt (x[1]^2 + ... + x[t]^2) |
     x[1] + a*x[2] + ... + pow (a, t-1) * x[t] is congruent to 0 (mod m) } */


  /* Variables. */
  unsigned int ui_t;
  unsigned int ui_i, ui_j, ui_k, ui_l;
  mpf_t f_tmp1, f_tmp2;
  mpz_t tmp1, tmp2, tmp3;
  mpz_t U[GMP_SPECT_MAXT][GMP_SPECT_MAXT],
    V[GMP_SPECT_MAXT][GMP_SPECT_MAXT],
    X[GMP_SPECT_MAXT],
    Y[GMP_SPECT_MAXT],
    Z[GMP_SPECT_MAXT];
  mpz_t h, hp, r, s, p, pp, q, u, v;

  /* GMP inits. */
  mpf_init (f_tmp1);
  mpf_init (f_tmp2);
  for (ui_i = 0; ui_i < GMP_SPECT_MAXT; ui_i++)
    {
      for (ui_j = 0; ui_j < GMP_SPECT_MAXT; ui_j++)
	{
	  mpz_init_set_ui (U[ui_i][ui_j], 0);
	  mpz_init_set_ui (V[ui_i][ui_j], 0);
	}
      mpz_init_set_ui (X[ui_i], 0);
      mpz_init_set_ui (Y[ui_i], 0);
      mpz_init (Z[ui_i]);
    }
  mpz_init (tmp1);
  mpz_init (tmp2);
  mpz_init (tmp3);
  mpz_init (h);
  mpz_init (hp);
  mpz_init (r);
  mpz_init (s);
  mpz_init (p);
  mpz_init (pp);
  mpz_init (q);
  mpz_init (u);
  mpz_init (v);

  /* Implementation inits. */
  if (T > GMP_SPECT_MAXT)
    T = GMP_SPECT_MAXT;			/* FIXME: Lazy. */

  /* S1 [Initialize.] */
  ui_t = 2 - 1;			/* NOTE: `t' in description == ui_t + 1
				   for easy indexing */
  mpz_set (h, a);
  mpz_set (hp, m);
  mpz_set_ui (p, 1);
  mpz_set_ui (pp, 0);
  mpz_set (r, a);
  mpz_pow_ui (s, a, 2);
  mpz_add_ui (s, s, 1);		/* s = 1 + a^2 */

  /* S2 [Euclidean step.] */
  while (1)
    {
      if (g_debug > DEBUG_1)
	{
	  mpz_mul (tmp1, h, pp);
	  mpz_mul (tmp2, hp, p);
	  mpz_sub (tmp1, tmp1, tmp2);
	  if (mpz_cmpabs (m, tmp1))
	    {
	      printf ("***BUG***: h*pp - hp*p = ");
	      mpz_out_str (stdout, 10, tmp1);
	      printf ("\n");
	    }
	}
      if (g_debug > DEBUG_2)
	{
	  printf ("hp = ");
	  mpz_out_str (stdout, 10, hp);
	  printf ("\nh = ");
	  mpz_out_str (stdout, 10, h);
	  printf ("\n");
	  fflush (stdout);
	}

      if (mpz_sgn (h))
	mpz_tdiv_q (q, hp, h);	/* q = floor(hp/h) */
      else
	mpz_set_ui (q, 1);

      if (g_debug > DEBUG_2)
	{
	  printf ("q = ");
	  mpz_out_str (stdout, 10, q);
	  printf ("\n");
	  fflush (stdout);
	}

      mpz_mul (tmp1, q, h);
      mpz_sub (u, hp, tmp1);	/* u = hp - q*h */

      mpz_mul (tmp1, q, p);
      mpz_sub (v, pp, tmp1);	/* v = pp - q*p */

      mpz_pow_ui (tmp1, u, 2);
      mpz_pow_ui (tmp2, v, 2);
      mpz_add (tmp1, tmp1, tmp2);
      if (mpz_cmp (tmp1, s) < 0)
	{
	  mpz_set (s, tmp1);	/* s = u^2 + v^2 */
	  mpz_set (hp, h);	/* hp = h */
	  mpz_set (h, u);	/* h = u */
	  mpz_set (pp, p);	/* pp = p */
	  mpz_set (p, v);	/* p = v */
	}
      else
	break;
    }

  /* S3 [Compute v2.] */
  mpz_sub (u, u, h);
  mpz_sub (v, v, p);

  mpz_pow_ui (tmp1, u, 2);
  mpz_pow_ui (tmp2, v, 2);
  mpz_add (tmp1, tmp1, tmp2);
  if (mpz_cmp (tmp1, s) < 0)
    {
      mpz_set (s, tmp1);	/* s = u^2 + v^2 */
      mpz_set (hp, u);
      mpz_set (pp, v);
    }
  mpf_set_z (f_tmp1, s);
  mpf_sqrt (rop[ui_t - 1], f_tmp1);

  /* S4 [Advance t.] */
  mpz_neg (U[0][0], h);
  mpz_set (U[0][1], p);
  mpz_neg (U[1][0], hp);
  mpz_set (U[1][1], pp);

  mpz_set (V[0][0], pp);
  mpz_set (V[0][1], hp);
  mpz_neg (V[1][0], p);
  mpz_neg (V[1][1], h);
  if (mpz_cmp_ui (pp, 0) > 0)
    {
      mpz_neg (V[0][0], V[0][0]);
      mpz_neg (V[0][1], V[0][1]);
      mpz_neg (V[1][0], V[1][0]);
      mpz_neg (V[1][1], V[1][1]);
    }

  while (ui_t + 1 != T)		/* S4 loop */
    {
      ui_t++;
      mpz_mul (r, a, r);
      mpz_mod (r, r, m);

      /* Add new row and column to U and V.  They are initialized with
	 all elements set to zero, so clearing is not necessary. */

      mpz_neg (U[ui_t][0], r); /* U: First col in new row. */
      mpz_set_ui (U[ui_t][ui_t], 1); /* U: Last col in new row. */

      mpz_set (V[ui_t][ui_t], m); /* V: Last col in new row. */

      /* "Finally, for 1 <= i < t,
	   set q = round (vi1 * r / m),
	   vit = vi1*r - q*m,
	   and Ut=Ut+q*Ui */

      for (ui_i = 0; ui_i < ui_t; ui_i++)
	{
	  mpz_mul (tmp1, V[ui_i][0], r); /* tmp1=vi1*r */
	  zdiv_round (q, tmp1, m); /* q=round(vi1*r/m) */
	  mpz_mul (tmp2, q, m);	/* tmp2=q*m */
	  mpz_sub (V[ui_i][ui_t], tmp1, tmp2);

	  for (ui_j = 0; ui_j <= ui_t; ui_j++) /* U[t] = U[t] + q*U[i] */
	    {
	      mpz_mul (tmp1, q, U[ui_i][ui_j]);	/* tmp=q*uij */
	      mpz_add (U[ui_t][ui_j], U[ui_t][ui_j], tmp1); /* utj = utj + q*uij */
	    }
	}

      /* s = min (s, zdot (U[t], U[t]) */
      vz_dot (tmp1, U[ui_t], U[ui_t], ui_t + 1);
      if (mpz_cmp (tmp1, s) < 0)
	mpz_set (s, tmp1);

      ui_k = ui_t;
      ui_j = 0;			/* WARNING: ui_j no longer a temp. */

      /* S5 [Transform.] */
      if (g_debug > DEBUG_2)
	printf ("(t, k, j, q1, q2, ...)\n");
      do
	{
	  if (g_debug > DEBUG_2)
	    printf ("(%u, %u, %u", ui_t + 1, ui_k + 1, ui_j + 1);

	  for (ui_i = 0; ui_i <= ui_t; ui_i++)
	    {
	      if (ui_i != ui_j)
		{
		  vz_dot (tmp1, V[ui_i], V[ui_j], ui_t + 1); /* tmp1=dot(Vi,Vj). */
		  mpz_abs (tmp2, tmp1);
		  mpz_mul_ui (tmp2, tmp2, 2); /* tmp2 = 2*abs(dot(Vi,Vj) */
		  vz_dot (tmp3, V[ui_j], V[ui_j], ui_t + 1); /* tmp3=dot(Vj,Vj). */

		  if (mpz_cmp (tmp2, tmp3) > 0)
		    {
		      zdiv_round (q, tmp1, tmp3); /* q=round(Vi.Vj/Vj.Vj) */
		      if (g_debug > DEBUG_2)
			{
			  printf (", ");
			  mpz_out_str (stdout, 10, q);
			}

		      for (ui_l = 0; ui_l <= ui_t; ui_l++)
			{
			  mpz_mul (tmp1, q, V[ui_j][ui_l]);
			  mpz_sub (V[ui_i][ui_l], V[ui_i][ui_l], tmp1); /* Vi=Vi-q*Vj */
			  mpz_mul (tmp1, q, U[ui_i][ui_l]);
			  mpz_add (U[ui_j][ui_l], U[ui_j][ui_l], tmp1); /* Uj=Uj+q*Ui */
			}

		      vz_dot (tmp1, U[ui_j], U[ui_j], ui_t + 1); /* tmp1=dot(Uj,Uj) */
		      if (mpz_cmp (tmp1, s) < 0) /* s = min(s,dot(Uj,Uj)) */
			mpz_set (s, tmp1);
		      ui_k = ui_j;
		    }
		  else if (g_debug > DEBUG_2)
		    printf (", #"); /* 2|Vi.Vj| <= Vj.Vj */
		}
	      else if (g_debug > DEBUG_2)
		printf (", *");	/* i == j */
	    }

	  if (g_debug > DEBUG_2)
	    printf (")\n");

	  /* S6 [Advance j.] */
	  if (ui_j == ui_t)
	    ui_j = 0;
	  else
	    ui_j++;
	}
      while (ui_j != ui_k);	/* S5 */

      /* From Knuth p. 104: "The exhaustive search in steps S8-S10
	 reduces the value of s only rarely." */
#ifdef DO_SEARCH
      /* S7 [Prepare for search.] */
      /* Find minimum in (x[1], ..., x[t]) satisfying condition
	 x[k]^2 <= f(y[1], ...,y[t]) * dot(V[k],V[k]) */

      ui_k = ui_t;
      if (g_debug > DEBUG_2)
	{
	  printf ("searching...");
	  /*for (f = 0; f < ui_t*/
	  fflush (stdout);
	}

      /* Z[i] = floor (sqrt (floor (dot(V[i],V[i]) * s / m^2))); */
      mpz_pow_ui (tmp1, m, 2);
      mpf_set_z (f_tmp1, tmp1);
      mpf_set_z (f_tmp2, s);
      mpf_div (f_tmp1, f_tmp2, f_tmp1);	/* f_tmp1 = s/m^2 */
      for (ui_i = 0; ui_i <= ui_t; ui_i++)
	{
	  vz_dot (tmp1, V[ui_i], V[ui_i], ui_t + 1);
	  mpf_set_z (f_tmp2, tmp1);
	  mpf_mul (f_tmp2, f_tmp2, f_tmp1);
	  f_floor (f_tmp2, f_tmp2);
	  mpf_sqrt (f_tmp2, f_tmp2);
	  mpz_set_f (Z[ui_i], f_tmp2);
	}

      /* S8 [Advance X[k].] */
      do
	{
	  if (g_debug > DEBUG_2)
	    {
	      printf ("X[%u] = ", ui_k);
	      mpz_out_str (stdout, 10, X[ui_k]);
	      printf ("\tZ[%u] = ", ui_k);
	      mpz_out_str (stdout, 10, Z[ui_k]);
	      printf ("\n");
	      fflush (stdout);
	    }

	  if (mpz_cmp (X[ui_k], Z[ui_k]))
	    {
	      mpz_add_ui (X[ui_k], X[ui_k], 1);
	      for (ui_i = 0; ui_i <= ui_t; ui_i++)
		mpz_add (Y[ui_i], Y[ui_i], U[ui_k][ui_i]);

	      /* S9 [Advance k.] */
	      while (++ui_k <= ui_t)
		{
		  mpz_neg (X[ui_k], Z[ui_k]);
		  mpz_mul_ui (tmp1, Z[ui_k], 2);
		  for (ui_i = 0; ui_i <= ui_t; ui_i++)
		    {
		      mpz_mul (tmp2, tmp1, U[ui_k][ui_i]);
		      mpz_sub (Y[ui_i], Y[ui_i], tmp2);
		    }
		}
	      vz_dot (tmp1, Y, Y, ui_t + 1);
	      if (mpz_cmp (tmp1, s) < 0)
		mpz_set (s, tmp1);
	    }
	}
      while (--ui_k);
#endif /* DO_SEARCH */
      mpf_set_z (f_tmp1, s);
      mpf_sqrt (rop[ui_t - 1], f_tmp1);
#ifdef DO_SEARCH
      if (g_debug > DEBUG_2)
	printf ("done.\n");
#endif /* DO_SEARCH */
    } /* S4 loop */

  /* Clear GMP variables. */

  mpf_clear (f_tmp1);
  mpf_clear (f_tmp2);
  for (ui_i = 0; ui_i < GMP_SPECT_MAXT; ui_i++)
    {
      for (ui_j = 0; ui_j < GMP_SPECT_MAXT; ui_j++)
	{
	  mpz_clear (U[ui_i][ui_j]);
	  mpz_clear (V[ui_i][ui_j]);
	}
      mpz_clear (X[ui_i]);
      mpz_clear (Y[ui_i]);
      mpz_clear (Z[ui_i]);
    }
  mpz_clear (tmp1);
  mpz_clear (tmp2);
  mpz_clear (tmp3);
  mpz_clear (h);
  mpz_clear (hp);
  mpz_clear (r);
  mpz_clear (s);
  mpz_clear (p);
  mpz_clear (pp);
  mpz_clear (q);
  mpz_clear (u);
  mpz_clear (v);

  return;
}
