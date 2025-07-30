/* Tune various threshold of MPFR

Copyright 2005-2017 Free Software Foundation, Inc.
Contributed by the AriC and Caramba projects, INRIA.

This file is part of the GNU MPFR Library.

The GNU MPFR Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The GNU MPFR Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the GNU MPFR Library; see the file COPYING.LESSER.  If not, see
http://www.gnu.org/licenses/ or write to the Free Software Foundation, Inc.,
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA. */

#include <stdlib.h>
#include <time.h>

#define MPFR_NEED_LONGLONG_H
#include "mpfr-impl.h"

#undef _PROTO
#define _PROTO __GMP_PROTO
#include "speed.h"

int verbose;

/* template for an unary function */
/* s->size: precision of both input and output
   s->xp  : Mantissa of first input
   s->yp  : mantissa of second input                    */
#define SPEED_MPFR_FUNC(mean_fun)                       \
  do                                                    \
    {                                                   \
      unsigned  i;                                      \
      mpfr_limb_ptr wp;                                 \
      double    t;                                      \
      mpfr_t    w, x;                                   \
      mp_size_t size;                                   \
      MPFR_TMP_DECL (marker);                           \
                                                        \
      SPEED_RESTRICT_COND (s->size >= MPFR_PREC_MIN);   \
      SPEED_RESTRICT_COND (s->size <= MPFR_PREC_MAX);   \
      MPFR_TMP_MARK (marker);                           \
                                                        \
      size = (s->size-1)/GMP_NUMB_BITS+1;               \
      s->xp[size-1] |= MPFR_LIMB_HIGHBIT;               \
      MPFR_TMP_INIT1 (s->xp, x, s->size);               \
      MPFR_SET_EXP (x, 0);                              \
                                                        \
      MPFR_TMP_INIT (wp, w, s->size, size);             \
                                                        \
      speed_operand_src (s, s->xp, size);               \
      speed_operand_dst (s, wp, size);                  \
      speed_cache_fill (s);                             \
                                                        \
      speed_starttime ();                               \
      i = s->reps;                                      \
      do                                                \
        mean_fun (w, x, MPFR_RNDN);                     \
      while (--i != 0);                                 \
      t = speed_endtime ();                             \
                                                        \
      MPFR_TMP_FREE (marker);                           \
      return t;                                         \
    }                                                   \
  while (0)

/* same as SPEED_MPFR_FUNC, but for say mpfr_sin_cos (y, z, x, r) */
#define SPEED_MPFR_FUNC2(mean_fun)                      \
  do                                                    \
    {                                                   \
      unsigned  i;                                      \
      mpfr_limb_ptr vp, wp;                             \
      double    t;                                      \
      mpfr_t    v, w, x;                                \
      mp_size_t size;                                   \
      MPFR_TMP_DECL (marker);                           \
                                                        \
      SPEED_RESTRICT_COND (s->size >= MPFR_PREC_MIN);   \
      SPEED_RESTRICT_COND (s->size <= MPFR_PREC_MAX);   \
      MPFR_TMP_MARK (marker);                           \
                                                        \
      size = (s->size-1)/GMP_NUMB_BITS+1;               \
      s->xp[size-1] |= MPFR_LIMB_HIGHBIT;               \
      MPFR_TMP_INIT1 (s->xp, x, s->size);               \
      MPFR_SET_EXP (x, 0);                              \
                                                        \
      MPFR_TMP_INIT (vp, v, s->size, size);             \
      MPFR_TMP_INIT (wp, w, s->size, size);             \
                                                        \
      speed_operand_src (s, s->xp, size);               \
      speed_operand_dst (s, vp, size);                  \
      speed_operand_dst (s, wp, size);                  \
      speed_cache_fill (s);                             \
                                                        \
      speed_starttime ();                               \
      i = s->reps;                                      \
      do                                                \
        mean_fun (v, w, x, MPFR_RNDN);                  \
      while (--i != 0);                                 \
      t = speed_endtime ();                             \
                                                        \
      MPFR_TMP_FREE (marker);                           \
      return t;                                         \
    }                                                   \
  while (0)

/* template for a function like mpfr_mul */
#define SPEED_MPFR_OP(mean_fun)                         \
  do                                                    \
    {                                                   \
      unsigned  i;                                      \
      mpfr_limb_ptr wp;                                 \
      double    t;                                      \
      mpfr_t    w, x, y;                                \
      mp_size_t size;                                   \
      MPFR_TMP_DECL (marker);                           \
                                                        \
      SPEED_RESTRICT_COND (s->size >= MPFR_PREC_MIN);   \
      SPEED_RESTRICT_COND (s->size <= MPFR_PREC_MAX);   \
      MPFR_TMP_MARK (marker);                           \
                                                        \
      size = (s->size-1)/GMP_NUMB_BITS+1;               \
      s->xp[size-1] |= MPFR_LIMB_HIGHBIT;               \
      MPFR_TMP_INIT1 (s->xp, x, s->size);               \
      MPFR_SET_EXP (x, 0);                              \
      s->yp[size-1] |= MPFR_LIMB_HIGHBIT;               \
      MPFR_TMP_INIT1 (s->yp, y, s->size);               \
      MPFR_SET_EXP (y, 0);                              \
                                                        \
      MPFR_TMP_INIT (wp, w, s->size, size);             \
                                                        \
      speed_operand_src (s, s->xp, size);               \
      speed_operand_src (s, s->yp, size);               \
      speed_operand_dst (s, wp, size);                  \
      speed_cache_fill (s);                             \
                                                        \
      speed_starttime ();                               \
      i = s->reps;                                      \
      do                                                \
        mean_fun (w, x, y, MPFR_RNDN);                  \
      while (--i != 0);                                 \
      t = speed_endtime ();                             \
                                                        \
      MPFR_TMP_FREE (marker);                           \
      return t;                                         \
    }                                                   \
  while (0)

/* special template for mpfr_mul(a,b,b) */
#define SPEED_MPFR_SQR(mean_fun)                        \
  do                                                    \
    {                                                   \
      unsigned  i;                                      \
      mpfr_limb_ptr wp;                                 \
      double    t;                                      \
      mpfr_t    w, x;                                   \
      mp_size_t size;                                   \
      MPFR_TMP_DECL (marker);                           \
                                                        \
      SPEED_RESTRICT_COND (s->size >= MPFR_PREC_MIN);   \
      SPEED_RESTRICT_COND (s->size <= MPFR_PREC_MAX);   \
      MPFR_TMP_MARK (marker);                           \
                                                        \
      size = (s->size-1)/GMP_NUMB_BITS+1;               \
      s->xp[size-1] |= MPFR_LIMB_HIGHBIT;               \
      MPFR_TMP_INIT1 (s->xp, x, s->size);               \
      MPFR_SET_EXP (x, 0);                              \
                                                        \
      MPFR_TMP_INIT (wp, w, s->size, size);             \
                                                        \
      speed_operand_src (s, s->xp, size);               \
      speed_operand_dst (s, wp, size);                  \
      speed_cache_fill (s);                             \
                                                        \
      speed_starttime ();                               \
      i = s->reps;                                      \
      do                                                \
        mean_fun (w, x, x, MPFR_RNDN);                  \
      while (--i != 0);                                 \
      t = speed_endtime ();                             \
                                                        \
      MPFR_TMP_FREE (marker);                           \
      return t;                                         \
    }                                                   \
  while (0)

/* s->size: precision of both input and output
   s->xp  : Mantissa of first input
   s->r   : exponent
   s->align_xp : sign (1 means positive, 2 means negative)
*/
#define SPEED_MPFR_FUNC_WITH_EXPONENT(mean_fun)         \
  do                                                    \
    {                                                   \
      unsigned  i;                                      \
      mpfr_limb_ptr wp;                                 \
      double    t;                                      \
      mpfr_t    w, x;                                   \
      mp_size_t size;                                   \
      MPFR_TMP_DECL (marker);                           \
                                                        \
      SPEED_RESTRICT_COND (s->size >= MPFR_PREC_MIN);   \
      SPEED_RESTRICT_COND (s->size <= MPFR_PREC_MAX);   \
      MPFR_TMP_MARK (marker);                           \
                                                        \
      size = (s->size-1)/GMP_NUMB_BITS+1;               \
      s->xp[size-1] |= MPFR_LIMB_HIGHBIT;               \
      MPFR_TMP_INIT1 (s->xp, x, s->size);               \
      MPFR_SET_EXP (x, s->r);                           \
      if (s->align_xp == 2) MPFR_SET_NEG (x);           \
                                                        \
      MPFR_TMP_INIT (wp, w, s->size, size);             \
                                                        \
      speed_operand_src (s, s->xp, size);               \
      speed_operand_dst (s, wp, size);                  \
      speed_cache_fill (s);                             \
                                                        \
      speed_starttime ();                               \
      i = s->reps;                                      \
      do                                                \
        mean_fun (w, x, MPFR_RNDN);                     \
      while (--i != 0);                                 \
      t = speed_endtime ();                             \
                                                        \
      MPFR_TMP_FREE (marker);                           \
      return t;                                         \
    }                                                   \
  while (0)

/* First we include all the functions we want to tune inside this program.
   We can't use the GNU MPFR library since the thresholds are fixed macros. */

/* Setup mpfr_exp_2 */
mpfr_prec_t mpfr_exp_2_threshold;
#undef  MPFR_EXP_2_THRESHOLD
#define MPFR_EXP_2_THRESHOLD mpfr_exp_2_threshold
#include "exp_2.c"
static double
speed_mpfr_exp_2 (struct speed_params *s)
{
  SPEED_MPFR_FUNC (mpfr_exp_2);
}

/* Setup mpfr_exp */
mpfr_prec_t mpfr_exp_threshold;
#undef  MPFR_EXP_THRESHOLD
#define MPFR_EXP_THRESHOLD mpfr_exp_threshold
#include "exp.c"
static double
speed_mpfr_exp (struct speed_params *s)
{
  SPEED_MPFR_FUNC (mpfr_exp);
}

/* Setup mpfr_sin_cos */
mpfr_prec_t mpfr_sincos_threshold;
#undef MPFR_SINCOS_THRESHOLD
#define MPFR_SINCOS_THRESHOLD mpfr_sincos_threshold
#include "sin_cos.c"
#include "cos.c"
static double
speed_mpfr_sincos (struct speed_params *s)
{
  SPEED_MPFR_FUNC2 (mpfr_sin_cos);
}

/* Setup mpfr_mul, mpfr_sqr and mpfr_div */
mpfr_prec_t mpfr_mul_threshold;
mpfr_prec_t mpfr_sqr_threshold;
mpfr_prec_t mpfr_div_threshold;
#undef  MPFR_MUL_THRESHOLD
#define MPFR_MUL_THRESHOLD mpfr_mul_threshold
#undef  MPFR_SQR_THRESHOLD
#define MPFR_SQR_THRESHOLD mpfr_sqr_threshold
#undef  MPFR_DIV_THRESHOLD
#define MPFR_DIV_THRESHOLD mpfr_div_threshold
#include "mul.c"
#include "div.c"
static double
speed_mpfr_mul (struct speed_params *s)
{
  SPEED_MPFR_OP (mpfr_mul);
}
static double
speed_mpfr_sqr (struct speed_params *s)
{
  SPEED_MPFR_SQR (mpfr_mul);
}
static double
speed_mpfr_div (struct speed_params *s)
{
  SPEED_MPFR_OP (mpfr_div);
}

/************************************************
 * Common functions (inspired by GMP function)  *
 ************************************************/
static int
analyze_data (double *dat, int ndat)
{
  double  x, min_x;
  int     j, min_j;

  x = 0.0;
  for (j = 0; j < ndat; j++)
    if (dat[j] > 0.0)
      x += dat[j];

  min_x = x;
  min_j = 0;

  for (j = 0; j < ndat; x -= dat[j], j++)
    {
      if (x < min_x)
        {
          min_x = x;
          min_j = j;
        }
    }
  return min_j;
}

static double
mpfr_speed_measure (speed_function_t fun, struct speed_params *s, char *m)
{
  double t = -1.0;
  int i;
  int number_of_iterations = 30;
  for (i = 1; i <= number_of_iterations && t == -1.0; i++)
    {
      t = speed_measure (fun, s);
      if ( (t == -1.0) && (i+1 <= number_of_iterations) )
        printf("speed_measure failed for size %lu. Trying again... (%d/%d)\n",
               s->size, i+1, number_of_iterations);
    }
  if (t == -1.0)
    {
      fprintf (stderr, "Failed to measure %s!\n", m);
      fprintf (stderr, "If CPU frequency scaling is enabled, please disable it:\n");
      fprintf (stderr, "   under Linux: cpufreq-selector -g performance\n");
      fprintf (stderr, "On a multi-core processor, you might also try to load all the cores\n");
      abort ();
    }
  return t;
}

#define THRESHOLD_WINDOW 16
#define THRESHOLD_FINAL_WINDOW 128
static double
domeasure (mpfr_prec_t *threshold,
           double (*func) (struct speed_params *),
           mpfr_prec_t p)
{
  struct speed_params s;
  mp_size_t size;
  double t1, t2, d;

  s.align_xp = s.align_yp = s.align_wp = 64;
  s.size = p;
  size = (p - 1)/GMP_NUMB_BITS+1;
  s.xp = malloc (2*size*sizeof (mp_limb_t));
  if (s.xp == NULL)
    {
      fprintf (stderr, "Can't allocate memory.\n");
      abort ();
    }
  mpn_random (s.xp, size);
  s.yp = s.xp + size;
  mpn_random (s.yp, size);
  *threshold = MPFR_PREC_MAX;
  t1 = mpfr_speed_measure (func, &s, "function 1");
  *threshold = 1;
  t2 = mpfr_speed_measure (func, &s, "function 2");
  free (s.xp);
  /* t1 is the time of the first algo (used for low prec) */
  if (t2 >= t1)
    d = (t2 - t1) / t2;
  else
    d = (t2 - t1) / t1;
  /* d > 0 if we have to use algo 1.
     d < 0 if we have to use algo 2 */
  return d;
}

/* Performs measures when both the precision and the point of evaluation
   shall vary. s.yp is ignored and not initialized.
   It assumes that func depends on three thresholds with a boundary of the
   form threshold1*x + threshold2*p = some scaling factor, if x<0,
   and  threshold3*x + threshold2*p = some scaling factor, if x>=0.
*/
static double
domeasure2 (long int *threshold1, long int *threshold2, long int *threshold3,
            double (*func) (struct speed_params *),
            mpfr_prec_t p,
            mpfr_t x)
{
  struct speed_params s;
  mp_size_t size;
  double t1, t2, d;
  mpfr_t xtmp;

  if (MPFR_IS_SINGULAR (x))
    {
      mpfr_fprintf (stderr, "x=%RNf is not a regular number.\n");
      abort ();
    }
  if (MPFR_IS_NEG (x))
    s.align_xp = 2;
  else
    s.align_xp = 1;

  s.align_yp = s.align_wp = 64;
  s.size = p;
  size = (p - 1)/GMP_NUMB_BITS+1;

  mpfr_init2 (xtmp, p);
  mpn_random (xtmp->_mpfr_d, size);
  xtmp->_mpfr_d[size-1] |= MPFR_LIMB_HIGHBIT;
  MPFR_SET_EXP (xtmp, -53);
  mpfr_add_ui (xtmp, xtmp, 1, MPFR_RNDN);
  mpfr_mul (xtmp, xtmp, x, MPFR_RNDN); /* xtmp = x*(1+perturb)       */
                                      /* where perturb ~ 2^(-53) is */
                                      /* randomly chosen.           */
  s.xp = xtmp->_mpfr_d;
  s.r = MPFR_GET_EXP (xtmp);

  *threshold1 = 0;
  *threshold2 = 0;
  *threshold3 = 0;
  t1 = mpfr_speed_measure (func, &s, "function 1");

  if (MPFR_IS_NEG (x))
    *threshold1 = INT_MIN;
  else
    *threshold3 = INT_MAX;
  *threshold2 = INT_MAX;
  t2 = mpfr_speed_measure (func, &s, "function 2");

  /* t1 is the time of the first algo (used for low prec) */
  if (t2 >= t1)
    d = (t2 - t1) / t2;
  else
    d = (t2 - t1) / t1;
  /* d > 0 if we have to use algo 1.
     d < 0 if we have to use algo 2 */
  mpfr_clear (xtmp);
  return d;
}

/* Tune a function with a simple THRESHOLD
   The function doesn't depend on another threshold.
   It assumes that it uses algo1 if p < THRESHOLD
   and algo2 otherwise.
   if algo2 is better for low prec, and algo1 better for high prec,
   the behaviour of this function is undefined. */
static void
tune_simple_func (mpfr_prec_t *threshold,
                  double (*func) (struct speed_params *),
                  mpfr_prec_t pstart)
{
  double measure[THRESHOLD_FINAL_WINDOW+1];
  double d;
  mpfr_prec_t pstep;
  int i, numpos, numneg, try;
  mpfr_prec_t pmin, pmax, p;

  /* first look for a lower bound within 10% */
  pmin = p = pstart;
  d = domeasure (threshold, func, pmin);
  if (d < 0.0)
    {
      if (verbose)
        printf ("Oops: even for %lu, algo 2 seems to be faster!\n",
                (unsigned long) pmin);
      *threshold = MPFR_PREC_MIN;
      return;
    }
  if (d >= 1.00)
    for (;;)
      {
        d = domeasure (threshold, func, pmin);
        if (d < 1.00)
          break;
        p = pmin;
        pmin += pmin/2;
      }
  pmin = p;
  for (;;)
    {
      d = domeasure (threshold, func, pmin);
      if (d < 0.10)
        break;
      pmin += GMP_NUMB_BITS;
    }

  /* then look for an upper bound within 20% */
  pmax = pmin * 2;
  for (;;)
    {
      d = domeasure (threshold, func, pmax);
      if (d < -0.20)
        break;
      pmax += pmin / 2; /* don't increase too rapidly */
    }

  /* The threshold is between pmin and pmax. Affine them */
  try = 0;
  while ((pmax-pmin) >= THRESHOLD_FINAL_WINDOW)
    {
      pstep = MAX(MIN(GMP_NUMB_BITS/2,(pmax-pmin)/(2*THRESHOLD_WINDOW)),1);
      if (verbose)
        printf ("Pmin = %8lu Pmax = %8lu Pstep=%lu\n", pmin, pmax, pstep);
      p = (pmin + pmax) / 2;
      for (i = numpos = numneg = 0 ; i < THRESHOLD_WINDOW + 1 ; i++)
        {
          measure[i] = domeasure (threshold, func,
                                  p+(i-THRESHOLD_WINDOW/2)*pstep);
          if (measure[i] > 0)
            numpos ++;
          else if (measure[i] < 0)
            numneg ++;
        }
      if (numpos > numneg)
        /* We use more often algo 1 than algo 2 */
        pmin = p - THRESHOLD_WINDOW/2*pstep;
      else if (numpos < numneg)
        pmax = p + THRESHOLD_WINDOW/2*pstep;
      else
        /* numpos == numneg ... */
        if (++ try > 2)
          {
            *threshold = p;
            if (verbose)
              printf ("Quick find: %lu\n", *threshold);
            return ;
          }
    }

  /* Final tune... */
  if (verbose)
    printf ("Finalizing in [%lu, %lu]... ", pmin, pmax);
  for (i = 0 ; i < THRESHOLD_FINAL_WINDOW+1 ; i++)
    measure[i] = domeasure (threshold, func, pmin+i);
  i = analyze_data (measure, THRESHOLD_FINAL_WINDOW+1);
  *threshold = pmin + i;
  if (verbose)
    printf ("%lu\n", *threshold);
  return;
}

/* Tune a function which behavior depends on both p and x,
   in a given direction.
   It assumes that for (x,p) close to zero, algo1 is used
   and algo2 is used when (x,p) is far from zero.
   If algo2 is better for low prec, and algo1 better for high prec,
   the behaviour of this function is undefined.
   This tuning function tries couples (x,p) of the form (ell*dirx, ell*dirp)
   until it finds a point on the boundary. It returns ell.
 */
static void
tune_simple_func_in_some_direction (long int *threshold1,
                                    long int *threshold2,
                                    long int *threshold3,
                                    double (*func) (struct speed_params *),
                                    mpfr_prec_t pstart,
                                    int dirx, int dirp,
                                    mpfr_t xres, mpfr_prec_t *pres)
{
  double measure[THRESHOLD_FINAL_WINDOW+1];
  double d;
  mpfr_prec_t pstep;
  int i, numpos, numneg, try;
  mpfr_prec_t pmin, pmax, p;
  mpfr_t xmin, xmax, x;
  mpfr_t ratio;

  mpfr_init2 (ratio, MPFR_SMALL_PRECISION);
  mpfr_set_si (ratio, dirx, MPFR_RNDN);
  mpfr_div_si (ratio, ratio, dirp, MPFR_RNDN);

  mpfr_init2 (xmin, MPFR_SMALL_PRECISION);
  mpfr_init2 (xmax, MPFR_SMALL_PRECISION);
  mpfr_init2 (x, MPFR_SMALL_PRECISION);

  /* first look for a lower bound within 10% */
  pmin = p = pstart;
  mpfr_mul_ui (xmin, ratio, (unsigned int)pmin, MPFR_RNDN);
  mpfr_set (x, xmin, MPFR_RNDN);

  d = domeasure2 (threshold1, threshold2, threshold3, func, pmin, xmin);
  if (d < 0.0)
    {
      if (verbose)
        printf ("Oops: even for %lu, algo 2 seems to be faster!\n",
                (unsigned long) pmin);
      *pres = MPFR_PREC_MIN;
      mpfr_mul_ui (xres, ratio, (unsigned int)*pres, MPFR_RNDN);
      mpfr_clear (ratio); mpfr_clear (x); mpfr_clear (xmin); mpfr_clear (xmax);
      return;
    }
  if (d >= 1.00)
    for (;;)
      {
        d = domeasure2 (threshold1, threshold2, threshold3, func, pmin, xmin);
        if (d < 1.00)
          break;
        p = pmin;
        mpfr_set (x, xmin, MPFR_RNDN);
        pmin += pmin/2;
        mpfr_mul_ui (xmin, ratio, (unsigned int)pmin, MPFR_RNDN);
      }
  pmin = p;
  mpfr_set (xmin, x, MPFR_RNDN);
  for (;;)
    {
      d = domeasure2 (threshold1, threshold2, threshold3, func, pmin, xmin);
      if (d < 0.10)
        break;
      pmin += GMP_NUMB_BITS;
      mpfr_mul_ui (xmin, ratio, (unsigned int)pmin, MPFR_RNDN);
    }

  /* then look for an upper bound within 20% */
  pmax = pmin * 2;
  mpfr_mul_ui (xmax, ratio, (unsigned int)pmax, MPFR_RNDN);
  for (;;)
    {
      d = domeasure2 (threshold1, threshold2, threshold3, func, pmax, xmax);
      if (d < -0.20)
        break;
      pmax += pmin / 2; /* don't increase too rapidly */
      mpfr_mul_ui (xmax, ratio, (unsigned int)pmax, MPFR_RNDN);
    }

  /* The threshold is between pmin and pmax. Affine them */
  try = 0;
  while ((pmax-pmin) >= THRESHOLD_FINAL_WINDOW)
    {
      pstep = MAX(MIN(GMP_NUMB_BITS/2,(pmax-pmin)/(2*THRESHOLD_WINDOW)),1);
      if (verbose)
        printf ("Pmin = %8lu Pmax = %8lu Pstep=%lu\n", pmin, pmax, pstep);
      p = (pmin + pmax) / 2;
      mpfr_mul_ui (x, ratio, (unsigned int)p, MPFR_RNDN);
      for (i = numpos = numneg = 0 ; i < THRESHOLD_WINDOW + 1 ; i++)
        {
          *pres = p+(i-THRESHOLD_WINDOW/2)*pstep;
          mpfr_mul_ui (xres, ratio, (unsigned int)*pres, MPFR_RNDN);
          measure[i] = domeasure2 (threshold1, threshold2, threshold3,
                                   func, *pres, xres);
          if (measure[i] > 0)
            numpos ++;
          else if (measure[i] < 0)
            numneg ++;
        }
      if (numpos > numneg)
        {
          /* We use more often algo 1 than algo 2 */
          pmin = p - THRESHOLD_WINDOW/2*pstep;
          mpfr_mul_ui (xmin, ratio, (unsigned int)pmin, MPFR_RNDN);
        }
      else if (numpos < numneg)
        {
          pmax = p + THRESHOLD_WINDOW/2*pstep;
          mpfr_mul_ui (xmax, ratio, (unsigned int)pmax, MPFR_RNDN);
        }
      else
        /* numpos == numneg ... */
        if (++ try > 2)
          {
            *pres = p;
            mpfr_mul_ui (xres, ratio, (unsigned int)*pres, MPFR_RNDN);
            if (verbose)
              printf ("Quick find: %lu\n", *pres);
            mpfr_clear (ratio);
            mpfr_clear (x); mpfr_clear (xmin); mpfr_clear (xmax);
            return ;
          }
    }

  /* Final tune... */
  if (verbose)
    printf ("Finalizing in [%lu, %lu]... ", pmin, pmax);
  for (i = 0 ; i < THRESHOLD_FINAL_WINDOW+1 ; i++)
    {
      *pres = pmin+i;
      mpfr_mul_ui (xres, ratio, (unsigned int)*pres, MPFR_RNDN);
      measure[i] = domeasure2 (threshold1, threshold2, threshold3,
                               func, *pres, xres);
    }
  i = analyze_data (measure, THRESHOLD_FINAL_WINDOW+1);
  *pres = pmin + i;
  mpfr_mul_ui (xres, ratio, (unsigned int)*pres, MPFR_RNDN);
  if (verbose)
    printf ("%lu\n", *pres);
  mpfr_clear (ratio); mpfr_clear (x); mpfr_clear (xmin); mpfr_clear (xmax);
  return;
}

/************************************
 * Tune Mulders' mulhigh function   *
 ************************************/
#define TOLERANCE 1.00
#define MULDERS_TABLE_SIZE 1024
#ifndef MPFR_MULHIGH_SIZE
# define MPFR_MULHIGH_SIZE MULDERS_TABLE_SIZE
#endif
#ifndef MPFR_SQRHIGH_SIZE
# define MPFR_SQRHIGH_SIZE MULDERS_TABLE_SIZE
#endif
#ifndef MPFR_DIVHIGH_SIZE
# define MPFR_DIVHIGH_SIZE MULDERS_TABLE_SIZE
#endif
#define MPFR_MULHIGH_TAB_SIZE MPFR_MULHIGH_SIZE
#define MPFR_SQRHIGH_TAB_SIZE MPFR_SQRHIGH_SIZE
#define MPFR_DIVHIGH_TAB_SIZE MPFR_DIVHIGH_SIZE
#include "mulders.c"

static double
speed_mpfr_mulhigh (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_MUL_N (mpfr_mulhigh_n);
}

static double
speed_mpfr_sqrhigh (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_SQR (mpfr_sqrhigh_n);
}

static double
speed_mpfr_divhigh (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_DC_DIVREM_CALL (mpfr_divhigh_n (q, a, d, s->size));
}

#define MAX_STEPS 513 /* maximum number of values of k tried for a given n */

/* Tune mpfr_mulhigh_n for size n */
static mp_size_t
tune_mul_mulders_upto (mp_size_t n)
{
  struct speed_params s;
  mp_size_t k, kbest, step;
  double t, tbest;
  MPFR_TMP_DECL (marker);

  if (n == 0)
    return -1;

  MPFR_TMP_MARK (marker);
  s.align_xp = s.align_yp = s.align_wp = 64;
  s.size = n;
  s.xp   = MPFR_TMP_ALLOC (n * sizeof (mp_limb_t));
  s.yp   = MPFR_TMP_ALLOC (n * sizeof (mp_limb_t));
  mpn_random (s.xp, n);
  mpn_random (s.yp, n);

  /* Check k == -1, mpn_mul_basecase */
  mulhigh_ktab[n] = -1;
  kbest = -1;
  tbest = mpfr_speed_measure (speed_mpfr_mulhigh, &s, "mpfr_mulhigh");

  /* Check k == 0, mpn_mulhigh_n_basecase */
  mulhigh_ktab[n] = 0;
  t = mpfr_speed_measure (speed_mpfr_mulhigh, &s, "mpfr_mulhigh");
  if (t * TOLERANCE < tbest)
    kbest = 0, tbest = t;

  /* Check Mulders with cutoff point k */
  step = 1 + n / (2 * MAX_STEPS);
  /* we need k >= (n+3)/2, which translates into k >= (n+4)/2 in C */
  for (k = (n + 4) / 2 ; k < n ; k += step)
    {
      mulhigh_ktab[n] = k;
      t = mpfr_speed_measure (speed_mpfr_mulhigh, &s, "mpfr_mulhigh");
      if (t * TOLERANCE < tbest)
        kbest = k, tbest = t;
    }

  mulhigh_ktab[n] = kbest;

  MPFR_TMP_FREE (marker);
  return kbest;
}

/* Tune mpfr_sqrhigh_n for size n */
static mp_size_t
tune_sqr_mulders_upto (mp_size_t n)
{
  struct speed_params s;
  mp_size_t k, kbest, step;
  double t, tbest;
  MPFR_TMP_DECL (marker);

  if (n == 0)
    return -1;

  MPFR_TMP_MARK (marker);
  s.align_xp = s.align_wp = 64;
  s.size = n;
  s.xp   = MPFR_TMP_ALLOC (n * sizeof (mp_limb_t));
  mpn_random (s.xp, n);

  /* Check k == -1, mpn_sqr_basecase */
  sqrhigh_ktab[n] = -1;
  kbest = -1;
  tbest = mpfr_speed_measure (speed_mpfr_sqrhigh, &s, "mpfr_sqrhigh");

  /* Check k == 0, mpfr_mulhigh_n_basecase */
  sqrhigh_ktab[n] = 0;
  t = mpfr_speed_measure (speed_mpfr_sqrhigh, &s, "mpfr_sqrhigh");
  if (t * TOLERANCE < tbest)
    kbest = 0, tbest = t;

  /* Check Mulders */
  step = 1 + n / (2 * MAX_STEPS);
  /* we need k >= (n+3)/2, which translates into k >= (n+4)/2 in C */
  for (k = (n + 4) / 2 ; k < n ; k += step)
    {
      sqrhigh_ktab[n] = k;
      t = mpfr_speed_measure (speed_mpfr_sqrhigh, &s, "mpfr_sqrhigh");
      if (t * TOLERANCE < tbest)
        kbest = k, tbest = t;
    }

  sqrhigh_ktab[n] = kbest;

  MPFR_TMP_FREE (marker);
  return kbest;
}

/* Tune mpfr_divhigh_n for size n */
static mp_size_t
tune_div_mulders_upto (mp_size_t n)
{
  struct speed_params s;
  mp_size_t k, kbest, step;
  double t, tbest;
  MPFR_TMP_DECL (marker);

  if (n == 0)
    return 0;

  MPFR_TMP_MARK (marker);
  s.align_xp = s.align_yp = s.align_wp = s.align_wp2 = 64;
  s.size = n;
  s.xp   = MPFR_TMP_ALLOC (n * sizeof (mp_limb_t));
  s.yp   = MPFR_TMP_ALLOC (n * sizeof (mp_limb_t));
  mpn_random (s.xp, n);
  mpn_random (s.yp, n);

  /* Check k == n, i.e., mpn_divrem */
  divhigh_ktab[n] = n;
  kbest = n;
  tbest = mpfr_speed_measure (speed_mpfr_divhigh, &s, "mpfr_divhigh");

  /* Check k == 0, i.e., mpfr_divhigh_n_basecase */
#if defined(WANT_GMP_INTERNALS) && defined(HAVE___GMPN_SBPI1_DIVAPPR_Q)
  if (n > 2) /* mpn_sbpi1_divappr_q requires dn > 2 */
#endif
    {
      divhigh_ktab[n] = 0;
      t = mpfr_speed_measure (speed_mpfr_divhigh, &s, "mpfr_divhigh");
      if (t * TOLERANCE < tbest)
        kbest = 0, tbest = t;
    }

  /* Check Mulders */
  step = 1 + n / (2 * MAX_STEPS);
  /* we should have (n+3)/2 <= k < n, which translates into
     (n+4)/2 <= k < n in C */
  for (k = (n + 4) / 2 ; k < n ; k += step)
    {
      divhigh_ktab[n] = k;
      t = mpfr_speed_measure (speed_mpfr_divhigh, &s, "mpfr_divhigh");
      if (t * TOLERANCE < tbest)
        kbest = k, tbest = t;
    }

  divhigh_ktab[n] = kbest;

  MPFR_TMP_FREE (marker);

  return kbest;
}

static void
tune_mul_mulders (FILE *f)
{
  mp_size_t k;

  if (verbose)
    printf ("Tuning mpfr_mulhigh_n[%d]", (int) MPFR_MULHIGH_TAB_SIZE);
  fprintf (f, "#define MPFR_MULHIGH_TAB  \\\n ");
  for (k = 0 ; k < MPFR_MULHIGH_TAB_SIZE ; k++)
    {
      fprintf (f, "%d", (int) tune_mul_mulders_upto (k));
      if (k != MPFR_MULHIGH_TAB_SIZE-1)
        fputc (',', f);
      if ((k+1) % 16 == 0)
        fprintf (f, " \\\n ");
      if (verbose)
        putchar ('.');
    }
  fprintf (f, " \n");
  if (verbose)
    putchar ('\n');
}

static void
tune_sqr_mulders (FILE *f)
{
  mp_size_t k;

  if (verbose)
    printf ("Tuning mpfr_sqrhigh_n[%d]", (int) MPFR_SQRHIGH_TAB_SIZE);
  fprintf (f, "#define MPFR_SQRHIGH_TAB  \\\n ");
  for (k = 0 ; k < MPFR_SQRHIGH_TAB_SIZE ; k++)
    {
      fprintf (f, "%d", (int) tune_sqr_mulders_upto (k));
      if (k != MPFR_SQRHIGH_TAB_SIZE-1)
        fputc (',', f);
      if ((k+1) % 16 == 0)
        fprintf (f, " \\\n ");
      if (verbose)
        putchar ('.');
    }
  fprintf (f, " \n");
  if (verbose)
    putchar ('\n');
}

static void
tune_div_mulders (FILE *f)
{
  mp_size_t k;

  if (verbose)
    printf ("Tuning mpfr_divhigh_n[%d]", (int) MPFR_DIVHIGH_TAB_SIZE);
  fprintf (f, "#define MPFR_DIVHIGH_TAB  \\\n ");
  for (k = 0 ; k < MPFR_DIVHIGH_TAB_SIZE ; k++)
    {
      fprintf (f, "%d", (int) tune_div_mulders_upto (k));
      if (k != MPFR_DIVHIGH_TAB_SIZE - 1)
        fputc (',', f);
      if ((k+1) % 16 == 0)
        fprintf (f, " /*%zu-%zu*/ \\\n ", k - 15, k);
      if (verbose)
        putchar ('.');
    }
  fprintf (f, " \n");
  if (verbose)
    putchar ('\n');
}

/*******************************************************
 *            Tuning functions for mpfr_ai             *
 *******************************************************/

long int mpfr_ai_threshold1;
long int mpfr_ai_threshold2;
long int mpfr_ai_threshold3;
#undef  MPFR_AI_THRESHOLD1
#define MPFR_AI_THRESHOLD1 mpfr_ai_threshold1
#undef  MPFR_AI_THRESHOLD2
#define MPFR_AI_THRESHOLD2 mpfr_ai_threshold2
#undef  MPFR_AI_THRESHOLD3
#define MPFR_AI_THRESHOLD3 mpfr_ai_threshold3

#include "ai.c"

static double
speed_mpfr_ai (struct speed_params *s)
{
  SPEED_MPFR_FUNC_WITH_EXPONENT (mpfr_ai);
}


/*******************************************************
 *            Tune all the threshold of MPFR           *
 * Warning: tune the function in their dependent order!*
 *******************************************************/
static void
all (const char *filename)
{
  FILE *f;
  time_t  start_time, end_time;
  struct tm  *tp;
  mpfr_t x1, x2, x3, tmp1, tmp2;
  mpfr_prec_t p1, p2, p3;

  f = fopen (filename, "w");
  if (f == NULL)
    {
      fprintf (stderr, "Can't open file '%s' for writing.\n", filename);
      abort ();
    }

  speed_time_init ();
  if (verbose)
    {
      printf ("Using: %s\n", speed_time_string);
      printf ("speed_precision %d", speed_precision);
      if (speed_unittime == 1.0)
        printf (", speed_unittime 1 cycle");
      else
        printf (", speed_unittime %.2e secs", speed_unittime);
      if (speed_cycletime == 1.0 || speed_cycletime == 0.0)
        printf (", CPU freq unknown\n");
      else
        printf (", CPU freq %.2f MHz\n\n", 1e-6/speed_cycletime);
    }

  time (&start_time);
  tp = localtime (&start_time);
  fprintf (f, "/* Generated by MPFR's tuneup.c, %d-%02d-%02d, ",
          tp->tm_year+1900, tp->tm_mon+1, tp->tm_mday);

#ifdef __ICC
  fprintf (f, "icc %d.%d.%d */\n", __ICC / 100, __ICC / 10 % 10, __ICC % 10);
#elif defined(__GNUC__)
#ifdef __GNUC_PATCHLEVEL__
  fprintf (f, "gcc %d.%d.%d */\n", __GNUC__, __GNUC_MINOR__,
           __GNUC_PATCHLEVEL__);
#else
  fprintf (f, "gcc %d.%d */\n", __GNUC__, __GNUC_MINOR__);
#endif
#elif defined (__SUNPRO_C)
  fprintf (f, "Sun C %d.%d */\n", __SUNPRO_C / 0x100, __SUNPRO_C % 0x100);
#elif defined (__sgi) && defined (_COMPILER_VERSION)
  fprintf (f, "MIPSpro C %d.%d.%d */\n",
           _COMPILER_VERSION / 100,
           _COMPILER_VERSION / 10 % 10,
           _COMPILER_VERSION % 10);
#elif defined (__DECC) && defined (__DECC_VER)
  fprintf (f, "DEC C %d */\n", __DECC_VER);
#else
  fprintf (f, "system compiler */\n");
#endif
  fprintf (f, "\n\n");
  fprintf (f, "#ifndef MPFR_TUNE_CASE\n");
  fprintf (f, "#define MPFR_TUNE_CASE \"src/mparam.h\"\n");
  fprintf (f, "#endif\n\n");

  /* Tune mulhigh */
  tune_mul_mulders (f);

  /* Tune sqrhigh */
  tune_sqr_mulders (f);

  /* Tune divhigh */
  tune_div_mulders (f);
  fflush (f);

  /* Tune mpfr_mul (threshold is in limbs, but it doesn't matter too much) */
  if (verbose)
    printf ("Tuning mpfr_mul...\n");
  tune_simple_func (&mpfr_mul_threshold, speed_mpfr_mul,
                    2*GMP_NUMB_BITS+1);
  fprintf (f, "#define MPFR_MUL_THRESHOLD %lu /* limbs */\n",
           (unsigned long) (mpfr_mul_threshold - 1) / GMP_NUMB_BITS + 1);

  /* Tune mpfr_sqr (threshold is in limbs, but it doesn't matter too much) */
  if (verbose)
    printf ("Tuning mpfr_sqr...\n");
  tune_simple_func (&mpfr_sqr_threshold, speed_mpfr_sqr,
                    2*GMP_NUMB_BITS+1);
  fprintf (f, "#define MPFR_SQR_THRESHOLD %lu /* limbs */\n",
           (unsigned long) (mpfr_sqr_threshold - 1) / GMP_NUMB_BITS + 1);

  /* Tune mpfr_div (threshold is in limbs, but it doesn't matter too much) */
  if (verbose)
    printf ("Tuning mpfr_div...\n");
  tune_simple_func (&mpfr_div_threshold, speed_mpfr_div,
                    2*GMP_NUMB_BITS+1);
  fprintf (f, "#define MPFR_DIV_THRESHOLD %lu /* limbs */\n",
           (unsigned long) (mpfr_div_threshold - 1) / GMP_NUMB_BITS + 1);

  /* Tune mpfr_exp_2 */
  if (verbose)
    printf ("Tuning mpfr_exp_2...\n");
  tune_simple_func (&mpfr_exp_2_threshold, speed_mpfr_exp_2, GMP_NUMB_BITS);
  fprintf (f, "#define MPFR_EXP_2_THRESHOLD %lu /* bits */\n",
           (unsigned long) mpfr_exp_2_threshold);

  /* Tune mpfr_exp */
  if (verbose)
    printf ("Tuning mpfr_exp...\n");
  tune_simple_func (&mpfr_exp_threshold, speed_mpfr_exp,
                    MPFR_PREC_MIN+3*GMP_NUMB_BITS);
  fprintf (f, "#define MPFR_EXP_THRESHOLD %lu /* bits */\n",
           (unsigned long) mpfr_exp_threshold);

  /* Tune mpfr_sin_cos */
  if (verbose)
    printf ("Tuning mpfr_sin_cos...\n");
  tune_simple_func (&mpfr_sincos_threshold, speed_mpfr_sincos,
                    MPFR_PREC_MIN+3*GMP_NUMB_BITS);
  fprintf (f, "#define MPFR_SINCOS_THRESHOLD %lu /* bits */\n",
           (unsigned long) mpfr_sincos_threshold);

  /* Tune mpfr_ai */
  if (verbose)
    printf ("Tuning mpfr_ai...\n");
  mpfr_init2 (x1, MPFR_SMALL_PRECISION);
  mpfr_init2 (x2, MPFR_SMALL_PRECISION);
  mpfr_init2 (x3, MPFR_SMALL_PRECISION);
  mpfr_init2 (tmp1, MPFR_SMALL_PRECISION);
  mpfr_init2 (tmp2, MPFR_SMALL_PRECISION);

  tune_simple_func_in_some_direction (&mpfr_ai_threshold1, &mpfr_ai_threshold2,
                                      &mpfr_ai_threshold3, speed_mpfr_ai,
                                      MPFR_PREC_MIN+GMP_NUMB_BITS,
                                      -60, 200, x1, &p1);
  tune_simple_func_in_some_direction (&mpfr_ai_threshold1, &mpfr_ai_threshold2,
                                      &mpfr_ai_threshold3, speed_mpfr_ai,
                                      MPFR_PREC_MIN+GMP_NUMB_BITS,
                                      -20, 500, x2, &p2);
  tune_simple_func_in_some_direction (&mpfr_ai_threshold1, &mpfr_ai_threshold2,
                                      &mpfr_ai_threshold3, speed_mpfr_ai,
                                      MPFR_PREC_MIN+GMP_NUMB_BITS,
                                      40, 200, x3, &p3);

  mpfr_mul_ui (tmp1, x2, (unsigned long)p1, MPFR_RNDN);
  mpfr_mul_ui (tmp2, x1, (unsigned long)p2, MPFR_RNDN);
  mpfr_sub (tmp1, tmp1, tmp2, MPFR_RNDN);
  mpfr_div_ui (tmp1, tmp1, MPFR_AI_SCALE, MPFR_RNDN);

  mpfr_set_ui (tmp2, (unsigned long)p1, MPFR_RNDN);
  mpfr_sub_ui (tmp2, tmp2, (unsigned long)p2, MPFR_RNDN);
  mpfr_div (tmp2, tmp2, tmp1, MPFR_RNDN);
  mpfr_ai_threshold1 = mpfr_get_si (tmp2, MPFR_RNDN);

  mpfr_sub (tmp2, x2, x1, MPFR_RNDN);
  mpfr_div (tmp2, tmp2, tmp1, MPFR_RNDN);
  mpfr_ai_threshold2 = mpfr_get_si (tmp2, MPFR_RNDN);

  mpfr_set_ui (tmp1, (unsigned long)p3, MPFR_RNDN);
  mpfr_mul_si (tmp1, tmp1, mpfr_ai_threshold2, MPFR_RNDN);
  mpfr_ui_sub (tmp1, MPFR_AI_SCALE, tmp1, MPFR_RNDN);
  mpfr_div (tmp1, tmp1, x3, MPFR_RNDN);
  mpfr_ai_threshold3 = mpfr_get_si (tmp1, MPFR_RNDN);

  fprintf (f, "#define MPFR_AI_THRESHOLD1 %ld /* threshold for negative input of mpfr_ai */\n", mpfr_ai_threshold1);
  fprintf (f, "#define MPFR_AI_THRESHOLD2 %ld\n", mpfr_ai_threshold2);
  fprintf (f, "#define MPFR_AI_THRESHOLD3 %ld\n", mpfr_ai_threshold3);

  mpfr_clear (x1); mpfr_clear (x2); mpfr_clear (x3);
  mpfr_clear (tmp1); mpfr_clear (tmp2);

  /* End of tuning */
  time (&end_time);
  fprintf (f, "/* Tuneup completed successfully, took %ld seconds */\n",
           (long) (end_time - start_time));
  if (verbose)
    printf ("Complete (took %ld seconds).\n", (long) (end_time - start_time));

  fclose (f);
}


/* Main function */
int main (int argc, char *argv[])
{
  /* Unbuffered so if output is redirected to a file it isn't lost if the
     program is killed part way through.  */
  setbuf (stdout, NULL);
  setbuf (stderr, NULL);

  verbose = argc > 1;

  if (verbose)
    printf ("Tuning MPFR (Coffee time?)...\n");

  all ("mparam.h");

  return 0;
}
