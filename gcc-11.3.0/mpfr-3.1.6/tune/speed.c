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

/* extracted from mulders.c */
#ifdef MPFR_MULHIGH_TAB_SIZE
static short mulhigh_ktab[MPFR_MULHIGH_TAB_SIZE];
#else
static short mulhigh_ktab[] = {MPFR_MULHIGH_TAB};
#define MPFR_MULHIGH_TAB_SIZE \
  ((mp_size_t) (sizeof(mulhigh_ktab) / sizeof(mulhigh_ktab[0])))
#endif

#undef _PROTO
#define _PROTO __GMP_PROTO
#include "speed.h"

int verbose;

/* s->size: precision of both input and output
   s->xp  : Mantissa of first input
   s->yp  : mantissa of second input                    */

#define SPEED_MPFR_FUNC(mean_fun) do {               \
  unsigned  i;                                       \
  mpfr_limb_ptr wp;                                  \
  double    t;                                       \
  mpfr_t    w, x;                                    \
  mp_size_t size;                                    \
  MPFR_TMP_DECL (marker);                            \
                                                     \
  SPEED_RESTRICT_COND (s->size >= MPFR_PREC_MIN);    \
  SPEED_RESTRICT_COND (s->size <= MPFR_PREC_MAX);    \
  MPFR_TMP_MARK (marker);                            \
                                                     \
  size = (s->size-1)/GMP_NUMB_BITS+1;                \
  s->xp[size-1] |= MPFR_LIMB_HIGHBIT;                \
  MPFR_TMP_INIT1 (s->xp, x, s->size);                \
  MPFR_SET_EXP (x, 0);                               \
                                                     \
  MPFR_TMP_INIT (wp, w, s->size, size);              \
                                                     \
  speed_operand_src (s, s->xp, size);                \
  speed_operand_dst (s, wp, size);                   \
  speed_cache_fill (s);                              \
                                                     \
  speed_starttime ();                                \
  i = s->reps;                                       \
  do                                                 \
    mean_fun (w, x, MPFR_RNDN);                      \
  while (--i != 0);                                  \
  t = speed_endtime ();                              \
                                                     \
  MPFR_TMP_FREE (marker);                            \
  return t;                                          \
} while (0)

#define SPEED_MPFR_OP(mean_fun) do {                 \
  unsigned  i;                                       \
  mpfr_limb_ptr wp;                                  \
  double    t;                                       \
  mpfr_t    w, x, y;                                 \
  mp_size_t size;                                    \
  MPFR_TMP_DECL (marker);                            \
                                                     \
  SPEED_RESTRICT_COND (s->size >= MPFR_PREC_MIN);    \
  SPEED_RESTRICT_COND (s->size <= MPFR_PREC_MAX);    \
  MPFR_TMP_MARK (marker);                            \
                                                     \
  size = (s->size-1)/GMP_NUMB_BITS+1;                \
  s->xp[size-1] |= MPFR_LIMB_HIGHBIT;                \
  MPFR_TMP_INIT1 (s->xp, x, s->size);                \
  MPFR_SET_EXP (x, 0);                               \
  s->yp[size-1] |= MPFR_LIMB_HIGHBIT;                \
  MPFR_TMP_INIT1 (s->yp, y, s->size);                \
  MPFR_SET_EXP (y, 0);                               \
                                                     \
  MPFR_TMP_INIT (wp, w, s->size, size);              \
                                                     \
  speed_operand_src (s, s->xp, size);                \
  speed_operand_src (s, s->yp, size);                \
  speed_operand_dst (s, wp, size);                   \
  speed_cache_fill (s);                              \
                                                     \
  speed_starttime ();                                \
  i = s->reps;                                       \
  do                                                 \
    mean_fun (w, x, y, MPFR_RNDN);                   \
  while (--i != 0);                                  \
  t = speed_endtime ();                              \
                                                     \
  MPFR_TMP_FREE (marker);                            \
  return t;                                          \
} while (0)


/* First we include all the functions we want to tune inside this program.
   We can't use GNU MPFR library since the THRESHOLD can't vary */

/* Setup mpfr_mul */
mpfr_prec_t mpfr_mul_threshold = MPFR_MUL_THRESHOLD;
static double speed_mpfr_mul (struct speed_params *s) {
  SPEED_MPFR_OP (mpfr_mul);
}



/************************************************
 * Common functions (inspired by GMP function)  *
 ************************************************/
#define THRESHOLD_WINDOW 16
#define THRESHOLD_FINAL_WINDOW 128
static double domeasure (mpfr_prec_t *threshold,
                         double (*func) (struct speed_params *),
                         mpfr_prec_t p)
{
  struct speed_params s;
  mp_size_t size;
  double t;

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
  t = speed_measure (func, &s);
  if (t == -1.0)
    {
      fprintf (stderr, "Failed to measure function!\n");
      abort ();
    }
  free (s.xp);
  return t;
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
                  mpfr_prec_t pstart, mpfr_prec_t pend)
{
  double measure;
  mpfr_prec_t p = pstart;
  mp_size_t k, n;

  while (p <= pend)
    {
      measure = domeasure (threshold, func, p);
      printf ("prec=%lu mpfr_mul=%e ", p, measure);
      n = 1 + (p - 1) / GMP_NUMB_BITS;
      if (n <= MPFR_MUL_THRESHOLD)
        k = MUL_FFT_THRESHOLD + 1;
      else if (n < MPFR_MULHIGH_TAB_SIZE)
        k = mulhigh_ktab[n];
      else
        k = 2*n/3;
      if (k < 0)
        printf ("[mpn_mul_basecase]\n");
      else if (k == 0)
        printf ("[mpfr_mulhigh_n_basecase]\n");
      else if (k > MUL_FFT_THRESHOLD)
        printf ("[mpn_mul_n]\n");
      else
        printf ("[mpfr_mulhigh_n]\n");
      p = p + p / 10;
    }
}

/*******************************************************
 *            Tune all the threshold of MPFR           *
 * Warning: tune the function in their dependent order!*
 *******************************************************/
static void
all (void)
{
  FILE *f = stdout;
  time_t  start_time, end_time;
  struct tm  *tp;

  speed_time_init ();
  if (verbose) {
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
  fprintf (f, "gcc %d.%d */\n", __GNUC__, __GNUC_MINOR__);
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
  fprintf (f, "\n");

  /* Tune mpfr_mul (threshold is in limbs, but it doesn't matter too much) */
  if (verbose)
    printf ("Measuring mpfr_mul with mpfr_mul_threshold=%lu...\n",
            mpfr_mul_threshold);
  tune_simple_func (&mpfr_mul_threshold, speed_mpfr_mul,
                    2*GMP_NUMB_BITS+1, 1000);

  /* End of tuning */
  time (&end_time);
  if (verbose)
    printf ("Complete (took %ld seconds).\n", end_time - start_time);
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

  all ();

  return 0;
}
