/* Create tuned thresholds for various algorithms.

Copyright 1999-2003, 2005, 2006, 2008-2012 Free Software Foundation, Inc.

This file is part of the GNU MP Library.

The GNU MP Library is free software; you can redistribute it and/or modify
it under the terms of either:

  * the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your
    option) any later version.

or

  * the GNU General Public License as published by the Free Software
    Foundation; either version 2 of the License, or (at your option) any
    later version.

or both in parallel, as here.

The GNU MP Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received copies of the GNU General Public License and the
GNU Lesser General Public License along with the GNU MP Library.  If not,
see https://www.gnu.org/licenses/.  */


/* Usage: tuneup [-t] [-t] [-p precision]

   -t turns on some diagnostic traces, a second -t turns on more traces.

   Notes:

   The code here isn't a vision of loveliness, mainly because it's subject
   to ongoing changes according to new things wanting to be tuned, and
   practical requirements of systems tested.

   Sometimes running the program twice produces slightly different results.
   This is probably because there's so little separating algorithms near
   their crossover, and on that basis it should make little or no difference
   to the final speed of the relevant routines, but nothing has been done to
   check that carefully.

   Algorithm:

   The thresholds are determined as follows.  A crossover may not be a
   single size but rather a range where it oscillates between method A or
   method B faster.  If the threshold is set making B used where A is faster
   (or vice versa) that's bad.  Badness is the percentage time lost and
   total badness is the sum of this over all sizes measured.  The threshold
   is set to minimize total badness.

   Suppose, as sizes increase, method B becomes faster than method A.  The
   effect of the rule is that, as you look at increasing sizes, isolated
   points where B is faster are ignored, but when it's consistently faster,
   or faster on balance, then the threshold is set there.  The same result
   is obtained thinking in the other direction of A becoming faster at
   smaller sizes.

   In practice the thresholds tend to be chosen to bring on the next
   algorithm fairly quickly.

   This rule is attractive because it's got a basis in reason and is fairly
   easy to implement, but no work has been done to actually compare it in
   absolute terms to other possibilities.

   Implementation:

   In a normal library build the thresholds are constants.  To tune them
   selected objects are recompiled with the thresholds as global variables
   instead.  #define TUNE_PROGRAM_BUILD does this, with help from code at
   the end of gmp-impl.h, and rules in tune/Makefile.am.

   MUL_TOOM22_THRESHOLD for example uses a recompiled mpn_mul_n.  The
   threshold is set to "size+1" to avoid karatsuba, or to "size" to use one
   level, but recurse into the basecase.

   MUL_TOOM33_THRESHOLD makes use of the tuned MUL_TOOM22_THRESHOLD value.
   Other routines in turn will make use of both of those.  Naturally the
   dependants must be tuned first.

   In a couple of cases, like DIVEXACT_1_THRESHOLD, there's no recompiling,
   just a threshold based on comparing two routines (mpn_divrem_1 and
   mpn_divexact_1), and no further use of the value determined.

   Flags like USE_PREINV_MOD_1 or JACOBI_BASE_METHOD are even simpler, being
   just comparisons between certain routines on representative data.

   Shortcuts are applied when native (assembler) versions of routines exist.
   For instance a native mpn_sqr_basecase is assumed to be always faster
   than mpn_mul_basecase, with no measuring.

   No attempt is made to tune within assembler routines, for instance
   DIVREM_1_NORM_THRESHOLD.  An assembler mpn_divrem_1 is expected to be
   written and tuned all by hand.  Assembler routines that might have hard
   limits are recompiled though, to make them accept a bigger range of sizes
   than normal, eg. mpn_sqr_basecase to compare against mpn_toom2_sqr.

   Limitations:

   The FFTs aren't subject to the same badness rule as the other thresholds,
   so each k is probably being brought on a touch early.  This isn't likely
   to make a difference, and the simpler probing means fewer tests.

*/

#define TUNE_PROGRAM_BUILD  1   /* for gmp-impl.h */

#include "config.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#if HAVE_UNISTD_H
#include <unistd.h>
#endif

#include "gmp.h"
#include "gmp-impl.h"
#include "longlong.h"

#include "tests.h"
#include "speed.h"

#if !HAVE_DECL_OPTARG
extern char *optarg;
extern int optind, opterr;
#endif


#define DEFAULT_MAX_SIZE   1000  /* limbs */

#if WANT_FFT
mp_size_t  option_fft_max_size = 50000;  /* limbs */
#else
mp_size_t  option_fft_max_size = 0;
#endif
int        option_trace = 0;
int        option_fft_trace = 0;
struct speed_params  s;

struct dat_t {
  mp_size_t  size;
  double     d;
} *dat = NULL;
int  ndat = 0;
int  allocdat = 0;

/* This is not defined if mpn_sqr_basecase doesn't declare a limit.  In that
   case use zero here, which for params.max_size means no limit.  */
#ifndef TUNE_SQR_TOOM2_MAX
#define TUNE_SQR_TOOM2_MAX  0
#endif

mp_size_t  mul_toom22_threshold         = MP_SIZE_T_MAX;
mp_size_t  mul_toom33_threshold         = MUL_TOOM33_THRESHOLD_LIMIT;
mp_size_t  mul_toom44_threshold         = MUL_TOOM44_THRESHOLD_LIMIT;
mp_size_t  mul_toom6h_threshold         = MUL_TOOM6H_THRESHOLD_LIMIT;
mp_size_t  mul_toom8h_threshold         = MUL_TOOM8H_THRESHOLD_LIMIT;
mp_size_t  mul_toom32_to_toom43_threshold = MP_SIZE_T_MAX;
mp_size_t  mul_toom32_to_toom53_threshold = MP_SIZE_T_MAX;
mp_size_t  mul_toom42_to_toom53_threshold = MP_SIZE_T_MAX;
mp_size_t  mul_toom42_to_toom63_threshold = MP_SIZE_T_MAX;
mp_size_t  mul_toom43_to_toom54_threshold = MP_SIZE_T_MAX;
mp_size_t  mul_fft_threshold            = MP_SIZE_T_MAX;
mp_size_t  mul_fft_modf_threshold       = MP_SIZE_T_MAX;
mp_size_t  sqr_basecase_threshold       = MP_SIZE_T_MAX;
mp_size_t  sqr_toom2_threshold
  = (TUNE_SQR_TOOM2_MAX == 0 ? MP_SIZE_T_MAX : TUNE_SQR_TOOM2_MAX);
mp_size_t  sqr_toom3_threshold          = SQR_TOOM3_THRESHOLD_LIMIT;
mp_size_t  sqr_toom4_threshold          = SQR_TOOM4_THRESHOLD_LIMIT;
mp_size_t  sqr_toom6_threshold          = SQR_TOOM6_THRESHOLD_LIMIT;
mp_size_t  sqr_toom8_threshold          = SQR_TOOM8_THRESHOLD_LIMIT;
mp_size_t  sqr_fft_threshold            = MP_SIZE_T_MAX;
mp_size_t  sqr_fft_modf_threshold       = MP_SIZE_T_MAX;
mp_size_t  mullo_basecase_threshold     = MP_SIZE_T_MAX;
mp_size_t  mullo_dc_threshold           = MP_SIZE_T_MAX;
mp_size_t  mullo_mul_n_threshold        = MP_SIZE_T_MAX;
mp_size_t  sqrlo_basecase_threshold     = MP_SIZE_T_MAX;
mp_size_t  sqrlo_dc_threshold           = MP_SIZE_T_MAX;
mp_size_t  sqrlo_sqr_threshold          = MP_SIZE_T_MAX;
mp_size_t  mulmid_toom42_threshold      = MP_SIZE_T_MAX;
mp_size_t  mulmod_bnm1_threshold        = MP_SIZE_T_MAX;
mp_size_t  sqrmod_bnm1_threshold        = MP_SIZE_T_MAX;
mp_size_t  div_qr_2_pi2_threshold       = MP_SIZE_T_MAX;
mp_size_t  dc_div_qr_threshold          = MP_SIZE_T_MAX;
mp_size_t  dc_divappr_q_threshold       = MP_SIZE_T_MAX;
mp_size_t  mu_div_qr_threshold          = MP_SIZE_T_MAX;
mp_size_t  mu_divappr_q_threshold       = MP_SIZE_T_MAX;
mp_size_t  mupi_div_qr_threshold        = MP_SIZE_T_MAX;
mp_size_t  mu_div_q_threshold           = MP_SIZE_T_MAX;
mp_size_t  dc_bdiv_qr_threshold         = MP_SIZE_T_MAX;
mp_size_t  dc_bdiv_q_threshold          = MP_SIZE_T_MAX;
mp_size_t  mu_bdiv_qr_threshold         = MP_SIZE_T_MAX;
mp_size_t  mu_bdiv_q_threshold          = MP_SIZE_T_MAX;
mp_size_t  inv_mulmod_bnm1_threshold    = MP_SIZE_T_MAX;
mp_size_t  inv_newton_threshold         = MP_SIZE_T_MAX;
mp_size_t  inv_appr_threshold           = MP_SIZE_T_MAX;
mp_size_t  binv_newton_threshold        = MP_SIZE_T_MAX;
mp_size_t  redc_1_to_redc_2_threshold   = MP_SIZE_T_MAX;
mp_size_t  redc_1_to_redc_n_threshold   = MP_SIZE_T_MAX;
mp_size_t  redc_2_to_redc_n_threshold   = MP_SIZE_T_MAX;
mp_size_t  matrix22_strassen_threshold  = MP_SIZE_T_MAX;
mp_size_t  hgcd_threshold               = MP_SIZE_T_MAX;
mp_size_t  hgcd_appr_threshold          = MP_SIZE_T_MAX;
mp_size_t  hgcd_reduce_threshold        = MP_SIZE_T_MAX;
mp_size_t  gcd_dc_threshold             = MP_SIZE_T_MAX;
mp_size_t  gcdext_dc_threshold          = MP_SIZE_T_MAX;
int	   div_qr_1n_pi1_method		= 0;
mp_size_t  div_qr_1_norm_threshold      = MP_SIZE_T_MAX;
mp_size_t  div_qr_1_unnorm_threshold    = MP_SIZE_T_MAX;
mp_size_t  divrem_1_norm_threshold      = MP_SIZE_T_MAX;
mp_size_t  divrem_1_unnorm_threshold    = MP_SIZE_T_MAX;
mp_size_t  mod_1_norm_threshold         = MP_SIZE_T_MAX;
mp_size_t  mod_1_unnorm_threshold       = MP_SIZE_T_MAX;
int	   mod_1_1p_method		= 0;
mp_size_t  mod_1n_to_mod_1_1_threshold  = MP_SIZE_T_MAX;
mp_size_t  mod_1u_to_mod_1_1_threshold  = MP_SIZE_T_MAX;
mp_size_t  mod_1_1_to_mod_1_2_threshold = MP_SIZE_T_MAX;
mp_size_t  mod_1_2_to_mod_1_4_threshold = MP_SIZE_T_MAX;
mp_size_t  preinv_mod_1_to_mod_1_threshold = MP_SIZE_T_MAX;
mp_size_t  divrem_2_threshold           = MP_SIZE_T_MAX;
mp_size_t  get_str_dc_threshold         = MP_SIZE_T_MAX;
mp_size_t  get_str_precompute_threshold = MP_SIZE_T_MAX;
mp_size_t  set_str_dc_threshold         = MP_SIZE_T_MAX;
mp_size_t  set_str_precompute_threshold = MP_SIZE_T_MAX;
mp_size_t  fac_odd_threshold            = 0;
mp_size_t  fac_dsc_threshold            = FAC_DSC_THRESHOLD_LIMIT;

mp_size_t  fft_modf_sqr_threshold = MP_SIZE_T_MAX;
mp_size_t  fft_modf_mul_threshold = MP_SIZE_T_MAX;

struct param_t {
  const char        *name;
  speed_function_t  function;
  speed_function_t  function2;
  double            step_factor;    /* how much to step relatively */
  int               step;           /* how much to step absolutely */
  double            function_fudge; /* multiplier for "function" speeds */
  int               stop_since_change;
  double            stop_factor;
  mp_size_t         min_size;
  int               min_is_always;
  mp_size_t         max_size;
  mp_size_t         check_size;
  mp_size_t         size_extra;

#define DATA_HIGH_LT_R  1
#define DATA_HIGH_GE_R  2
  int               data_high;

  int               noprint;
};


/* These are normally undefined when false, which suits "#if" fine.
   But give them zero values so they can be used in plain C "if"s.  */
#ifndef UDIV_PREINV_ALWAYS
#define UDIV_PREINV_ALWAYS 0
#endif
#ifndef HAVE_NATIVE_mpn_divexact_1
#define HAVE_NATIVE_mpn_divexact_1 0
#endif
#ifndef HAVE_NATIVE_mpn_div_qr_1n_pi1
#define HAVE_NATIVE_mpn_div_qr_1n_pi1 0
#endif
#ifndef HAVE_NATIVE_mpn_divrem_1
#define HAVE_NATIVE_mpn_divrem_1 0
#endif
#ifndef HAVE_NATIVE_mpn_divrem_2
#define HAVE_NATIVE_mpn_divrem_2 0
#endif
#ifndef HAVE_NATIVE_mpn_mod_1
#define HAVE_NATIVE_mpn_mod_1 0
#endif
#ifndef HAVE_NATIVE_mpn_mod_1_1p
#define HAVE_NATIVE_mpn_mod_1_1p 0
#endif
#ifndef HAVE_NATIVE_mpn_modexact_1_odd
#define HAVE_NATIVE_mpn_modexact_1_odd 0
#endif
#ifndef HAVE_NATIVE_mpn_preinv_divrem_1
#define HAVE_NATIVE_mpn_preinv_divrem_1 0
#endif
#ifndef HAVE_NATIVE_mpn_preinv_mod_1
#define HAVE_NATIVE_mpn_preinv_mod_1 0
#endif
#ifndef HAVE_NATIVE_mpn_sqr_basecase
#define HAVE_NATIVE_mpn_sqr_basecase 0
#endif


#define MAX3(a,b,c)  MAX (MAX (a, b), c)

mp_limb_t
randlimb_norm (void)
{
  mp_limb_t  n;
  mpn_random (&n, 1);
  n |= GMP_NUMB_HIGHBIT;
  return n;
}

#define GMP_NUMB_HALFMASK  ((CNST_LIMB(1) << (GMP_NUMB_BITS/2)) - 1)

mp_limb_t
randlimb_half (void)
{
  mp_limb_t  n;
  mpn_random (&n, 1);
  n &= GMP_NUMB_HALFMASK;
  n += (n==0);
  return n;
}


/* Add an entry to the end of the dat[] array, reallocing to make it bigger
   if necessary.  */
void
add_dat (mp_size_t size, double d)
{
#define ALLOCDAT_STEP  500

  ASSERT_ALWAYS (ndat <= allocdat);

  if (ndat == allocdat)
    {
      dat = (struct dat_t *) __gmp_allocate_or_reallocate
        (dat, allocdat * sizeof(dat[0]),
         (allocdat+ALLOCDAT_STEP) * sizeof(dat[0]));
      allocdat += ALLOCDAT_STEP;
    }

  dat[ndat].size = size;
  dat[ndat].d = d;
  ndat++;
}


/* Return the threshold size based on the data accumulated. */
mp_size_t
analyze_dat (int final)
{
  double  x, min_x;
  int     j, min_j;

  /* If the threshold is set at dat[0].size, any positive values are bad. */
  x = 0.0;
  for (j = 0; j < ndat; j++)
    if (dat[j].d > 0.0)
      x += dat[j].d;

  if (option_trace >= 2 && final)
    {
      printf ("\n");
      printf ("x is the sum of the badness from setting thresh at given size\n");
      printf ("  (minimum x is sought)\n");
      printf ("size=%ld  first x=%.4f\n", (long) dat[j].size, x);
    }

  min_x = x;
  min_j = 0;


  /* When stepping to the next dat[j].size, positive values are no longer
     bad (so subtracted), negative values become bad (so add the absolute
     value, meaning subtract). */
  for (j = 0; j < ndat; x -= dat[j].d, j++)
    {
      if (option_trace >= 2 && final)
        printf ("size=%ld  x=%.4f\n", (long) dat[j].size, x);

      if (x < min_x)
        {
          min_x = x;
          min_j = j;
        }
    }

  return min_j;
}


/* Measuring for recompiled mpn/generic/div_qr_1.c,
 * mpn/generic/divrem_1.c, mpn/generic/mod_1.c and mpz/fac_ui.c */

mp_limb_t mpn_div_qr_1_tune (mp_ptr, mp_limb_t *, mp_srcptr, mp_size_t, mp_limb_t);

#if defined (__cplusplus)
extern "C" {
#endif

mp_limb_t mpn_divrem_1_tune (mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t);
mp_limb_t mpn_mod_1_tune (mp_srcptr, mp_size_t, mp_limb_t);
void mpz_fac_ui_tune (mpz_ptr, unsigned long);

#if defined (__cplusplus)
}
#endif

double
speed_mpn_mod_1_tune (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_MOD_1 (mpn_mod_1_tune);
}
double
speed_mpn_divrem_1_tune (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_DIVREM_1 (mpn_divrem_1_tune);
}
double
speed_mpz_fac_ui_tune (struct speed_params *s)
{
  SPEED_ROUTINE_MPZ_FAC_UI (mpz_fac_ui_tune);
}
double
speed_mpn_div_qr_1_tune (struct speed_params *s)
{
  SPEED_ROUTINE_MPN_DIV_QR_1 (mpn_div_qr_1_tune);
}

double
tuneup_measure (speed_function_t fun,
                const struct param_t *param,
                struct speed_params *s)
{
  static struct param_t  dummy;
  double   t;
  TMP_DECL;

  if (! param)
    param = &dummy;

  s->size += param->size_extra;

  TMP_MARK;
  SPEED_TMP_ALLOC_LIMBS (s->xp, s->size, 0);
  SPEED_TMP_ALLOC_LIMBS (s->yp, s->size, 0);

  mpn_random (s->xp, s->size);
  mpn_random (s->yp, s->size);

  switch (param->data_high) {
  case DATA_HIGH_LT_R:
    s->xp[s->size-1] %= s->r;
    s->yp[s->size-1] %= s->r;
    break;
  case DATA_HIGH_GE_R:
    s->xp[s->size-1] |= s->r;
    s->yp[s->size-1] |= s->r;
    break;
  }

  t = speed_measure (fun, s);

  s->size -= param->size_extra;

  TMP_FREE;
  return t;
}


#define PRINT_WIDTH  31

void
print_define_start (const char *name)
{
  printf ("#define %-*s  ", PRINT_WIDTH, name);
  if (option_trace)
    printf ("...\n");
}

void
print_define_end_remark (const char *name, mp_size_t value, const char *remark)
{
  if (option_trace)
    printf ("#define %-*s  ", PRINT_WIDTH, name);

  if (value == MP_SIZE_T_MAX)
    printf ("MP_SIZE_T_MAX");
  else
    printf ("%5ld", (long) value);

  if (remark != NULL)
    printf ("  /* %s */", remark);
  printf ("\n");
  fflush (stdout);
}

void
print_define_end (const char *name, mp_size_t value)
{
  const char  *remark;
  if (value == MP_SIZE_T_MAX)
    remark = "never";
  else if (value == 0)
    remark = "always";
  else
    remark = NULL;
  print_define_end_remark (name, value, remark);
}

void
print_define (const char *name, mp_size_t value)
{
  print_define_start (name);
  print_define_end (name, value);
}

void
print_define_remark (const char *name, mp_size_t value, const char *remark)
{
  print_define_start (name);
  print_define_end_remark (name, value, remark);
}


void
one (mp_size_t *threshold, struct param_t *param)
{
  int  since_positive, since_thresh_change;
  int  thresh_idx, new_thresh_idx;

#define DEFAULT(x,n)  do { if (! (x))  (x) = (n); } while (0)

  DEFAULT (param->function_fudge, 1.0);
  DEFAULT (param->function2, param->function);
  DEFAULT (param->step_factor, 0.01);  /* small steps by default */
  DEFAULT (param->step, 1);            /* small steps by default */
  DEFAULT (param->stop_since_change, 80);
  DEFAULT (param->stop_factor, 1.2);
  DEFAULT (param->min_size, 10);
  DEFAULT (param->max_size, DEFAULT_MAX_SIZE);

  if (param->check_size != 0)
    {
      double   t1, t2;
      s.size = param->check_size;

      *threshold = s.size+1;
      t1 = tuneup_measure (param->function, param, &s);

      *threshold = s.size;
      t2 = tuneup_measure (param->function2, param, &s);
      if (t1 == -1.0 || t2 == -1.0)
        {
          printf ("Oops, can't run both functions at size %ld\n",
                  (long) s.size);
          abort ();
        }
      t1 *= param->function_fudge;

      /* ask that t2 is at least 4% below t1 */
      if (t1 < t2*1.04)
        {
          if (option_trace)
            printf ("function2 never enough faster: t1=%.9f t2=%.9f\n", t1, t2);
          *threshold = MP_SIZE_T_MAX;
          if (! param->noprint)
            print_define (param->name, *threshold);
          return;
        }

      if (option_trace >= 2)
        printf ("function2 enough faster at size=%ld: t1=%.9f t2=%.9f\n",
                (long) s.size, t1, t2);
    }

  if (! param->noprint || option_trace)
    print_define_start (param->name);

  ndat = 0;
  since_positive = 0;
  since_thresh_change = 0;
  thresh_idx = 0;

  if (option_trace >= 2)
    {
      printf ("             algorithm-A  algorithm-B   ratio  possible\n");
      printf ("              (seconds)    (seconds)    diff    thresh\n");
    }

  for (s.size = param->min_size;
       s.size < param->max_size;
       s.size += MAX ((mp_size_t) floor (s.size * param->step_factor), param->step))
    {
      double   ti, tiplus1, d;

      /*
        FIXME: check minimum size requirements are met, possibly by just
        checking for the -1 returns from the speed functions.
      */

      /* using method A at this size */
      *threshold = s.size+1;
      ti = tuneup_measure (param->function, param, &s);
      if (ti == -1.0)
        abort ();
      ti *= param->function_fudge;

      /* using method B at this size */
      *threshold = s.size;
      tiplus1 = tuneup_measure (param->function2, param, &s);
      if (tiplus1 == -1.0)
        abort ();

      /* Calculate the fraction by which the one or the other routine is
         slower.  */
      if (tiplus1 >= ti)
        d = (tiplus1 - ti) / tiplus1;  /* negative */
      else
        d = (tiplus1 - ti) / ti;       /* positive */

      add_dat (s.size, d);

      new_thresh_idx = analyze_dat (0);

      if (option_trace >= 2)
        printf ("size=%ld  %.9f  %.9f  % .4f %c  %ld\n",
                (long) s.size, ti, tiplus1, d,
                ti > tiplus1 ? '#' : ' ',
                (long) dat[new_thresh_idx].size);

      /* Stop if the last time method i was faster was more than a
         certain number of measurements ago.  */
#define STOP_SINCE_POSITIVE  200
      if (d >= 0)
        since_positive = 0;
      else
        if (++since_positive > STOP_SINCE_POSITIVE)
          {
            if (option_trace >= 1)
              printf ("stopped due to since_positive (%d)\n",
                      STOP_SINCE_POSITIVE);
            break;
          }

      /* Stop if method A has become slower by a certain factor. */
      if (ti >= tiplus1 * param->stop_factor)
        {
          if (option_trace >= 1)
            printf ("stopped due to ti >= tiplus1 * factor (%.1f)\n",
                    param->stop_factor);
          break;
        }

      /* Stop if the threshold implied hasn't changed in a certain
         number of measurements.  (It's this condition that usually
         stops the loop.) */
      if (thresh_idx != new_thresh_idx)
        since_thresh_change = 0, thresh_idx = new_thresh_idx;
      else
        if (++since_thresh_change > param->stop_since_change)
          {
            if (option_trace >= 1)
              printf ("stopped due to since_thresh_change (%d)\n",
                      param->stop_since_change);
            break;
          }

      /* Stop if the threshold implied is more than a certain number of
         measurements ago.  */
#define STOP_SINCE_AFTER   500
      if (ndat - thresh_idx > STOP_SINCE_AFTER)
        {
          if (option_trace >= 1)
            printf ("stopped due to ndat - thresh_idx > amount (%d)\n",
                    STOP_SINCE_AFTER);
          break;
        }

      /* Stop when the size limit is reached before the end of the
         crossover, but only show this as an error for >= the default max
         size.  FIXME: Maybe should make it a param choice whether this is
         an error.  */
      if (s.size >= param->max_size && param->max_size >= DEFAULT_MAX_SIZE)
        {
          fprintf (stderr, "%s\n", param->name);
          fprintf (stderr, "sizes %ld to %ld total %d measurements\n",
                   (long) dat[0].size, (long) dat[ndat-1].size, ndat);
          fprintf (stderr, "    max size reached before end of crossover\n");
          break;
        }
    }

  if (option_trace >= 1)
    printf ("sizes %ld to %ld total %d measurements\n",
            (long) dat[0].size, (long) dat[ndat-1].size, ndat);

  *threshold = dat[analyze_dat (1)].size;

  if (param->min_is_always)
    {
      if (*threshold == param->min_size)
        *threshold = 0;
    }

  if (! param->noprint || option_trace)
    print_define_end (param->name, *threshold);
}


/* Special probing for the fft thresholds.  The size restrictions on the
   FFTs mean the graph of time vs size has a step effect.  See this for
   example using

       ./speed -s 4096-16384 -t 128 -P foo mpn_mul_fft.8 mpn_mul_fft.9
       gnuplot foo.gnuplot

   The current approach is to compare routines at the midpoint of relevant
   steps.  Arguably a more sophisticated system of threshold data is wanted
   if this step effect remains. */

struct fft_param_t {
  const char        *table_name;
  const char        *threshold_name;
  const char        *modf_threshold_name;
  mp_size_t         *p_threshold;
  mp_size_t         *p_modf_threshold;
  mp_size_t         first_size;
  mp_size_t         max_size;
  speed_function_t  function;
  speed_function_t  mul_modf_function;
  speed_function_t  mul_function;
  mp_size_t         sqr;
};


/* mpn_mul_fft requires pl a multiple of 2^k limbs, but with
   N=pl*BIT_PER_MP_LIMB it internally also pads out so N/2^k is a multiple
   of 2^(k-1) bits. */

mp_size_t
fft_step_size (int k)
{
  mp_size_t  step;

  step = MAX ((mp_size_t) 1 << (k-1), GMP_LIMB_BITS) / GMP_LIMB_BITS;
  step *= (mp_size_t) 1 << k;

  if (step <= 0)
    {
      printf ("Can't handle k=%d\n", k);
      abort ();
    }

  return step;
}

mp_size_t
fft_next_size (mp_size_t pl, int k)
{
  mp_size_t  m = fft_step_size (k);

/*    printf ("[k=%d %ld] %ld ->", k, m, pl); */

  if (pl == 0 || (pl & (m-1)) != 0)
    pl = (pl | (m-1)) + 1;

/*    printf (" %ld\n", pl); */
  return pl;
}

#define NMAX_DEFAULT 1000000
#define MAX_REPS 25
#define MIN_REPS 5

static inline size_t
mpn_mul_fft_lcm (size_t a, unsigned int k)
{
  unsigned int l = k;

  while (a % 2 == 0 && k > 0)
    {
      a >>= 1;
      k--;
    }
  return a << l;
}

mp_size_t
fftfill (mp_size_t pl, int k, int sqr)
{
  mp_size_t maxLK;
  mp_bitcnt_t N, Nprime, nprime, M;

  N = pl * GMP_NUMB_BITS;
  M = N >> k;

  maxLK = mpn_mul_fft_lcm ((unsigned long) GMP_NUMB_BITS, k);

  Nprime = (1 + (2 * M + k + 2) / maxLK) * maxLK;
  nprime = Nprime / GMP_NUMB_BITS;
  if (nprime >= (sqr ? SQR_FFT_MODF_THRESHOLD : MUL_FFT_MODF_THRESHOLD))
    {
      size_t K2;
      for (;;)
	{
	  K2 = 1L << mpn_fft_best_k (nprime, sqr);
	  if ((nprime & (K2 - 1)) == 0)
	    break;
	  nprime = (nprime + K2 - 1) & -K2;
	  Nprime = nprime * GMP_LIMB_BITS;
	}
    }
  ASSERT_ALWAYS (nprime < pl);

  return Nprime;
}

static int
compare_double (const void *ap, const void *bp)
{
  double a = * (const double *) ap;
  double b = * (const double *) bp;

  if (a < b)
    return -1;
  else if (a > b)
    return 1;
  else
    return 0;
}

double
median (double *times, int n)
{
  qsort (times, n, sizeof (double), compare_double);
  return times[n/2];
}

#define FFT_CACHE_SIZE 25
typedef struct fft_cache
{
  mp_size_t n;
  double time;
} fft_cache_t;

fft_cache_t fft_cache[FFT_CACHE_SIZE];

double
cached_measure (mp_ptr rp, mp_srcptr ap, mp_srcptr bp, mp_size_t n, int k,
		int n_measurements)
{
  int i;
  double t, ttab[MAX_REPS];

  if (fft_cache[k].n == n)
    return fft_cache[k].time;

  for (i = 0; i < n_measurements; i++)
    {
      speed_starttime ();
      mpn_mul_fft (rp, n, ap, n, bp, n, k);
      ttab[i] = speed_endtime ();
    }

  t = median (ttab, n_measurements);
  fft_cache[k].n = n;
  fft_cache[k].time = t;
  return t;
}

#define INSERT_FFTTAB(idx, nval, kval)					\
  do {									\
    fft_tab[idx].n = nval;						\
    fft_tab[idx].k = kval;						\
    fft_tab[idx+1].n = (1 << 27) - 1;	/* sentinel, 27b wide field */	\
    fft_tab[idx+1].k = (1 <<  5) - 1;					\
  } while (0)

int
fftmes (mp_size_t nmin, mp_size_t nmax, int initial_k, struct fft_param_t *p, int idx, int print)
{
  mp_size_t n, n1, prev_n1;
  int k, best_k, last_best_k, kmax;
  int eff, prev_eff;
  double t0, t1;
  int n_measurements;
  mp_limb_t *ap, *bp, *rp;
  mp_size_t alloc;
  struct fft_table_nk *fft_tab;

  fft_tab = mpn_fft_table3[p->sqr];

  for (k = 0; k < FFT_CACHE_SIZE; k++)
    fft_cache[k].n = 0;

  if (nmin < (p->sqr ? SQR_FFT_MODF_THRESHOLD : MUL_FFT_MODF_THRESHOLD))
    {
      nmin = (p->sqr ? SQR_FFT_MODF_THRESHOLD : MUL_FFT_MODF_THRESHOLD);
    }

  if (print)
    printf ("#define %s%*s", p->table_name, 38, "");

  if (idx == 0)
    {
      INSERT_FFTTAB (0, nmin, initial_k);

      if (print)
	{
	  printf ("\\\n  { ");
	  printf ("{%7u,%2u}", fft_tab[0].n, fft_tab[0].k);
	}

      idx = 1;
    }

  ap = (mp_ptr) malloc (sizeof (mp_limb_t));
  if (p->sqr)
    bp = ap;
  else
    bp = (mp_ptr) malloc (sizeof (mp_limb_t));
  rp = (mp_ptr) malloc (sizeof (mp_limb_t));
  alloc = 1;

  /* Round n to comply to initial k value */
  n = (nmin + ((1ul << initial_k) - 1)) & (MP_SIZE_T_MAX << initial_k);

  n_measurements = (18 - initial_k) | 1;
  n_measurements = MAX (n_measurements, MIN_REPS);
  n_measurements = MIN (n_measurements, MAX_REPS);

  last_best_k = initial_k;
  best_k = initial_k;

  while (n < nmax)
    {
      int start_k, end_k;

      /* Assume the current best k is best until we hit its next FFT step.  */
      t0 = 99999;

      prev_n1 = n + 1;

      start_k = MAX (4, best_k - 4);
      end_k = MIN (24, best_k + 4);
      for (k = start_k; k <= end_k; k++)
	{
          n1 = mpn_fft_next_size (prev_n1, k);

	  eff = 200 * (n1 * GMP_NUMB_BITS >> k) / fftfill (n1, k, p->sqr);

	  if (eff < 70)		/* avoid measuring too slow fft:s */
	    continue;

	  if (n1 > alloc)
	    {
	      alloc = n1;
	      if (p->sqr)
		{
		  ap = (mp_ptr) realloc (ap, sizeof (mp_limb_t));
		  rp = (mp_ptr) realloc (rp, sizeof (mp_limb_t));
		  ap = bp = (mp_ptr) realloc (ap, alloc * sizeof (mp_limb_t));
		  mpn_random (ap, alloc);
		  rp = (mp_ptr) realloc (rp, alloc * sizeof (mp_limb_t));
		}
	      else
		{
		  ap = (mp_ptr) realloc (ap, sizeof (mp_limb_t));
		  bp = (mp_ptr) realloc (bp, sizeof (mp_limb_t));
		  rp = (mp_ptr) realloc (rp, sizeof (mp_limb_t));
		  ap = (mp_ptr) realloc (ap, alloc * sizeof (mp_limb_t));
		  mpn_random (ap, alloc);
		  bp = (mp_ptr) realloc (bp, alloc * sizeof (mp_limb_t));
		  mpn_random (bp, alloc);
		  rp = (mp_ptr) realloc (rp, alloc * sizeof (mp_limb_t));
		}
	    }

	  t1 = cached_measure (rp, ap, bp, n1, k, n_measurements);

	  if (t1 * n_measurements > 0.3)
	    n_measurements -= 2;
	  n_measurements = MAX (n_measurements, MIN_REPS);

	  if (t1 < t0)
	    {
	      best_k = k;
	      t0 = t1;
	    }
	}

      n1 = mpn_fft_next_size (prev_n1, best_k);

      if (last_best_k != best_k)
	{
	  ASSERT_ALWAYS ((prev_n1 & ((1ul << last_best_k) - 1)) == 1);

	  if (idx >= FFT_TABLE3_SIZE)
	    {
	      printf ("FFT table exhausted, increase FFT_TABLE3_SIZE in gmp-impl.h\n");
	      abort ();
	    }
	  INSERT_FFTTAB (idx, prev_n1 >> last_best_k, best_k);

	  if (print)
	    {
	      printf (", ");
	      if (idx % 4 == 0)
		printf ("\\\n    ");
	      printf ("{%7u,%2u}", fft_tab[idx].n, fft_tab[idx].k);
	    }

	  if (option_trace >= 2)
	    {
	      printf ("{%lu,%u}\n", prev_n1, best_k);
	      fflush (stdout);
	    }

	  last_best_k = best_k;
	  idx++;
	}

      for (;;)
	{
	  prev_n1 = n1;
	  prev_eff = fftfill (prev_n1, best_k, p->sqr);
	  n1 = mpn_fft_next_size (prev_n1 + 1, best_k);
	  eff = fftfill (n1, best_k, p->sqr);

	  if (eff != prev_eff)
	    break;
	}

      n = prev_n1;
    }

  kmax = sizeof (mp_size_t) * 4;	/* GMP_MP_SIZE_T_BITS / 2 */
  kmax = MIN (kmax, 25-1);
  for (k = last_best_k + 1; k <= kmax; k++)
    {
      if (idx >= FFT_TABLE3_SIZE)
	{
	  printf ("FFT table exhausted, increase FFT_TABLE3_SIZE in gmp-impl.h\n");
	  abort ();
	}
      INSERT_FFTTAB (idx, ((1ul << (2*k-2)) + 1) >> (k-1), k);

      if (print)
	{
	  printf (", ");
	  if (idx % 4 == 0)
	    printf ("\\\n    ");
	  printf ("{%7u,%2u}", fft_tab[idx].n, fft_tab[idx].k);
	}

      idx++;
    }

  if (print)
    printf (" }\n");

  free (ap);
  if (! p->sqr)
    free (bp);
  free (rp);

  return idx;
}

void
fft (struct fft_param_t *p)
{
  mp_size_t  size;
  int        k, idx, initial_k;

  /*** Generate MUL_FFT_MODF_THRESHOLD / SQR_FFT_MODF_THRESHOLD ***/

#if 1
  {
    /* Use plain one() mechanism, for some reasonable initial values of k.  The
       advantage is that we don't depend on mpn_fft_table3, which can therefore
       leave it completely uninitialized.  */

    static struct param_t param;
    mp_size_t thres, best_thres;
    int best_k;
    char buf[20];

    best_thres = MP_SIZE_T_MAX;
    best_k = -1;

    for (k = 5; k <= 7; k++)
      {
	param.name = p->modf_threshold_name;
	param.min_size = 100;
	param.max_size = 2000;
	param.function  = p->mul_function;
	param.step_factor = 0.0;
	param.step = 4;
	param.function2 = p->mul_modf_function;
	param.noprint = 1;
	s.r = k;
	one (&thres, &param);
	if (thres < best_thres)
	  {
	    best_thres = thres;
	    best_k = k;
	  }
      }

    *(p->p_modf_threshold) = best_thres;
    sprintf (buf, "k = %d", best_k);
    print_define_remark (p->modf_threshold_name, best_thres, buf);
    initial_k = best_k;
  }
#else
  size = p->first_size;
  for (;;)
    {
      double  tk, tm;

      size = mpn_fft_next_size (size+1, mpn_fft_best_k (size+1, p->sqr));
      k = mpn_fft_best_k (size, p->sqr);

      if (size >= p->max_size)
        break;

      s.size = size + fft_step_size (k) / 2;
      s.r = k;
      tk = tuneup_measure (p->mul_modf_function, NULL, &s);
      if (tk == -1.0)
        abort ();

      tm = tuneup_measure (p->mul_function, NULL, &s);
      if (tm == -1.0)
        abort ();

      if (option_trace >= 2)
        printf ("at %ld   size=%ld  k=%d  %.9f   size=%ld modf %.9f\n",
                (long) size,
                (long) size + fft_step_size (k) / 2, k, tk,
                (long) s.size, tm);

      if (tk < tm)
        {
	  *p->p_modf_threshold = s.size;
	  print_define (p->modf_threshold_name, *p->p_modf_threshold);
	  break;
        }
    }
  initial_k = ?;
#endif

  /*** Generate MUL_FFT_TABLE3 / SQR_FFT_TABLE3 ***/

  idx = fftmes (*p->p_modf_threshold, p->max_size, initial_k, p, 0, 1);
  printf ("#define %s_SIZE %d\n", p->table_name, idx);

  /*** Generate MUL_FFT_THRESHOLD / SQR_FFT_THRESHOLD ***/

  size = 2 * *p->p_modf_threshold;	/* OK? */
  for (;;)
    {
      double  tk, tm;
      mp_size_t mulmod_size, mul_size;;

      if (size >= p->max_size)
        break;

      mulmod_size = mpn_mulmod_bnm1_next_size (2 * (size + 1)) / 2;
      mul_size = (size + mulmod_size) / 2;	/* middle of step */

      s.size = mulmod_size;
      tk = tuneup_measure (p->function, NULL, &s);
      if (tk == -1.0)
        abort ();

      s.size = mul_size;
      tm = tuneup_measure (p->mul_function, NULL, &s);
      if (tm == -1.0)
        abort ();

      if (option_trace >= 2)
        printf ("at %ld   size=%ld  %.9f   size=%ld mul %.9f\n",
                (long) size,
                (long) mulmod_size, tk,
                (long) mul_size, tm);

      size = mulmod_size;

      if (tk < tm)
        {
	  *p->p_threshold = s.size;
	  print_define (p->threshold_name, *p->p_threshold);
	  break;
        }
    }
}



/* Start karatsuba from 4, since the Cray t90 ieee code is much faster at 2,
   giving wrong results.  */
void
tune_mul_n (void)
{
  static struct param_t  param;
  mp_size_t next_toom_start;
  int something_changed;

  param.function = speed_mpn_mul_n;

  param.name = "MUL_TOOM22_THRESHOLD";
  param.min_size = MAX (4, MPN_TOOM22_MUL_MINSIZE);
  param.max_size = MUL_TOOM22_THRESHOLD_LIMIT-1;
  one (&mul_toom22_threshold, &param);

  param.noprint = 1;

  /* Threshold sequence loop.  Disable functions that would be used in a very
     narrow range, re-measuring things when that happens.  */
  something_changed = 1;
  while (something_changed)
    {
      something_changed = 0;

	next_toom_start = mul_toom22_threshold;

	if (mul_toom33_threshold != 0)
	  {
	    param.name = "MUL_TOOM33_THRESHOLD";
	    param.min_size = MAX (next_toom_start, MPN_TOOM33_MUL_MINSIZE);
	    param.max_size = MUL_TOOM33_THRESHOLD_LIMIT-1;
	    one (&mul_toom33_threshold, &param);

	    if (next_toom_start * 1.05 >= mul_toom33_threshold)
	      {
		mul_toom33_threshold = 0;
		something_changed = 1;
	      }
	  }

	next_toom_start = MAX (next_toom_start, mul_toom33_threshold);

	if (mul_toom44_threshold != 0)
	  {
	    param.name = "MUL_TOOM44_THRESHOLD";
	    param.min_size = MAX (next_toom_start, MPN_TOOM44_MUL_MINSIZE);
	    param.max_size = MUL_TOOM44_THRESHOLD_LIMIT-1;
	    one (&mul_toom44_threshold, &param);

	    if (next_toom_start * 1.05 >= mul_toom44_threshold)
	      {
		mul_toom44_threshold = 0;
		something_changed = 1;
	      }
	  }

	next_toom_start = MAX (next_toom_start, mul_toom44_threshold);

	if (mul_toom6h_threshold != 0)
	  {
	    param.name = "MUL_TOOM6H_THRESHOLD";
	    param.min_size = MAX (next_toom_start, MPN_TOOM6H_MUL_MINSIZE);
	    param.max_size = MUL_TOOM6H_THRESHOLD_LIMIT-1;
	    one (&mul_toom6h_threshold, &param);

	    if (next_toom_start * 1.05 >= mul_toom6h_threshold)
	      {
		mul_toom6h_threshold = 0;
		something_changed = 1;
	      }
	  }

	next_toom_start = MAX (next_toom_start, mul_toom6h_threshold);

	if (mul_toom8h_threshold != 0)
	  {
	    param.name = "MUL_TOOM8H_THRESHOLD";
	    param.min_size = MAX (next_toom_start, MPN_TOOM8H_MUL_MINSIZE);
	    param.max_size = MUL_TOOM8H_THRESHOLD_LIMIT-1;
	    one (&mul_toom8h_threshold, &param);

	    if (next_toom_start * 1.05 >= mul_toom8h_threshold)
	      {
		mul_toom8h_threshold = 0;
		something_changed = 1;
	      }
	  }
    }

    print_define ("MUL_TOOM33_THRESHOLD", MUL_TOOM33_THRESHOLD);
    print_define ("MUL_TOOM44_THRESHOLD", MUL_TOOM44_THRESHOLD);
    print_define ("MUL_TOOM6H_THRESHOLD", MUL_TOOM6H_THRESHOLD);
    print_define ("MUL_TOOM8H_THRESHOLD", MUL_TOOM8H_THRESHOLD);

  /* disabled until tuned */
  MUL_FFT_THRESHOLD = MP_SIZE_T_MAX;
}

void
tune_mul (void)
{
  static struct param_t  param;
  mp_size_t thres;

  param.noprint = 1;

  param.function = speed_mpn_toom32_for_toom43_mul;
  param.function2 = speed_mpn_toom43_for_toom32_mul;
  param.name = "MUL_TOOM32_TO_TOOM43_THRESHOLD";
  param.min_size = MPN_TOOM43_MUL_MINSIZE * 24 / 17;
  one (&thres, &param);
  mul_toom32_to_toom43_threshold = thres * 17 / 24;
  print_define ("MUL_TOOM32_TO_TOOM43_THRESHOLD", mul_toom32_to_toom43_threshold);

  param.function = speed_mpn_toom32_for_toom53_mul;
  param.function2 = speed_mpn_toom53_for_toom32_mul;
  param.name = "MUL_TOOM32_TO_TOOM53_THRESHOLD";
  param.min_size = MPN_TOOM53_MUL_MINSIZE * 30 / 19;
  one (&thres, &param);
  mul_toom32_to_toom53_threshold = thres * 19 / 30;
  print_define ("MUL_TOOM32_TO_TOOM53_THRESHOLD", mul_toom32_to_toom53_threshold);

  param.function = speed_mpn_toom42_for_toom53_mul;
  param.function2 = speed_mpn_toom53_for_toom42_mul;
  param.name = "MUL_TOOM42_TO_TOOM53_THRESHOLD";
  param.min_size = MPN_TOOM53_MUL_MINSIZE * 20 / 11;
  one (&thres, &param);
  mul_toom42_to_toom53_threshold = thres * 11 / 20;
  print_define ("MUL_TOOM42_TO_TOOM53_THRESHOLD", mul_toom42_to_toom53_threshold);

  param.function = speed_mpn_toom42_mul;
  param.function2 = speed_mpn_toom63_mul;
  param.name = "MUL_TOOM42_TO_TOOM63_THRESHOLD";
  param.min_size = MPN_TOOM63_MUL_MINSIZE * 2;
  one (&thres, &param);
  mul_toom42_to_toom63_threshold = thres / 2;
  print_define ("MUL_TOOM42_TO_TOOM63_THRESHOLD", mul_toom42_to_toom63_threshold);

  /* Use ratio 5/6 when measuring, the middle of the range 2/3 to 1. */
  param.function = speed_mpn_toom43_for_toom54_mul;
  param.function2 = speed_mpn_toom54_for_toom43_mul;
  param.name = "MUL_TOOM43_TO_TOOM54_THRESHOLD";
  param.min_size = MPN_TOOM54_MUL_MINSIZE * 6 / 5;
  one (&thres, &param);
  mul_toom43_to_toom54_threshold = thres * 5 / 6;
  print_define ("MUL_TOOM43_TO_TOOM54_THRESHOLD", mul_toom43_to_toom54_threshold);
}


void
tune_mullo (void)
{
  static struct param_t  param;

  param.function = speed_mpn_mullo_n;

  param.name = "MULLO_BASECASE_THRESHOLD";
  param.min_size = 1;
  param.min_is_always = 1;
  param.max_size = MULLO_BASECASE_THRESHOLD_LIMIT-1;
  param.stop_factor = 1.5;
  param.noprint = 1;
  one (&mullo_basecase_threshold, &param);

  param.name = "MULLO_DC_THRESHOLD";
  param.min_size = 8;
  param.min_is_always = 0;
  param.max_size = 1000;
  one (&mullo_dc_threshold, &param);

  if (mullo_basecase_threshold >= mullo_dc_threshold)
    {
      print_define ("MULLO_BASECASE_THRESHOLD", mullo_dc_threshold);
      print_define_remark ("MULLO_DC_THRESHOLD", 0, "never mpn_mullo_basecase");
    }
  else
    {
      print_define ("MULLO_BASECASE_THRESHOLD", mullo_basecase_threshold);
      print_define ("MULLO_DC_THRESHOLD", mullo_dc_threshold);
    }

  if (WANT_FFT && mul_fft_threshold < MP_SIZE_T_MAX / 2)
    {
      param.name = "MULLO_MUL_N_THRESHOLD";
      param.min_size = mullo_dc_threshold;
      param.max_size = 2 * mul_fft_threshold;
      param.noprint = 0;
      param.step_factor = 0.03;
      one (&mullo_mul_n_threshold, &param);
    }
  else
    print_define_remark ("MULLO_MUL_N_THRESHOLD", MP_SIZE_T_MAX,
			 "without FFT use mullo forever");
}

void
tune_sqrlo (void)
{
  static struct param_t  param;

  param.function = speed_mpn_sqrlo;

  param.name = "SQRLO_BASECASE_THRESHOLD";
  param.min_size = 1;
  param.min_is_always = 1;
  param.max_size = SQRLO_BASECASE_THRESHOLD_LIMIT-1;
  param.stop_factor = 1.5;
  param.noprint = 1;
  one (&sqrlo_basecase_threshold, &param);

  param.name = "SQRLO_DC_THRESHOLD";
  param.min_size = 8;
  param.min_is_always = 0;
  param.max_size = SQRLO_DC_THRESHOLD_LIMIT-1;
  one (&sqrlo_dc_threshold, &param);

  if (sqrlo_basecase_threshold >= sqrlo_dc_threshold)
    {
      print_define ("SQRLO_BASECASE_THRESHOLD", sqrlo_dc_threshold);
      print_define_remark ("SQRLO_DC_THRESHOLD", 0, "never mpn_sqrlo_basecase");
    }
  else
    {
      print_define ("SQRLO_BASECASE_THRESHOLD", sqrlo_basecase_threshold);
      print_define ("SQRLO_DC_THRESHOLD", sqrlo_dc_threshold);
    }

  if (WANT_FFT && sqr_fft_threshold < MP_SIZE_T_MAX / 2)
    {
      param.name = "SQRLO_SQR_THRESHOLD";
      param.min_size = sqrlo_dc_threshold;
      param.max_size = 2 * sqr_fft_threshold;
      param.noprint = 0;
      param.step_factor = 0.03;
      one (&sqrlo_sqr_threshold, &param);
    }
  else
    print_define_remark ("SQRLO_SQR_THRESHOLD", MP_SIZE_T_MAX,
			 "without FFT use sqrlo forever");
}

void
tune_mulmid (void)
{
  static struct param_t  param;

  param.name = "MULMID_TOOM42_THRESHOLD";
  param.function = speed_mpn_mulmid_n;
  param.min_size = 4;
  param.max_size = 100;
  one (&mulmid_toom42_threshold, &param);
}

void
tune_mulmod_bnm1 (void)
{
  static struct param_t  param;

  param.name = "MULMOD_BNM1_THRESHOLD";
  param.function = speed_mpn_mulmod_bnm1;
  param.min_size = 4;
  param.max_size = 100;
  one (&mulmod_bnm1_threshold, &param);
}

void
tune_sqrmod_bnm1 (void)
{
  static struct param_t  param;

  param.name = "SQRMOD_BNM1_THRESHOLD";
  param.function = speed_mpn_sqrmod_bnm1;
  param.min_size = 4;
  param.max_size = 100;
  one (&sqrmod_bnm1_threshold, &param);
}


/* Start the basecase from 3, since 1 is a special case, and if mul_basecase
   is faster only at size==2 then we don't want to bother with extra code
   just for that.  Start karatsuba from 4 same as MUL above.  */

void
tune_sqr (void)
{
  /* disabled until tuned */
  SQR_FFT_THRESHOLD = MP_SIZE_T_MAX;

  if (HAVE_NATIVE_mpn_sqr_basecase)
    {
      print_define_remark ("SQR_BASECASE_THRESHOLD", 0, "always (native)");
      sqr_basecase_threshold = 0;
    }
  else
    {
      static struct param_t  param;
      param.name = "SQR_BASECASE_THRESHOLD";
      param.function = speed_mpn_sqr;
      param.min_size = 3;
      param.min_is_always = 1;
      param.max_size = TUNE_SQR_TOOM2_MAX;
      param.noprint = 1;
      one (&sqr_basecase_threshold, &param);
    }

  {
    static struct param_t  param;
    param.name = "SQR_TOOM2_THRESHOLD";
    param.function = speed_mpn_sqr;
    param.min_size = MAX (4, MPN_TOOM2_SQR_MINSIZE);
    param.max_size = TUNE_SQR_TOOM2_MAX;
    param.noprint = 1;
    one (&sqr_toom2_threshold, &param);

    if (! HAVE_NATIVE_mpn_sqr_basecase
        && sqr_toom2_threshold < sqr_basecase_threshold)
      {
        /* Karatsuba becomes faster than mul_basecase before
           sqr_basecase does.  Arrange for the expression
           "BELOW_THRESHOLD (un, SQR_TOOM2_THRESHOLD))" which
           selects mpn_sqr_basecase in mpn_sqr to be false, by setting
           SQR_TOOM2_THRESHOLD to zero, making
           SQR_BASECASE_THRESHOLD the toom2 threshold.  */

        sqr_basecase_threshold = SQR_TOOM2_THRESHOLD;
        SQR_TOOM2_THRESHOLD = 0;

        print_define_remark ("SQR_BASECASE_THRESHOLD", sqr_basecase_threshold,
                             "toom2");
        print_define_remark ("SQR_TOOM2_THRESHOLD",SQR_TOOM2_THRESHOLD,
                             "never sqr_basecase");
      }
    else
      {
        if (! HAVE_NATIVE_mpn_sqr_basecase)
          print_define ("SQR_BASECASE_THRESHOLD", sqr_basecase_threshold);
        print_define ("SQR_TOOM2_THRESHOLD", SQR_TOOM2_THRESHOLD);
      }
  }

  {
    static struct param_t  param;
    mp_size_t next_toom_start;
    int something_changed;

    param.function = speed_mpn_sqr;
    param.noprint = 1;

  /* Threshold sequence loop.  Disable functions that would be used in a very
     narrow range, re-measuring things when that happens.  */
    something_changed = 1;
    while (something_changed)
      {
	something_changed = 0;

	next_toom_start = MAX (sqr_toom2_threshold, sqr_basecase_threshold);

	sqr_toom3_threshold = SQR_TOOM3_THRESHOLD_LIMIT;
	param.name = "SQR_TOOM3_THRESHOLD";
	param.min_size = MAX (next_toom_start, MPN_TOOM3_SQR_MINSIZE);
	param.max_size = SQR_TOOM3_THRESHOLD_LIMIT-1;
	one (&sqr_toom3_threshold, &param);

	next_toom_start = MAX (next_toom_start, sqr_toom3_threshold);

	if (sqr_toom4_threshold != 0)
	  {
	    param.name = "SQR_TOOM4_THRESHOLD";
	    sqr_toom4_threshold = SQR_TOOM4_THRESHOLD_LIMIT;
	    param.min_size = MAX (next_toom_start, MPN_TOOM4_SQR_MINSIZE);
	    param.max_size = SQR_TOOM4_THRESHOLD_LIMIT-1;
	    one (&sqr_toom4_threshold, &param);

	    if (next_toom_start * 1.05 >= sqr_toom4_threshold)
	      {
		sqr_toom4_threshold = 0;
		something_changed = 1;
	      }
	  }

	next_toom_start = MAX (next_toom_start, sqr_toom4_threshold);

	if (sqr_toom6_threshold != 0)
	  {
	    param.name = "SQR_TOOM6_THRESHOLD";
	    sqr_toom6_threshold = SQR_TOOM6_THRESHOLD_LIMIT;
	    param.min_size = MAX (next_toom_start, MPN_TOOM6_SQR_MINSIZE);
	    param.max_size = SQR_TOOM6_THRESHOLD_LIMIT-1;
	    one (&sqr_toom6_threshold, &param);

	    if (next_toom_start * 1.05 >= sqr_toom6_threshold)
	      {
		sqr_toom6_threshold = 0;
		something_changed = 1;
	      }
	  }

	next_toom_start = MAX (next_toom_start, sqr_toom6_threshold);

	if (sqr_toom8_threshold != 0)
	  {
	    param.name = "SQR_TOOM8_THRESHOLD";
	    sqr_toom8_threshold = SQR_TOOM8_THRESHOLD_LIMIT;
	    param.min_size = MAX (next_toom_start, MPN_TOOM8_SQR_MINSIZE);
	    param.max_size = SQR_TOOM8_THRESHOLD_LIMIT-1;
	    one (&sqr_toom8_threshold, &param);

	    if (next_toom_start * 1.05 >= sqr_toom8_threshold)
	      {
		sqr_toom8_threshold = 0;
		something_changed = 1;
	      }
	  }
      }

    print_define ("SQR_TOOM3_THRESHOLD", SQR_TOOM3_THRESHOLD);
    print_define ("SQR_TOOM4_THRESHOLD", SQR_TOOM4_THRESHOLD);
    print_define ("SQR_TOOM6_THRESHOLD", SQR_TOOM6_THRESHOLD);
    print_define ("SQR_TOOM8_THRESHOLD", SQR_TOOM8_THRESHOLD);
  }
}


void
tune_dc_div (void)
{
  s.r = 0;		/* clear to make speed function do 2n/n */
  {
    static struct param_t  param;
    param.name = "DC_DIV_QR_THRESHOLD";
    param.function = speed_mpn_sbpi1_div_qr;
    param.function2 = speed_mpn_dcpi1_div_qr;
    param.min_size = 6;
    one (&dc_div_qr_threshold, &param);
  }
  {
    static struct param_t  param;
    param.name = "DC_DIVAPPR_Q_THRESHOLD";
    param.function = speed_mpn_sbpi1_divappr_q;
    param.function2 = speed_mpn_dcpi1_divappr_q;
    param.min_size = 6;
    one (&dc_divappr_q_threshold, &param);
  }
}

static double
speed_mpn_sbordcpi1_div_qr (struct speed_params *s)
{
  if (s->size < DC_DIV_QR_THRESHOLD)
    return speed_mpn_sbpi1_div_qr (s);
  else
    return speed_mpn_dcpi1_div_qr (s);
}

void
tune_mu_div (void)
{
  s.r = 0;		/* clear to make speed function do 2n/n */
  {
    static struct param_t  param;
    param.name = "MU_DIV_QR_THRESHOLD";
    param.function = speed_mpn_dcpi1_div_qr;
    param.function2 = speed_mpn_mu_div_qr;
    param.min_size = mul_toom22_threshold;
    param.max_size = 5000;
    param.step_factor = 0.02;
    one (&mu_div_qr_threshold, &param);
  }
  {
    static struct param_t  param;
    param.name = "MU_DIVAPPR_Q_THRESHOLD";
    param.function = speed_mpn_dcpi1_divappr_q;
    param.function2 = speed_mpn_mu_divappr_q;
    param.min_size = mul_toom22_threshold;
    param.max_size = 5000;
    param.step_factor = 0.02;
    one (&mu_divappr_q_threshold, &param);
  }
  {
    static struct param_t  param;
    param.name = "MUPI_DIV_QR_THRESHOLD";
    param.function = speed_mpn_sbordcpi1_div_qr;
    param.function2 = speed_mpn_mupi_div_qr;
    param.min_size = 6;
    param.min_is_always = 1;
    param.max_size = 1000;
    param.step_factor = 0.02;
    one (&mupi_div_qr_threshold, &param);
  }
}

void
tune_dc_bdiv (void)
{
  s.r = 0;		/* clear to make speed function do 2n/n*/
  {
    static struct param_t  param;
    param.name = "DC_BDIV_QR_THRESHOLD";
    param.function = speed_mpn_sbpi1_bdiv_qr;
    param.function2 = speed_mpn_dcpi1_bdiv_qr;
    param.min_size = 4;
    one (&dc_bdiv_qr_threshold, &param);
  }
  {
    static struct param_t  param;
    param.name = "DC_BDIV_Q_THRESHOLD";
    param.function = speed_mpn_sbpi1_bdiv_q;
    param.function2 = speed_mpn_dcpi1_bdiv_q;
    param.min_size = 4;
    one (&dc_bdiv_q_threshold, &param);
  }
}

void
tune_mu_bdiv (void)
{
  s.r = 0;		/* clear to make speed function do 2n/n*/
  {
    static struct param_t  param;
    param.name = "MU_BDIV_QR_THRESHOLD";
    param.function = speed_mpn_dcpi1_bdiv_qr;
    param.function2 = speed_mpn_mu_bdiv_qr;
    param.min_size = dc_bdiv_qr_threshold;
    param.max_size = 5000;
    param.step_factor = 0.02;
    one (&mu_bdiv_qr_threshold, &param);
  }
  {
    static struct param_t  param;
    param.name = "MU_BDIV_Q_THRESHOLD";
    param.function = speed_mpn_dcpi1_bdiv_q;
    param.function2 = speed_mpn_mu_bdiv_q;
    param.min_size = dc_bdiv_q_threshold;
    param.max_size = 5000;
    param.step_factor = 0.02;
    one (&mu_bdiv_q_threshold, &param);
  }
}

void
tune_invertappr (void)
{
  static struct param_t  param;

  param.function = speed_mpn_ni_invertappr;
  param.name = "INV_MULMOD_BNM1_THRESHOLD";
  param.min_size = 5;
  one (&inv_mulmod_bnm1_threshold, &param);

  param.function = speed_mpn_invertappr;
  param.name = "INV_NEWTON_THRESHOLD";
  param.min_size = 5;
  one (&inv_newton_threshold, &param);
}

void
tune_invert (void)
{
  static struct param_t  param;

  param.function = speed_mpn_invert;
  param.name = "INV_APPR_THRESHOLD";
  param.min_size = 5;
  one (&inv_appr_threshold, &param);
}

void
tune_binvert (void)
{
  static struct param_t  param;

  param.function = speed_mpn_binvert;
  param.name = "BINV_NEWTON_THRESHOLD";
  param.min_size = 8;		/* pointless with smaller operands */
  one (&binv_newton_threshold, &param);
}

void
tune_redc (void)
{
#define TUNE_REDC_2_MAX 100
#if HAVE_NATIVE_mpn_addmul_2 || HAVE_NATIVE_mpn_redc_2
#define WANT_REDC_2 1
#endif

#if WANT_REDC_2
  {
    static struct param_t  param;
    param.name = "REDC_1_TO_REDC_2_THRESHOLD";
    param.function = speed_mpn_redc_1;
    param.function2 = speed_mpn_redc_2;
    param.min_size = 1;
    param.min_is_always = 1;
    param.max_size = TUNE_REDC_2_MAX;
    param.noprint = 1;
    param.stop_factor = 1.5;
    one (&redc_1_to_redc_2_threshold, &param);
  }
  {
    static struct param_t  param;
    param.name = "REDC_2_TO_REDC_N_THRESHOLD";
    param.function = speed_mpn_redc_2;
    param.function2 = speed_mpn_redc_n;
    param.min_size = 16;
    param.noprint = 1;
    one (&redc_2_to_redc_n_threshold, &param);
  }
  if (redc_1_to_redc_2_threshold >= redc_2_to_redc_n_threshold)
    {
      redc_2_to_redc_n_threshold = 0;	/* disable redc_2 */

      /* Never use redc2, measure redc_1 -> redc_n cutoff, store result as
	 REDC_1_TO_REDC_2_THRESHOLD.  */
      {
	static struct param_t  param;
	param.name = "REDC_1_TO_REDC_2_THRESHOLD";
	param.function = speed_mpn_redc_1;
	param.function2 = speed_mpn_redc_n;
	param.min_size = 16;
	param.noprint = 1;
	one (&redc_1_to_redc_2_threshold, &param);
      }
    }
  print_define ("REDC_1_TO_REDC_2_THRESHOLD", REDC_1_TO_REDC_2_THRESHOLD);
  print_define ("REDC_2_TO_REDC_N_THRESHOLD", REDC_2_TO_REDC_N_THRESHOLD);
#else
  {
    static struct param_t  param;
    param.name = "REDC_1_TO_REDC_N_THRESHOLD";
    param.function = speed_mpn_redc_1;
    param.function2 = speed_mpn_redc_n;
    param.min_size = 16;
    one (&redc_1_to_redc_n_threshold, &param);
  }
#endif
}

void
tune_matrix22_mul (void)
{
  static struct param_t  param;
  param.name = "MATRIX22_STRASSEN_THRESHOLD";
  param.function = speed_mpn_matrix22_mul;
  param.min_size = 2;
  one (&matrix22_strassen_threshold, &param);
}

void
tune_hgcd (void)
{
  static struct param_t  param;
  param.name = "HGCD_THRESHOLD";
  param.function = speed_mpn_hgcd;
  /* We seem to get strange results for small sizes */
  param.min_size = 30;
  one (&hgcd_threshold, &param);
}

void
tune_hgcd_appr (void)
{
  static struct param_t  param;
  param.name = "HGCD_APPR_THRESHOLD";
  param.function = speed_mpn_hgcd_appr;
  /* We seem to get strange results for small sizes */
  param.min_size = 50;
  param.stop_since_change = 150;
  one (&hgcd_appr_threshold, &param);
}

void
tune_hgcd_reduce (void)
{
  static struct param_t  param;
  param.name = "HGCD_REDUCE_THRESHOLD";
  param.function = speed_mpn_hgcd_reduce;
  param.min_size = 30;
  param.max_size = 7000;
  param.step_factor = 0.04;
  one (&hgcd_reduce_threshold, &param);
}

void
tune_gcd_dc (void)
{
  static struct param_t  param;
  param.name = "GCD_DC_THRESHOLD";
  param.function = speed_mpn_gcd;
  param.min_size = hgcd_threshold;
  param.max_size = 3000;
  param.step_factor = 0.02;
  one (&gcd_dc_threshold, &param);
}

void
tune_gcdext_dc (void)
{
  static struct param_t  param;
  param.name = "GCDEXT_DC_THRESHOLD";
  param.function = speed_mpn_gcdext;
  param.min_size = hgcd_threshold;
  param.max_size = 3000;
  param.step_factor = 0.02;
  one (&gcdext_dc_threshold, &param);
}

/* In tune_powm_sec we compute the table used by the win_size function.  The
   cutoff points are in exponent bits, disregarding other operand sizes.  It is
   not possible to use the one framework since it currently uses a granularity
   of full limbs.
*/

/* This win_size replaces the variant in the powm code, allowing us to
   control k in the k-ary algorithms.  */
int winsize;
int
win_size (mp_bitcnt_t eb)
{
  return winsize;
}

void
tune_powm_sec (void)
{
  mp_size_t n;
  int k, i;
  mp_size_t itch;
  mp_bitcnt_t nbits, nbits_next, possible_nbits_cutoff;
  const int n_max = 3000 / GMP_NUMB_BITS;
  const int n_measurements = 5;
  mp_ptr rp, bp, ep, mp, tp;
  double ttab[n_measurements], tk, tkp1;
  TMP_DECL;
  TMP_MARK;

  possible_nbits_cutoff = 0;

  k = 1;

  winsize = 10;			/* the itch function needs this */
  itch = mpn_sec_powm_itch (n_max, n_max * GMP_NUMB_BITS, n_max);

  rp = TMP_ALLOC_LIMBS (n_max);
  bp = TMP_ALLOC_LIMBS (n_max);
  ep = TMP_ALLOC_LIMBS (n_max);
  mp = TMP_ALLOC_LIMBS (n_max);
  tp = TMP_ALLOC_LIMBS (itch);

  mpn_random (bp, n_max);
  mpn_random (mp, n_max);
  mp[0] |= 1;

/* How about taking the M operand size into account?

   An operation R=powm(B,E,N) will take time O(log(E)*M(log(N))) (assuming
   B = O(M)).

   Using k-ary and no sliding window, the precomputation will need time
   O(2^(k-1)*M(log(N))) and the main computation will need O(log(E)*S(N)) +
   O(log(E)/k*M(N)), for the squarings, multiplications, respectively.

   An operation R=powm_sec(B,E,N) will take time like powm.

   Using k-ary, the precomputation will need time O(2^k*M(log(N))) and the
   main computation will need O(log(E)*S(N)) + O(log(E)/k*M(N)) +
   O(log(E)/k*2^k*log(N)), for the squarings, multiplications, and full
   table reads, respectively.  */

  printf ("#define POWM_SEC_TABLE  ");

  /* For nbits == 1, we should always use k == 1, so no need to tune
     that. Starting with nbits == 2 also ensure that nbits always is
     larger than the windowsize k+1. */
  for (nbits = 2; nbits <= n_max * GMP_NUMB_BITS; )
    {
      n = (nbits - 1) / GMP_NUMB_BITS + 1;

      /* Generate E such that sliding-window for k and k+1 works equally
	 well/poorly (but sliding is not used in powm_sec, of course). */
      for (i = 0; i < n; i++)
	ep[i] = ~CNST_LIMB(0);

      winsize = k;
      for (i = 0; i < n_measurements; i++)
	{
	  speed_starttime ();
	  mpn_sec_powm (rp, bp, n, ep, nbits, mp, n, tp);
	  ttab[i] = speed_endtime ();
	}
      tk = median (ttab, n_measurements);

      winsize = k + 1;
      speed_starttime ();
      for (i = 0; i < n_measurements; i++)
	{
	  speed_starttime ();
	  mpn_sec_powm (rp, bp, n, ep, nbits, mp, n, tp);
	  ttab[i] = speed_endtime ();
	}
      tkp1 = median (ttab, n_measurements);
/*
      printf ("testing: %ld, %d", nbits, k, ep[n-1]);
      printf ("   %10.5f  %10.5f\n", tk, tkp1);
*/
      if (tkp1 < tk)
	{
	  if (possible_nbits_cutoff)
	    {
	      /* Two consecutive sizes indicate k increase, obey.  */

	      /* Must always have x[k] >= k */
	      ASSERT_ALWAYS (possible_nbits_cutoff >= k);

	      if (k > 1)
		printf (",");
	      printf ("%ld", (long) possible_nbits_cutoff);
	      k++;
	      possible_nbits_cutoff = 0;
	    }
	  else
	    {
	      /* One measurement indicate k increase, save nbits for further
		 consideration.  */
	      /* The new larger k gets used for sizes > the cutoff
		 value, hence the cutoff should be one less than the
		 smallest size where it gives a speedup. */
	      possible_nbits_cutoff = nbits - 1;
	    }
	}
      else
	possible_nbits_cutoff = 0;

      nbits_next = nbits * 65 / 64;
      nbits = nbits_next + (nbits_next == nbits);
    }
  printf ("\n");
  TMP_FREE;
}


/* size_extra==1 reflects the fact that with high<divisor one division is
   always skipped.  Forcing high<divisor while testing ensures consistency
   while stepping through sizes, ie. that size-1 divides will be done each
   time.

   min_size==2 and min_is_always are used so that if plain division is only
   better at size==1 then don't bother including that code just for that
   case, instead go with preinv always and get a size saving.  */

#define DIV_1_PARAMS                    \
  param.check_size = 256;               \
  param.min_size = 2;                   \
  param.min_is_always = 1;              \
  param.data_high = DATA_HIGH_LT_R;     \
  param.size_extra = 1;                 \
  param.stop_factor = 2.0;


double (*tuned_speed_mpn_divrem_1) (struct speed_params *);

void
tune_divrem_1 (void)
{
  /* plain version by default */
  tuned_speed_mpn_divrem_1 = speed_mpn_divrem_1;

  /* No support for tuning native assembler code, do that by hand and put
     the results in the .asm file, there's no need for such thresholds to
     appear in gmp-mparam.h.  */
  if (HAVE_NATIVE_mpn_divrem_1)
    return;

  if (GMP_NAIL_BITS != 0)
    {
      print_define_remark ("DIVREM_1_NORM_THRESHOLD", MP_SIZE_T_MAX,
                           "no preinv with nails");
      print_define_remark ("DIVREM_1_UNNORM_THRESHOLD", MP_SIZE_T_MAX,
                           "no preinv with nails");
      return;
    }

  if (UDIV_PREINV_ALWAYS)
    {
      print_define_remark ("DIVREM_1_NORM_THRESHOLD", 0L, "preinv always");
      print_define ("DIVREM_1_UNNORM_THRESHOLD", 0L);
      return;
    }

  tuned_speed_mpn_divrem_1 = speed_mpn_divrem_1_tune;

  /* Tune for the integer part of mpn_divrem_1.  This will very possibly be
     a bit out for the fractional part, but that's too bad, the integer part
     is more important. */
  {
    static struct param_t  param;
    param.name = "DIVREM_1_NORM_THRESHOLD";
    DIV_1_PARAMS;
    s.r = randlimb_norm ();
    param.function = speed_mpn_divrem_1_tune;
    one (&divrem_1_norm_threshold, &param);
  }
  {
    static struct param_t  param;
    param.name = "DIVREM_1_UNNORM_THRESHOLD";
    DIV_1_PARAMS;
    s.r = randlimb_half ();
    param.function = speed_mpn_divrem_1_tune;
    one (&divrem_1_unnorm_threshold, &param);
  }
}

void
tune_div_qr_1 (void)
{
  static struct param_t  param;
  double            t1, t2;

  if (!HAVE_NATIVE_mpn_div_qr_1n_pi1)
    {
      static struct param_t  param;
      double   t1, t2;

      s.size = 10;
      s.r = randlimb_norm ();

      t1 = tuneup_measure (speed_mpn_div_qr_1n_pi1_1, &param, &s);
      t2 = tuneup_measure (speed_mpn_div_qr_1n_pi1_2, &param, &s);

      if (t1 == -1.0 || t2 == -1.0)
	{
	  printf ("Oops, can't measure all mpn_div_qr_1n_pi1 methods at %ld\n",
		  (long) s.size);
	  abort ();
	}
      div_qr_1n_pi1_method = (t1 < t2) ? 1 : 2;
      print_define ("DIV_QR_1N_PI1_METHOD", div_qr_1n_pi1_method);
    }

  {
    static struct param_t  param;
    param.name = "DIV_QR_1_NORM_THRESHOLD";
    DIV_1_PARAMS;
    param.min_size = 1;
    param.min_is_always = 0;
    s.r = randlimb_norm ();
    param.function = speed_mpn_div_qr_1_tune;
    one (&div_qr_1_norm_threshold, &param);
  }
  {
    static struct param_t  param;
    param.name = "DIV_QR_1_UNNORM_THRESHOLD";
    DIV_1_PARAMS;
    param.min_size = 1;
    param.min_is_always = 0;
    s.r = randlimb_half();
    param.function = speed_mpn_div_qr_1_tune;
    one (&div_qr_1_unnorm_threshold, &param);
  }
}


void
tune_mod_1 (void)
{
  /* No support for tuning native assembler code, do that by hand and put
     the results in the .asm file, there's no need for such thresholds to
     appear in gmp-mparam.h.  */
  if (HAVE_NATIVE_mpn_mod_1)
    return;

  if (GMP_NAIL_BITS != 0)
    {
      print_define_remark ("MOD_1_NORM_THRESHOLD", MP_SIZE_T_MAX,
                           "no preinv with nails");
      print_define_remark ("MOD_1_UNNORM_THRESHOLD", MP_SIZE_T_MAX,
                           "no preinv with nails");
      return;
    }

  if (!HAVE_NATIVE_mpn_mod_1_1p)
    {
      static struct param_t  param;
      double   t1, t2;

      s.size = 10;
      s.r = randlimb_half ();

      t1 = tuneup_measure (speed_mpn_mod_1_1_1, &param, &s);
      t2 = tuneup_measure (speed_mpn_mod_1_1_2, &param, &s);

      if (t1 == -1.0 || t2 == -1.0)
	{
	  printf ("Oops, can't measure all mpn_mod_1_1 methods at %ld\n",
		  (long) s.size);
	  abort ();
	}
      mod_1_1p_method = (t1 < t2) ? 1 : 2;
      print_define ("MOD_1_1P_METHOD", mod_1_1p_method);
    }

  if (UDIV_PREINV_ALWAYS)
    {
      print_define ("MOD_1_NORM_THRESHOLD", 0L);
      print_define ("MOD_1_UNNORM_THRESHOLD", 0L);
    }
  else
    {
      {
	static struct param_t  param;
	param.name = "MOD_1_NORM_THRESHOLD";
	DIV_1_PARAMS;
	s.r = randlimb_norm ();
	param.function = speed_mpn_mod_1_tune;
	one (&mod_1_norm_threshold, &param);
      }
      {
	static struct param_t  param;
	param.name = "MOD_1_UNNORM_THRESHOLD";
	DIV_1_PARAMS;
	s.r = randlimb_half ();
	param.function = speed_mpn_mod_1_tune;
	one (&mod_1_unnorm_threshold, &param);
      }
    }
  {
    static struct param_t  param;

    param.check_size = 256;

    s.r = randlimb_norm ();
    param.function = speed_mpn_mod_1_tune;

    param.name = "MOD_1N_TO_MOD_1_1_THRESHOLD";
    param.min_size = 2;
    one (&mod_1n_to_mod_1_1_threshold, &param);
  }

  {
    static struct param_t  param;

    param.check_size = 256;
    s.r = randlimb_half ();
    param.noprint = 1;

    param.function = speed_mpn_mod_1_1;
    param.function2 = speed_mpn_mod_1_2;
    param.min_is_always = 1;
    param.name = "MOD_1_1_TO_MOD_1_2_THRESHOLD";
    param.min_size = 2;
    one (&mod_1_1_to_mod_1_2_threshold, &param);

    param.function = speed_mpn_mod_1_2;
    param.function2 = speed_mpn_mod_1_4;
    param.min_is_always = 1;
    param.name = "MOD_1_2_TO_MOD_1_4_THRESHOLD";
    param.min_size = 1;
    one (&mod_1_2_to_mod_1_4_threshold, &param);

    if (mod_1_1_to_mod_1_2_threshold >= mod_1_2_to_mod_1_4_threshold)
      {
	/* Never use mod_1_2, measure mod_1_1 -> mod_1_4 */
	mod_1_2_to_mod_1_4_threshold = 0;

	param.function = speed_mpn_mod_1_1;
	param.function2 = speed_mpn_mod_1_4;
	param.min_is_always = 1;
	param.name = "MOD_1_1_TO_MOD_1_4_THRESHOLD fake";
	param.min_size = 2;
	one (&mod_1_1_to_mod_1_2_threshold, &param);
      }

    param.function = speed_mpn_mod_1_tune;
    param.function2 = NULL;
    param.name = "MOD_1U_TO_MOD_1_1_THRESHOLD";
    param.min_size = 2;
    param.min_is_always = 0;
    one (&mod_1u_to_mod_1_1_threshold, &param);

    if (mod_1u_to_mod_1_1_threshold >= mod_1_1_to_mod_1_2_threshold)
      mod_1_1_to_mod_1_2_threshold = 0;
    if (mod_1u_to_mod_1_1_threshold >= mod_1_2_to_mod_1_4_threshold)
      mod_1_2_to_mod_1_4_threshold = 0;

    print_define_remark ("MOD_1U_TO_MOD_1_1_THRESHOLD", mod_1u_to_mod_1_1_threshold, NULL);
    print_define_remark ("MOD_1_1_TO_MOD_1_2_THRESHOLD", mod_1_1_to_mod_1_2_threshold,
			 mod_1_1_to_mod_1_2_threshold == 0 ? "never mpn_mod_1_1p" : NULL);
    print_define_remark ("MOD_1_2_TO_MOD_1_4_THRESHOLD", mod_1_2_to_mod_1_4_threshold,
			 mod_1_2_to_mod_1_4_threshold == 0 ? "never mpn_mod_1s_2p" : NULL);
  }

  {
    static struct param_t  param;

    param.check_size = 256;

    param.name = "PREINV_MOD_1_TO_MOD_1_THRESHOLD";
    s.r = randlimb_norm ();
    param.function = speed_mpn_preinv_mod_1;
    param.function2 = speed_mpn_mod_1_tune;
    param.min_size = 1;
    one (&preinv_mod_1_to_mod_1_threshold, &param);
  }
}


/* A non-zero DIVREM_1_UNNORM_THRESHOLD (or DIVREM_1_NORM_THRESHOLD) would
   imply that udiv_qrnnd_preinv is worth using, but it seems most
   straightforward to compare mpn_preinv_divrem_1 and mpn_divrem_1_div
   directly.  */

void
tune_preinv_divrem_1 (void)
{
  static struct param_t  param;
  speed_function_t  divrem_1;
  const char        *divrem_1_name;
  double            t1, t2;

  if (GMP_NAIL_BITS != 0)
    {
      print_define_remark ("USE_PREINV_DIVREM_1", 0, "no preinv with nails");
      return;
    }

  /* Any native version of mpn_preinv_divrem_1 is assumed to exist because
     it's faster than mpn_divrem_1.  */
  if (HAVE_NATIVE_mpn_preinv_divrem_1)
    {
      print_define_remark ("USE_PREINV_DIVREM_1", 1, "native");
      return;
    }

  /* If udiv_qrnnd_preinv is the only division method then of course
     mpn_preinv_divrem_1 should be used.  */
  if (UDIV_PREINV_ALWAYS)
    {
      print_define_remark ("USE_PREINV_DIVREM_1", 1, "preinv always");
      return;
    }

  /* If we've got an assembler version of mpn_divrem_1, then compare against
     that, not the mpn_divrem_1_div generic C.  */
  if (HAVE_NATIVE_mpn_divrem_1)
    {
      divrem_1 = speed_mpn_divrem_1;
      divrem_1_name = "mpn_divrem_1";
    }
  else
    {
      divrem_1 = speed_mpn_divrem_1_div;
      divrem_1_name = "mpn_divrem_1_div";
    }

  param.data_high = DATA_HIGH_LT_R; /* allow skip one division */
  s.size = 200;                     /* generous but not too big */
  /* Divisor, nonzero.  Unnormalized so as to exercise the shift!=0 case,
     since in general that's probably most common, though in fact for a
     64-bit limb mp_bases[10].big_base is normalized.  */
  s.r = urandom() & (GMP_NUMB_MASK >> 4);
  if (s.r == 0) s.r = 123;

  t1 = tuneup_measure (speed_mpn_preinv_divrem_1, &param, &s);
  t2 = tuneup_measure (divrem_1, &param, &s);
  if (t1 == -1.0 || t2 == -1.0)
    {
      printf ("Oops, can't measure mpn_preinv_divrem_1 and %s at %ld\n",
              divrem_1_name, (long) s.size);
      abort ();
    }
  if (option_trace >= 1)
    printf ("size=%ld, mpn_preinv_divrem_1 %.9f, %s %.9f\n",
            (long) s.size, t1, divrem_1_name, t2);

  print_define_remark ("USE_PREINV_DIVREM_1", (mp_size_t) (t1 < t2), NULL);
}



void
tune_divrem_2 (void)
{
  static struct param_t  param;

  /* No support for tuning native assembler code, do that by hand and put
     the results in the .asm file, and there's no need for such thresholds
     to appear in gmp-mparam.h.  */
  if (HAVE_NATIVE_mpn_divrem_2)
    return;

  if (GMP_NAIL_BITS != 0)
    {
      print_define_remark ("DIVREM_2_THRESHOLD", MP_SIZE_T_MAX,
                           "no preinv with nails");
      return;
    }

  if (UDIV_PREINV_ALWAYS)
    {
      print_define_remark ("DIVREM_2_THRESHOLD", 0L, "preinv always");
      return;
    }

  /* Tune for the integer part of mpn_divrem_2.  This will very possibly be
     a bit out for the fractional part, but that's too bad, the integer part
     is more important.

     min_size must be >=2 since nsize>=2 is required, but is set to 4 to save
     code space if plain division is better only at size==2 or size==3. */
  param.name = "DIVREM_2_THRESHOLD";
  param.check_size = 256;
  param.min_size = 4;
  param.min_is_always = 1;
  param.size_extra = 2;      /* does qsize==nsize-2 divisions */
  param.stop_factor = 2.0;

  s.r = randlimb_norm ();
  param.function = speed_mpn_divrem_2;
  one (&divrem_2_threshold, &param);
}

void
tune_div_qr_2 (void)
{
  static struct param_t  param;
  param.name = "DIV_QR_2_PI2_THRESHOLD";
  param.function = speed_mpn_div_qr_2n;
  param.check_size = 500;
  param.min_size = 4;
  one (&div_qr_2_pi2_threshold, &param);
}

/* mpn_divexact_1 is vaguely expected to be used on smallish divisors, so
   tune for that.  Its speed can differ on odd or even divisor, so take an
   average threshold for the two.

   mpn_divrem_1 can vary with high<divisor or not, whereas mpn_divexact_1
   might not vary that way, but don't test this since high<divisor isn't
   expected to occur often with small divisors.  */

void
tune_divexact_1 (void)
{
  static struct param_t  param;
  mp_size_t  thresh[2], average;
  int        low, i;

  /* Any native mpn_divexact_1 is assumed to incorporate all the speed of a
     full mpn_divrem_1.  */
  if (HAVE_NATIVE_mpn_divexact_1)
    {
      print_define_remark ("DIVEXACT_1_THRESHOLD", 0, "always (native)");
      return;
    }

  ASSERT_ALWAYS (tuned_speed_mpn_divrem_1 != NULL);

  param.name = "DIVEXACT_1_THRESHOLD";
  param.data_high = DATA_HIGH_GE_R;
  param.check_size = 256;
  param.min_size = 2;
  param.stop_factor = 1.5;
  param.function  = tuned_speed_mpn_divrem_1;
  param.function2 = speed_mpn_divexact_1;
  param.noprint = 1;

  print_define_start (param.name);

  for (low = 0; low <= 1; low++)
    {
      s.r = randlimb_half();
      if (low == 0)
        s.r |= 1;
      else
        s.r &= ~CNST_LIMB(7);

      one (&thresh[low], &param);
      if (option_trace)
        printf ("low=%d thresh %ld\n", low, (long) thresh[low]);

      if (thresh[low] == MP_SIZE_T_MAX)
        {
          average = MP_SIZE_T_MAX;
          goto divexact_1_done;
        }
    }

  if (option_trace)
    {
      printf ("average of:");
      for (i = 0; i < numberof(thresh); i++)
        printf (" %ld", (long) thresh[i]);
      printf ("\n");
    }

  average = 0;
  for (i = 0; i < numberof(thresh); i++)
    average += thresh[i];
  average /= numberof(thresh);

  /* If divexact turns out to be better as early as 3 limbs, then use it
     always, so as to reduce code size and conditional jumps.  */
  if (average <= 3)
    average = 0;

 divexact_1_done:
  print_define_end (param.name, average);
}


/* The generic mpn_modexact_1_odd skips a divide step if high<divisor, the
   same as mpn_mod_1, but this might not be true of an assembler
   implementation.  The threshold used is an average based on data where a
   divide can be skipped and where it can't.

   If modexact turns out to be better as early as 3 limbs, then use it
   always, so as to reduce code size and conditional jumps.  */

void
tune_modexact_1_odd (void)
{
  static struct param_t  param;
  mp_size_t  thresh_lt, thresh_ge, average;

#if 0
  /* Any native mpn_modexact_1_odd is assumed to incorporate all the speed
     of a full mpn_mod_1.  */
  if (HAVE_NATIVE_mpn_modexact_1_odd)
    {
      print_define_remark ("BMOD_1_TO_MOD_1_THRESHOLD", MP_SIZE_T_MAX, "always bmod_1");
      return;
    }
#endif

  param.name = "BMOD_1_TO_MOD_1_THRESHOLD";
  param.check_size = 256;
  param.min_size = 2;
  param.stop_factor = 1.5;
  param.function  = speed_mpn_modexact_1c_odd;
  param.function2 = speed_mpn_mod_1_tune;
  param.noprint = 1;
  s.r = randlimb_half () | 1;

  print_define_start (param.name);

  param.data_high = DATA_HIGH_LT_R;
  one (&thresh_lt, &param);
  if (option_trace)
    printf ("lt thresh %ld\n", (long) thresh_lt);

  average = thresh_lt;
  if (thresh_lt != MP_SIZE_T_MAX)
    {
      param.data_high = DATA_HIGH_GE_R;
      one (&thresh_ge, &param);
      if (option_trace)
        printf ("ge thresh %ld\n", (long) thresh_ge);

      if (thresh_ge != MP_SIZE_T_MAX)
        {
          average = (thresh_ge + thresh_lt) / 2;
          if (thresh_ge <= 3)
            average = 0;
        }
    }

  print_define_end (param.name, average);
}


void
tune_jacobi_base (void)
{
  static struct param_t  param;
  double   t1, t2, t3, t4;
  int      method;

  s.size = GMP_LIMB_BITS * 3 / 4;

  t1 = tuneup_measure (speed_mpn_jacobi_base_1, &param, &s);
  if (option_trace >= 1)
    printf ("size=%ld, mpn_jacobi_base_1 %.9f\n", (long) s.size, t1);

  t2 = tuneup_measure (speed_mpn_jacobi_base_2, &param, &s);
  if (option_trace >= 1)
    printf ("size=%ld, mpn_jacobi_base_2 %.9f\n", (long) s.size, t2);

  t3 = tuneup_measure (speed_mpn_jacobi_base_3, &param, &s);
  if (option_trace >= 1)
    printf ("size=%ld, mpn_jacobi_base_3 %.9f\n", (long) s.size, t3);

  t4 = tuneup_measure (speed_mpn_jacobi_base_4, &param, &s);
  if (option_trace >= 1)
    printf ("size=%ld, mpn_jacobi_base_4 %.9f\n", (long) s.size, t4);

  if (t1 == -1.0 || t2 == -1.0 || t3 == -1.0 || t4 == -1.0)
    {
      printf ("Oops, can't measure all mpn_jacobi_base methods at %ld\n",
              (long) s.size);
      abort ();
    }

  if (t1 < t2 && t1 < t3 && t1 < t4)
    method = 1;
  else if (t2 < t3 && t2 < t4)
    method = 2;
  else if (t3 < t4)
    method = 3;
  else
    method = 4;

  print_define ("JACOBI_BASE_METHOD", method);
}


void
tune_get_str (void)
{
  /* Tune for decimal, it being most common.  Some rough testing suggests
     other bases are different, but not by very much.  */
  s.r = 10;
  {
    static struct param_t  param;
    GET_STR_PRECOMPUTE_THRESHOLD = 0;
    param.name = "GET_STR_DC_THRESHOLD";
    param.function = speed_mpn_get_str;
    param.min_size = 4;
    param.max_size = GET_STR_THRESHOLD_LIMIT;
    one (&get_str_dc_threshold, &param);
  }
  {
    static struct param_t  param;
    param.name = "GET_STR_PRECOMPUTE_THRESHOLD";
    param.function = speed_mpn_get_str;
    param.min_size = GET_STR_DC_THRESHOLD;
    param.max_size = GET_STR_THRESHOLD_LIMIT;
    one (&get_str_precompute_threshold, &param);
  }
}


double
speed_mpn_pre_set_str (struct speed_params *s)
{
  unsigned char *str;
  mp_ptr     wp;
  mp_size_t  wn;
  unsigned   i;
  int        base;
  double     t;
  mp_ptr powtab_mem, tp;
  powers_t powtab[GMP_LIMB_BITS];
  mp_size_t un;
  int chars_per_limb;
  TMP_DECL;

  SPEED_RESTRICT_COND (s->size >= 1);

  base = s->r == 0 ? 10 : s->r;
  SPEED_RESTRICT_COND (base >= 2 && base <= 256);

  TMP_MARK;

  str = (unsigned char *) TMP_ALLOC (s->size);
  for (i = 0; i < s->size; i++)
    str[i] = s->xp[i] % base;

  LIMBS_PER_DIGIT_IN_BASE (wn, s->size, base);
  SPEED_TMP_ALLOC_LIMBS (wp, wn, s->align_wp);

  /* use this during development to check wn is big enough */
  /*
  ASSERT_ALWAYS (mpn_set_str (wp, str, s->size, base) <= wn);
  */

  speed_operand_src (s, (mp_ptr) str, s->size/GMP_LIMB_BYTES);
  speed_operand_dst (s, wp, wn);
  speed_cache_fill (s);

  chars_per_limb = mp_bases[base].chars_per_limb;
  un = s->size / chars_per_limb + 1;
  powtab_mem = TMP_BALLOC_LIMBS (mpn_dc_set_str_powtab_alloc (un));
  mpn_set_str_compute_powtab (powtab, powtab_mem, un, base);
  tp = TMP_BALLOC_LIMBS (mpn_dc_set_str_itch (un));

  speed_starttime ();
  i = s->reps;
  do
    {
      mpn_pre_set_str (wp, str, s->size, powtab, tp);
    }
  while (--i != 0);
  t = speed_endtime ();

  TMP_FREE;
  return t;
}

void
tune_set_str (void)
{
  s.r = 10;  /* decimal */
  {
    static struct param_t  param;
    SET_STR_PRECOMPUTE_THRESHOLD = 0;
    param.step_factor = 0.01;
    param.name = "SET_STR_DC_THRESHOLD";
    param.function = speed_mpn_pre_set_str;
    param.min_size = 100;
    param.max_size = 50000;
    one (&set_str_dc_threshold, &param);
  }
  {
    static struct param_t  param;
    param.step_factor = 0.02;
    param.name = "SET_STR_PRECOMPUTE_THRESHOLD";
    param.function = speed_mpn_set_str;
    param.min_size = SET_STR_DC_THRESHOLD;
    param.max_size = 100000;
    one (&set_str_precompute_threshold, &param);
  }
}


void
tune_fft_mul (void)
{
  static struct fft_param_t  param;

  if (option_fft_max_size == 0)
    return;

  param.table_name          = "MUL_FFT_TABLE3";
  param.threshold_name      = "MUL_FFT_THRESHOLD";
  param.p_threshold         = &mul_fft_threshold;
  param.modf_threshold_name = "MUL_FFT_MODF_THRESHOLD";
  param.p_modf_threshold    = &mul_fft_modf_threshold;
  param.first_size          = MUL_TOOM33_THRESHOLD / 2;
  param.max_size            = option_fft_max_size;
  param.function            = speed_mpn_fft_mul;
  param.mul_modf_function   = speed_mpn_mul_fft;
  param.mul_function        = speed_mpn_mul_n;
  param.sqr = 0;
  fft (&param);
}


void
tune_fft_sqr (void)
{
  static struct fft_param_t  param;

  if (option_fft_max_size == 0)
    return;

  param.table_name          = "SQR_FFT_TABLE3";
  param.threshold_name      = "SQR_FFT_THRESHOLD";
  param.p_threshold         = &sqr_fft_threshold;
  param.modf_threshold_name = "SQR_FFT_MODF_THRESHOLD";
  param.p_modf_threshold    = &sqr_fft_modf_threshold;
  param.first_size          = SQR_TOOM3_THRESHOLD / 2;
  param.max_size            = option_fft_max_size;
  param.function            = speed_mpn_fft_sqr;
  param.mul_modf_function   = speed_mpn_mul_fft_sqr;
  param.mul_function        = speed_mpn_sqr;
  param.sqr = 1;
  fft (&param);
}

void
tune_fac_ui (void)
{
  static struct param_t  param;

  param.function = speed_mpz_fac_ui_tune;

  param.name = "FAC_DSC_THRESHOLD";
  param.min_size = 70;
  param.max_size = FAC_DSC_THRESHOLD_LIMIT;
  one (&fac_dsc_threshold, &param);

  param.name = "FAC_ODD_THRESHOLD";
  param.min_size = 22;
  param.stop_factor = 1.7;
  param.min_is_always = 1;
  one (&fac_odd_threshold, &param);
}

void
all (void)
{
  time_t  start_time, end_time;
  TMP_DECL;

  TMP_MARK;
  SPEED_TMP_ALLOC_LIMBS (s.xp_block, SPEED_BLOCK_SIZE, 0);
  SPEED_TMP_ALLOC_LIMBS (s.yp_block, SPEED_BLOCK_SIZE, 0);

  mpn_random (s.xp_block, SPEED_BLOCK_SIZE);
  mpn_random (s.yp_block, SPEED_BLOCK_SIZE);

  fprintf (stderr, "Parameters for %s\n", GMP_MPARAM_H_SUGGEST);

  speed_time_init ();
  fprintf (stderr, "Using: %s\n", speed_time_string);

  fprintf (stderr, "speed_precision %d", speed_precision);
  if (speed_unittime == 1.0)
    fprintf (stderr, ", speed_unittime 1 cycle");
  else
    fprintf (stderr, ", speed_unittime %.2e secs", speed_unittime);
  if (speed_cycletime == 1.0 || speed_cycletime == 0.0)
    fprintf (stderr, ", CPU freq unknown\n");
  else
    fprintf (stderr, ", CPU freq %.2f MHz\n", 1e-6/speed_cycletime);

  fprintf (stderr, "DEFAULT_MAX_SIZE %d, fft_max_size %ld\n",
           DEFAULT_MAX_SIZE, (long) option_fft_max_size);
  fprintf (stderr, "\n");

  time (&start_time);
  {
    struct tm  *tp;
    tp = localtime (&start_time);
    printf ("/* Generated by tuneup.c, %d-%02d-%02d, ",
            tp->tm_year+1900, tp->tm_mon+1, tp->tm_mday);

#ifdef __GNUC__
    /* gcc sub-minor version doesn't seem to come through as a define */
    printf ("gcc %d.%d */\n", __GNUC__, __GNUC_MINOR__);
#define PRINTED_COMPILER
#endif
#if defined (__SUNPRO_C)
    printf ("Sun C %d.%d */\n", __SUNPRO_C / 0x100, __SUNPRO_C % 0x100);
#define PRINTED_COMPILER
#endif
#if ! defined (__GNUC__) && defined (__sgi) && defined (_COMPILER_VERSION)
    /* gcc defines __sgi and _COMPILER_VERSION on irix 6, avoid that */
    printf ("MIPSpro C %d.%d.%d */\n",
	    _COMPILER_VERSION / 100,
	    _COMPILER_VERSION / 10 % 10,
	    _COMPILER_VERSION % 10);
#define PRINTED_COMPILER
#endif
#if defined (__DECC) && defined (__DECC_VER)
    printf ("DEC C %d */\n", __DECC_VER);
#define PRINTED_COMPILER
#endif
#if ! defined (PRINTED_COMPILER)
    printf ("system compiler */\n");
#endif
  }
  printf ("\n");

  tune_divrem_1 ();
  tune_mod_1 ();
  tune_preinv_divrem_1 ();
  tune_div_qr_1 ();
#if 0
  tune_divrem_2 ();
#endif
  tune_div_qr_2 ();
  tune_divexact_1 ();
  tune_modexact_1_odd ();
  printf("\n");

  tune_mul_n ();
  printf("\n");

  tune_mul ();
  printf("\n");

  tune_sqr ();
  printf("\n");

  tune_mulmid ();
  printf("\n");

  tune_mulmod_bnm1 ();
  tune_sqrmod_bnm1 ();
  printf("\n");

  tune_fft_mul ();
  printf("\n");

  tune_fft_sqr ();
  printf ("\n");

  tune_mullo ();
  tune_sqrlo ();
  printf("\n");

  tune_dc_div ();
  tune_dc_bdiv ();

  printf("\n");
  tune_invertappr ();
  tune_invert ();
  printf("\n");

  tune_binvert ();
  tune_redc ();
  printf("\n");

  tune_mu_div ();
  tune_mu_bdiv ();
  printf("\n");

  tune_powm_sec ();
  printf("\n");

  tune_get_str ();
  tune_set_str ();
  printf("\n");

  tune_fac_ui ();
  printf("\n");

  tune_matrix22_mul ();
  tune_hgcd ();
  tune_hgcd_appr ();
  tune_hgcd_reduce();
  tune_gcd_dc ();
  tune_gcdext_dc ();
  tune_jacobi_base ();
  printf("\n");

  time (&end_time);
  printf ("/* Tuneup completed successfully, took %ld seconds */\n",
          (long) (end_time - start_time));

  TMP_FREE;
}


int
main (int argc, char *argv[])
{
  int  opt;

  /* Unbuffered so if output is redirected to a file it isn't lost if the
     program is killed part way through.  */
  setbuf (stdout, NULL);
  setbuf (stderr, NULL);

  while ((opt = getopt(argc, argv, "f:o:p:t")) != EOF)
    {
      switch (opt) {
      case 'f':
        if (optarg[0] == 't')
          option_fft_trace = 2;
        else
          option_fft_max_size = atol (optarg);
        break;
      case 'o':
        speed_option_set (optarg);
        break;
      case 'p':
        speed_precision = atoi (optarg);
        break;
      case 't':
        option_trace++;
        break;
      case '?':
        exit(1);
      }
    }

  all ();
  exit (0);
}
