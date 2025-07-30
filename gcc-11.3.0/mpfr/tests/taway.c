/* Test file for round away.

Copyright 2000-2017 Free Software Foundation, Inc.
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

#include <stdio.h>
#include <stdlib.h>

#include "mpfr-test.h"

#define DISP(s, t) {printf(s); mpfr_out_str(stdout, 2, 0, t, MPFR_RNDN); }
#define DISP2(s,t) {DISP(s,t); putchar('\n');}

#define SPECIAL_MAX 12

static void
set_special (mpfr_ptr x, unsigned int select)
{
  MPFR_ASSERTN (select < SPECIAL_MAX);
  switch (select)
    {
    case 0:
      MPFR_SET_NAN (x);
      break;
    case 1:
      MPFR_SET_INF (x);
      MPFR_SET_POS (x);
      break;
    case 2:
      MPFR_SET_INF (x);
      MPFR_SET_NEG (x);
      break;
    case 3:
      MPFR_SET_ZERO (x);
      MPFR_SET_POS  (x);
      break;
    case 4:
      MPFR_SET_ZERO (x);
      MPFR_SET_NEG  (x);
      break;
    case 5:
      mpfr_set_str_binary (x, "1");
      break;
    case 6:
      mpfr_set_str_binary (x, "-1");
      break;
    case 7:
      mpfr_set_str_binary (x, "1e-1");
      break;
    case 8:
      mpfr_set_str_binary (x, "1e+1");
      break;
    case 9:
      mpfr_const_pi (x, MPFR_RNDN);
      break;
    case 10:
      mpfr_const_pi (x, MPFR_RNDN);
      MPFR_SET_EXP (x, MPFR_GET_EXP (x)-1);
      break;
    default:
      mpfr_urandomb (x, RANDS);
      break;
    }
}
/* same than mpfr_cmp, but returns 0 for both NaN's */
static int
mpfr_compare (mpfr_srcptr a, mpfr_srcptr b)
{
  return (MPFR_IS_NAN(a)) ? !MPFR_IS_NAN(b) :
    (MPFR_IS_NAN(b) || mpfr_cmp(a, b));
}

static void
test3 (int (*testfunc)(mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t),
       const char *foo)
{
  mpfr_t ref1, ref2, ref3;
  mpfr_t res1;
  mpfr_prec_t p1, p2, p3;
  int i, inexa, inexd;
  mpfr_rnd_t r;

  p1 = (randlimb () % 200) + MPFR_PREC_MIN;
  p2 = (randlimb () % 200) + MPFR_PREC_MIN;
  p3 = (randlimb () % 200) + MPFR_PREC_MIN;

  mpfr_init2 (ref1, p1);
  mpfr_init2 (ref2, p2);
  mpfr_init2 (ref3, p3);
  mpfr_init2 (res1, p1);

  /* for each variable, consider each of the following 6 possibilities:
     NaN, +Infinity, -Infinity, +0, -0 or a random number */
  for (i = 0; i < SPECIAL_MAX * SPECIAL_MAX; i++)
    {
      set_special (ref2, i%SPECIAL_MAX);
      set_special (ref3, i/SPECIAL_MAX);

      inexa = testfunc (res1, ref2, ref3, MPFR_RNDA);
      r = MPFR_SIGN(res1) > 0 ? MPFR_RNDU : MPFR_RNDD;
      inexd = testfunc (ref1, ref2, ref3, r);

      if (mpfr_compare (res1, ref1) || inexa != inexd)
        {
          printf ("Error with RNDA for %s with ", foo);
          DISP("x=",ref2); DISP2(", y=",ref3);
          printf ("inexa=%d inexd=%d\n", inexa, inexd);
          printf ("expected "); mpfr_print_binary (ref1); puts ("");
          printf ("got      "); mpfr_print_binary (res1); puts ("");
          exit (1);
        }
    }

  mpfr_clear (ref1);
  mpfr_clear (ref2);
  mpfr_clear (ref3);
  mpfr_clear (res1);
}

static void
test4 (int (*testfunc)(mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_srcptr,
                       mpfr_rnd_t), const char *foo)
{
  mpfr_t ref, op1, op2, op3;
  mpfr_prec_t pout, p1, p2, p3;
  mpfr_t res;
  int i, j, k, inexa, inexd;
  mpfr_rnd_t r;

  pout = (randlimb () % 200) + MPFR_PREC_MIN;
  p1 = (randlimb () % 200) + MPFR_PREC_MIN;
  p2 = (randlimb () % 200) + MPFR_PREC_MIN;
  p3 = (randlimb () % 200) + MPFR_PREC_MIN;

  mpfr_init2 (ref, pout);
  mpfr_init2 (res, pout);
  mpfr_init2 (op1, p1);
  mpfr_init2 (op2, p2);
  mpfr_init2 (op3, p3);

  /* for each variable, consider each of the following 6 possibilities:
     NaN, +Infinity, -Infinity, +0, -0 or a random number */

  for (i = 0; i < SPECIAL_MAX; i++)
    {
      set_special (op1, i);
      for (j = 0; j < SPECIAL_MAX; j++)
        {
          set_special (op2, j);
          for (k = 0; k < SPECIAL_MAX; k++)
            {
              set_special (op3, k);

              inexa = testfunc (res, op1, op2, op3, MPFR_RNDA);
              r = MPFR_SIGN(res) > 0 ? MPFR_RNDU : MPFR_RNDD;
              inexd = testfunc (ref, op1, op2, op3, r);

              if (mpfr_compare (res, ref) || inexa != inexd)
                {
                  printf ("Error with RNDA for %s with ", foo);
                  DISP("a=", op1); DISP(", b=", op2); DISP2(", c=", op3);
                  printf ("inexa=%d inexd=%d\n", inexa, inexd);
                  DISP("expected ", ref); DISP2(", got ", res);
                  exit (1);
                }
            }
        }
    }

  mpfr_clear (ref);
  mpfr_clear (op1);
  mpfr_clear (op2);
  mpfr_clear (op3);
  mpfr_clear (res);
}

static void
test2ui (int (*testfunc)(mpfr_ptr, mpfr_srcptr, unsigned long int, mpfr_rnd_t),
         const char *foo)
{
  mpfr_t ref1, ref2;
  unsigned int ref3;
  mpfr_t res1;
  mpfr_prec_t p1, p2;
  int i, inexa, inexd;
  mpfr_rnd_t r;

  p1 = (randlimb () % 200) + MPFR_PREC_MIN;
  p2 = (randlimb () % 200) + MPFR_PREC_MIN;

  mpfr_init2 (ref1, p1);
  mpfr_init2 (ref2, p2);
  mpfr_init2 (res1, p1);

  /* ref2 can be NaN, +Inf, -Inf, +0, -0 or any number
     ref3 can be 0 or any number */
  for (i = 0; i < SPECIAL_MAX * 2; i++)
    {
      set_special (ref2, i % SPECIAL_MAX);
      ref3 = i / SPECIAL_MAX == 0 ? 0 : randlimb ();

      inexa = testfunc (res1, ref2, ref3, MPFR_RNDA);
      r = MPFR_SIGN(res1) > 0 ? MPFR_RNDU : MPFR_RNDD;
      inexd = testfunc (ref1, ref2, ref3, r);

      if (mpfr_compare (res1, ref1) || inexa != inexd)
        {
          printf ("Error with RNDA for %s for c=%u\n", foo, ref3);
          DISP2("a=",ref2);
          printf ("inexa=%d inexd=%d\n", inexa, inexd);
          printf ("expected "); mpfr_print_binary (ref1); puts ("");
          printf ("got      "); mpfr_print_binary (res1); puts ("");
          exit (1);
        }
    }

  mpfr_clear (ref1);
  mpfr_clear (ref2);
  mpfr_clear (res1);
}

static void
testui2 (int (*testfunc)(mpfr_ptr, unsigned long int, mpfr_srcptr, mpfr_rnd_t),
         const char *foo)
{
  mpfr_t ref1, ref3;
  unsigned int ref2;
  mpfr_t res1;
  mpfr_prec_t p1, p3;
  int i, inexa, inexd;
  mpfr_rnd_t r;

  p1 = (randlimb () % 200) + MPFR_PREC_MIN;
  p3 = (randlimb () % 200) + MPFR_PREC_MIN;

  mpfr_init2 (ref1, p1);
  mpfr_init2 (ref3, p3);
  mpfr_init2 (res1, p1);

  for (i = 0; i < SPECIAL_MAX * 2; i++)
    {
      set_special (ref3, i % SPECIAL_MAX);
      ref2 = i / SPECIAL_MAX == 0 ? 0 : randlimb ();

      inexa = testfunc (res1, ref2, ref3, MPFR_RNDA);
      r = MPFR_SIGN(res1) > 0 ? MPFR_RNDU : MPFR_RNDD;
      inexd = testfunc (ref1, ref2, ref3, r);

      if (mpfr_compare (res1, ref1) || inexa != inexd)
        {
          printf ("Error with RNDA for %s for b=%u\n", foo, ref2);
          DISP2("a=", ref3);
          printf ("inexa=%d inexd=%d\n", inexa, inexd);
          DISP("expected ", ref1); DISP2(", got ", res1);
          exit (1);
        }
    }

  mpfr_clear (ref1);
  mpfr_clear (ref3);
  mpfr_clear (res1);
}

/* foo(mpfr_ptr, mpfr_srcptr, mp_rndt) */
static void
test2 (int (*testfunc)(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t), const char *foo)
{
  mpfr_t ref1, ref2;
  mpfr_t res1;
  mpfr_prec_t p1, p2;
  int i, inexa, inexd;
  mpfr_rnd_t r;

  p1 = (randlimb () % 200) + MPFR_PREC_MIN;
  p2 = (randlimb () % 200) + MPFR_PREC_MIN;

  mpfr_init2 (ref1, p1);
  mpfr_init2 (ref2, p2);
  mpfr_init2 (res1, p1);

  for (i = 0; i < SPECIAL_MAX; i++)
    {
      set_special (ref2, i);

      /* first round to away */
      inexa = testfunc (res1, ref2, MPFR_RNDA);

      r = MPFR_SIGN(res1) > 0 ? MPFR_RNDU : MPFR_RNDD;
      inexd = testfunc (ref1, ref2, r);
      if (mpfr_compare (res1, ref1) || inexa != inexd)
        {
          printf ("Error with RNDA for %s with ", foo);
          DISP2("x=", ref2);
          printf ("inexa=%d inexd=%d\n", inexa, inexd);
          DISP("expected ", ref1); DISP2(", got ", res1);
          exit (1);
        }
    }

  mpfr_clear (ref1);
  mpfr_clear (ref2);
  mpfr_clear (res1);
}

/* one operand, two results, like mpfr_sin_cos */
static void
test3a (int (*testfunc)(mpfr_ptr, mpfr_ptr, mpfr_srcptr, mpfr_rnd_t),
        const char *foo)
{
  mpfr_t ref1, ref2, ref3;
  mpfr_t res1, res2;
  mpfr_prec_t p1, p2, p3;
  int i, inexa, inexd;
  mpfr_rnd_t r;

  p1 = (randlimb () % 200) + MPFR_PREC_MIN;
  p2 = (randlimb () % 200) + MPFR_PREC_MIN;
  p3 = (randlimb () % 200) + MPFR_PREC_MIN;

  mpfr_init2 (ref1, p1);
  mpfr_init2 (ref2, p2);
  mpfr_init2 (ref3, p3);
  mpfr_init2 (res1, p1);
  mpfr_init2 (res2, p2);

  for (i = 0; i < SPECIAL_MAX; i++)
    {
      set_special (ref3, i);

      inexa = testfunc (res1, res2, ref3, MPFR_RNDA);

      /* first check wrt the first operand */
      r = MPFR_SIGN(res1) > 0 ? MPFR_RNDU : MPFR_RNDD;
      inexd = testfunc (ref1, ref2, ref3, r);
      /* the low 2 bits of the inexact flag concern the 1st operand */
      if (mpfr_compare (res1, ref1) || (inexa & 3) != (inexd & 3))
        {
          printf ("Error with RNDA for %s (1st operand)\n", foo);
          DISP2("a=",ref3);
          DISP("expected ", ref1); printf ("\n");
          DISP("got      ", res1); printf ("\n");
          printf ("inexa=%d inexd=%d\n", inexa & 3, inexd & 3);
          exit (1);
        }

      /* now check wrt the second operand */
      r = MPFR_SIGN(res2) > 0 ? MPFR_RNDU : MPFR_RNDD;
      inexd = testfunc (ref1, ref2, ref3, r);
      /* bits 2..3 of the inexact flag concern the 2nd operand */
      if (mpfr_compare (res2, ref2) || (inexa >> 2) != (inexd >> 2))
        {
          printf ("Error with RNDA for %s (2nd operand)\n", foo);
          DISP2("a=",ref3);
          DISP("expected ", ref2); printf ("\n");
          DISP("got      ", res2); printf ("\n");
          printf ("inexa=%d inexd=%d\n", inexa >> 2, inexd >> 2);
          exit (1);
        }

    }

  mpfr_clear (ref1);
  mpfr_clear (ref2);
  mpfr_clear (ref3);
  mpfr_clear (res1);
  mpfr_clear (res2);
}

int
main (void)
{
  int N = 20;

  tests_start_mpfr ();

  while (N--)
    {
      /* no need to test mpfr_round, mpfr_ceil, mpfr_floor, mpfr_trunc since
         they take no rounding mode */

      test2ui (mpfr_add_ui, "mpfr_add_ui");
      test2ui (mpfr_div_2exp, "mpfr_div_2exp");
      test2ui (mpfr_div_ui, "mpfr_div_ui");
      test2ui (mpfr_mul_2exp, "mpfr_mul_2exp");
      test2ui (mpfr_mul_ui, "mpfr_mul_ui");
      test2ui (mpfr_pow_ui, "mpfr_pow_ui");
      test2ui (mpfr_sub_ui, "mpfr_sub_ui");

      testui2 (mpfr_ui_div, "mpfr_ui_div");
      testui2 (mpfr_ui_sub, "mpfr_ui_sub");
      testui2 (mpfr_ui_pow, "mpfr_ui_pow");

      test2 (mpfr_sqr, "mpfr_sqr");
      test2 (mpfr_sqrt, "mpfr_sqrt");
      test2 (mpfr_abs, "mpfr_abs");
      test2 (mpfr_neg, "mpfr_neg");

      test2 (mpfr_log, "mpfr_log");
      test2 (mpfr_log2, "mpfr_log2");
      test2 (mpfr_log10, "mpfr_log10");
      test2 (mpfr_log1p, "mpfr_log1p");

      test2 (mpfr_exp, "mpfr_exp");
      test2 (mpfr_exp2, "mpfr_exp2");
      test2 (mpfr_exp10, "mpfr_exp10");
      test2 (mpfr_expm1, "mpfr_expm1");
      test2 (mpfr_eint, "mpfr_eint");

      test2 (mpfr_sinh, "mpfr_sinh");
      test2 (mpfr_cosh, "mpfr_cosh");
      test2 (mpfr_tanh, "mpfr_tanh");
      test2 (mpfr_asinh, "mpfr_asinh");
      test2 (mpfr_acosh, "mpfr_acosh");
      test2 (mpfr_atanh, "mpfr_atanh");
      test2 (mpfr_sech, "mpfr_sech");
      test2 (mpfr_csch, "mpfr_csch");
      test2 (mpfr_coth, "mpfr_coth");

      test2 (mpfr_asin, "mpfr_asin");
      test2 (mpfr_acos, "mpfr_acos");
      test2 (mpfr_atan, "mpfr_atan");
      test2 (mpfr_cos, "mpfr_cos");
      test2 (mpfr_sin, "mpfr_sin");
      test2 (mpfr_tan, "mpfr_tan");
      test2 (mpfr_sec, "mpfr_sec");
      test2 (mpfr_csc, "mpfr_csc");
      test2 (mpfr_cot, "mpfr_cot");

      test2 (mpfr_erf,  "mpfr_erf");
      test2 (mpfr_erfc, "mpfr_erfc");
      test2 (mpfr_j0,   "mpfr_j0");
      test2 (mpfr_j1,   "mpfr_j1");
      test2 (mpfr_y0,   "mpfr_y0");
      test2 (mpfr_y1,   "mpfr_y1");
      test2 (mpfr_zeta, "mpfr_zeta");
      test2 (mpfr_gamma, "mpfr_gamma");
      test2 (mpfr_lngamma, "mpfr_lngamma");

      test2 (mpfr_rint, "mpfr_rint");
      test2 (mpfr_rint_ceil, "mpfr_rint_ceil");
      test2 (mpfr_rint_floor, "mpfr_rint_floor");
      test2 (mpfr_rint_round, "mpfr_rint_round");
      test2 (mpfr_rint_trunc, "mpfr_rint_trunc");
      test2 (mpfr_frac, "mpfr_frac");

      test3 (mpfr_add, "mpfr_add");
      test3 (mpfr_sub, "mpfr_sub");
      test3 (mpfr_mul, "mpfr_mul");
      test3 (mpfr_div, "mpfr_div");

      test3 (mpfr_agm, "mpfr_agm");
      test3 (mpfr_min, "mpfr_min");
      test3 (mpfr_max, "mpfr_max");

      /* we don't test reldiff since it does not guarantee correct rounding,
         thus we can get different results with RNDA and RNDU or RNDD. */
      test3 (mpfr_dim, "mpfr_dim");

      test3 (mpfr_remainder, "mpfr_remainder");
      test3 (mpfr_pow, "mpfr_pow");
      test3 (mpfr_atan2, "mpfr_atan2");
      test3 (mpfr_hypot, "mpfr_hypot");

      test3a (mpfr_sin_cos, "mpfr_sin_cos");

      test4 (mpfr_fma, "mpfr_fma");
      test4 (mpfr_fms, "mpfr_fms");

#if MPFR_VERSION >= MPFR_VERSION_NUM(2,4,0)
      test2 (mpfr_li2, "mpfr_li2");
      test2 (mpfr_rec_sqrt, "mpfr_rec_sqrt");
      test3 (mpfr_fmod, "mpfr_fmod");
      test3a (mpfr_modf, "mpfr_modf");
      test3a (mpfr_sinh_cosh, "mpfr_sinh_cosh");
#endif
    }

  tests_end_mpfr ();
  return 0;
}
