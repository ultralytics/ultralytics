/* tgeneric.c -- File for generic tests.

Copyright (C) 2008, 2009, 2010, 2011, 2012 INRIA

This file is part of GNU MPC.

GNU MPC is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the
Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

GNU MPC is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
more details.

You should have received a copy of the GNU Lesser General Public License
along with this program. If not, see http://www.gnu.org/licenses/ .
*/

#include "mpc-tests.h"

/* Warning: unlike the MPFR macro (defined in mpfr-impl.h), this one returns
   true when b is singular */
#define MPFR_CAN_ROUND(b,err,prec,rnd)                                  \
  (mpfr_zero_p (b) || mpfr_inf_p (b)                                    \
   || mpfr_can_round (b, (long)mpfr_get_prec (b) - (err), (rnd),        \
                      GMP_RNDZ, (prec) + ((rnd)==GMP_RNDN)))

/* functions with one input, one output */
static void
tgeneric_cc (mpc_function *function, mpc_ptr op, mpc_ptr rop,
             mpc_ptr rop4, mpc_ptr rop4rnd, mpc_rnd_t rnd)
{
  known_signs_t ks = {1, 1};

  /* We compute the result with four times the precision and check whether the
     rounding is correct. Error reports in this part of the algorithm might
     still be wrong, though, since there are two consecutive roundings (but we
     try to avoid them).  */
  function->pointer.CC (rop4, op, rnd);
  function->pointer.CC (rop, op, rnd);

  /* can't use the mpfr_can_round function when argument is singular,
     use a custom macro instead. */
  if (MPFR_CAN_ROUND (mpc_realref (rop4), 1, MPC_PREC_RE (rop),
                      MPC_RND_RE (rnd))
      && MPFR_CAN_ROUND (mpc_imagref (rop4), 1, MPC_PREC_IM (rop),
                         MPC_RND_IM (rnd)))
    mpc_set (rop4rnd, rop4, rnd);
  else
    /* avoid double rounding error */
    return;

  if (same_mpc_value (rop, rop4rnd, ks))
    return;

  /* rounding failed */
  printf ("Rounding in %s might be incorrect for\n", function->name);
  MPC_OUT (op);

  printf ("with rounding mode (%s, %s)",
          mpfr_print_rnd_mode (MPC_RND_RE (rnd)),
          mpfr_print_rnd_mode (MPC_RND_IM (rnd)));

  printf ("\n%s                     gives ", function->name);
  MPC_OUT (rop);
  printf ("%s quadruple precision gives ", function->name);
  MPC_OUT (rop4);
  printf ("and is rounded to                  ");
  MPC_OUT (rop4rnd);

  exit (1);
}

static void
tgeneric_cc_c (mpc_function *function, mpc_ptr op, mpc_ptr rop1, mpc_ptr rop2,
   mpc_ptr rop14, mpc_ptr rop24, mpc_ptr rop14rnd, mpc_ptr rop24rnd,
   mpc_rnd_t rnd1, mpc_rnd_t rnd2)
{
   /* same as the previous function, but for mpc functions computing two
      results from one argument                                          */
   known_signs_t ks = {1, 1};

   function->pointer.CC_C (rop14, rop24, op, rnd1, rnd2);
   function->pointer.CC_C (rop1,  rop2,  op, rnd1, rnd2);

   if (   MPFR_CAN_ROUND (mpc_realref (rop14), 1, MPC_PREC_RE (rop1),
                          MPC_RND_RE (rnd1))
       && MPFR_CAN_ROUND (mpc_imagref (rop14), 1, MPC_PREC_IM (rop1),
                          MPC_RND_IM (rnd1))
       && MPFR_CAN_ROUND (mpc_realref (rop24), 1, MPC_PREC_RE (rop2),
                          MPC_RND_RE (rnd2))
       && MPFR_CAN_ROUND (mpc_imagref (rop24), 1, MPC_PREC_IM (rop2),
                          MPC_RND_IM (rnd2))) {
     mpc_set (rop14rnd, rop14, rnd1);
     mpc_set (rop24rnd, rop24, rnd2);
   }
   else
     return;

   if (!same_mpc_value (rop1, rop14rnd, ks)) {
      /* rounding failed for first result */
      printf ("Rounding might be incorrect for the first result of %s at\n", function->name);
      MPC_OUT (op);
      printf ("with rounding mode (%s, %s)",
          mpfr_print_rnd_mode (MPC_RND_RE (rnd1)),
          mpfr_print_rnd_mode (MPC_RND_IM (rnd1)));
      printf ("\n%s                     gives ", function->name);
      MPC_OUT (rop1);
      printf ("%s quadruple precision gives ", function->name);
      MPC_OUT (rop14);
      printf ("and is rounded to                  ");
      MPC_OUT (rop14rnd);
      exit (1);
   }
   else if (!same_mpc_value (rop2, rop24rnd, ks)) {
      /* rounding failed for second result */
      printf ("Rounding might be incorrect for the second result of %s at\n", function->name);
      MPC_OUT (op);
      printf ("with rounding mode (%s, %s)",
          mpfr_print_rnd_mode (MPC_RND_RE (rnd2)),
          mpfr_print_rnd_mode (MPC_RND_IM (rnd2)));
      printf ("\n%s                     gives ", function->name);
      MPC_OUT (rop2);
      printf ("%s quadruple precision gives ", function->name);
      MPC_OUT (rop24);
      printf ("and is rounded to                  ");
      MPC_OUT (rop24rnd);
      exit (1);
   }
}

static void
tgeneric_fc (mpc_function *function, mpc_ptr op, mpfr_ptr rop,
             mpfr_ptr rop4, mpfr_ptr rop4rnd, mpfr_rnd_t rnd)
{
  function->pointer.FC (rop4, op, rnd);
  function->pointer.FC (rop, op, rnd);
  if (MPFR_CAN_ROUND (rop4, 1, mpfr_get_prec (rop), rnd))
    mpfr_set (rop4rnd, rop4, rnd);
  else
    return;

  if (same_mpfr_value (rop, rop4rnd, 1))
    return;

  printf ("Rounding in %s might be incorrect for\n", function->name);
  MPC_OUT (op);
  printf ("with rounding mode %s", mpfr_print_rnd_mode (rnd));

  printf ("\n%s                     gives ", function->name);
  MPFR_OUT (rop);
  printf ("%s quadruple precision gives ", function->name);
  MPFR_OUT (rop4);
  printf ("and is rounded to                  ");
  MPFR_OUT (rop4rnd);

  exit (1);
}

static void
tgeneric_cfc (mpc_function *function, mpfr_ptr op1, mpc_ptr op2,
              mpc_ptr rop, mpc_ptr rop4, mpc_ptr rop4rnd, mpc_rnd_t rnd)
{
  known_signs_t ks = {1, 1};

  function->pointer.CFC (rop4, op1, op2, rnd);
  function->pointer.CFC (rop, op1, op2, rnd);
  if (MPFR_CAN_ROUND (mpc_realref (rop4), 1, MPC_PREC_RE (rop),
                      MPC_RND_RE (rnd))
      && MPFR_CAN_ROUND (mpc_imagref (rop4), 1, MPC_PREC_IM (rop),
                         MPC_RND_IM (rnd)))
    mpc_set (rop4rnd, rop4, rnd);
  else
    return;

  if (same_mpc_value (rop, rop4rnd, ks))
    return;

  printf ("Rounding in %s might be incorrect for\n", function->name);
  MPFR_OUT (op1);
  MPC_OUT (op2);
  printf ("with rounding mode (%s, %s)",
          mpfr_print_rnd_mode (MPC_RND_RE (rnd)),
          mpfr_print_rnd_mode (MPC_RND_IM (rnd)));

  printf ("\n%s                     gives ", function->name);
  MPC_OUT (rop);
  printf ("%s quadruple precision gives ", function->name);
  MPC_OUT (rop4);
  printf ("and is rounded to                  ");
  MPC_OUT (rop4rnd);

  exit (1);
}

static void
tgeneric_ccf (mpc_function *function, mpc_ptr op1, mpfr_ptr op2,
              mpc_ptr rop, mpc_ptr rop4, mpc_ptr rop4rnd, mpc_rnd_t rnd)
{
  known_signs_t ks = {1, 1};

  function->pointer.CCF (rop4, op1, op2, rnd);
  function->pointer.CCF (rop, op1, op2, rnd);
  if (MPFR_CAN_ROUND (mpc_realref (rop4), 1, MPC_PREC_RE (rop),
                      MPC_RND_RE (rnd))
      && MPFR_CAN_ROUND (mpc_imagref (rop4), 1, MPC_PREC_IM (rop),
                         MPC_RND_IM (rnd)))
    mpc_set (rop4rnd, rop4, rnd);
  else
    return;

  if (same_mpc_value (rop, rop4rnd, ks))
    return;

  printf ("Rounding in %s might be incorrect for\n", function->name);
  MPC_OUT (op1);
  MPFR_OUT (op2);
  printf ("with rounding mode (%s, %s)",
          mpfr_print_rnd_mode (MPC_RND_RE (rnd)),
          mpfr_print_rnd_mode (MPC_RND_IM (rnd)));

  printf ("\n%s                     gives ", function->name);
  MPC_OUT (rop);
  printf ("%s quadruple precision gives ", function->name);
  MPC_OUT (rop4);
  printf ("and is rounded to                  ");
  MPC_OUT (rop4rnd);

  exit (1);
}

/* for functions with one mpc_t output, two mpc_t inputs */
static void
tgeneric_c_cc (mpc_function *function, mpc_ptr op1, mpc_ptr op2,
	       mpc_ptr rop, mpc_ptr rop4, mpc_ptr rop4rnd, mpc_rnd_t rnd)
{
  known_signs_t ks = {1, 1};

  /* We compute the result with four times the precision and check whether the
     rounding is correct. Error reports in this part of the algorithm might
     still be wrong, though, since there are two consecutive roundings (but we
     try to avoid them).  */
  function->pointer.C_CC (rop4, op1, op2, rnd);
  function->pointer.C_CC (rop, op1, op2, rnd);

  /* can't use mpfr_can_round when argument is singular */
  if (MPFR_CAN_ROUND (mpc_realref (rop4), 1, MPC_PREC_RE (rop),
                      MPC_RND_RE (rnd))
      && MPFR_CAN_ROUND (mpc_imagref (rop4), 1, MPC_PREC_IM (rop),
                         MPC_RND_IM (rnd)))
    mpc_set (rop4rnd, rop4, rnd);
  else
    /* avoid double rounding error */
    return;

  if (same_mpc_value (rop, rop4rnd, ks))
    return;

  /* rounding failed */
  printf ("Rounding in %s might be incorrect for\n", function->name);
  MPC_OUT (op1);
  MPC_OUT (op2);
  printf ("with rounding mode (%s, %s)",
          mpfr_print_rnd_mode (MPC_RND_RE (rnd)),
          mpfr_print_rnd_mode (MPC_RND_IM (rnd)));

  printf ("\n%s                     gives ", function->name);
  MPC_OUT (rop);
  printf ("%s quadruple precision gives ", function->name);
  MPC_OUT (rop4);
  printf ("and is rounded to                  ");
  MPC_OUT (rop4rnd);

  exit (1);
}

static void
tgeneric_cccc (mpc_function *function, mpc_ptr op1, mpc_ptr op2, mpc_ptr op3,
              mpc_ptr rop, mpc_ptr rop4, mpc_ptr rop4rnd, mpc_rnd_t rnd)
{
  known_signs_t ks = {1, 1};

  /* We compute the result with four times the precision and check whether the
     rounding is correct. Error reports in this part of the algorithm might
     still be wrong, though, since there are two consecutive roundings (but we
     try to avoid them).  */
  function->pointer.CCCC (rop4, op1, op2, op3, rnd);
  function->pointer.CCCC (rop, op1, op2, op3, rnd);

  /* can't use mpfr_can_round when argument is singular */
  if (MPFR_CAN_ROUND (mpc_realref (rop4), 1, MPC_PREC_RE (rop),
                      MPC_RND_RE (rnd))
      && MPFR_CAN_ROUND (mpc_imagref (rop4), 1, MPC_PREC_IM (rop),
                         MPC_RND_IM (rnd)))
    mpc_set (rop4rnd, rop4, rnd);
  else
    /* avoid double rounding error */
    return;

  if (same_mpc_value (rop, rop4rnd, ks))
    return;

  /* rounding failed */
  printf ("Rounding in %s might be incorrect for\n", function->name);
  MPC_OUT (op1);
  MPC_OUT (op2);
  MPC_OUT (op3);
  printf ("with rounding mode (%s, %s)",
          mpfr_print_rnd_mode (MPC_RND_RE (rnd)),
          mpfr_print_rnd_mode (MPC_RND_IM (rnd)));

  printf ("\n%s                     gives ", function->name);
  MPC_OUT (rop);
  printf ("%s quadruple precision gives ", function->name);
  MPC_OUT (rop4);
  printf ("and is rounded to                  ");
  MPC_OUT (rop4rnd);

  exit (1);
}

static void
tgeneric_ccu (mpc_function *function, mpc_ptr op1, unsigned long int op2,
              mpc_ptr rop, mpc_ptr rop4, mpc_ptr rop4rnd, mpc_rnd_t rnd)
{
  known_signs_t ks = {1, 1};

  function->pointer.CCU (rop4, op1, op2, rnd);
  function->pointer.CCU (rop, op1, op2, rnd);
  if (MPFR_CAN_ROUND (mpc_realref (rop4), 1, MPC_PREC_RE (rop),
                      MPC_RND_RE (rnd))
      && MPFR_CAN_ROUND (mpc_imagref (rop4), 1, MPC_PREC_IM (rop),
                         MPC_RND_IM (rnd)))
    mpc_set (rop4rnd, rop4, rnd);
  else
    return;

  if (same_mpc_value (rop, rop4rnd, ks))
    return;

  printf ("Rounding in %s might be incorrect for\n", function->name);
  MPC_OUT (op1);
  printf ("op2=%lu\n", op2);
  printf ("with rounding mode (%s, %s)",
          mpfr_print_rnd_mode (MPC_RND_RE (rnd)),
          mpfr_print_rnd_mode (MPC_RND_IM (rnd)));

  printf ("\n%s                     gives ", function->name);
  MPC_OUT (rop);
  printf ("%s quadruple precision gives ", function->name);
  MPC_OUT (rop4);
  printf ("and is rounded to                  ");
  MPC_OUT (rop4rnd);

  exit (1);
}

static void
tgeneric_cuc (mpc_function *function, unsigned long int op1, mpc_ptr op2,
              mpc_ptr rop, mpc_ptr rop4, mpc_ptr rop4rnd, mpc_rnd_t rnd)
{
  known_signs_t ks = {1, 1};

  function->pointer.CUC (rop4, op1, op2, rnd);
  function->pointer.CUC (rop, op1, op2, rnd);
  if (MPFR_CAN_ROUND (mpc_realref (rop4), 1, MPC_PREC_RE (rop),
                      MPC_RND_RE (rnd))
      && MPFR_CAN_ROUND (mpc_imagref (rop4), 1, MPC_PREC_IM (rop),
                         MPC_RND_IM (rnd)))
    mpc_set (rop4rnd, rop4, rnd);
  else
    return;

  if (same_mpc_value (rop, rop4rnd, ks))
    return;

  printf ("Rounding in %s might be incorrect for\n", function->name);
  printf ("op1=%lu\n", op1);
  MPC_OUT (op2);
  printf ("with rounding mode (%s, %s)",
          mpfr_print_rnd_mode (MPC_RND_RE (rnd)),
          mpfr_print_rnd_mode (MPC_RND_IM (rnd)));

  printf ("\n%s                     gives ", function->name);
  MPC_OUT (rop);
  printf ("%s quadruple precision gives ", function->name);
  MPC_OUT (rop4);
  printf ("and is rounded to                  ");
  MPC_OUT (rop4rnd);

  exit (1);
}

static void
tgeneric_ccs (mpc_function *function, mpc_ptr op1, long int op2,
              mpc_ptr rop, mpc_ptr rop4, mpc_ptr rop4rnd, mpc_rnd_t rnd)
{
  known_signs_t ks = {1, 1};

  function->pointer.CCS (rop4, op1, op2, rnd);
  function->pointer.CCS (rop, op1, op2, rnd);
  if (MPFR_CAN_ROUND (mpc_realref (rop4), 1, MPC_PREC_RE (rop),
                      MPC_RND_RE (rnd))
      && MPFR_CAN_ROUND (mpc_imagref (rop4), 1, MPC_PREC_IM (rop),
                         MPC_RND_IM (rnd)))
    mpc_set (rop4rnd, rop4, rnd);
  else
    return;

  if (same_mpc_value (rop, rop4rnd, ks))
    return;

  printf ("Rounding in %s might be incorrect for\n", function->name);
  MPC_OUT (op1);
  printf ("op2=%ld\n", op2);
  printf ("with rounding mode (%s, %s)",
          mpfr_print_rnd_mode (MPC_RND_RE (rnd)),
          mpfr_print_rnd_mode (MPC_RND_IM (rnd)));

  printf ("\n%s                     gives ", function->name);
  MPC_OUT (rop);
  printf ("%s quadruple precision gives ", function->name);
  MPC_OUT (rop4);
  printf ("and is rounded to                  ");
  MPC_OUT (rop4rnd);

  exit (1);
}


static void
tgeneric_cci (mpc_function *function, mpc_ptr op1, int op2,
              mpc_ptr rop, mpc_ptr rop4, mpc_ptr rop4rnd, mpc_rnd_t rnd)
{
  known_signs_t ks = {1, 1};

  function->pointer.CCI (rop4, op1, op2, rnd);
  function->pointer.CCI (rop, op1, op2, rnd);
  if (MPFR_CAN_ROUND (mpc_realref (rop4), 1, MPC_PREC_RE (rop),
                      MPC_RND_RE (rnd))
      && MPFR_CAN_ROUND (mpc_imagref (rop4), 1, MPC_PREC_IM (rop),
                         MPC_RND_IM (rnd)))
    mpc_set (rop4rnd, rop4, rnd);
  else
    return;

  if (same_mpc_value (rop, rop4rnd, ks))
    return;

  printf ("Rounding in %s might be incorrect for\n", function->name);
  MPC_OUT (op1);
  printf ("op2=%d\n", op2);
  printf ("with rounding mode (%s, %s)",
          mpfr_print_rnd_mode (MPC_RND_RE (rnd)),
          mpfr_print_rnd_mode (MPC_RND_IM (rnd)));

  printf ("\n%s                     gives ", function->name);
  MPC_OUT (rop);
  printf ("%s quadruple precision gives ", function->name);
  MPC_OUT (rop4);
  printf ("and is rounded to                  ");
  MPC_OUT (rop4rnd);

  exit (1);
}

static void
tgeneric_cuuc (mpc_function *function, unsigned long int op1,
               unsigned long int op2, mpc_ptr op3, mpc_ptr rop,
               mpc_ptr rop4, mpc_ptr rop4rnd, mpc_rnd_t rnd)
{
  known_signs_t ks = {1, 1};

  function->pointer.CUUC (rop4, op1, op2, op3, rnd);
  function->pointer.CUUC (rop, op1, op2, op3, rnd);
  if (MPFR_CAN_ROUND (mpc_realref (rop4), 1, MPC_PREC_RE (rop),
                      MPC_RND_RE (rnd))
      && MPFR_CAN_ROUND (mpc_imagref (rop4), 1, MPC_PREC_IM (rop),
                         MPC_RND_IM (rnd)))
    mpc_set (rop4rnd, rop4, rnd);
  else
    return;

  if (same_mpc_value (rop, rop4rnd, ks))
    return;

  printf ("Rounding in %s might be incorrect for\n", function->name);
  printf ("op1=%lu\n", op1);
  printf ("op2=%lu\n", op2);
  MPC_OUT (op3);
  printf ("with rounding mode (%s, %s)",
          mpfr_print_rnd_mode (MPC_RND_RE (rnd)),
          mpfr_print_rnd_mode (MPC_RND_IM (rnd)));

  printf ("\n%s                     gives ", function->name);
  MPC_OUT (rop);
  printf ("%s quadruple precision gives ", function->name);
  MPC_OUT (rop4);
  printf ("and is rounded to                  ");
  MPC_OUT (rop4rnd);

  exit (1);
}


/* Test parameter reuse: the function should not use its output parameter in
   internal computations. */
static void
reuse_cc (mpc_function* function, mpc_srcptr z, mpc_ptr got, mpc_ptr expected)
{
  known_signs_t ks = {1, 1};

  mpc_set (got, z, MPC_RNDNN); /* exact */
  function->pointer.CC (expected, z, MPC_RNDNN);
  function->pointer.CC (got, got, MPC_RNDNN);
  if (!same_mpc_value (got, expected, ks))
    {
      printf ("Reuse error for %s(z, z) for\n", function->name);
      MPC_OUT (z);
      MPC_OUT (expected);
      MPC_OUT (got);

      exit (1);
    }
}

static void
reuse_cc_c (mpc_function* function, mpc_srcptr z, mpc_ptr got1, mpc_ptr got2,
            mpc_ptr exp1, mpc_ptr exp2)
{
   known_signs_t ks = {1, 1};

   function->pointer.CC_C (exp1, exp2, z, MPC_RNDNN, MPC_RNDNN);
   mpc_set (got1, z, MPC_RNDNN); /* exact */
   function->pointer.CC_C (got1, got2, got1, MPC_RNDNN, MPC_RNDNN);
   if (   !same_mpc_value (got1, exp1, ks)
       || !same_mpc_value (got2, exp2, ks)) {
      printf ("Reuse error in first result of %s for\n", function->name);
      MPC_OUT (z);
      MPC_OUT (exp1);
      MPC_OUT (got1);
      MPC_OUT (exp2);
      MPC_OUT (got2);
      exit (1);
   }
   mpc_set (got2, z, MPC_RNDNN); /* exact */
   function->pointer.CC_C (got1, got2, got2, MPC_RNDNN, MPC_RNDNN);
   if (   !same_mpc_value (got1, exp1, ks)
       || !same_mpc_value (got2, exp2, ks)) {
      printf ("Reuse error in second result of %s for\n", function->name);
      MPC_OUT (z);
      MPC_OUT (exp1);
      MPC_OUT (got1);
      MPC_OUT (exp2);
      MPC_OUT (got2);
      exit (1);
   }
}

static void
reuse_fc (mpc_function* function, mpc_ptr z, mpc_ptr x, mpfr_ptr expected)
{
  mpc_set (x, z, MPC_RNDNN); /* exact */
  function->pointer.FC (expected, z, GMP_RNDN);
  function->pointer.FC (mpc_realref (x), x, GMP_RNDN);
  if (!same_mpfr_value (mpc_realref (x), expected, 1))
    {
      mpfr_t got;
      got[0] = mpc_realref(x)[0]; /* display sensible name */
      printf ("Reuse error for %s(mpc_realref(z), z) for\n", function->name);
      MPC_OUT (z);
      MPFR_OUT (expected);
      MPFR_OUT (got);

      exit (1);
    }
  mpc_set (x, z, MPC_RNDNN); /* exact */
  function->pointer.FC (mpc_imagref (x), x, GMP_RNDN);
  if (!same_mpfr_value (mpc_imagref (x), expected, 1))
    {
      mpfr_t got;
      got[0] = mpc_imagref(x)[0]; /* display sensible name */
      printf ("Reuse error for %s(mpc_imagref(z), z) for \n", function->name);
      MPC_OUT (z);
      MPFR_OUT (expected);
      MPFR_OUT (got);

      exit (1);
    }
}

static void
reuse_cfc (mpc_function* function, mpc_srcptr z, mpfr_srcptr x, mpc_ptr got,
           mpc_ptr expected)
{
  known_signs_t ks = {1, 1};

  mpc_set (got, z, MPC_RNDNN); /* exact */
  function->pointer.CFC (expected, x, z, MPC_RNDNN);
  function->pointer.CFC (got, x, got, MPC_RNDNN);
  if (!same_mpc_value (got, expected, ks))
    {
      printf ("Reuse error for %s(z, x, z) for\n", function->name);
      MPFR_OUT (x);
      MPC_OUT (z);
      MPC_OUT (expected);
      MPC_OUT (got);

      exit (1);
    }
}

static void
reuse_ccf (mpc_function* function, mpc_srcptr z, mpfr_srcptr x, mpc_ptr got,
           mpc_ptr expected)
{
  known_signs_t ks = {1, 1};

  mpc_set (got, z, MPC_RNDNN); /* exact */
  function->pointer.CCF (expected, z, x, MPC_RNDNN);
  function->pointer.CCF (got, got, x, MPC_RNDNN);
  if (!same_mpc_value (got, expected, ks))
    {
      printf ("Reuse error for %s(z, z, x, RNDNN) for\n", function->name);
      MPC_OUT (z);
      MPFR_OUT (x);
      MPC_OUT (expected);
      MPC_OUT (got);

      exit (1);
    }
}

/* for functions with one mpc_t output, two mpc_t inputs */
static void
reuse_c_cc (mpc_function* function, mpc_srcptr z, mpc_srcptr x,
           mpc_ptr got, mpc_ptr expected)
{
  known_signs_t ks = {1, 1};

  mpc_set (got, z, MPC_RNDNN); /* exact */
  function->pointer.C_CC (expected, z, x, MPC_RNDNN);
  function->pointer.C_CC (got, got, x, MPC_RNDNN);
  if (!same_mpc_value (got, expected, ks))
    {
      printf ("Reuse error for %s(z, z, x) for\n", function->name);
      MPC_OUT (z);
      MPC_OUT (x);
      MPC_OUT (expected);
      MPC_OUT (got);

      exit (1);
    }
  mpc_set (got, x, MPC_RNDNN); /* exact */
  function->pointer.C_CC (expected, z, x, MPC_RNDNN);
  function->pointer.C_CC (got, z, got, MPC_RNDNN);
  if (!same_mpc_value (got, expected, ks))
    {
      printf ("Reuse error for %s(x, z, x) for\n", function->name);
      MPC_OUT (z);
      MPC_OUT (x);
      MPC_OUT (expected);
      MPC_OUT (got);

      exit (1);
    }
  mpc_set (got, x, MPC_RNDNN); /* exact */
  function->pointer.C_CC (expected, x, x, MPC_RNDNN);
  function->pointer.C_CC (got, got, got, MPC_RNDNN);
  if (!same_mpc_value (got, expected, ks))
    {
      printf ("Reuse error for %s(x, x, x) for\n", function->name);
      MPC_OUT (x);
      MPC_OUT (expected);
      MPC_OUT (got);

      exit (1);
    }
}

static void
reuse_cccc (mpc_function* function, mpc_srcptr z, mpc_srcptr x, mpc_srcptr y,
	    mpc_ptr got, mpc_ptr expected)
{
  known_signs_t ks = {1, 1};

  mpc_set (got, z, MPC_RNDNN); /* exact */
  function->pointer.CCCC (expected, z, x, y, MPC_RNDNN);
  function->pointer.CCCC (got, got, x, y, MPC_RNDNN);
  if (!same_mpc_value (got, expected, ks))
    {
      printf ("Reuse error for %s(z, z, x, y) for\n", function->name);
      MPC_OUT (z);
      MPC_OUT (x);
      MPC_OUT (y);
      MPC_OUT (expected);
      MPC_OUT (got);

      exit (1);
    }

  mpc_set (got, x, MPC_RNDNN); /* exact */
  function->pointer.CCCC (expected, z, x, y, MPC_RNDNN);
  function->pointer.CCCC (got, z, got, y, MPC_RNDNN);
  if (!same_mpc_value (got, expected, ks))
    {
      printf ("Reuse error for %s(x, z, x, y) for\n", function->name);
      MPC_OUT (z);
      MPC_OUT (x);
      MPC_OUT (y);
      MPC_OUT (expected);
      MPC_OUT (got);

      exit (1);
    }

  mpc_set (got, y, MPC_RNDNN); /* exact */
  function->pointer.CCCC (expected, z, x, y, MPC_RNDNN);
  function->pointer.CCCC (got, z, x, got, MPC_RNDNN);
  if (!same_mpc_value (got, expected, ks))
    {
      printf ("Reuse error for %s(y, z, x, y) for\n", function->name);
      MPC_OUT (z);
      MPC_OUT (x);
      MPC_OUT (y);
      MPC_OUT (expected);
      MPC_OUT (got);

      exit (1);
    }

  mpc_set (got, x, MPC_RNDNN); /* exact */
  function->pointer.CCCC (expected, x, x, x, MPC_RNDNN);
  function->pointer.CCCC (got, got, got, got, MPC_RNDNN);
  if (!same_mpc_value (got, expected, ks))
    {
      printf ("Reuse error for %s(x, x, x, x) for\n", function->name);
      MPC_OUT (x);
      MPC_OUT (expected);
      MPC_OUT (got);

      exit (1);
    }
}

static void
reuse_ccu (mpc_function* function, mpc_srcptr z, unsigned long ul,
           mpc_ptr got, mpc_ptr expected)
{
  known_signs_t ks = {1, 1};

  mpc_set (got, z, MPC_RNDNN); /* exact */
  function->pointer.CCU (expected, z, ul, MPC_RNDNN);
  function->pointer.CCU (got, got, ul, MPC_RNDNN);
  if (!same_mpc_value (got, expected, ks))
    {
      printf ("Reuse error for %s(z, z, n) for\n", function->name);
      MPC_OUT (z);
      printf ("n=%lu\n", ul);
      MPC_OUT (expected);
      MPC_OUT (got);

      exit (1);
    }
}

static void
reuse_cuc (mpc_function* function, unsigned long ul, mpc_srcptr z,
           mpc_ptr got, mpc_ptr expected)
{
  known_signs_t ks = {1, 1};

  mpc_set (got, z, MPC_RNDNN); /* exact */
  function->pointer.CUC (expected, ul, z,MPC_RNDNN);
  function->pointer.CUC (got, ul, got, MPC_RNDNN);
  if (!same_mpc_value (got, expected, ks))
    {
      printf ("Reuse error for %s(z, n, z) for\n", function->name);
      printf ("n=%lu\n", ul);
      MPC_OUT (z);
      MPC_OUT (expected);
      MPC_OUT (got);

      exit (1);
    }
}

static void
reuse_ccs (mpc_function* function, mpc_srcptr z, long lo,
           mpc_ptr got, mpc_ptr expected)
{
  known_signs_t ks = {1, 1};

  mpc_set (got, z, MPC_RNDNN); /* exact */
  function->pointer.CCS (expected, z, lo, MPC_RNDNN);
  function->pointer.CCS (got, got, lo, MPC_RNDNN);
  if (!same_mpc_value (got, expected, ks))
    {
      printf ("Reuse error for %s(z, z, n) for\n", function->name);
      MPC_OUT (z);
      printf ("n=%ld\n", lo);
      MPC_OUT (expected);
      MPC_OUT (got);

      exit (1);
    }
}

static void
reuse_cci (mpc_function* function, mpc_srcptr z, int i,
           mpc_ptr got, mpc_ptr expected)
{
  known_signs_t ks = {1, 1};

  mpc_set (got, z, MPC_RNDNN); /* exact */
  function->pointer.CCI (expected, z, i, MPC_RNDNN);
  function->pointer.CCI (got, got, i, MPC_RNDNN);
  if (!same_mpc_value (got, expected, ks))
    {
      printf ("Reuse error for %s(z, z, n) for\n", function->name);
      MPC_OUT (z);
      printf ("n=%d\n", i);
      MPC_OUT (expected);
      MPC_OUT (got);

      exit (1);
    }
}

static void
reuse_cuuc (mpc_function* function, unsigned long ul1, unsigned long ul2,
            mpc_srcptr z, mpc_ptr got, mpc_ptr expected)
{
  known_signs_t ks = {1, 1};

  mpc_set (got, z, MPC_RNDNN); /* exact */
  function->pointer.CUUC (expected, ul1, ul2, z,MPC_RNDNN);
  function->pointer.CUUC (got, ul1, ul2, got, MPC_RNDNN);
  if (!same_mpc_value (got, expected, ks))
    {
      printf ("Reuse error for %s(z, m, n, z) for\n", function->name);
      printf ("m=%lu\n", ul1);
      printf ("n=%lu\n", ul2);
      MPC_OUT (z);
      MPC_OUT (expected);
      MPC_OUT (got);

      exit (1);
    }
}


/* helper functions for iterating over mpfr rounding modes */
static mpfr_rnd_t
first_rnd_mode (void)
{
   return GMP_RNDN;
}

static mpfr_rnd_t
next_rnd_mode (mpfr_rnd_t curr)
   /* assumes that all rounding modes are non-negative, and returns -1
      when curr is the last rounding mode                              */
{
   switch (curr) {
      case GMP_RNDN:
         return GMP_RNDZ;
      case GMP_RNDZ:
         return GMP_RNDU;
      case GMP_RNDU:
         return GMP_RNDD;
      default:
         /* return invalid guard value in mpfr_rnd_t */
#if MPFR_VERSION_MAJOR < 3
         return GMP_RNDNA;
#else
         return MPFR_RNDA; /* valid rounding type, but not (yet) used in mpc */
#endif
   }
}

static int
is_valid_rnd_mode (mpfr_rnd_t curr)
   /* returns 1 if curr is a valid rounding mode, and 0otherwise */
{
   if (   curr == GMP_RNDN || curr == GMP_RNDZ
       || curr == GMP_RNDU || curr == GMP_RNDD)
      return 1;
   else
      return 0;
}

/* tgeneric(prec_min, prec_max, step, exp_max) checks rounding with random
   numbers:
   - with precision ranging from prec_min to prec_max with an increment of
   step,
   - with exponent between -exp_max and exp_max.

   It also checks parameter reuse (it is assumed here that either two mpc_t
   variables are equal or they are different, in the sense that the real part
   of one of them cannot be the imaginary part of the other). */
void
tgeneric (mpc_function function, mpfr_prec_t prec_min,
          mpfr_prec_t prec_max, mpfr_prec_t step, mpfr_exp_t exp_max)
{
  unsigned long ul1 = 0, ul2 = 0;
  long lo = 0;
  int i = 0;
  mpfr_t x1, x2, xxxx;
  mpc_t  z1, z2, z3, z4, z5, zzzz, zzzz2;

  mpfr_rnd_t rnd_re, rnd_im, rnd2_re, rnd2_im;
  mpfr_prec_t prec;
  mpfr_exp_t exp_min;
  int special, special_cases;

  mpc_init2 (z1, prec_max);
  switch (function.type)
    {
    case C_CC:
      mpc_init2 (z2, prec_max);
      mpc_init2 (z3, prec_max);
      mpc_init2 (z4, prec_max);
      mpc_init2 (zzzz, 4*prec_max);
      special_cases = 8;
      break;
    case CCCC:
      mpc_init2 (z2, prec_max);
      mpc_init2 (z3, prec_max);
      mpc_init2 (z4, prec_max);
      mpc_init2 (z5, prec_max);
      mpc_init2 (zzzz, 4*prec_max);
      special_cases = 8;
      break;
    case FC:
      mpfr_init2 (x1, prec_max);
      mpfr_init2 (x2, prec_max);
      mpfr_init2 (xxxx, 4*prec_max);
      mpc_init2 (z2, prec_max);
      special_cases = 4;
      break;
    case CCF: case CFC:
      mpfr_init2 (x1, prec_max);
      mpc_init2 (z2, prec_max);
      mpc_init2 (z3, prec_max);
      mpc_init2 (zzzz, 4*prec_max);
      special_cases = 6;
      break;
    case CCI: case CCS:
    case CCU: case CUC:
      mpc_init2 (z2, prec_max);
      mpc_init2 (z3, prec_max);
      mpc_init2 (zzzz, 4*prec_max);
      special_cases = 5;
      break;
    case CUUC:
      mpc_init2 (z2, prec_max);
      mpc_init2 (z3, prec_max);
      mpc_init2 (zzzz, 4*prec_max);
      special_cases = 6;
      break;
    case CC_C:
      mpc_init2 (z2, prec_max);
      mpc_init2 (z3, prec_max);
      mpc_init2 (z4, prec_max);
      mpc_init2 (z5, prec_max);
      mpc_init2 (zzzz, 4*prec_max);
      mpc_init2 (zzzz2, 4*prec_max);
      special_cases = 4;
      break;
    case CC:
    default:
      mpc_init2 (z2, prec_max);
      mpc_init2 (z3, prec_max);
      mpc_init2 (zzzz, 4*prec_max);
      special_cases = 4;
    }

  exp_min = mpfr_get_emin ();
  if (exp_max <= 0 || exp_max > mpfr_get_emax ())
    exp_max = mpfr_get_emax();
  if (-exp_max > exp_min)
    exp_min = - exp_max;

  if (step < 1)
    step = 1;

  for (prec = prec_min, special = 0;
       prec <= prec_max || special <= special_cases;
       prec+=step, special += (prec > prec_max ? 1 : 0)) {
       /* In the end, test functions in special cases of purely real, purely
          imaginary or infinite arguments. */

      /* probability of one zero part in 256th (25 is almost 10%) */
      const unsigned int zero_probability = special != 0 ? 0 : 25;

      mpc_set_prec (z1, prec);
      test_default_random (z1, exp_min, exp_max, 128, zero_probability);

      switch (function.type)
        {
        case C_CC:
          mpc_set_prec (z2, prec);
          test_default_random (z2, exp_min, exp_max, 128, zero_probability);
          mpc_set_prec (z3, prec);
          mpc_set_prec (z4, prec);
          mpc_set_prec (zzzz, 4*prec);
          switch (special)
            {
            case 1:
              mpfr_set_ui (mpc_realref (z1), 0, GMP_RNDN);
              break;
            case 2:
              mpfr_set_inf (mpc_realref (z1), +1);
              break;
            case 3:
              mpfr_set_ui (mpc_imagref (z1), 0, GMP_RNDN);
              break;
            case 4:
              mpfr_set_inf (mpc_imagref (z1), -1);
              break;
            case 5:
              mpfr_set_ui (mpc_realref (z2), 0, GMP_RNDN);
              break;
            case 6:
              mpfr_set_inf (mpc_realref (z2), -1);
              break;
            case 7:
              mpfr_set_ui (mpc_imagref (z2), 0, GMP_RNDN);
              break;
            case 8:
              mpfr_set_inf (mpc_imagref (z2), +1);
              break;
            }
          break;
        case CCCC:
          mpc_set_prec (z2, prec);
          test_default_random (z2, exp_min, exp_max, 128, zero_probability);
          mpc_set_prec (z3, prec);
          mpc_set_prec (z4, prec);
          mpc_set_prec (z5, prec);
          mpc_set_prec (zzzz, 4*prec);
          switch (special)
            {
            case 1:
              mpfr_set_ui (mpc_realref (z1), 0, GMP_RNDN);
              break;
            case 2:
              mpfr_set_inf (mpc_realref (z1), +1);
              break;
            case 3:
              mpfr_set_ui (mpc_imagref (z1), 0, GMP_RNDN);
              break;
            case 4:
              mpfr_set_inf (mpc_imagref (z1), -1);
              break;
            case 5:
              mpfr_set_ui (mpc_realref (z2), 0, GMP_RNDN);
              break;
            case 6:
              mpfr_set_inf (mpc_realref (z2), -1);
              break;
            case 7:
              mpfr_set_ui (mpc_imagref (z2), 0, GMP_RNDN);
              break;
            case 8:
              mpfr_set_inf (mpc_imagref (z2), +1);
              break;
            }
          break;
        case FC:
          mpc_set_prec (z2, prec);
          mpfr_set_prec (x1, prec);
          mpfr_set_prec (x2, prec);
          mpfr_set_prec (xxxx, 4*prec);
          switch (special)
            {
            case 1:
              mpfr_set_ui (mpc_realref (z1), 0, GMP_RNDN);
              break;
            case 2:
              mpfr_set_inf (mpc_realref (z1), +1);
              break;
            case 3:
              mpfr_set_ui (mpc_imagref (z1), 0, GMP_RNDN);
              break;
            case 4:
              mpfr_set_inf (mpc_imagref (z1), -1);
              break;
            }
          break;
        case CCU: case CUC:
          mpc_set_prec (z2, 128);
          do {
            test_default_random (z2, 0, 64, 128, zero_probability);
          } while (!mpfr_fits_ulong_p (mpc_realref (z2), GMP_RNDN));
          ul1 = mpfr_get_ui (mpc_realref(z2), GMP_RNDN);
          mpc_set_prec (z2, prec);
          mpc_set_prec (z3, prec);
          mpc_set_prec (zzzz, 4*prec);
          switch (special)
            {
            case 1:
              mpfr_set_ui (mpc_realref (z1), 0, GMP_RNDN);
              break;
            case 2:
              mpfr_set_inf (mpc_realref (z1), +1);
              break;
            case 3:
              mpfr_set_ui (mpc_imagref (z1), 0, GMP_RNDN);
              break;
            case 4:
              mpfr_set_inf (mpc_imagref (z1), -1);
              break;
            case 5:
              ul1 = 0;
              break;
            }
          break;
        case CUUC:
          mpc_set_prec (z2, 128);
          do {
            test_default_random (z2, 0, 64, 128, zero_probability);
          } while (!mpfr_fits_ulong_p (mpc_realref (z2), GMP_RNDN)
                   ||!mpfr_fits_ulong_p (mpc_imagref (z2), GMP_RNDN));
          ul1 = mpfr_get_ui (mpc_realref(z2), GMP_RNDN);
          ul2 = mpfr_get_ui (mpc_imagref(z2), GMP_RNDN);
          mpc_set_prec (z2, prec);
          mpc_set_prec (z3, prec);
          mpc_set_prec (zzzz, 4*prec);
          switch (special)
            {
            case 1:
              mpfr_set_ui (mpc_realref (z1), 0, GMP_RNDN);
              break;
            case 2:
              mpfr_set_inf (mpc_realref (z1), +1);
              break;
            case 3:
              mpfr_set_ui (mpc_imagref (z1), 0, GMP_RNDN);
              break;
            case 4:
              mpfr_set_inf (mpc_imagref (z1), -1);
              break;
            case 5:
              ul1 = 0;
              break;
            case 6:
              ul2 = 0;
              break;
            }
          break;
        case CCS:
          mpc_set_prec (z2, 128);
          do {
            test_default_random (z2, 0, 64, 128, zero_probability);
          } while (!mpfr_fits_slong_p (mpc_realref (z2), GMP_RNDN));
          lo = mpfr_get_si (mpc_realref(z2), GMP_RNDN);
          mpc_set_prec (z2, prec);
          mpc_set_prec (z3, prec);
          mpc_set_prec (zzzz, 4*prec);
          switch (special)
            {
            case 1:
              mpfr_set_ui (mpc_realref (z1), 0, GMP_RNDN);
              break;
            case 2:
              mpfr_set_inf (mpc_realref (z1), +1);
              break;
            case 3:
              mpfr_set_ui (mpc_imagref (z1), 0, GMP_RNDN);
              break;
            case 4:
              mpfr_set_inf (mpc_imagref (z1), -1);
              break;
            case 5:
              lo = 0;
              break;
            }
          break;
        case CCI:
          mpc_set_prec (z2, 128);
          do {
            test_default_random (z2, 0, 64, 128, zero_probability);
          } while (!mpfr_fits_slong_p (mpc_realref (z2), GMP_RNDN));
          i = (int)mpfr_get_si (mpc_realref(z2), GMP_RNDN);
          mpc_set_prec (z2, prec);
          mpc_set_prec (z3, prec);
          mpc_set_prec (zzzz, 4*prec);
          switch (special)
            {
            case 1:
              mpfr_set_ui (mpc_realref (z1), 0, GMP_RNDN);
              break;
            case 2:
              mpfr_set_inf (mpc_realref (z1), +1);
              break;
            case 3:
              mpfr_set_ui (mpc_imagref (z1), 0, GMP_RNDN);
              break;
            case 4:
              mpfr_set_inf (mpc_imagref (z1), -1);
              break;
            case 5:
              i = 0;
              break;
            }
          break;
        case CCF: case CFC:
          mpfr_set_prec (x1, prec);
          mpfr_set (x1, mpc_realref (z1), GMP_RNDN);
          test_default_random (z1, exp_min, exp_max, 128, zero_probability);
          mpc_set_prec (z2, prec);
          mpc_set_prec (z3, prec);
          mpc_set_prec (zzzz, 4*prec);
          switch (special)
            {
            case 1:
              mpfr_set_ui (mpc_realref (z1), 0, GMP_RNDN);
              break;
            case 2:
              mpfr_set_inf (mpc_realref (z1), +1);
              break;
            case 3:
              mpfr_set_ui (mpc_imagref (z1), 0, GMP_RNDN);
              break;
            case 4:
              mpfr_set_inf (mpc_imagref (z1), -1);
              break;
            case 5:
              mpfr_set_ui (x1, 0, GMP_RNDN);
              break;
            case 6:
              mpfr_set_inf (x1, +1);
              break;
            }
          break;
        case CC_C:
          mpc_set_prec (z2, prec);
          mpc_set_prec (z3, prec);
          mpc_set_prec (z4, prec);
          mpc_set_prec (z5, prec);
          mpc_set_prec (zzzz, 4*prec);
          mpc_set_prec (zzzz2, 4*prec);
          switch (special)
            {
            case 1:
              mpfr_set_ui (mpc_realref (z1), 0, GMP_RNDN);
              break;
            case 2:
              mpfr_set_inf (mpc_realref (z1), +1);
              break;
            case 3:
              mpfr_set_ui (mpc_imagref (z1), 0, GMP_RNDN);
              break;
            case 4:
              mpfr_set_inf (mpc_imagref (z1), -1);
              break;
            }
          break;
        case CC:
        default:
          mpc_set_prec (z2, prec);
          mpc_set_prec (z3, prec);
          mpc_set_prec (zzzz, 4*prec);
          switch (special)
            {
            case 1:
              mpfr_set_ui (mpc_realref (z1), 0, GMP_RNDN);
              break;
            case 2:
              mpfr_set_inf (mpc_realref (z1), +1);
              break;
            case 3:
              mpfr_set_ui (mpc_imagref (z1), 0, GMP_RNDN);
              break;
            case 4:
              mpfr_set_inf (mpc_imagref (z1), -1);
              break;
            }
        }

      for (rnd_re = first_rnd_mode (); is_valid_rnd_mode (rnd_re); rnd_re = next_rnd_mode (rnd_re))
        switch (function.type)
          {
          case C_CC:
            for (rnd_im = first_rnd_mode (); is_valid_rnd_mode (rnd_im); rnd_im = next_rnd_mode (rnd_im))
              tgeneric_c_cc (&function, z1, z2, z3, zzzz, z4,
			     MPC_RND (rnd_re, rnd_im));
            reuse_c_cc (&function, z1, z2, z3, z4);
            break;
          case CCCC:
            for (rnd_im = first_rnd_mode (); is_valid_rnd_mode (rnd_im); rnd_im = next_rnd_mode (rnd_im))
              tgeneric_cccc (&function, z1, z2, z3, z4, zzzz, z5,
                            MPC_RND (rnd_re, rnd_im));
            reuse_cccc (&function, z1, z2, z3, z4, z5);
            break;
          case FC:
            tgeneric_fc (&function, z1, x1, xxxx, x2, rnd_re);
            reuse_fc (&function, z1, z2, x1);
            break;
          case CC:
            for (rnd_im = first_rnd_mode (); is_valid_rnd_mode (rnd_im); rnd_im = next_rnd_mode (rnd_im))
              tgeneric_cc (&function, z1, z2, zzzz, z3,
                           MPC_RND (rnd_re, rnd_im));
            reuse_cc (&function, z1, z2, z3);
            break;
          case CC_C:
            for (rnd_im = first_rnd_mode (); is_valid_rnd_mode (rnd_im); rnd_im = next_rnd_mode (rnd_im))
               for (rnd2_re = first_rnd_mode (); is_valid_rnd_mode (rnd2_re); rnd2_re = next_rnd_mode (rnd2_re))
                  for (rnd2_im = first_rnd_mode (); is_valid_rnd_mode (rnd2_im); rnd2_im = next_rnd_mode (rnd2_im))
                     tgeneric_cc_c (&function, z1, z2, z3, zzzz, zzzz2, z4, z5,
                           MPC_RND (rnd_re, rnd_im), MPC_RND (rnd2_re, rnd2_im));
             reuse_cc_c (&function, z1, z2, z3, z4, z5);
            break;
          case CFC:
            for (rnd_im = first_rnd_mode (); is_valid_rnd_mode (rnd_im); rnd_im = next_rnd_mode (rnd_im))
              tgeneric_cfc (&function, x1, z1, z2, zzzz, z3,
                            MPC_RND (rnd_re, rnd_im));
            reuse_cfc (&function, z1, x1, z2, z3);
            break;
          case CCF:
            for (rnd_im = first_rnd_mode (); is_valid_rnd_mode (rnd_im); rnd_im = next_rnd_mode (rnd_im))
              tgeneric_ccf (&function, z1, x1, z2, zzzz, z3,
                            MPC_RND (rnd_re, rnd_im));
            reuse_ccf (&function, z1, x1, z2, z3);
            break;
          case CCU:
            for (rnd_im = first_rnd_mode (); is_valid_rnd_mode (rnd_im); rnd_im = next_rnd_mode (rnd_im))
              tgeneric_ccu (&function, z1, ul1, z2, zzzz, z3,
                            MPC_RND (rnd_re, rnd_im));
            reuse_ccu (&function, z1, ul1, z2, z3);
            break;
          case CUC:
            for (rnd_im = first_rnd_mode (); is_valid_rnd_mode (rnd_im); rnd_im = next_rnd_mode (rnd_im))
              tgeneric_cuc (&function, ul1, z1, z2, zzzz, z3,
                            MPC_RND (rnd_re, rnd_im));
            reuse_cuc (&function, ul1, z1, z2, z3);
            break;
          case CCS:
            for (rnd_im = first_rnd_mode (); is_valid_rnd_mode (rnd_im); rnd_im = next_rnd_mode (rnd_im))
              tgeneric_ccs (&function, z1, lo, z2, zzzz, z3,
                            MPC_RND (rnd_re, rnd_im));
            reuse_ccs (&function, z1, lo, z2, z3);
            break;
          case CCI:
            for (rnd_im = first_rnd_mode (); is_valid_rnd_mode (rnd_im); rnd_im = next_rnd_mode (rnd_im))
              tgeneric_cci (&function, z1, i, z2, zzzz, z3,
                            MPC_RND (rnd_re, rnd_im));
            reuse_cci (&function, z1, i, z2, z3);
            break;
          case CUUC:
            for (rnd_im = first_rnd_mode (); is_valid_rnd_mode (rnd_im); rnd_im = next_rnd_mode (rnd_im))
              tgeneric_cuuc (&function, ul1, ul2, z1, z2, zzzz, z3,
                             MPC_RND (rnd_re, rnd_im));
            reuse_cuuc (&function, ul1, ul2, z1, z2, z3);
            break;
          default:
            printf ("tgeneric not yet implemented for this kind of"
                    "function\n");
            exit (1);
          }
    }

  mpc_clear (z1);
  switch (function.type)
    {
    case C_CC:
      mpc_clear (z2);
      mpc_clear (z3);
      mpc_clear (z4);
      mpc_clear (zzzz);
      break;
    case CCCC:
      mpc_clear (z2);
      mpc_clear (z3);
      mpc_clear (z4);
      mpc_clear (z5);
      mpc_clear (zzzz);
      break;
    case FC:
      mpc_clear (z2);
      mpfr_clear (x1);
      mpfr_clear (x2);
      mpfr_clear (xxxx);
      break;
    case CCF: case CFC:
      mpfr_clear (x1);
      mpc_clear (z2);
      mpc_clear (z3);
      mpc_clear (zzzz);
      break;
    case CC_C:
      mpc_clear (z2);
      mpc_clear (z3);
      mpc_clear (z4);
      mpc_clear (z5);
      mpc_clear (zzzz);
      mpc_clear (zzzz2);
      break;
    case CUUC:
    case CCI: case CCS:
    case CCU: case CUC:
    case CC:
    default:
      mpc_clear (z2);
      mpc_clear (z3);
      mpc_clear (zzzz);
    }
}
