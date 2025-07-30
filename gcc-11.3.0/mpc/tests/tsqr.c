/* tsqr -- test file for mpc_sqr.

Copyright (C) 2002, 2005, 2008, 2010, 2011 INRIA

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

#include <stdlib.h>
#include "mpc-tests.h"

static void
cmpsqr (mpc_srcptr x, mpc_rnd_t rnd)
   /* computes the square of x with the specific function or by simple     */
   /* multiplication using the rounding mode rnd and compares the results  */
   /* and return values.                                                   */
   /* In our current test suite, the real and imaginary parts of x have    */
   /* the same precision, and we use this precision also for the result.   */
   /* Furthermore, we check whether computing the square in the same       */
   /* place yields the same result.                                        */
   /* We also compute the result with four times the precision and check   */
   /* whether the rounding is correct. Error reports in this part of the   */
   /* algorithm might still be wrong, though, since there are two          */
   /* consecutive roundings.                                               */
{
  mpc_t z, t, u;
  int   inexact_z, inexact_t;

  mpc_init2 (z, MPC_MAX_PREC (x));
  mpc_init2 (t, MPC_MAX_PREC (x));
  mpc_init2 (u, 4 * MPC_MAX_PREC (x));

  inexact_z = mpc_sqr (z, x, rnd);
  inexact_t = mpc_mul (t, x, x, rnd);

  if (mpc_cmp (z, t))
    {
      fprintf (stderr, "sqr and mul differ for rnd=(%s,%s) \nx=",
               mpfr_print_rnd_mode(MPC_RND_RE(rnd)),
               mpfr_print_rnd_mode(MPC_RND_IM(rnd)));
      mpc_out_str (stderr, 2, 0, x, MPC_RNDNN);
      fprintf (stderr, "\nmpc_sqr gives ");
      mpc_out_str (stderr, 2, 0, z, MPC_RNDNN);
      fprintf (stderr, "\nmpc_mul gives ");
      mpc_out_str (stderr, 2, 0, t, MPC_RNDNN);
      fprintf (stderr, "\n");
      exit (1);
    }
  if (inexact_z != inexact_t)
    {
      fprintf (stderr, "The return values of sqr and mul differ for rnd=(%s,%s) \nx=  ",
               mpfr_print_rnd_mode(MPC_RND_RE(rnd)),
               mpfr_print_rnd_mode(MPC_RND_IM(rnd)));
      mpc_out_str (stderr, 2, 0, x, MPC_RNDNN);
      fprintf (stderr, "\nx^2=");
      mpc_out_str (stderr, 2, 0, z, MPC_RNDNN);
      fprintf (stderr, "\nmpc_sqr gives %i", inexact_z);
      fprintf (stderr, "\nmpc_mul gives %i", inexact_t);
      fprintf (stderr, "\n");
      exit (1);
    }

  mpc_set (t, x, MPC_RNDNN);
  inexact_t = mpc_sqr (t, t, rnd);
  if (mpc_cmp (z, t))
    {
      fprintf (stderr, "sqr and sqr in place differ for rnd=(%s,%s) \nx=",
               mpfr_print_rnd_mode(MPC_RND_RE(rnd)),
               mpfr_print_rnd_mode(MPC_RND_IM(rnd)));
      mpc_out_str (stderr, 2, 0, x, MPC_RNDNN);
      fprintf (stderr, "\nmpc_sqr          gives ");
      mpc_out_str (stderr, 2, 0, z, MPC_RNDNN);
      fprintf (stderr, "\nmpc_sqr in place gives ");
      mpc_out_str (stderr, 2, 0, t, MPC_RNDNN);
      fprintf (stderr, "\n");
      exit (1);
    }
  if (inexact_z != inexact_t)
    {
      fprintf (stderr, "The return values of sqr and sqr in place differ for rnd=(%s,%s) \nx=  ",
               mpfr_print_rnd_mode(MPC_RND_RE(rnd)),
               mpfr_print_rnd_mode(MPC_RND_IM(rnd)));
      mpc_out_str (stderr, 2, 0, x, MPC_RNDNN);
      fprintf (stderr, "\nx^2=");
      mpc_out_str (stderr, 2, 0, z, MPC_RNDNN);
      fprintf (stderr, "\nmpc_sqr          gives %i", inexact_z);
      fprintf (stderr, "\nmpc_sqr in place gives %i", inexact_t);
      fprintf (stderr, "\n");
      exit (1);
    }

  mpc_sqr (u, x, rnd);
  mpc_set (t, u, rnd);
  if (mpc_cmp (z, t))
    {
      fprintf (stderr, "rounding in sqr might be incorrect for rnd=(%s,%s) \nx=",
               mpfr_print_rnd_mode(MPC_RND_RE(rnd)),
               mpfr_print_rnd_mode(MPC_RND_IM(rnd)));
      mpc_out_str (stderr, 2, 0, x, MPC_RNDNN);
      fprintf (stderr, "\nmpc_sqr                     gives ");
      mpc_out_str (stderr, 2, 0, z, MPC_RNDNN);
      fprintf (stderr, "\nmpc_sqr quadruple precision gives ");
      mpc_out_str (stderr, 2, 0, u, MPC_RNDNN);
      fprintf (stderr, "\nand is rounded to                 ");
      mpc_out_str (stderr, 2, 0, t, MPC_RNDNN);
      fprintf (stderr, "\n");
      exit (1);
    }

  mpc_clear (z);
  mpc_clear (t);
  mpc_clear (u);
}


static void
testsqr (long a, long b, mpfr_prec_t prec, mpc_rnd_t rnd)
{
  mpc_t x;

  mpc_init2 (x, prec);

  mpc_set_si_si (x, a, b, rnd);

  cmpsqr (x, rnd);

  mpc_clear (x);
}


static void
reuse_bug (void)
{
  mpc_t z1;

  /* reuse bug found by Paul Zimmermann 20081021 */
  mpc_init2 (z1, 2);
  /* RE (z1^2) overflows, IM(z^2) = -0 */
  mpfr_set_str (mpc_realref (z1), "0.11", 2, GMP_RNDN);
  mpfr_mul_2si (mpc_realref (z1), mpc_realref (z1), mpfr_get_emax (), GMP_RNDN);
  mpfr_set_ui (mpc_imagref (z1), 0, GMP_RNDN);
  mpc_conj (z1, z1, MPC_RNDNN);
  mpc_sqr (z1, z1, MPC_RNDNN);
  if (!mpfr_inf_p (mpc_realref (z1)) || mpfr_signbit (mpc_realref (z1))
      ||!mpfr_zero_p (mpc_imagref (z1)) || !mpfr_signbit (mpc_imagref (z1)))
    {
      printf ("Error: Regression, bug 20081021 reproduced\n");
      MPC_OUT (z1);
      exit (1);
    }

  mpc_clear (z1);
}

int
main (void)
{
  DECL_FUNC (CC, f, mpc_sqr);
  test_start ();

  testsqr (247, -65, 8, 24);
  testsqr (5, -896, 3, 2);
  testsqr (-3, -512, 2, 16);
  testsqr (266013312, 121990769, 27, 0);
  testsqr (170, 9, 8, 0);
  testsqr (768, 85, 8, 16);
  testsqr (145, 1816, 8, 24);
  testsqr (0, 1816, 8, 24);
  testsqr (145, 0, 8, 24);

  data_check (f, "sqr.dat");
  tgeneric (f, 2, 1024, 1, 0);

  reuse_bug ();

  test_end ();

  return 0;
}
