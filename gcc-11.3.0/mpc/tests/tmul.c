/* tmul -- test file for mpc_mul.

Copyright (C) 2002, 2005, 2008, 2009, 2010, 2011, 2012 INRIA

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
#ifdef TIMING
#include <sys/times.h>
#endif
#include "mpc-tests.h"

static void
cmpmul (mpc_srcptr x, mpc_srcptr y, mpc_rnd_t rnd)
   /* computes the product of x and y with the naive and Karatsuba methods */
   /* using the rounding mode rnd and compares the results and return      */
   /* values.                                                              */
   /* In our current test suite, the real and imaginary parts of x and y   */
   /* all have the same precision, and we use this precision also for the  */
   /* result.                                                              */
{
   mpc_t z, t;
   int   inex_z, inex_t;

   mpc_init2 (z, MPC_MAX_PREC (x));
   mpc_init2 (t, MPC_MAX_PREC (x));

   inex_z = mpc_mul_naive (z, x, y, rnd);
   inex_t = mpc_mul_karatsuba (t, x, y, rnd);

   if (mpc_cmp (z, t) != 0 || inex_z != inex_t) {
      fprintf (stderr, "mul_naive and mul_karatsuba differ for rnd=(%s,%s)\n",
               mpfr_print_rnd_mode(MPC_RND_RE(rnd)),
               mpfr_print_rnd_mode(MPC_RND_IM(rnd)));
      MPC_OUT (x);
      MPC_OUT (y);
      MPC_OUT (z);
      MPC_OUT (t);
      if (inex_z != inex_t) {
         fprintf (stderr, "inex_re (z): %s\n", MPC_INEX_STR (inex_z));
         fprintf (stderr, "inex_re (t): %s\n", MPC_INEX_STR (inex_t));
      }
      exit (1);
   }

   mpc_clear (z);
   mpc_clear (t);
}


static void
testmul (long a, long b, long c, long d, mpfr_prec_t prec, mpc_rnd_t rnd)
{
  mpc_t x, y;

  mpc_init2 (x, prec);
  mpc_init2 (y, prec);

  mpc_set_si_si (x, a, b, rnd);
  mpc_set_si_si (y, c, d, rnd);

  cmpmul (x, y, rnd);

  mpc_clear (x);
  mpc_clear (y);
}


static void
check_regular (void)
{
  mpc_t x, y;
  int rnd_re, rnd_im;
  mpfr_prec_t prec;

  testmul (247, -65, -223, 416, 8, 24);
  testmul (5, -896, 5, -32, 3, 2);
  testmul (-3, -512, -1, -1, 2, 16);
  testmul (266013312, 121990769, 110585572, 116491059, 27, 0);
  testmul (170, 9, 450, 251, 8, 0);
  testmul (768, 85, 169, 440, 8, 16);
  testmul (145, 1816, 848, 169, 8, 24);

  mpc_init2 (x, 1000);
  mpc_init2 (y, 1000);

  /* Bug 20081114: mpc_mul_karatsuba returned wrong inexact value for
     imaginary part */
  mpc_set_prec (x, 7);
  mpc_set_prec (y, 7);
  mpfr_set_str (mpc_realref (x), "0xB4p+733", 16, GMP_RNDN);
  mpfr_set_str (mpc_imagref (x), "0x90p+244", 16, GMP_RNDN);
  mpfr_set_str (mpc_realref (y), "0xECp-146", 16, GMP_RNDN);
  mpfr_set_str (mpc_imagref (y), "0xACp-471", 16, GMP_RNDN);
  cmpmul (x, y, MPC_RNDNN);
  mpfr_set_str (mpc_realref (x), "0xB4p+733", 16, GMP_RNDN);
  mpfr_set_str (mpc_imagref (x), "0x90p+244", 16, GMP_RNDN);
  mpfr_set_str (mpc_realref (y), "0xACp-471", 16, GMP_RNDN);
  mpfr_set_str (mpc_imagref (y), "-0xECp-146", 16, GMP_RNDN);
  cmpmul (x, y, MPC_RNDNN);

  for (prec = 2; prec < 1000; prec = (mpfr_prec_t) (prec * 1.1 + 1))
    {
      mpc_set_prec (x, prec);
      mpc_set_prec (y, prec);

      test_default_random (x, -1024, 1024, 128, 0);
      test_default_random (y, -1024, 1024, 128, 0);

      for (rnd_re = 0; rnd_re < 4; rnd_re ++)
        for (rnd_im = 0; rnd_im < 4; rnd_im ++)
          cmpmul (x, y, MPC_RND (rnd_re, rnd_im));
    }

  mpc_clear (x);
  mpc_clear (y);
}


#ifdef TIMING
static void
timemul (void)
{
  /* measures the time needed with different precisions for naive and */
  /* Karatsuba multiplication                                         */

  mpc_t             x, y, z;
  unsigned long int i, j;
  const unsigned long int tests = 10000;
  struct tms        time_old, time_new;
  double            passed1, passed2;

  mpc_init (x);
  mpc_init (y);
  mpc_init_set_ui_ui (z, 1, 0, MPC_RNDNN);

  for (i = 1; i < 50; i++)
    {
      mpc_set_prec (x, i * BITS_PER_MP_LIMB);
      mpc_set_prec (y, i * BITS_PER_MP_LIMB);
      mpc_set_prec (z, i * BITS_PER_MP_LIMB);
      test_default_random (x, -1, 1, 128, 25);
      test_default_random (y, -1, 1, 128, 25);

      times (&time_old);
      for (j = 0; j < tests; j++)
        mpc_mul_naive (z, x, y, MPC_RNDNN);
      times (&time_new);
      passed1 = ((double) (time_new.tms_utime - time_old.tms_utime)) / 100;

      times (&time_old);
      for (j = 0; j < tests; j++)
        mpc_mul_karatsuba (z, x, y, MPC_RNDNN);
      times (&time_new);
      passed2 = ((double) (time_new.tms_utime - time_old.tms_utime)) / 100;

      printf ("Time for %3li limbs naive/Karatsuba: %5.2f %5.2f\n", i,
              passed1, passed2);
    }

  mpc_clear (x);
  mpc_clear (y);
  mpc_clear (z);
}
#endif


int
main (void)
{
  DECL_FUNC (C_CC, f, mpc_mul);
  f.properties = FUNC_PROP_SYMETRIC;

  test_start ();

#ifdef TIMING
  timemul ();
#endif

  check_regular ();

  data_check (f, "mul.dat");
  tgeneric (f, 2, 4096, 41, 100);

  test_end ();
  return 0;
}
