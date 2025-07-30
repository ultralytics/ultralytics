/* tpow_ui -- test file for mpc_pow_ui.

Copyright (C) 2009, 2010 INRIA

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

#include <limits.h> /* for CHAR_BIT */
#include "mpc-tests.h"

static void
compare_mpc_pow (mpfr_prec_t pmax, int iter, unsigned long nbits)
{
  mpfr_prec_t p;
  mpc_t x, y, z, t;
  unsigned long n;
  int i, inex_pow, inex_pow_ui;
  mpc_rnd_t rnd;

  mpc_init3 (y, sizeof (unsigned long) * CHAR_BIT, MPFR_PREC_MIN);
  for (p = MPFR_PREC_MIN; p <= pmax; p++)
    for (i = 0; i < iter; i++)
      {
        mpc_init2 (x, p);
        mpc_init2 (z, p);
        mpc_init2 (t, p);
        mpc_urandom (x, rands);
        n = gmp_urandomb_ui (rands, nbits); /* 0 <= n < 2^nbits */
        mpc_set_ui (y, n, MPC_RNDNN);
        for (rnd = 0; rnd < 16; rnd ++)
          {
            inex_pow = mpc_pow (z, x, y, rnd);
            inex_pow_ui = mpc_pow_ui (t, x, n, rnd);
            if (mpc_cmp (z, t) != 0)
              {
                printf ("mpc_pow and mpc_pow_ui differ for x=");
                mpc_out_str (stdout, 10, 0, x, MPC_RNDNN);
                printf (" n=%lu\n", n);
                printf ("mpc_pow gives ");
                mpc_out_str (stdout, 10, 0, z, MPC_RNDNN);
                printf ("\nmpc_pow_ui gives ");
                mpc_out_str (stdout, 10, 0, t, MPC_RNDNN);
                printf ("\n");
                exit (1);
              }
            if (inex_pow != inex_pow_ui)
              {
                printf ("mpc_pow and mpc_pow_ui give different flags for x=");
                mpc_out_str (stdout, 10, 0, x, MPC_RNDNN);
                printf (" n=%lu\n", n);
                printf ("mpc_pow gives %d\n", inex_pow);
                printf ("mpc_pow_ui gives %d\n", inex_pow_ui);
                exit (1);
              }
          }
        mpc_clear (x);
        mpc_clear (z);
        mpc_clear (t);
      }
  mpc_clear (y);
}

int
main (int argc, char *argv[])
{
  mpc_t z;

  DECL_FUNC (CCU, f, mpc_pow_ui);

  if (argc != 1)
    {
      mpfr_prec_t p;
      long int n, k;
      mpc_t res;
      if (argc != 3 && argc != 4)
        {
          printf ("Usage: tpow_ui precision exponent [k]\n");
          exit (1);
        }
      p = atoi (argv[1]);
      n = atoi (argv[2]);
      MPC_ASSERT (n >= 0);
      k = (argc > 3) ? atoi (argv[3]) : 1;
      MPC_ASSERT (k >= 0);
      mpc_init2 (z, p);
      mpc_init2 (res, p);
      mpfr_const_pi (mpc_realref (z), GMP_RNDN);
      mpfr_div_2exp (mpc_realref (z), mpc_realref (z), 2, GMP_RNDN);
      mpfr_const_log2 (mpc_imagref (z), GMP_RNDN);
      while (k--)
        mpc_pow_ui (res, z, (unsigned long int) n, MPC_RNDNN);
      mpc_clear (z);
      mpc_clear (res);
      return 0;
    }

  test_start ();
  data_check (f, "pow_ui.dat");

  compare_mpc_pow (100, 5, 19);

  test_end ();

  return 0;
}
