/* Test mpz_perfect_square_p.

Copyright 2000-2002 Free Software Foundation, Inc.

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

#include <stdio.h>
#include <stdlib.h>

#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"

#include "mpn/perfsqr.h"


/* check_modulo() exercises mpz_perfect_square_p on squares which cover each
   possible quadratic residue to each divisor used within
   mpn_perfect_square_p, ensuring those residues aren't incorrectly claimed
   to be non-residues.

   Each divisor is taken separately.  It's arranged that n is congruent to 0
   modulo the other divisors, 0 of course being a quadratic residue to any
   modulus.

   The values "(j*others)^2" cover all quadratic residues mod divisor[i],
   but in no particular order.  j is run from 1<=j<=divisor[i] so that zero
   is excluded.  A literal n==0 doesn't reach the residue tests.  */

void
check_modulo (void)
{
  static const unsigned long  divisor[] = PERFSQR_DIVISORS;
  unsigned long  i, j;

  mpz_t  alldiv, others, n;

  mpz_init (alldiv);
  mpz_init (others);
  mpz_init (n);

  /* product of all divisors */
  mpz_set_ui (alldiv, 1L);
  for (i = 0; i < numberof (divisor); i++)
    mpz_mul_ui (alldiv, alldiv, divisor[i]);

  for (i = 0; i < numberof (divisor); i++)
    {
      /* product of all divisors except i */
      mpz_set_ui (others, 1L);
      for (j = 0; j < numberof (divisor); j++)
        if (i != j)
          mpz_mul_ui (others, others, divisor[j]);

      for (j = 1; j <= divisor[i]; j++)
        {
          /* square */
          mpz_mul_ui (n, others, j);
          mpz_mul (n, n, n);
          if (! mpz_perfect_square_p (n))
            {
              printf ("mpz_perfect_square_p got 0, want 1\n");
              mpz_trace ("  n", n);
              abort ();
            }
        }
    }

  mpz_clear (alldiv);
  mpz_clear (others);
  mpz_clear (n);
}


/* Exercise mpz_perfect_square_p compared to what mpz_sqrt says. */
void
check_sqrt (int reps)
{
  mpz_t x2, x2t, x;
  mp_size_t x2n;
  int res;
  int i;
  /* int cnt = 0; */
  gmp_randstate_ptr rands = RANDS;
  mpz_t bs;

  mpz_init (bs);

  mpz_init (x2);
  mpz_init (x);
  mpz_init (x2t);

  for (i = 0; i < reps; i++)
    {
      mpz_urandomb (bs, rands, 9);
      x2n = mpz_get_ui (bs);
      mpz_rrandomb (x2, rands, x2n);
      /* mpz_out_str (stdout, -16, x2); puts (""); */

      res = mpz_perfect_square_p (x2);
      mpz_sqrt (x, x2);
      mpz_mul (x2t, x, x);

      if (res != (mpz_cmp (x2, x2t) == 0))
        {
          printf    ("mpz_perfect_square_p and mpz_sqrt differ\n");
          mpz_trace ("   x  ", x);
          mpz_trace ("   x2 ", x2);
          mpz_trace ("   x2t", x2t);
          printf    ("   mpz_perfect_square_p %d\n", res);
          printf    ("   mpz_sqrt             %d\n", mpz_cmp (x2, x2t) == 0);
          abort ();
        }

      /* cnt += res != 0; */
    }
  /* printf ("%d/%d perfect squares\n", cnt, reps); */

  mpz_clear (bs);
  mpz_clear (x2);
  mpz_clear (x);
  mpz_clear (x2t);
}


int
main (int argc, char **argv)
{
  int reps = 200000;

  tests_start ();
  mp_trace_base = -16;

  if (argc == 2)
     reps = atoi (argv[1]);

  check_modulo ();
  check_sqrt (reps);

  tests_end ();
  exit (0);
}
