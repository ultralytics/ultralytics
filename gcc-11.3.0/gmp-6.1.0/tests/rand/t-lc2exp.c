/* Exercise the lc2exp random functions.

Copyright 2002, 2011 Free Software Foundation, Inc.

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


/* a=0 and c=0 produces zero results always. */
void
check_zero (unsigned long m2exp)
{
  gmp_randstate_t  r;
  mpz_t            a;
  unsigned long    c;
  int              i;

  mpz_init_set_ui (a, 0L);
  c = 0L;

  gmp_randinit_lc_2exp (r, a, c, m2exp);
  gmp_randseed_ui (r, 0L);

  for (i = 0; i < 5; i++)
    {
      mpz_urandomb (a, r, 123L);
      if (mpz_sgn (a) != 0)
        {
          printf ("check_zero m2exp=%lu: didn't get zero\n", m2exp);
          gmp_printf ("  rand=%#Zx\n", a);
          abort ();
        }
    }

  mpz_clear (a);
  gmp_randclear (r);
}

/* negative a */
void
check_nega (void)
{
  gmp_randstate_t  r;
  mpz_t            a;
  unsigned long    c, m2exp;
  int              i;

  mpz_init (a);
  mpz_setbit (a, 1000L);
  mpz_neg (a, a);
  c = 0L;
  m2exp = 45L;

  gmp_randinit_lc_2exp (r, a, c, m2exp);
  gmp_randseed_ui (r, 0L);

  for (i = 0; i < 5; i++)
    {
      mpz_urandomb (a, r, 123L);
      if (mpz_sgn (a) != 0)
        printf ("check_nega m2exp=%lu: didn't get zero\n", m2exp);
    }

  mpz_clear (a);
  gmp_randclear (r);
}

void
check_bigc (void)
{
  gmp_randstate_t  r;
  mpz_t            a;
  unsigned long    c, m2exp, bits;
  int              i;

  mpz_init_set_ui (a, 0L);
  c = ULONG_MAX;
  m2exp = 8;

  gmp_randinit_lc_2exp (r, a, c, m2exp);
  gmp_randseed_ui (r, 0L);

  for (i = 0; i < 20; i++)
    {
      bits = 123L;
      mpz_urandomb (a, r, bits);
      if (mpz_sgn (a) < 0 || mpz_sizeinbase (a, 2) > bits)
        {
          printf     ("check_bigc: mpz_urandomb out of range\n");
          printf     ("   m2exp=%lu\n", m2exp);
          gmp_printf ("   rand=%#ZX\n", a);
          gmp_printf ("   sizeinbase2=%u\n", mpz_sizeinbase (a, 2));
	  abort ();
        }
    }

  mpz_clear (a);
  gmp_randclear (r);
}

void
check_bigc1 (void)
{
  gmp_randstate_t  r;
  mpz_t            a;
  unsigned long    c, m2exp;
  int              i;

  mpz_init_set_ui (a, 0L);
  c = ULONG_MAX;
  m2exp = 2;

  gmp_randinit_lc_2exp (r, a, c, m2exp);
  gmp_randseed_ui (r, 0L);

  for (i = 0; i < 20; i++)
    {
      mpz_urandomb (a, r, 1L);
      if (mpz_cmp_ui (a, 1L) != 0)
        {
          printf     ("check_bigc1: mpz_urandomb didn't give 1\n");
          printf     ("   m2exp=%lu\n", m2exp);
          gmp_printf ("   got rand=%#ZX\n", a);
          abort ();
        }
    }

  mpz_clear (a);
  gmp_randclear (r);
}

/* Checks parameters which triggered an assertion failure in the past.
   Happened when limbs(a)+limbs(c) < bits_to_limbs(m2exp).  */
void
check_bigm (void)
{
  gmp_randstate_t rstate;
  mpz_t a;

  mpz_init_set_ui (a, 5L);
  gmp_randinit_lc_2exp (rstate, a, 1L, 384L);

  mpz_urandomb (a, rstate, 20L);

  gmp_randclear (rstate);
  mpz_clear (a);
}

/* Checks for seeds bigger than the modulus.  */
void
check_bigs (void)
{
  gmp_randstate_t rstate;
  mpz_t sd, a;
  int i;

  mpz_init (sd);
  mpz_setbit (sd, 300L);
  mpz_sub_ui (sd, sd, 1L);
  mpz_clrbit (sd, 13L);
  mpz_init_set_ui (a, 123456789L);

  gmp_randinit_lc_2exp (rstate, a, 5L, 64L);

  for (i = 0; i < 20; i++)
    {
      mpz_neg (sd, sd);
      gmp_randseed (rstate, sd);
      mpz_mul_ui (sd, sd, 7L);

      mpz_urandomb (a, rstate, 80L);
    }

  gmp_randclear (rstate);
  mpz_clear (a);
  mpz_clear (sd);
}

int
main (void)
{
  tests_start ();

  check_zero (2L);
  check_zero (7L);
  check_zero (32L);
  check_zero (64L);
  check_zero (1000L);

  check_nega ();
  check_bigc ();
  check_bigc1 ();

  check_bigm ();
  check_bigs ();

  tests_end ();
  exit (0);
}
