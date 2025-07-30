/* Test mpz_perfect_power_p.

   Contributed to the GNU project by Torbjorn Granlund and Martin Boij.

Copyright 2008-2010, 2014 Free Software Foundation, Inc.

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

struct
{
  const char *num_as_str;
  char want;
} static tests[] =
  {
    { "0", 1},
    { "1", 1},
    {"-1", 1},
    { "2", 0},
    {"-2", 0},
    { "3", 0},
    {"-3", 0},
    { "4", 1},
    {"-4", 0},
    { "64", 1},
    {"-64", 1},
    { "128", 1},
    {"-128", 1},
    { "256", 1},
    {"-256", 0},
    { "512", 1},
    {"-512", 1},
    { "0x4000000", 1},
    {"-0x4000000", 1},
    { "0x3cab640", 1},
    {"-0x3cab640", 0},
    { "0x3e23840", 1},
    {"-0x3e23840", 0},
    { "0x3d3a7ed1", 1},
    {"-0x3d3a7ed1", 1},
    { "0x30a7a6000", 1},
    {"-0x30a7a6000", 1},
    { "0xf33e5a5a59", 1},
    {"-0xf33e5a5a59", 0},
    { "0xed1b1182118135d", 1},
    {"-0xed1b1182118135d", 1},
    { "0xe71f6eb7689cc276b2f1", 1},
    {"-0xe71f6eb7689cc276b2f1", 0},
    { "0x12644507fe78cf563a4b342c92e7da9fe5e99cb75a01", 1},
    {"-0x12644507fe78cf563a4b342c92e7da9fe5e99cb75a01", 0},
    { "0x1ff2e7c581bb0951df644885bd33f50e472b0b73a204e13cbe98fdb424d66561e4000000", 1},
    {"-0x1ff2e7c581bb0951df644885bd33f50e472b0b73a204e13cbe98fdb424d66561e4000000", 1},
    { "0x2b9b44db2d91a6f8165c8c7339ef73633228ea29e388592e80354e4380004aad84000000", 1},
    {"-0x2b9b44db2d91a6f8165c8c7339ef73633228ea29e388592e80354e4380004aad84000000", 1},
    { "0x28d5a2b8f330910a9d3cda06036ae0546442e5b1a83b26a436efea5b727bf1bcbe7e12b47d81", 1},
    {"-0x28d5a2b8f330910a9d3cda06036ae0546442e5b1a83b26a436efea5b727bf1bcbe7e12b47d81", 1},
    {NULL, 0}
  };


void
check_tests ()
{
  mpz_t x;
  int i;
  int got, want;

  mpz_init (x);

  for (i = 0; tests[i].num_as_str != NULL; i++)
    {
      mpz_set_str (x, tests[i].num_as_str, 0);
      got = mpz_perfect_power_p (x);
      want = tests[i].want;
      if (got != want)
	{
	  fprintf (stderr, "mpz_perfect_power_p returns %d when %d was expected\n", got, want);
	  fprintf (stderr, "fault operand: %s\n", tests[i].num_as_str);
	  abort ();
	}
    }

  mpz_clear (x);
}

#define NRP 15

void
check_random (int reps)
{
  mpz_t n, np, temp, primes[NRP];
  int i, j, k, unique, destroy, res;
  unsigned long int nrprimes, primebits;
  mp_limb_t g, exp[NRP], e;
  gmp_randstate_ptr rands;

  rands = RANDS;

  mpz_init (n);
  mpz_init (np);
  mpz_init (temp);

  for (i = 0; i < NRP; i++)
    mpz_init (primes[i]);

  for (i = 0; i < reps; i++)
    {
      mpz_urandomb (np, rands, 32);
      nrprimes = mpz_get_ui (np) % NRP + 1; /* 1-NRP unique primes */

      mpz_urandomb (np, rands, 32);
      g = mpz_get_ui (np) % 32 + 2; /* gcd 2-33 */

      for (j = 0; j < nrprimes;)
	{
	  mpz_urandomb (np, rands, 32);
	  primebits = mpz_get_ui (np) % 100 + 3; /* 3-102 bit primes */
	  mpz_urandomb (primes[j], rands, primebits);
	  mpz_nextprime (primes[j], primes[j]);
	  unique = 1;
	  for (k = 0; k < j; k++)
	    {
	      if (mpz_cmp (primes[j], primes[k]) == 0)
		{
		  unique = 0;
		  break;
		}
	    }
	  if (unique)
	    {
	      mpz_urandomb (np, rands, 32);
	      e = 371 / (10 * primebits) + mpz_get_ui (np) % 11 + 1; /* Magic constants */
	      exp[j++] = g * e;
	    }
	}

      if (nrprimes > 1)
	{
	  /* Destroy d exponents, d in [1, nrprimes - 1] */
	  if (nrprimes == 2)
	    {
	      destroy = 1;
	    }
	  else
	    {
	      mpz_urandomb (np, rands, 32);
	      destroy = mpz_get_ui (np) % (nrprimes - 2);
	    }

	  g = exp[destroy];
	  for (k = destroy + 1; k < nrprimes; k++)
	    g = mpn_gcd_1 (&g, 1, exp[k]);

	  for (j = 0; j < destroy; j++)
	    {
	      mpz_urandomb (np, rands, 32);
	      e = mpz_get_ui (np) % 50 + 1;
	      while (mpn_gcd_1 (&g, 1, e) > 1)
		e++;

	      exp[j] = e;
	    }
	}

      /* Compute n */
      mpz_pow_ui (n, primes[0], exp[0]);
      for (j = 1; j < nrprimes; j++)
	{
	  mpz_pow_ui (temp, primes[j], exp[j]);
	  mpz_mul (n, n, temp);
	}

      res = mpz_perfect_power_p (n);

      if (nrprimes == 1)
	{
	if (res == 0 && exp[0] > 1)
	  {
	    printf("n is a perfect power, perfpow_p disagrees\n");
	    gmp_printf("n = %Zu\nprimes[0] = %Zu\nexp[0] = %lu\n", n, primes[0], exp[0]);
	    abort ();
	  }
	else if (res == 1 && exp[0] == 1)
	  {
	    gmp_printf("n = %Zu\n", n);
	    printf("n is now a prime number, but perfpow_p still believes n is a perfect power\n");
	    abort ();
	  }
	}
      else
	{
	  if (res == 1 && destroy != 0)
	    {
	      gmp_printf("n = %Zu\nn was destroyed, but perfpow_p still believes n is a perfect power\n", n);
	      abort ();
	    }
	  else if (res == 0 && destroy == 0)
	    {
	      gmp_printf("n = %Zu\nn is a perfect power, perfpow_p disagrees\n", n);
	      abort ();
	    }
	}
    }

  mpz_clear (n);
  mpz_clear (np);
  mpz_clear (temp);
  for (i = 0; i < NRP; i++)
    mpz_clear (primes[i]);
}

int
main (int argc, char **argv)
{
  int n_tests;

  tests_start ();
  mp_trace_base = -16;

  check_tests ();

  n_tests = 500;
  if (argc == 2)
    n_tests = atoi (argv[1]);
  check_random (n_tests);

  tests_end ();
  exit (0);
}
