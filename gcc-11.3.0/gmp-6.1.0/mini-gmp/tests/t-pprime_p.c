/* test mpz_probab_prime_p

Copyright 2001, 2002, 2004, 2011, 2012, 2014 Free Software Foundation, Inc.

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

#include "testutils.h"

static int
isprime (unsigned long int t)
{
  unsigned long int q, r, d;

  if (t < 32)
    return (0xa08a28acUL >> t) & 1;
  if ((t & 1) == 0)
    return 0;

  if (t % 3 == 0)
    return 0;
  if (t % 5 == 0)
    return 0;
  if (t % 7 == 0)
    return 0;

  for (d = 11;;)
    {
      q = t / d;
      r = t - q * d;
      if (q < d)
	return 1;
      if (r == 0)
	break;
      d += 2;
      q = t / d;
      r = t - q * d;
      if (q < d)
	return 1;
      if (r == 0)
	break;
      d += 4;
    }
  return 0;
}

static void
check_one (mpz_srcptr n, int want)
{
  int  got;

  got = mpz_probab_prime_p (n, 25);

  /* "definitely prime" is fine if we only wanted "probably prime" */
  if (got == 2 && want == 1)
    want = 2;

  if (got != want)
    {
      printf ("mpz_probab_prime_p\n");
      dump   ("  n    ", n);
      printf ("  got =%d", got);
      printf ("  want=%d\n", want);
      abort ();
    }
}

static void
check_pn (mpz_ptr n, int want)
{
  check_one (n, want);
  mpz_neg (n, n);
  check_one (n, want);
}

static void
check_small (void)
{
  mpz_t  n;
  long   i;

  mpz_init (n);

  for (i = 0; i < 1700; i++)
    {
      mpz_set_si (n, i);
      check_pn (n, isprime (i));
    }

  mpz_clear (n);
}

void
check_composites (void)
{
  int i;
  int reps = 1000;
  mpz_t a, b, n, bs;
  unsigned long size_range, size;

  mpz_init (a);
  mpz_init (b);
  mpz_init (n);
  mpz_init (bs);

  for (i = 0; i < reps; i++)
    {
      mini_urandomb (bs, 32);
      size_range = mpz_get_ui (bs) % 12 + 1; /* 0..4096 bit operands */

      mini_urandomb (bs, size_range);
      size = mpz_get_ui (bs);
      mini_rrandomb (a, size);

      mini_urandomb (bs, 32);
      size_range = mpz_get_ui (bs) % 12 + 1; /* 0..4096 bit operands */
      mini_rrandomb (b, size);

      /* Exclude trivial factors */
      if (mpz_cmp_ui (a, 1) == 0)
	mpz_set_ui (a, 2);
      if (mpz_cmp_ui (b, 1) == 0)
	mpz_set_ui (b, 2);

      mpz_mul (n, a, b);

      check_pn (n, 0);
    }
  mpz_clear (a);
  mpz_clear (b);
  mpz_clear (n);
  mpz_clear (bs);
}

static void
check_primes (void)
{
  static const char * const primes[] = {
    "2", "17", "65537",
    /* diffie-hellman-group1-sha1, also "Well known group 2" in RFC
       2412, 2^1024 - 2^960 - 1 + 2^64 * { [2^894 pi] + 129093 } */
    "0xFFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1"
    "29024E088A67CC74020BBEA63B139B22514A08798E3404DD"
    "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245"
    "E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
    "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE65381"
    "FFFFFFFFFFFFFFFF",
    NULL
  };

  mpz_t n;
  int i;

  mpz_init (n);

  for (i = 0; primes[i]; i++)
    {
      mpz_set_str_or_abort (n, primes[i], 0);
      check_one (n, 1);
    }
  mpz_clear (n);
}

void
testmain (int argc, char *argv[])
{
  check_small ();
  check_composites ();
  check_primes ();
}
