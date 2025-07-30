/* Copyright 2006, 2007, 2009, 2010 Free Software Foundation, Inc.

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


#include <stdlib.h>		/* for strtol */
#include <stdio.h>		/* for printf */

#include "gmp.h"
#include "gmp-impl.h"
#include "longlong.h"
#include "tests/tests.h"


static void
dumpy (mp_srcptr p, mp_size_t n)
{
  mp_size_t i;
  if (n > 20)
    {
      for (i = n - 1; i >= n - 4; i--)
	{
	  printf ("%0*lx", (int) (2 * sizeof (mp_limb_t)), p[i]);
	  printf (" ");
	}
      printf ("... ");
      for (i = 3; i >= 0; i--)
	{
	  printf ("%0*lx", (int) (2 * sizeof (mp_limb_t)), p[i]);
	  printf (i == 0 ? "" : " ");
	}
    }
  else
    {
      for (i = n - 1; i >= 0; i--)
	{
	  printf ("%0*lx", (int) (2 * sizeof (mp_limb_t)), p[i]);
	  printf (i == 0 ? "" : " ");
	}
    }
  puts ("");
}

static unsigned long test;

void
check_one (mp_ptr qp, mp_srcptr rp, mp_limb_t rh,
	   mp_srcptr np, mp_size_t nn, mp_srcptr dp, mp_size_t dn, const char *fname)
{
  mp_size_t qn;
  int cmp;
  mp_ptr tp;
  mp_limb_t cy = 4711;		/* silence warnings */
  TMP_DECL;

  qn = nn - dn;

  if (qn == 0)
    return;

  TMP_MARK;

  tp = TMP_ALLOC_LIMBS (nn + 1);

  if (dn >= qn)
    mpn_mul (tp, dp, dn, qp, qn);
  else
    mpn_mul (tp, qp, qn, dp, dn);

  if (rp != NULL)
    {
      cy = mpn_add_n (tp + qn, tp + qn, rp, dn);
      cmp = cy != rh || mpn_cmp (tp, np, nn) != 0;
    }
  else
    cmp = mpn_cmp (tp, np, nn - dn) != 0;

  if (cmp != 0)
    {
      printf ("\r*******************************************************************************\n");
      printf ("%s inconsistent in test %lu\n", fname, test);
      printf ("N=   "); dumpy (np, nn);
      printf ("D=   "); dumpy (dp, dn);
      printf ("Q=   "); dumpy (qp, qn);
      if (rp != NULL)
	{
	  printf ("R=   "); dumpy (rp, dn);
	  printf ("Rb=  %d, Cy=%d\n", (int) cy, (int) rh);
	}
      printf ("T=   "); dumpy (tp, nn);
      printf ("nn = %ld, dn = %ld, qn = %ld", nn, dn, qn);
      printf ("\n*******************************************************************************\n");
      abort ();
    }

  TMP_FREE;
}


/* These are *bit* sizes. */
#define SIZE_LOG 16
#define MAX_DN (1L << SIZE_LOG)
#define MAX_NN (1L << (SIZE_LOG + 1))

#define COUNT 500

mp_limb_t
random_word (gmp_randstate_ptr rs)
{
  mpz_t x;
  mp_limb_t r;
  TMP_DECL;
  TMP_MARK;

  MPZ_TMP_INIT (x, 2);
  mpz_urandomb (x, rs, 32);
  r = mpz_get_ui (x);
  TMP_FREE;
  return r;
}

int
main (int argc, char **argv)
{
  gmp_randstate_ptr rands;
  unsigned long maxnbits, maxdbits, nbits, dbits;
  mpz_t n, d, tz;
  mp_size_t maxnn, maxdn, nn, dn, clearn, i;
  mp_ptr np, dp, qp, rp;
  mp_limb_t rh;
  mp_limb_t t;
  mp_limb_t dinv;
  int count = COUNT;
  mp_ptr scratch;
  mp_limb_t ran;
  mp_size_t alloc, itch;
  mp_limb_t rran0, rran1, qran0, qran1;
  TMP_DECL;

  if (argc > 1)
    {
      char *end;
      count = strtol (argv[1], &end, 0);
      if (*end || count <= 0)
	{
	  fprintf (stderr, "Invalid test count: %s.\n", argv[1]);
	  return 1;
	}
    }


  maxdbits = MAX_DN;
  maxnbits = MAX_NN;

  tests_start ();
  rands = RANDS;

  mpz_init (n);
  mpz_init (d);
  mpz_init (tz);

  maxnn = maxnbits / GMP_NUMB_BITS + 1;
  maxdn = maxdbits / GMP_NUMB_BITS + 1;

  TMP_MARK;

  qp = TMP_ALLOC_LIMBS (maxnn + 2) + 1;
  rp = TMP_ALLOC_LIMBS (maxnn + 2) + 1;

  alloc = 1;
  scratch = __GMP_ALLOCATE_FUNC_LIMBS (alloc);

  for (test = 0; test < count;)
    {
      nbits = random_word (rands) % (maxnbits - GMP_NUMB_BITS) + 2 * GMP_NUMB_BITS;
      if (maxdbits > nbits)
	dbits = random_word (rands) % nbits + 1;
      else
	dbits = random_word (rands) % maxdbits + 1;

#if RAND_UNIFORM
#define RANDFUNC mpz_urandomb
#else
#define RANDFUNC mpz_rrandomb
#endif

      do
	{
	  RANDFUNC (n, rands, nbits);
	  do
	    {
	      RANDFUNC (d, rands, dbits);
	    }
	  while (mpz_sgn (d) == 0);

	  np = PTR (n);
	  dp = PTR (d);
	  nn = SIZ (n);
	  dn = SIZ (d);
	}
      while (nn < dn);

      dp[0] |= 1;

      mpz_urandomb (tz, rands, 32);
      t = mpz_get_ui (tz);

      if (t % 17 == 0)
	dp[0] = GMP_NUMB_MAX;

      switch ((int) t % 16)
	{
	case 0:
	  clearn = random_word (rands) % nn;
	  for (i = 0; i <= clearn; i++)
	    np[i] = 0;
	  break;
	case 1:
	  mpn_sub_1 (np + nn - dn, dp, dn, random_word (rands));
	  break;
	case 2:
	  mpn_add_1 (np + nn - dn, dp, dn, random_word (rands));
	  break;
	}

      test++;

      binvert_limb (dinv, dp[0]);

      rran0 = random_word (rands);
      rran1 = random_word (rands);
      qran0 = random_word (rands);
      qran1 = random_word (rands);

      qp[-1] = qran0;
      qp[nn - dn + 1] = qran1;
      rp[-1] = rran0;

      ran = random_word (rands);

      if ((double) (nn - dn) * dn < 1e5)
	{
	  if (nn > dn)
	    {
	      /* Test mpn_sbpi1_bdiv_qr */
	      MPN_ZERO (qp, nn - dn);
	      MPN_ZERO (rp, dn);
	      MPN_COPY (rp, np, nn);
	      rh = mpn_sbpi1_bdiv_qr (qp, rp, nn, dp, dn, -dinv);
	      ASSERT_ALWAYS (qp[-1] == qran0);  ASSERT_ALWAYS (qp[nn - dn + 1] == qran1);
	      ASSERT_ALWAYS (rp[-1] == rran0);
	      check_one (qp, rp + nn - dn, rh, np, nn, dp, dn, "mpn_sbpi1_bdiv_qr");
	    }

	  if (nn > dn)
	    {
	      /* Test mpn_sbpi1_bdiv_q */
	      MPN_COPY (rp, np, nn);
	      MPN_ZERO (qp, nn - dn);
	      mpn_sbpi1_bdiv_q (qp, rp, nn - dn, dp, MIN(dn,nn-dn), -dinv);
	      ASSERT_ALWAYS (qp[-1] == qran0);  ASSERT_ALWAYS (qp[nn - dn + 1] == qran1);
	      ASSERT_ALWAYS (rp[-1] == rran0);
	      check_one (qp, NULL, 0, np, nn, dp, dn, "mpn_sbpi1_bdiv_q");
	    }
	}

      if (dn >= 4 && nn - dn >= 2)
	{
	  /* Test mpn_dcpi1_bdiv_qr */
	  MPN_COPY (rp, np, nn);
	  MPN_ZERO (qp, nn - dn);
	  rh = mpn_dcpi1_bdiv_qr (qp, rp, nn, dp, dn, -dinv);
	  ASSERT_ALWAYS (qp[-1] == qran0);  ASSERT_ALWAYS (qp[nn - dn + 1] == qran1);
	  ASSERT_ALWAYS (rp[-1] == rran0);
	  check_one (qp, rp + nn - dn, rh, np, nn, dp, dn, "mpn_dcpi1_bdiv_qr");
	}

      if (dn >= 4 && nn - dn >= 2)
	{
	  /* Test mpn_dcpi1_bdiv_q */
	  MPN_COPY (rp, np, nn);
	  MPN_ZERO (qp, nn - dn);
	  mpn_dcpi1_bdiv_q (qp, rp, nn - dn, dp, MIN(dn,nn-dn), -dinv);
	  ASSERT_ALWAYS (qp[-1] == qran0);  ASSERT_ALWAYS (qp[nn - dn + 1] == qran1);
	  ASSERT_ALWAYS (rp[-1] == rran0);
	  check_one (qp, NULL, 0, np, nn, dp, dn, "mpn_dcpi1_bdiv_q");
	}

      if (nn > dn)
	{
	  /* Test mpn_bdiv_qr */
	  itch = mpn_bdiv_qr_itch (nn, dn);
	  if (itch + 1 > alloc)
	    {
	      scratch = __GMP_REALLOCATE_FUNC_LIMBS (scratch, alloc, itch + 1);
	      alloc = itch + 1;
	    }
	  scratch[itch] = ran;
	  MPN_ZERO (qp, nn - dn);
	  MPN_ZERO (rp, dn);
	  rp[dn] = rran1;
	  rh = mpn_bdiv_qr (qp, rp, np, nn, dp, dn, scratch);
	  ASSERT_ALWAYS (ran == scratch[itch]);
	  ASSERT_ALWAYS (qp[-1] == qran0);  ASSERT_ALWAYS (qp[nn - dn + 1] == qran1);
	  ASSERT_ALWAYS (rp[-1] == rran0);  ASSERT_ALWAYS (rp[dn] == rran1);

	  check_one (qp, rp, rh, np, nn, dp, dn, "mpn_bdiv_qr");
	}

      if (nn - dn < 2 || dn < 2)
	continue;

      /* Test mpn_mu_bdiv_qr */
      itch = mpn_mu_bdiv_qr_itch (nn, dn);
      if (itch + 1 > alloc)
	{
	  scratch = __GMP_REALLOCATE_FUNC_LIMBS (scratch, alloc, itch + 1);
	  alloc = itch + 1;
	}
      scratch[itch] = ran;
      MPN_ZERO (qp, nn - dn);
      MPN_ZERO (rp, dn);
      rp[dn] = rran1;
      rh = mpn_mu_bdiv_qr (qp, rp, np, nn, dp, dn, scratch);
      ASSERT_ALWAYS (ran == scratch[itch]);
      ASSERT_ALWAYS (qp[-1] == qran0);  ASSERT_ALWAYS (qp[nn - dn + 1] == qran1);
      ASSERT_ALWAYS (rp[-1] == rran0);  ASSERT_ALWAYS (rp[dn] == rran1);
      check_one (qp, rp, rh, np, nn, dp, dn, "mpn_mu_bdiv_qr");

      /* Test mpn_mu_bdiv_q */
      itch = mpn_mu_bdiv_q_itch (nn, dn);
      if (itch + 1 > alloc)
	{
	  scratch = __GMP_REALLOCATE_FUNC_LIMBS (scratch, alloc, itch + 1);
	  alloc = itch + 1;
	}
      scratch[itch] = ran;
      MPN_ZERO (qp, nn - dn + 1);
      mpn_mu_bdiv_q (qp, np, nn - dn, dp, dn, scratch);
      ASSERT_ALWAYS (ran == scratch[itch]);
      ASSERT_ALWAYS (qp[-1] == qran0);  ASSERT_ALWAYS (qp[nn - dn + 1] == qran1);
      check_one (qp, NULL, 0, np, nn, dp, dn, "mpn_mu_bdiv_q");
    }

  __GMP_FREE_FUNC_LIMBS (scratch, alloc);

  TMP_FREE;

  mpz_clear (n);
  mpz_clear (d);
  mpz_clear (tz);

  tests_end ();
  return 0;
}
