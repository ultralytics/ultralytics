/* Copyright 2006, 2007, 2009, 2010, 2013-2015 Free Software Foundation, Inc.

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

static signed long test;

static void
check_one (mp_ptr qp, mp_srcptr rp,
	   mp_srcptr np, mp_size_t nn, mp_srcptr dp, mp_size_t dn,
	   const char *fname, mp_limb_t q_allowed_err)
{
  mp_size_t qn = nn - dn + 1;
  mp_ptr tp;
  const char *msg;
  const char *tvalue;
  mp_limb_t i;
  TMP_DECL;
  TMP_MARK;

  tp = TMP_ALLOC_LIMBS (nn + 1);
  if (dn >= qn)
    refmpn_mul (tp, dp, dn, qp, qn);
  else
    refmpn_mul (tp, qp, qn, dp, dn);

  for (i = 0; i < q_allowed_err && (tp[nn] > 0 || mpn_cmp (tp, np, nn) > 0); i++)
    ASSERT_NOCARRY (refmpn_sub (tp, tp, nn+1, dp, dn));

  if (tp[nn] > 0 || mpn_cmp (tp, np, nn) > 0)
    {
      msg = "q too large";
      tvalue = "Q*D";
    error:
      printf ("\r*******************************************************************************\n");
      printf ("%s failed test %ld: %s\n", fname, test, msg);
      printf ("N=    "); dumpy (np, nn);
      printf ("D=    "); dumpy (dp, dn);
      printf ("Q=    "); dumpy (qp, qn);
      if (rp)
	{ printf ("R=    "); dumpy (rp, dn); }
      printf ("%5s=", tvalue); dumpy (tp, nn+1);
      printf ("nn = %ld, dn = %ld, qn = %ld\n", nn, dn, qn);
      abort ();
    }

  ASSERT_NOCARRY (refmpn_sub_n (tp, np, tp, nn));
  tvalue = "N-Q*D";
  if (!(nn == dn || mpn_zero_p (tp + dn, nn - dn)) || mpn_cmp (tp, dp, dn) >= 0)
    {
      msg = "q too small";
      goto error;
    }

  if (rp && mpn_cmp (rp, tp, dn) != 0)
    {
      msg = "r incorrect";
      goto error;
    }

  TMP_FREE;
}


/* These are *bit* sizes. */
#ifndef SIZE_LOG
#define SIZE_LOG 17
#endif
#define MAX_DN (1L << SIZE_LOG)
#define MAX_NN (1L << (SIZE_LOG + 1))

#define COUNT 200

int
main (int argc, char **argv)
{
  gmp_randstate_ptr rands;
  unsigned long maxnbits, maxdbits, nbits, dbits;
  mpz_t n, d, q, r, tz, junk;
  mp_size_t maxnn, maxdn, nn, dn, clearn, i;
  mp_ptr np, dup, dnp, qp, rp, junkp;
  mp_limb_t t;
  gmp_pi1_t dinv;
  long count = COUNT;
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
  mpz_init (q);
  mpz_init (r);
  mpz_init (tz);
  mpz_init (junk);

  maxnn = maxnbits / GMP_NUMB_BITS + 1;
  maxdn = maxdbits / GMP_NUMB_BITS + 1;

  TMP_MARK;

  qp = TMP_ALLOC_LIMBS (maxnn + 2) + 1;
  rp = TMP_ALLOC_LIMBS (maxnn + 2) + 1;
  dnp = TMP_ALLOC_LIMBS (maxdn);

  alloc = 1;
  scratch = __GMP_ALLOCATE_FUNC_LIMBS (alloc);

  for (test = -300; test < count; test++)
    {
      nbits = urandom () % (maxnbits - GMP_NUMB_BITS) + 2 * GMP_NUMB_BITS;

      if (test < 0)
	dbits = (test + 300) % (nbits - 1) + 1;
      else
	dbits = urandom () % (nbits - 1) % maxdbits + 1;

#if RAND_UNIFORM
#define RANDFUNC mpz_urandomb
#else
#define RANDFUNC mpz_rrandomb
#endif

      do
	RANDFUNC (d, rands, dbits);
      while (mpz_sgn (d) == 0);
      dn = SIZ (d);
      dup = PTR (d);
      MPN_COPY (dnp, dup, dn);
      dnp[dn - 1] |= GMP_NUMB_HIGHBIT;

      if (test % 2 == 0)
	{
	  RANDFUNC (n, rands, nbits);
	  nn = SIZ (n);
	  ASSERT_ALWAYS (nn >= dn);
	}
      else
	{
	  do
	    {
	      RANDFUNC (q, rands, urandom () % (nbits - dbits + 1));
	      RANDFUNC (r, rands, urandom () % mpz_sizeinbase (d, 2));
	      mpz_mul (n, q, d);
	      mpz_add (n, n, r);
	      nn = SIZ (n);
	    }
	  while (nn > maxnn || nn < dn);
	}

      ASSERT_ALWAYS (nn <= maxnn);
      ASSERT_ALWAYS (dn <= maxdn);

      mpz_urandomb (junk, rands, nbits);
      junkp = PTR (junk);

      np = PTR (n);

      mpz_urandomb (tz, rands, 32);
      t = mpz_get_ui (tz);

      if (t % 17 == 0)
	{
	  dnp[dn - 1] = GMP_NUMB_MAX;
	  dup[dn - 1] = GMP_NUMB_MAX;
	}

      switch ((int) t % 16)
	{
	case 0:
	  clearn = urandom () % nn;
	  for (i = clearn; i < nn; i++)
	    np[i] = 0;
	  break;
	case 1:
	  mpn_sub_1 (np + nn - dn, dnp, dn, urandom ());
	  break;
	case 2:
	  mpn_add_1 (np + nn - dn, dnp, dn, urandom ());
	  break;
	}

      if (dn >= 2)
	invert_pi1 (dinv, dnp[dn - 1], dnp[dn - 2]);

      rran0 = urandom ();
      rran1 = urandom ();
      qran0 = urandom ();
      qran1 = urandom ();

      qp[-1] = qran0;
      qp[nn - dn + 1] = qran1;
      rp[-1] = rran0;

      ran = urandom ();

      if ((double) (nn - dn) * dn < 1e5)
	{
	  /* Test mpn_sbpi1_div_qr */
	  if (dn > 2)
	    {
	      MPN_COPY (rp, np, nn);
	      if (nn > dn)
		MPN_COPY (qp, junkp, nn - dn);
	      qp[nn - dn] = mpn_sbpi1_div_qr (qp, rp, nn, dnp, dn, dinv.inv32);
	      check_one (qp, rp, np, nn, dnp, dn, "mpn_sbpi1_div_qr", 0);
	    }

	  /* Test mpn_sbpi1_divappr_q */
	  if (dn > 2)
	    {
	      MPN_COPY (rp, np, nn);
	      if (nn > dn)
		MPN_COPY (qp, junkp, nn - dn);
	      qp[nn - dn] = mpn_sbpi1_divappr_q (qp, rp, nn, dnp, dn, dinv.inv32);
	      check_one (qp, NULL, np, nn, dnp, dn, "mpn_sbpi1_divappr_q", 1);
	    }

	  /* Test mpn_sbpi1_div_q */
	  if (dn > 2)
	    {
	      MPN_COPY (rp, np, nn);
	      if (nn > dn)
		MPN_COPY (qp, junkp, nn - dn);
	      qp[nn - dn] = mpn_sbpi1_div_q (qp, rp, nn, dnp, dn, dinv.inv32);
	      check_one (qp, NULL, np, nn, dnp, dn, "mpn_sbpi1_div_q", 0);
	    }

	  /* Test mpn_sec_div_qr */
	  itch = mpn_sec_div_qr_itch (nn, dn);
	  if (itch + 1 > alloc)
	    {
	      scratch = __GMP_REALLOCATE_FUNC_LIMBS (scratch, alloc, itch + 1);
	      alloc = itch + 1;
	    }
	  scratch[itch] = ran;
	  MPN_COPY (rp, np, nn);
	  if (nn >= dn)
	    MPN_COPY (qp, junkp, nn - dn + 1);
	  qp[nn - dn] = mpn_sec_div_qr (qp, rp, nn, dup, dn, scratch);
	  ASSERT_ALWAYS (ran == scratch[itch]);
	  check_one (qp, rp, np, nn, dup, dn, "mpn_sec_div_qr (unnorm)", 0);

	  /* Test mpn_sec_div_r */
	  itch = mpn_sec_div_r_itch (nn, dn);
	  if (itch + 1 > alloc)
	    {
	      scratch = __GMP_REALLOCATE_FUNC_LIMBS (scratch, alloc, itch + 1);
	      alloc = itch + 1;
	    }
	  scratch[itch] = ran;
	  MPN_COPY (rp, np, nn);
	  mpn_sec_div_r (rp, nn, dup, dn, scratch);
	  ASSERT_ALWAYS (ran == scratch[itch]);
	  /* Note: Since check_one cannot cope with remainder-only functions, we
	     pass qp[] from the previous function, mpn_sec_div_qr.  */
	  check_one (qp, rp, np, nn, dup, dn, "mpn_sec_div_r (unnorm)", 0);

	  /* Normalised case, mpn_sec_div_qr */
	  itch = mpn_sec_div_qr_itch (nn, dn);
	  scratch[itch] = ran;

	  MPN_COPY (rp, np, nn);
	  if (nn >= dn)
	    MPN_COPY (qp, junkp, nn - dn + 1);
	  qp[nn - dn] = mpn_sec_div_qr (qp, rp, nn, dnp, dn, scratch);
	  ASSERT_ALWAYS (ran == scratch[itch]);
	  check_one (qp, rp, np, nn, dnp, dn, "mpn_sec_div_qr (norm)", 0);

	  /* Normalised case, mpn_sec_div_r */
	  itch = mpn_sec_div_r_itch (nn, dn);
	  scratch[itch] = ran;
	  MPN_COPY (rp, np, nn);
	  mpn_sec_div_r (rp, nn, dnp, dn, scratch);
	  ASSERT_ALWAYS (ran == scratch[itch]);
	  /* Note: Since check_one cannot cope with remainder-only functions, we
	     pass qp[] from the previous function, mpn_sec_div_qr.  */
	  check_one (qp, rp, np, nn, dnp, dn, "mpn_sec_div_r (norm)", 0);
	}

      /* Test mpn_dcpi1_div_qr */
      if (dn >= 6 && nn - dn >= 3)
	{
	  MPN_COPY (rp, np, nn);
	  if (nn > dn)
	    MPN_COPY (qp, junkp, nn - dn);
	  qp[nn - dn] = mpn_dcpi1_div_qr (qp, rp, nn, dnp, dn, &dinv);
	  ASSERT_ALWAYS (qp[-1] == qran0);  ASSERT_ALWAYS (qp[nn - dn + 1] == qran1);
	  ASSERT_ALWAYS (rp[-1] == rran0);
	  check_one (qp, rp, np, nn, dnp, dn, "mpn_dcpi1_div_qr", 0);
	}

      /* Test mpn_dcpi1_divappr_q */
      if (dn >= 6 && nn - dn >= 3)
	{
	  MPN_COPY (rp, np, nn);
	  if (nn > dn)
	    MPN_COPY (qp, junkp, nn - dn);
	  qp[nn - dn] = mpn_dcpi1_divappr_q (qp, rp, nn, dnp, dn, &dinv);
	  ASSERT_ALWAYS (qp[-1] == qran0);  ASSERT_ALWAYS (qp[nn - dn + 1] == qran1);
	  ASSERT_ALWAYS (rp[-1] == rran0);
	  check_one (qp, NULL, np, nn, dnp, dn, "mpn_dcpi1_divappr_q", 1);
	}

      /* Test mpn_dcpi1_div_q */
      if (dn >= 6 && nn - dn >= 3)
	{
	  MPN_COPY (rp, np, nn);
	  if (nn > dn)
	    MPN_COPY (qp, junkp, nn - dn);
	  qp[nn - dn] = mpn_dcpi1_div_q (qp, rp, nn, dnp, dn, &dinv);
	  ASSERT_ALWAYS (qp[-1] == qran0);  ASSERT_ALWAYS (qp[nn - dn + 1] == qran1);
	  ASSERT_ALWAYS (rp[-1] == rran0);
	  check_one (qp, NULL, np, nn, dnp, dn, "mpn_dcpi1_div_q", 0);
	}

     /* Test mpn_mu_div_qr */
      if (nn - dn > 2 && dn >= 2)
	{
	  itch = mpn_mu_div_qr_itch (nn, dn, 0);
	  if (itch + 1 > alloc)
	    {
	      scratch = __GMP_REALLOCATE_FUNC_LIMBS (scratch, alloc, itch + 1);
	      alloc = itch + 1;
	    }
	  scratch[itch] = ran;
	  MPN_COPY (qp, junkp, nn - dn);
	  MPN_ZERO (rp, dn);
	  rp[dn] = rran1;
	  qp[nn - dn] = mpn_mu_div_qr (qp, rp, np, nn, dnp, dn, scratch);
	  ASSERT_ALWAYS (ran == scratch[itch]);
	  ASSERT_ALWAYS (qp[-1] == qran0);  ASSERT_ALWAYS (qp[nn - dn + 1] == qran1);
	  ASSERT_ALWAYS (rp[-1] == rran0);  ASSERT_ALWAYS (rp[dn] == rran1);
	  check_one (qp, rp, np, nn, dnp, dn, "mpn_mu_div_qr", 0);
	}

      /* Test mpn_mu_divappr_q */
      if (nn - dn > 2 && dn >= 2)
	{
	  itch = mpn_mu_divappr_q_itch (nn, dn, 0);
	  if (itch + 1 > alloc)
	    {
	      scratch = __GMP_REALLOCATE_FUNC_LIMBS (scratch, alloc, itch + 1);
	      alloc = itch + 1;
	    }
	  scratch[itch] = ran;
	  MPN_COPY (qp, junkp, nn - dn);
	  qp[nn - dn] = mpn_mu_divappr_q (qp, np, nn, dnp, dn, scratch);
	  ASSERT_ALWAYS (ran == scratch[itch]);
	  ASSERT_ALWAYS (qp[-1] == qran0);  ASSERT_ALWAYS (qp[nn - dn + 1] == qran1);
	  check_one (qp, NULL, np, nn, dnp, dn, "mpn_mu_divappr_q", 4);
	}

      /* Test mpn_mu_div_q */
      if (nn - dn > 2 && dn >= 2)
	{
	  itch = mpn_mu_div_q_itch (nn, dn, 0);
	  if (itch + 1> alloc)
	    {
	      scratch = __GMP_REALLOCATE_FUNC_LIMBS (scratch, alloc, itch + 1);
	      alloc = itch + 1;
	    }
	  scratch[itch] = ran;
	  MPN_COPY (qp, junkp, nn - dn);
	  qp[nn - dn] = mpn_mu_div_q (qp, np, nn, dnp, dn, scratch);
	  ASSERT_ALWAYS (ran == scratch[itch]);
	  ASSERT_ALWAYS (qp[-1] == qran0);  ASSERT_ALWAYS (qp[nn - dn + 1] == qran1);
	  check_one (qp, NULL, np, nn, dnp, dn, "mpn_mu_div_q", 0);
	}

      if (1)
	{
	  itch = nn + 1;
	  if (itch + 1> alloc)
	    {
	      scratch = __GMP_REALLOCATE_FUNC_LIMBS (scratch, alloc, itch + 1);
	      alloc = itch + 1;
	    }
	  scratch[itch] = ran;
	  mpn_div_q (qp, np, nn, dup, dn, scratch);
	  ASSERT_ALWAYS (ran == scratch[itch]);
	  ASSERT_ALWAYS (qp[-1] == qran0);  ASSERT_ALWAYS (qp[nn - dn + 1] == qran1);
	  check_one (qp, NULL, np, nn, dup, dn, "mpn_div_q", 0);
	}

      if (dn >= 2 && nn >= 2)
	{
	  mp_limb_t qh;

	  /* mpn_divrem_2 */
	  MPN_COPY (rp, np, nn);
	  qp[nn - 2] = qp[nn-1] = qran1;

	  qh = mpn_divrem_2 (qp, 0, rp, nn, dnp + dn - 2);
	  ASSERT_ALWAYS (qp[nn - 2] == qran1);
	  ASSERT_ALWAYS (qp[-1] == qran0);  ASSERT_ALWAYS (qp[nn - 1] == qran1);
	  qp[nn - 2] = qh;
	  check_one (qp, rp, np, nn, dnp + dn - 2, 2, "mpn_divrem_2", 0);

	  /* Missing: divrem_2 with fraction limbs. */

	  /* mpn_div_qr_2 */
	  qp[nn - 2] = qran1;

	  qh = mpn_div_qr_2 (qp, rp, np, nn, dup + dn - 2);
	  ASSERT_ALWAYS (qp[nn - 2] == qran1);
	  ASSERT_ALWAYS (qp[-1] == qran0);  ASSERT_ALWAYS (qp[nn - 1] == qran1);
	  qp[nn - 2] = qh;
	  check_one (qp, rp, np, nn, dup + dn - 2, 2, "mpn_div_qr_2", 0);
	}
      if (dn >= 1 && nn >= 1)
	{
	  /* mpn_div_qr_1 */
	  mp_limb_t qh;
	  qp[nn-1] = qran1;
	  rp[0] = mpn_div_qr_1 (qp, &qh, np, nn, dnp[dn - 1]);
	  ASSERT_ALWAYS (qp[-1] == qran0); ASSERT_ALWAYS (qp[nn - 1] == qran1);
	  qp[nn - 1] = qh;
	  check_one (qp, rp, np, nn,  dnp + dn - 1, 1, "mpn_div_qr_1", 0);
	}
    }

  __GMP_FREE_FUNC_LIMBS (scratch, alloc);

  TMP_FREE;

  mpz_clear (n);
  mpz_clear (d);
  mpz_clear (q);
  mpz_clear (r);
  mpz_clear (tz);
  mpz_clear (junk);

  tests_end ();
  return 0;
}
