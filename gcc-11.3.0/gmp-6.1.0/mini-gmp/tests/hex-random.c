/*

Copyright 2011, Free Software Foundation, Inc.

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

#include <time.h>
#include <unistd.h>

#include "gmp.h"

#include "hex-random.h"

static gmp_randstate_t state;

void
hex_random_init (void)
{
  unsigned long seed;
  char *env_seed;

  env_seed = getenv("GMP_CHECK_RANDOMIZE");
  if (env_seed && env_seed[0])
    {
      seed = strtoul (env_seed, NULL, 0);
      if (seed)
	printf ("Re-seeding with GMP_CHECK_RANDOMIZE=%lu\n", seed);
      else
	{
	  seed = time(NULL) + getpid();
	  printf ("Seed GMP_CHECK_RANDOMIZE=%lu (include this in bug reports)\n", seed);
	}
    }
  else
    seed = 4711;

  gmp_randinit_default (state);
  gmp_randseed_ui (state, seed);
}

char *
hex_urandomb (unsigned long bits)
{
  char *res;
  mpz_t x;

  mpz_init (x);
  mpz_urandomb (x, state, bits);
  gmp_asprintf (&res, "%Zx", x);
  mpz_clear (x);
  return res;
}

char *
hex_rrandomb (unsigned long bits)
{
  char *res;
  mpz_t x;

  mpz_init (x);
  mpz_rrandomb (x, state, bits);
  gmp_asprintf (&res, "%Zx", x);
  mpz_clear (x);
  return res;
}

char *
hex_rrandomb_export (void *dst, size_t *countp,
		     int order, size_t size, int endian, unsigned long bits)
{
  char *res;
  mpz_t x;
  mpz_init (x);
  mpz_rrandomb (x, state, bits);
  gmp_asprintf (&res, "%Zx", x);
  mpz_export (dst, countp, order, size, endian, 0, x);
  mpz_clear (x);
  return res;
}

void hex_random_op2 (enum hex_random_op op,  unsigned long maxbits,
		     char **ap, char **rp)
{
  mpz_t a, r;
  unsigned long abits;
  unsigned signs;

  mpz_init (a);
  mpz_init (r);

  abits = gmp_urandomb_ui (state, 32) % maxbits;

  mpz_rrandomb (a, state, abits);

  signs = gmp_urandomb_ui (state, 1);
  if (signs & 1)
    mpz_neg (a, a);

  switch (op)
    {
    default:
      abort ();
    case OP_SQR:
      mpz_mul (r, a, a);
      break;
    }

  gmp_asprintf (ap, "%Zx", a);
  gmp_asprintf (rp, "%Zx", r);

  mpz_clear (a);
  mpz_clear (r);
}

void
hex_random_op3 (enum hex_random_op op,  unsigned long maxbits,
		char **ap, char **bp, char **rp)
{
  mpz_t a, b, r;
  unsigned long abits, bbits;
  unsigned signs;

  mpz_init (a);
  mpz_init (b);
  mpz_init (r);

  abits = gmp_urandomb_ui (state, 32) % maxbits;
  bbits = gmp_urandomb_ui (state, 32) % maxbits;

  mpz_rrandomb (a, state, abits);
  mpz_rrandomb (b, state, bbits);

  signs = gmp_urandomb_ui (state, 3);
  if (signs & 1)
    mpz_neg (a, a);
  if (signs & 2)
    mpz_neg (b, b);

  switch (op)
    {
    default:
      abort ();
    case OP_ADD:
      mpz_add (r, a, b);
      break;
    case OP_SUB:
      mpz_sub (r, a, b);
      break;
    case OP_MUL:
      mpz_mul (r, a, b);
      break;
    case OP_GCD:
      if (signs & 4)
	{
	  /* Produce a large gcd */
	  unsigned long gbits = gmp_urandomb_ui (state, 32) % maxbits;
	  mpz_rrandomb (r, state, gbits);
	  mpz_mul (a, a, r);
	  mpz_mul (b, b, r);
	}
      mpz_gcd (r, a, b);
      break;
    case OP_LCM:
      if (signs & 4)
	{
	  /* Produce a large gcd */
	  unsigned long gbits = gmp_urandomb_ui (state, 32) % maxbits;
	  mpz_rrandomb (r, state, gbits);
	  mpz_mul (a, a, r);
	  mpz_mul (b, b, r);
	}
      mpz_lcm (r, a, b);
      break;
    case OP_AND:
      mpz_and (r, a, b);
      break;
    case OP_IOR:
      mpz_ior (r, a, b);
      break;
    case OP_XOR:
      mpz_xor (r, a, b);
      break;
    }

  gmp_asprintf (ap, "%Zx", a);
  gmp_asprintf (bp, "%Zx", b);
  gmp_asprintf (rp, "%Zx", r);

  mpz_clear (a);
  mpz_clear (b);
  mpz_clear (r);
}

void
hex_random_op4 (enum hex_random_op op, unsigned long maxbits,
		char **ap, char **bp, char **cp, char **dp)
{
  mpz_t a, b, c, d;
  unsigned long abits, bbits;
  unsigned signs;

  mpz_init (a);
  mpz_init (b);
  mpz_init (c);
  mpz_init (d);

  if (op == OP_POWM)
    {
      unsigned long cbits;
      abits = gmp_urandomb_ui (state, 32) % maxbits;
      bbits = 1 + gmp_urandomb_ui (state, 32) % maxbits;
      cbits = 2 + gmp_urandomb_ui (state, 32) % maxbits;

      mpz_rrandomb (a, state, abits);
      mpz_rrandomb (b, state, bbits);
      mpz_rrandomb (c, state, cbits);

      signs = gmp_urandomb_ui (state, 3);
      if (signs & 1)
	mpz_neg (a, a);
      if (signs & 2)
	{
	  mpz_t g;

	  /* If we negate the exponent, must make sure that gcd(a, c) = 1 */
	  if (mpz_sgn (a) == 0)
	    mpz_set_ui (a, 1);
	  else
	    {
	      mpz_init (g);

	      for (;;)
		{
		  mpz_gcd (g, a, c);
		  if (mpz_cmp_ui (g, 1) == 0)
		    break;
		  mpz_divexact (a, a, g);
		}
	      mpz_clear (g);
	    }
	  mpz_neg (b, b);
	}
      if (signs & 4)
	mpz_neg (c, c);

      mpz_powm (d, a, b, c);
    }
  else
    {
      unsigned long qbits;
      bbits = 1 + gmp_urandomb_ui (state, 32) % maxbits;
      qbits = gmp_urandomb_ui (state, 32) % maxbits;
      abits = bbits + qbits;
      if (abits > 30)
	abits -= 30;
      else
	abits = 0;

      mpz_rrandomb (a, state, abits);
      mpz_rrandomb (b, state, bbits);

      signs = gmp_urandomb_ui (state, 2);
      if (signs & 1)
	mpz_neg (a, a);
      if (signs & 2)
	mpz_neg (b, b);

      switch (op)
	{
	default:
	  abort ();
	case OP_CDIV:
	  mpz_cdiv_qr (c, d, a, b);
	  break;
	case OP_FDIV:
	  mpz_fdiv_qr (c, d, a, b);
	  break;
	case OP_TDIV:
	  mpz_tdiv_qr (c, d, a, b);
	  break;
	}
    }
  gmp_asprintf (ap, "%Zx", a);
  gmp_asprintf (bp, "%Zx", b);
  gmp_asprintf (cp, "%Zx", c);
  gmp_asprintf (dp, "%Zx", d);

  mpz_clear (a);
  mpz_clear (b);
  mpz_clear (c);
  mpz_clear (d);
}

void
hex_random_bit_op (enum hex_random_op op, unsigned long maxbits,
		   char **ap, unsigned long *b, char **rp)
{
  mpz_t a, r;
  unsigned long abits, bbits;
  unsigned signs;

  mpz_init (a);
  mpz_init (r);

  abits = gmp_urandomb_ui (state, 32) % maxbits;
  bbits = gmp_urandomb_ui (state, 32) % (maxbits + 100);

  mpz_rrandomb (a, state, abits);

  signs = gmp_urandomb_ui (state, 1);
  if (signs & 1)
    mpz_neg (a, a);

  switch (op)
    {
    default:
      abort ();

    case OP_SETBIT:
      mpz_set (r, a);
      mpz_setbit (r, bbits);
      break;
    case OP_CLRBIT:
      mpz_set (r, a);
      mpz_clrbit (r, bbits);
      break;
    case OP_COMBIT:
      mpz_set (r, a);
      mpz_combit (r, bbits);
      break;
    case OP_CDIV_Q_2:
      mpz_cdiv_q_2exp (r, a, bbits);
      break;
    case OP_CDIV_R_2:
      mpz_cdiv_r_2exp (r, a, bbits);
      break;
    case OP_FDIV_Q_2:
      mpz_fdiv_q_2exp (r, a, bbits);
      break;
    case OP_FDIV_R_2:
      mpz_fdiv_r_2exp (r, a, bbits);
      break;
    case OP_TDIV_Q_2:
      mpz_tdiv_q_2exp (r, a, bbits);
      break;
    case OP_TDIV_R_2:
      mpz_tdiv_r_2exp (r, a, bbits);
      break;
    }

  gmp_asprintf (ap, "%Zx", a);
  *b = bbits;
  gmp_asprintf (rp, "%Zx", r);

  mpz_clear (a);
  mpz_clear (r);
}

void
hex_random_scan_op (enum hex_random_op op, unsigned long maxbits,
		    char **ap, unsigned long *b, unsigned long *r)
{
  mpz_t a;
  unsigned long abits, bbits;
  unsigned signs;

  mpz_init (a);

  abits = gmp_urandomb_ui (state, 32) % maxbits;
  bbits = gmp_urandomb_ui (state, 32) % (maxbits + 100);

  mpz_rrandomb (a, state, abits);

  signs = gmp_urandomb_ui (state, 1);
  if (signs & 1)
    mpz_neg (a, a);

  switch (op)
    {
    default:
      abort ();

    case OP_SCAN0:
      *r = mpz_scan0 (a, bbits);
      break;
    case OP_SCAN1:
      *r = mpz_scan1 (a, bbits);
      break;
    }
  gmp_asprintf (ap, "%Zx", a);
  *b = bbits;

  mpz_clear (a);
}

void
hex_random_str_op (unsigned long maxbits,
		   int base, char **ap, char **rp)
{
  mpz_t a;
  unsigned long abits;
  unsigned signs;

  mpz_init (a);

  abits = gmp_urandomb_ui (state, 32) % maxbits;

  mpz_rrandomb (a, state, abits);

  signs = gmp_urandomb_ui (state, 2);
  if (signs & 1)
    mpz_neg (a, a);

  *ap = mpz_get_str (NULL, 16, a);
  *rp = mpz_get_str (NULL, base, a);

  mpz_clear (a);
}
