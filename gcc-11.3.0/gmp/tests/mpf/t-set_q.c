/* Test mpf_set_q.

Copyright 2004 Free Software Foundation, Inc.

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


void
check_one (mpf_ptr got, mpq_srcptr q)
{
  mpf_t  n, d;

  mpf_set_q (got, q);

  PTR(n) = PTR(&q->_mp_num);
  SIZ(n) = SIZ(&q->_mp_num);
  EXP(n) = ABSIZ(&q->_mp_num);

  PTR(d) = PTR(&q->_mp_den);
  SIZ(d) = SIZ(&q->_mp_den);
  EXP(d) = ABSIZ(&q->_mp_den);

  if (! refmpf_validate_division ("mpf_set_q", got, n, d))
    {
      mp_trace_base = -16;
      mpq_trace ("   q", q);
      abort ();
    }
}

void
check_rand (void)
{
  unsigned long  min_prec = __GMPF_BITS_TO_PREC (1);
  gmp_randstate_ptr  rands = RANDS;
  unsigned long  prec;
  mpf_t  got;
  mpq_t  q;
  int    i;

  mpf_init (got);
  mpq_init (q);

  for (i = 0; i < 400; i++)
    {
      /* result precision */
      prec = min_prec + gmp_urandomm_ui (rands, 20L);
      refmpf_set_prec_limbs (got, prec);

      /* num */
      prec = gmp_urandomm_ui (rands, 20L * GMP_NUMB_BITS);
      mpz_rrandomb (mpq_numref(q), rands, prec);

      /* possibly negative num */
      if (gmp_urandomb_ui (rands, 1L))
        mpz_neg (mpq_numref(q), mpq_numref(q));

      /* den, non-zero */
      do {
        prec = gmp_urandomm_ui (rands, 20L * GMP_NUMB_BITS);
        mpz_rrandomb (mpq_denref(q), rands, prec);
      } while (mpz_sgn (mpq_denref(q)) <= 0);

      check_one (got, q);
    }

  mpf_clear (got);
  mpq_clear (q);
}

void
check_various (void)
{
  mpf_t got;
  mpq_t q;

  mpf_init (got);
  mpq_init (q);

  /* 1/1 == 1 */
  mpf_set_prec (got, 20L);
  mpq_set_ui (q, 1L, 1L);
  mpf_set_q (got, q);
  MPF_CHECK_FORMAT (got);
  ASSERT_ALWAYS (mpf_cmp_ui (got, 1L) == 0);

  /* 1/(2^n+1), a case where truncating the divisor would be wrong */
  mpf_set_prec (got, 500L);
  mpq_set_ui (q, 1L, 1L);
  mpz_mul_2exp (mpq_denref(q), mpq_denref(q), 800L);
  mpz_add_ui (mpq_denref(q), mpq_denref(q), 1L);
  check_one (got, q);

  mpf_clear (got);
  mpq_clear (q);
}

int
main (void)
{
  tests_start ();

  check_various ();
  check_rand ();

  tests_end ();
  exit (0);
}
