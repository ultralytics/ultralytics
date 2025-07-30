/* Functions for evaluating Gamma(1/3) and Gamma(2/3). Used by mpfr_ai.

Copyright 2010-2017 Free Software Foundation, Inc.
Contributed by the AriC and Caramba projects, INRIA.

This file is part of the GNU MPFR Library.

The GNU MPFR Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The GNU MPFR Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the GNU MPFR Library; see the file COPYING.LESSER.  If not, see
http://www.gnu.org/licenses/ or write to the Free Software Foundation, Inc.,
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA. */

#define MPFR_NEED_LONGLONG_H
#include "mpfr-impl.h"

#define MPFR_ACC_OR_MUL(v)                              \
  do                                                    \
    {                                                   \
      if (v <= ULONG_MAX / acc)                         \
        acc *= v;                                       \
      else                                              \
        {                                               \
          mpfr_mul_ui (y, y, acc, mode); acc = v;       \
        }                                               \
    }                                                   \
  while (0)

#define MPFR_ACC_OR_DIV(v)                              \
  do                                                    \
    {                                                   \
      if (v <= ULONG_MAX / acc)                         \
        acc *= v;                                       \
      else                                              \
        {                                               \
          mpfr_div_ui (y, y, acc, mode); acc = v;       \
        }                                               \
    }                                                   \
  while (0)

static void
mpfr_mul_ui5 (mpfr_ptr y, mpfr_srcptr x,
              unsigned long int v1, unsigned long int v2,
              unsigned long int v3, unsigned long int v4,
              unsigned long int v5, mpfr_rnd_t mode)
{
  unsigned long int acc = v1;
  mpfr_set (y, x, mode);
  MPFR_ACC_OR_MUL (v2);
  MPFR_ACC_OR_MUL (v3);
  MPFR_ACC_OR_MUL (v4);
  MPFR_ACC_OR_MUL (v5);
  mpfr_mul_ui (y, y, acc, mode);
}

void
mpfr_div_ui2 (mpfr_ptr y, mpfr_srcptr x,
              unsigned long int v1, unsigned long int v2, mpfr_rnd_t mode)
{
  unsigned long int acc = v1;
  mpfr_set (y, x, mode);
  MPFR_ACC_OR_DIV (v2);
  mpfr_div_ui (y, y, acc, mode);
}

static void
mpfr_div_ui8 (mpfr_ptr y, mpfr_srcptr x,
              unsigned long int v1, unsigned long int v2,
              unsigned long int v3, unsigned long int v4,
              unsigned long int v5, unsigned long int v6,
              unsigned long int v7, unsigned long int v8, mpfr_rnd_t mode)
{
  unsigned long int acc = v1;
  mpfr_set (y, x, mode);
  MPFR_ACC_OR_DIV (v2);
  MPFR_ACC_OR_DIV (v3);
  MPFR_ACC_OR_DIV (v4);
  MPFR_ACC_OR_DIV (v5);
  MPFR_ACC_OR_DIV (v6);
  MPFR_ACC_OR_DIV (v7);
  MPFR_ACC_OR_DIV (v8);
  mpfr_div_ui (y, y, acc, mode);
}


/* Gives an approximation of omega = Gamma(1/3)^6 * sqrt(10) / (12pi^4) */
/* using C. H. Brown's formula.                                         */
/* The computed value s satisfies |s-omega| <= 2^{1-prec}*omega         */
/* As usual, the variable s is supposed to be initialized.              */
static void
mpfr_Browns_const (mpfr_ptr s, mpfr_prec_t prec)
{
  mpfr_t uk;
  unsigned long int k;

  mpfr_prec_t working_prec = prec + 10 + MPFR_INT_CEIL_LOG2 (2 + prec / 10);

  mpfr_init2 (uk, working_prec);
  mpfr_set_prec (s, working_prec);

  mpfr_set_ui (uk, 1, MPFR_RNDN);
  mpfr_set (s, uk, MPFR_RNDN);
  k = 1;

  /* Invariants: uk ~ u(k-1) and s ~ sum(i=0..k-1, u(i)) */
  for (;;)
    {
      mpfr_mul_ui5 (uk, uk, 6 * k - 5, 6 * k - 4, 6 * k - 3, 6 * k - 2,
                    6 * k - 1, MPFR_RNDN);
      mpfr_div_ui8 (uk, uk, k, k, 3 * k - 2, 3 * k - 1, 3 * k, 80, 160, 160,
                    MPFR_RNDN);
      MPFR_CHANGE_SIGN (uk);

      mpfr_add (s, s, uk, MPFR_RNDN);
      k++;
      if (MPFR_GET_EXP (uk) + prec <= MPFR_GET_EXP (s) + 7)
        break;
    }

  mpfr_clear (uk);
  return;
}

/* Returns y such that |Gamma(1/3)-y| <= 2^{1-prec}*Gamma(1/3) */
static void
mpfr_gamma_one_third (mpfr_ptr y, mpfr_prec_t prec)
{
  mpfr_t tmp, tmp2, tmp3;

  mpfr_init2 (tmp, prec + 9);
  mpfr_init2 (tmp2, prec + 9);
  mpfr_init2 (tmp3, prec + 4);
  mpfr_set_prec (y, prec + 2);

  mpfr_const_pi (tmp, MPFR_RNDN);
  mpfr_sqr (tmp, tmp, MPFR_RNDN);
  mpfr_sqr (tmp, tmp, MPFR_RNDN);
  mpfr_mul_ui (tmp, tmp, 12, MPFR_RNDN);

  mpfr_Browns_const (tmp2, prec + 9);
  mpfr_mul (tmp, tmp, tmp2, MPFR_RNDN);

  mpfr_set_ui (tmp2, 10, MPFR_RNDN);
  mpfr_sqrt (tmp2, tmp2, MPFR_RNDN);
  mpfr_div (tmp, tmp, tmp2, MPFR_RNDN);

  mpfr_sqrt (tmp3, tmp, MPFR_RNDN);
  mpfr_cbrt (y, tmp3, MPFR_RNDN);

  mpfr_clear (tmp);
  mpfr_clear (tmp2);
  mpfr_clear (tmp3);
  return;
}

/* Computes y1 and y2 such that:                                      */
/*        |y1-Gamma(1/3)| <= 2^{1-prec}Gamma(1/3)                     */
/*  and   |y2-Gamma(2/3)| <= 2^{1-prec}Gamma(2/3)                     */
/*                                                                    */
/* Uses the formula Gamma(z)Gamma(1-z) = pi / sin(pi*z)               */
/* to compute Gamma(2/3) from Gamma(1/3).                             */
void
mpfr_gamma_one_and_two_third (mpfr_ptr y1, mpfr_ptr y2, mpfr_prec_t prec)
{
  mpfr_t temp;

  mpfr_init2 (temp, prec + 4);
  mpfr_set_prec (y2, prec + 4);

  mpfr_gamma_one_third (y1, prec + 4);

  mpfr_set_ui (temp, 3, MPFR_RNDN);
  mpfr_sqrt (temp, temp, MPFR_RNDN);
  mpfr_mul (temp, y1, temp, MPFR_RNDN);

  mpfr_const_pi (y2, MPFR_RNDN);
  mpfr_mul_2ui (y2, y2, 1, MPFR_RNDN);

  mpfr_div (y2, y2, temp, MPFR_RNDN);

  mpfr_clear (temp);
}
