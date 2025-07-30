/* Functions needed for bootstrapping the gmp build, based on mini-gmp.

Copyright 2001, 2002, 2004, 2011, 2012 Free Software Foundation, Inc.

This file is part of the GNU MP Library.

The GNU MP Library is free software; you can redistribute it and/or modify
it under the terms of either:

  * the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your
    option) any later version.

or

  * the GNU General Public License as published by the Free Software
    Foundation; either version 2 of the License, or (at your option) any
    later version.

or both in parallel, as here.

The GNU MP Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received copies of the GNU General Public License and the
GNU Lesser General Public License along with the GNU MP Library.  If not,
see https://www.gnu.org/licenses/.  */


#include "mini-gmp/mini-gmp.c"

#define MIN(l,o) ((l) < (o) ? (l) : (o))
#define PTR(x)   ((x)->_mp_d)
#define SIZ(x)   ((x)->_mp_size)

#define xmalloc gmp_default_alloc

int
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

int
log2_ceil (int n)
{
  int  e;
  assert (n >= 1);
  for (e = 0; ; e++)
    if ((1 << e) >= n)
      break;
  return e;
}

/* Set inv to the inverse of d, in the style of invert_limb, ie. for
   udiv_qrnnd_preinv.  */
void
mpz_preinv_invert (mpz_t inv, mpz_t d, int numb_bits)
{
  mpz_t  t;
  int    norm;
  assert (SIZ(d) > 0);

  norm = numb_bits - mpz_sizeinbase (d, 2);
  assert (norm >= 0);
  mpz_init_set_ui (t, 1L);
  mpz_mul_2exp (t, t, 2*numb_bits - norm);
  mpz_tdiv_q (inv, t, d);
  mpz_set_ui (t, 1L);
  mpz_mul_2exp (t, t, numb_bits);
  mpz_sub (inv, inv, t);

  mpz_clear (t);
}

/* Calculate r satisfying r*d == 1 mod 2^n. */
void
mpz_invert_2exp (mpz_t r, mpz_t a, unsigned long n)
{
  unsigned long  i;
  mpz_t  inv, prod;

  assert (mpz_odd_p (a));

  mpz_init_set_ui (inv, 1L);
  mpz_init (prod);

  for (i = 1; i < n; i++)
    {
      mpz_mul (prod, inv, a);
      if (mpz_tstbit (prod, i) != 0)
	mpz_setbit (inv, i);
    }

  mpz_mul (prod, inv, a);
  mpz_tdiv_r_2exp (prod, prod, n);
  assert (mpz_cmp_ui (prod, 1L) == 0);

  mpz_set (r, inv);

  mpz_clear (inv);
  mpz_clear (prod);
}

/* Calculate inv satisfying r*a == 1 mod 2^n. */
void
mpz_invert_ui_2exp (mpz_t r, unsigned long a, unsigned long n)
{
  mpz_t  az;
  mpz_init_set_ui (az, a);
  mpz_invert_2exp (r, az, n);
  mpz_clear (az);
}
