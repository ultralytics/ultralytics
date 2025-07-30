/* Test mpz_set_d and mpz_init_set_d.

Copyright 2000-2003, 2006 Free Software Foundation, Inc.

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
check_data (void)
{
  static const struct {
    double     d;
    mp_size_t  want_size;
    mp_limb_t  want_data[2];
  } data[] = {

    {  0.0,  0 },
    {  1.0,  1, { 1 } },
    { -1.0, -1, { 1 } },

    {  123.0,  1, { 123 } },
    { -123.0, -1, { 123 } },

    { 1e-1, 0, { 0 } },
    { -1e-1, 0, { 0 } },
    { 2.328306436538696e-10, 0, { 0 } },
    { -2.328306436538696e-10, 0, { 0 } },
    { 5.421010862427522e-20, 0, { 0 } },
    { -5.421010862427522e-20, 0, { 0 } },
    { 2.938735877055719e-39, 0, { 0 } },
    { -2.938735877055719e-39, 0, { 0 } },
  };

  mpz_t  z;
  int    i;

  for (i = 0; i < numberof (data); i++)
    {
      mpz_init (z);
      mpz_set_d (z, data[i].d);
      MPZ_CHECK_FORMAT (z);
      if (z->_mp_size != data[i].want_size
          || refmpn_cmp_allowzero (z->_mp_d, data[i].want_data,
                                   ABS (data[i].want_size)) != 0)
        {
          printf ("mpz_set_d wrong on data[%d]\n", i);
        bad:
          d_trace   ("  d  ", data[i].d);
          printf    ("  got  size %ld\n", (long) z->_mp_size);
          printf    ("  want size %ld\n", (long) data[i].want_size);
          mpn_trace ("  got  z", z->_mp_d, z->_mp_size);
          mpn_trace ("  want z", data[i].want_data, data[i].want_size);
          abort();
        }
      mpz_clear (z);

      mpz_init_set_d (z, data[i].d);
      MPZ_CHECK_FORMAT (z);
      if (z->_mp_size != data[i].want_size
          || refmpn_cmp_allowzero (z->_mp_d, data[i].want_data,
                                   ABS (data[i].want_size)) != 0)
        {
          printf ("mpz_init_set_d wrong on data[%d]\n", i);
          goto bad;
        }
      mpz_clear (z);
    }
}

/* Try mpz_set_d on values 2^i+1, while such a value fits a double. */
void
check_2n_plus_1 (void)
{
  volatile double  p, d, diff;
  mpz_t  want, got;
  int    i;

  mpz_init (want);
  mpz_init (got);

  p = 1.0;
  mpz_set_ui (want, 2L);  /* gives 3 on first step */

  for (i = 1; i < 500; i++)
    {
      mpz_mul_2exp (want, want, 1L);
      mpz_sub_ui (want, want, 1L);   /* want = 2^i+1 */

      p *= 2.0;  /* p = 2^i */
      d = p + 1.0;
      diff = d - p;
      if (diff != 1.0)
        break;   /* rounding occurred, stop now */

      mpz_set_d (got, d);
      MPZ_CHECK_FORMAT (got);
      if (mpz_cmp (got, want) != 0)
        {
          printf ("mpz_set_d wrong on 2^%d+1\n", i);
          d_trace   ("  d ", d);
          mpz_trace ("  got  ", got);
          mpz_trace ("  want ", want);
          abort ();
        }
    }

  mpz_clear (want);
  mpz_clear (got);
}

int
main (void)
{
  tests_start ();

  check_data ();
  check_2n_plus_1 ();

  tests_end ();
  exit (0);
}
