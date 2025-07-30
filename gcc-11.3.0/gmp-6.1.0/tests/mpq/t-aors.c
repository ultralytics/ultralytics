/* Test mpq_add and mpq_sub.

Copyright 2001 Free Software Foundation, Inc.

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

#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"


void
check_all (mpq_ptr x, mpq_ptr y, mpq_ptr want_add, mpq_ptr want_sub)
{
  mpq_t  got;
  int    neg_x, neg_y, swap;

  mpq_init (got);

  MPQ_CHECK_FORMAT (want_add);
  MPQ_CHECK_FORMAT (want_sub);
  MPQ_CHECK_FORMAT (x);
  MPQ_CHECK_FORMAT (y);

  for (swap = 0; swap <= 1; swap++)
    {
      for (neg_x = 0; neg_x <= 1; neg_x++)
        {
          for (neg_y = 0; neg_y <= 1; neg_y++)
            {
              mpq_add (got, x, y);
              MPQ_CHECK_FORMAT (got);
              if (! mpq_equal (got, want_add))
                {
                  printf ("mpq_add wrong\n");
                  mpq_trace ("  x   ", x);
                  mpq_trace ("  y   ", y);
                  mpq_trace ("  got ", got);
                  mpq_trace ("  want", want_add);
                  abort ();
                }

              mpq_sub (got, x, y);
              MPQ_CHECK_FORMAT (got);
              if (! mpq_equal (got, want_sub))
                {
                  printf ("mpq_sub wrong\n");
                  mpq_trace ("  x   ", x);
                  mpq_trace ("  y   ", y);
                  mpq_trace ("  got ", got);
                  mpq_trace ("  want", want_sub);
                  abort ();
                }


              mpq_neg (y, y);
              mpq_swap (want_add, want_sub);
            }

          mpq_neg (x, x);
          mpq_swap (want_add, want_sub);
          mpq_neg (want_add, want_add);
          mpq_neg (want_sub, want_sub);
        }

      mpq_swap (x, y);
      mpq_neg (want_sub, want_sub);
    }

  mpq_clear (got);
}


void
check_data (void)
{
  static const struct {
    const char  *x;
    const char  *y;
    const char  *want_add;
    const char  *want_sub;

  } data[] = {

    { "0", "0", "0", "0" },
    { "1", "0", "1", "1" },
    { "1", "1", "2", "0" },

    { "1/2", "1/2", "1", "0" },
    { "5/6", "14/15", "53/30", "-1/10" },
  };

  mpq_t  x, y, want_add, want_sub;
  int i;

  mpq_init (x);
  mpq_init (y);
  mpq_init (want_add);
  mpq_init (want_sub);

  for (i = 0; i < numberof (data); i++)
    {
      mpq_set_str_or_abort (x, data[i].x, 0);
      mpq_set_str_or_abort (y, data[i].y, 0);
      mpq_set_str_or_abort (want_add, data[i].want_add, 0);
      mpq_set_str_or_abort (want_sub, data[i].want_sub, 0);

      check_all (x, y, want_add, want_sub);
    }

  mpq_clear (x);
  mpq_clear (y);
  mpq_clear (want_add);
  mpq_clear (want_sub);
}


void
check_rand (void)
{
  mpq_t  x, y, want_add, want_sub;
  int i;
  gmp_randstate_ptr  rands = RANDS;

  mpq_init (x);
  mpq_init (y);
  mpq_init (want_add);
  mpq_init (want_sub);

  for (i = 0; i < 500; i++)
    {
      mpz_errandomb (mpq_numref(x), rands, 512L);
      mpz_errandomb_nonzero (mpq_denref(x), rands, 512L);
      mpq_canonicalize (x);

      mpz_errandomb (mpq_numref(y), rands, 512L);
      mpz_errandomb_nonzero (mpq_denref(y), rands, 512L);
      mpq_canonicalize (y);

      refmpq_add (want_add, x, y);
      refmpq_sub (want_sub, x, y);

      check_all (x, y, want_add, want_sub);
    }

  mpq_clear (x);
  mpq_clear (y);
  mpq_clear (want_add);
  mpq_clear (want_sub);
}


int
main (void)
{
  tests_start ();

  check_data ();
  check_rand ();

  tests_end ();

  exit (0);
}
