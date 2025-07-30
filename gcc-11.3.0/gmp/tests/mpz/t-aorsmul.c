/* Test mpz_addmul, mpz_addmul_ui, mpz_submul, mpz_submul_ui.

Copyright 2001, 2002 Free Software Foundation, Inc.

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
#include <string.h>

#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"


#define M GMP_NUMB_MAX


void
check_one_inplace (mpz_srcptr w, mpz_srcptr y)
{
  mpz_t  want, got;

  mpz_init (want);
  mpz_init (got);

  mpz_mul (want, w, y);
  mpz_add (want, w, want);
  mpz_set (got, w);
  mpz_addmul (got, got, y);
  MPZ_CHECK_FORMAT (got);
  if (mpz_cmp (want, got) != 0)
    {
      printf ("mpz_addmul inplace fail\n");
    fail:
      mpz_trace ("w", w);
      mpz_trace ("y", y);
      mpz_trace ("want", want);
      mpz_trace ("got ", got);
      abort ();
    }

  mpz_mul (want, w, y);
  mpz_sub (want, w, want);
  mpz_set (got, w);
  mpz_submul (got, got, y);
  MPZ_CHECK_FORMAT (got);
  if (mpz_cmp (want, got) != 0)
    {
      printf ("mpz_submul inplace fail\n");
      goto fail;
    }

  mpz_clear (want);
  mpz_clear (got);
}

void
check_one_ui_inplace (mpz_ptr w, unsigned long y)
{
  mpz_t  want, got;

  mpz_init (want);
  mpz_init (got);

  mpz_mul_ui (want, w, (unsigned long) y);
  mpz_add (want, w, want);
  mpz_set (got, w);
  mpz_addmul_ui (got, got, (unsigned long) y);
  MPZ_CHECK_FORMAT (got);
  if (mpz_cmp (want, got) != 0)
    {
      printf ("mpz_addmul_ui fail\n");
    fail:
      mpz_trace ("w", w);
      printf    ("y=0x%lX   %lu\n", y, y);
      mpz_trace ("want", want);
      mpz_trace ("got ", got);
      abort ();
    }

  mpz_mul_ui (want, w, y);
  mpz_sub (want, w, want);
  mpz_set (got, w);
  mpz_submul_ui (got, got, y);
  MPZ_CHECK_FORMAT (got);
  if (mpz_cmp (want, got) != 0)
    {
      printf ("mpz_submul_ui fail\n");
      goto fail;
    }

  mpz_clear (want);
  mpz_clear (got);
}

void
check_all_inplace (mpz_ptr w, mpz_ptr y)
{
  int  wneg, yneg;

  MPZ_CHECK_FORMAT (w);
  MPZ_CHECK_FORMAT (y);

  for (wneg = 0; wneg < 2; wneg++)
    {
      for (yneg = 0; yneg < 2; yneg++)
        {
          check_one_inplace (w, y);

          if (mpz_fits_ulong_p (y))
            check_one_ui_inplace (w, mpz_get_ui (y));

          mpz_neg (y, y);
        }
      mpz_neg (w, w);
    }
}

void
check_one (mpz_srcptr w, mpz_srcptr x, mpz_srcptr y)
{
  mpz_t  want, got;

  mpz_init (want);
  mpz_init (got);

  mpz_mul (want, x, y);
  mpz_add (want, w, want);
  mpz_set (got, w);
  mpz_addmul (got, x, y);
  MPZ_CHECK_FORMAT (got);
  if (mpz_cmp (want, got) != 0)
    {
      printf ("mpz_addmul fail\n");
    fail:
      mpz_trace ("w", w);
      mpz_trace ("x", x);
      mpz_trace ("y", y);
      mpz_trace ("want", want);
      mpz_trace ("got ", got);
      abort ();
    }

  mpz_mul (want, x, y);
  mpz_sub (want, w, want);
  mpz_set (got, w);
  mpz_submul (got, x, y);
  MPZ_CHECK_FORMAT (got);
  if (mpz_cmp (want, got) != 0)
    {
      printf ("mpz_submul fail\n");
      goto fail;
    }

  mpz_clear (want);
  mpz_clear (got);
}

void
check_one_ui (mpz_ptr w, mpz_ptr x, unsigned long y)
{
  mpz_t  want, got;

  mpz_init (want);
  mpz_init (got);

  mpz_mul_ui (want, x, (unsigned long) y);
  mpz_add (want, w, want);
  mpz_set (got, w);
  mpz_addmul_ui (got, x, (unsigned long) y);
  MPZ_CHECK_FORMAT (got);
  if (mpz_cmp (want, got) != 0)
    {
      printf ("mpz_addmul_ui fail\n");
    fail:
      mpz_trace ("w", w);
      mpz_trace ("x", x);
      printf    ("y=0x%lX   %lu\n", y, y);
      mpz_trace ("want", want);
      mpz_trace ("got ", got);
      abort ();
    }

  mpz_mul_ui (want, x, y);
  mpz_sub (want, w, want);
  mpz_set (got, w);
  mpz_submul_ui (got, x, y);
  MPZ_CHECK_FORMAT (got);
  if (mpz_cmp (want, got) != 0)
    {
      printf ("mpz_submul_ui fail\n");
      goto fail;
    }

  mpz_clear (want);
  mpz_clear (got);
}


void
check_all (mpz_ptr w, mpz_ptr x, mpz_ptr y)
{
  int    swap, wneg, xneg, yneg;

  MPZ_CHECK_FORMAT (w);
  MPZ_CHECK_FORMAT (x);
  MPZ_CHECK_FORMAT (y);

  for (swap = 0; swap < 2; swap++)
    {
      for (wneg = 0; wneg < 2; wneg++)
        {
          for (xneg = 0; xneg < 2; xneg++)
            {
              for (yneg = 0; yneg < 2; yneg++)
                {
                  check_one (w, x, y);

                  if (mpz_fits_ulong_p (y))
                    check_one_ui (w, x, mpz_get_ui (y));

                  mpz_neg (y, y);
                }
              mpz_neg (x, x);
            }
          mpz_neg (w, w);
        }
      mpz_swap (x, y);
    }
}

void
check_data_inplace_ui (void)
{
  static const struct {
    mp_limb_t      w[6];
    unsigned long  y;

  } data[] = {

    { { 0 }, 0 },
    { { 0 }, 1 },
    { { 1 }, 1 },
    { { 2 }, 1 },

    { { 123 }, 1 },
    { { 123 }, ULONG_MAX },
    { { M }, 1 },
    { { M }, ULONG_MAX },

    { { 123, 456 }, 1 },
    { { M, M }, 1 },
    { { 123, 456 }, ULONG_MAX },
    { { M, M }, ULONG_MAX },

    { { 123, 456, 789 }, 1 },
    { { M, M, M }, 1 },
    { { 123, 456, 789 }, ULONG_MAX },
    { { M, M, M }, ULONG_MAX },
  };

  mpz_t  w, y;
  int    i;

  mpz_init (w);
  mpz_init (y);

  for (i = 0; i < numberof (data); i++)
    {
      mpz_set_n (w, data[i].w, (mp_size_t) numberof(data[i].w));
      mpz_set_ui (y, data[i].y);
      check_all_inplace (w, y);
    }

  mpz_clear (w);
  mpz_clear (y);
}

void
check_data (void)
{
  static const struct {
    mp_limb_t  w[6];
    mp_limb_t  x[6];
    mp_limb_t  y[6];

  } data[] = {

    /* reducing to zero */
    { { 1 }, { 1 }, { 1 } },
    { { 2 }, { 1 }, { 2 } },
    { { 0,1 }, { 0,1 }, { 1 } },

    /* reducing to 1 */
    { { 0,1 },       { M },       { 1 } },
    { { 0,0,1 },     { M,M },     { 1 } },
    { { 0,0,0,1 },   { M,M,M },   { 1 } },
    { { 0,0,0,0,1 }, { M,M,M,M }, { 1 } },

    /* reducing to -1 */
    { { M },       { 0,1 },       { 1 } },
    { { M,M },     { 0,0,1 },     { 1 } },
    { { M,M,M },   { 0,0,0,1 },   { 1 } },
    { { M,M,M,M }, { 0,0,0,0,1 }, { 1 } },

    /* carry out of addmul */
    { { M },     { 1 }, { 1 } },
    { { M,M },   { 1 }, { 1 } },
    { { M,M,M }, { 1 }, { 1 } },

    /* borrow from submul */
    { { 0,1 },     { 1 }, { 1 } },
    { { 0,0,1 },   { 1 }, { 1 } },
    { { 0,0,0,1 }, { 1 }, { 1 } },

    /* borrow from submul */
    { { 0,0,1 },     { 0,1 }, { 1 } },
    { { 0,0,0,1 },   { 0,1 }, { 1 } },
    { { 0,0,0,0,1 }, { 0,1 }, { 1 } },

    /* more borrow from submul */
    { { M }, { 0,1 },       { 1 } },
    { { M }, { 0,0,1 },     { 1 } },
    { { M }, { 0,0,0,1 },   { 1 } },
    { { M }, { 0,0,0,0,1 }, { 1 } },

    /* big borrow from submul */
    { { 0,0,1 },     { M,M }, { M } },
    { { 0,0,0,1 },   { M,M }, { M } },
    { { 0,0,0,0,1 }, { M,M }, { M } },

    /* small w */
    { { 0,1 }, { M,M },       { M } },
    { { 0,1 }, { M,M,M },     { M } },
    { { 0,1 }, { M,M,M,M },   { M } },
    { { 0,1 }, { M,M,M,M,M }, { M } },
  };

  mpz_t  w, x, y;
  int    i;

  mpz_init (w);
  mpz_init (x);
  mpz_init (y);

  for (i = 0; i < numberof (data); i++)
    {
      mpz_set_n (w, data[i].w, (mp_size_t) numberof(data[i].w));
      mpz_set_n (x, data[i].x, (mp_size_t) numberof(data[i].x));
      mpz_set_n (y, data[i].y, (mp_size_t) numberof(data[i].y));
      check_all (w, x, y);
    }

  mpz_clear (w);
  mpz_clear (x);
  mpz_clear (y);
}


void
check_random (int argc, char *argv[])
{
  gmp_randstate_ptr rands = RANDS;
  mpz_t  w, x, y;
  int    i, reps = 2000;

  mpz_init (w);
  mpz_init (x);
  mpz_init (y);

  if (argc == 2)
    reps = atoi (argv[1]);

  for (i = 0; i < reps; i++)
    {
      mpz_errandomb (w, rands, 5*GMP_LIMB_BITS);
      mpz_errandomb (x, rands, 5*GMP_LIMB_BITS);
      mpz_errandomb (y, rands, 5*GMP_LIMB_BITS);
      check_all (w, x, y);
      check_all_inplace (w, y);

      mpz_errandomb (w, rands, 5*GMP_LIMB_BITS);
      mpz_errandomb (x, rands, 5*GMP_LIMB_BITS);
      mpz_errandomb (y, rands, BITS_PER_ULONG);
      check_all (w, x, y);
      check_all_inplace (w, y);
    }

  mpz_clear (w);
  mpz_clear (x);
  mpz_clear (y);
}


int
main (int argc, char *argv[])
{
  tests_start ();
  mp_trace_base = -16;

  check_data ();
  check_data_inplace_ui ();
  check_random (argc, argv);

  tests_end ();
  exit (0);
}
