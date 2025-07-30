/* Test mpq_set_f.

Copyright 2000, 2001 Free Software Foundation, Inc.

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

int
main (int argc, char **argv)
{
#if GMP_NAIL_BITS == 0
  static const struct {
    int         f_base;
    const char  *f;
    int         z_base;
    const char  *want_num;
    const char  *want_den;

  } data[] = {

    { -2, "0",    16, "0", "1" },
    { -2, "1",    16, "1", "1" },
    { -2, "1@1",  16, "2", "1" },
    { -2, "1@2",  16, "4", "1" },
    { -2, "1@3",  16, "8", "1" },

    { -2, "1@30", 16,  "40000000", "1" },
    { -2, "1@31", 16,  "80000000", "1" },
    { -2, "1@32", 16, "100000000", "1" },
    { -2, "1@33", 16, "200000000", "1" },
    { -2, "1@34", 16, "400000000", "1" },

    { -2, "1@62", 16,  "4000000000000000", "1" },
    { -2, "1@63", 16,  "8000000000000000", "1" },
    { -2, "1@64", 16, "10000000000000000", "1" },
    { -2, "1@65", 16, "20000000000000000", "1" },
    { -2, "1@66", 16, "40000000000000000", "1" },

    { -2, "1@126", 16,  "40000000000000000000000000000000", "1" },
    { -2, "1@127", 16,  "80000000000000000000000000000000", "1" },
    { -2, "1@128", 16, "100000000000000000000000000000000", "1" },
    { -2, "1@129", 16, "200000000000000000000000000000000", "1" },
    { -2, "1@130", 16, "400000000000000000000000000000000", "1" },

    { -2, "1@-1",  16, "1", "2" },
    { -2, "1@-2",  16, "1", "4" },
    { -2, "1@-3",  16, "1", "8" },

    { -2, "1@-30", 16, "1",  "40000000" },
    { -2, "1@-31", 16, "1",  "80000000" },
    { -2, "1@-32", 16, "1", "100000000" },
    { -2, "1@-33", 16, "1", "200000000" },
    { -2, "1@-34", 16, "1", "400000000" },

    { -2, "1@-62", 16, "1",  "4000000000000000" },
    { -2, "1@-63", 16, "1",  "8000000000000000" },
    { -2, "1@-64", 16, "1", "10000000000000000" },
    { -2, "1@-65", 16, "1", "20000000000000000" },
    { -2, "1@-66", 16, "1", "40000000000000000" },

    { -2, "1@-126", 16, "1",  "40000000000000000000000000000000" },
    { -2, "1@-127", 16, "1",  "80000000000000000000000000000000" },
    { -2, "1@-128", 16, "1", "100000000000000000000000000000000" },
    { -2, "1@-129", 16, "1", "200000000000000000000000000000000" },
    { -2, "1@-130", 16, "1", "400000000000000000000000000000000" },

    { -2, "1@-30", 16, "1",  "40000000" },
    { -2, "1@-31", 16, "1",  "80000000" },
    { -2, "1@-32", 16, "1", "100000000" },
    { -2, "1@-33", 16, "1", "200000000" },
    { -2, "1@-34", 16, "1", "400000000" },

    { -2, "11@-62", 16, "3",  "4000000000000000" },
    { -2, "11@-63", 16, "3",  "8000000000000000" },
    { -2, "11@-64", 16, "3", "10000000000000000" },
    { -2, "11@-65", 16, "3", "20000000000000000" },
    { -2, "11@-66", 16, "3", "40000000000000000" },

    { 16, "80000000.00000001", 16, "8000000000000001", "100000000" },
    { 16, "80000000.00000008", 16, "1000000000000001",  "20000000" },
    { 16, "80000000.8",        16, "100000001", "2" },

  };

  mpf_t  f;
  mpq_t  got;
  mpz_t  want_num, want_den;
  int    i, neg;

  tests_start ();

  mpf_init2 (f, 1024L);
  mpq_init (got);
  mpz_init (want_num);
  mpz_init (want_den);

  for (i = 0; i < numberof (data); i++)
    {
      for (neg = 0; neg <= 1; neg++)
        {
          mpf_set_str_or_abort (f, data[i].f, data[i].f_base);
          mpz_set_str_or_abort (want_num, data[i].want_num, data[i].z_base);
          mpz_set_str_or_abort (want_den, data[i].want_den, data[i].z_base);

          if (neg)
            {
              mpf_neg (f, f);
              mpz_neg (want_num, want_num);
            }

          mpq_set_f (got, f);
          MPQ_CHECK_FORMAT (got);

          if (mpz_cmp (mpq_numref(got), want_num) != 0
              || mpz_cmp (mpq_denref(got), want_den) != 0)
            {
              printf ("wrong at data[%d]\n", i);
              printf ("   f_base %d, z_base %d\n",
                      data[i].f_base, data[i].z_base);

              printf ("   f \"%s\" hex ", data[i].f);
              mpf_out_str (stdout, 16, 0, f);
              printf ("\n");

              printf ("   want num 0x");
              mpz_out_str (stdout, 16, want_num);
              printf ("\n");
              printf ("   want den 0x");
              mpz_out_str (stdout, 16, want_den);
              printf ("\n");

              printf ("   got num 0x");
              mpz_out_str (stdout, 16, mpq_numref(got));
              printf ("\n");
              printf ("   got den 0x");
              mpz_out_str (stdout, 16, mpq_denref(got));
              printf ("\n");

              abort ();
            }
        }
    }

  mpf_clear (f);
  mpq_clear (got);
  mpz_clear (want_num);
  mpz_clear (want_den);

  tests_end ();
#endif
  exit (0);
}
