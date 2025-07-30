/* Test mpf_get_d_2exp.

Copyright 2002, 2003 Free Software Foundation, Inc.

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


static void
check_onebit (void)
{
  static const long data[] = {
    -513, -512, -511, -65, -64, -63, -32, -1,
    0, 1, 32, 53, 54, 64, 128, 256, 511, 512, 513
  };
  mpf_t   f;
  double  got, want;
  long    got_exp, want_exp;
  int     i;

  mpf_init2 (f, 1024L);

  for (i = 0; i < numberof (data); i++)
    {
      mpf_set_ui (f, 1L);
      if (data[i] >= 0)
        mpf_mul_2exp (f, f, data[i]);
      else
        mpf_div_2exp (f, f, -data[i]);
      want = 0.5;
      want_exp = data[i] + 1;

      got = mpf_get_d_2exp (&got_exp, f);
      if (got != want || got_exp != want_exp)
        {
          printf    ("mpf_get_d_2exp wrong on 2**%ld\n", data[i]);
          mpf_trace ("   f    ", f);
          d_trace   ("   want ", want);
          d_trace   ("   got  ", got);
          printf    ("   want exp %ld\n", want_exp);
          printf    ("   got exp  %ld\n", got_exp);
          abort();
        }
    }
  mpf_clear (f);
}

/* Check that hardware rounding doesn't make mpf_get_d_2exp return a value
   outside its defined range. */
static void
check_round (void)
{
  static const unsigned long data[] = { 1, 32, 53, 54, 64, 128, 256, 512 };
  mpf_t   f;
  double  got;
  long    got_exp;
  int     i, rnd_mode, old_rnd_mode;

  mpf_init2 (f, 1024L);
  old_rnd_mode = tests_hardware_getround ();

  for (rnd_mode = 0; rnd_mode < 4; rnd_mode++)
    {
      tests_hardware_setround (rnd_mode);

      for (i = 0; i < numberof (data); i++)
        {
          mpf_set_ui (f, 1L);
          mpf_mul_2exp (f, f, data[i]);
          mpf_sub_ui (f, f, 1L);

          got = mpf_get_d_2exp (&got_exp, f);
          if (got < 0.5 || got >= 1.0)
            {
              printf    ("mpf_get_d_2exp bad on 2**%lu-1\n", data[i]);
              printf    ("result out of range, expect 0.5 <= got < 1.0\n");
              printf    ("   rnd_mode = %d\n", rnd_mode);
              printf    ("   data[i]  = %lu\n", data[i]);
              mpf_trace ("   f    ", f);
              d_trace   ("   got  ", got);
              printf    ("   got exp  %ld\n", got_exp);
              abort();
            }
        }
    }

  mpf_clear (f);
  tests_hardware_setround (old_rnd_mode);
}


int
main (void)
{
  tests_start ();
  mp_trace_base = 16;

  check_onebit ();
  check_round ();

  tests_end ();
  exit (0);
}
