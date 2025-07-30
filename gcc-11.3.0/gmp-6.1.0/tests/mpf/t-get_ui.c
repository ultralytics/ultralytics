/* Exercise mpf_get_ui.

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
check_limbdata (void)
{
#define M  GMP_NUMB_MAX

  static const struct {
    mp_exp_t       exp;
    mp_size_t      size;
    mp_limb_t      d[10];
    unsigned long  want;

  } data[] = {

    /* in the comments here, a "_" indicates a digit (ie. limb) position not
       included in the d data, and therefore zero */

    { 0, 0, { 0 }, 0L },    /* 0 */

    { 1,  1, { 1 }, 1L },   /* 1 */
    { 1, -1, { 1 }, 1L },   /* -1 */

    { 0,  1, { 1 }, 0L },   /* .1 */
    { 0, -1, { 1 }, 0L },   /* -.1 */

    { -1,  1, { 1 }, 0L },  /* ._1 */
    { -1, -1, { 1 }, 0L },  /* -._1 */

    { -999,          1, { 1 }, 0L },   /* .___1 small */
    { MP_EXP_T_MIN,  1, { 1 }, 0L },   /* .____1 very small */

    { 999,          1, { 1 }, 0L },    /* 1____. big */
    { MP_EXP_T_MAX, 1, { 1 }, 0L },    /* 1_____. very big */

    { 1, 2, { 999, 2 }, 2L },                  /* 2.9 */
    { 5, 8, { 7, 8, 9, 3, 0, 0, 0, 1 }, 3L },  /* 10003.987 */

    { 2, 2, { M, M },    ULONG_MAX }, /* FF. */
    { 2, 2, { M, M, M }, ULONG_MAX }, /* FF.F */
    { 3, 3, { M, M, M }, ULONG_MAX }, /* FFF. */

#if GMP_NUMB_BITS >= BITS_PER_ULONG
    /* normal case, numb bigger than long */
    { 2,  1, { 1 },    0L },      /* 1_. */
    { 2,  2, { 0, 1 }, 0L },      /* 10. */
    { 2,  2, { 999, 1 }, 999L },  /* 19. */
    { 3,  2, { 999, 1 }, 0L },    /* 19_. */

#else
    /* nails case, numb smaller than long */
    { 2,  1, { 1 }, 1L << GMP_NUMB_BITS },  /* 1_. */
    { 3,  1, { 1 }, 0L },                   /* 1__. */

    { 2,  2, { 99, 1 },    99L + (1L << GMP_NUMB_BITS) },  /* 19. */
    { 3,  2, { 1, 99 },    1L << GMP_NUMB_BITS },          /* 91_. */
    { 3,  3, { 0, 1, 99 }, 1L << GMP_NUMB_BITS },          /* 910. */

#endif
  };

  mpf_t          f;
  unsigned long  got;
  int            i;
  mp_limb_t      buf[20 + numberof(data[i].d)];

  for (i = 0; i < numberof (data); i++)
    {
      refmpn_fill (buf, 10, CNST_LIMB(0xDEADBEEF));
      refmpn_copy (buf+10, data[i].d, ABS(data[i].size));
      refmpn_fill (buf+10+ABS(data[i].size), 10, CNST_LIMB(0xDEADBEEF));

      PTR(f) = buf+10;
      EXP(f) = data[i].exp;
      SIZ(f) = data[i].size;
      PREC(f) = numberof (data[i].d);
      MPF_CHECK_FORMAT (f);

      got = mpf_get_ui (f);
      if (got != data[i].want)
	{
	  printf    ("mpf_get_ui wrong at limb data[%d]\n", i);
	  mpf_trace ("  f", f);
	  mpn_trace ("  d", data[i].d, data[i].size);
	  printf    ("  size %ld\n", (long) data[i].size);
	  printf    ("  exp %ld\n", (long) data[i].exp);
	  printf    ("  got   %lu (0x%lX)\n", got, got);
	  printf    ("  want  %lu (0x%lX)\n", data[i].want, data[i].want);
	  abort();
	}
    }
}

int
main (void)
{
  tests_start ();
  mp_trace_base = 16;

  check_limbdata ();

  tests_end ();
  exit (0);
}
