/* Test mpn_add_1 and mpn_sub_1.

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

#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"


#define M      GMP_NUMB_MAX
#define ASIZE  10
#define MAGIC  0x1234

#define SETUP()                         \
  do {                                  \
    refmpn_random (got, data[i].size);  \
    got[data[i].size] = MAGIC;          \
  } while (0)

#define SETUP_INPLACE()                                 \
  do {                                                  \
    refmpn_copyi (got, data[i].src, data[i].size);      \
    got[data[i].size] = MAGIC;                          \
  } while (0)

#define VERIFY(name)                            \
  do {                                          \
    verify (name, i, data[i].src, data[i].n,    \
            got_c, data[i].want_c,              \
            got, data[i].want, data[i].size);   \
  } while (0)

typedef mp_limb_t (*mpn_aors_1_t) (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);
mpn_aors_1_t fudge (mpn_aors_1_t);


void
verify (const char *name, int i,
        mp_srcptr src, mp_limb_t n,
        mp_limb_t got_c, mp_limb_t want_c,
        mp_srcptr got, mp_srcptr want, mp_size_t size)
{
  if (got[size] != MAGIC)
    {
      printf ("Overwrite at %s i=%d\n", name, i);
      abort ();
    }

  if (got_c != want_c || ! refmpn_equal_anynail (got, want, size))
    {
      printf ("Wrong at %s i=%d size=%ld\n", name, i, size);
      mpn_trace ("   src", src,  size);
      mpn_trace ("     n", &n,   (mp_size_t) 1);
      mpn_trace ("   got", got,  size);
      mpn_trace ("  want", want, size);
      mpn_trace (" got c", &got_c,  (mp_size_t) 1);
      mpn_trace ("want c", &want_c, (mp_size_t) 1);
      abort ();
    }
}


void
check_add_1 (void)
{
  static const struct {
    mp_size_t        size;
    mp_limb_t        n;
    const mp_limb_t  src[ASIZE];
    mp_limb_t        want_c;
    const mp_limb_t  want[ASIZE];
  } data[] = {
    { 1, 0, { 0 },  0, { 0 } },
    { 1, 0, { 1 },  0, { 1 } },
    { 1, 1, { 0 },  0, { 1 } },
    { 1, 0, { M },  0, { M } },
    { 1, M, { 0 },  0, { M } },
    { 1, 1, { 123 }, 0, { 124 } },

    { 1, 1, { M },  1, { 0 } },
    { 1, M, { 1 },  1, { 0 } },
    { 1, M, { M },  1, { M-1 } },

    { 2, 0, { 0, 0 },  0, { 0, 0 } },
    { 2, 0, { 1, 0 },  0, { 1, 0 } },
    { 2, 1, { 0, 0 },  0, { 1, 0 } },
    { 2, 0, { M, 0 },  0, { M, 0 } },
    { 2, M, { 0, 0 },  0, { M, 0 } },
    { 2, 1, { M, 0 },  0, { 0, 1 } },
    { 2, M, { 1, 0 },  0, { 0, 1 } },
    { 2, M, { M, 0 },  0, { M-1, 1 } },
    { 2, M, { M, 0 },  0, { M-1, 1 } },

    { 2, 1, { M, M },  1, { 0, 0 } },
    { 2, M, { 1, M },  1, { 0, 0 } },
    { 2, M, { M, M },  1, { M-1, 0 } },
    { 2, M, { M, M },  1, { M-1, 0 } },

    { 3, 1, { M, M, M },  1, { 0, 0, 0 } },
    { 3, M, { 1, M, M },  1, { 0, 0, 0 } },
    { 3, M, { M, M, M },  1, { M-1, 0, 0 } },
    { 3, M, { M, M, M },  1, { M-1, 0, 0 } },

    { 4, 1, { M, M, M, M },  1, { 0, 0, 0, 0 } },
    { 4, M, { 1, M, M, M },  1, { 0, 0, 0, 0 } },
    { 4, M, { M, M, M, M },  1, { M-1, 0, 0, 0 } },
    { 4, M, { M, M, M, M },  1, { M-1, 0, 0, 0 } },

    { 4, M, { M, 0,   M, M },  0, { M-1, 1, M, M } },
    { 4, M, { M, M-1, M, M },  0, { M-1, M, M, M } },

    { 4, M, { M, M, 0,   M },  0, { M-1, 0, 1, M } },
    { 4, M, { M, M, M-1, M },  0, { M-1, 0, M, M } },
  };

  mp_limb_t  got[ASIZE];
  mp_limb_t  got_c;
  /* mpn_sec_add_a_itch(n) <= n */
  mp_limb_t  scratch[ASIZE];
  int        i;

  for (i = 0; i < numberof (data); i++)
    {
      SETUP ();
      got_c = mpn_add_1 (got, data[i].src, data[i].size, data[i].n);
      VERIFY ("check_add_1 (separate)");

      SETUP_INPLACE ();
      got_c = mpn_add_1 (got, got, data[i].size, data[i].n);
      VERIFY ("check_add_1 (in-place)");

      SETUP ();
      scratch [mpn_sec_add_1_itch(data[i].size)] = MAGIC;
      got_c = mpn_sec_add_1 (got, data[i].src, data[i].size, data[i].n, scratch);
      got_c ^= scratch [mpn_sec_add_1_itch(data[i].size)] ^ MAGIC;
      VERIFY ("check_sec_add_1 (separate)");

      SETUP_INPLACE ();
      got_c = mpn_sec_add_1 (got, got, data[i].size, data[i].n, scratch);
      VERIFY ("check_sec_add_1 (in-place)");

      if (data[i].n == 1)
        {
          SETUP ();
          got_c = mpn_add_1 (got, data[i].src, data[i].size, CNST_LIMB(1));
          VERIFY ("check_add_1 (separate, const 1)");

          SETUP_INPLACE ();
          got_c = mpn_add_1 (got, got, data[i].size, CNST_LIMB(1));
          VERIFY ("check_add_1 (in-place, const 1)");

          SETUP ();
          got_c = mpn_sec_add_1 (got, data[i].src, data[i].size,
				 CNST_LIMB(1), scratch);
          VERIFY ("check_sec_add_1 (separate, const 1)");

          SETUP_INPLACE ();
          got_c = mpn_sec_add_1 (got, got, data[i].size,
				 CNST_LIMB(1), scratch);
          VERIFY ("check_sec_add_1 (in-place, const 1)");
        }

      /* Same again on functions, not inlines. */
      SETUP ();
      got_c = (*fudge(mpn_add_1)) (got, data[i].src, data[i].size, data[i].n);
      VERIFY ("check_add_1 (function, separate)");

      SETUP_INPLACE ();
      got_c = (*fudge(mpn_add_1)) (got, got, data[i].size, data[i].n);
      VERIFY ("check_add_1 (function, in-place)");
    }
}

void
check_sub_1 (void)
{
  static const struct {
    mp_size_t        size;
    mp_limb_t        n;
    const mp_limb_t  src[ASIZE];
    mp_limb_t        want_c;
    const mp_limb_t  want[ASIZE];
  } data[] = {
    { 1, 0, { 0 },  0, { 0 } },
    { 1, 0, { 1 },  0, { 1 } },
    { 1, 1, { 1 },  0, { 0 } },
    { 1, 0, { M },  0, { M } },
    { 1, 1, { M },  0, { M-1 } },
    { 1, 1, { 123 }, 0, { 122 } },

    { 1, 1, { 0 },  1, { M } },
    { 1, M, { 0 },  1, { 1 } },

    { 2, 0, { 0, 0 },  0, { 0, 0 } },
    { 2, 0, { 1, 0 },  0, { 1, 0 } },
    { 2, 1, { 1, 0 },  0, { 0, 0 } },
    { 2, 0, { M, 0 },  0, { M, 0 } },
    { 2, 1, { M, 0 },  0, { M-1, 0 } },
    { 2, 1, { 123, 0 }, 0, { 122, 0 } },

    { 2, 1, { 0, 0 },  1, { M, M } },
    { 2, M, { 0, 0 },  1, { 1, M } },

    { 3, 0, { 0,   0, 0 },  0, { 0,   0, 0 } },
    { 3, 0, { 123, 0, 0 },  0, { 123, 0, 0 } },

    { 3, 1, { 0, 0, 0 },  1, { M, M, M } },
    { 3, M, { 0, 0, 0 },  1, { 1, M, M } },

    { 4, 1, { 0, 0, 0, 0 },  1, { M, M, M, M } },
    { 4, M, { 0, 0, 0, 0 },  1, { 1, M, M, M } },

    { 4, 1, { 0, 0, 1,   42 },  0, { M, M, 0,   42 } },
    { 4, M, { 0, 0, 123, 24 },  0, { 1, M, 122, 24 } },
  };

  mp_limb_t  got[ASIZE];
  mp_limb_t  got_c;
  /* mpn_sec_sub_1_itch(n) <= n */
  mp_limb_t  scratch[ASIZE];
  int        i;

  for (i = 0; i < numberof (data); i++)
    {
      SETUP ();
      got_c = mpn_sub_1 (got, data[i].src, data[i].size, data[i].n);
      VERIFY ("check_sub_1 (separate)");

      SETUP_INPLACE ();
      got_c = mpn_sub_1 (got, got, data[i].size, data[i].n);
      VERIFY ("check_sub_1 (in-place)");

      SETUP ();
      scratch [mpn_sec_sub_1_itch(data[i].size)] = MAGIC;
      got_c = mpn_sec_sub_1 (got, data[i].src, data[i].size, data[i].n, scratch);
      got_c ^= scratch [mpn_sec_sub_1_itch(data[i].size)] ^ MAGIC;
      VERIFY ("check_sec_sub_1 (separate)");

      SETUP_INPLACE ();
      got_c = mpn_sec_sub_1 (got, got, data[i].size, data[i].n, scratch);
      VERIFY ("check_sec_sub_1 (in-place)");

      if (data[i].n == 1)
        {
          SETUP ();
          got_c = mpn_sub_1 (got, data[i].src, data[i].size, CNST_LIMB(1));
          VERIFY ("check_sub_1 (separate, const 1)");

          SETUP_INPLACE ();
          got_c = mpn_sub_1 (got, got, data[i].size, CNST_LIMB(1));
          VERIFY ("check_sub_1 (in-place, const 1)");

          SETUP ();
          got_c = mpn_sec_sub_1 (got, data[i].src, data[i].size,
				 CNST_LIMB(1), scratch);
          VERIFY ("check_sec_sub_1 (separate, const 1)");

          SETUP_INPLACE ();
          got_c = mpn_sec_sub_1 (got, got, data[i].size,
				 CNST_LIMB(1), scratch);
          VERIFY ("check_sec_sub_1 (in-place, const 1)");
        }

      /* Same again on functions, not inlines. */
      SETUP ();
      got_c = (*fudge(mpn_sub_1)) (got, data[i].src, data[i].size, data[i].n);
      VERIFY ("check_sub_1 (function, separate)");

      SETUP_INPLACE ();
      got_c = (*fudge(mpn_sub_1)) (got, got, data[i].size, data[i].n);
      VERIFY ("check_sub_1 (function, in-place)");
    }
}

/* Try to prevent the optimizer inlining. */
mpn_aors_1_t
fudge (mpn_aors_1_t f)
{
  return f;
}

int
main (void)
{
  tests_start ();
  mp_trace_base = -16;

  check_add_1 ();
  check_sub_1 ();

  tests_end ();
  exit (0);
}
