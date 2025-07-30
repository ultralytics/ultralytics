/* Test BSWAP_LIMB.

Copyright 2002 Free Software Foundation, Inc.

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
main (void)
{
  mp_limb_t  src, want, got;
  int        i;

  tests_start ();
  mp_trace_base = -16;

  for (i = 0; i < 1000; i++)
    {
      mpn_random (&src, (mp_size_t) 1);

      want = ref_bswap_limb (src);

      BSWAP_LIMB (got, src);
      if (got != want)
        {
          printf ("BSWAP_LIMB wrong result\n");
        error:
          mpn_trace ("  src ", &src,  (mp_size_t) 1);
          mpn_trace ("  want", &want, (mp_size_t) 1);
          mpn_trace ("  got ", &got,  (mp_size_t) 1);
          abort ();
        }

      BSWAP_LIMB_FETCH (got, &src);
      if (got != want)
        {
          printf ("BSWAP_LIMB_FETCH wrong result\n");
          goto error;
        }

      BSWAP_LIMB_STORE (&got, src);
      if (got != want)
        {
          printf ("BSWAP_LIMB_STORE wrong result\n");
          goto error;
        }
    }

  tests_end ();
  exit (0);
}
