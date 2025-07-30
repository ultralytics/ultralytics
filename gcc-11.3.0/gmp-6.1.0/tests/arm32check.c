/* ARM32 calling conventions checking.

Copyright 2000, 2001, 2004, 2007 Free Software Foundation, Inc.

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
#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"


/* Vector if constants and register values.  */
mp_limb_t calling_conventions_values[29] =
{
  0x12345678,	/*  0 want_r4 */
  0x87654321,	/*  1 want_r5 */
  0x89ABCDEF,	/*  2 want_r6 */
  0xFEDCBA98,	/*  3 want_r7 */
  0xDEADBEEF,	/*  4 want_r8 */
  0xBADECAFE,	/*  5 want_r9 */
  0xFFEEDDCC,	/*  6 want_r10 */
  0xBBAA9988,	/*  7 want_r11 */

  0x00000000,	/*  8 save_r4 */
  0x00000000,	/*  9 save_r5 */
  0x00000000,	/* 10 save_r6 */
  0x00000000,	/* 11 save_r7 */
  0x00000000,	/* 12 save_r8 */
  0x00000000,	/* 13 save_r9 */
  0x00000000,	/* 14 save_r10 */
  0x00000000,	/* 15 save_r11 */
  0x00000000,	/* 16 save_r14 */

  0x00000000,	/* 17 got_r4 */
  0x00000000,	/* 18 got_r5 */
  0x00000000,	/* 19 got_r6 */
  0x00000000,	/* 20 got_r7 */
  0x00000000,	/* 21 got_r8 */
  0x00000000,	/* 22 got_r9 */
  0x00000000,	/* 23 got_r10 */
  0x00000000,	/* 24 got_r11 */

  0x00112233,	/* 25 junk_r0 */
  0x44556677,	/* 26 junk_r1 */
  0x12344321,	/* 27 junk_r2 */
  0x56788765,	/* 28 junk_r3 */
};

/* Index starts for various regions in above vector.  */
#define WANT_CALLEE_SAVES	0
#define SAVE_CALLEE_SAVES	8
#define RETADDR			16
#define GOT_CALLEE_SAVES	17
#define JUNK_PARAMS		25

/* Return 1 if ok, 0 if not */

int
calling_conventions_check (void)
{
  const char  *header = "Violated calling conventions:\n";
  int  ret = 1;
  int i;

#define CHECK(callreg, regnum, value)					\
  if (callreg != value)							\
    {									\
      printf ("%s   r%d	got 0x%08lX want 0x%08lX\n",			\
	      header, regnum, callreg, value);				\
      header = "";							\
      ret = 0;								\
    }

  for (i = 0; i < 8; i++)
    {
      CHECK (calling_conventions_values[GOT_CALLEE_SAVES + i],
	     i + 4,
	     calling_conventions_values[WANT_CALLEE_SAVES + i]);
    }

  return ret;
}
