/* Test mpf_trunc, mpf_ceil, mpf_floor.

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


void
check_print (mpf_srcptr src, mpf_srcptr got, mpf_srcptr want)
{
  mp_trace_base = 16;
  mpf_trace ("src ", src);
  mpf_trace ("got ", got);
  mpf_trace ("want", want);

  printf ("got  size=%d exp=%ld\n", SIZ(got), EXP(got));
  mpn_trace ("     limbs=", PTR(got), (mp_size_t) ABSIZ(got));

  printf ("want size=%d exp=%ld\n", SIZ(want), EXP(want));
  mpn_trace ("     limbs=", PTR(want), (mp_size_t) ABSIZ(want));
}

void
check_one (mpf_srcptr src, mpf_srcptr trunc, mpf_srcptr ceil, mpf_srcptr floor)
{
  mpf_t  got;

  mpf_init2 (got, mpf_get_prec (trunc));
  ASSERT_ALWAYS (PREC(got) == PREC(trunc));
  ASSERT_ALWAYS (PREC(got) == PREC(ceil));
  ASSERT_ALWAYS (PREC(got) == PREC(floor));

#define CHECK_SEP(name, fun, want)              \
  mpf_set_ui (got, 54321L); /* initial junk */  \
  fun (got, src);                               \
  MPF_CHECK_FORMAT (got);                       \
  if (mpf_cmp (got, want) != 0)                 \
    {                                           \
	printf ("%s wrong\n", name);            \
	check_print (src, got, want);           \
	abort ();                               \
    }

  CHECK_SEP ("mpf_trunc", mpf_trunc, trunc);
  CHECK_SEP ("mpf_ceil",  mpf_ceil,  ceil);
  CHECK_SEP ("mpf_floor", mpf_floor, floor);

#define CHECK_INPLACE(name, fun, want)  \
  mpf_set (got, src);                   \
  fun (got, got);                       \
  MPF_CHECK_FORMAT (got);               \
  if (mpf_cmp (got, want) != 0)         \
    {                                   \
	printf ("%s wrong\n", name);    \
	check_print (src, got, want);   \
	abort ();                       \
    }

  CHECK_INPLACE ("mpf_trunc", mpf_trunc, trunc);

  /* Can't do these unconditionally in case truncation by mpf_set strips
     some low non-zero limbs which would have rounded the result.  */
  if (ABSIZ(src) <= PREC(trunc)+1)
    {
      CHECK_INPLACE ("mpf_ceil",  mpf_ceil,  ceil);
      CHECK_INPLACE ("mpf_floor", mpf_floor, floor);
    }

  mpf_clear (got);
}

void
check_all (mpf_ptr src, mpf_ptr trunc, mpf_ptr ceil, mpf_ptr floor)
{
  /* some of these values are generated with direct field assignments */
  MPF_CHECK_FORMAT (src);
  MPF_CHECK_FORMAT (trunc);
  MPF_CHECK_FORMAT (ceil);
  MPF_CHECK_FORMAT (floor);

  check_one (src, trunc, ceil, floor);

  mpf_neg (src,   src);
  mpf_neg (trunc, trunc);
  mpf_neg (ceil,  ceil);
  mpf_neg (floor, floor);
  check_one (src, trunc, floor, ceil);
}

void
check_various (void)
{
  mpf_t  src, trunc, ceil, floor;
  int    n, i;

  mpf_init2 (src, 512L);
  mpf_init2 (trunc, 256L);
  mpf_init2 (ceil,  256L);
  mpf_init2 (floor, 256L);

  /* 0 */
  mpf_set_ui (src, 0L);
  mpf_set_ui (trunc, 0L);
  mpf_set_ui (ceil, 0L);
  mpf_set_ui (floor, 0L);
  check_all (src, trunc, ceil, floor);

  /* 1 */
  mpf_set_ui (src, 1L);
  mpf_set_ui (trunc, 1L);
  mpf_set_ui (ceil, 1L);
  mpf_set_ui (floor, 1L);
  check_all (src, trunc, ceil, floor);

  /* 2^1024 */
  mpf_set_ui (src, 1L);
  mpf_mul_2exp (src,   src,   1024L);
  mpf_set (trunc, src);
  mpf_set (ceil,  src);
  mpf_set (floor, src);
  check_all (src, trunc, ceil, floor);

  /* 1/2^1024, fraction only */
  mpf_set_ui (src, 1L);
  mpf_div_2exp (src,  src, 1024L);
  mpf_set_si (trunc, 0L);
  mpf_set_si (ceil, 1L);
  mpf_set_si (floor, 0L);
  check_all (src, trunc, ceil, floor);

  /* 1/2 */
  mpf_set_ui (src, 1L);
  mpf_div_2exp (src,  src, 1L);
  mpf_set_si (trunc, 0L);
  mpf_set_si (ceil, 1L);
  mpf_set_si (floor, 0L);
  check_all (src, trunc, ceil, floor);

  /* 123+1/2^64 */
  mpf_set_ui (src, 1L);
  mpf_div_2exp (src,  src, 64L);
  mpf_add_ui (src,  src, 123L);
  mpf_set_si (trunc, 123L);
  mpf_set_si (ceil, 124L);
  mpf_set_si (floor, 123L);
  check_all (src, trunc, ceil, floor);

  /* integer of full prec+1 limbs, unchanged */
  n = PREC(trunc)+1;
  ASSERT_ALWAYS (n <= PREC(src)+1);
  EXP(src) = n;
  SIZ(src) = n;
  for (i = 0; i < SIZ(src); i++)
    PTR(src)[i] = i+100;
  mpf_set (trunc, src);
  mpf_set (ceil, src);
  mpf_set (floor, src);
  check_all (src, trunc, ceil, floor);

  /* full prec+1 limbs, 1 trimmed for integer */
  n = PREC(trunc)+1;
  ASSERT_ALWAYS (n <= PREC(src)+1);
  EXP(src) = n-1;
  SIZ(src) = n;
  for (i = 0; i < SIZ(src); i++)
    PTR(src)[i] = i+200;
  EXP(trunc) = n-1;
  SIZ(trunc) = n-1;
  for (i = 0; i < SIZ(trunc); i++)
    PTR(trunc)[i] = i+201;
  mpf_set (floor, trunc);
  mpf_add_ui (ceil, trunc, 1L);
  check_all (src, trunc, ceil, floor);

  /* prec+3 limbs, 2 trimmed for size */
  n = PREC(trunc)+3;
  ASSERT_ALWAYS (n <= PREC(src)+1);
  EXP(src) = n;
  SIZ(src) = n;
  for (i = 0; i < SIZ(src); i++)
    PTR(src)[i] = i+300;
  EXP(trunc) = n;
  SIZ(trunc) = n-2;
  for (i = 0; i < SIZ(trunc); i++)
    PTR(trunc)[i] = i+302;
  mpf_set (floor, trunc);
  mpf_set (ceil, trunc);
  PTR(ceil)[0]++;
  check_all (src, trunc, ceil, floor);

  /* prec+4 limbs, 2 trimmed for size, 1 trimmed for integer */
  n = PREC(trunc)+4;
  ASSERT_ALWAYS (n <= PREC(src)+1);
  EXP(src) = n-1;
  SIZ(src) = n;
  for (i = 0; i < SIZ(src); i++)
    PTR(src)[i] = i+400;
  EXP(trunc) = n-1;
  SIZ(trunc) = n-3;
  for (i = 0; i < SIZ(trunc); i++)
    PTR(trunc)[i] = i+403;
  mpf_set (floor, trunc);
  mpf_set (ceil, trunc);
  PTR(ceil)[0]++;
  check_all (src, trunc, ceil, floor);

  /* F.F, carry out of ceil */
  EXP(src) = 1;
  SIZ(src) = 2;
  PTR(src)[0] = GMP_NUMB_MAX;
  PTR(src)[1] = GMP_NUMB_MAX;
  EXP(trunc) = 1;
  SIZ(trunc) = 1;
  PTR(trunc)[0] = GMP_NUMB_MAX;
  mpf_set (floor, trunc);
  EXP(ceil) = 2;
  SIZ(ceil) = 1;
  PTR(ceil)[0] = 1;
  check_all (src, trunc, ceil, floor);

  /* FF.F, carry out of ceil */
  EXP(src) = 2;
  SIZ(src) = 3;
  PTR(src)[0] = GMP_NUMB_MAX;
  PTR(src)[1] = GMP_NUMB_MAX;
  PTR(src)[2] = GMP_NUMB_MAX;
  EXP(trunc) = 2;
  SIZ(trunc) = 2;
  PTR(trunc)[0] = GMP_NUMB_MAX;
  PTR(trunc)[1] = GMP_NUMB_MAX;
  mpf_set (floor, trunc);
  EXP(ceil) = 3;
  SIZ(ceil) = 1;
  PTR(ceil)[0] = 1;
  check_all (src, trunc, ceil, floor);

  mpf_clear (src);
  mpf_clear (trunc);
  mpf_clear (ceil);
  mpf_clear (floor);
}

int
main (void)
{
  tests_start ();

  check_various ();

  tests_end ();
  exit (0);
}
