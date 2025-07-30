/* Test fat binary setups.

Copyright 2003, 2012 Free Software Foundation, Inc.

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
#include "longlong.h"
#include "tests.h"


/* In this program we're aiming to pick up certain subtle problems that
   might creep into a fat binary.

   1. We want to ensure the application entry point routines like
      __gmpn_add_n dispatch to the correct field of __gmpn_cpuvec.

      Note that these routines are not exercised as a side effect of other
      tests (eg. the mpz routines).  Internally the fields of __gmpn_cpuvec
      are used directly, so we need to write test code explicitly calling
      the mpn functions, like an application will have.

   2. We want to ensure the initial __gmpn_cpuvec data has the initializer
      function pointers in the correct fields, and that those initializer
      functions dispatch to their correct corresponding field once
      initialization has been done.

      Only one of the initializer routines executes in a normal program,
      since that routine sets all the pointers to actual mpn functions.  We
      forcibly reset __gmpn_cpuvec so we can run each.

   In both cases for the above, the data put through the functions is
   nothing special, just enough to verify that for instance an add_n is
   really doing an add_n and has not for instance mistakenly gone to sub_n
   or something.

   The loop around each test will exercise the initializer routine on the
   first iteration, and the dispatcher routine on the second.

   If the dispatcher and/or initializer routines are generated mechanically
   via macros (eg. mpn/x86/fat/fat_entry.asm) then there shouldn't be too
   much risk of them going wrong, provided the structure layout is correctly
   expressed.  But if they're in C then it's good to guard against typos in
   what is rather repetitive code.  The initializer data for __gmpn_cpuvec
   in fat.c is always done by hand and is likewise a bit repetitive.  */


/* dummies when not a fat binary */
#if ! WANT_FAT_BINARY
struct cpuvec_t {
  int  dummy;
};
struct cpuvec_t __gmpn_cpuvec;
#define ITERATE_FAT_THRESHOLDS()  do { } while (0)
#endif

/* saved from program startup */
struct cpuvec_t  initial_cpuvec;

void
check_functions (void)
{
  mp_limb_t  wp[2], xp[2], yp[2], r;
  int  i;

  memcpy (&__gmpn_cpuvec, &initial_cpuvec, sizeof (__gmpn_cpuvec));
  for (i = 0; i < 2; i++)
    {
      xp[0] = 123;
      yp[0] = 456;
      mpn_add_n (wp, xp, yp, (mp_size_t) 1);
      ASSERT_ALWAYS (wp[0] == 579);
    }

  memcpy (&__gmpn_cpuvec, &initial_cpuvec, sizeof (__gmpn_cpuvec));
  for (i = 0; i < 2; i++)
    {
      xp[0] = 123;
      wp[0] = 456;
      r = mpn_addmul_1 (wp, xp, (mp_size_t) 1, CNST_LIMB(2));
      ASSERT_ALWAYS (wp[0] == 702);
      ASSERT_ALWAYS (r == 0);
    }

#if HAVE_NATIVE_mpn_copyd
  memcpy (&__gmpn_cpuvec, &initial_cpuvec, sizeof (__gmpn_cpuvec));
  for (i = 0; i < 2; i++)
    {
      xp[0] = 123;
      xp[1] = 456;
      mpn_copyd (xp+1, xp, (mp_size_t) 1);
      ASSERT_ALWAYS (xp[1] == 123);
    }
#endif

#if HAVE_NATIVE_mpn_copyi
  memcpy (&__gmpn_cpuvec, &initial_cpuvec, sizeof (__gmpn_cpuvec));
  for (i = 0; i < 2; i++)
    {
      xp[0] = 123;
      xp[1] = 456;
      mpn_copyi (xp, xp+1, (mp_size_t) 1);
      ASSERT_ALWAYS (xp[0] == 456);
    }
#endif

  memcpy (&__gmpn_cpuvec, &initial_cpuvec, sizeof (__gmpn_cpuvec));
  for (i = 0; i < 2; i++)
    {
      xp[0] = 1605;
      mpn_divexact_1 (wp, xp, (mp_size_t) 1, CNST_LIMB(5));
      ASSERT_ALWAYS (wp[0] == 321);
    }

  memcpy (&__gmpn_cpuvec, &initial_cpuvec, sizeof (__gmpn_cpuvec));
  for (i = 0; i < 2; i++)
    {
      xp[0] = 1296;
      r = mpn_divexact_by3c (wp, xp, (mp_size_t) 1, CNST_LIMB(0));
      ASSERT_ALWAYS (wp[0] == 432);
      ASSERT_ALWAYS (r == 0);
    }

  memcpy (&__gmpn_cpuvec, &initial_cpuvec, sizeof (__gmpn_cpuvec));
  for (i = 0; i < 2; i++)
    {
      xp[0] = 287;
      r = mpn_divrem_1 (wp, (mp_size_t) 1, xp, (mp_size_t) 1, CNST_LIMB(7));
      ASSERT_ALWAYS (wp[1] == 41);
      ASSERT_ALWAYS (wp[0] == 0);
      ASSERT_ALWAYS (r == 0);
    }

  memcpy (&__gmpn_cpuvec, &initial_cpuvec, sizeof (__gmpn_cpuvec));
  for (i = 0; i < 2; i++)
    {
      xp[0] = 12;
      r = mpn_gcd_1 (xp, (mp_size_t) 1, CNST_LIMB(9));
      ASSERT_ALWAYS (r == 3);
    }

  memcpy (&__gmpn_cpuvec, &initial_cpuvec, sizeof (__gmpn_cpuvec));
  for (i = 0; i < 2; i++)
    {
      xp[0] = 0x1001;
      mpn_lshift (wp, xp, (mp_size_t) 1, 1);
      ASSERT_ALWAYS (wp[0] == 0x2002);
    }

  memcpy (&__gmpn_cpuvec, &initial_cpuvec, sizeof (__gmpn_cpuvec));
  for (i = 0; i < 2; i++)
    {
      xp[0] = 14;
      r = mpn_mod_1 (xp, (mp_size_t) 1, CNST_LIMB(4));
      ASSERT_ALWAYS (r == 2);
    }

#if (GMP_NUMB_BITS % 4) == 0
  memcpy (&__gmpn_cpuvec, &initial_cpuvec, sizeof (__gmpn_cpuvec));
  for (i = 0; i < 2; i++)
    {
      int  bits = (GMP_NUMB_BITS / 4) * 3;
      mp_limb_t  mod = (CNST_LIMB(1) << bits) - 1;
      mp_limb_t  want = GMP_NUMB_MAX % mod;
      xp[0] = GMP_NUMB_MAX;
      r = mpn_mod_34lsub1 (xp, (mp_size_t) 1);
      ASSERT_ALWAYS (r % mod == want);
    }
#endif

  /*   DECL_modexact_1c_odd ((*modexact_1c_odd)); */

  memcpy (&__gmpn_cpuvec, &initial_cpuvec, sizeof (__gmpn_cpuvec));
  for (i = 0; i < 2; i++)
    {
      xp[0] = 14;
      r = mpn_mul_1 (wp, xp, (mp_size_t) 1, CNST_LIMB(4));
      ASSERT_ALWAYS (wp[0] == 56);
      ASSERT_ALWAYS (r == 0);
    }

  memcpy (&__gmpn_cpuvec, &initial_cpuvec, sizeof (__gmpn_cpuvec));
  for (i = 0; i < 2; i++)
    {
      xp[0] = 5;
      yp[0] = 7;
      mpn_mul_basecase (wp, xp, (mp_size_t) 1, yp, (mp_size_t) 1);
      ASSERT_ALWAYS (wp[0] == 35);
      ASSERT_ALWAYS (wp[1] == 0);
    }

  memcpy (&__gmpn_cpuvec, &initial_cpuvec, sizeof (__gmpn_cpuvec));
  for (i = 0; i < 2; i++)
    {
      xp[0] = 5;
      yp[0] = 7;
      mpn_mullo_basecase (wp, xp, yp, (mp_size_t) 1);
      ASSERT_ALWAYS (wp[0] == 35);
    }

#if HAVE_NATIVE_mpn_preinv_divrem_1 && GMP_NAIL_BITS == 0
  memcpy (&__gmpn_cpuvec, &initial_cpuvec, sizeof (__gmpn_cpuvec));
  for (i = 0; i < 2; i++)
    {
      xp[0] = 0x101;
      r = mpn_preinv_divrem_1 (wp, (mp_size_t) 1, xp, (mp_size_t) 1,
                               GMP_LIMB_HIGHBIT,
                               refmpn_invert_limb (GMP_LIMB_HIGHBIT), 0);
      ASSERT_ALWAYS (wp[0] == 0x202);
      ASSERT_ALWAYS (wp[1] == 0);
      ASSERT_ALWAYS (r == 0);
    }
#endif

#if GMP_NAIL_BITS == 0
  memcpy (&__gmpn_cpuvec, &initial_cpuvec, sizeof (__gmpn_cpuvec));
  for (i = 0; i < 2; i++)
    {
      xp[0] = GMP_LIMB_HIGHBIT+123;
      r = mpn_preinv_mod_1 (xp, (mp_size_t) 1, GMP_LIMB_HIGHBIT,
                            refmpn_invert_limb (GMP_LIMB_HIGHBIT));
      ASSERT_ALWAYS (r == 123);
    }
#endif

  memcpy (&__gmpn_cpuvec, &initial_cpuvec, sizeof (__gmpn_cpuvec));
  for (i = 0; i < 2; i++)
    {
      xp[0] = 0x8008;
      mpn_rshift (wp, xp, (mp_size_t) 1, 1);
      ASSERT_ALWAYS (wp[0] == 0x4004);
    }

  memcpy (&__gmpn_cpuvec, &initial_cpuvec, sizeof (__gmpn_cpuvec));
  for (i = 0; i < 2; i++)
    {
      xp[0] = 5;
      mpn_sqr_basecase (wp, xp, (mp_size_t) 1);
      ASSERT_ALWAYS (wp[0] == 25);
      ASSERT_ALWAYS (wp[1] == 0);
    }

  memcpy (&__gmpn_cpuvec, &initial_cpuvec, sizeof (__gmpn_cpuvec));
  for (i = 0; i < 2; i++)
    {
      xp[0] = 999;
      yp[0] = 666;
      mpn_sub_n (wp, xp, yp, (mp_size_t) 1);
      ASSERT_ALWAYS (wp[0] == 333);
    }

  memcpy (&__gmpn_cpuvec, &initial_cpuvec, sizeof (__gmpn_cpuvec));
  for (i = 0; i < 2; i++)
    {
      xp[0] = 123;
      wp[0] = 456;
      r = mpn_submul_1 (wp, xp, (mp_size_t) 1, CNST_LIMB(2));
      ASSERT_ALWAYS (wp[0] == 210);
      ASSERT_ALWAYS (r == 0);
    }
}

/* Expect the first use of each fat threshold to invoke the necessary
   initialization.  */
void
check_thresholds (void)
{
#define ITERATE(name,field)                                             \
  do {                                                                  \
    __gmpn_cpuvec_initialized = 0;					\
    memcpy (&__gmpn_cpuvec, &initial_cpuvec, sizeof (__gmpn_cpuvec));   \
    ASSERT_ALWAYS (name != 0);                                          \
    ASSERT_ALWAYS (name == __gmpn_cpuvec.field);                        \
    ASSERT_ALWAYS (__gmpn_cpuvec_initialized);                          \
  } while (0)

  ITERATE_FAT_THRESHOLDS ();
}


int
main (void)
{
  memcpy (&initial_cpuvec, &__gmpn_cpuvec, sizeof (__gmpn_cpuvec));

  tests_start ();

  check_functions ();
  check_thresholds ();

  tests_end ();
  exit (0);
}
