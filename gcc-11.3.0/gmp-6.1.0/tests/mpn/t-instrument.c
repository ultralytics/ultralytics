/* Test assembler support for --enable-profiling=instrument.

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
#include "longlong.h"
#include "tests.h"


#if WANT_PROFILING_INSTRUMENT

/* This program exercises each mpn routine that might be implemented in
   assembler.  It ensures the __cyg_profile_func_enter and exit calls have
   come out right, and that in the x86 code "ret_internal" is correctly used
   for PIC setups.  */


/* Changes to enter_seen done by __cyg_profile_func_enter are essentially
   unknown to the optimizer, so must use volatile.  */
volatile int  enter_seen;

/* Dummy used to stop various calls going dead. */
unsigned long  notdead;

const char     *name = "<none>";
int  old_ncall;

struct {
  void  *this_fn;
  void  *call_site;
} call[100];
int  ncall;


void __cyg_profile_func_enter (void *, void *)
  __attribute__ ((no_instrument_function));

void
__cyg_profile_func_enter (void *this_fn, void *call_site)
{
#if 0
  printf ("%24s %p %p\n", name, this_fn, call_site);
#endif
  ASSERT_ALWAYS (ncall >= 0);
  ASSERT_ALWAYS (ncall <= numberof (call));

  if (ncall >= numberof (call))
    {
      printf ("__cyg_profile_func_enter: oops, call stack full, from %s\n", name);
      abort ();
    }

  enter_seen = 1;
  call[ncall].this_fn = this_fn;
  call[ncall].call_site = call_site;
  ncall++;
}

void __cyg_profile_func_exit (void *, void *)
  __attribute__ ((no_instrument_function));

void
__cyg_profile_func_exit  (void *this_fn, void *call_site)
{
  ASSERT_ALWAYS (ncall >= 0);
  ASSERT_ALWAYS (ncall <= numberof (call));

  if (ncall == 0)
    {
      printf ("__cyg_profile_func_exit: call stack empty, from %s\n", name);
      abort ();
    }

  ncall--;
  if (this_fn != call[ncall].this_fn || call_site != call[ncall].call_site)
    {
      printf ("__cyg_profile_func_exit: unbalanced this_fn/call_site from %s\n", name);
      printf ("  this_fn got  %p\n", this_fn);
      printf ("          want %p\n", call[ncall].this_fn);
      printf ("  call_site got  %p\n", call_site);
      printf ("            want %p\n", call[ncall].call_site);
      abort ();
    }
}


void
pre (const char *str)
{
  name = str;
  enter_seen = 0;
  old_ncall = ncall;
}

void
post (void)
{
  if (! enter_seen)
    {
      printf ("did not reach __cyg_profile_func_enter from %s\n", name);
      abort ();
    }

  if (ncall != old_ncall)
    {
      printf ("unbalance enter/exit calls from %s\n", name);
      printf ("  ncall     %d\n", ncall);
      printf ("  old_ncall %d\n", old_ncall);
      abort ();
    }
}

void
check (void)
{
  mp_limb_t  wp[100], xp[100], yp[100];
  mp_size_t  size = 100;

  refmpn_zero (xp, size);
  refmpn_zero (yp, size);
  refmpn_zero (wp, size);

  pre ("mpn_add_n");
  mpn_add_n (wp, xp, yp, size);
  post ();

#if HAVE_NATIVE_mpn_add_nc
  pre ("mpn_add_nc");
  mpn_add_nc (wp, xp, yp, size, CNST_LIMB(0));
  post ();
#endif

#if HAVE_NATIVE_mpn_addlsh1_n
  pre ("mpn_addlsh1_n");
  mpn_addlsh1_n (wp, xp, yp, size);
  post ();
#endif

#if HAVE_NATIVE_mpn_and_n
  pre ("mpn_and_n");
  mpn_and_n (wp, xp, yp, size);
  post ();
#endif

#if HAVE_NATIVE_mpn_andn_n
  pre ("mpn_andn_n");
  mpn_andn_n (wp, xp, yp, size);
  post ();
#endif

  pre ("mpn_addmul_1");
  mpn_addmul_1 (wp, xp, size, yp[0]);
  post ();

#if HAVE_NATIVE_mpn_addmul_1c
  pre ("mpn_addmul_1c");
  mpn_addmul_1c (wp, xp, size, yp[0], CNST_LIMB(0));
  post ();
#endif

#if HAVE_NATIVE_mpn_com
  pre ("mpn_com");
  mpn_com (wp, xp, size);
  post ();
#endif

#if HAVE_NATIVE_mpn_copyd
  pre ("mpn_copyd");
  mpn_copyd (wp, xp, size);
  post ();
#endif

#if HAVE_NATIVE_mpn_copyi
  pre ("mpn_copyi");
  mpn_copyi (wp, xp, size);
  post ();
#endif

  pre ("mpn_divexact_1");
  mpn_divexact_1 (wp, xp, size, CNST_LIMB(123));
  post ();

  pre ("mpn_divexact_by3c");
  mpn_divexact_by3c (wp, xp, size, CNST_LIMB(0));
  post ();

  pre ("mpn_divrem_1");
  mpn_divrem_1 (wp, (mp_size_t) 0, xp, size, CNST_LIMB(123));
  post ();

#if HAVE_NATIVE_mpn_divrem_1c
  pre ("mpn_divrem_1c");
  mpn_divrem_1c (wp, (mp_size_t) 0, xp, size, CNST_LIMB(123), CNST_LIMB(122));
  post ();
#endif

  pre ("mpn_gcd_1");
  xp[0] |= 1;
  notdead += (unsigned long) mpn_gcd_1 (xp, size, CNST_LIMB(123));
  post ();

  pre ("mpn_hamdist");
  notdead += mpn_hamdist (xp, yp, size);
  post ();

#if HAVE_NATIVE_mpn_ior_n
  pre ("mpn_ior_n");
  mpn_ior_n (wp, xp, yp, size);
  post ();
#endif

#if HAVE_NATIVE_mpn_iorn_n
  pre ("mpn_iorn_n");
  mpn_iorn_n (wp, xp, yp, size);
  post ();
#endif

  pre ("mpn_lshift");
  mpn_lshift (wp, xp, size, 1);
  post ();

  pre ("mpn_mod_1");
  notdead += mpn_mod_1 (xp, size, CNST_LIMB(123));
  post ();

#if HAVE_NATIVE_mpn_mod_1c
  pre ("mpn_mod_1c");
  notdead += mpn_mod_1c (xp, size, CNST_LIMB(123), CNST_LIMB(122));
  post ();
#endif

#if GMP_NUMB_BITS % 4 == 0
  pre ("mpn_mod_34lsub1");
  notdead += mpn_mod_34lsub1 (xp, size);
  post ();
#endif

  pre ("mpn_modexact_1_odd");
  notdead += mpn_modexact_1_odd (xp, size, CNST_LIMB(123));
  post ();

  pre ("mpn_modexact_1c_odd");
  notdead += mpn_modexact_1c_odd (xp, size, CNST_LIMB(123), CNST_LIMB(456));
  post ();

  pre ("mpn_mul_1");
  mpn_mul_1 (wp, xp, size, yp[0]);
  post ();

#if HAVE_NATIVE_mpn_mul_1c
  pre ("mpn_mul_1c");
  mpn_mul_1c (wp, xp, size, yp[0], CNST_LIMB(0));
  post ();
#endif

#if HAVE_NATIVE_mpn_mul_2
  pre ("mpn_mul_2");
  mpn_mul_2 (wp, xp, size-1, yp);
  post ();
#endif

  pre ("mpn_mul_basecase");
  mpn_mul_basecase (wp, xp, (mp_size_t) 3, yp, (mp_size_t) 3);
  post ();

#if HAVE_NATIVE_mpn_nand_n
  pre ("mpn_nand_n");
  mpn_nand_n (wp, xp, yp, size);
  post ();
#endif

#if HAVE_NATIVE_mpn_nior_n
  pre ("mpn_nior_n");
  mpn_nior_n (wp, xp, yp, size);
  post ();
#endif

  pre ("mpn_popcount");
  notdead += mpn_popcount (xp, size);
  post ();

  pre ("mpn_preinv_mod_1");
  notdead += mpn_preinv_mod_1 (xp, size, GMP_NUMB_MAX,
                               refmpn_invert_limb (GMP_NUMB_MAX));
  post ();

#if USE_PREINV_DIVREM_1 || HAVE_NATIVE_mpn_preinv_divrem_1
  pre ("mpn_preinv_divrem_1");
  mpn_preinv_divrem_1 (wp, (mp_size_t) 0, xp, size, GMP_NUMB_MAX,
                       refmpn_invert_limb (GMP_NUMB_MAX), 0);
  post ();
#endif

#if HAVE_NATIVE_mpn_rsh1add_n
  pre ("mpn_rsh1add_n");
  mpn_rsh1add_n (wp, xp, yp, size);
  post ();
#endif

#if HAVE_NATIVE_mpn_rsh1sub_n
  pre ("mpn_rsh1sub_n");
  mpn_rsh1sub_n (wp, xp, yp, size);
  post ();
#endif

  pre ("mpn_rshift");
  mpn_rshift (wp, xp, size, 1);
  post ();

  pre ("mpn_sqr_basecase");
  mpn_sqr_basecase (wp, xp, (mp_size_t) 3);
  post ();

  pre ("mpn_submul_1");
  mpn_submul_1 (wp, xp, size, yp[0]);
  post ();

#if HAVE_NATIVE_mpn_submul_1c
  pre ("mpn_submul_1c");
  mpn_submul_1c (wp, xp, size, yp[0], CNST_LIMB(0));
  post ();
#endif

  pre ("mpn_sub_n");
  mpn_sub_n (wp, xp, yp, size);
  post ();

#if HAVE_NATIVE_mpn_sub_nc
  pre ("mpn_sub_nc");
  mpn_sub_nc (wp, xp, yp, size, CNST_LIMB(0));
  post ();
#endif

#if HAVE_NATIVE_mpn_sublsh1_n
  pre ("mpn_sublsh1_n");
  mpn_sublsh1_n (wp, xp, yp, size);
  post ();
#endif

#if HAVE_NATIVE_mpn_udiv_qrnnd
  pre ("mpn_udiv_qrnnd");
  mpn_udiv_qrnnd (&wp[0], CNST_LIMB(122), xp[0], CNST_LIMB(123));
  post ();
#endif

#if HAVE_NATIVE_mpn_udiv_qrnnd_r
  pre ("mpn_udiv_qrnnd_r");
  mpn_udiv_qrnnd (CNST_LIMB(122), xp[0], CNST_LIMB(123), &wp[0]);
  post ();
#endif

#if HAVE_NATIVE_mpn_umul_ppmm
  pre ("mpn_umul_ppmm");
  mpn_umul_ppmm (&wp[0], xp[0], yp[0]);
  post ();
#endif

#if HAVE_NATIVE_mpn_umul_ppmm_r
  pre ("mpn_umul_ppmm_r");
  mpn_umul_ppmm_r (&wp[0], xp[0], yp[0]);
  post ();
#endif

#if HAVE_NATIVE_mpn_xor_n
  pre ("mpn_xor_n");
  mpn_xor_n (wp, xp, yp, size);
  post ();
#endif

#if HAVE_NATIVE_mpn_xnor_n
  pre ("mpn_xnor_n");
  mpn_xnor_n (wp, xp, yp, size);
  post ();
#endif
}


int
main (void)
{
  tests_start ();

  check ();

  tests_end ();
  exit (0);
}


#else /* ! WANT_PROFILING_INSTRUMENT */

int
main (void)
{
  exit (0);
}

#endif
