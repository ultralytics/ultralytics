/* Test that routines allow reusing a source variable as destination.

   Test all relevant functions except:
	mpz_bin_ui
	mpz_nextprime
	mpz_mul_si
	mpz_addmul_ui (should this really allow a+=a*c?)

Copyright 1996, 1999-2002, 2009, 2012, 2013 Free Software Foundation, Inc.

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
#include <unistd.h>

#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"

#if __GMP_LIBGMP_DLL

/* FIXME: When linking to a DLL libgmp, mpz_add etc can't be used as
   initializers for global variables because they're effectively global
   variables (function pointers) themselves.  Perhaps calling a test
   function successively with mpz_add etc would be better.  */

int
main (void)
{
  printf ("Test suppressed for windows DLL\n");
  exit (0);
}


#else /* ! DLL_EXPORT */

void dump (const char *, mpz_t, mpz_t, mpz_t);

typedef void (*dss_func) (mpz_ptr, mpz_srcptr, mpz_srcptr);
typedef void (*dsi_func) (mpz_ptr, mpz_srcptr, unsigned long int);
typedef unsigned long int (*dsi_div_func) (mpz_ptr, mpz_srcptr, unsigned long int);
typedef unsigned long int (*ddsi_div_func) (mpz_ptr, mpz_ptr, mpz_srcptr, unsigned long int);
typedef void (*ddss_div_func) (mpz_ptr, mpz_ptr, mpz_srcptr, mpz_srcptr);
typedef void (*ds_func) (mpz_ptr, mpz_srcptr);


void
mpz_xinvert (mpz_ptr r, mpz_srcptr a, mpz_srcptr b)
{
  int res;
  res = mpz_invert (r, a, b);
  if (res == 0)
    mpz_set_ui (r, 0);
}

struct {
  dss_func fptr;
  const char *fname;
  int isdivision;
  int isslow;
} static dss[] =
  { { mpz_add,     "mpz_add",	  0, 0 },
    { mpz_sub,     "mpz_sub",	  0, 0 },
    { mpz_mul,     "mpz_mul",	  0, 0 },
    { mpz_cdiv_q,  "mpz_cdiv_q",  1, 0 },
    { mpz_cdiv_r,  "mpz_cdiv_r",  1, 0 },
    { mpz_fdiv_q,  "mpz_fdiv_q",  1, 0 },
    { mpz_fdiv_r,  "mpz_fdiv_r",  1, 0 },
    { mpz_tdiv_q,  "mpz_tdiv_q",  1, 0 },
    { mpz_tdiv_r,  "mpz_tdiv_r",  1, 0 },
    { mpz_mod,     "mpz_mod",	  1, 0 },
    { mpz_xinvert, "mpz_xinvert", 1, 1 },
    { mpz_gcd,     "mpz_gcd",	  0, 1 },
    { mpz_lcm,     "mpz_lcm",	  0, 1 },
    { mpz_and,     "mpz_and",	  0, 0 },
    { mpz_ior,     "mpz_ior",	  0, 0 },
    { mpz_xor,     "mpz_xor",     0, 0 }
  };


struct {
  dsi_func fptr;
  const char *fname;
  int mod;
} static dsi[] =
{
  /* Don't change order here without changing the code in main(). */
  { mpz_add_ui,         "mpz_add_ui",	     0 },
  { mpz_mul_ui,		"mpz_mul_ui",	     0 },
  { mpz_sub_ui,		"mpz_sub_ui",	     0 },
  { mpz_fdiv_q_2exp,    "mpz_fdiv_q_2exp",   0x1000 },
  { mpz_fdiv_r_2exp,    "mpz_fdiv_r_2exp",   0x1000 },
  { mpz_cdiv_q_2exp,    "mpz_cdiv_q_2exp",   0x1000 },
  { mpz_cdiv_r_2exp,    "mpz_cdiv_r_2exp",   0x1000 },
  { mpz_tdiv_q_2exp,    "mpz_tdiv_q_2exp",   0x1000 },
  { mpz_tdiv_r_2exp,    "mpz_tdiv_r_2exp",   0x1000 },
  { mpz_mul_2exp,	"mpz_mul_2exp",      0x100 },
  { mpz_pow_ui,		"mpz_pow_ui",        0x10 }
};

struct {
  dsi_div_func fptr;
  const char *fname;
} static dsi_div[] =
{
  { mpz_cdiv_q_ui,       "mpz_cdiv_q_ui" },
  { mpz_cdiv_r_ui,       "mpz_cdiv_r_ui" },
  { mpz_fdiv_q_ui,       "mpz_fdiv_q_ui" },
  { mpz_fdiv_r_ui,       "mpz_fdiv_r_ui" },
  { mpz_tdiv_q_ui,       "mpz_tdiv_q_ui" },
  { mpz_tdiv_r_ui,       "mpz_tdiv_r_ui" }
};

struct {
  ddsi_div_func fptr;
  const char *fname;
  int isslow;
} static ddsi_div[] =
{
  { mpz_cdiv_qr_ui,     "mpz_cdiv_qr_ui",    0 },
  { mpz_fdiv_qr_ui,     "mpz_fdiv_qr_ui",    0 },
  { mpz_tdiv_qr_ui,     "mpz_tdiv_qr_ui",    0 },
};


struct {
  ddss_div_func fptr;
  const char *fname;
  int isslow;
} static ddss_div[] =
{
  { mpz_cdiv_qr,  "mpz_cdiv_qr",    0 },
  { mpz_fdiv_qr,  "mpz_fdiv_qr",    0 },
  { mpz_tdiv_qr,  "mpz_tdiv_qr",    0 },
};

struct {
  ds_func fptr;
  const char *fname;
  int nonneg;
} static ds[] =
{
  { mpz_abs,    "mpz_abs",    0 },
  { mpz_com,    "mpz_com",    0 },
  { mpz_neg,    "mpz_neg",    0 },
  { mpz_sqrt,   "mpz_sqrt",   1 },
};

#define FAIL(class,indx,op1,op2,op3)					\
  do {									\
    dump (class[indx].fname, op1, op2, op3);				\
    exit (1);								\
  } while (0)

#define FAIL2(fname,op1,op2,op3)					\
  do {									\
    dump (#fname, op1, op2, op3);					\
    exit (1);								\
  } while (0)


#define INVOKE_RRS(desc,r1,r2,i1)					\
  do {									\
    if (pass & 1) _mpz_realloc (r1, ABSIZ(r1));				\
    if (pass & 2) _mpz_realloc (r2, ABSIZ(r2));				\
    (desc).fptr (r1, r2, i1);						\
  } while (0)
#define INVOKE_RS(desc,r1,i1)						\
  do {									\
    if (pass & 1) _mpz_realloc (r1, ABSIZ(r1));				\
    (desc).fptr (r1, i1);						\
  } while (0)
#define INVOKE_RRSS(desc,r1,r2,i1,i2)					\
  do {									\
    if (pass & 1) _mpz_realloc (r1, ABSIZ(r1));				\
    if (pass & 2) _mpz_realloc (r2, ABSIZ(r2));				\
    (desc).fptr (r1, r2, i1, i2);					\
  } while (0)
#define INVOKE_RSS(desc,r1,i1,i2)					\
  do {									\
    if (pass & 1) _mpz_realloc (r1, ABSIZ(r1));				\
    (desc).fptr (r1, i1, i2);						\
  } while (0)

int
main (int argc, char **argv)
{
  int i;
  int pass, reps = 400;
  mpz_t in1, in2, in3;
  unsigned long int in2i;
  mp_size_t size;
  mpz_t res1, res2, res3;
  mpz_t ref1, ref2, ref3;
  mpz_t t;
  unsigned long int r1, r2;
  gmp_randstate_ptr rands;
  mpz_t bs;
  unsigned long bsi, size_range;

  tests_start ();
  TESTS_REPS (reps, argv, argc);

  rands = RANDS;

  mpz_init (bs);

  mpz_init (in1);
  mpz_init (in2);
  mpz_init (in3);
  mpz_init (ref1);
  mpz_init (ref2);
  mpz_init (ref3);
  mpz_init (res1);
  mpz_init (res2);
  mpz_init (res3);
  mpz_init (t);

  for (pass = 1; pass <= reps; pass++)
    {
      if (isatty (fileno (stdout)))
	{
	  printf ("\r%d/%d passes", pass, reps);
	  fflush (stdout);
	}

      mpz_urandomb (bs, rands, 32);
      size_range = mpz_get_ui (bs) % 21 + 2;

      if ((pass & 1) == 0)
	{
	  /* Make all input operands have quite different sizes */
	  mpz_urandomb (bs, rands, 32);
	  size = mpz_get_ui (bs) % size_range;
	  mpz_rrandomb (in1, rands, size);

	  mpz_urandomb (bs, rands, 32);
	  size = mpz_get_ui (bs) % size_range;
	  mpz_rrandomb (in2, rands, size);

	  mpz_urandomb (bs, rands, 32);
	  size = mpz_get_ui (bs) % size_range;
	  mpz_rrandomb (in3, rands, size);
	}
      else
	{
	  /* Make all input operands have about the same size */
	  mpz_urandomb (bs, rands, size_range);
	  size = mpz_get_ui (bs);
	  mpz_rrandomb (in1, rands, size);

	  mpz_urandomb (bs, rands, size_range);
	  size = mpz_get_ui (bs);
	  mpz_rrandomb (in2, rands, size);

	  mpz_urandomb (bs, rands, size_range);
	  size = mpz_get_ui (bs);
	  mpz_rrandomb (in3, rands, size);
	}

      mpz_urandomb (bs, rands, 3);
      bsi = mpz_get_ui (bs);
      if ((bsi & 1) != 0)
	mpz_neg (in1, in1);
      if ((bsi & 2) != 0)
	mpz_neg (in2, in2);
      if ((bsi & 4) != 0)
	mpz_neg (in3, in3);

      for (i = 0; i < numberof (dss); i++)
	{
	  if (dss[i].isdivision && mpz_sgn (in2) == 0)
	    continue;
	  if (dss[i].isslow && size_range > 19)
	    continue;

	  (dss[i].fptr) (ref1, in1, in2);
	  MPZ_CHECK_FORMAT (ref1);

	  mpz_set (res1, in1);
	  INVOKE_RSS (dss[i], res1, res1, in2);
	  MPZ_CHECK_FORMAT (res1);
	  if (mpz_cmp (ref1, res1) != 0)
	    FAIL (dss, i, in1, in2, NULL);

	  mpz_set (res1, in2);
	  INVOKE_RSS (dss[i], res1, in1, res1);
	  MPZ_CHECK_FORMAT (res1);
	  if (mpz_cmp (ref1, res1) != 0)
	    FAIL (dss, i, in1, in2, NULL);
	}

      for (i = 0; i < numberof (ddss_div); i++)
	{
	  if (mpz_sgn (in2) == 0)
	    continue;

	  (ddss_div[i].fptr) (ref1, ref2, in1, in2);
	  MPZ_CHECK_FORMAT (ref1);
	  MPZ_CHECK_FORMAT (ref2);

	  mpz_set (res1, in1);
	  INVOKE_RRSS (ddss_div[i], res1, res2, res1, in2);
	  MPZ_CHECK_FORMAT (res1);
	  MPZ_CHECK_FORMAT (res2);
	  if (mpz_cmp (ref1, res1) != 0 || mpz_cmp (ref2, res2) != 0)
	    FAIL (ddss_div, i, in1, in2, NULL);

	  mpz_set (res2, in1);
	  INVOKE_RRSS (ddss_div[i], res1, res2, res2, in2);
	  MPZ_CHECK_FORMAT (res1);
	  MPZ_CHECK_FORMAT (res2);
	  if (mpz_cmp (ref1, res1) != 0 || mpz_cmp (ref2, res2) != 0)
	    FAIL (ddss_div, i, in1, in2, NULL);

	  mpz_set (res1, in2);
	  INVOKE_RRSS (ddss_div[i], res1, res2, in1, res1);
	  MPZ_CHECK_FORMAT (res1);
	  MPZ_CHECK_FORMAT (res2);
	  if (mpz_cmp (ref1, res1) != 0 || mpz_cmp (ref2, res2) != 0)
	    FAIL (ddss_div, i, in1, in2, NULL);

	  mpz_set (res2, in2);
	  INVOKE_RRSS (ddss_div[i], res1, res2, in1, res2);
	  MPZ_CHECK_FORMAT (res1);
	  MPZ_CHECK_FORMAT (res2);
	  if (mpz_cmp (ref1, res1) != 0 || mpz_cmp (ref2, res2) != 0)
	    FAIL (ddss_div, i, in1, in2, NULL);
	}

      for (i = 0; i < numberof (ds); i++)
	{
	  if (ds[i].nonneg && mpz_sgn (in1) < 0)
	    continue;

	  (ds[i].fptr) (ref1, in1);
	  MPZ_CHECK_FORMAT (ref1);

	  mpz_set (res1, in1);
	  INVOKE_RS (ds[i], res1, res1);
	  MPZ_CHECK_FORMAT (res1);
	  if (mpz_cmp (ref1, res1) != 0)
	    FAIL (ds, i, in1, in2, NULL);
	}

      in2i = mpz_get_ui (in2);

      for (i = 0; i < numberof (dsi); i++)
	{
	  if (dsi[i].mod != 0)
	    in2i = mpz_get_ui (in2) % dsi[i].mod;

	  (dsi[i].fptr) (ref1, in1, in2i);
	  MPZ_CHECK_FORMAT (ref1);

	  mpz_set (res1, in1);
	  INVOKE_RRS (dsi[i], res1, res1, in2i);
	  MPZ_CHECK_FORMAT (res1);
	  if (mpz_cmp (ref1, res1) != 0)
	    FAIL (dsi, i, in1, in2, NULL);
	}

      if (in2i != 0)	  /* Don't divide by 0.  */
	{
	  for (i = 0; i < numberof (dsi_div); i++)
	    {
	      r1 = (dsi_div[i].fptr) (ref1, in1, in2i);
	      MPZ_CHECK_FORMAT (ref1);

	      mpz_set (res1, in1);
	      r2 = (dsi_div[i].fptr) (res1, res1, in2i);
	      MPZ_CHECK_FORMAT (res1);
	      if (mpz_cmp (ref1, res1) != 0 || r1 != r2)
		FAIL (dsi_div, i, in1, in2, NULL);
	    }

	  for (i = 0; i < numberof (ddsi_div); i++)
	    {
	      r1 = (ddsi_div[i].fptr) (ref1, ref2, in1, in2i);
	      MPZ_CHECK_FORMAT (ref1);

	      mpz_set (res1, in1);
	      r2 = (ddsi_div[i].fptr) (res1, res2, res1, in2i);
	      MPZ_CHECK_FORMAT (res1);
	      if (mpz_cmp (ref1, res1) != 0 || mpz_cmp (ref2, res2) != 0 || r1 != r2)
		FAIL (ddsi_div, i, in1, in2, NULL);

	      mpz_set (res2, in1);
	      (ddsi_div[i].fptr) (res1, res2, res2, in2i);
	      MPZ_CHECK_FORMAT (res1);
	      if (mpz_cmp (ref1, res1) != 0 || mpz_cmp (ref2, res2) != 0 || r1 != r2)
		FAIL (ddsi_div, i, in1, in2, NULL);
	    }
	}

      if (mpz_sgn (in1) >= 0)
	{
	  mpz_sqrtrem (ref1, ref2, in1);
	  MPZ_CHECK_FORMAT (ref1);
	  MPZ_CHECK_FORMAT (ref2);

	  mpz_set (res1, in1);
	  mpz_sqrtrem (res1, res2, res1);
	  MPZ_CHECK_FORMAT (res1);
	  MPZ_CHECK_FORMAT (res2);
	  if (mpz_cmp (ref1, res1) != 0 || mpz_cmp (ref2, res2) != 0)
	    FAIL2 (mpz_sqrtrem, in1, NULL, NULL);

	  mpz_set (res2, in1);
	  mpz_sqrtrem (res1, res2, res2);
	  MPZ_CHECK_FORMAT (res1);
	  MPZ_CHECK_FORMAT (res2);
	  if (mpz_cmp (ref1, res1) != 0 || mpz_cmp (ref2, res2) != 0)
	    FAIL2 (mpz_sqrtrem, in1, NULL, NULL);

	  mpz_set (res1, in1);
	  mpz_sqrtrem (res1, res1, res1);
	  MPZ_CHECK_FORMAT (res1);
	  if (mpz_cmp (ref2, res1) != 0)
	    FAIL2 (mpz_sqrtrem, in1, NULL, NULL);
	}

      if (mpz_sgn (in1) >= 0)
	{
	  mpz_root (ref1, in1, in2i % 0x100 + 1);
	  MPZ_CHECK_FORMAT (ref1);

	  mpz_set (res1, in1);
	  mpz_root (res1, res1, in2i % 0x100 + 1);
	  MPZ_CHECK_FORMAT (res1);
	  if (mpz_cmp (ref1, res1) != 0)
	    FAIL2 (mpz_root, in1, in2, NULL);
	}

      if (mpz_sgn (in1) >= 0)
	{
	  mpz_rootrem (ref1, ref2, in1, in2i % 0x100 + 1);
	  MPZ_CHECK_FORMAT (ref1);
	  MPZ_CHECK_FORMAT (ref2);

	  mpz_set (res1, in1);
	  mpz_rootrem (res1, res2, res1, in2i % 0x100 + 1);
	  MPZ_CHECK_FORMAT (res1);
	  MPZ_CHECK_FORMAT (res2);
	  if (mpz_cmp (ref1, res1) != 0 || mpz_cmp (ref2, res2) != 0)
	    FAIL2 (mpz_rootrem, in1, in2, NULL);

	  mpz_set (res2, in1);
	  mpz_rootrem (res1, res2, res2, in2i % 0x100 + 1);
	  MPZ_CHECK_FORMAT (res1);
	  MPZ_CHECK_FORMAT (res2);
	  if (mpz_cmp (ref1, res1) != 0 || mpz_cmp (ref2, res2) != 0)
	    FAIL2 (mpz_rootrem, in1, in2, NULL);
	}

      if (size_range < 18)	/* run fewer tests since gcdext is slow */
	{
	  mpz_gcdext (ref1, ref2, ref3, in1, in2);
	  MPZ_CHECK_FORMAT (ref1);
	  MPZ_CHECK_FORMAT (ref2);
	  MPZ_CHECK_FORMAT (ref3);

	  mpz_set (res1, in1);
	  mpz_gcdext (res1, res2, res3, res1, in2);
	  MPZ_CHECK_FORMAT (res1);
	  MPZ_CHECK_FORMAT (res2);
	  MPZ_CHECK_FORMAT (res3);
	  if (mpz_cmp (ref1, res1) != 0 || mpz_cmp (ref2, res2) != 0
	      || mpz_cmp (ref3, res3) != 0)
	    FAIL2 (mpz_gcdext, in1, in2, NULL);

	  mpz_set (res2, in1);
	  mpz_gcdext (res1, res2, res3, res2, in2);
	  MPZ_CHECK_FORMAT (res1);
	  MPZ_CHECK_FORMAT (res2);
	  MPZ_CHECK_FORMAT (res3);
	  if (mpz_cmp (ref1, res1) != 0 || mpz_cmp (ref2, res2) != 0
	      || mpz_cmp (ref3, res3) != 0)
	    FAIL2 (mpz_gcdext, in1, in2, NULL);

	  mpz_set (res3, in1);
	  mpz_gcdext (res1, res2, res3, res3, in2);
	  MPZ_CHECK_FORMAT (res1);
	  MPZ_CHECK_FORMAT (res2);
	  MPZ_CHECK_FORMAT (res3);
	  if (mpz_cmp (ref1, res1) != 0 || mpz_cmp (ref2, res2) != 0
	      || mpz_cmp (ref3, res3) != 0)
	    FAIL2 (mpz_gcdext, in1, in2, NULL);

	  mpz_set (res1, in2);
	  mpz_gcdext (res1, res2, res3, in1, res1);
	  MPZ_CHECK_FORMAT (res1);
	  MPZ_CHECK_FORMAT (res2);
	  MPZ_CHECK_FORMAT (res3);
	  if (mpz_cmp (ref1, res1) != 0 || mpz_cmp (ref2, res2) != 0
	      || mpz_cmp (ref3, res3) != 0)
	    FAIL2 (mpz_gcdext, in1, in2, NULL);

	  mpz_set (res2, in2);
	  mpz_gcdext (res1, res2, res3, in1, res2);
	  MPZ_CHECK_FORMAT (res1);
	  MPZ_CHECK_FORMAT (res2);
	  MPZ_CHECK_FORMAT (res3);
	  if (mpz_cmp (ref1, res1) != 0 || mpz_cmp (ref2, res2) != 0
	      || mpz_cmp (ref3, res3) != 0)
	    FAIL2 (mpz_gcdext, in1, in2, NULL);

	  mpz_set (res3, in2);
	  mpz_gcdext (res1, res2, res3, in1, res3);
	  MPZ_CHECK_FORMAT (res1);
	  MPZ_CHECK_FORMAT (res2);
	  MPZ_CHECK_FORMAT (res3);
	  if (mpz_cmp (ref1, res1) != 0 || mpz_cmp (ref2, res2) != 0
	      || mpz_cmp (ref3, res3) != 0)
	    FAIL2 (mpz_gcdext, in1, in2, NULL);

	  mpz_set (res1, in1);
	  mpz_gcdext (res1, res2, NULL, res1, in2);
	  MPZ_CHECK_FORMAT (res1);
	  MPZ_CHECK_FORMAT (res2);
	  if (mpz_cmp (ref1, res1) != 0 || mpz_cmp (ref2, res2) != 0
	      || mpz_cmp (ref3, res3) != 0)
	    FAIL2 (mpz_gcdext, in1, in2, NULL);

	  mpz_set (res2, in1);
	  mpz_gcdext (res1, res2, NULL, res2, in2);
	  MPZ_CHECK_FORMAT (res1);
	  MPZ_CHECK_FORMAT (res2);
	  if (mpz_cmp (ref1, res1) != 0 || mpz_cmp (ref2, res2) != 0
	      || mpz_cmp (ref3, res3) != 0)
	    FAIL2 (mpz_gcdext, in1, in2, NULL);

	  mpz_set (res1, in2);
	  mpz_gcdext (res1, res2, NULL, in1, res1);
	  MPZ_CHECK_FORMAT (res1);
	  MPZ_CHECK_FORMAT (res2);
	  if (mpz_cmp (ref1, res1) != 0 || mpz_cmp (ref2, res2) != 0
	      || mpz_cmp (ref3, res3) != 0)
	    FAIL2 (mpz_gcdext, in1, in2, NULL);

	  mpz_set (res2, in2);
	  mpz_gcdext (res1, res2, NULL, in1, res2);
	  MPZ_CHECK_FORMAT (res1);
	  MPZ_CHECK_FORMAT (res2);
	  if (mpz_cmp (ref1, res1) != 0 || mpz_cmp (ref2, res2) != 0
	      || mpz_cmp (ref3, res3) != 0)
	    FAIL2 (mpz_gcdext, in1, in2, NULL);
	}

      /* Don't run mpz_powm for huge exponents or when undefined.  */
      if (size_range < 17 && mpz_sizeinbase (in2, 2) < 250 && mpz_sgn (in3) != 0
	  && (mpz_sgn (in2) >= 0 || mpz_invert (t, in1, in3)))
	{
	  mpz_powm (ref1, in1, in2, in3);
	  MPZ_CHECK_FORMAT (ref1);

	  mpz_set (res1, in1);
	  mpz_powm (res1, res1, in2, in3);
	  MPZ_CHECK_FORMAT (res1);
	  if (mpz_cmp (ref1, res1) != 0)
	    FAIL2 (mpz_powm, in1, in2, in3);

	  mpz_set (res1, in2);
	  mpz_powm (res1, in1, res1, in3);
	  MPZ_CHECK_FORMAT (res1);
	  if (mpz_cmp (ref1, res1) != 0)
	    FAIL2 (mpz_powm, in1, in2, in3);

	  mpz_set (res1, in3);
	  mpz_powm (res1, in1, in2, res1);
	  MPZ_CHECK_FORMAT (res1);
	  if (mpz_cmp (ref1, res1) != 0)
	    FAIL2 (mpz_powm, in1, in2, in3);
	}

      /* Don't run mpz_powm_ui when undefined.  */
      if (size_range < 17 && mpz_sgn (in3) != 0)
	{
	  mpz_powm_ui (ref1, in1, in2i, in3);
	  MPZ_CHECK_FORMAT (ref1);

	  mpz_set (res1, in1);
	  mpz_powm_ui (res1, res1, in2i, in3);
	  MPZ_CHECK_FORMAT (res1);
	  if (mpz_cmp (ref1, res1) != 0)
	    FAIL2 (mpz_powm_ui, in1, in2, in3);

	  mpz_set (res1, in3);
	  mpz_powm_ui (res1, in1, in2i, res1);
	  MPZ_CHECK_FORMAT (res1);
	  if (mpz_cmp (ref1, res1) != 0)
	    FAIL2 (mpz_powm_ui, in1, in2, in3);
	}

      {
	r1 = mpz_gcd_ui (ref1, in1, in2i);
	MPZ_CHECK_FORMAT (ref1);

	mpz_set (res1, in1);
	r2 = mpz_gcd_ui (res1, res1, in2i);
	MPZ_CHECK_FORMAT (res1);
	if (mpz_cmp (ref1, res1) != 0)
	  FAIL2 (mpz_gcd_ui, in1, in2, NULL);
      }

      if (mpz_sgn (in2) != 0)
	{
	  /* Test mpz_remove */
	  mp_bitcnt_t refretval, retval;
	  refretval = mpz_remove (ref1, in1, in2);
	  MPZ_CHECK_FORMAT (ref1);

	  mpz_set (res1, in1);
	  retval = mpz_remove (res1, res1, in2);
	  MPZ_CHECK_FORMAT (res1);
	  if (mpz_cmp (ref1, res1) != 0 || refretval != retval)
	    FAIL2 (mpz_remove, in1, in2, NULL);

	  mpz_set (res1, in2);
	  retval = mpz_remove (res1, in1, res1);
	  MPZ_CHECK_FORMAT (res1);
	  if (mpz_cmp (ref1, res1) != 0 || refretval != retval)
	    FAIL2 (mpz_remove, in1, in2, NULL);
	}

      if (mpz_sgn (in2) != 0)
	{
	  /* Test mpz_divexact */
	  mpz_mul (t, in1, in2);
	  mpz_divexact (ref1, t, in2);
	  MPZ_CHECK_FORMAT (ref1);

	  mpz_set (res1, t);
	  mpz_divexact (res1, res1, in2);
	  MPZ_CHECK_FORMAT (res1);
	  if (mpz_cmp (ref1, res1) != 0)
	    FAIL2 (mpz_divexact, t, in2, NULL);

	  mpz_set (res1, in2);
	  mpz_divexact (res1, t, res1);
	  MPZ_CHECK_FORMAT (res1);
	  if (mpz_cmp (ref1, res1) != 0)
	    FAIL2 (mpz_divexact, t, in2, NULL);
	}

      if (mpz_sgn (in2) > 0)
	{
	  /* Test mpz_divexact_gcd, same as mpz_divexact */
	  mpz_mul (t, in1, in2);
	  mpz_divexact_gcd (ref1, t, in2);
	  MPZ_CHECK_FORMAT (ref1);

	  mpz_set (res1, t);
	  mpz_divexact_gcd (res1, res1, in2);
	  MPZ_CHECK_FORMAT (res1);
	  if (mpz_cmp (ref1, res1) != 0)
	    FAIL2 (mpz_divexact_gcd, t, in2, NULL);

	  mpz_set (res1, in2);
	  mpz_divexact_gcd (res1, t, res1);
	  MPZ_CHECK_FORMAT (res1);
	  if (mpz_cmp (ref1, res1) != 0)
	    FAIL2 (mpz_divexact_gcd, t, in2, NULL);
	}
    }

  if (isatty (fileno (stdout)))
    printf ("\r%20s", "");

  mpz_clear (bs);
  mpz_clear (in1);
  mpz_clear (in2);
  mpz_clear (in3);
  mpz_clear (ref1);
  mpz_clear (ref2);
  mpz_clear (ref3);
  mpz_clear (res1);
  mpz_clear (res2);
  mpz_clear (res3);
  mpz_clear (t);

  if (isatty (fileno (stdout)))
    printf ("\r");

  tests_end ();
  exit (0);
}

void
dump (const char *name, mpz_t in1, mpz_t in2, mpz_t in3)
{
  printf ("failure in %s (", name);
  0 && mpz_out_str (stdout, -16, in1);
  if (in2 != NULL)
    {
      printf (" ");
      0 && mpz_out_str (stdout, -16, in2);
    }
  if (in3 != NULL)
    {
      printf (" ");
      0 && mpz_out_str (stdout, -16, in3);
    }
  printf (")\n");
}

#endif /* ! DLL_EXPORT */
