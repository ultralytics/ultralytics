/* Test that routines allow reusing a source variable as destination.

Copyright 1996, 2000-2002, 2012 Free Software Foundation, Inc.

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
#include "tests.h"

#if __GMP_LIBGMP_DLL

/* FIXME: When linking to a DLL libgmp, mpf_add etc can't be used as
   initializers for global variables because they're effectively global
   variables (function pointers) themselves.  Perhaps calling a test
   function successively with mpf_add etc would be better.  */

int
main (void)
{
  printf ("Test suppressed for windows DLL\n");
  exit (0);
}


#else /* ! DLL_EXPORT */

#ifndef SIZE
#define SIZE 16
#endif

#ifndef EXPO
#define EXPO 32
#endif

void dump_abort (const char *, mpf_t, mpf_t);

typedef void (*dss_func) (mpf_ptr, mpf_srcptr, mpf_srcptr);

dss_func dss_funcs[] =
{
  mpf_div, mpf_add, mpf_mul, mpf_sub,
};

const char *dss_func_names[] =
{
  "mpf_div", "mpf_add", "mpf_mul", "mpf_sub",
};

typedef void (*dsi_func) (mpf_ptr, mpf_srcptr, unsigned long int);

dsi_func dsi_funcs[] =
{
  mpf_div_ui, mpf_add_ui, mpf_mul_ui, mpf_sub_ui,
  mpf_mul_2exp, mpf_div_2exp, mpf_pow_ui
};

const char *dsi_func_names[] =
{
  "mpf_div_ui", "mpf_add_ui", "mpf_mul_ui", "mpf_sub_ui",
  "mpf_mul_2exp", "mpf_div_2exp", "mpf_pow_ui"
};

typedef void (*dis_func) (mpf_ptr, unsigned long int, mpf_srcptr);

dis_func dis_funcs[] =
{
  mpf_ui_div, mpf_ui_sub,
};

const char *dis_func_names[] =
{
  "mpf_ui_div", "mpf_ui_sub",
};

int
main (int argc, char **argv)
{
  int i;
  int pass, reps = 10000;
  mpf_t in1, in2, out1;
  unsigned long int in1i, in2i;
  mpf_t res1, res2, res3;
  mp_size_t bprec = 100;

  tests_start ();

  if (argc > 1)
    {
      reps = strtol (argv[1], 0, 0);
      if (argc > 2)
	bprec = strtol (argv[2], 0, 0);
    }

  mpf_set_default_prec (bprec);

  mpf_init (in1);
  mpf_init (in2);
  mpf_init (out1);
  mpf_init (res1);
  mpf_init (res2);
  mpf_init (res3);

  for (pass = 1; pass <= reps; pass++)
    {
      mpf_random2 (in1, urandom () % SIZE - SIZE/2, urandom () % EXPO);
      mpf_random2 (in2, urandom () % SIZE - SIZE/2, urandom () % EXPO);

      for (i = 0; i < sizeof (dss_funcs) / sizeof (dss_func); i++)
	{
	  /* Don't divide by 0.  */
	  if (i == 0 && mpf_cmp_ui (in2, 0) == 0)
	    continue;

	  (dss_funcs[i]) (res1, in1, in2);

	  mpf_set (out1, in1);
	  (dss_funcs[i]) (out1, out1, in2);
	  mpf_set (res2, out1);

	  mpf_set (out1, in2);
	  (dss_funcs[i]) (out1, in1, out1);
	  mpf_set (res3, out1);

	  if (mpf_cmp (res1, res2) != 0)
	    dump_abort (dss_func_names[i], res1, res2);
	  if (mpf_cmp (res1, res3) != 0)
	    dump_abort (dss_func_names[i], res1, res3);
	}

      in2i = urandom ();
      for (i = 0; i < sizeof (dsi_funcs) / sizeof (dsi_func); i++)
	{
	  unsigned long this_in2i = in2i;

	  /* Don't divide by 0.  */
	  if (dsi_funcs[i] == mpf_div_ui && this_in2i == 0)
	    continue;

	  /* Avoid overflow/underflow in the exponent.  */
	  if (dsi_funcs[i] == mpf_mul_2exp || dsi_funcs[i] == mpf_div_2exp)
	    this_in2i %= 0x100000;
	  else if (dsi_funcs[i] == mpf_pow_ui)
	    this_in2i %= 0x1000;

	  (dsi_funcs[i]) (res1, in1, this_in2i);

	  mpf_set (out1, in1);
	  (dsi_funcs[i]) (out1, out1, this_in2i);
	  mpf_set (res2, out1);

	  if (mpf_cmp (res1, res2) != 0)
	    dump_abort (dsi_func_names[i], res1, res2);
	}

      in1i = urandom ();
      for (i = 0; i < sizeof (dis_funcs) / sizeof (dis_func); i++)
	{
	  /* Don't divide by 0.  */
	  if (dis_funcs[i] == mpf_ui_div
	      && mpf_cmp_ui (in2, 0) == 0)
	    continue;

	  (dis_funcs[i]) (res1, in1i, in2);

	  mpf_set (out1, in2);
	  (dis_funcs[i]) (out1, in1i, out1);
	  mpf_set (res2, out1);

	  if (mpf_cmp (res1, res2) != 0)
	    dump_abort (dis_func_names[i], res1, res2);
	}

    }

  mpf_clear (in1);
  mpf_clear (in2);
  mpf_clear (out1);
  mpf_clear (res1);
  mpf_clear (res2);
  mpf_clear (res3);

  tests_end ();
  exit (0);
}

void
dump_abort (const char *name, mpf_t res1, mpf_t res2)
{
  printf ("failure in %s:\n", name);
  mpf_dump (res1);
  mpf_dump (res2);
  abort ();
}

#if 0
void mpf_abs		(mpf_ptr, mpf_srcptr);
void mpf_sqrt		(mpf_ptr, mpf_srcptr);
void mpf_neg		(mpf_ptr, mpf_srcptr);
#endif

#endif /* ! DLL_EXPORT */
