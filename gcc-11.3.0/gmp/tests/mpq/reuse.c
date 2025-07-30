/* Test that routines allow reusing a source variable as destination.

Copyright 1996, 2000-2002, 2012, 2015 Free Software Foundation, Inc.

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

/* FIXME: When linking to a DLL libgmp, mpq_add etc can't be used as
   initializers for global variables because they're effectively global
   variables (function pointers) themselves.  Perhaps calling a test
   function successively with mpq_add etc would be better.  */

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

void dump_abort (const char *, mpq_t, mpq_t);

typedef void (*dss_func) (mpq_ptr, mpq_srcptr, mpq_srcptr);

dss_func dss_funcs[] =
{
  mpq_div, mpq_add, mpq_mul, mpq_sub,
};

const char *dss_func_names[] =
{
  "mpq_div", "mpq_add", "mpq_mul", "mpq_sub",
};

typedef void (*ds_func) (mpq_ptr, mpq_srcptr);

ds_func ds_funcs[] =
{
  mpq_abs, mpq_neg,
};

const char *ds_func_names[] =
{
  "mpq_abs", "mpq_neg",
};

typedef void (*dsi_func) (mpq_ptr, mpq_srcptr, unsigned long int);

dsi_func dsi_funcs[] =
{
  mpq_mul_2exp, mpq_div_2exp
};

const char *dsi_func_names[] =
{
  "mpq_mul_2exp", "mpq_div_2exp"
};

int
main (int argc, char **argv)
{
  int i;
  int pass, reps = 100;
  mpq_t in1, in2, out1;
  unsigned long int randbits, in2i;
  mpq_t res1, res2;
  gmp_randstate_ptr  rands;

  tests_start ();

  if (argc > 1)
    reps = strtol (argv[1], 0, 0);

  rands = RANDS;

  mpq_init (in1);
  mpq_init (in2);
  mpq_init (out1);
  mpq_init (res1);
  mpq_init (res2);

  for (pass = 1; pass <= reps; pass++)
    {
      randbits = urandom ();

      if (randbits & 1)
	{
	  mpq_clear (in1);
	  mpq_init (in1);
	}
      randbits >>= 1;
      mpz_errandomb (mpq_numref(in1), rands, 512L);
      mpz_errandomb_nonzero (mpq_denref(in1), rands, 512L);
      if (randbits & 1)
	mpz_neg (mpq_numref(in1),mpq_numref(in1));
      randbits >>= 1;
      mpq_canonicalize (in1);

      if (randbits & 1)
	{
	  mpq_clear (in2);
	  mpq_init (in2);
	}
      randbits >>= 1;
      mpz_errandomb (mpq_numref(in2), rands, 512L);
      mpz_errandomb_nonzero (mpq_denref(in2), rands, 512L);
      if (randbits & 1)
	mpz_neg (mpq_numref(in2),mpq_numref(in2));
      randbits >>= 1;
      mpq_canonicalize (in2);

      for (i = 0; i < sizeof (dss_funcs) / sizeof (dss_func); i++)
	{
	  /* Don't divide by 0.  */
	  if (i == 0 && mpq_cmp_ui (in2, 0, 1) == 0)
	    continue;

	  if (randbits & 1)
	    {
	      mpq_clear (res1);
	      mpq_init (res1);
	    }
	  randbits >>= 1;

	  (dss_funcs[i]) (res1, in1, in2);
	  MPQ_CHECK_FORMAT(res1);

	  mpq_set (out1, in1);
	  (dss_funcs[i]) (out1, out1, in2);
	  MPQ_CHECK_FORMAT(out1);

	  if (mpq_cmp (res1, out1) != 0)
	    dump_abort (dss_func_names[i], res1, out1);

	  mpq_set (out1, in2);
	  (dss_funcs[i]) (out1, in1, out1);
	  MPQ_CHECK_FORMAT(out1);

	  if (mpq_cmp (res1, out1) != 0)
	    dump_abort (dss_func_names[i], res1, out1);

	  mpq_set (out1, in2);
	  (dss_funcs[i]) (res1, out1, in2);
	  MPQ_CHECK_FORMAT(res1);

	  (dss_funcs[i]) (res2, in2, in2);
	  MPQ_CHECK_FORMAT(res2);

	  (dss_funcs[i]) (out1, out1, out1);
	  MPQ_CHECK_FORMAT(out1);

	  if (mpq_cmp (res1, res2) != 0)
	    dump_abort (dss_func_names[i], res1, res2);
	  if (mpq_cmp (res1, out1) != 0)
	    dump_abort (dss_func_names[i], res1, out1);
	}

      for (i = 0; i < sizeof (ds_funcs) / sizeof (ds_func); i++)
	{
	  if (randbits & 1)
	    {
	      mpq_clear (res1);
	      mpq_init (res1);
	    }
	  randbits >>= 1;
	  (ds_funcs[i]) (res1, in1);
	  MPQ_CHECK_FORMAT(res1);

	  mpq_set (out1, in1);
	  (ds_funcs[i]) (out1, out1);
	  MPQ_CHECK_FORMAT(out1);

	  if (mpq_cmp (res1, out1) != 0)
	    dump_abort (ds_func_names[i], res1, out1);
	}

      in2i = urandom () % 65536;
      for (i = 0; i < sizeof (dsi_funcs) / sizeof (dsi_func); i++)
	{
	  if (randbits & 1)
	    {
	      mpq_clear (res1);
	      mpq_init (res1);
	    }
	  randbits >>= 1;

	  (dsi_funcs[i]) (res1, in1, in2i);
	  MPQ_CHECK_FORMAT(res1);

	  mpq_set (out1, in1);
	  (dsi_funcs[i]) (out1, out1, in2i);
	  MPQ_CHECK_FORMAT(out1);

	  if (mpq_cmp (res1, out1) != 0)
	    dump_abort (dsi_func_names[i], res1, out1);
	}

    }

  mpq_clear (in1);
  mpq_clear (in2);
  mpq_clear (out1);
  mpq_clear (res1);
  mpq_clear (res2);

  tests_end ();
  exit (0);
}

void
dump_abort (const char *name, mpq_t res1, mpq_t res2)
{
  printf ("failure in %s:\n", name);
  mpq_trace ("  res1  ", res1);
  mpq_trace ("  res2  ", res2);
  abort ();
}

#endif /* ! DLL_EXPORT */
