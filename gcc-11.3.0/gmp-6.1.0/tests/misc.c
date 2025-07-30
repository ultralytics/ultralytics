/* Miscellaneous test program support routines.

Copyright 2000-2003, 2005, 2013 Free Software Foundation, Inc.

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

#include "config.h"

#include <ctype.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>     /* for getenv */
#include <string.h>

#if HAVE_FLOAT_H
#include <float.h>      /* for DBL_MANT_DIG */
#endif

#if TIME_WITH_SYS_TIME
# include <sys/time.h>  /* for struct timeval */
# include <time.h>
#else
# if HAVE_SYS_TIME_H
#  include <sys/time.h>
# else
#  include <time.h>
# endif
#endif

#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"


/* The various tests setups and final checks, collected up together. */
void
tests_start (void)
{
  /* don't buffer, so output is not lost if a test causes a segv etc */
  setbuf (stdout, NULL);
  setbuf (stderr, NULL);

  tests_memory_start ();
  tests_rand_start ();
}
void
tests_end (void)
{
  tests_rand_end ();
  tests_memory_end ();
}


void
tests_rand_start (void)
{
  gmp_randstate_ptr  rands;
  char           *perform_seed;
  unsigned long  seed;

  if (__gmp_rands_initialized)
    {
      printf ("Please let tests_start() initialize the global __gmp_rands.\n");
      printf ("ie. ensure that function is called before the first use of RANDS.\n");
      abort ();
    }

  gmp_randinit_default (__gmp_rands);
  __gmp_rands_initialized = 1;
  rands = __gmp_rands;

  perform_seed = getenv ("GMP_CHECK_RANDOMIZE");
  if (perform_seed != NULL)
    {
#ifdef HAVE_STRTOUL
      seed = strtoul (perform_seed, 0, 0);
#else
      /* This will not work right for seeds >= 2^31 on 64-bit machines.
	 Perhaps use atol unconditionally?  Is that ubiquitous?  */
      seed = atoi (perform_seed);
#endif
      if (! (seed == 0 || seed == 1))
        {
          printf ("Re-seeding with GMP_CHECK_RANDOMIZE=%lu\n", seed);
          gmp_randseed_ui (rands, seed);
        }
      else
        {
#if HAVE_GETTIMEOFDAY
          struct timeval  tv;
          gettimeofday (&tv, NULL);
          seed = tv.tv_sec ^ (tv.tv_usec << 12);
	  seed &= 0xffffffff;
#else
          time_t  tv;
          time (&tv);
          seed = tv;
#endif
          gmp_randseed_ui (rands, seed);
          printf ("Seed GMP_CHECK_RANDOMIZE=%lu (include this in bug reports)\n", seed);
        }
      fflush (stdout);
    }
}
void
tests_rand_end (void)
{
  RANDS_CLEAR ();
}


/* Only used if CPU calling conventions checking is available. */
mp_limb_t (*calling_conventions_function) (ANYARGS);


/* Return p advanced to the next multiple of "align" bytes.  "align" must be
   a power of 2.  Care is taken not to assume sizeof(int)==sizeof(pointer).
   Using "unsigned long" avoids a warning on hpux.  */
void *
align_pointer (void *p, size_t align)
{
  gmp_intptr_t d;
  d = ((gmp_intptr_t) p) & (align-1);
  d = (d != 0 ? align-d : 0);
  return (void *) (((char *) p) + d);
}


/* Note that memory allocated with this function can never be freed, because
   the start address of the block allocated is lost. */
void *
__gmp_allocate_func_aligned (size_t bytes, size_t align)
{
  return align_pointer ((*__gmp_allocate_func) (bytes + align-1), align);
}


void *
__gmp_allocate_or_reallocate (void *ptr, size_t oldsize, size_t newsize)
{
  if (ptr == NULL)
    return (*__gmp_allocate_func) (newsize);
  else
    return (*__gmp_reallocate_func) (ptr, oldsize, newsize);
}

char *
__gmp_allocate_strdup (const char *s)
{
  size_t  len;
  char    *t;
  len = strlen (s);
  t = (char *) (*__gmp_allocate_func) (len+1);
  memcpy (t, s, len+1);
  return t;
}


char *
strtoupper (char *s_orig)
{
  char  *s;
  for (s = s_orig; *s != '\0'; s++)
    if (isascii (*s))
      *s = toupper (*s);
  return s_orig;
}


void
mpz_set_n (mpz_ptr z, mp_srcptr p, mp_size_t size)
{
  ASSERT (size >= 0);
  MPN_NORMALIZE (p, size);
  MPZ_REALLOC (z, size);
  MPN_COPY (PTR(z), p, size);
  SIZ(z) = size;
}

void
mpz_init_set_n (mpz_ptr z, mp_srcptr p, mp_size_t size)
{
  ASSERT (size >= 0);

  MPN_NORMALIZE (p, size);
  ALLOC(z) = MAX (size, 1);
  PTR(z) = __GMP_ALLOCATE_FUNC_LIMBS (ALLOC(z));
  SIZ(z) = size;
  MPN_COPY (PTR(z), p, size);
}


/* Find least significant limb position where p1,size and p2,size differ.  */
mp_size_t
mpn_diff_lowest (mp_srcptr p1, mp_srcptr p2, mp_size_t size)
{
  mp_size_t  i;

  for (i = 0; i < size; i++)
    if (p1[i] != p2[i])
      return i;

  /* no differences */
  return -1;
}


/* Find most significant limb position where p1,size and p2,size differ.  */
mp_size_t
mpn_diff_highest (mp_srcptr p1, mp_srcptr p2, mp_size_t size)
{
  mp_size_t  i;

  for (i = size-1; i >= 0; i--)
    if (p1[i] != p2[i])
      return i;

  /* no differences */
  return -1;
}


/* Find least significant byte position where p1,size and p2,size differ.  */
mp_size_t
byte_diff_lowest (const void *p1, const void *p2, mp_size_t size)
{
  mp_size_t  i;

  for (i = 0; i < size; i++)
    if (((const char *) p1)[i] != ((const char *) p2)[i])
      return i;

  /* no differences */
  return -1;
}


/* Find most significant limb position where p1,size and p2,size differ.  */
mp_size_t
byte_diff_highest (const void *p1, const void *p2, mp_size_t size)
{
  mp_size_t  i;

  for (i = size-1; i >= 0; i--)
    if (((const char *) p1)[i] != ((const char *) p2)[i])
      return i;

  /* no differences */
  return -1;
}


void
mpz_set_str_or_abort (mpz_ptr z, const char *str, int base)
{
  if (mpz_set_str (z, str, base) != 0)
    {
      fprintf (stderr, "ERROR: mpz_set_str failed\n");
      fprintf (stderr, "   str  = \"%s\"\n", str);
      fprintf (stderr, "   base = %d\n", base);
      abort();
    }
}

void
mpq_set_str_or_abort (mpq_ptr q, const char *str, int base)
{
  if (mpq_set_str (q, str, base) != 0)
    {
      fprintf (stderr, "ERROR: mpq_set_str failed\n");
      fprintf (stderr, "   str  = \"%s\"\n", str);
      fprintf (stderr, "   base = %d\n", base);
      abort();
    }
}

void
mpf_set_str_or_abort (mpf_ptr f, const char *str, int base)
{
  if (mpf_set_str (f, str, base) != 0)
    {
      fprintf (stderr, "ERROR mpf_set_str failed\n");
      fprintf (stderr, "   str  = \"%s\"\n", str);
      fprintf (stderr, "   base = %d\n", base);
      abort();
    }
}


/* Whether the absolute value of z is a power of 2. */
int
mpz_pow2abs_p (mpz_srcptr z)
{
  mp_size_t  size, i;
  mp_srcptr  ptr;

  size = SIZ (z);
  if (size == 0)
    return 0;  /* zero is not a power of 2 */
  size = ABS (size);

  ptr = PTR (z);
  for (i = 0; i < size-1; i++)
    if (ptr[i] != 0)
      return 0;  /* non-zero low limb means not a power of 2 */

  return POW2_P (ptr[i]);  /* high limb power of 2 */
}


/* Exponentially distributed between 0 and 2^nbits-1, meaning the number of
   bits in the result is uniformly distributed between 0 and nbits-1.

   FIXME: This is not a proper exponential distribution, since the
   probability function will have a stepped shape due to using a uniform
   distribution after choosing how many bits.  */

void
mpz_erandomb (mpz_ptr rop, gmp_randstate_t rstate, unsigned long nbits)
{
  mpz_urandomb (rop, rstate, gmp_urandomm_ui (rstate, nbits));
}

void
mpz_erandomb_nonzero (mpz_ptr rop, gmp_randstate_t rstate, unsigned long nbits)
{
  mpz_erandomb (rop, rstate, nbits);
  if (mpz_sgn (rop) == 0)
    mpz_set_ui (rop, 1L);
}

void
mpz_errandomb (mpz_ptr rop, gmp_randstate_t rstate, unsigned long nbits)
{
  mpz_rrandomb (rop, rstate, gmp_urandomm_ui (rstate, nbits));
}

void
mpz_errandomb_nonzero (mpz_ptr rop, gmp_randstate_t rstate, unsigned long nbits)
{
  mpz_errandomb (rop, rstate, nbits);
  if (mpz_sgn (rop) == 0)
    mpz_set_ui (rop, 1L);
}

void
mpz_negrandom (mpz_ptr rop, gmp_randstate_t rstate)
{
  mp_limb_t  n;
  _gmp_rand (&n, rstate, 1);
  if (n != 0)
    mpz_neg (rop, rop);
}

mp_limb_t
urandom (void)
{
#if GMP_NAIL_BITS == 0
  mp_limb_t  n;
  _gmp_rand (&n, RANDS, GMP_LIMB_BITS);
  return n;
#else
  mp_limb_t n[2];
  _gmp_rand (n, RANDS, GMP_LIMB_BITS);
  return n[0] + (n[1] << GMP_NUMB_BITS);
#endif
}


/* Call (*func)() with various random number generators. */
void
call_rand_algs (void (*func) (const char *, gmp_randstate_ptr))
{
  gmp_randstate_t  rstate;
  mpz_t            a;

  mpz_init (a);

  gmp_randinit_default (rstate);
  (*func) ("gmp_randinit_default", rstate);
  gmp_randclear (rstate);

  gmp_randinit_mt (rstate);
  (*func) ("gmp_randinit_mt", rstate);
  gmp_randclear (rstate);

  gmp_randinit_lc_2exp_size (rstate, 8L);
  (*func) ("gmp_randinit_lc_2exp_size 8", rstate);
  gmp_randclear (rstate);

  gmp_randinit_lc_2exp_size (rstate, 16L);
  (*func) ("gmp_randinit_lc_2exp_size 16", rstate);
  gmp_randclear (rstate);

  gmp_randinit_lc_2exp_size (rstate, 128L);
  (*func) ("gmp_randinit_lc_2exp_size 128", rstate);
  gmp_randclear (rstate);

  /* degenerate always zeros */
  mpz_set_ui (a, 0L);
  gmp_randinit_lc_2exp (rstate, a, 0L, 8L);
  (*func) ("gmp_randinit_lc_2exp a=0 c=0 m=8", rstate);
  gmp_randclear (rstate);

  /* degenerate always FFs */
  mpz_set_ui (a, 0L);
  gmp_randinit_lc_2exp (rstate, a, 0xFFL, 8L);
  (*func) ("gmp_randinit_lc_2exp a=0 c=0xFF m=8", rstate);
  gmp_randclear (rstate);

  mpz_clear (a);
}


/* Return +infinity if available, or 0 if not.
   We don't want to use libm, so INFINITY or other system values are not
   used here.  */
double
tests_infinity_d (void)
{
#if _GMP_IEEE_FLOATS
  union ieee_double_extract x;
  x.s.exp = 2047;
  x.s.manl = 0;
  x.s.manh = 0;
  x.s.sig = 0;
  return x.d;
#else
  return 0;
#endif
}


/* Return non-zero if d is an infinity (either positive or negative).
   Don't want libm, so don't use isinf() or other system tests.  */
int
tests_isinf (double d)
{
#if _GMP_IEEE_FLOATS
  union ieee_double_extract x;
  x.d = d;
  return (x.s.exp == 2047 && x.s.manl == 0 && x.s.manh == 0);
#else
  return 0;
#endif
}


/* Set the hardware floating point rounding mode.  Same mode values as mpfr,
   namely 0=nearest, 1=tozero, 2=up, 3=down.  Return 1 if successful, 0 if
   not.  */
int
tests_hardware_setround (int mode)
{
#if WANT_ASSEMBLY && HAVE_HOST_CPU_FAMILY_x86
  int  rc;
  switch (mode) {
  case 0: rc = 0; break;  /* nearest */
  case 1: rc = 3; break;  /* tozero  */
  case 2: rc = 2; break;  /* up      */
  case 3: rc = 1; break;  /* down    */
  default:
    return 0;
  }
  x86_fldcw ((x86_fstcw () & ~0xC00) | (rc << 10));
  return 1;
#endif

  return 0;
}

/* Return the hardware floating point rounding mode, or -1 if unknown. */
int
tests_hardware_getround (void)
{
#if WANT_ASSEMBLY && HAVE_HOST_CPU_FAMILY_x86
  switch ((x86_fstcw () & ~0xC00) >> 10) {
  case 0: return 0; break;  /* nearest */
  case 1: return 3; break;  /* down    */
  case 2: return 2; break;  /* up      */
  case 3: return 1; break;  /* tozero  */
  }
#endif

  return -1;
}


/* tests_dbl_mant_bits() determines by experiment the number of bits in the
   mantissa of a "double".  If it's not possible to find a value (perhaps
   due to the compiler optimizing too aggressively), then return 0.

   This code is used rather than DBL_MANT_DIG from <float.h> since ancient
   systems like SunOS don't have that file, and since one GNU/Linux ARM
   system was seen where the float emulation seemed to have only 32 working
   bits, not the 53 float.h claimed.  */

int
tests_dbl_mant_bits (void)
{
  static int n = -1;
  volatile double x, y, d;

  if (n != -1)
    return n;

  n = 1;
  x = 2.0;
  for (;;)
    {
      /* see if 2^(n+1)+1 can be formed without rounding, if so then
         continue, if not then "n" is the answer */
      y = x + 1.0;
      d = y - x;
      if (d != 1.0)
        {
#if defined (DBL_MANT_DIG) && DBL_RADIX == 2
          if (n != DBL_MANT_DIG)
            printf ("Warning, tests_dbl_mant_bits got %d but DBL_MANT_DIG says %d\n", n, DBL_MANT_DIG);
#endif
          break;
        }

      x *= 2;
      n++;

      if (n > 1000)
        {
          printf ("Oops, tests_dbl_mant_bits can't determine mantissa size\n");
          n = 0;
          break;
        }
    }
  return n;
}


/* See tests_setjmp_sigfpe in tests.h. */

jmp_buf    tests_sigfpe_target;

RETSIGTYPE
tests_sigfpe_handler (int sig)
{
  longjmp (tests_sigfpe_target, 1);
}

void
tests_sigfpe_done (void)
{
  signal (SIGFPE, SIG_DFL);
}
