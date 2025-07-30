/*
 * Output various information about GMP and MPFR.
 */

/*
Copyright 2010-2017 Free Software Foundation, Inc.
Contributed by the AriC and Caramba projects, INRIA.

This file is part of the GNU MPFR Library.

The GNU MPFR Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The GNU MPFR Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the GNU MPFR Library; see the file COPYING.LESSER.  If not, see
http://www.gnu.org/licenses/ or write to the Free Software Foundation, Inc.,
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA.
*/

#include <stdio.h>
#include <limits.h>
#include <gmp.h>
#include <mpfr.h>

/* The following failure can occur if GMP has been rebuilt with
 * a different ABI, e.g.
 *   1. GMP built with ABI=mode32.
 *   2. MPFR built against this GMP version.
 *   3. GMP rebuilt with ABI=32.
 */
static void failure_test (void)
{
  mpfr_t x;

  mpfr_init2 (x, 128);
  mpfr_set_str (x, "17", 0, MPFR_RNDN);
  if (mpfr_cmp_ui (x, 17) != 0)
    printf ("\nFailure in mpfr_set_str! Probably an unmatched ABI!\n");
  mpfr_clear (x);
}

static void patches (void)
{
  const char *p = mpfr_get_patches ();
  printf ("MPFR patches: %s\n", p[0] ? p : "[none]");
}

#define STRINGIZE(S) #S
#define MAKE_STR(S) STRINGIZE(S)

#define SIGNED_STR(V) ((V) < 0 ? "signed" : "unsigned")
#define SIGNED(I) SIGNED_STR((I) - (I) - 1)

int main (void)
{
  unsigned long c;
  mp_limb_t t[4];
  int i;

  /* Casts are for C++ compilers. */
  for (i = 0; i < (int) (sizeof (t) / sizeof (mp_limb_t)); i++)
    t[i] = (mp_limb_t) -1;

  /**************** Information about the C implementation ****************/

  /* This is useful, as this can affect the behavior of MPFR. */

#define COMP "Compiler: "
#ifdef __INTEL_COMPILER
# ifdef __VERSION__
#  define ICCV " [" __VERSION__ "]"
# else
#  define ICCV ""
# endif
  printf (COMP "ICC %d.%d.%d" ICCV "\n", __INTEL_COMPILER / 100,
          __INTEL_COMPILER % 100, __INTEL_COMPILER_UPDATE);
#elif (defined(__GNUC__) || defined(__clang__)) && defined(__VERSION__)
# ifdef __clang__
#  define COMP2 COMP
# else
#  define COMP2 COMP "GCC "
# endif
  printf (COMP2 "%s\n", __VERSION__);
#endif

#if defined(__STDC__) || defined(__STDC_VERSION__)
  printf ("C/C++: __STDC__ = "
#if defined(__STDC__)
          MAKE_STR(__STDC__)
#else
          "undef"
#endif
          ", __STDC_VERSION__ = "
#if defined(__STDC_VERSION__)
          MAKE_STR(__STDC_VERSION__)
#else
          "undef"
#endif
#if defined(__cplusplus)
          ", C++"
#endif
          "\n");
#endif

#if defined(__GNUC__)
  printf ("GNU compatibility: __GNUC__ = " MAKE_STR(__GNUC__)
          ", __GNUC_MINOR__ = "
#if defined(__GNUC_MINOR__)
          MAKE_STR(__GNUC_MINOR__)
#else
          "undef"
#endif
          "\n");
#endif

#if defined(__ICC) || defined(__INTEL_COMPILER)
  printf ("Intel compiler: __ICC = "
#if defined(__ICC)
          MAKE_STR(__ICC)
#else
          "undef"
#endif
          ", __INTEL_COMPILER = "
#if defined(__INTEL_COMPILER)
          MAKE_STR(__INTEL_COMPILER)
#else
          "undef"
#endif
          "\n");
#endif

#if defined(_WIN32) || defined(_MSC_VER)
  printf ("MS Windows: _WIN32 = "
#if defined(_WIN32)
          MAKE_STR(_WIN32)
#else
          "undef"
#endif
          ", _MSC_VER = "
#if defined(_MSC_VER)
          MAKE_STR(_MSC_VER)
#else
          "undef"
#endif
          "\n");
#endif

#if defined(__GLIBC__)
  printf ("GNU C library: __GLIBC__ = " MAKE_STR(__GLIBC__)
          ", __GLIBC_MINOR__ = "
#if defined(__GLIBC_MINOR__)
          MAKE_STR(__GLIBC_MINOR__)
#else
          "undef"
#endif
          "\n");
#endif

  printf ("\n");

  /************************************************************************/

#if defined(__MPIR_VERSION)
  printf ("MPIR ....  Library: %-12s  Header: %d.%d.%d\n",
          mpir_version, __MPIR_VERSION, __MPIR_VERSION_MINOR,
          __MPIR_VERSION_PATCHLEVEL);
#else
  printf ("GMP .....  Library: %-12s  Header: %d.%d.%d\n",
          gmp_version, __GNU_MP_VERSION, __GNU_MP_VERSION_MINOR,
          __GNU_MP_VERSION_PATCHLEVEL);
#endif

  printf ("MPFR ....  Library: %-12s  Header: %s (based on %d.%d.%d)\n",
          mpfr_get_version (), MPFR_VERSION_STRING, MPFR_VERSION_MAJOR,
          MPFR_VERSION_MINOR, MPFR_VERSION_PATCHLEVEL);

  printf ("MPFR features: TLS = %s, decimal = %s",
          mpfr_buildopt_tls_p () ? "yes" : "no",
          mpfr_buildopt_decimal_p () ? "yes" : "no");
#if MPFR_VERSION_MAJOR > 3 || MPFR_VERSION_MINOR >= 1
  printf (", GMP internals = %s\nMPFR tuning: %s",
          mpfr_buildopt_gmpinternals_p () ? "yes" : "no",
          mpfr_buildopt_tune_case ());
#endif  /* 3.1 */
  printf ("\n");

  patches ();

  printf ("\n");
#ifdef __GMP_CC
  printf ("__GMP_CC = \"%s\"\n", __GMP_CC);
#endif
#ifdef __GMP_CFLAGS
  printf ("__GMP_CFLAGS = \"%s\"\n", __GMP_CFLAGS);
#endif
  printf ("GMP_LIMB_BITS     = %d\n", (int) GMP_LIMB_BITS);
  printf ("GMP_NAIL_BITS     = %d\n", (int) GMP_NAIL_BITS);
  printf ("GMP_NUMB_BITS     = %d\n", (int) GMP_NUMB_BITS);
  printf ("mp_bits_per_limb  = %d\n", (int) mp_bits_per_limb);
  printf ("sizeof(mp_limb_t) = %d\n", (int) sizeof(mp_limb_t));
  if (mp_bits_per_limb != GMP_LIMB_BITS)
    printf ("Warning! mp_bits_per_limb != GMP_LIMB_BITS\n");
  if (GMP_LIMB_BITS != sizeof(mp_limb_t) * CHAR_BIT)
    printf ("Warning! GMP_LIMB_BITS != sizeof(mp_limb_t) * CHAR_BIT\n");

  c = mpn_popcount (t, 1);
  printf ("The GMP library expects %lu bits in a mp_limb_t.\n", c);
  if (c != GMP_LIMB_BITS)
    printf ("Warning! This is different from GMP_LIMB_BITS!\n"
            "Different ABI caused by a GMP library upgrade?\n");

  printf ("\n");
  printf ("sizeof(mpfr_prec_t) = %d (%s type)\n", (int) sizeof(mpfr_prec_t),
          SIGNED_STR((mpfr_prec_t) -1));
  printf ("sizeof(mpfr_exp_t)  = %d (%s type)\n", (int) sizeof(mpfr_exp_t),
          SIGNED_STR((mpfr_exp_t) -1));
#ifdef _MPFR_PREC_FORMAT
  printf ("_MPFR_PREC_FORMAT = %d\n", (int) _MPFR_PREC_FORMAT);
#endif
  /* Note: "long" is sufficient for all current _MPFR_PREC_FORMAT values
     (1, 2, 3). Thus we do not need to depend on ISO C99 or later. */
  printf ("MPFR_PREC_MIN = %ld (%s)\n", (long) MPFR_PREC_MIN,
          SIGNED (MPFR_PREC_MIN));
  printf ("MPFR_PREC_MAX = %ld (%s)\n", (long) MPFR_PREC_MAX,
          SIGNED (MPFR_PREC_MAX));
#ifdef _MPFR_EXP_FORMAT
  printf ("_MPFR_EXP_FORMAT = %d\n", (int) _MPFR_EXP_FORMAT);
#endif
  printf ("sizeof(mpfr_t) = %d\n", (int) sizeof(mpfr_t));
  printf ("sizeof(mpfr_ptr) = %d\n", (int) sizeof(mpfr_ptr));
  failure_test ();

  return 0;
}
