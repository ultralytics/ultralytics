/* Check the values of some constants.

Copyright 2000-2003, 2014 Free Software Foundation, Inc.

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
#include "tests.h"


#ifdef ULONG_MAX
const char *ulong_max_def = "defined";
#else
const char *ulong_max_def = "not defined";
#endif
#ifdef LONG_MAX
const char *long_max_def = "defined";
#else
const char *long_max_def = "not defined";
#endif

#ifdef UINT_MAX
const char *uint_max_def = "defined";
#else
const char *uint_max_def = "not defined";
#endif
#ifdef INT_MAX
const char *int_max_def = "defined";
#else
const char *int_max_def = "not defined";
#endif

#ifdef USHRT_MAX
const char *ushrt_max_def = "defined";
#else
const char *ushrt_max_def = "not defined";
#endif
#ifdef SHRT_MAX
const char *shrt_max_def = "defined";
#else
const char *shrt_max_def = "not defined";
#endif

#include "gmp-impl.h"
#include "longlong.h"


#ifdef _LONG_LONG_LIMB
#define LL(l,ll)  ll
#else
#define LL(l,ll)  l
#endif

#if __GMP_MP_SIZE_T_INT
#define SS(i,l)   i
#else
#define SS(i,l)   l
#endif


#define CHECK_LIMB_S(x, xname, y, yname)                \
  do {                                                  \
    if ((x) != (y))                                     \
      {                                                 \
        printf (LL("%s == %lx, but %s == %lx\n",        \
                   "%s == %llx, but %s == %llx\n"),     \
                xname, x, yname, y);                    \
        error = 1;                                      \
      }                                                 \
  } while (0)

#define CHECK_INT_S(x, xname, y, yname)                                 \
  do {                                                                  \
    if ((x) != (y))                                                     \
      {                                                                 \
        printf ("%s == %d, but %s == %d\n", xname, x, yname, y);        \
        error = 1;                                                      \
      }                                                                 \
  } while (0)



#define CHECK_CONDITION_S(x, xname)             \
  do {                                          \
    if (!(x))                                   \
      {                                         \
        printf ("%s is false\n", xname);        \
        error = 1;                              \
      }                                         \
  } while (0)


/* How many bits seem to work in the given type. */
#define CALC_BITS(result, type) \
  do {                          \
    type  n = 1;                \
    result = 0;                 \
    while (n != 0)              \
      {                         \
        n <<= 1;                \
        result++;               \
      }                         \
  } while (0)

#define CHECK_BITS_S(constant, constant_name, type)     \
  do {                                                  \
    int   calculated;                                   \
    CALC_BITS (calculated, type);                       \
    if (calculated != constant)                         \
      {                                                 \
        printf ("%s == %d, but calculated %d\n",        \
                constant_name, constant, calculated);   \
        error = 1;                                      \
      }                                                 \
  } while (0)


#define CHECK_HIGHBIT_S(value, value_name, type, format)        \
  do {                                                          \
    type  n = value;                                            \
    if (n == 0)                                                 \
      {                                                         \
        printf ("%s == 0\n", value_name);                       \
        error = 1;                                              \
      }                                                         \
    n <<= 1;                                                    \
    if (n != 0)                                                 \
      {                                                         \
        printf ("%s << 1 = ", value_name);                      \
        printf (format, n);                                     \
        printf (" != 0\n");                                     \
        error = 1;                                              \
      }                                                         \
  } while (0)


#define CHECK_MAX_S(max_val, max_name, min_val, min_name, type, format) \
  do {                                                                  \
    type  maxval = max_val;                                             \
    type  minval = min_val;                                             \
    type  n = maxval;                                                   \
    n++;                                                                \
    if (n != minval)                                                    \
      {                                                                 \
        printf ("%s + 1 = ", max_name);                                 \
        printf (format, n);                                             \
        printf (" != %s = ", min_name);                                 \
        printf (format, minval);                                        \
        printf ("\n");                                                  \
        error = 1;                                                      \
      }                                                                 \
    if (maxval <= minval)                                               \
      {                                                                 \
        printf ("%s = ", max_name);                                     \
        printf (format, maxval);                                        \
        printf (" <= %s = ", min_name);                                 \
        printf (format, minval);                                        \
        printf ("\n");                                                  \
        error = 1;                                                      \
      }                                                                 \
  } while (0)


#define CHECK_LIMB(x,y)      CHECK_LIMB_S (x, #x, y, #y)
#define CHECK_INT(x,y)       CHECK_INT_S (x, #x, y, #y)
#define CHECK_CONDITION(x)   CHECK_CONDITION_S (x, #x)
#define CHECK_BITS(c,t)      CHECK_BITS_S (c, #c, t)
#define CHECK_MAX(m,n,t,f)   CHECK_MAX_S (m, #m, n, #n, t, f)
#define CHECK_HIGHBIT(n,t,f) CHECK_HIGHBIT_S (n, #n, t, f)


/* The tests below marked "Bad!" fail on Cray T90 systems, where int, short
   and mp_size_t are 48 bits or some such but don't wraparound in a plain
   twos complement fashion.  In particular,

       INT_HIGHBIT << 1 = 0xFFFFC00000000000 != 0
       INT_MAX + 1 = 35184372088832 != INT_MIN = -35184372088832

   This is a bit bizarre, but doesn't matter because GMP doesn't rely on any
   particular overflow behaviour for int or short, only for mp_limb_t.  */

int
main (int argc, char *argv[])
{
  int  error = 0;

  CHECK_INT (GMP_LIMB_BYTES, (int) sizeof(mp_limb_t));
  CHECK_INT (mp_bits_per_limb, GMP_LIMB_BITS);

  CHECK_BITS (GMP_LIMB_BITS, mp_limb_t);
  CHECK_BITS (BITS_PER_ULONG, unsigned long);

  CHECK_HIGHBIT (GMP_LIMB_HIGHBIT, mp_limb_t,      LL("0x%lX","0x%llX"));
  CHECK_HIGHBIT (ULONG_HIGHBIT,     unsigned long,  "0x%lX");
  CHECK_HIGHBIT (UINT_HIGHBIT,      unsigned int,   "0x%X");
  CHECK_HIGHBIT (USHRT_HIGHBIT,     unsigned short, "0x%hX");
#if 0 /* Bad! */
  CHECK_HIGHBIT (LONG_HIGHBIT,      long,           "0x%lX");
  CHECK_HIGHBIT (INT_HIGHBIT,       int,            "0x%X");
  CHECK_HIGHBIT (SHRT_HIGHBIT,      short,          "0x%hX");
#endif

#if 0 /* Bad! */
  CHECK_MAX (LONG_MAX,      LONG_MIN,      long,           "%ld");
  CHECK_MAX (INT_MAX,       INT_MIN,       int,            "%d");
  CHECK_MAX (SHRT_MAX,      SHRT_MIN,      short,          "%hd");
#endif
  CHECK_MAX (ULONG_MAX,     0,             unsigned long,  "%lu");
  CHECK_MAX (UINT_MAX,      0,             unsigned int,   "%u");
  CHECK_MAX (USHRT_MAX,     0,             unsigned short, "%hu");
#if 0 /* Bad! */
  CHECK_MAX (MP_SIZE_T_MAX, MP_SIZE_T_MIN, mp_size_t,      SS("%d","%ld"));
#endif

  /* UHWtype should have at least enough bits for half a UWtype */
  {
    int  bits_per_UWtype, bits_per_UHWtype;
    CALC_BITS (bits_per_UWtype,  UWtype);
    CALC_BITS (bits_per_UHWtype, UHWtype);
    CHECK_CONDITION (2*bits_per_UHWtype >= bits_per_UWtype);
  }

  ASSERT_ALWAYS_LIMB (MODLIMB_INVERSE_3);
  {
    mp_limb_t  modlimb_inverse_3_calc;
    binvert_limb (modlimb_inverse_3_calc, CNST_LIMB(3));
    ASSERT_ALWAYS_LIMB (modlimb_inverse_3_calc);
    CHECK_LIMB (MODLIMB_INVERSE_3, modlimb_inverse_3_calc);
  }
  {
    mp_limb_t  MODLIMB_INVERSE_3_times_3
      = (MODLIMB_INVERSE_3 * CNST_LIMB(3)) & GMP_NUMB_MASK;
    CHECK_LIMB (MODLIMB_INVERSE_3_times_3, CNST_LIMB(1));
  }

  {
    mp_limb_t  hi, lo;
    hi = refmpn_umul_ppmm (&lo, GMP_NUMB_CEIL_MAX_DIV3-1,
                           CNST_LIMB(3) << GMP_NAIL_BITS);
    if (! (hi < 1))
      {
        printf ("GMP_NUMB_CEIL_MAX_DIV3 too big\n");
        error = 1;
      }
    hi = refmpn_umul_ppmm (&lo, GMP_NUMB_CEIL_MAX_DIV3,
                           CNST_LIMB(3) << GMP_NAIL_BITS);
    if (! (hi >= 1))
      {
        printf ("GMP_NUMB_CEIL_MAX_DIV3 too small\n");
        error = 1;
      }
  }

  {
    mp_limb_t  hi, lo;
    hi = refmpn_umul_ppmm (&lo, GMP_NUMB_CEIL_2MAX_DIV3-1,
                           CNST_LIMB(3) << GMP_NAIL_BITS);
    if (! (hi < 2))
      {
        printf ("GMP_NUMB_CEIL_2MAX_DIV3 too big\n");
        error = 1;
      }
    hi = refmpn_umul_ppmm (&lo, GMP_NUMB_CEIL_2MAX_DIV3,
                           CNST_LIMB(3) << GMP_NAIL_BITS);
    if (! (hi >= 2))
      {
        printf ("GMP_NUMB_CEIL_2MAX_DIV3 too small\n");
        error = 1;
      }
  }

#ifdef PP_INVERTED
  {
    mp_limb_t  pp_inverted_calc;
    invert_limb (pp_inverted_calc, PP);
    CHECK_LIMB (PP_INVERTED, pp_inverted_calc);
  }
#endif

  if (argc >= 2 || error)
    {
      int  bits;

      printf ("\n");
      printf ("After gmp.h,\n");
      printf ("  ULONG_MAX  %s\n", ulong_max_def);
      printf ("  LONG_MAX   %s\n", long_max_def);
      printf ("  UINT_MAX   %s\n", uint_max_def);
      printf ("  INT_MAX    %s\n", int_max_def);
      printf ("  USHRT_MAX  %s\n", ushrt_max_def);
      printf ("  SHRT_MAX   %s\n", shrt_max_def);
      printf ("\n");

#ifdef _CRAY
      printf ("_CRAY is defined, so limits.h is being used\n");
#endif

      printf ("ULONG_MAX      %lX\n", ULONG_MAX);
      printf ("ULONG_HIGHBIT  %lX\n", ULONG_HIGHBIT);
      printf ("LONG_MAX       %lX\n", LONG_MAX);
      printf ("LONG_MIN       %lX\n", LONG_MIN);

      printf ("UINT_MAX       %X\n", UINT_MAX);
      printf ("UINT_HIGHBIT   %X\n", UINT_HIGHBIT);
      printf ("INT_MAX        %X\n", INT_MAX);
      printf ("INT_MIN        %X\n", INT_MIN);

      printf ("USHRT_MAX      %X\n", USHRT_MAX);
      printf ("USHRT_HIGHBIT  %X\n", USHRT_HIGHBIT);
      printf ("SHRT_MAX       %X\n", SHRT_MAX);
      printf ("SHRT_MIN       %X\n", SHRT_MIN);

      printf ("\n");
      printf ("Bits\n");
      CALC_BITS (bits, long);           printf ("  long           %d\n", bits);
      CALC_BITS (bits, int);            printf ("  int            %d\n", bits);
      CALC_BITS (bits, short);          printf ("  short          %d\n", bits);
      CALC_BITS (bits, unsigned long);  printf ("  unsigned long  %d\n", bits);
      CALC_BITS (bits, unsigned int);   printf ("  unsigned int   %d\n", bits);
      CALC_BITS (bits, unsigned short); printf ("  unsigned short %d\n", bits);
      CALC_BITS (bits, mp_size_t);      printf ("  mp_size_t      %d\n", bits);
    }

  if (error)
    abort ();

  exit (0);
}
