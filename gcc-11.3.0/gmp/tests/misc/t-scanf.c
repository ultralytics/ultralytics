/* Test gmp_scanf and related functions.

Copyright 2001-2004 Free Software Foundation, Inc.

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


/* Usage: t-scanf [-s]

   -s  Check the data against the system scanf, where possible.  This is
       only an option since we don't want to fail if the system scanf is
       faulty or strange.

   There's some fairly unattractive repetition between check_z, check_q and
   check_f, but enough differences to make a common loop or a set of macros
   seem like too much trouble. */

#include "config.h"	/* needed for the HAVE_, could also move gmp incls */

#include <stdarg.h>

#include <stddef.h>    /* for ptrdiff_t */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if HAVE_INTTYPES_H
# include <inttypes.h> /* for intmax_t */
#else
# if HAVE_STDINT_H
#  include <stdint.h>
# endif
#endif

#if HAVE_UNISTD_H
#include <unistd.h>  /* for unlink */
#endif

#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"


#define TEMPFILE  "t-scanf.tmp"

int   option_libc_scanf = 0;

typedef int (*fun_t) (const char *, const char *, void *, void *);


/* This problem was seen on powerpc7450-apple-darwin7.0.0, sscanf returns 0
   where it should return EOF.  A workaround in gmp_sscanf would be a bit
   tedious, and since this is a rather obvious libc bug, quite likely
   affecting other programs, we'll just suppress affected tests for now.  */
int
test_sscanf_eof_ok (void)
{
  static int  result = -1;

  if (result == -1)
    {
      int  x;
      if (sscanf ("", "%d", &x) == EOF)
        {
          result = 1;
        }
      else
        {
          printf ("Warning, sscanf(\"\",\"%%d\",&x) doesn't return EOF.\n");
          printf ("This affects gmp_sscanf, tests involving it will be suppressed.\n");
          printf ("You should try to get a fix for your libc.\n");
          result = 0;
        }
    }
  return result;
}


/* Convert fmt from a GMP scanf format string to an equivalent for a plain
   libc scanf, for example "%Zd" becomes "%ld".  Return 1 if this succeeds,
   0 if it cannot (or should not) be done.  */
int
libc_scanf_convert (char *fmt)
{
  char  *p = fmt;

  if (! option_libc_scanf)
    return 0;

  for ( ; *fmt != '\0'; fmt++)
    {
      switch (*fmt) {
      case 'F':
      case 'Q':
      case 'Z':
        /* transmute */
        *p++ = 'l';
        break;
      default:
        *p++ = *fmt;
        break;
      }
    }
  *p = '\0';
  return 1;
}


long  got_ftell;
int   fromstring_next_c;

/* Call gmp_fscanf, reading the "input" string data provided. */
int
fromstring_gmp_fscanf (const char *input, const char *fmt, ...)
{
  va_list  ap;
  FILE     *fp;
  int      ret;
  va_start (ap, fmt);

  fp = fopen (TEMPFILE, "w+");
  ASSERT_ALWAYS (fp != NULL);
  ASSERT_ALWAYS (fputs (input, fp) != EOF);
  ASSERT_ALWAYS (fflush (fp) == 0);
  rewind (fp);

  ret = gmp_vfscanf (fp, fmt, ap);
  got_ftell = ftell (fp);
  ASSERT_ALWAYS (got_ftell != -1L);

  fromstring_next_c = getc (fp);

  ASSERT_ALWAYS (fclose (fp) == 0);
  va_end (ap);
  return ret;
}


int
fun_gmp_sscanf (const char *input, const char *fmt, void *a1, void *a2)
{
  if (a2 == NULL)
    return gmp_sscanf (input, fmt, a1);
  else
    return gmp_sscanf (input, fmt, a1, a2);
}

int
fun_gmp_fscanf (const char *input, const char *fmt, void *a1, void *a2)
{
  if (a2 == NULL)
    return fromstring_gmp_fscanf (input, fmt, a1);
  else
    return fromstring_gmp_fscanf (input, fmt, a1, a2);
}


int
fun_fscanf (const char *input, const char *fmt, void *a1, void *a2)
{
  FILE  *fp;
  int   ret;

  fp = fopen (TEMPFILE, "w+");
  ASSERT_ALWAYS (fp != NULL);
  ASSERT_ALWAYS (fputs (input, fp) != EOF);
  ASSERT_ALWAYS (fflush (fp) == 0);
  rewind (fp);

  if (a2 == NULL)
    ret = fscanf (fp, fmt, a1);
  else
    ret = fscanf (fp, fmt, a1, a2);

  got_ftell = ftell (fp);
  ASSERT_ALWAYS (got_ftell != -1L);

  fromstring_next_c = getc (fp);

  ASSERT_ALWAYS (fclose (fp) == 0);
  return ret;
}


/* On various old systems, for instance HP-UX 9, the C library sscanf needs
   to be able to write into the input string.  Ensure that this is possible,
   when gcc is putting the test data into a read-only section.

   Actually we ought to only need this under SSCANF_WRITABLE_INPUT from
   configure, but it's just as easy to do it unconditionally, and in any
   case this code is only executed under the -s option.  */

int
fun_sscanf (const char *input, const char *fmt, void *a1, void *a2)
{
  char    *input_writable;
  size_t  size;
  int     ret;

  size = strlen (input) + 1;
  input_writable = (char *) (*__gmp_allocate_func) (size);
  memcpy (input_writable, input, size);

  if (a2 == NULL)
    ret = sscanf (input_writable, fmt, a1);
  else
    ret = sscanf (input_writable, fmt, a1, a2);

  (*__gmp_free_func) (input_writable, size);
  return ret;
}


/* whether the format string consists entirely of ignored fields */
int
fmt_allignore (const char *fmt)
{
  int  saw_star = 1;
  for ( ; *fmt != '\0'; fmt++)
    {
      switch (*fmt) {
      case '%':
        if (! saw_star)
          return 0;
        saw_star = 0;
        break;
      case '*':
        saw_star = 1;
        break;
      }
    }
  return 1;
}

void
check_z (void)
{
  static const struct {
    const char  *fmt;
    const char  *input;
    const char  *want;
    int         want_ret;
    long        want_ftell;
    int         want_upto;
    int         not_glibc;

  } data[] = {

    { "%Zd",    "0",    "0", 1, -1, -1 },
    { "%Zd",    "1",    "1", 1, -1, -1 },
    { "%Zd",  "123",  "123", 1, -1, -1 },
    { "%Zd",   "+0",    "0", 1, -1, -1 },
    { "%Zd",   "+1",    "1", 1, -1, -1 },
    { "%Zd", "+123",  "123", 1, -1, -1 },
    { "%Zd",   "-0",    "0", 1, -1, -1 },
    { "%Zd",   "-1",   "-1", 1, -1, -1 },
    { "%Zd", "-123", "-123", 1, -1, -1 },

    { "%Zo",    "0",    "0", 1, -1, -1 },
    { "%Zo",  "173",  "123", 1, -1, -1 },
    { "%Zo",   "+0",    "0", 1, -1, -1 },
    { "%Zo", "+173",  "123", 1, -1, -1 },
    { "%Zo",   "-0",    "0", 1, -1, -1 },
    { "%Zo", "-173", "-123", 1, -1, -1 },

    { "%Zx",    "0",    "0", 1, -1, -1 },
    { "%Zx",   "7b",  "123", 1, -1, -1 },
    { "%Zx",   "7b",  "123", 1, -1, -1 },
    { "%Zx",   "+0",    "0", 1, -1, -1 },
    { "%Zx",  "+7b",  "123", 1, -1, -1 },
    { "%Zx",  "+7b",  "123", 1, -1, -1 },
    { "%Zx",   "-0",   "-0", 1, -1, -1 },
    { "%Zx",  "-7b", "-123", 1, -1, -1 },
    { "%Zx",  "-7b", "-123", 1, -1, -1 },
    { "%ZX",    "0",    "0", 1, -1, -1 },
    { "%ZX",   "7b",  "123", 1, -1, -1 },
    { "%ZX",   "7b",  "123", 1, -1, -1 },
    { "%ZX",   "+0",    "0", 1, -1, -1 },
    { "%ZX",  "+7b",  "123", 1, -1, -1 },
    { "%ZX",  "+7b",  "123", 1, -1, -1 },
    { "%ZX",   "-0",   "-0", 1, -1, -1 },
    { "%ZX",  "-7b", "-123", 1, -1, -1 },
    { "%ZX",  "-7b", "-123", 1, -1, -1 },
    { "%Zx",    "0",    "0", 1, -1, -1 },
    { "%Zx",   "7B",  "123", 1, -1, -1 },
    { "%Zx",   "7B",  "123", 1, -1, -1 },
    { "%Zx",   "+0",    "0", 1, -1, -1 },
    { "%Zx",  "+7B",  "123", 1, -1, -1 },
    { "%Zx",  "+7B",  "123", 1, -1, -1 },
    { "%Zx",   "-0",   "-0", 1, -1, -1 },
    { "%Zx",  "-7B", "-123", 1, -1, -1 },
    { "%Zx",  "-7B", "-123", 1, -1, -1 },
    { "%ZX",    "0",    "0", 1, -1, -1 },
    { "%ZX",   "7B",  "123", 1, -1, -1 },
    { "%ZX",   "7B",  "123", 1, -1, -1 },
    { "%ZX",   "+0",    "0", 1, -1, -1 },
    { "%ZX",  "+7B",  "123", 1, -1, -1 },
    { "%ZX",  "+7B",  "123", 1, -1, -1 },
    { "%ZX",   "-0",   "-0", 1, -1, -1 },
    { "%ZX",  "-7B", "-123", 1, -1, -1 },
    { "%ZX",  "-7B", "-123", 1, -1, -1 },

    { "%Zi",    "0",    "0", 1, -1, -1 },
    { "%Zi",    "1",    "1", 1, -1, -1 },
    { "%Zi",  "123",  "123", 1, -1, -1 },
    { "%Zi",   "+0",    "0", 1, -1, -1 },
    { "%Zi",   "+1",    "1", 1, -1, -1 },
    { "%Zi", "+123",  "123", 1, -1, -1 },
    { "%Zi",   "-0",    "0", 1, -1, -1 },
    { "%Zi",   "-1",   "-1", 1, -1, -1 },
    { "%Zi", "-123", "-123", 1, -1, -1 },

    { "%Zi",    "00",    "0", 1, -1, -1 },
    { "%Zi",  "0173",  "123", 1, -1, -1 },
    { "%Zi",   "+00",    "0", 1, -1, -1 },
    { "%Zi", "+0173",  "123", 1, -1, -1 },
    { "%Zi",   "-00",    "0", 1, -1, -1 },
    { "%Zi", "-0173", "-123", 1, -1, -1 },

    { "%Zi",    "0x0",    "0", 1, -1, -1 },
    { "%Zi",   "0x7b",  "123", 1, -1, -1 },
    { "%Zi",   "0x7b",  "123", 1, -1, -1 },
    { "%Zi",   "+0x0",    "0", 1, -1, -1 },
    { "%Zi",  "+0x7b",  "123", 1, -1, -1 },
    { "%Zi",  "+0x7b",  "123", 1, -1, -1 },
    { "%Zi",   "-0x0",   "-0", 1, -1, -1 },
    { "%Zi",  "-0x7b", "-123", 1, -1, -1 },
    { "%Zi",  "-0x7b", "-123", 1, -1, -1 },
    { "%Zi",    "0X0",    "0", 1, -1, -1 },
    { "%Zi",   "0X7b",  "123", 1, -1, -1 },
    { "%Zi",   "0X7b",  "123", 1, -1, -1 },
    { "%Zi",   "+0X0",    "0", 1, -1, -1 },
    { "%Zi",  "+0X7b",  "123", 1, -1, -1 },
    { "%Zi",  "+0X7b",  "123", 1, -1, -1 },
    { "%Zi",   "-0X0",   "-0", 1, -1, -1 },
    { "%Zi",  "-0X7b", "-123", 1, -1, -1 },
    { "%Zi",  "-0X7b", "-123", 1, -1, -1 },
    { "%Zi",    "0x0",    "0", 1, -1, -1 },
    { "%Zi",   "0x7B",  "123", 1, -1, -1 },
    { "%Zi",   "0x7B",  "123", 1, -1, -1 },
    { "%Zi",   "+0x0",    "0", 1, -1, -1 },
    { "%Zi",  "+0x7B",  "123", 1, -1, -1 },
    { "%Zi",  "+0x7B",  "123", 1, -1, -1 },
    { "%Zi",   "-0x0",   "-0", 1, -1, -1 },
    { "%Zi",  "-0x7B", "-123", 1, -1, -1 },
    { "%Zi",  "-0x7B", "-123", 1, -1, -1 },
    { "%Zi",    "0X0",    "0", 1, -1, -1 },
    { "%Zi",   "0X7B",  "123", 1, -1, -1 },
    { "%Zi",   "0X7B",  "123", 1, -1, -1 },
    { "%Zi",   "+0X0",    "0", 1, -1, -1 },
    { "%Zi",  "+0X7B",  "123", 1, -1, -1 },
    { "%Zi",  "+0X7B",  "123", 1, -1, -1 },
    { "%Zi",   "-0X0",   "-0", 1, -1, -1 },
    { "%Zi",  "-0X7B", "-123", 1, -1, -1 },
    { "%Zi",  "-0X7B", "-123", 1, -1, -1 },

    { "%Zd",    " 0",    "0", 1, -1, -1 },
    { "%Zd",   "  0",    "0", 1, -1, -1 },
    { "%Zd",  "   0",    "0", 1, -1, -1 },
    { "%Zd",   "\t0",    "0", 1, -1, -1 },
    { "%Zd", "\t\t0",    "0", 1, -1, -1 },

    { "hello%Zd",      "hello0",       "0", 1, -1, -1 },
    { "hello%Zd",      "hello 0",      "0", 1, -1, -1 },
    { "hello%Zd",      "hello \t0",    "0", 1, -1, -1 },
    { "hello%Zdworld", "hello 0world", "0", 1, -1, -1 },

    { "hello%*Zd",      "hello0",       "-999", 0, -1, -1 },
    { "hello%*Zd",      "hello 0",      "-999", 0, -1, -1 },
    { "hello%*Zd",      "hello \t0",    "-999", 0, -1, -1 },
    { "hello%*Zdworld", "hello 0world", "-999", 0, -1, -1 },

    { "%Zd",    "",     "-999", -1, -1, -555 },
    { "%Zd",    " ",    "-999", -1, -1, -555 },
    { " %Zd",   "",     "-999", -1, -1, -555 },
    { "xyz%Zd", "",     "-999", -1, -1, -555 },

    { "%*Zd",    "",     "-999", -1, -1, -555 },
    { " %*Zd",   "",     "-999", -1, -1, -555 },
    { "xyz%*Zd", "",     "-999", -1, -1, -555 },

    { "%Zd",    "xyz",  "0",     0, 0, -555 },

    /* match something, but invalid */
    { "%Zd",    "-",    "-999",  0, 1, -555 },
    { "%Zd",    "+",    "-999",  0, 1, -555 },
    { "xyz%Zd", "xyz-", "-999",  0, 4, -555 },
    { "xyz%Zd", "xyz+", "-999",  0, 4, -555 },
    { "%Zi",    "0x",   "-999",  0, 2, -555 },
    { "%Zi",    "0X",   "-999",  0, 2, -555 },
    { "%Zi",    "0x-",  "-999",  0, 2, -555 },
    { "%Zi",    "0X+",  "-999",  0, 2, -555 },
    { "%Zi",    "-0x",  "-999",  0, 3, -555 },
    { "%Zi",    "-0X",  "-999",  0, 3, -555 },
    { "%Zi",    "+0x",  "-999",  0, 3, -555 },
    { "%Zi",    "+0X",  "-999",  0, 3, -555 },

    { "%1Zi",  "1234", "1",    1, 1, 1 },
    { "%2Zi",  "1234", "12",   1, 2, 2 },
    { "%3Zi",  "1234", "123",  1, 3, 3 },
    { "%4Zi",  "1234", "1234", 1, 4, 4 },
    { "%5Zi",  "1234", "1234", 1, 4, 4 },
    { "%6Zi",  "1234", "1234", 1, 4, 4 },

    { "%1Zi",  "01234", "0",     1, 1, 1 },
    { "%2Zi",  "01234", "01",    1, 2, 2 },
    { "%3Zi",  "01234", "012",   1, 3, 3 },
    { "%4Zi",  "01234", "0123",  1, 4, 4 },
    { "%5Zi",  "01234", "01234", 1, 5, 5 },
    { "%6Zi",  "01234", "01234", 1, 5, 5 },
    { "%7Zi",  "01234", "01234", 1, 5, 5 },

    { "%1Zi",  "0x1234", "0",      1, 1, 1 },
    { "%2Zi",  "0x1234", "-999",   0, 2, -555 },
    { "%3Zi",  "0x1234", "0x1",    1, 3, 3 },
    { "%4Zi",  "0x1234", "0x12",   1, 4, 4 },
    { "%5Zi",  "0x1234", "0x123",  1, 5, 5 },
    { "%6Zi",  "0x1234", "0x1234", 1, 6, 6 },
    { "%7Zi",  "0x1234", "0x1234", 1, 6, 6 },
    { "%8Zi",  "0x1234", "0x1234", 1, 6, 6 },

    { "%%xyz%Zd",  "%xyz123",  "123", 1, -1, -1 },
    { "12%%34%Zd", "12%34567", "567", 1, -1, -1 },
    { "%%%%%Zd",   "%%123",    "123", 1, -1, -1 },

    /* various subtle EOF cases */
    { "x",       "",    "-999", EOF, 0, -555 },
    { " x",      "",    "-999", EOF, 0, -555 },
    { "xyz",     "",    "-999", EOF, 0, -555 },
    { " ",       "",    "-999",   0, 0,    0 },
    { " ",       " ",   "-999",   0, 1,    1 },
    { "%*Zd%Zd", "",    "-999", EOF, 0, -555 },
    { "%*Zd%Zd", "123", "-999", EOF, 3, -555 },
    { "x",       "x",   "-999",   0, 1,    1 },
    { "xyz",     "x",   "-999", EOF, 1, -555 },
    { "xyz",     "xy",  "-999", EOF, 2, -555 },
    { "xyz",     "xyz", "-999",   0, 3,    3 },
    { "%Zn",     "",    "0",      0, 0,    0 },
    { " %Zn",    "",    "0",      0, 0,    0 },
    { " x%Zn",   "",    "-999", EOF, 0, -555 },
    { "xyz%Zn",  "",    "-999", EOF, 0, -555 },
    { " x%Zn",   "",    "-999", EOF, 0, -555 },
    { " %Zn x",  " ",   "-999", EOF, 1, -555 },

    /* these seem to tickle a bug in glibc 2.2.4 */
    { " x",      " ",   "-999", EOF, 1, -555, 1 },
    { " xyz",    " ",   "-999", EOF, 1, -555, 1 },
    { " x%Zn",   " ",   "-999", EOF, 1, -555, 1 },
  };

  int         i, j, ignore;
  int         got_ret, want_ret, got_upto, want_upto;
  mpz_t       got, want;
  long        got_l, want_ftell;
  int         error = 0;
  fun_t       fun;
  const char  *name;
  char        fmt[128];

  mpz_init (got);
  mpz_init (want);

  for (i = 0; i < numberof (data); i++)
    {
      mpz_set_str_or_abort (want, data[i].want, 0);

      ASSERT_ALWAYS (strlen (data[i].fmt) + 2 < sizeof (fmt));
      strcpy (fmt, data[i].fmt);
      strcat (fmt, "%n");

      ignore = fmt_allignore (fmt);

      for (j = 0; j <= 3; j++)
        {
          want_ret = data[i].want_ret;

          want_ftell = data[i].want_ftell;
          if (want_ftell == -1)
            want_ftell = strlen (data[i].input);

          want_upto = data[i].want_upto;
          if (want_upto == -1)
            want_upto = strlen (data[i].input);

          switch (j) {
          case 0:
            name = "gmp_sscanf";
            fun = fun_gmp_sscanf;
            break;
          case 1:
            name = "gmp_fscanf";
            fun = fun_gmp_fscanf;
            break;
          case 2:
#ifdef __GLIBC__
            if (data[i].not_glibc)
              continue;
#endif
            if (! libc_scanf_convert (fmt))
              continue;
            name = "standard sscanf";
            fun = fun_sscanf;
            break;
          case 3:
#ifdef __GLIBC__
            if (data[i].not_glibc)
              continue;
#endif
            if (! libc_scanf_convert (fmt))
              continue;
            name = "standard fscanf";
            fun = fun_fscanf;
            break;
          default:
            ASSERT_ALWAYS (0);
            break;
          }

          got_upto = -555;
          got_ftell = -1L;

          switch (j) {
          case 0:
          case 1:
            mpz_set_si (got, -999L);
            if (ignore)
              got_ret = (*fun) (data[i].input, fmt, &got_upto, NULL);
            else
              got_ret = (*fun) (data[i].input, fmt, got, &got_upto);
            break;
          case 2:
          case 3:
            got_l = -999L;
            if (ignore)
              got_ret = (*fun) (data[i].input, fmt, &got_upto, NULL);
            else
              got_ret = (*fun) (data[i].input, fmt, &got_l, &got_upto);
            mpz_set_si (got, got_l);
            break;
          default:
            ASSERT_ALWAYS (0);
            break;
          }

          MPZ_CHECK_FORMAT (got);

          if (got_ret != want_ret)
            {
              printf ("%s wrong return value\n", name);
              error = 1;
            }
          if (want_ret == 1 && mpz_cmp (want, got) != 0)
            {
              printf ("%s wrong result\n", name);
              error = 1;
            }
          if (got_upto != want_upto)
            {
              printf ("%s wrong upto\n", name);
              error = 1;
            }
          if (got_ftell != -1 && want_ftell != -1 && got_ftell != want_ftell)
            {
              printf ("%s wrong ftell\n", name);
              error = 1;
            }
          if (error)
            {
              printf    ("  fmt   \"%s\"\n", data[i].fmt);
              printf    ("  input \"%s\"\n", data[i].input);
              printf    ("  ignore %d\n", ignore);
              printf    ("  ret   want=%d\n", want_ret);
              printf    ("        got =%d\n", got_ret);
              mpz_trace ("  value want", want);
              mpz_trace ("        got ", got);
              printf    ("  upto  want =%d\n", want_upto);
              printf    ("        got  =%d\n", got_upto);
              if (got_ftell != -1)
                {
                  printf    ("  ftell want =%ld\n", want_ftell);
                  printf    ("        got  =%ld\n", got_ftell);
                }
              abort ();
            }
        }
    }

  mpz_clear (got);
  mpz_clear (want);
}

void
check_q (void)
{
  static const struct {
    const char  *fmt;
    const char  *input;
    const char  *want;
    int         ret;
    long        ftell;

  } data[] = {

    { "%Qd",    "0",    "0", 1, -1 },
    { "%Qd",    "1",    "1", 1, -1 },
    { "%Qd",  "123",  "123", 1, -1 },
    { "%Qd",   "+0",    "0", 1, -1 },
    { "%Qd",   "+1",    "1", 1, -1 },
    { "%Qd", "+123",  "123", 1, -1 },
    { "%Qd",   "-0",    "0", 1, -1 },
    { "%Qd",   "-1",   "-1", 1, -1 },
    { "%Qd", "-123", "-123", 1, -1 },

    { "%Qo",    "0",    "0", 1, -1 },
    { "%Qo",  "173",  "123", 1, -1 },
    { "%Qo",   "+0",    "0", 1, -1 },
    { "%Qo", "+173",  "123", 1, -1 },
    { "%Qo",   "-0",    "0", 1, -1 },
    { "%Qo", "-173", "-123", 1, -1 },

    { "%Qx",    "0",    "0", 1, -1 },
    { "%Qx",   "7b",  "123", 1, -1 },
    { "%Qx",   "7b",  "123", 1, -1 },
    { "%Qx",   "+0",    "0", 1, -1 },
    { "%Qx",  "+7b",  "123", 1, -1 },
    { "%Qx",  "+7b",  "123", 1, -1 },
    { "%Qx",   "-0",   "-0", 1, -1 },
    { "%Qx",  "-7b", "-123", 1, -1 },
    { "%Qx",  "-7b", "-123", 1, -1 },
    { "%QX",    "0",    "0", 1, -1 },
    { "%QX",   "7b",  "123", 1, -1 },
    { "%QX",   "7b",  "123", 1, -1 },
    { "%QX",   "+0",    "0", 1, -1 },
    { "%QX",  "+7b",  "123", 1, -1 },
    { "%QX",  "+7b",  "123", 1, -1 },
    { "%QX",   "-0",   "-0", 1, -1 },
    { "%QX",  "-7b", "-123", 1, -1 },
    { "%QX",  "-7b", "-123", 1, -1 },
    { "%Qx",    "0",    "0", 1, -1 },
    { "%Qx",   "7B",  "123", 1, -1 },
    { "%Qx",   "7B",  "123", 1, -1 },
    { "%Qx",   "+0",    "0", 1, -1 },
    { "%Qx",  "+7B",  "123", 1, -1 },
    { "%Qx",  "+7B",  "123", 1, -1 },
    { "%Qx",   "-0",   "-0", 1, -1 },
    { "%Qx",  "-7B", "-123", 1, -1 },
    { "%Qx",  "-7B", "-123", 1, -1 },
    { "%QX",    "0",    "0", 1, -1 },
    { "%QX",   "7B",  "123", 1, -1 },
    { "%QX",   "7B",  "123", 1, -1 },
    { "%QX",   "+0",    "0", 1, -1 },
    { "%QX",  "+7B",  "123", 1, -1 },
    { "%QX",  "+7B",  "123", 1, -1 },
    { "%QX",   "-0",   "-0", 1, -1 },
    { "%QX",  "-7B", "-123", 1, -1 },
    { "%QX",  "-7B", "-123", 1, -1 },

    { "%Qi",    "0",    "0", 1, -1 },
    { "%Qi",    "1",    "1", 1, -1 },
    { "%Qi",  "123",  "123", 1, -1 },
    { "%Qi",   "+0",    "0", 1, -1 },
    { "%Qi",   "+1",    "1", 1, -1 },
    { "%Qi", "+123",  "123", 1, -1 },
    { "%Qi",   "-0",    "0", 1, -1 },
    { "%Qi",   "-1",   "-1", 1, -1 },
    { "%Qi", "-123", "-123", 1, -1 },

    { "%Qi",    "00",    "0", 1, -1 },
    { "%Qi",  "0173",  "123", 1, -1 },
    { "%Qi",   "+00",    "0", 1, -1 },
    { "%Qi", "+0173",  "123", 1, -1 },
    { "%Qi",   "-00",    "0", 1, -1 },
    { "%Qi", "-0173", "-123", 1, -1 },

    { "%Qi",    "0x0",    "0", 1, -1 },
    { "%Qi",   "0x7b",  "123", 1, -1 },
    { "%Qi",   "0x7b",  "123", 1, -1 },
    { "%Qi",   "+0x0",    "0", 1, -1 },
    { "%Qi",  "+0x7b",  "123", 1, -1 },
    { "%Qi",  "+0x7b",  "123", 1, -1 },
    { "%Qi",   "-0x0",   "-0", 1, -1 },
    { "%Qi",  "-0x7b", "-123", 1, -1 },
    { "%Qi",  "-0x7b", "-123", 1, -1 },
    { "%Qi",    "0X0",    "0", 1, -1 },
    { "%Qi",   "0X7b",  "123", 1, -1 },
    { "%Qi",   "0X7b",  "123", 1, -1 },
    { "%Qi",   "+0X0",    "0", 1, -1 },
    { "%Qi",  "+0X7b",  "123", 1, -1 },
    { "%Qi",  "+0X7b",  "123", 1, -1 },
    { "%Qi",   "-0X0",   "-0", 1, -1 },
    { "%Qi",  "-0X7b", "-123", 1, -1 },
    { "%Qi",  "-0X7b", "-123", 1, -1 },
    { "%Qi",    "0x0",    "0", 1, -1 },
    { "%Qi",   "0x7B",  "123", 1, -1 },
    { "%Qi",   "0x7B",  "123", 1, -1 },
    { "%Qi",   "+0x0",    "0", 1, -1 },
    { "%Qi",  "+0x7B",  "123", 1, -1 },
    { "%Qi",  "+0x7B",  "123", 1, -1 },
    { "%Qi",   "-0x0",   "-0", 1, -1 },
    { "%Qi",  "-0x7B", "-123", 1, -1 },
    { "%Qi",  "-0x7B", "-123", 1, -1 },
    { "%Qi",    "0X0",    "0", 1, -1 },
    { "%Qi",   "0X7B",  "123", 1, -1 },
    { "%Qi",   "0X7B",  "123", 1, -1 },
    { "%Qi",   "+0X0",    "0", 1, -1 },
    { "%Qi",  "+0X7B",  "123", 1, -1 },
    { "%Qi",  "+0X7B",  "123", 1, -1 },
    { "%Qi",   "-0X0",   "-0", 1, -1 },
    { "%Qi",  "-0X7B", "-123", 1, -1 },
    { "%Qi",  "-0X7B", "-123", 1, -1 },

    { "%Qd",    " 0",    "0", 1, -1 },
    { "%Qd",   "  0",    "0", 1, -1 },
    { "%Qd",  "   0",    "0", 1, -1 },
    { "%Qd",   "\t0",    "0", 1, -1 },
    { "%Qd", "\t\t0",    "0", 1, -1 },

    { "%Qd",  "3/2",   "3/2", 1, -1 },
    { "%Qd", "+3/2",   "3/2", 1, -1 },
    { "%Qd", "-3/2",  "-3/2", 1, -1 },

    { "%Qx",  "f/10", "15/16", 1, -1 },
    { "%Qx",  "F/10", "15/16", 1, -1 },
    { "%QX",  "f/10", "15/16", 1, -1 },
    { "%QX",  "F/10", "15/16", 1, -1 },

    { "%Qo",  "20/21",  "16/17", 1, -1 },
    { "%Qo", "-20/21", "-16/17", 1, -1 },

    { "%Qi",    "10/11",  "10/11", 1, -1 },
    { "%Qi",   "+10/11",  "10/11", 1, -1 },
    { "%Qi",   "-10/11", "-10/11", 1, -1 },
    { "%Qi",   "010/11",   "8/11", 1, -1 },
    { "%Qi",  "+010/11",   "8/11", 1, -1 },
    { "%Qi",  "-010/11",  "-8/11", 1, -1 },
    { "%Qi",  "0x10/11",  "16/11", 1, -1 },
    { "%Qi", "+0x10/11",  "16/11", 1, -1 },
    { "%Qi", "-0x10/11", "-16/11", 1, -1 },

    { "%Qi",    "10/011",  "10/9", 1, -1 },
    { "%Qi",   "+10/011",  "10/9", 1, -1 },
    { "%Qi",   "-10/011", "-10/9", 1, -1 },
    { "%Qi",   "010/011",   "8/9", 1, -1 },
    { "%Qi",  "+010/011",   "8/9", 1, -1 },
    { "%Qi",  "-010/011",  "-8/9", 1, -1 },
    { "%Qi",  "0x10/011",  "16/9", 1, -1 },
    { "%Qi", "+0x10/011",  "16/9", 1, -1 },
    { "%Qi", "-0x10/011", "-16/9", 1, -1 },

    { "%Qi",    "10/0x11",  "10/17", 1, -1 },
    { "%Qi",   "+10/0x11",  "10/17", 1, -1 },
    { "%Qi",   "-10/0x11", "-10/17", 1, -1 },
    { "%Qi",   "010/0x11",   "8/17", 1, -1 },
    { "%Qi",  "+010/0x11",   "8/17", 1, -1 },
    { "%Qi",  "-010/0x11",  "-8/17", 1, -1 },
    { "%Qi",  "0x10/0x11",  "16/17", 1, -1 },
    { "%Qi", "+0x10/0x11",  "16/17", 1, -1 },
    { "%Qi", "-0x10/0x11", "-16/17", 1, -1 },

    { "hello%Qd",      "hello0",         "0", 1, -1 },
    { "hello%Qd",      "hello 0",        "0", 1, -1 },
    { "hello%Qd",      "hello \t0",      "0", 1, -1 },
    { "hello%Qdworld", "hello 0world",   "0", 1, -1 },
    { "hello%Qd",      "hello3/2",     "3/2", 1, -1 },

    { "hello%*Qd",      "hello0",        "-999/121", 0, -1 },
    { "hello%*Qd",      "hello 0",       "-999/121", 0, -1 },
    { "hello%*Qd",      "hello \t0",     "-999/121", 0, -1 },
    { "hello%*Qdworld", "hello 0world",  "-999/121", 0, -1 },
    { "hello%*Qdworld", "hello3/2world", "-999/121", 0, -1 },

    { "%Qd",    "",     "-999/121", -1, -1 },
    { "%Qd",   " ",     "-999/121", -1, -1 },
    { " %Qd",   "",     "-999/121", -1, -1 },
    { "xyz%Qd", "",     "-999/121", -1, -1 },

    { "%*Qd",    "",     "-999/121", -1, -1 },
    { " %*Qd",   "",     "-999/121", -1, -1 },
    { "xyz%*Qd", "",     "-999/121", -1, -1 },

    /* match something, but invalid */
    { "%Qd",    "-",     "-999/121",  0, 1 },
    { "%Qd",    "+",     "-999/121",  0, 1 },
    { "%Qd",    "/-",    "-999/121",  0, 1 },
    { "%Qd",    "/+",    "-999/121",  0, 1 },
    { "%Qd",    "-/",    "-999/121",  0, 1 },
    { "%Qd",    "+/",    "-999/121",  0, 1 },
    { "%Qd",    "-/-",   "-999/121",  0, 1 },
    { "%Qd",    "-/+",   "-999/121",  0, 1 },
    { "%Qd",    "+/+",   "-999/121",  0, 1 },
    { "%Qd",    "/123",  "-999/121",  0, 1 },
    { "%Qd",    "-/123", "-999/121",  0, 1 },
    { "%Qd",    "+/123", "-999/121",  0, 1 },
    { "%Qd",    "123/",  "-999/121",  0, 1 },
    { "%Qd",    "123/-", "-999/121",  0, 1 },
    { "%Qd",    "123/+", "-999/121",  0, 1 },
    { "xyz%Qd", "xyz-",  "-999/121",  0, 4 },
    { "xyz%Qd", "xyz+",  "-999/121",  0, 4 },

    { "%1Qi",  "12/57", "1",        1, 1 },
    { "%2Qi",  "12/57", "12",       1, 2 },
    { "%3Qi",  "12/57", "-999/121", 0, -1 },
    { "%4Qi",  "12/57", "12/5",     1, 4 },
    { "%5Qi",  "12/57", "12/57",    1, 5 },
    { "%6Qi",  "12/57", "12/57",    1, 5 },
    { "%7Qi",  "12/57", "12/57",    1, 5 },

    { "%1Qi",  "012/057", "0",        1, 1 },
    { "%2Qi",  "012/057", "01",       1, 2 },
    { "%3Qi",  "012/057", "012",      1, 3 },
    { "%4Qi",  "012/057", "-999/121", 0, -1 },
    { "%5Qi",  "012/057", "012/0",    1, 5 },
    { "%6Qi",  "012/057", "012/5",    1, 6 },
    { "%7Qi",  "012/057", "012/057",  1, 7 },
    { "%8Qi",  "012/057", "012/057",  1, 7 },
    { "%9Qi",  "012/057", "012/057",  1, 7 },

    { "%1Qi",  "0x12/0x57", "0",         1, 1 },
    { "%2Qi",  "0x12/0x57", "-999",      0, 2 },
    { "%3Qi",  "0x12/0x57", "0x1",       1, 3 },
    { "%4Qi",  "0x12/0x57", "0x12",      1, 4 },
    { "%5Qi",  "0x12/0x57", "-999/121",  0, 5 },
    { "%6Qi",  "0x12/0x57", "0x12/0",    1, 6 },
    { "%7Qi",  "0x12/0x57", "-999/121",  0, 7 },
    { "%8Qi",  "0x12/0x57", "0x12/0x5",  1, 8 },
    { "%9Qi",  "0x12/0x57", "0x12/0x57", 1, 9 },
    { "%10Qi", "0x12/0x57", "0x12/0x57", 1, 9 },
    { "%11Qi", "0x12/0x57", "0x12/0x57", 1, 9 },

    { "%Qd",  "xyz", "0", 0, 0 },
  };

  int         i, j, ignore, got_ret, want_ret, got_upto, want_upto;
  mpq_t       got, want;
  long        got_l, want_ftell;
  int         error = 0;
  fun_t       fun;
  const char  *name;
  char        fmt[128];

  mpq_init (got);
  mpq_init (want);

  for (i = 0; i < numberof (data); i++)
    {
      mpq_set_str_or_abort (want, data[i].want, 0);

      ASSERT_ALWAYS (strlen (data[i].fmt) + 2 < sizeof (fmt));
      strcpy (fmt, data[i].fmt);
      strcat (fmt, "%n");

      ignore = (strchr (fmt, '*') != NULL);

      for (j = 0; j <= 3; j++)
        {
          want_ret = data[i].ret;

          want_ftell = data[i].ftell;
          if (want_ftell == -1)
            want_ftell = strlen (data[i].input);
          want_upto = want_ftell;

          if (want_ret == -1 || (want_ret == 0 && ! ignore))
            {
              want_ftell = -1;
              want_upto = -555;
            }

          switch (j) {
          case 0:
            name = "gmp_sscanf";
            fun = fun_gmp_sscanf;
            break;
          case 1:
            name = "gmp_fscanf";
            fun = fun_gmp_fscanf;
            break;
          case 2:
            if (strchr (data[i].input, '/') != NULL)
              continue;
            if (! libc_scanf_convert (fmt))
              continue;
            name = "standard sscanf";
            fun = fun_sscanf;
            break;
          case 3:
            if (strchr (data[i].input, '/') != NULL)
              continue;
            if (! libc_scanf_convert (fmt))
              continue;
            name = "standard fscanf";
            fun = fun_fscanf;
            break;
          default:
            ASSERT_ALWAYS (0);
            break;
          }

          got_upto = -555;
          got_ftell = -1;

          switch (j) {
          case 0:
          case 1:
            mpq_set_si (got, -999L, 121L);
            if (ignore)
              got_ret = (*fun) (data[i].input, fmt, &got_upto, NULL);
            else
              got_ret = (*fun) (data[i].input, fmt, got, &got_upto);
            break;
          case 2:
          case 3:
            got_l = -999L;
            if (ignore)
              got_ret = (*fun) (data[i].input, fmt, &got_upto, NULL);
            else
              got_ret = (*fun) (data[i].input, fmt, &got_l, &got_upto);
            mpq_set_si (got, got_l, (got_l == -999L ? 121L : 1L));
            break;
          default:
            ASSERT_ALWAYS (0);
            break;
          }

          MPZ_CHECK_FORMAT (mpq_numref (got));
          MPZ_CHECK_FORMAT (mpq_denref (got));

          if (got_ret != want_ret)
            {
              printf ("%s wrong return value\n", name);
              error = 1;
            }
          /* use direct mpz compares, since some of the test data is
             non-canonical and can trip ASSERTs in mpq_equal */
          if (want_ret == 1
              && ! (mpz_cmp (mpq_numref(want), mpq_numref(got)) == 0
                    && mpz_cmp (mpq_denref(want), mpq_denref(got)) == 0))
            {
              printf ("%s wrong result\n", name);
              error = 1;
            }
          if (got_upto != want_upto)
            {
              printf ("%s wrong upto\n", name);
              error = 1;
            }
          if (got_ftell != -1 && want_ftell != -1 && got_ftell != want_ftell)
            {
              printf ("%s wrong ftell\n", name);
              error = 1;
            }
          if (error)
            {
              printf    ("  fmt   \"%s\"\n", data[i].fmt);
              printf    ("  input \"%s\"\n", data[i].input);
              printf    ("  ret   want=%d\n", want_ret);
              printf    ("        got =%d\n", got_ret);
              mpq_trace ("  value want", want);
              mpq_trace ("        got ", got);
              printf    ("  upto  want=%d\n", want_upto);
              printf    ("        got =%d\n", got_upto);
              if (got_ftell != -1)
                {
                  printf    ("  ftell want =%ld\n", want_ftell);
                  printf    ("        got  =%ld\n", got_ftell);
                }
              abort ();
            }
        }
    }

  mpq_clear (got);
  mpq_clear (want);
}

void
check_f (void)
{
  static const struct {
    const char  *fmt;
    const char  *input;
    const char  *want;
    int         ret;
    long        ftell;    /* or -1 for length of input string */

  } data[] = {

    { "%Ff",    "0",    "0", 1, -1 },
    { "%Fe",    "0",    "0", 1, -1 },
    { "%FE",    "0",    "0", 1, -1 },
    { "%Fg",    "0",    "0", 1, -1 },
    { "%FG",    "0",    "0", 1, -1 },

    { "%Ff",  "123",    "123", 1, -1 },
    { "%Ff", "+123",    "123", 1, -1 },
    { "%Ff", "-123",   "-123", 1, -1 },
    { "%Ff",  "123.",   "123", 1, -1 },
    { "%Ff", "+123.",   "123", 1, -1 },
    { "%Ff", "-123.",  "-123", 1, -1 },
    { "%Ff",  "123.0",  "123", 1, -1 },
    { "%Ff", "+123.0",  "123", 1, -1 },
    { "%Ff", "-123.0", "-123", 1, -1 },
    { "%Ff",  "0123",   "123", 1, -1 },
    { "%Ff", "-0123",  "-123", 1, -1 },

    { "%Ff",  "123.456e3",   "123456", 1, -1 },
    { "%Ff", "-123.456e3",  "-123456", 1, -1 },
    { "%Ff",  "123.456e+3",  "123456", 1, -1 },
    { "%Ff", "-123.456e+3", "-123456", 1, -1 },
    { "%Ff",  "123000e-3",      "123", 1, -1 },
    { "%Ff", "-123000e-3",     "-123", 1, -1 },
    { "%Ff",  "123000.e-3",     "123", 1, -1 },
    { "%Ff", "-123000.e-3",    "-123", 1, -1 },

    { "%Ff",  "123.456E3",   "123456", 1, -1 },
    { "%Ff", "-123.456E3",  "-123456", 1, -1 },
    { "%Ff",  "123.456E+3",  "123456", 1, -1 },
    { "%Ff", "-123.456E+3", "-123456", 1, -1 },
    { "%Ff",  "123000E-3",      "123", 1, -1 },
    { "%Ff", "-123000E-3",     "-123", 1, -1 },
    { "%Ff",  "123000.E-3",     "123", 1, -1 },
    { "%Ff", "-123000.E-3",    "-123", 1, -1 },

    { "%Ff",  ".456e3",   "456", 1, -1 },
    { "%Ff", "-.456e3",  "-456", 1, -1 },
    { "%Ff",  ".456e+3",  "456", 1, -1 },
    { "%Ff", "-.456e+3", "-456", 1, -1 },

    { "%Ff",    " 0",    "0", 1, -1 },
    { "%Ff",   "  0",    "0", 1, -1 },
    { "%Ff",  "   0",    "0", 1, -1 },
    { "%Ff",   "\t0",    "0", 1, -1 },
    { "%Ff", "\t\t0",    "0", 1, -1 },

    { "hello%Fg",      "hello0",       "0",   1, -1 },
    { "hello%Fg",      "hello 0",      "0",   1, -1 },
    { "hello%Fg",      "hello \t0",    "0",   1, -1 },
    { "hello%Fgworld", "hello 0world", "0",   1, -1 },
    { "hello%Fg",      "hello3.0",     "3.0", 1, -1 },

    { "hello%*Fg",      "hello0",        "-999", 0, -1 },
    { "hello%*Fg",      "hello 0",       "-999", 0, -1 },
    { "hello%*Fg",      "hello \t0",     "-999", 0, -1 },
    { "hello%*Fgworld", "hello 0world",  "-999", 0, -1 },
    { "hello%*Fgworld", "hello3.0world", "-999", 0, -1 },

    { "%Ff",     "",   "-999", -1, -1 },
    { "%Ff",    " ",   "-999", -1, -1 },
    { "%Ff",   "\t",   "-999", -1, -1 },
    { "%Ff",  " \t",   "-999", -1, -1 },
    { " %Ff",    "",   "-999", -1, -1 },
    { "xyz%Ff",  "",   "-999", -1, -1 },

    { "%*Ff",    "",   "-999", -1, -1 },
    { " %*Ff",   "",   "-999", -1, -1 },
    { "xyz%*Ff", "",   "-999", -1, -1 },

    { "%Ff",    "xyz", "0", 0 },

    /* various non-empty but invalid */
    { "%Ff",    "-",      "-999",  0, 1 },
    { "%Ff",    "+",      "-999",  0, 1 },
    { "xyz%Ff", "xyz-",   "-999",  0, 4 },
    { "xyz%Ff", "xyz+",   "-999",  0, 4 },
    { "%Ff",    "-.",     "-999",  0, 2 },
    { "%Ff",    "+.",     "-999",  0, 2 },
    { "%Ff",    ".e",     "-999",  0, 1 },
    { "%Ff",   "-.e",     "-999",  0, 2 },
    { "%Ff",   "+.e",     "-999",  0, 2 },
    { "%Ff",    ".E",     "-999",  0, 1 },
    { "%Ff",   "-.E",     "-999",  0, 2 },
    { "%Ff",   "+.E",     "-999",  0, 2 },
    { "%Ff",    ".e123",  "-999",  0, 1 },
    { "%Ff",   "-.e123",  "-999",  0, 2 },
    { "%Ff",   "+.e123",  "-999",  0, 2 },
    { "%Ff",    "123e",   "-999",  0, 4 },
    { "%Ff",   "-123e",   "-999",  0, 5 },
    { "%Ff",    "123e-",  "-999",  0, 5 },
    { "%Ff",   "-123e-",  "-999",  0, 6 },
    { "%Ff",    "123e+",  "-999",  0, 5 },
    { "%Ff",   "-123e+",  "-999",  0, 6 },
    { "%Ff",   "123e-Z",  "-999",  0, 5 },

    /* hex floats */
    { "%Ff", "0x123p0",       "291",  1, -1 },
    { "%Ff", "0x123P0",       "291",  1, -1 },
    { "%Ff", "0X123p0",       "291",  1, -1 },
    { "%Ff", "0X123P0",       "291",  1, -1 },
    { "%Ff", "-0x123p0",     "-291",  1, -1 },
    { "%Ff", "+0x123p0",      "291",  1, -1 },
    { "%Ff", "0x123.p0",      "291",  1, -1 },
    { "%Ff", "0x12.3p4",      "291",  1, -1 },
    { "%Ff", "-0x12.3p4",    "-291",  1, -1 },
    { "%Ff", "+0x12.3p4",     "291",  1, -1 },
    { "%Ff", "0x1230p-4",     "291",  1, -1 },
    { "%Ff", "-0x1230p-4",   "-291",  1, -1 },
    { "%Ff", "+0x1230p-4",    "291",  1, -1 },
    { "%Ff", "+0x.1230p12",   "291",  1, -1 },
    { "%Ff", "+0x123000p-12", "291",  1, -1 },
    { "%Ff", "0x123 p12",     "291",  1, 5 },
    { "%Ff", "0x9 9",           "9",  1, 3 },
    { "%Ff", "0x01",            "1",  1, 4 },
    { "%Ff", "0x23",           "35",  1, 4 },
    { "%Ff", "0x45",           "69",  1, 4 },
    { "%Ff", "0x67",          "103",  1, 4 },
    { "%Ff", "0x89",          "137",  1, 4 },
    { "%Ff", "0xAB",          "171",  1, 4 },
    { "%Ff", "0xCD",          "205",  1, 4 },
    { "%Ff", "0xEF",          "239",  1, 4 },
    { "%Ff", "0xab",          "171",  1, 4 },
    { "%Ff", "0xcd",          "205",  1, 4 },
    { "%Ff", "0xef",          "239",  1, 4 },
    { "%Ff", "0x100p0A",      "256",  1, 7 },
    { "%Ff", "0x1p9",         "512",  1, -1 },

    /* invalid hex floats */
    { "%Ff", "0x",     "-999",  0, 2 },
    { "%Ff", "-0x",    "-999",  0, 3 },
    { "%Ff", "+0x",    "-999",  0, 3 },
    { "%Ff", "0x-",    "-999",  0, 2 },
    { "%Ff", "0x+",    "-999",  0, 2 },
    { "%Ff", "0x.",    "-999",  0, 3 },
    { "%Ff", "-0x.",   "-999",  0, 4 },
    { "%Ff", "+0x.",   "-999",  0, 4 },
    { "%Ff", "0x.p",   "-999",  0, 3 },
    { "%Ff", "-0x.p",  "-999",  0, 4 },
    { "%Ff", "+0x.p",  "-999",  0, 4 },
    { "%Ff", "0x.P",   "-999",  0, 3 },
    { "%Ff", "-0x.P",  "-999",  0, 4 },
    { "%Ff", "+0x.P",  "-999",  0, 4 },
    { "%Ff", ".p123",  "-999",  0, 1 },
    { "%Ff", "-.p123", "-999",  0, 2 },
    { "%Ff", "+.p123", "-999",  0, 2 },
    { "%Ff", "0x1p",   "-999",  0, 4 },
    { "%Ff", "0x1p-",  "-999",  0, 5 },
    { "%Ff", "0x1p+",  "-999",  0, 5 },
    { "%Ff", "0x123p 12", "291",  0, 6 },
    { "%Ff", "0x 123p12", "291",  0, 2 },

  };

  int         i, j, ignore, got_ret, want_ret, got_upto, want_upto;
  mpf_t       got, want;
  double      got_d;
  long        want_ftell;
  int         error = 0;
  fun_t       fun;
  const char  *name;
  char        fmt[128];

  mpf_init (got);
  mpf_init (want);

  for (i = 0; i < numberof (data); i++)
    {
      mpf_set_str_or_abort (want, data[i].want, 10);

      ASSERT_ALWAYS (strlen (data[i].fmt) + 2 < sizeof (fmt));
      strcpy (fmt, data[i].fmt);
      strcat (fmt, "%n");

      ignore = (strchr (fmt, '*') != NULL);

      for (j = 0; j <= 3; j++)
        {
          want_ret = data[i].ret;

          want_ftell = data[i].ftell;
          if (want_ftell == -1)
            want_ftell = strlen (data[i].input);
          want_upto = want_ftell;

          if (want_ret == -1 || (want_ret == 0 && ! ignore))
            want_upto = -555;

          switch (j) {
          case 0:
            name = "gmp_sscanf";
            fun = fun_gmp_sscanf;
            break;
          case 1:
            name = "gmp_fscanf";
            fun = fun_gmp_fscanf;
            break;
          case 2:
            if (! libc_scanf_convert (fmt))
              continue;
            name = "standard sscanf";
            fun = fun_sscanf;
            break;
          case 3:
            if (! libc_scanf_convert (fmt))
              continue;
            name = "standard fscanf";
            fun = fun_fscanf;
            break;
          default:
            ASSERT_ALWAYS (0);
            break;
          }

          got_upto = -555;
          got_ftell = -1;

          switch (j) {
          case 0:
          case 1:
            mpf_set_si (got, -999L);
            if (ignore)
              got_ret = (*fun) (data[i].input, fmt, &got_upto, NULL);
            else
              got_ret = (*fun) (data[i].input, fmt, got, &got_upto);
            break;
          case 2:
          case 3:
            got_d = -999L;
            if (ignore)
              got_ret = (*fun) (data[i].input, fmt, &got_upto, NULL);
            else
              got_ret = (*fun) (data[i].input, fmt, &got_d, &got_upto);
            mpf_set_d (got, got_d);
            break;
          default:
            ASSERT_ALWAYS (0);
            break;
          }

          MPF_CHECK_FORMAT (got);

          if (got_ret != want_ret)
            {
              printf ("%s wrong return value\n", name);
              error = 1;
            }
          if (want_ret == 1 && mpf_cmp (want, got) != 0)
            {
              printf ("%s wrong result\n", name);
              error = 1;
            }
          if (got_upto != want_upto)
            {
              printf ("%s wrong upto\n", name);
              error = 1;
            }
          if (got_ftell != -1 && want_ftell != -1 && got_ftell != want_ftell)
            {
              printf ("%s wrong ftell\n", name);
              error = 1;
            }
          if (error)
            {
              printf    ("  fmt   \"%s\"\n", data[i].fmt);
              printf    ("  input \"%s\"\n", data[i].input);
              printf    ("  ret   want=%d\n", want_ret);
              printf    ("        got =%d\n", got_ret);
              mpf_trace ("  value want", want);
              mpf_trace ("        got ", got);
              printf    ("  upto  want=%d\n", want_upto);
              printf    ("        got =%d\n", got_upto);
              if (got_ftell != -1)
                {
                  printf    ("  ftell want =%ld\n", want_ftell);
                  printf    ("        got  =%ld\n", got_ftell);
                }
              abort ();
            }
        }
    }

  mpf_clear (got);
  mpf_clear (want);
}


void
check_n (void)
{
  int    ret;

  /* %n suppressed */
  {
    int n = 123;
    gmp_sscanf ("   ", " %*n", &n);
    ASSERT_ALWAYS (n == 123);
  }
  {
    int n = 123;
    fromstring_gmp_fscanf ("   ", " %*n", &n);
    ASSERT_ALWAYS (n == 123);
  }


#define CHECK_N(type, string)                           \
  do {                                                  \
    type  x[2];                                         \
    char  fmt[128];                                     \
    int   ret;                                          \
                                                        \
    x[0] = ~ (type) 0;                                  \
    x[1] = ~ (type) 0;                                  \
    sprintf (fmt, "abc%%%sn", string);                  \
    ret = gmp_sscanf ("abc", fmt, &x[0]);               \
                                                        \
    ASSERT_ALWAYS (ret == 0);                           \
                                                        \
    /* should write whole of x[0] and none of x[1] */   \
    ASSERT_ALWAYS (x[0] == 3);                          \
    ASSERT_ALWAYS (x[1] == (type) ~ (type) 0);		\
                                                        \
  } while (0)

  CHECK_N (char,      "hh");
  CHECK_N (long,      "l");
#if HAVE_LONG_LONG
  CHECK_N (long long, "L");
#endif
#if HAVE_INTMAX_T
  CHECK_N (intmax_t,  "j");
#endif
#if HAVE_PTRDIFF_T
  CHECK_N (ptrdiff_t, "t");
#endif
  CHECK_N (short,     "h");
  CHECK_N (size_t,    "z");

  /* %Zn */
  {
    mpz_t  x[2];
    mpz_init_set_si (x[0], -987L);
    mpz_init_set_si (x[1],  654L);
    ret = gmp_sscanf ("xyz   ", "xyz%Zn", x[0]);
    MPZ_CHECK_FORMAT (x[0]);
    MPZ_CHECK_FORMAT (x[1]);
    ASSERT_ALWAYS (ret == 0);
    ASSERT_ALWAYS (mpz_cmp_ui (x[0], 3L) == 0);
    ASSERT_ALWAYS (mpz_cmp_ui (x[1], 654L) == 0);
    mpz_clear (x[0]);
    mpz_clear (x[1]);
  }
  {
    mpz_t  x;
    mpz_init (x);
    ret = fromstring_gmp_fscanf ("xyz   ", "xyz%Zn", x);
    ASSERT_ALWAYS (ret == 0);
    ASSERT_ALWAYS (mpz_cmp_ui (x, 3L) == 0);
    mpz_clear (x);
  }

  /* %Qn */
  {
    mpq_t  x[2];
    mpq_init (x[0]);
    mpq_init (x[1]);
    mpq_set_ui (x[0], 987L, 654L);
    mpq_set_ui (x[1], 4115L, 226L);
    ret = gmp_sscanf ("xyz   ", "xyz%Qn", x[0]);
    MPQ_CHECK_FORMAT (x[0]);
    MPQ_CHECK_FORMAT (x[1]);
    ASSERT_ALWAYS (ret == 0);
    ASSERT_ALWAYS (mpq_cmp_ui (x[0], 3L, 1L) == 0);
    ASSERT_ALWAYS (mpq_cmp_ui (x[1], 4115L, 226L) == 0);
    mpq_clear (x[0]);
    mpq_clear (x[1]);
  }
  {
    mpq_t  x;
    mpq_init (x);
    ret = fromstring_gmp_fscanf ("xyz   ", "xyz%Qn", x);
    ASSERT_ALWAYS (ret == 0);
    ASSERT_ALWAYS (mpq_cmp_ui (x, 3L, 1L) == 0);
    mpq_clear (x);
  }

  /* %Fn */
  {
    mpf_t  x[2];
    mpf_init (x[0]);
    mpf_init (x[1]);
    mpf_set_ui (x[0], 987L);
    mpf_set_ui (x[1], 654L);
    ret = gmp_sscanf ("xyz   ", "xyz%Fn", x[0]);
    MPF_CHECK_FORMAT (x[0]);
    MPF_CHECK_FORMAT (x[1]);
    ASSERT_ALWAYS (ret == 0);
    ASSERT_ALWAYS (mpf_cmp_ui (x[0], 3L) == 0);
    ASSERT_ALWAYS (mpf_cmp_ui (x[1], 654L) == 0);
    mpf_clear (x[0]);
    mpf_clear (x[1]);
  }
  {
    mpf_t  x;
    mpf_init (x);
    ret = fromstring_gmp_fscanf ("xyz   ", "xyz%Fn", x);
    ASSERT_ALWAYS (ret == 0);
    ASSERT_ALWAYS (mpf_cmp_ui (x, 3L) == 0);
    mpf_clear (x);
  }
}


void
check_misc (void)
{
  int  ret, cmp;
  {
    int  a=9, b=8, c=7, n=66;
    mpz_t  z;
    mpz_init (z);
    ret = gmp_sscanf ("1 2 3 4", "%d %d %d %Zd%n",
                      &a, &b, &c, z, &n);
    ASSERT_ALWAYS (ret == 4);
    ASSERT_ALWAYS (a == 1);
    ASSERT_ALWAYS (b == 2);
    ASSERT_ALWAYS (c == 3);
    ASSERT_ALWAYS (n == 7);
    ASSERT_ALWAYS (mpz_cmp_ui (z, 4L) == 0);
    mpz_clear (z);
  }
  {
    int  a=9, b=8, c=7, n=66;
    mpz_t  z;
    mpz_init (z);
    ret = fromstring_gmp_fscanf ("1 2 3 4", "%d %d %d %Zd%n",
                                 &a, &b, &c, z, &n);
    ASSERT_ALWAYS (ret == 4);
    ASSERT_ALWAYS (a == 1);
    ASSERT_ALWAYS (b == 2);
    ASSERT_ALWAYS (c == 3);
    ASSERT_ALWAYS (mpz_cmp_ui (z, 4L) == 0);
    ASSERT_ALWAYS (n == 7);
    ASSERT_ALWAYS (got_ftell == 7);
    mpz_clear (z);
  }

  {
    int  a=9, n=8;
    mpz_t  z;
    mpz_init (z);
    ret = gmp_sscanf ("1 2 3 4", "%d %*d %*d %Zd%n", &a, z, &n);
    ASSERT_ALWAYS (ret == 2);
    ASSERT_ALWAYS (a == 1);
    ASSERT_ALWAYS (mpz_cmp_ui (z, 4L) == 0);
    ASSERT_ALWAYS (n == 7);
    mpz_clear (z);
  }
  {
    int  a=9, n=8;
    mpz_t  z;
    mpz_init (z);
    ret = fromstring_gmp_fscanf ("1 2 3 4", "%d %*d %*d %Zd%n",
                                 &a, z, &n);
    ASSERT_ALWAYS (ret == 2);
    ASSERT_ALWAYS (a == 1);
    ASSERT_ALWAYS (mpz_cmp_ui (z, 4L) == 0);
    ASSERT_ALWAYS (n == 7);
    ASSERT_ALWAYS (got_ftell == 7);
    mpz_clear (z);
  }

  /* EOF for no matching */
  {
    char buf[128];
    ret = gmp_sscanf ("   ", "%s", buf);
    ASSERT_ALWAYS (ret == EOF);
    ret = fromstring_gmp_fscanf ("   ", "%s", buf);
    ASSERT_ALWAYS (ret == EOF);
    if (option_libc_scanf)
      {
        ret = sscanf ("   ", "%s", buf);
        ASSERT_ALWAYS (ret == EOF);
        ret = fun_fscanf ("   ", "%s", buf, NULL);
        ASSERT_ALWAYS (ret == EOF);
      }
  }

  /* suppressed field, then eof */
  {
    int  x;
    if (test_sscanf_eof_ok ())
      {
        ret = gmp_sscanf ("123", "%*d%d", &x);
        ASSERT_ALWAYS (ret == EOF);
      }
    ret = fromstring_gmp_fscanf ("123", "%*d%d", &x);
    ASSERT_ALWAYS (ret == EOF);
    if (option_libc_scanf)
      {
        ret = sscanf ("123", "%*d%d", &x);
        ASSERT_ALWAYS (ret == EOF);
        ret = fun_fscanf ("123", "%*d%d", &x, NULL);
        ASSERT_ALWAYS (ret == EOF);
      }
  }
  {
    mpz_t  x;
    mpz_init (x);
    ret = gmp_sscanf ("123", "%*Zd%Zd", x);
    ASSERT_ALWAYS (ret == EOF);
    ret = fromstring_gmp_fscanf ("123", "%*Zd%Zd", x);
    ASSERT_ALWAYS (ret == EOF);
    mpz_clear (x);
  }

  /* %[...], glibc only */
#ifdef __GLIBC__
  {
    char  buf[128];
    int   n = -1;
    buf[0] = '\0';
    ret = gmp_sscanf ("abcdefgh", "%[a-d]ef%n", buf, &n);
    ASSERT_ALWAYS (ret == 1);
    cmp = strcmp (buf, "abcd");
    ASSERT_ALWAYS (cmp == 0);
    ASSERT_ALWAYS (n == 6);
  }
  {
    char  buf[128];
    int   n = -1;
    buf[0] = '\0';
    ret = gmp_sscanf ("xyza", "%[^a]a%n", buf, &n);
    ASSERT_ALWAYS (ret == 1);
    cmp = strcmp (buf, "xyz");
    ASSERT_ALWAYS (cmp == 0);
    ASSERT_ALWAYS (n == 4);
  }
  {
    char  buf[128];
    int   n = -1;
    buf[0] = '\0';
    ret = gmp_sscanf ("ab]ab]", "%[]ab]%n", buf, &n);
    ASSERT_ALWAYS (ret == 1);
    cmp = strcmp (buf, "ab]ab]");
    ASSERT_ALWAYS (cmp == 0);
    ASSERT_ALWAYS (n == 6);
  }
  {
    char  buf[128];
    int   n = -1;
    buf[0] = '\0';
    ret = gmp_sscanf ("xyzb", "%[^]ab]b%n", buf, &n);
    ASSERT_ALWAYS (ret == 1);
    cmp = strcmp (buf, "xyz");
    ASSERT_ALWAYS (cmp == 0);
    ASSERT_ALWAYS (n == 4);
  }
#endif

  /* %zd etc won't be accepted by sscanf on old systems, and running
     something to see if they work might be bad, so only try it on glibc,
     and only on a new enough version (glibc 2.0 doesn't have %zd) */
#if __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 0)
  {
    mpz_t   z;
    size_t  s = -1;
    mpz_init (z);
    ret = gmp_sscanf ("456 789", "%zd %Zd", &s, z);
    ASSERT_ALWAYS (ret == 2);
    ASSERT_ALWAYS (s == 456);
    ASSERT_ALWAYS (mpz_cmp_ui (z, 789L) == 0);
    mpz_clear (z);
  }
  {
    mpz_t      z;
    ptrdiff_t  d = -1;
    mpz_init (z);
    ret = gmp_sscanf ("456 789", "%td %Zd", &d, z);
    ASSERT_ALWAYS (ret == 2);
    ASSERT_ALWAYS (d == 456);
    ASSERT_ALWAYS (mpz_cmp_ui (z, 789L) == 0);
    mpz_clear (z);
  }
  {
    mpz_t      z;
    long long  ll = -1;
    mpz_init (z);
    ret = gmp_sscanf ("456 789", "%Ld %Zd", &ll, z);
    ASSERT_ALWAYS (ret == 2);
    ASSERT_ALWAYS (ll == 456);
    ASSERT_ALWAYS (mpz_cmp_ui (z, 789L) == 0);
    mpz_clear (z);
  }
#endif
}

int
main (int argc, char *argv[])
{
  if (argc > 1 && strcmp (argv[1], "-s") == 0)
    option_libc_scanf = 1;

  tests_start ();

  mp_trace_base = 16;

  check_z ();
  check_q ();
  check_f ();
  check_n ();
  check_misc ();

  unlink (TEMPFILE);
  tests_end ();
  exit (0);
}
