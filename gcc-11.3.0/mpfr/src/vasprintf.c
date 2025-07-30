/* mpfr_vasprintf -- main function for the printf functions family
   plus helper macros & functions.

Copyright 2007-2017 Free Software Foundation, Inc.
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
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA. */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

/* The mpfr_printf-like functions are defined only if <stdarg.h> exists */
#ifdef HAVE_STDARG

#include <stdarg.h>

#ifndef HAVE_VA_COPY
# ifdef HAVE___VA_COPY
#  define va_copy(dst,src) __va_copy(dst, src)
# else
/* autoconf manual advocates this fallback.
   This is also the solution chosen by gmp */
#  define va_copy(dst,src) \
  do { memcpy(&(dst), &(src), sizeof(va_list)); } while (0)
# endif /* HAVE___VA_COPY */
#endif /* HAVE_VA_COPY */

#ifdef HAVE_WCHAR_H
#include <wchar.h>
#endif

#if defined (__cplusplus)
#include <cstddef>
#define __STDC_LIMIT_MACROS   /* SIZE_MAX defined with <stdint.h> inclusion */
#else
#include <stddef.h>             /* for ptrdiff_t */
#endif

#include <errno.h>

#define MPFR_NEED_LONGLONG_H
#include "mpfr-intmax.h"
#include "mpfr-impl.h"

/* Define a length modifier corresponding to mpfr_prec_t.
   We use literal string instead of literal character so as to permit future
   extension to long long int ("ll"). */
#if   _MPFR_PREC_FORMAT == 1
#define MPFR_PREC_FORMAT_TYPE "h"
#define MPFR_PREC_FORMAT_SIZE 1
#elif _MPFR_PREC_FORMAT == 2
#define MPFR_PREC_FORMAT_TYPE ""
#define MPFR_PREC_FORMAT_SIZE 0
#elif _MPFR_PREC_FORMAT == 3
#define MPFR_PREC_FORMAT_TYPE "l"
#define MPFR_PREC_FORMAT_SIZE 1
#else
#error "mpfr_prec_t size not supported"
#endif

/* Output for special values defined in the C99 standard */
#define MPFR_NAN_STRING_LC "nan"
#define MPFR_NAN_STRING_UC "NAN"
#define MPFR_NAN_STRING_LENGTH 3
#define MPFR_INF_STRING_LC "inf"
#define MPFR_INF_STRING_UC "INF"
#define MPFR_INF_STRING_LENGTH 3

/* The implicit \0 is useless, but we do not write num_to_text[16]
   otherwise g++ complains. */
static const char num_to_text[] = "0123456789abcdef";

/* some macro and functions for parsing format string */

/* Read an integer; saturate to INT_MAX. */
#define READ_INT(ap, format, specinfo, field, label_out)                \
  do {                                                                  \
    while (*(format))                                                   \
      {                                                                 \
        int _i;                                                         \
        switch (*(format))                                              \
          {                                                             \
          case '0':                                                     \
          case '1':                                                     \
          case '2':                                                     \
          case '3':                                                     \
          case '4':                                                     \
          case '5':                                                     \
          case '6':                                                     \
          case '7':                                                     \
          case '8':                                                     \
          case '9':                                                     \
            specinfo.field = (specinfo.field <= INT_MAX / 10) ?         \
              specinfo.field * 10 : INT_MAX;                            \
            _i = *(format) - '0';                                       \
            MPFR_ASSERTN (_i >= 0 && _i <= 9);                          \
            specinfo.field = (specinfo.field <= INT_MAX - _i) ?         \
              specinfo.field + _i : INT_MAX;                            \
            ++(format);                                                 \
            break;                                                      \
          case '*':                                                     \
            specinfo.field = va_arg ((ap), int);                        \
            ++(format);                                                 \
          default:                                                      \
            goto label_out;                                             \
          }                                                             \
      }                                                                 \
  } while (0)

/* arg_t contains all the types described by the 'type' field of the
   format string */
enum arg_t
  {
    NONE,
    CHAR_ARG,
    SHORT_ARG,
    LONG_ARG,
    LONG_LONG_ARG,
    INTMAX_ARG,
    SIZE_ARG,
    PTRDIFF_ARG,
    LONG_DOUBLE_ARG,
    MPF_ARG,
    MPQ_ARG,
    MP_LIMB_ARG,
    MP_LIMB_ARRAY_ARG,
    MPZ_ARG,
    MPFR_PREC_ARG,
    MPFR_ARG,
    UNSUPPORTED
  };

/* Each conversion specification of the format string will be translated in a
   printf_spec structure by the parser.
   This structure is adapted from the GNU libc one. */
struct printf_spec
{
  unsigned int alt:1;           /* # flag */
  unsigned int space:1;         /* Space flag */
  unsigned int left:1;          /* - flag */
  unsigned int showsign:1;      /* + flag */
  unsigned int group:1;         /* ' flag */

  int width;                    /* Width */
  int prec;                     /* Precision */

  enum arg_t arg_type;          /* Type of argument */
  mpfr_rnd_t rnd_mode;            /* Rounding mode */
  char spec;                    /* Conversion specifier */

  char pad;                     /* Padding character */
};

static void
specinfo_init (struct printf_spec *specinfo)
{
  specinfo->alt = 0;
  specinfo->space = 0;
  specinfo->left = 0;
  specinfo->showsign = 0;
  specinfo->group = 0;
  specinfo->width = 0;
  specinfo->prec = 0;
  specinfo->arg_type = NONE;
  specinfo->rnd_mode = MPFR_RNDN;
  specinfo->spec = '\0';
  specinfo->pad = ' ';
}

#define FLOATING_POINT_ARG_TYPE(at) \
  ((at) == MPFR_ARG || (at) == MPF_ARG || (at) == LONG_DOUBLE_ARG)

#define INTEGER_LIKE_ARG_TYPE(at)                                       \
  ((at) == SHORT_ARG || (at) == LONG_ARG || (at) == LONG_LONG_ARG       \
   || (at) == INTMAX_ARG  || (at) == MPFR_PREC_ARG || (at) == MPZ_ARG   \
   || (at) == MPQ_ARG || (at) == MP_LIMB_ARG || (at) == MP_LIMB_ARRAY_ARG \
   || (at) == CHAR_ARG || (at) == SIZE_ARG || (at) == PTRDIFF_ARG)

static int
specinfo_is_valid (struct printf_spec spec)
{
  switch (spec.spec)
    {
    case 'n':
      return -1;

    case 'a':    case 'A':
    case 'e':    case 'E':
    case 'f':    case 'F':
    case 'g':    case 'G':
      return (spec.arg_type == NONE
              || FLOATING_POINT_ARG_TYPE (spec.arg_type));

    case 'b':
      return spec.arg_type == MPFR_ARG;

    case 'd':    case 'i':
    case 'u':    case 'o':
    case 'x':    case 'X':
      return (spec.arg_type == NONE
              || INTEGER_LIKE_ARG_TYPE (spec.arg_type));

    case 'c':
    case 's':
      return (spec.arg_type == NONE || spec.arg_type == LONG_ARG);

    case 'p':
      return spec.arg_type == NONE;

    default:
      return 0;
    }
}

static const char *
parse_flags (const char *format, struct printf_spec *specinfo)
{
  while (*format)
    {
      switch (*format)
        {
        case '0':
          specinfo->pad = '0';
          ++format;
          break;
        case '#':
          specinfo->alt = 1;
          ++format;
          break;
        case '+':
          specinfo->showsign = 1;
          ++format;
          break;
        case ' ':
          specinfo->space = 1;
          ++format;
          break;
        case '-':
          specinfo->left = 1;
          ++format;
          break;
        case '\'':
          /* Single UNIX Specification for thousand separator */
          specinfo->group = 1;
          ++format;
          break;
        default:
          return format;
        }
    }
  return format;
}

static const char *
parse_arg_type (const char *format, struct printf_spec *specinfo)
{
  switch (*format)
    {
    case '\0':
      break;
    case 'h':
      if (*++format == 'h')
#ifndef NPRINTF_HH
        {
          ++format;
          specinfo->arg_type = CHAR_ARG;
        }
#else
        specinfo->arg_type = UNSUPPORTED;
#endif
      else
        specinfo->arg_type = SHORT_ARG;
      break;
    case 'l':
      if (*++format == 'l')
        {
          ++format;
#if defined (HAVE_LONG_LONG) && !defined(NPRINTF_LL)
          specinfo->arg_type = LONG_LONG_ARG;
#else
          specinfo->arg_type = UNSUPPORTED;
#endif
          break;
        }
      else
        {
          specinfo->arg_type = LONG_ARG;
          break;
        }
    case 'j':
      ++format;
#if defined(_MPFR_H_HAVE_INTMAX_T) && !defined(NPRINTF_J)
      specinfo->arg_type = INTMAX_ARG;
#else
      specinfo->arg_type = UNSUPPORTED;
#endif
      break;
    case 'z':
      ++format;
      specinfo->arg_type = SIZE_ARG;
      break;
    case 't':
      ++format;
#ifndef NPRINTF_T
      specinfo->arg_type = PTRDIFF_ARG;
#else
      specinfo->arg_type = UNSUPPORTED;
#endif
      break;
    case 'L':
      ++format;
#ifndef NPRINTF_L
      specinfo->arg_type = LONG_DOUBLE_ARG;
#else
      specinfo->arg_type = UNSUPPORTED;
#endif
      break;
    case 'F':
      ++format;
      specinfo->arg_type = MPF_ARG;
      break;
    case 'Q':
      ++format;
      specinfo->arg_type = MPQ_ARG;
      break;
    case 'M':
      ++format;
      /* The 'M' specifier was added in gmp 4.2.0 */
      specinfo->arg_type = MP_LIMB_ARG;
      break;
    case 'N':
      ++format;
      specinfo->arg_type = MP_LIMB_ARRAY_ARG;
      break;
    case 'Z':
      ++format;
      specinfo->arg_type = MPZ_ARG;
      break;

      /* mpfr specific specifiers */
    case 'P':
      ++format;
      specinfo->arg_type = MPFR_PREC_ARG;
      break;
    case 'R':
      ++format;
      specinfo->arg_type = MPFR_ARG;
    }
  return format;
}


/* some macros and functions filling the buffer */

/* CONSUME_VA_ARG removes from va_list AP the type expected by SPECINFO */

/* With a C++ compiler wchar_t and enumeration in va_list are converted to
   integer type : int, unsigned int, long or unsigned long (unfortunately,
   this is implementation dependent).
   We follow gmp which assumes in print/doprnt.c that wchar_t is converted
   to int (because wchar_t <= int).
   For wint_t, we assume that the case WINT_MAX < INT_MAX yields an
   integer promotion. */
#ifdef HAVE_WCHAR_H
#if defined(WINT_MAX) && WINT_MAX < INT_MAX
typedef int    mpfr_va_wint;  /* integer promotion */
#else
typedef wint_t mpfr_va_wint;
#endif
#define CASE_LONG_ARG(specinfo, ap)                                     \
  case LONG_ARG:                                                        \
  if (((specinfo).spec == 'd') || ((specinfo).spec == 'i')              \
      || ((specinfo).spec == 'o') || ((specinfo).spec == 'u')           \
      || ((specinfo).spec == 'x') || ((specinfo).spec == 'X'))          \
    (void) va_arg ((ap), long);                                         \
  else if ((specinfo).spec == 'c')                                      \
    (void) va_arg ((ap), mpfr_va_wint);                                 \
  else if ((specinfo).spec == 's')                                      \
    (void) va_arg ((ap), int); /* we assume integer promotion */        \
  break;
#else
#define CASE_LONG_ARG(specinfo, ap)             \
  case LONG_ARG:                                \
  (void) va_arg ((ap), long);                   \
  break;
#endif

#if defined(_MPFR_H_HAVE_INTMAX_T)
#define CASE_INTMAX_ARG(specinfo, ap)           \
  case INTMAX_ARG:                              \
  (void) va_arg ((ap), intmax_t);               \
  break;
#else
#define CASE_INTMAX_ARG(specinfo, ap)
#endif

#ifdef HAVE_LONG_LONG
#define CASE_LONG_LONG_ARG(specinfo, ap)        \
  case LONG_LONG_ARG:                           \
  (void) va_arg ((ap), long long);              \
  break;
#else
#define CASE_LONG_LONG_ARG(specinfo, ap)
#endif

#define CONSUME_VA_ARG(specinfo, ap)            \
  do {                                          \
    switch ((specinfo).arg_type)                \
      {                                         \
      case CHAR_ARG:                            \
      case SHORT_ARG:                           \
        (void) va_arg ((ap), int);              \
        break;                                  \
      CASE_LONG_ARG (specinfo, ap)              \
      CASE_LONG_LONG_ARG (specinfo, ap)         \
      CASE_INTMAX_ARG (specinfo, ap)            \
      case SIZE_ARG:                            \
        (void) va_arg ((ap), size_t);           \
        break;                                  \
      case PTRDIFF_ARG:                         \
        (void) va_arg ((ap), ptrdiff_t);        \
        break;                                  \
      case LONG_DOUBLE_ARG:                     \
        (void) va_arg ((ap), long double);      \
        break;                                  \
      case MPF_ARG:                             \
        (void) va_arg ((ap), mpf_srcptr);       \
        break;                                  \
      case MPQ_ARG:                             \
        (void) va_arg ((ap), mpq_srcptr);       \
        break;                                  \
      case MP_LIMB_ARG:                         \
        (void) va_arg ((ap), mp_limb_t);        \
        break;                                  \
      case MP_LIMB_ARRAY_ARG:                   \
        (void) va_arg ((ap), mpfr_limb_ptr);    \
        (void) va_arg ((ap), mp_size_t);        \
        break;                                  \
      case MPZ_ARG:                             \
        (void) va_arg ((ap), mpz_srcptr);       \
        break;                                  \
      default:                                  \
        switch ((specinfo).spec)                \
          {                                     \
          case 'd':                             \
          case 'i':                             \
          case 'o':                             \
          case 'u':                             \
          case 'x':                             \
          case 'X':                             \
          case 'c':                             \
            (void) va_arg ((ap), int);          \
            break;                              \
          case 'f':                             \
          case 'F':                             \
          case 'e':                             \
          case 'E':                             \
          case 'g':                             \
          case 'G':                             \
          case 'a':                             \
          case 'A':                             \
            (void) va_arg ((ap), double);       \
            break;                              \
          case 's':                             \
            (void) va_arg ((ap), char *);       \
            break;                              \
          case 'p':                             \
            (void) va_arg ((ap), void *);       \
          }                                     \
      }                                         \
  } while (0)

/* process the format part which does not deal with mpfr types,
   jump to external label 'error' if gmp_asprintf return -1. */
#define FLUSH(flag, start, end, ap, buf_ptr)                            \
  do {                                                                  \
    const size_t n = (end) - (start);                                   \
    if ((flag))                                                         \
      /* previous specifiers are understood by gmp_printf */            \
      {                                                                 \
        MPFR_TMP_DECL (marker);                                         \
        char *fmt_copy;                                                 \
        MPFR_TMP_MARK (marker);                                         \
        fmt_copy = (char*) MPFR_TMP_ALLOC (n + 1);                      \
        strncpy (fmt_copy, (start), n);                                 \
        fmt_copy[n] = '\0';                                             \
        if (sprntf_gmp ((buf_ptr), (fmt_copy), (ap)) == -1)             \
          {                                                             \
            MPFR_TMP_FREE (marker);                                     \
            goto error;                                                 \
          }                                                             \
        (flag) = 0;                                                     \
        MPFR_TMP_FREE (marker);                                         \
      }                                                                 \
    else if ((start) != (end))                                          \
      /* no conversion specification, just simple characters */         \
      buffer_cat ((buf_ptr), (start), n);                               \
  } while (0)

struct string_buffer
{
  char *start;                  /* beginning of the buffer */
  char *curr;                   /* null terminating character */
  size_t size;                  /* buffer capacity */
};

static void
buffer_init (struct string_buffer *b, size_t s)
{
  b->start = (char *) (*__gmp_allocate_func) (s);
  b->start[0] = '\0';
  b->curr = b->start;
  b->size = s;
}

/* Increase buffer size by a number of character being the least multiple of
   4096 greater than LEN+1. */
static void
buffer_widen (struct string_buffer *b, size_t len)
{
  const size_t pos = b->curr - b->start;
  const size_t n = 0x1000 + (len & ~((size_t) 0xfff));
  MPFR_ASSERTD (pos < b->size);

  MPFR_ASSERTN ((len & ~((size_t) 4095)) <= (size_t)(SIZE_MAX - 4096));
  MPFR_ASSERTN (b->size < SIZE_MAX - n);

  b->start =
    (char *) (*__gmp_reallocate_func) (b->start, b->size, b->size + n);
  b->size += n;
  b->curr = b->start + pos;

  MPFR_ASSERTD (pos < b->size);
  MPFR_ASSERTD (*b->curr == '\0');
}

/* Concatenate the LEN first characters of the string S to the buffer B and
   expand it if needed. */
static void
buffer_cat (struct string_buffer *b, const char *s, size_t len)
{
  MPFR_ASSERTD (len != 0);
  MPFR_ASSERTD (len <= strlen (s));

  if (MPFR_UNLIKELY ((b->curr + len) >= (b->start + b->size)))
    buffer_widen (b, len);

  strncat (b->curr, s, len);
  b->curr += len;

  MPFR_ASSERTD (b->curr < b->start + b->size);
  MPFR_ASSERTD (*b->curr == '\0');
}

/* Add N characters C to the end of buffer B */
static void
buffer_pad (struct string_buffer *b, const char c, const size_t n)
{
  MPFR_ASSERTD (n != 0);

  MPFR_ASSERTN (b->size < SIZE_MAX - n - 1);
  if (MPFR_UNLIKELY ((b->curr + n + 1) > (b->start + b->size)))
    buffer_widen (b, n);

  if (n == 1)
    *b->curr = c;
  else
    memset (b->curr, c, n);
  b->curr += n;
  *b->curr = '\0';

  MPFR_ASSERTD (b->curr < b->start + b->size);
}

/* Form a string by concatenating the first LEN characters of STR to TZ
   zero(s), insert into one character C each 3 characters starting from end
   to begining and concatenate the result to the buffer B. */
static void
buffer_sandwich (struct string_buffer *b, char *str, size_t len,
                 const size_t tz, const char c)
{
  const size_t step = 3;
  const size_t size = len + tz;
  const size_t r = size % step == 0 ? step : size % step;
  const size_t q = size % step == 0 ? size / step - 1 : size / step;
  size_t i;

  MPFR_ASSERTD (size != 0);
  if (c == '\0')
    {
      buffer_cat (b, str, len);
      buffer_pad (b, '0', tz);
      return;
    }

  MPFR_ASSERTN (b->size < SIZE_MAX - size - 1 - q);
  MPFR_ASSERTD (len <= strlen (str));
  if (MPFR_UNLIKELY ((b->curr + size + 1 + q) > (b->start + b->size)))
    buffer_widen (b, size + q);

  /* first R significant digits */
  memcpy (b->curr, str, r);
  b->curr += r;
  str += r;
  len -= r;

  /* blocks of thousands. Warning: STR might end in the middle of a block */
  for (i = 0; i < q; ++i)
    {
      *b->curr++ = c;
      if (MPFR_LIKELY (len > 0))
        {
          if (MPFR_LIKELY (len >= step))
            /* step significant digits */
            {
              memcpy (b->curr, str, step);
              len -= step;
            }
          else
            /* last digits in STR, fill up thousand block with zeros */
            {
              memcpy (b->curr, str, len);
              memset (b->curr + len, '0', step - len);
              len = 0;
            }
        }
      else
        /* trailing zeros */
        memset (b->curr, '0', step);

      b->curr += step;
      str += step;
    }

  *b->curr = '\0';

  MPFR_ASSERTD (b->curr < b->start + b->size);
}

/* let gmp_xprintf process the part it can understand */
static int
sprntf_gmp (struct string_buffer *b, const char *fmt, va_list ap)
{
  int length;
  char *s;

  length = gmp_vasprintf (&s, fmt, ap);
  if (length > 0)
    buffer_cat (b, s, length);

  mpfr_free_str (s);
  return length;
}

/* Helper struct and functions for temporary strings management */
/* struct for easy string clearing */
struct string_list
{
  char *string;
  struct string_list *next; /* NULL in last node */
};

/* initialisation */
static void
init_string_list (struct string_list *sl)
{
  sl->string = NULL;
  sl->next = NULL;
}

/* clear all strings in the list */
static void
clear_string_list (struct string_list *sl)
{
  struct string_list *n;

  while (sl)
    {
      if (sl->string)
        mpfr_free_str (sl->string);
      n = sl->next;
      (*__gmp_free_func) (sl, sizeof(struct string_list));
      sl = n;
    }
}

/* add a string in the list */
static char *
register_string (struct string_list *sl, char *new_string)
{
  /* look for the last node */
  while (sl->next)
    sl = sl->next;

  sl->next = (struct string_list*)
    (*__gmp_allocate_func) (sizeof (struct string_list));

  sl = sl->next;
  sl->next = NULL;
  return sl->string = new_string;
}

/* padding type: where are the padding characters */
enum pad_t
  {
    LEFT,          /* spaces in left hand side for right justification */
    LEADING_ZEROS, /* padding with '0' characters in integral part */
    RIGHT          /* spaces in right hand side for left justification */
  };

/* number_parts details how much characters are needed in each part of a float
   print.  */
struct number_parts
{
  enum pad_t pad_type;    /* Padding type */
  size_t pad_size;        /* Number of padding characters */

  char sign;              /* Sign character */

  char *prefix_ptr;       /* Pointer to prefix part */
  size_t prefix_size;     /* Number of characters in *prefix_ptr */

  char thousands_sep;     /* Thousands separator (only with style 'f') */

  char *ip_ptr;           /* Pointer to integral part characters*/
  size_t ip_size;         /* Number of digits in *ip_ptr */
  int ip_trailing_zeros;  /* Number of additional null digits in integral
                             part */

  char point;             /* Decimal point character */

  int fp_leading_zeros;   /* Number of additional leading zeros in fractional
                             part */
  char *fp_ptr;           /* Pointer to fractional part characters */
  size_t fp_size;         /* Number of digits in *fp_ptr */
  int fp_trailing_zeros;  /* Number of additional trailing zeros in fractional
                             part */

  char *exp_ptr;          /* Pointer to exponent part */
  size_t exp_size;        /* Number of characters in *exp_ptr */

  struct string_list *sl; /* List of string buffers in use: we need such a
                             mechanism because fp_ptr may point into the same
                             string as ip_ptr */
};

/* For a real non zero number x, what is the base exponent f when rounding x
   with rounding mode r to r(x) = m*b^f, where m is a digit and 1 <= m < b ?
   Return non zero value if x is rounded up to b^f, return zero otherwise */
static int
next_base_power_p (mpfr_srcptr x, int base, mpfr_rnd_t rnd)
{
  mpfr_prec_t nbits;
  mp_limb_t pm;
  mp_limb_t xm;

  MPFR_ASSERTD (MPFR_IS_PURE_FP (x));
  MPFR_ASSERTD (base == 2 || base == 16);

  /* Warning: the decimal point is AFTER THE FIRST DIGIT in this output
     representation. */
  nbits = base == 2 ? 1 : 4;

  if (rnd == MPFR_RNDZ
      || (rnd == MPFR_RNDD && MPFR_IS_POS (x))
      || (rnd == MPFR_RNDU && MPFR_IS_NEG (x))
      || MPFR_PREC (x) <= nbits)
    /* no rounding when printing x with 1 digit */
    return 0;

  xm = MPFR_MANT (x) [MPFR_LIMB_SIZE (x) - 1];
  pm = MPFR_LIMB_MASK (GMP_NUMB_BITS - nbits);
  if ((xm & ~pm) ^ ~pm)
    /* do no round up if some of the nbits first bits are 0s. */
    return 0;

  if (rnd == MPFR_RNDN)
    /* mask for rounding bit */
    pm = (MPFR_LIMB_ONE << (GMP_NUMB_BITS - nbits - 1));

  /* round up if some remaining bits are 1 */
  /* warning: the return value must be an int */
  return xm & pm ? 1 : 0;
}

/* Record information from mpfr_get_str() so as to avoid multiple
   calls to this expensive function. */
struct decimal_info
{
  mpfr_exp_t exp;
  char *str;
};

/* For a real non zero number x, what is the exponent f so that
   10^f <= x < 10^(f+1). */
static mpfr_exp_t
floor_log10 (mpfr_srcptr x)
{
  mpfr_t y;
  mpfr_exp_t exp;

  /* make sure first that y can represent a mpfr_exp_t exactly
     and can compare with x */
  mpfr_prec_t prec = sizeof (mpfr_exp_t) * CHAR_BIT;
  mpfr_init2 (y, MAX (prec, MPFR_PREC (x)));

  exp = mpfr_ceil_mul (MPFR_GET_EXP (x), 10, 1) - 1;
  mpfr_set_exp_t (y, exp, MPFR_RNDU);
  /* The following call to mpfr_ui_pow should be fast: y is an integer
     (not too large), so that mpfr_pow_z will be used internally. */
  mpfr_ui_pow (y, 10, y, MPFR_RNDU);
  if (mpfr_cmpabs (x, y) < 0)
    exp--;

  mpfr_clear (y);
  return exp;
}

/* Determine the different parts of the string representation of the regular
   number P when SPEC.SPEC is 'a', 'A', or 'b'.

   return -1 if some field > INT_MAX */
static int
regular_ab (struct number_parts *np, mpfr_srcptr p,
            const struct printf_spec spec)
{
  int uppercase;
  int base;
  char *str;
  mpfr_exp_t exp;

  uppercase = spec.spec == 'A';

  /* sign */
  if (MPFR_IS_NEG (p))
    np->sign = '-';
  else if (spec.showsign || spec.space)
    np->sign = spec.showsign ? '+' : ' ';

  if (spec.spec == 'a' || spec.spec == 'A')
    /* prefix part */
    {
      np->prefix_size = 2;
      str = (char *) (*__gmp_allocate_func) (1 + np->prefix_size);
      str[0] = '0';
      str[1] = uppercase ? 'X' : 'x';
      str[2] = '\0';
      np->prefix_ptr = register_string (np->sl, str);
    }

  /* integral part */
  np->ip_size = 1;
  base = (spec.spec == 'b') ? 2 : 16;

  if (spec.prec != 0)
    {
      size_t nsd;

      /* Number of significant digits:
         - if no given precision, let mpfr_get_str determine it;
         - if a non-zero precision is specified, then one digit before decimal
         point plus SPEC.PREC after it. */
      nsd = spec.prec < 0 ? 0 : spec.prec + np->ip_size;
      str = mpfr_get_str (0, &exp, base, nsd, p, spec.rnd_mode);
      register_string (np->sl, str);
      np->ip_ptr = MPFR_IS_NEG (p) ? ++str : str;  /* skip sign if any */

      if (base == 16)
        /* EXP is the exponent for radix sixteen with decimal point BEFORE the
           first digit, we want the exponent for radix two and the decimal
           point AFTER the first digit. */
        {
          /* An integer overflow is normally not possible since MPFR_EXP_MIN
             is twice as large as MPFR_EMIN_MIN. */
          MPFR_ASSERTN (exp > (MPFR_EXP_MIN + 3) / 4);
          exp = (exp - 1) * 4;
        }
      else
        /* EXP is the exponent for decimal point BEFORE the first digit, we
           want the exponent for decimal point AFTER the first digit. */
        {
          /* An integer overflow is normally not possible since MPFR_EXP_MIN
             is twice as large as MPFR_EMIN_MIN. */
          MPFR_ASSERTN (exp > MPFR_EXP_MIN);
          --exp;
        }
    }
  else if (next_base_power_p (p, base, spec.rnd_mode))
    {
      str = (char *)(*__gmp_allocate_func) (2);
      str[0] = '1';
      str[1] = '\0';
      np->ip_ptr = register_string (np->sl, str);

      exp = MPFR_GET_EXP (p);
    }
  else if (base == 2)
    {
      str = (char *)(*__gmp_allocate_func) (2);
      str[0] = '1';
      str[1] = '\0';
      np->ip_ptr = register_string (np->sl, str);

      exp = MPFR_GET_EXP (p) - 1;
    }
  else
    {
      int digit;
      mp_limb_t msl = MPFR_MANT (p)[MPFR_LIMB_SIZE (p) - 1];
      int rnd_bit = GMP_NUMB_BITS - 5;

      /* pick up the 4 first bits */
      digit = msl >> (rnd_bit+1);
      if (spec.rnd_mode == MPFR_RNDA
          || (spec.rnd_mode == MPFR_RNDU && MPFR_IS_POS (p))
          || (spec.rnd_mode == MPFR_RNDD && MPFR_IS_NEG (p))
          || (spec.rnd_mode == MPFR_RNDN
              && (msl & (MPFR_LIMB_ONE << rnd_bit))))
        digit++;
      MPFR_ASSERTD ((0 <= digit) && (digit <= 15));

      str = (char *)(*__gmp_allocate_func) (1 + np->ip_size);
      str[0] = num_to_text [digit];
      str[1] = '\0';
      np->ip_ptr = register_string (np->sl, str);

      exp = MPFR_GET_EXP (p) - 4;
    }

  if (uppercase)
    /* All digits in upper case */
    {
      char *s1 = str;
      while (*s1)
        {
          switch (*s1)
            {
            case 'a':
              *s1 = 'A';
              break;
            case 'b':
              *s1 = 'B';
              break;
            case 'c':
              *s1 = 'C';
              break;
            case 'd':
              *s1 = 'D';
              break;
            case 'e':
              *s1 = 'E';
              break;
            case 'f':
              *s1 = 'F';
              break;
            }
          s1++;
        }
    }

  if (spec.spec == 'b' || spec.prec != 0)
    /* compute the number of digits in fractional part */
    {
      char *ptr;
      size_t str_len;

      /* the sign has been skipped, skip also the first digit */
      ++str;
      str_len = strlen (str);
      ptr = str + str_len - 1; /* points to the end of str */

      if (spec.prec < 0)
        /* remove trailing zeros, if any */
        {
          while ((*ptr == '0') && (str_len != 0))
            {
              --ptr;
              --str_len;
            }
        }

      if (str_len > INT_MAX)
        /* too many digits in fractional part */
        return -1;

      if (str_len != 0)
        /* there are some non-zero digits in fractional part */
        {
          np->fp_ptr = str;
          np->fp_size = str_len;
          if ((int) str_len < spec.prec)
            np->fp_trailing_zeros = spec.prec - str_len;
        }
    }

  /* decimal point */
  if ((np->fp_size != 0) || spec.alt)
    np->point = MPFR_DECIMAL_POINT;

  /* the exponent part contains the character 'p', or 'P' plus the sign
     character plus at least one digit and only as many more digits as
     necessary to represent the exponent.
     We assume that |EXP| < 10^INT_MAX. */
  np->exp_size = 3;
  {
    mpfr_uexp_t x;

    x = SAFE_ABS (mpfr_uexp_t, exp);
    while (x > 9)
      {
        np->exp_size++;
        x /= 10;
      }
  }
  str = (char *) (*__gmp_allocate_func) (1 + np->exp_size);
  np->exp_ptr = register_string (np->sl, str);
  {
    char exp_fmt[8];  /* contains at most 7 characters like in "p%+.1i",
                         or "P%+.2li" */

    exp_fmt[0] = uppercase ? 'P' : 'p';
    exp_fmt[1] = '\0';
    strcat (exp_fmt, "%+.1" MPFR_EXP_FSPEC "d");

    if (sprintf (str, exp_fmt, (mpfr_eexp_t) exp) < 0)
      return -1;
  }

  return 0;
}

/* Determine the different parts of the string representation of the regular
   number P when spec.spec is 'e', 'E', 'g', or 'G'.
   DEC_INFO contains the previously computed exponent and string or is NULL.

   return -1 if some field > INT_MAX */
static int
regular_eg (struct number_parts *np, mpfr_srcptr p,
            const struct printf_spec spec, struct decimal_info *dec_info)
{
  char *str;
  mpfr_exp_t exp;

  const int uppercase = spec.spec == 'E' || spec.spec == 'G';
  const int spec_g = spec.spec == 'g' || spec.spec == 'G';
  const int keep_trailing_zeros = (spec_g && spec.alt)
    || (!spec_g && (spec.prec > 0));

  /* sign */
  if (MPFR_IS_NEG (p))
    np->sign = '-';
  else if (spec.showsign || spec.space)
    np->sign = spec.showsign ? '+' : ' ';

  /* integral part */
  np->ip_size = 1;
  if (dec_info == NULL)
    {
      size_t nsd;

      /* Number of significant digits:
         - if no given precision, then let mpfr_get_str determine it,
         - if a precision is specified, then one digit before decimal point
         plus SPEC.PREC after it.
         We use the fact here that mpfr_get_str allows us to ask for only one
         significant digit when the base is not a power of 2. */
      nsd = (spec.prec < 0) ? 0 : spec.prec + np->ip_size;
      str = mpfr_get_str (0, &exp, 10, nsd, p, spec.rnd_mode);
      register_string (np->sl, str);
    }
  else
    {
      exp = dec_info->exp;
      str = dec_info->str;
    }
  np->ip_ptr = MPFR_IS_NEG (p) ? ++str : str;  /* skip sign if any */

  if (spec.prec != 0)
    /* compute the number of digits in fractional part */
    {
      char *ptr;
      size_t str_len;

      /* the sign has been skipped, skip also the first digit */
      ++str;
      str_len = strlen (str);
      ptr = str + str_len - 1; /* points to the end of str */

      if (!keep_trailing_zeros)
        /* remove trailing zeros, if any */
        {
          while ((*ptr == '0') && (str_len != 0))
            {
              --ptr;
              --str_len;
            }
        }

      if (str_len > INT_MAX)
        /* too many digits in fractional part */
        return -1;

      if (str_len != 0)
        /* there are some non-zero digits in fractional part */
        {
          np->fp_ptr = str;
          np->fp_size = str_len;
          if ((!spec_g || spec.alt) && (spec.prec > 0)
              && ((int)str_len < spec.prec))
            /* add missing trailing zeros */
            np->fp_trailing_zeros = spec.prec - str_len;
        }
    }

  /* decimal point */
  if (np->fp_size != 0 || spec.alt)
    np->point = MPFR_DECIMAL_POINT;

  /* EXP is the exponent for decimal point BEFORE the first digit, we want
     the exponent for decimal point AFTER the first digit.
     Here, no possible overflow because exp < MPFR_EXP (p) / 3 */
  exp--;

  /* the exponent part contains the character 'e', or 'E' plus the sign
     character plus at least two digits and only as many more digits as
     necessary to represent the exponent.
     We assume that |EXP| < 10^INT_MAX. */
  np->exp_size = 3;
  {
    mpfr_uexp_t x;

    x = SAFE_ABS (mpfr_uexp_t, exp);
    while (x > 9)
      {
        np->exp_size++;
        x /= 10;
      }
  }
  if (np->exp_size < 4)
    np->exp_size = 4;

  str = (char *) (*__gmp_allocate_func) (1 + np->exp_size);
  np->exp_ptr = register_string (np->sl, str);

  {
    char exp_fmt[8];  /* e.g. "e%+.2i", or "E%+.2li" */

    exp_fmt[0] = uppercase ? 'E' : 'e';
    exp_fmt[1] = '\0';
    strcat (exp_fmt, "%+.2" MPFR_EXP_FSPEC "d");

    if (sprintf (str, exp_fmt, (mpfr_eexp_t) exp) < 0)
      return -1;
  }

  return 0;
}

/* Determine the different parts of the string representation of the regular
   number P when spec.spec is 'f', 'F', 'g', or 'G'.
   DEC_INFO contains the previously computed exponent and string or is NULL.

   return -1 if some field of number_parts is greater than INT_MAX */
static int
regular_fg (struct number_parts *np, mpfr_srcptr p,
            const struct printf_spec spec, struct decimal_info *dec_info)
{
  mpfr_exp_t exp;
  char * str;
  const int spec_g = (spec.spec == 'g' || spec.spec == 'G');
  const int keep_trailing_zeros = !spec_g || spec.alt;

  /* WARNING: an empty precision field is forbidden (it means precision = 6
     and it should have been changed to 6 before the function call) */
  MPFR_ASSERTD (spec.prec >= 0);

  /* sign */
  if (MPFR_IS_NEG (p))
    np->sign = '-';
  else if (spec.showsign || spec.space)
    np->sign = spec.showsign ? '+' : ' ';

  if (MPFR_GET_EXP (p) <= 0)
    /* 0 < |p| < 1 */
    {
      /* Most of the time, integral part is 0 */
      np->ip_size = 1;
      str = (char *) (*__gmp_allocate_func) (1 + np->ip_size);
      str[0] = '0';
      str[1] = '\0';
      np->ip_ptr = register_string (np->sl, str);

      if (spec.prec == 0)
        /* only two possibilities: either 1 or 0. */
        {
          mpfr_t y;
          /* y = abs(p) */
          MPFR_ALIAS (y, p, 1, MPFR_EXP (p));

          if (spec.rnd_mode == MPFR_RNDA
              || (spec.rnd_mode == MPFR_RNDD && MPFR_IS_NEG (p))
              || (spec.rnd_mode == MPFR_RNDU && MPFR_IS_POS (p))
              || (spec.rnd_mode == MPFR_RNDN && mpfr_cmp_d (y, 0.5) > 0))
            /* rounded up to 1: one digit '1' in integral part.
               note that 0.5 is rounded to 0 with RNDN (round ties to even) */
            np->ip_ptr[0] = '1';
        }
      else
        {
          /* exp =  position of the most significant decimal digit. */
          exp = floor_log10 (p);
          MPFR_ASSERTD (exp < 0);

          if (exp < -spec.prec)
            /* only the last digit may be non zero */
            {
              int round_away;
              switch (spec.rnd_mode)
                {
                case MPFR_RNDA:
                  round_away = 1;
                  break;
                case MPFR_RNDD:
                  round_away = MPFR_IS_NEG (p);
                  break;
                case MPFR_RNDU:
                  round_away = MPFR_IS_POS (p);
                  break;
                case MPFR_RNDN:
                  {
                    /* compare |p| to y = 0.5*10^(-spec.prec) */
                    mpfr_t y;
                    mpfr_exp_t e = MAX (MPFR_PREC (p), 56);
                    mpfr_init2 (y, e + 8);
                    do
                      {
                        /* find a lower approximation of
                           0.5*10^(-spec.prec) different from |p| */
                        e += 8;
                        mpfr_set_prec (y, e);
                        mpfr_set_si (y, -spec.prec, MPFR_RNDN);
                        mpfr_exp10 (y, y, MPFR_RNDD);
                        mpfr_div_2ui (y, y, 1, MPFR_RNDN);
                      } while (mpfr_cmpabs (y, p) == 0);

                    round_away = mpfr_cmpabs (y, p) < 0;
                    mpfr_clear (y);
                  }
                  break;
                default:
                  round_away = 0;
                }

              if (round_away)
                /* round away from zero: the last output digit is '1' */
                {
                  np->fp_leading_zeros = spec.prec - 1;

                  np->fp_size = 1;
                  str =
                    (char *) (*__gmp_allocate_func) (1 + np->fp_size);
                  str[0] = '1';
                  str[1] = '\0';
                  np->fp_ptr = register_string (np->sl, str);
                }
              else
                /* only zeros in fractional part */
                {
                  MPFR_ASSERTD (!spec_g);
                  np->fp_leading_zeros = spec.prec;
                }
            }
          else
            /* the most significant digits are the last
               spec.prec + exp + 1 digits in fractional part */
            {
              char *ptr;
              size_t str_len;
              if (dec_info == NULL)
                {
                  size_t nsd = spec.prec + exp + 1;
                  /* WARNING: nsd may equal 1, but here we use the
                     fact that mpfr_get_str can return one digit with
                     base ten (undocumented feature, see comments in
                     get_str.c) */

                  str = mpfr_get_str (NULL, &exp, 10, nsd, p, spec.rnd_mode);
                  register_string (np->sl, str);
                }
              else
                {
                  exp = dec_info->exp;
                  str = dec_info->str;
                }
              if (MPFR_IS_NEG (p))
                /* skip sign */
                ++str;
              if (exp == 1)
                /* round up to 1 */
                {
                  MPFR_ASSERTD (str[0] == '1');
                  np->ip_ptr[0] = '1';
                  if (!spec_g || spec.alt)
                    np->fp_leading_zeros = spec.prec;
                }
              else
                {
                  np->fp_ptr = str;
                  np->fp_leading_zeros = -exp;
                  MPFR_ASSERTD (exp <= 0);

                  str_len = strlen (str); /* the sign has been skipped */
                  ptr = str + str_len - 1; /* points to the end of str */

                  if (!keep_trailing_zeros)
                    /* remove trailing zeros, if any */
                    {
                      while ((*ptr == '0') && str_len)
                        {
                          --ptr;
                          --str_len;
                        }
                    }

                  if (str_len > INT_MAX)
                    /* too many digits in fractional part */
                    return -1;

                  MPFR_ASSERTD (str_len > 0);
                  np->fp_size = str_len;

                  if ((!spec_g || spec.alt)
                      && spec.prec > 0
                      && (np->fp_leading_zeros + np->fp_size < spec.prec))
                    /* add missing trailing zeros */
                    np->fp_trailing_zeros = spec.prec - np->fp_leading_zeros
                      - np->fp_size;
                }
            }
        }

      if (spec.alt || np->fp_leading_zeros != 0 || np->fp_size != 0
          || np->fp_trailing_zeros != 0)
        np->point = MPFR_DECIMAL_POINT;
    }
  else
    /* 1 <= |p| */
    {
      size_t str_len;

      /* Determine the position of the most significant decimal digit. */
      exp = floor_log10 (p);
      MPFR_ASSERTD (exp >= 0);
      if (exp > INT_MAX)
        /* P is too large to print all its integral part digits */
        return -1;

      if (dec_info == NULL)
        { /* this case occurs with mpfr_printf ("%.0RUf", x) with x=9.5 */
          str =
            mpfr_get_str (NULL, &exp, 10, spec.prec+exp+1, p, spec.rnd_mode);
          register_string (np->sl, str);
        }
      else
        {
          exp = dec_info->exp;
          str = dec_info->str;
        }
      np->ip_ptr = MPFR_IS_NEG (p) ? ++str : str; /* skip sign */
      str_len = strlen (str);

      /* integral part */
      if (exp > str_len)
        /* mpfr_get_str gives no trailing zero when p is rounded up to the next
           power of 10 (p integer, so no fractional part) */
        {
          np->ip_trailing_zeros = exp - str_len;
          np->ip_size = str_len;
        }
      else
        np->ip_size = exp;

      if (spec.group)
        /* thousands separator in integral part */
        np->thousands_sep = MPFR_THOUSANDS_SEPARATOR;

      /* fractional part */
      str += np->ip_size;
      str_len -= np->ip_size;
      if (!keep_trailing_zeros)
        /* remove trailing zeros, if any */
        {
          char *ptr = str + str_len - 1; /* pointer to the last digit of
                                            str */
          while ((*ptr == '0') && (str_len != 0))
            {
              --ptr;
              --str_len;
            }
        }

      if (str_len > 0)
        /* some nonzero digits in fractional part */
        {
          if (str_len > INT_MAX)
            /* too many digits in fractional part */
            return -1;

          np->point = MPFR_DECIMAL_POINT;
          np->fp_ptr = str;
          np->fp_size = str_len;
        }

      if (keep_trailing_zeros && str_len < spec.prec)
        /* add missing trailing zeros */
        {
          np->point = MPFR_DECIMAL_POINT;
          np->fp_trailing_zeros = spec.prec - np->fp_size;
        }

      if (spec.alt)
        /* add decimal point even if no digits follow it */
        np->point = MPFR_DECIMAL_POINT;
    }

  return 0;
}

/* partition_number determines the different parts of the string
   representation of the number p according to the given specification.
   partition_number initializes the given structure np, so all previous
   information in that variable is lost.
   return the total number of characters to be written.
   return -1 if an error occured, in that case np's fields are in an undefined
   state but all string buffers have been freed. */
static int
partition_number (struct number_parts *np, mpfr_srcptr p,
                  struct printf_spec spec)
{
  char *str;
  unsigned int total;  /* can hold the sum of two non-negative int's + 1 */
  int uppercase;

  /* WARNING: left justification means right space padding */
  np->pad_type = spec.left ? RIGHT : spec.pad == '0' ? LEADING_ZEROS : LEFT;
  np->pad_size = 0;
  np->sign = '\0';
  np->prefix_ptr =NULL;
  np->prefix_size = 0;
  np->thousands_sep = '\0';
  np->ip_ptr = NULL;
  np->ip_size = 0;
  np->ip_trailing_zeros = 0;
  np->point = '\0';
  np->fp_leading_zeros = 0;
  np->fp_ptr = NULL;
  np->fp_size = 0;
  np->fp_trailing_zeros = 0;
  np->exp_ptr = NULL;
  np->exp_size = 0;
  np->sl = (struct string_list *)
    (*__gmp_allocate_func) (sizeof (struct string_list));
  init_string_list (np->sl);

  uppercase = spec.spec == 'A' || spec.spec == 'E' || spec.spec == 'F'
    || spec.spec == 'G';

  if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (p)))
    {
      if (MPFR_IS_NAN (p))
        {
          if (np->pad_type == LEADING_ZEROS)
            /* don't want "0000nan", change to right justification padding
               with left spaces instead */
            np->pad_type = LEFT;

          if (uppercase)
            {
              np->ip_size = MPFR_NAN_STRING_LENGTH;
              str = (char *) (*__gmp_allocate_func) (1 + np->ip_size);
              strcpy (str, MPFR_NAN_STRING_UC);
              np->ip_ptr = register_string (np->sl, str);
            }
          else
            {
              np->ip_size = MPFR_NAN_STRING_LENGTH;
              str = (char *) (*__gmp_allocate_func) (1 + np->ip_size);
              strcpy (str, MPFR_NAN_STRING_LC);
              np->ip_ptr = register_string (np->sl, str);
            }
        }
      else if (MPFR_IS_INF (p))
        {
          if (np->pad_type == LEADING_ZEROS)
            /* don't want "0000inf", change to right justification padding
               with left spaces instead */
            np->pad_type = LEFT;

          if (MPFR_IS_NEG (p))
            np->sign = '-';

          if (uppercase)
            {
              np->ip_size = MPFR_INF_STRING_LENGTH;
              str = (char *) (*__gmp_allocate_func) (1 + np->ip_size);
              strcpy (str, MPFR_INF_STRING_UC);
              np->ip_ptr = register_string (np->sl, str);
            }
          else
            {
              np->ip_size = MPFR_INF_STRING_LENGTH;
              str = (char *) (*__gmp_allocate_func) (1 + np->ip_size);
              strcpy (str, MPFR_INF_STRING_LC);
              np->ip_ptr = register_string (np->sl, str);
            }
        }
      else
        /* p == 0 */
        {
          /* note: for 'g' spec, zero is always displayed with 'f'-style with
             precision spec.prec - 1 and the trailing zeros are removed unless
             the flag '#' is used. */
          if (MPFR_IS_NEG (p))
            /* signed zero */
            np->sign = '-';
          else if (spec.showsign || spec.space)
            np->sign = spec.showsign ? '+' : ' ';

          if (spec.spec == 'a' || spec.spec == 'A')
            /* prefix part */
            {
              np->prefix_size = 2;
              str = (char *) (*__gmp_allocate_func) (1 + np->prefix_size);
              str[0] = '0';
              str[1] = uppercase ? 'X' : 'x';
              str[2] = '\0';
              np->prefix_ptr = register_string (np->sl, str);
            }

          /* integral part */
          np->ip_size = 1;
          str = (char *) (*__gmp_allocate_func) (1 + np->ip_size);
          str[0] = '0';
          str[1] = '\0';
          np->ip_ptr = register_string (np->sl, str);

          if (spec.prec > 0
              && ((spec.spec != 'g' && spec.spec != 'G') || spec.alt))
            /* fractional part */
            {
              np->point = MPFR_DECIMAL_POINT;
              np->fp_trailing_zeros = (spec.spec == 'g' || spec.spec == 'G') ?
                spec.prec - 1 : spec.prec;
            }
          else if (spec.alt)
            np->point = MPFR_DECIMAL_POINT;

          if (spec.spec == 'a' || spec.spec == 'A' || spec.spec == 'b'
              || spec.spec == 'e' || spec.spec == 'E')
            /* exponent part */
            {
              np->exp_size = (spec.spec == 'e' || spec.spec == 'E') ? 4 : 3;
              str = (char *) (*__gmp_allocate_func) (1 + np->exp_size);
              if (spec.spec == 'e' || spec.spec == 'E')
                strcpy (str, uppercase ? "E+00" : "e+00");
              else
                strcpy (str, uppercase ? "P+0" : "p+0");
              np->exp_ptr = register_string (np->sl, str);
            }
        }
    }
  else
    /* regular p, p != 0 */
    {
      if (spec.spec == 'a' || spec.spec == 'A' || spec.spec == 'b')
        {
          if (regular_ab (np, p, spec) == -1)
            goto error;
        }
      else if (spec.spec == 'f' || spec.spec == 'F')
        {
          if (spec.prec < 0)
            spec.prec = 6;
          if (regular_fg (np, p, spec, NULL) == -1)
            goto error;
        }
      else if (spec.spec == 'e' || spec.spec == 'E')
        {
          if (regular_eg (np, p, spec, NULL) == -1)
            goto error;
        }
      else
        /* %g case */
        {
          /* Use the C99 rules:
             if T > X >= -4 then the conversion is with style 'f'/'F' and
             precision T-(X+1).
             otherwise, the conversion is with style 'e'/'E' and
             precision T-1.
             where T is the threshold computed below and X is the exponent
             that would be displayed with style 'e' and precision T-1. */
          int threshold;
          mpfr_exp_t x;
          struct decimal_info dec_info;

          threshold = (spec.prec < 0) ? 6 : (spec.prec == 0) ? 1 : spec.prec;
          dec_info.str = mpfr_get_str (NULL, &dec_info.exp, 10, threshold,
                                        p, spec.rnd_mode);
          register_string (np->sl, dec_info.str);
          /* mpfr_get_str corresponds to a significand between 0.1 and 1,
             whereas here we want a significand between 1 and 10. */
          x = dec_info.exp - 1;

          if (threshold > x && x >= -4)
            {
              /* the conversion is with style 'f' */
              spec.prec = threshold - x - 1;

              if (regular_fg (np, p, spec, &dec_info) == -1)
                goto error;
            }
          else
            {
              spec.prec = threshold - 1;

              if (regular_eg (np, p, spec, &dec_info) == -1)
                goto error;
            }
        }
    }

  /* compute the number of characters to be written verifying it is not too
     much */

#define INCR_TOTAL(v)                           \
  do {                                          \
    MPFR_ASSERTD ((v) >= 0);                    \
    if (MPFR_UNLIKELY ((v) > INT_MAX))          \
      goto error;                               \
    total += (v);                               \
    if (MPFR_UNLIKELY (total > INT_MAX))        \
      goto error;                               \
  } while (0)

  total = np->sign ? 1 : 0;
  INCR_TOTAL (np->prefix_size);
  INCR_TOTAL (np->ip_size);
  INCR_TOTAL (np->ip_trailing_zeros);
  MPFR_ASSERTD (np->ip_size + np->ip_trailing_zeros >= 1);
  if (np->thousands_sep)
    /* ' flag, style f and the thousands separator in current locale is not
       reduced to the null character */
    INCR_TOTAL ((np->ip_size + np->ip_trailing_zeros - 1) / 3);
  if (np->point)
    ++total;
  INCR_TOTAL (np->fp_leading_zeros);
  INCR_TOTAL (np->fp_size);
  INCR_TOTAL (np->fp_trailing_zeros);
  INCR_TOTAL (np->exp_size);

  if (spec.width > total)
    /* pad with spaces or zeros depending on np->pad_type */
    {
      np->pad_size = spec.width - total;
      total += np->pad_size; /* here total == spec.width,
                                so 0 < total <= INT_MAX */
      MPFR_ASSERTD (total == spec.width);
    }

  MPFR_ASSERTD (total > 0 && total <= INT_MAX);
  return total;

 error:
  clear_string_list (np->sl);
  np->prefix_ptr = NULL;
  np->ip_ptr = NULL;
  np->fp_ptr = NULL;
  np->exp_ptr = NULL;
  return -1;
}

/* sprnt_fp prints a mpfr_t according to spec.spec specification.

   return the size of the string (not counting the terminating '\0')
   return -1 if the built string is too long (i.e. has more than
   INT_MAX characters). */
static int
sprnt_fp (struct string_buffer *buf, mpfr_srcptr p,
          const struct printf_spec spec)
{
  int length;
  struct number_parts np;

  length = partition_number (&np, p, spec);
  if (length < 0)
    return -1;

  /* right justification padding with left spaces */
  if (np.pad_type == LEFT && np.pad_size != 0)
    buffer_pad (buf, ' ', np.pad_size);

  /* sign character (may be '-', '+', or ' ') */
  if (np.sign)
    buffer_pad (buf, np.sign, 1);

  /* prefix part */
  if (np.prefix_ptr)
    buffer_cat (buf, np.prefix_ptr, np.prefix_size);

  /* right justification  padding with leading zeros */
  if (np.pad_type == LEADING_ZEROS && np.pad_size != 0)
    buffer_pad (buf, '0', np.pad_size);

  /* integral part (may also be "nan" or "inf") */
  MPFR_ASSERTN (np.ip_ptr != NULL); /* never empty */
  if (MPFR_UNLIKELY (np.thousands_sep))
    buffer_sandwich (buf, np.ip_ptr, np.ip_size, np.ip_trailing_zeros,
                     np.thousands_sep);
  else
    {
      buffer_cat (buf, np.ip_ptr, np.ip_size);

      /* trailing zeros in integral part */
      if (np.ip_trailing_zeros != 0)
        buffer_pad (buf, '0', np.ip_trailing_zeros);
    }

  /* decimal point */
  if (np.point)
    buffer_pad (buf, np.point, 1);

  /* leading zeros in fractional part */
  if (np.fp_leading_zeros != 0)
    buffer_pad (buf, '0', np.fp_leading_zeros);

  /* significant digits in fractional part */
  if (np.fp_ptr)
    buffer_cat (buf, np.fp_ptr, np.fp_size);

  /* trailing zeros in fractional part */
  if (np.fp_trailing_zeros != 0)
    buffer_pad (buf, '0', np.fp_trailing_zeros);

  /* exponent part */
  if (np.exp_ptr)
    buffer_cat (buf, np.exp_ptr, np.exp_size);

  /* left justication padding with right spaces */
  if (np.pad_type == RIGHT && np.pad_size != 0)
    buffer_pad (buf, ' ', np.pad_size);

  clear_string_list (np.sl);
  return length;
}

int
mpfr_vasprintf (char **ptr, const char *fmt, va_list ap)
{
  struct string_buffer buf;
  size_t nbchar;

  /* informations on the conversion specification filled by the parser */
  struct printf_spec spec;
  /* flag raised when previous part of fmt need to be processed by
     gmp_vsnprintf */
  int xgmp_fmt_flag;
  /* beginning and end of the previous unprocessed part of fmt */
  const char *start, *end;
  /* pointer to arguments for gmp_vasprintf */
  va_list ap2;

  MPFR_SAVE_EXPO_DECL (expo);
  MPFR_SAVE_EXPO_MARK (expo);

  nbchar = 0;
  buffer_init (&buf, 4096);
  xgmp_fmt_flag = 0;
  va_copy (ap2, ap);
  start = fmt;
  while (*fmt)
    {
      /* Look for the next format specification */
      while ((*fmt) && (*fmt != '%'))
        ++fmt;

      if (*fmt == '\0')
        break;

      if (*++fmt == '%')
        /* %%: go one step further otherwise the second '%' would be
           considered as a new conversion specification introducing
           character */
        {
          ++fmt;
          xgmp_fmt_flag = 1;
          continue;
        }

      end = fmt - 1;

      /* format string analysis */
      specinfo_init (&spec);
      fmt = parse_flags (fmt, &spec);

      READ_INT (ap, fmt, spec, width, width_analysis);
    width_analysis:
      if (spec.width < 0)
        {
          spec.left = 1;
          spec.width = -spec.width;
          MPFR_ASSERTN (spec.width < INT_MAX);
        }
      if (*fmt == '.')
        {
          const char *f = ++fmt;
          READ_INT (ap, fmt, spec, prec, prec_analysis);
        prec_analysis:
          if (f == fmt)
            spec.prec = -1;
        }
      else
        spec.prec = -1;

      fmt = parse_arg_type (fmt, &spec);
      if (spec.arg_type == UNSUPPORTED)
        /* the current architecture doesn't support this type */
        {
          goto error;
        }
      else if (spec.arg_type == MPFR_ARG)
        {
          switch (*fmt)
            {
            case '\0':
              break;
            case '*':
              ++fmt;
              spec.rnd_mode = (mpfr_rnd_t) va_arg (ap, int);
              break;
            case 'D':
              ++fmt;
              spec.rnd_mode = MPFR_RNDD;
              break;
            case 'U':
              ++fmt;
              spec.rnd_mode = MPFR_RNDU;
              break;
            case 'Y':
              ++fmt;
              spec.rnd_mode = MPFR_RNDA;
              break;
            case 'Z':
              ++fmt;
              spec.rnd_mode = MPFR_RNDZ;
              break;
            case 'N':
              ++fmt;
            default:
              spec.rnd_mode = MPFR_RNDN;
            }
        }

      spec.spec = *fmt;
      if (!specinfo_is_valid (spec))
        goto error;

      if (*fmt)
        fmt++;

      /* Format processing */
      if (spec.spec == '\0')
        /* end of the format string */
        break;
      else if (spec.spec == 'n')
        /* put the number of characters written so far in the location pointed
           by the next va_list argument; the types of pointer accepted are the
           same as in GMP (except unsupported quad_t) plus pointer to a mpfr_t
           so as to be able to accept the same format strings. */
        {
          void *p;
          size_t nchar;

          p = va_arg (ap, void *);
          FLUSH (xgmp_fmt_flag, start, end, ap2, &buf);
          va_end (ap2);
          start = fmt;
          nchar = buf.curr - buf.start;

          switch (spec.arg_type)
            {
            case CHAR_ARG:
              *(char *) p = (char) nchar;
              break;
            case SHORT_ARG:
              *(short *) p = (short) nchar;
              break;
            case LONG_ARG:
              *(long *) p = (long) nchar;
              break;
#ifdef HAVE_LONG_LONG
            case LONG_LONG_ARG:
              *(long long *) p = (long long) nchar;
              break;
#endif
#ifdef _MPFR_H_HAVE_INTMAX_T
            case INTMAX_ARG:
              *(intmax_t *) p = (intmax_t) nchar;
              break;
#endif
            case SIZE_ARG:
              *(size_t *) p = nchar;
              break;
            case PTRDIFF_ARG:
              *(ptrdiff_t *) p = (ptrdiff_t) nchar;
              break;
            case MPF_ARG:
              mpf_set_ui ((mpf_ptr) p, (unsigned long) nchar);
              break;
            case MPQ_ARG:
              mpq_set_ui ((mpq_ptr) p, (unsigned long) nchar, 1L);
              break;
            case MP_LIMB_ARG:
              *(mp_limb_t *) p = (mp_limb_t) nchar;
              break;
            case MP_LIMB_ARRAY_ARG:
              {
                mp_limb_t *q = (mp_limb_t *) p;
                mp_size_t n;
                n = va_arg (ap, mp_size_t);
                if (n < 0)
                  n = -n;
                else if (n == 0)
                  break;

                /* we assume here that mp_limb_t is wider than int */
                *q = (mp_limb_t) nchar;
                while (--n != 0)
                  {
                    q++;
                    *q = (mp_limb_t) 0;
                  }
              }
              break;
            case MPZ_ARG:
              mpz_set_ui ((mpz_ptr) p, (unsigned long) nchar);
              break;

            case MPFR_ARG:
              mpfr_set_ui ((mpfr_ptr) p, (unsigned long) nchar,
                           spec.rnd_mode);
              break;

            default:
              *(int *) p = (int) nchar;
            }
          va_copy (ap2, ap); /* after the switch, due to MP_LIMB_ARRAY_ARG
                                case */
        }
      else if (spec.arg_type == MPFR_PREC_ARG)
        /* output mpfr_prec_t variable */
        {
          char *s;
          char format[MPFR_PREC_FORMAT_SIZE + 6]; /* see examples below */
          size_t length;
          mpfr_prec_t prec;
          prec = va_arg (ap, mpfr_prec_t);

          FLUSH (xgmp_fmt_flag, start, end, ap2, &buf);
          va_end (ap2);
          va_copy (ap2, ap);
          start = fmt;

          /* construct format string, like "%*.*hd" "%*.*d" or "%*.*ld" */
          format[0] = '%';
          format[1] = '*';
          format[2] = '.';
          format[3] = '*';
          format[4] = '\0';
          strcat (format, MPFR_PREC_FORMAT_TYPE);
          format[4 + MPFR_PREC_FORMAT_SIZE] = spec.spec;
          format[5 + MPFR_PREC_FORMAT_SIZE] = '\0';
          length = gmp_asprintf (&s, format, spec.width, spec.prec, prec);
          if (buf.size <= INT_MAX - length)
            {
              buffer_cat (&buf, s, length);
              mpfr_free_str (s);
            }
          else
            {
              mpfr_free_str (s);
              goto overflow_error;
            }
        }
      else if (spec.arg_type == MPFR_ARG)
        /* output a mpfr_t variable */
        {
          mpfr_srcptr p;

          p = va_arg (ap, mpfr_srcptr);

          FLUSH (xgmp_fmt_flag, start, end, ap2, &buf);
          va_end (ap2);
          va_copy (ap2, ap);
          start = fmt;

          switch (spec.spec)
            {
            case 'a':
            case 'A':
            case 'b':
            case 'e':
            case 'E':
            case 'f':
            case 'F':
            case 'g':
            case 'G':
              if (sprnt_fp (&buf, p, spec) < 0)
                goto overflow_error;
              break;

            default:
              /* unsupported specifier */
              goto error;
            }
        }
      else
        /* gmp_printf specification, step forward in the va_list */
        {
          CONSUME_VA_ARG (spec, ap);
          xgmp_fmt_flag = 1;
        }
    }

  if (start != fmt)
    FLUSH (xgmp_fmt_flag, start, fmt, ap2, &buf);

  va_end (ap2);
  nbchar = buf.curr - buf.start;
  MPFR_ASSERTD (nbchar == strlen (buf.start));
  buf.start =
    (char *) (*__gmp_reallocate_func) (buf.start, buf.size, nbchar + 1);
  buf.size = nbchar + 1; /* update needed for __gmp_free_func below when
                            nbchar is too large (overflow_error) */
  *ptr = buf.start;

  /* If nbchar is larger than INT_MAX, the ISO C99 standard is silent, but
     POSIX says concerning the snprintf() function:
     "[EOVERFLOW] The value of n is greater than {INT_MAX} or the
     number of bytes needed to hold the output excluding the
     terminating null is greater than {INT_MAX}." See:
     http://www.opengroup.org/onlinepubs/009695399/functions/fprintf.html
     But it doesn't say anything concerning the other printf-like functions.
     A defect report has been submitted to austin-review-l (item 2532).
     So, for the time being, we return a negative value and set the erange
     flag, and set errno to EOVERFLOW in POSIX system. */
  if (nbchar <= INT_MAX)
    {
      MPFR_SAVE_EXPO_FREE (expo);
      return nbchar;
    }

 overflow_error:
  MPFR_SAVE_EXPO_UPDATE_FLAGS(expo, MPFR_FLAGS_ERANGE);
#ifdef EOVERFLOW
  errno = EOVERFLOW;
#endif

 error:
  MPFR_SAVE_EXPO_FREE (expo);
  *ptr = NULL;
  (*__gmp_free_func) (buf.start, buf.size);

  return -1;
}

#endif /* HAVE_STDARG */
