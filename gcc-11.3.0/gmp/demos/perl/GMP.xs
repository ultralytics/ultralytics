/* GMP module external subroutines.

Copyright 2001-2003, 2015 Free Software Foundation, Inc.

This file is part of the GNU MP Library.

The GNU MP Library is free software; you can redistribute it and/or modify
it under the terms of either:

  * the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your
    option) any later version.

or

  * the GNU General Public License as published by the Free Software
    Foundation; either version 2 of the License, or (at your option) any
    later version.

or both in parallel, as here.

The GNU MP Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received copies of the GNU General Public License and the
GNU Lesser General Public License along with the GNU MP Library.  If not,
see https://www.gnu.org/licenses/.


/* Notes:

   Routines are grouped with the alias feature and a table of function
   pointers where possible, since each xsub routine ends up with quite a bit
   of code size.  Different combinations of arguments and return values have
   to be separate though.

   The "INTERFACE:" feature isn't available in perl 5.005 and so isn't used.
   "ALIAS:" requires a table lookup with CvXSUBANY(cv).any_i32 (which is
   "ix") whereas "INTERFACE:" would have CvXSUBANY(cv).any_dptr as the
   function pointer immediately.

   Mixed-type swapped-order assignments like "$a = 123; $a += mpz(456);"
   invoke the plain overloaded "+", not "+=", which makes life easier.

   mpz_assume etc types are used with the overloaded operators since such
   operators are always called with a class object as the first argument, we
   don't need an sv_derived_from() lookup to check.  There's assert()s in
   MPX_ASSUME() for this though.

   The overload_constant routines reached via overload::constant get 4
   arguments in perl 5.6, not the 3 as documented.  This is apparently a
   bug, using "..." lets us ignore the extra one.

   There's only a few "si" functions in gmp, so usually SvIV values get
   handled with an mpz_set_si into a temporary and then a full precision mpz
   routine.  This is reasonably efficient.

   Argument types are checked, with a view to preserving all bits in the
   operand.  Perl is a bit looser in its arithmetic, allowing rounding or
   truncation to an intended operand type (IV, UV or NV).

   Bugs:

   The memory leak detection attempted in GMP::END() doesn't work when mpz's
   are created as constants because END() is called before they're
   destroyed.  What's the right place to hook such a check?

   See the bugs section of GMP.pm too.  */


/* Comment this out to get assertion checking. */
#define NDEBUG

/* Change this to "#define TRACE(x) x" for some diagnostics. */
#define TRACE(x)


#include <assert.h>
#include <float.h>

#include "EXTERN.h"
#include "perl.h"
#include "XSUB.h"
#include "patchlevel.h"

#include "gmp.h"


/* Perl 5.005 doesn't have SvIsUV, only 5.6 and up.
   Perl 5.8 has SvUOK, but not 5.6, so we don't use that.  */
#ifndef SvIsUV
#define SvIsUV(sv)  0
#endif
#ifndef SvUVX
#define SvUVX(sv)  (croak("GMP: oops, shouldn't be using SvUVX"), 0)
#endif


/* Code which doesn't check anything itself, but exists to support other
   assert()s.  */
#ifdef NDEBUG
#define assert_support(x)
#else
#define assert_support(x) x
#endif

/* LONG_MAX + 1 and ULONG_MAX + 1, as a doubles */
#define LONG_MAX_P1_AS_DOUBLE   ((double) ((unsigned long) LONG_MAX + 1))
#define ULONG_MAX_P1_AS_DOUBLE  (2.0 * (double) ((unsigned long) ULONG_MAX/2 + 1))

/* Check for perl version "major.minor".
   Perl 5.004 doesn't have PERL_REVISION and PERL_VERSION, but that's ok,
   we're only interested in tests above that.  */
#if defined (PERL_REVISION) && defined (PERL_VERSION)
#define PERL_GE(major,minor)                                    \
    (PERL_REVISION > (major)                                    \
     || ((major) == PERL_REVISION && PERL_VERSION >= (minor)))
#else
#define PERL_GE(major,minor)  (0)
#endif
#define PERL_LT(major,minor)  (! PERL_GE(major,minor))

/* sv_derived_from etc in 5.005 took "char *" rather than "const char *".
   Avoid some compiler warnings by using const only where it works.  */
#if PERL_LT (5,6)
#define classconst
#else
#define classconst const
#endif

/* In a MINGW or Cygwin DLL build of gmp, the various gmp functions are
   given with dllimport directives, which prevents them being used as
   initializers for constant data.  We give function tables as
   "static_functable const ...", which is normally "static const", but for
   mingw expands to just "const" making the table an automatic with a
   run-time initializer.

   In gcc 3.3.1, the function tables initialized like this end up getting
   all the __imp__foo values fetched, even though just one or two will be
   used.  This is wasteful, but probably not too bad.  */

#if defined (__MINGW32__) || defined (__CYGWIN__)
#define static_functable
#else
#define static_functable  static
#endif

#define GMP_MALLOC_ID  42

static classconst char mpz_class[]  = "GMP::Mpz";
static classconst char mpq_class[]  = "GMP::Mpq";
static classconst char mpf_class[]  = "GMP::Mpf";
static classconst char rand_class[] = "GMP::Rand";

static HV *mpz_class_hv;
static HV *mpq_class_hv;
static HV *mpf_class_hv;

assert_support (static long mpz_count = 0;)
assert_support (static long mpq_count = 0;)
assert_support (static long mpf_count = 0;)
assert_support (static long rand_count = 0;)

#define TRACE_ACTIVE()                                                   \
  assert_support                                                         \
  (TRACE (printf ("  active %ld mpz, %ld mpq, %ld mpf, %ld randstate\n", \
                  mpz_count, mpq_count, mpf_count, rand_count)))


/* Each "struct mpz_elem" etc is an mpz_t with a link field tacked on the
   end so they can be held on a linked list.  */

#define CREATE_MPX(type)                                \
                                                        \
  /* must have mpz_t etc first, for sprintf below */    \
  struct type##_elem {                                  \
    type##_t            m;                              \
    struct type##_elem  *next;                          \
  };                                                    \
  typedef struct type##_elem  *type;                    \
  typedef struct type##_elem  *type##_assume;           \
  typedef type##_ptr          type##_coerce;            \
                                                        \
  static type type##_freelist = NULL;                   \
                                                        \
  static type                                           \
  new_##type (void)                                     \
  {                                                     \
    type p;                                             \
    TRACE (printf ("new %s\n", type##_class));          \
    if (type##_freelist != NULL)                        \
      {                                                 \
        p = type##_freelist;                            \
        type##_freelist = type##_freelist->next;        \
      }                                                 \
    else                                                \
      {                                                 \
        New (GMP_MALLOC_ID, p, 1, struct type##_elem);  \
        type##_init (p->m);                             \
      }                                                 \
    TRACE (printf ("  p=%p\n", p));                     \
    assert_support (type##_count++);                    \
    TRACE_ACTIVE ();                                    \
    return p;                                           \
  }                                                     \

CREATE_MPX (mpz)
CREATE_MPX (mpq)

typedef mpf_ptr  mpf;
typedef mpf_ptr  mpf_assume;
typedef mpf_ptr  mpf_coerce_st0;
typedef mpf_ptr  mpf_coerce_def;


static mpf
new_mpf (unsigned long prec)
{
  mpf p;
  New (GMP_MALLOC_ID, p, 1, __mpf_struct);
  mpf_init2 (p, prec);
  TRACE (printf ("  mpf p=%p\n", p));
  assert_support (mpf_count++);
  TRACE_ACTIVE ();
  return p;
}


/* tmp_mpf_t records an allocated precision with an mpf_t so changes of
   precision can be done with just an mpf_set_prec_raw.  */

struct tmp_mpf_struct {
  mpf_t          m;
  unsigned long  allocated_prec;
};
typedef const struct tmp_mpf_struct  *tmp_mpf_srcptr;
typedef struct tmp_mpf_struct        *tmp_mpf_ptr;
typedef struct tmp_mpf_struct        tmp_mpf_t[1];

#define tmp_mpf_init(f)                         \
  do {                                          \
    mpf_init (f->m);                            \
    f->allocated_prec = mpf_get_prec (f->m);    \
  } while (0)

static void
tmp_mpf_grow (tmp_mpf_ptr f, unsigned long prec)
{
  mpf_set_prec_raw (f->m, f->allocated_prec);
  mpf_set_prec (f->m, prec);
  f->allocated_prec = mpf_get_prec (f->m);
}

#define tmp_mpf_shrink(f)  tmp_mpf_grow (f, 1L)

#define tmp_mpf_set_prec(f,prec)        \
  do {                                  \
    if (prec > f->allocated_prec)       \
      tmp_mpf_grow (f, prec);           \
    else                                \
      mpf_set_prec_raw (f->m, prec);    \
  } while (0)


static mpz_t  tmp_mpz_0, tmp_mpz_1, tmp_mpz_2;
static mpq_t  tmp_mpq_0, tmp_mpq_1;
static tmp_mpf_t tmp_mpf_0, tmp_mpf_1;

/* for GMP::Mpz::export */
#define tmp_mpz_4  tmp_mpz_2


#define FREE_MPX_FREELIST(p,type)               \
  do {                                          \
    TRACE (printf ("free %s\n", type##_class)); \
    p->next = type##_freelist;                  \
    type##_freelist = p;                        \
    assert_support (type##_count--);            \
    TRACE_ACTIVE ();                            \
    assert (type##_count >= 0);                 \
  } while (0)

/* this version for comparison, if desired */
#define FREE_MPX_NOFREELIST(p,type)             \
  do {                                          \
    TRACE (printf ("free %s\n", type##_class)); \
    type##_clear (p->m);                        \
    Safefree (p);                               \
    assert_support (type##_count--);            \
    TRACE_ACTIVE ();                            \
    assert (type##_count >= 0);                 \
  } while (0)

#define free_mpz(z)    FREE_MPX_FREELIST (z, mpz)
#define free_mpq(q)    FREE_MPX_FREELIST (q, mpq)


/* Return a new mortal SV holding the given mpx_ptr pointer.
   class_hv should be one of mpz_class_hv etc.  */
#define MPX_NEWMORTAL(mpx_ptr, class_hv)                                \
    sv_bless (sv_setref_pv (sv_newmortal(), NULL, mpx_ptr), class_hv)

/* Aliases for use in typemaps */
typedef char           *malloced_string;
typedef const char     *const_string;
typedef const char     *const_string_assume;
typedef char           *string;
typedef SV             *order_noswap;
typedef SV             *dummy;
typedef SV             *SV_copy_0;
typedef unsigned long  ulong_coerce;
typedef __gmp_randstate_struct *randstate;
typedef UV             gmp_UV;

#define SvMPX(s,type)  ((type) SvIV((SV*) SvRV(s)))
#define SvMPZ(s)       SvMPX(s,mpz)
#define SvMPQ(s)       SvMPX(s,mpq)
#define SvMPF(s)       SvMPX(s,mpf)
#define SvRANDSTATE(s) SvMPX(s,randstate)

#define MPX_ASSUME(x,sv,type)                           \
  do {                                                  \
    assert (sv_derived_from (sv, type##_class));        \
    x = SvMPX(sv,type);                                 \
  } while (0)

#define MPZ_ASSUME(z,sv)    MPX_ASSUME(z,sv,mpz)
#define MPQ_ASSUME(q,sv)    MPX_ASSUME(q,sv,mpq)
#define MPF_ASSUME(f,sv)    MPX_ASSUME(f,sv,mpf)

#define numberof(x)  (sizeof (x) / sizeof ((x)[0]))
#define SGN(x)       ((x)<0 ? -1 : (x) != 0)
#define ABS(x)       ((x)>=0 ? (x) : -(x))
#define double_integer_p(d)  (floor (d) == (d))

#define x_mpq_integer_p(q) \
  (mpz_cmp_ui (mpq_denref(q), 1L) == 0)

#define assert_table(ix)  assert (ix >= 0 && ix < numberof (table))

#define SV_PTR_SWAP(x,y) \
  do { SV *__tmp = (x); (x) = (y); (y) = __tmp; } while (0)
#define MPF_PTR_SWAP(x,y) \
  do { mpf_ptr __tmp = (x); (x) = (y); (y) = __tmp; } while (0)


static void
class_or_croak (SV *sv, classconst char *cl)
{
  if (! sv_derived_from (sv, cl))
    croak("not type %s", cl);
}


/* These are macros, wrap them in functions. */
static int
x_mpz_odd_p (mpz_srcptr z)
{
  return mpz_odd_p (z);
}
static int
x_mpz_even_p (mpz_srcptr z)
{
  return mpz_even_p (z);
}

static void
x_mpq_pow_ui (mpq_ptr r, mpq_srcptr b, unsigned long e)
{
  mpz_pow_ui (mpq_numref(r), mpq_numref(b), e);
  mpz_pow_ui (mpq_denref(r), mpq_denref(b), e);
}


static void *
my_gmp_alloc (size_t n)
{
  void *p;
  TRACE (printf ("my_gmp_alloc %u\n", n));
  New (GMP_MALLOC_ID, p, n, char);
  TRACE (printf ("  p=%p\n", p));
  return p;
}

static void *
my_gmp_realloc (void *p, size_t oldsize, size_t newsize)
{
  TRACE (printf ("my_gmp_realloc %p, %u to %u\n", p, oldsize, newsize));
  Renew (p, newsize, char);
  TRACE (printf ("  p=%p\n", p));
  return p;
}

static void
my_gmp_free (void *p, size_t n)
{
  TRACE (printf ("my_gmp_free %p %u\n", p, n));
  Safefree (p);
}


#define my_mpx_set_svstr(type)                                  \
  static void                                                   \
  my_##type##_set_svstr (type##_ptr x, SV *sv)                  \
  {                                                             \
    const char  *str;                                           \
    STRLEN      len;                                            \
    TRACE (printf ("  my_" #type "_set_svstr\n"));              \
    assert (SvPOK(sv) || SvPOKp(sv));                           \
    str = SvPV (sv, len);                                       \
    TRACE (printf ("  str \"%s\"\n", str));                     \
    if (type##_set_str (x, str, 0) != 0)                        \
      croak ("%s: invalid string: %s", type##_class, str);      \
  }

my_mpx_set_svstr(mpz)
my_mpx_set_svstr(mpq)
my_mpx_set_svstr(mpf)


/* very slack */
static int
x_mpq_cmp_si (mpq_srcptr x, long yn, unsigned long yd)
{
  mpq  y;
  int  ret;
  y = new_mpq ();
  mpq_set_si (y->m, yn, yd);
  ret = mpq_cmp (x, y->m);
  free_mpq (y);
  return ret;
}

static int
x_mpq_fits_slong_p (mpq_srcptr q)
{
  return x_mpq_cmp_si (q, LONG_MIN, 1L) >= 0
    && mpq_cmp_ui (q, LONG_MAX, 1L) <= 0;
}

static int
x_mpz_cmp_q (mpz_ptr x, mpq_srcptr y)
{
  int  ret;
  mpz_set_ui (mpq_denref(tmp_mpq_0), 1L);
  mpz_swap (mpq_numref(tmp_mpq_0), x);
  ret = mpq_cmp (tmp_mpq_0, y);
  mpz_swap (mpq_numref(tmp_mpq_0), x);
  return ret;
}

static int
x_mpz_cmp_f (mpz_srcptr x, mpf_srcptr y)
{
  tmp_mpf_set_prec (tmp_mpf_0, mpz_sizeinbase (x, 2));
  mpf_set_z (tmp_mpf_0->m, x);
  return mpf_cmp (tmp_mpf_0->m, y);
}


#define USE_UNKNOWN  0
#define USE_IVX      1
#define USE_UVX      2
#define USE_NVX      3
#define USE_PVX      4
#define USE_MPZ      5
#define USE_MPQ      6
#define USE_MPF      7

/* mg_get is called every time we get a value, even if the private flags are
   still set from a previous such call.  This is the same as as SvIV and
   friends do.

   When POK, we use the PV, even if there's an IV or NV available.  This is
   because it's hard to be sure there wasn't any rounding in establishing
   the IV and/or NV.  Cases of overflow, where the PV should definitely be
   used, are easy enough to spot, but rounding is hard.  So although IV or
   NV would be more efficient, we must use the PV to be sure of getting all
   the data.  Applications should convert once to mpz, mpq or mpf when using
   a value repeatedly.

   Zany dual-type scalars like $! where the IV is an error code and the PV
   is an error description string won't work with this preference for PV,
   but that's too bad.  Such scalars should be rare, and unlikely to be used
   in bignum calculations.

   When IOK and NOK are both set, we would prefer to use the IV since it can
   be converted more efficiently, and because on a 64-bit system the NV may
   have less bits than the IV.  The following rules are applied,

   - If the NV is not an integer, then we must use that NV, since clearly
     the IV was merely established by rounding and is not the full value.

   - In perl prior to 5.8, an NV too big for an IV leaves an overflow value
     0xFFFFFFFF.  If the NV is too big to fit an IV then clearly it's the NV
     which is the true value and must be used.

   - In perl 5.8 and up, such an overflow doesn't set IOK, so that test is
     unnecessary.  However when coming from get-magic, IOKp _is_ set, and we
     must check for overflow the same as in older perl.

   FIXME:

   We'd like to call mg_get just once, but unfortunately sv_derived_from()
   will call it for each of our checks.  We could do a string compare like
   sv_isa ourselves, but that only tests the exact class, it doesn't
   recognise subclassing.  There doesn't seem to be a public interface to
   the subclassing tests (in the internal isa_lookup() function).  */

int
use_sv (SV *sv)
{
  double  d;

  if (SvGMAGICAL(sv))
    {
      mg_get(sv);

      if (SvPOKp(sv))
        return USE_PVX;

      if (SvIOKp(sv))
        {
          if (SvIsUV(sv))
            {
              if (SvNOKp(sv))
                goto u_or_n;
              return USE_UVX;
            }
          else
            {
              if (SvNOKp(sv))
                goto i_or_n;
              return USE_IVX;
            }
        }

      if (SvNOKp(sv))
        return USE_NVX;

      goto rok_or_unknown;
    }

  if (SvPOK(sv))
    return USE_PVX;

  if (SvIOK(sv))
    {
      if (SvIsUV(sv))
        {
          if (SvNOK(sv))
            {
              if (PERL_LT (5, 8))
                {
                u_or_n:
                  d = SvNVX(sv);
                  if (d >= ULONG_MAX_P1_AS_DOUBLE || d < 0.0)
                    return USE_NVX;
                }
              d = SvNVX(sv);
              if (d != floor (d))
                return USE_NVX;
            }
          return USE_UVX;
        }
      else
        {
          if (SvNOK(sv))
            {
              if (PERL_LT (5, 8))
                {
                i_or_n:
                  d = SvNVX(sv);
                  if (d >= LONG_MAX_P1_AS_DOUBLE || d < (double) LONG_MIN)
                    return USE_NVX;
                }
              d = SvNVX(sv);
              if (d != floor (d))
                return USE_NVX;
            }
          return USE_IVX;
        }
    }

  if (SvNOK(sv))
    return USE_NVX;

 rok_or_unknown:
  if (SvROK(sv))
    {
      if (sv_derived_from (sv, mpz_class))
        return USE_MPZ;
      if (sv_derived_from (sv, mpq_class))
        return USE_MPQ;
      if (sv_derived_from (sv, mpf_class))
        return USE_MPF;
    }

  return USE_UNKNOWN;
}


/* Coerce sv to an mpz.  Use tmp to hold the converted value if sv isn't
   already an mpz (or an mpq of which the numerator can be used).  Return
   the chosen mpz (tmp or the contents of sv).  */

static mpz_ptr
coerce_mpz_using (mpz_ptr tmp, SV *sv, int use)
{
  switch (use) {
  case USE_IVX:
    mpz_set_si (tmp, SvIVX(sv));
    return tmp;

  case USE_UVX:
    mpz_set_ui (tmp, SvUVX(sv));
    return tmp;

  case USE_NVX:
    {
      double d;
      d = SvNVX(sv);
      if (! double_integer_p (d))
        croak ("cannot coerce non-integer double to mpz");
      mpz_set_d (tmp, d);
      return tmp;
    }

  case USE_PVX:
    my_mpz_set_svstr (tmp, sv);
    return tmp;

  case USE_MPZ:
    return SvMPZ(sv)->m;

  case USE_MPQ:
    {
      mpq q = SvMPQ(sv);
      if (! x_mpq_integer_p (q->m))
        croak ("cannot coerce non-integer mpq to mpz");
      return mpq_numref(q->m);
    }

  case USE_MPF:
    {
      mpf f = SvMPF(sv);
      if (! mpf_integer_p (f))
        croak ("cannot coerce non-integer mpf to mpz");
      mpz_set_f (tmp, f);
      return tmp;
    }

  default:
    croak ("cannot coerce to mpz");
  }
}
static mpz_ptr
coerce_mpz (mpz_ptr tmp, SV *sv)
{
  return coerce_mpz_using (tmp, sv, use_sv (sv));
}


/* Coerce sv to an mpq.  If sv is an mpq then just return that, otherwise
   use tmp to hold the converted value and return that.  */

static mpq_ptr
coerce_mpq_using (mpq_ptr tmp, SV *sv, int use)
{
  TRACE (printf ("coerce_mpq_using %p %d\n", tmp, use));
  switch (use) {
  case USE_IVX:
    mpq_set_si (tmp, SvIVX(sv), 1L);
    return tmp;

  case USE_UVX:
    mpq_set_ui (tmp, SvUVX(sv), 1L);
    return tmp;

  case USE_NVX:
    mpq_set_d (tmp, SvNVX(sv));
    return tmp;

  case USE_PVX:
    my_mpq_set_svstr (tmp, sv);
    return tmp;

  case USE_MPZ:
    mpq_set_z (tmp, SvMPZ(sv)->m);
    return tmp;

  case USE_MPQ:
    return SvMPQ(sv)->m;

  case USE_MPF:
    mpq_set_f (tmp, SvMPF(sv));
    return tmp;

  default:
    croak ("cannot coerce to mpq");
  }
}
static mpq_ptr
coerce_mpq (mpq_ptr tmp, SV *sv)
{
  return coerce_mpq_using (tmp, sv, use_sv (sv));
}


static void
my_mpf_set_sv_using (mpf_ptr f, SV *sv, int use)
{
  switch (use) {
  case USE_IVX:
    mpf_set_si (f, SvIVX(sv));
    break;

  case USE_UVX:
    mpf_set_ui (f, SvUVX(sv));
    break;

  case USE_NVX:
    mpf_set_d (f, SvNVX(sv));
    break;

  case USE_PVX:
    my_mpf_set_svstr (f, sv);
    break;

  case USE_MPZ:
    mpf_set_z (f, SvMPZ(sv)->m);
    break;

  case USE_MPQ:
    mpf_set_q (f, SvMPQ(sv)->m);
    break;

  case USE_MPF:
    mpf_set (f, SvMPF(sv));
    break;

  default:
    croak ("cannot coerce to mpf");
  }
}

/* Coerce sv to an mpf.  If sv is an mpf then just return that, otherwise
   use tmp to hold the converted value (with prec precision).  */
static mpf_ptr
coerce_mpf_using (tmp_mpf_ptr tmp, SV *sv, unsigned long prec, int use)
{
  if (use == USE_MPF)
    return SvMPF(sv);

  tmp_mpf_set_prec (tmp, prec);
  my_mpf_set_sv_using (tmp->m, sv, use);
  return tmp->m;
}
static mpf_ptr
coerce_mpf (tmp_mpf_ptr tmp, SV *sv, unsigned long prec)
{
  return coerce_mpf_using (tmp, sv, prec, use_sv (sv));
}


/* Coerce xv to an mpf and store the pointer in x, ditto for yv to x.  If
   one of xv or yv is an mpf then use it for the precision, otherwise use
   the default precision.  */
unsigned long
coerce_mpf_pair (mpf *xp, SV *xv, mpf *yp, SV *yv)
{
  int x_use = use_sv (xv);
  int y_use = use_sv (yv);
  unsigned long  prec;
  mpf  x, y;

  if (x_use == USE_MPF)
    {
      x = SvMPF(xv);
      prec = mpf_get_prec (x);
      y = coerce_mpf_using (tmp_mpf_0, yv, prec, y_use);
    }
  else
    {
      y = coerce_mpf_using (tmp_mpf_0, yv, mpf_get_default_prec(), y_use);
      prec = mpf_get_prec (y);
      x = coerce_mpf_using (tmp_mpf_1, xv, prec, x_use);
    }
  *xp = x;
  *yp = y;
  return prec;
}


/* Note that SvUV is not used, since it merely treats the signed IV as if it
   was unsigned.  We get an IV and check its sign. */
static unsigned long
coerce_ulong (SV *sv)
{
  long  n;

  switch (use_sv (sv)) {
  case USE_IVX:
    n = SvIVX(sv);
  negative_check:
    if (n < 0)
      goto range_error;
    return n;

  case USE_UVX:
    return SvUVX(sv);

  case USE_NVX:
    {
      double d;
      d = SvNVX(sv);
      if (! double_integer_p (d))
        goto integer_error;
      n = SvIV(sv);
    }
    goto negative_check;

  case USE_PVX:
    /* FIXME: Check the string is an integer. */
    n = SvIV(sv);
    goto negative_check;

  case USE_MPZ:
    {
      mpz z = SvMPZ(sv);
      if (! mpz_fits_ulong_p (z->m))
        goto range_error;
      return mpz_get_ui (z->m);
    }

  case USE_MPQ:
    {
      mpq q = SvMPQ(sv);
      if (! x_mpq_integer_p (q->m))
        goto integer_error;
      if (! mpz_fits_ulong_p (mpq_numref (q->m)))
        goto range_error;
      return mpz_get_ui (mpq_numref (q->m));
    }

  case USE_MPF:
    {
      mpf f = SvMPF(sv);
      if (! mpf_integer_p (f))
        goto integer_error;
      if (! mpf_fits_ulong_p (f))
        goto range_error;
      return mpf_get_ui (f);
    }

  default:
    croak ("cannot coerce to ulong");
  }

 integer_error:
  croak ("not an integer");

 range_error:
  croak ("out of range for ulong");
}


static long
coerce_long (SV *sv)
{
  switch (use_sv (sv)) {
  case USE_IVX:
    return SvIVX(sv);

  case USE_UVX:
    {
      UV u = SvUVX(sv);
      if (u > (UV) LONG_MAX)
        goto range_error;
      return u;
    }

  case USE_NVX:
    {
      double d = SvNVX(sv);
      if (! double_integer_p (d))
        goto integer_error;
      return SvIV(sv);
    }

  case USE_PVX:
    /* FIXME: Check the string is an integer. */
    return SvIV(sv);

  case USE_MPZ:
    {
      mpz z = SvMPZ(sv);
      if (! mpz_fits_slong_p (z->m))
        goto range_error;
      return mpz_get_si (z->m);
    }

  case USE_MPQ:
    {
      mpq q = SvMPQ(sv);
      if (! x_mpq_integer_p (q->m))
        goto integer_error;
      if (! mpz_fits_slong_p (mpq_numref (q->m)))
        goto range_error;
      return mpz_get_si (mpq_numref (q->m));
    }

  case USE_MPF:
    {
      mpf f = SvMPF(sv);
      if (! mpf_integer_p (f))
        goto integer_error;
      if (! mpf_fits_slong_p (f))
        goto range_error;
      return mpf_get_si (f);
    }

  default:
    croak ("cannot coerce to long");
  }

 integer_error:
  croak ("not an integer");

 range_error:
  croak ("out of range for ulong");
}


/* ------------------------------------------------------------------------- */

MODULE = GMP         PACKAGE = GMP

BOOT:
    TRACE (printf ("GMP boot\n"));
    mp_set_memory_functions (my_gmp_alloc, my_gmp_realloc, my_gmp_free);
    mpz_init (tmp_mpz_0);
    mpz_init (tmp_mpz_1);
    mpz_init (tmp_mpz_2);
    mpq_init (tmp_mpq_0);
    mpq_init (tmp_mpq_1);
    tmp_mpf_init (tmp_mpf_0);
    tmp_mpf_init (tmp_mpf_1);
    mpz_class_hv = gv_stashpv (mpz_class, 1);
    mpq_class_hv = gv_stashpv (mpq_class, 1);
    mpf_class_hv = gv_stashpv (mpf_class, 1);


void
END()
CODE:
    TRACE (printf ("GMP end\n"));
    TRACE_ACTIVE ();
    /* These are not always true, see Bugs at the top of the file. */
    /* assert (mpz_count == 0); */
    /* assert (mpq_count == 0); */
    /* assert (mpf_count == 0); */
    /* assert (rand_count == 0); */


const_string
version()
CODE:
    RETVAL = gmp_version;
OUTPUT:
    RETVAL


bool
fits_slong_p (sv)
    SV *sv
CODE:
    switch (use_sv (sv)) {
    case USE_IVX:
      RETVAL = 1;
      break;

    case USE_UVX:
      {
        UV u = SvUVX(sv);
        RETVAL = (u <= LONG_MAX);
      }
      break;

    case USE_NVX:
      {
        double  d = SvNVX(sv);
        RETVAL = (d >= (double) LONG_MIN && d < LONG_MAX_P1_AS_DOUBLE);
      }
      break;

    case USE_PVX:
      {
        STRLEN len;
        const char *str = SvPV (sv, len);
        if (mpq_set_str (tmp_mpq_0, str, 0) == 0)
          RETVAL = x_mpq_fits_slong_p (tmp_mpq_0);
        else
          {
            /* enough precision for a long */
            tmp_mpf_set_prec (tmp_mpf_0, 2*mp_bits_per_limb);
            if (mpf_set_str (tmp_mpf_0->m, str, 10) != 0)
              croak ("GMP::fits_slong_p invalid string format");
            RETVAL = mpf_fits_slong_p (tmp_mpf_0->m);
          }
      }
      break;

    case USE_MPZ:
      RETVAL = mpz_fits_slong_p (SvMPZ(sv)->m);
      break;

    case USE_MPQ:
      RETVAL = x_mpq_fits_slong_p (SvMPQ(sv)->m);
      break;

    case USE_MPF:
      RETVAL = mpf_fits_slong_p (SvMPF(sv));
      break;

    default:
      croak ("GMP::fits_slong_p invalid argument");
    }
OUTPUT:
    RETVAL


double
get_d (sv)
    SV *sv
CODE:
    switch (use_sv (sv)) {
    case USE_IVX:
      RETVAL = (double) SvIVX(sv);
      break;

    case USE_UVX:
      RETVAL = (double) SvUVX(sv);
      break;

    case USE_NVX:
      RETVAL = SvNVX(sv);
      break;

    case USE_PVX:
      {
        STRLEN len;
        RETVAL = atof(SvPV(sv, len));
      }
      break;

    case USE_MPZ:
      RETVAL = mpz_get_d (SvMPZ(sv)->m);
      break;

    case USE_MPQ:
      RETVAL = mpq_get_d (SvMPQ(sv)->m);
      break;

    case USE_MPF:
      RETVAL = mpf_get_d (SvMPF(sv));
      break;

    default:
      croak ("GMP::get_d invalid argument");
    }
OUTPUT:
    RETVAL


void
get_d_2exp (sv)
    SV *sv
PREINIT:
    double ret;
    long   exp;
PPCODE:
    switch (use_sv (sv)) {
    case USE_IVX:
      ret = (double) SvIVX(sv);
      goto use_frexp;

    case USE_UVX:
      ret = (double) SvUVX(sv);
      goto use_frexp;

    case USE_NVX:
      {
        int i_exp;
        ret = SvNVX(sv);
      use_frexp:
        ret = frexp (ret, &i_exp);
        exp = i_exp;
      }
      break;

    case USE_PVX:
      /* put strings through mpf to give full exp range */
      tmp_mpf_set_prec (tmp_mpf_0, DBL_MANT_DIG);
      my_mpf_set_svstr (tmp_mpf_0->m, sv);
      ret = mpf_get_d_2exp (&exp, tmp_mpf_0->m);
      break;

    case USE_MPZ:
      ret = mpz_get_d_2exp (&exp, SvMPZ(sv)->m);
      break;

    case USE_MPQ:
      tmp_mpf_set_prec (tmp_mpf_0, DBL_MANT_DIG);
      mpf_set_q (tmp_mpf_0->m, SvMPQ(sv)->m);
      ret = mpf_get_d_2exp (&exp, tmp_mpf_0->m);
      break;

    case USE_MPF:
      ret = mpf_get_d_2exp (&exp, SvMPF(sv));
      break;

    default:
      croak ("GMP::get_d_2exp invalid argument");
    }
    PUSHs (sv_2mortal (newSVnv (ret)));
    PUSHs (sv_2mortal (newSViv (exp)));


long
get_si (sv)
    SV *sv
CODE:
    switch (use_sv (sv)) {
    case USE_IVX:
      RETVAL = SvIVX(sv);
      break;

    case USE_UVX:
      RETVAL = SvUVX(sv);
      break;

    case USE_NVX:
      RETVAL = (long) SvNVX(sv);
      break;

    case USE_PVX:
      RETVAL = SvIV(sv);
      break;

    case USE_MPZ:
      RETVAL = mpz_get_si (SvMPZ(sv)->m);
      break;

    case USE_MPQ:
      mpz_set_q (tmp_mpz_0, SvMPQ(sv)->m);
      RETVAL = mpz_get_si (tmp_mpz_0);
      break;

    case USE_MPF:
      RETVAL = mpf_get_si (SvMPF(sv));
      break;

    default:
      croak ("GMP::get_si invalid argument");
    }
OUTPUT:
    RETVAL


void
get_str (sv, ...)
    SV *sv
PREINIT:
    char      *str;
    mp_exp_t  exp;
    mpz_ptr   z;
    mpq_ptr   q;
    mpf       f;
    int       base;
    int       ndigits;
PPCODE:
    TRACE (printf ("GMP::get_str\n"));

    if (items >= 2)
      base = coerce_long (ST(1));
    else
      base = 10;
    TRACE (printf (" base=%d\n", base));

    if (items >= 3)
      ndigits = coerce_long (ST(2));
    else
      ndigits = 10;
    TRACE (printf (" ndigits=%d\n", ndigits));

    EXTEND (SP, 2);

    switch (use_sv (sv)) {
    case USE_IVX:
      mpz_set_si (tmp_mpz_0, SvIVX(sv));
    get_tmp_mpz_0:
      z = tmp_mpz_0;
      goto get_mpz;

    case USE_UVX:
      mpz_set_ui (tmp_mpz_0, SvUVX(sv));
      goto get_tmp_mpz_0;

    case USE_NVX:
      /* only digits in the original double, not in the coerced form */
      if (ndigits == 0)
        ndigits = DBL_DIG;
      mpf_set_d (tmp_mpf_0->m, SvNVX(sv));
      f = tmp_mpf_0->m;
      goto get_mpf;

    case USE_PVX:
      {
        /* get_str on a string is not much more than a base conversion */
        STRLEN len;
        str = SvPV (sv, len);
        if (mpz_set_str (tmp_mpz_0, str, 0) == 0)
          {
            z = tmp_mpz_0;
            goto get_mpz;
          }
        else if (mpq_set_str (tmp_mpq_0, str, 0) == 0)
          {
            q = tmp_mpq_0;
            goto get_mpq;
          }
        else
          {
            /* FIXME: Would like perhaps a precision equivalent to the
               number of significant digits of the string, in its given
               base.  */
            tmp_mpf_set_prec (tmp_mpf_0, strlen(str));
            if (mpf_set_str (tmp_mpf_0->m, str, 10) == 0)
              {
                f = tmp_mpf_0->m;
                goto get_mpf;
              }
            else
              croak ("GMP::get_str invalid string format");
          }
      }
      break;

    case USE_MPZ:
      z = SvMPZ(sv)->m;
    get_mpz:
      str = mpz_get_str (NULL, base, z);
    push_str:
      PUSHs (sv_2mortal (newSVpv (str, 0)));
      break;

    case USE_MPQ:
      q = SvMPQ(sv)->m;
    get_mpq:
      str = mpq_get_str (NULL, base, q);
      goto push_str;

    case USE_MPF:
      f = SvMPF(sv);
    get_mpf:
      str = mpf_get_str (NULL, &exp, base, 0, f);
      PUSHs (sv_2mortal (newSVpv (str, 0)));
      PUSHs (sv_2mortal (newSViv (exp)));
      break;

    default:
      croak ("GMP::get_str invalid argument");
    }


bool
integer_p (sv)
    SV *sv
CODE:
    switch (use_sv (sv)) {
    case USE_IVX:
    case USE_UVX:
      RETVAL = 1;
      break;

    case USE_NVX:
      RETVAL = double_integer_p (SvNVX(sv));
      break;

    case USE_PVX:
      {
        /* FIXME: Maybe this should be done by parsing the string, not by an
           actual conversion.  */
        STRLEN len;
        const char *str = SvPV (sv, len);
        if (mpq_set_str (tmp_mpq_0, str, 0) == 0)
          RETVAL = x_mpq_integer_p (tmp_mpq_0);
        else
          {
            /* enough for all digits of the string */
            tmp_mpf_set_prec (tmp_mpf_0, strlen(str)+64);
            if (mpf_set_str (tmp_mpf_0->m, str, 10) == 0)
              RETVAL = mpf_integer_p (tmp_mpf_0->m);
            else
              croak ("GMP::integer_p invalid string format");
          }
      }
      break;

    case USE_MPZ:
      RETVAL = 1;
      break;

    case USE_MPQ:
      RETVAL = x_mpq_integer_p (SvMPQ(sv)->m);
      break;

    case USE_MPF:
      RETVAL = mpf_integer_p (SvMPF(sv));
      break;

    default:
      croak ("GMP::integer_p invalid argument");
    }
OUTPUT:
    RETVAL


int
sgn (sv)
    SV *sv
CODE:
    switch (use_sv (sv)) {
    case USE_IVX:
      RETVAL = SGN (SvIVX(sv));
      break;

    case USE_UVX:
      RETVAL = (SvUVX(sv) > 0);
      break;

    case USE_NVX:
      RETVAL = SGN (SvNVX(sv));
      break;

    case USE_PVX:
      {
        /* FIXME: Maybe this should be done by parsing the string, not by an
           actual conversion.  */
        STRLEN len;
        const char *str = SvPV (sv, len);
        if (mpq_set_str (tmp_mpq_0, str, 0) == 0)
          RETVAL = mpq_sgn (tmp_mpq_0);
        else
          {
            /* enough for all digits of the string */
            tmp_mpf_set_prec (tmp_mpf_0, strlen(str)+64);
            if (mpf_set_str (tmp_mpf_0->m, str, 10) == 0)
              RETVAL = mpf_sgn (tmp_mpf_0->m);
            else
              croak ("GMP::sgn invalid string format");
          }
      }
      break;

    case USE_MPZ:
      RETVAL = mpz_sgn (SvMPZ(sv)->m);
      break;

    case USE_MPQ:
      RETVAL = mpq_sgn (SvMPQ(sv)->m);
      break;

    case USE_MPF:
      RETVAL = mpf_sgn (SvMPF(sv));
      break;

    default:
      croak ("GMP::sgn invalid argument");
    }
OUTPUT:
    RETVAL


# currently undocumented
void
shrink ()
CODE:
#define x_mpz_shrink(z) \
    mpz_set_ui (z, 0L); _mpz_realloc (z, 1)
#define x_mpq_shrink(q) \
    x_mpz_shrink (mpq_numref(q)); x_mpz_shrink (mpq_denref(q))

    x_mpz_shrink (tmp_mpz_0);
    x_mpz_shrink (tmp_mpz_1);
    x_mpz_shrink (tmp_mpz_2);
    x_mpq_shrink (tmp_mpq_0);
    x_mpq_shrink (tmp_mpq_1);
    tmp_mpf_shrink (tmp_mpf_0);
    tmp_mpf_shrink (tmp_mpf_1);



malloced_string
sprintf_internal (fmt, sv)
    const_string fmt
    SV           *sv
CODE:
    assert (strlen (fmt) >= 3);
    assert (SvROK(sv));
    assert ((sv_derived_from (sv, mpz_class)    && fmt[strlen(fmt)-2] == 'Z')
            || (sv_derived_from (sv, mpq_class) && fmt[strlen(fmt)-2] == 'Q')
            || (sv_derived_from (sv, mpf_class) && fmt[strlen(fmt)-2] == 'F'));
    TRACE (printf ("GMP::sprintf_internal\n");
           printf ("  fmt  |%s|\n", fmt);
           printf ("  sv   |%p|\n", SvMPZ(sv)));

    /* cheat a bit here, SvMPZ works for mpq and mpf too */
    gmp_asprintf (&RETVAL, fmt, SvMPZ(sv));

    TRACE (printf ("  result |%s|\n", RETVAL));
OUTPUT:
    RETVAL



#------------------------------------------------------------------------------

MODULE = GMP         PACKAGE = GMP::Mpz

mpz
mpz (...)
ALIAS:
    GMP::Mpz::new = 1
PREINIT:
    SV *sv;
CODE:
    TRACE (printf ("%s new, ix=%ld, items=%d\n", mpz_class, ix, (int) items));
    RETVAL = new_mpz();

    switch (items) {
    case 0:
      mpz_set_ui (RETVAL->m, 0L);
      break;

    case 1:
      sv = ST(0);
      TRACE (printf ("  use %d\n", use_sv (sv)));
      switch (use_sv (sv)) {
      case USE_IVX:
        mpz_set_si (RETVAL->m, SvIVX(sv));
        break;

      case USE_UVX:
        mpz_set_ui (RETVAL->m, SvUVX(sv));
        break;

      case USE_NVX:
        mpz_set_d (RETVAL->m, SvNVX(sv));
        break;

      case USE_PVX:
        my_mpz_set_svstr (RETVAL->m, sv);
        break;

      case USE_MPZ:
        mpz_set (RETVAL->m, SvMPZ(sv)->m);
        break;

      case USE_MPQ:
        mpz_set_q (RETVAL->m, SvMPQ(sv)->m);
        break;

      case USE_MPF:
        mpz_set_f (RETVAL->m, SvMPF(sv));
        break;

      default:
        goto invalid;
      }
      break;

    default:
    invalid:
      croak ("%s new: invalid arguments", mpz_class);
    }
OUTPUT:
    RETVAL


void
overload_constant (str, pv, d1, ...)
    const_string_assume str
    SV                  *pv
    dummy               d1
PREINIT:
    mpz z;
PPCODE:
    TRACE (printf ("%s constant: %s\n", mpz_class, str));
    z = new_mpz();
    if (mpz_set_str (z->m, str, 0) == 0)
      {
        PUSHs (MPX_NEWMORTAL (z, mpz_class_hv));
      }
    else
      {
        free_mpz (z);
        PUSHs(pv);
      }


mpz
overload_copy (z, d1, d2)
    mpz_assume z
    dummy      d1
    dummy      d2
CODE:
    RETVAL = new_mpz();
    mpz_set (RETVAL->m, z->m);
OUTPUT:
    RETVAL


void
DESTROY (z)
    mpz_assume z
CODE:
    TRACE (printf ("%s DESTROY %p\n", mpz_class, z));
    free_mpz (z);


malloced_string
overload_string (z, d1, d2)
    mpz_assume z
    dummy      d1
    dummy      d2
CODE:
    TRACE (printf ("%s overload_string %p\n", mpz_class, z));
    RETVAL = mpz_get_str (NULL, 10, z->m);
OUTPUT:
    RETVAL


mpz
overload_add (xv, yv, order)
    SV *xv
    SV *yv
    SV *order
ALIAS:
    GMP::Mpz::overload_sub = 1
    GMP::Mpz::overload_mul = 2
    GMP::Mpz::overload_div = 3
    GMP::Mpz::overload_rem = 4
    GMP::Mpz::overload_and = 5
    GMP::Mpz::overload_ior = 6
    GMP::Mpz::overload_xor = 7
PREINIT:
    static_functable const struct {
      void (*op) (mpz_ptr, mpz_srcptr, mpz_srcptr);
    } table[] = {
      { mpz_add    }, /* 0 */
      { mpz_sub    }, /* 1 */
      { mpz_mul    }, /* 2 */
      { mpz_tdiv_q }, /* 3 */
      { mpz_tdiv_r }, /* 4 */
      { mpz_and    }, /* 5 */
      { mpz_ior    }, /* 6 */
      { mpz_xor    }, /* 7 */
    };
CODE:
    assert_table (ix);
    if (order == &PL_sv_yes)
      SV_PTR_SWAP (xv, yv);
    RETVAL = new_mpz();
    (*table[ix].op) (RETVAL->m,
                     coerce_mpz (tmp_mpz_0, xv),
                     coerce_mpz (tmp_mpz_1, yv));
OUTPUT:
    RETVAL


void
overload_addeq (x, y, o)
    mpz_assume   x
    mpz_coerce   y
    order_noswap o
ALIAS:
    GMP::Mpz::overload_subeq = 1
    GMP::Mpz::overload_muleq = 2
    GMP::Mpz::overload_diveq = 3
    GMP::Mpz::overload_remeq = 4
    GMP::Mpz::overload_andeq = 5
    GMP::Mpz::overload_ioreq = 6
    GMP::Mpz::overload_xoreq = 7
PREINIT:
    static_functable const struct {
      void (*op) (mpz_ptr, mpz_srcptr, mpz_srcptr);
    } table[] = {
      { mpz_add    }, /* 0 */
      { mpz_sub    }, /* 1 */
      { mpz_mul    }, /* 2 */
      { mpz_tdiv_q }, /* 3 */
      { mpz_tdiv_r }, /* 4 */
      { mpz_and    }, /* 5 */
      { mpz_ior    }, /* 6 */
      { mpz_xor    }, /* 7 */
    };
PPCODE:
    assert_table (ix);
    (*table[ix].op) (x->m, x->m, y);
    XPUSHs (ST(0));


mpz
overload_lshift (zv, nv, order)
    SV *zv
    SV *nv
    SV *order
ALIAS:
    GMP::Mpz::overload_rshift   = 1
    GMP::Mpz::overload_pow      = 2
PREINIT:
    static_functable const struct {
      void (*op) (mpz_ptr, mpz_srcptr, unsigned long);
    } table[] = {
      { mpz_mul_2exp }, /* 0 */
      { mpz_fdiv_q_2exp }, /* 1 */
      { mpz_pow_ui   }, /* 2 */
    };
CODE:
    assert_table (ix);
    if (order == &PL_sv_yes)
      SV_PTR_SWAP (zv, nv);
    RETVAL = new_mpz();
    (*table[ix].op) (RETVAL->m, coerce_mpz (RETVAL->m, zv), coerce_ulong (nv));
OUTPUT:
    RETVAL


void
overload_lshifteq (z, n, o)
    mpz_assume   z
    ulong_coerce n
    order_noswap o
ALIAS:
    GMP::Mpz::overload_rshifteq   = 1
    GMP::Mpz::overload_poweq      = 2
PREINIT:
    static_functable const struct {
      void (*op) (mpz_ptr, mpz_srcptr, unsigned long);
    } table[] = {
      { mpz_mul_2exp }, /* 0 */
      { mpz_fdiv_q_2exp }, /* 1 */
      { mpz_pow_ui   }, /* 2 */
    };
PPCODE:
    assert_table (ix);
    (*table[ix].op) (z->m, z->m, n);
    XPUSHs(ST(0));


mpz
overload_abs (z, d1, d2)
    mpz_assume z
    dummy      d1
    dummy      d2
ALIAS:
    GMP::Mpz::overload_neg  = 1
    GMP::Mpz::overload_com  = 2
    GMP::Mpz::overload_sqrt = 3
PREINIT:
    static_functable const struct {
      void (*op) (mpz_ptr w, mpz_srcptr x);
    } table[] = {
      { mpz_abs  }, /* 0 */
      { mpz_neg  }, /* 1 */
      { mpz_com  }, /* 2 */
      { mpz_sqrt }, /* 3 */
    };
CODE:
    assert_table (ix);
    RETVAL = new_mpz();
    (*table[ix].op) (RETVAL->m, z->m);
OUTPUT:
    RETVAL


void
overload_inc (z, d1, d2)
    mpz_assume z
    dummy      d1
    dummy      d2
ALIAS:
    GMP::Mpz::overload_dec = 1
PREINIT:
    static_functable const struct {
      void (*op) (mpz_ptr w, mpz_srcptr x, unsigned long y);
    } table[] = {
      { mpz_add_ui }, /* 0 */
      { mpz_sub_ui }, /* 1 */
    };
CODE:
    assert_table (ix);
    (*table[ix].op) (z->m, z->m, 1L);


int
overload_spaceship (xv, yv, order)
    SV *xv
    SV *yv
    SV *order
PREINIT:
    mpz x;
CODE:
    TRACE (printf ("%s overload_spaceship\n", mpz_class));
    MPZ_ASSUME (x, xv);
    switch (use_sv (yv)) {
    case USE_IVX:
      RETVAL = mpz_cmp_si (x->m, SvIVX(yv));
      break;
    case USE_UVX:
      RETVAL = mpz_cmp_ui (x->m, SvUVX(yv));
      break;
    case USE_PVX:
      RETVAL = mpz_cmp (x->m, coerce_mpz (tmp_mpz_0, yv));
      break;
    case USE_NVX:
      RETVAL = mpz_cmp_d (x->m, SvNVX(yv));
      break;
    case USE_MPZ:
      RETVAL = mpz_cmp (x->m, SvMPZ(yv)->m);
      break;
    case USE_MPQ:
      RETVAL = x_mpz_cmp_q (x->m, SvMPQ(yv)->m);
      break;
    case USE_MPF:
      RETVAL = x_mpz_cmp_f (x->m, SvMPF(yv));
      break;
    default:
      croak ("%s <=>: invalid operand", mpz_class);
    }
    RETVAL = SGN (RETVAL);
    if (order == &PL_sv_yes)
      RETVAL = -RETVAL;
OUTPUT:
    RETVAL


bool
overload_bool (z, d1, d2)
    mpz_assume z
    dummy      d1
    dummy      d2
ALIAS:
    GMP::Mpz::overload_not = 1
CODE:
    RETVAL = (mpz_sgn (z->m) != 0) ^ ix;
OUTPUT:
    RETVAL


mpz
bin (n, k)
    mpz_coerce   n
    ulong_coerce k
ALIAS:
    GMP::Mpz::root = 1
PREINIT:
    /* mpz_root returns an int, hence the cast */
    static_functable const struct {
      void (*op) (mpz_ptr, mpz_srcptr, unsigned long);
    } table[] = {
      {                                                mpz_bin_ui }, /* 0 */
      { (void (*)(mpz_ptr, mpz_srcptr, unsigned long)) mpz_root   }, /* 1 */
    };
CODE:
    assert_table (ix);
    RETVAL = new_mpz();
    (*table[ix].op) (RETVAL->m, n, k);
OUTPUT:
    RETVAL


void
cdiv (a, d)
    mpz_coerce a
    mpz_coerce d
ALIAS:
    GMP::Mpz::fdiv = 1
    GMP::Mpz::tdiv = 2
PREINIT:
    static_functable const struct {
      void (*op) (mpz_ptr, mpz_ptr, mpz_srcptr, mpz_srcptr);
    } table[] = {
      { mpz_cdiv_qr }, /* 0 */
      { mpz_fdiv_qr }, /* 1 */
      { mpz_tdiv_qr }, /* 2 */
    };
    mpz q, r;
PPCODE:
    assert_table (ix);
    q = new_mpz();
    r = new_mpz();
    (*table[ix].op) (q->m, r->m, a, d);
    EXTEND (SP, 2);
    PUSHs (MPX_NEWMORTAL (q, mpz_class_hv));
    PUSHs (MPX_NEWMORTAL (r, mpz_class_hv));


void
cdiv_2exp (a, d)
    mpz_coerce   a
    ulong_coerce d
ALIAS:
    GMP::Mpz::fdiv_2exp = 1
    GMP::Mpz::tdiv_2exp = 2
PREINIT:
    static_functable const struct {
      void (*q) (mpz_ptr, mpz_srcptr, unsigned long);
      void (*r) (mpz_ptr, mpz_srcptr, unsigned long);
    } table[] = {
      { mpz_cdiv_q_2exp, mpz_cdiv_r_2exp }, /* 0 */
      { mpz_fdiv_q_2exp, mpz_fdiv_r_2exp }, /* 1 */
      { mpz_tdiv_q_2exp, mpz_tdiv_r_2exp }, /* 2 */
    };
    mpz q, r;
PPCODE:
    assert_table (ix);
    q = new_mpz();
    r = new_mpz();
    (*table[ix].q) (q->m, a, d);
    (*table[ix].r) (r->m, a, d);
    EXTEND (SP, 2);
    PUSHs (MPX_NEWMORTAL (q, mpz_class_hv));
    PUSHs (MPX_NEWMORTAL (r, mpz_class_hv));


bool
congruent_p (a, c, d)
    mpz_coerce a
    mpz_coerce c
    mpz_coerce d
PREINIT:
CODE:
    RETVAL = mpz_congruent_p (a, c, d);
OUTPUT:
    RETVAL


bool
congruent_2exp_p (a, c, d)
    mpz_coerce   a
    mpz_coerce   c
    ulong_coerce d
PREINIT:
CODE:
    RETVAL = mpz_congruent_2exp_p (a, c, d);
OUTPUT:
    RETVAL


mpz
divexact (a, d)
    mpz_coerce a
    mpz_coerce d
ALIAS:
    GMP::Mpz::mod = 1
PREINIT:
    static_functable const struct {
      void (*op) (mpz_ptr, mpz_srcptr, mpz_srcptr);
    } table[] = {
      { mpz_divexact }, /* 0 */
      { mpz_mod      }, /* 1 */
    };
CODE:
    assert_table (ix);
    RETVAL = new_mpz();
    (*table[ix].op) (RETVAL->m, a, d);
OUTPUT:
    RETVAL


bool
divisible_p (a, d)
    mpz_coerce a
    mpz_coerce d
CODE:
    RETVAL = mpz_divisible_p (a, d);
OUTPUT:
    RETVAL


bool
divisible_2exp_p (a, d)
    mpz_coerce   a
    ulong_coerce d
CODE:
    RETVAL = mpz_divisible_2exp_p (a, d);
OUTPUT:
    RETVAL


bool
even_p (z)
    mpz_coerce z
ALIAS:
    GMP::Mpz::odd_p            = 1
    GMP::Mpz::perfect_square_p = 2
    GMP::Mpz::perfect_power_p  = 3
PREINIT:
    static_functable const struct {
      int (*op) (mpz_srcptr z);
    } table[] = {
      { x_mpz_even_p         }, /* 0 */
      { x_mpz_odd_p          }, /* 1 */
      { mpz_perfect_square_p }, /* 2 */
      { mpz_perfect_power_p  }, /* 3 */
    };
CODE:
    assert_table (ix);
    RETVAL = (*table[ix].op) (z);
OUTPUT:
    RETVAL


mpz
fac (n)
    ulong_coerce n
ALIAS:
    GMP::Mpz::fib    = 1
    GMP::Mpz::lucnum = 2
PREINIT:
    static_functable const struct {
      void (*op) (mpz_ptr r, unsigned long n);
    } table[] = {
      { mpz_fac_ui },    /* 0 */
      { mpz_fib_ui },    /* 1 */
      { mpz_lucnum_ui }, /* 2 */
    };
CODE:
    assert_table (ix);
    RETVAL = new_mpz();
    (*table[ix].op) (RETVAL->m, n);
OUTPUT:
    RETVAL


void
fib2 (n)
    ulong_coerce n
ALIAS:
    GMP::Mpz::lucnum2 = 1
PREINIT:
    static_functable const struct {
      void (*op) (mpz_ptr r, mpz_ptr r2, unsigned long n);
    } table[] = {
      { mpz_fib2_ui },    /* 0 */
      { mpz_lucnum2_ui }, /* 1 */
    };
    mpz  r, r2;
PPCODE:
    assert_table (ix);
    r = new_mpz();
    r2 = new_mpz();
    (*table[ix].op) (r->m, r2->m, n);
    EXTEND (SP, 2);
    PUSHs (MPX_NEWMORTAL (r,  mpz_class_hv));
    PUSHs (MPX_NEWMORTAL (r2, mpz_class_hv));


mpz
gcd (x, ...)
    mpz_coerce x
ALIAS:
    GMP::Mpz::lcm = 1
PREINIT:
    static_functable const struct {
      void (*op) (mpz_ptr w, mpz_srcptr x, mpz_srcptr y);
      void (*op_ui) (mpz_ptr w, mpz_srcptr x, unsigned long y);
    } table[] = {
      /* cast to ignore ulong return from mpz_gcd_ui */
      { mpz_gcd,
        (void (*) (mpz_ptr, mpz_srcptr, unsigned long)) mpz_gcd_ui }, /* 0 */
      { mpz_lcm, mpz_lcm_ui },                                        /* 1 */
    };
    int  i;
    SV   *yv;
CODE:
    assert_table (ix);
    RETVAL = new_mpz();
    if (items == 1)
      mpz_set (RETVAL->m, x);
    else
      {
        for (i = 1; i < items; i++)
          {
            yv = ST(i);
            if (SvIOK(yv))
              (*table[ix].op_ui) (RETVAL->m, x, ABS(SvIVX(yv)));
            else
              (*table[ix].op) (RETVAL->m, x, coerce_mpz (tmp_mpz_1, yv));
            x = RETVAL->m;
          }
      }
OUTPUT:
    RETVAL


void
gcdext (a, b)
    mpz_coerce a
    mpz_coerce b
PREINIT:
    mpz g, x, y;
    SV  *sv;
PPCODE:
    g = new_mpz();
    x = new_mpz();
    y = new_mpz();
    mpz_gcdext (g->m, x->m, y->m, a, b);
    EXTEND (SP, 3);
    PUSHs (MPX_NEWMORTAL (g, mpz_class_hv));
    PUSHs (MPX_NEWMORTAL (x, mpz_class_hv));
    PUSHs (MPX_NEWMORTAL (y, mpz_class_hv));


unsigned long
hamdist (x, y)
    mpz_coerce x
    mpz_coerce y
CODE:
    RETVAL = mpz_hamdist (x, y);
OUTPUT:
    RETVAL


mpz
invert (a, m)
    mpz_coerce a
    mpz_coerce m
CODE:
    RETVAL = new_mpz();
    if (! mpz_invert (RETVAL->m, a, m))
      {
        free_mpz (RETVAL);
        XSRETURN_UNDEF;
      }
OUTPUT:
    RETVAL


int
jacobi (a, b)
    mpz_coerce a
    mpz_coerce b
CODE:
    RETVAL = mpz_jacobi (a, b);
OUTPUT:
    RETVAL


int
kronecker (a, b)
    SV *a
    SV *b
CODE:
    if (SvIOK(b))
      RETVAL = mpz_kronecker_si (coerce_mpz(tmp_mpz_0,a), SvIVX(b));
    else if (SvIOK(a))
      RETVAL = mpz_si_kronecker (SvIVX(a), coerce_mpz(tmp_mpz_0,b));
    else
      RETVAL = mpz_kronecker (coerce_mpz(tmp_mpz_0,a),
                              coerce_mpz(tmp_mpz_1,b));
OUTPUT:
    RETVAL


void
mpz_export (order, size, endian, nails, z)
    int        order
    size_t     size
    int        endian
    size_t     nails
    mpz_coerce z
PREINIT:
    size_t  numb, count, bytes, actual_count;
    char    *data;
    SV      *sv;
PPCODE:
    numb = 8*size - nails;
    count = (mpz_sizeinbase (z, 2) + numb-1) / numb;
    bytes = count * size;
    New (GMP_MALLOC_ID, data, bytes+1, char);
    mpz_export (data, &actual_count, order, size, endian, nails, z);
    assert (count == actual_count);
    data[bytes] = '\0';
    sv = sv_newmortal(); sv_usepvn_mg (sv, data, bytes); PUSHs(sv);


mpz
mpz_import (order, size, endian, nails, sv)
    int     order
    size_t  size
    int     endian
    size_t  nails
    SV      *sv
PREINIT:
    size_t      count;
    const char  *data;
    STRLEN      len;
CODE:
    data = SvPV (sv, len);
    if ((len % size) != 0)
      croak ("%s mpz_import: string not a multiple of the given size",
             mpz_class);
    count = len / size;
    RETVAL = new_mpz();
    mpz_import (RETVAL->m, count, order, size, endian, nails, data);
OUTPUT:
    RETVAL


mpz
nextprime (z)
    mpz_coerce z
CODE:
    RETVAL = new_mpz();
    mpz_nextprime (RETVAL->m, z);
OUTPUT:
    RETVAL


unsigned long
popcount (x)
    mpz_coerce x
CODE:
    RETVAL = mpz_popcount (x);
OUTPUT:
    RETVAL


mpz
powm (b, e, m)
    mpz_coerce b
    mpz_coerce e
    mpz_coerce m
CODE:
    RETVAL = new_mpz();
    mpz_powm (RETVAL->m, b, e, m);
OUTPUT:
    RETVAL


bool
probab_prime_p (z, n)
    mpz_coerce   z
    ulong_coerce n
CODE:
    RETVAL = mpz_probab_prime_p (z, n);
OUTPUT:
    RETVAL


# No attempt to coerce here, only an mpz makes sense.
void
realloc (z, limbs)
    mpz z
    int limbs
CODE:
    _mpz_realloc (z->m, limbs);


void
remove (z, f)
    mpz_coerce z
    mpz_coerce f
PREINIT:
    SV             *sv;
    mpz            rem;
    unsigned long  mult;
PPCODE:
    rem = new_mpz();
    mult = mpz_remove (rem->m, z, f);
    EXTEND (SP, 2);
    PUSHs (MPX_NEWMORTAL (rem, mpz_class_hv));
    PUSHs (sv_2mortal (newSViv (mult)));


void
roote (z, n)
    mpz_coerce   z
    ulong_coerce n
PREINIT:
    SV  *sv;
    mpz root;
    int exact;
PPCODE:
    root = new_mpz();
    exact = mpz_root (root->m, z, n);
    EXTEND (SP, 2);
    PUSHs (MPX_NEWMORTAL (root, mpz_class_hv));
    sv = (exact ? &PL_sv_yes : &PL_sv_no); sv_2mortal(sv); PUSHs(sv);


void
rootrem (z, n)
    mpz_coerce   z
    ulong_coerce n
PREINIT:
    SV  *sv;
    mpz root;
    mpz rem;
PPCODE:
    root = new_mpz();
    rem = new_mpz();
    mpz_rootrem (root->m, rem->m, z, n);
    EXTEND (SP, 2);
    PUSHs (MPX_NEWMORTAL (root, mpz_class_hv));
    PUSHs (MPX_NEWMORTAL (rem,  mpz_class_hv));


# In the past scan0 and scan1 were described as returning ULONG_MAX which
# could be obtained in perl with ~0.  That wasn't true on 64-bit systems
# (eg. alpha) with perl 5.005, since in that version IV and UV were still
# 32-bits.
#
# We changed in gmp 4.2 to just say ~0 for the not-found return.  It's
# likely most people have used ~0 rather than POSIX::ULONG_MAX(), so this
# change should match existing usage.  It only actually makes a difference
# in old perl, since recent versions have gone to 64-bits for IV and UV, the
# same as a ulong.
#
# In perl 5.005 we explicitly mask the mpz return down to 32-bits to get ~0.
# UV_MAX is no good, it reflects the size of the UV type (64-bits), rather
# than the size of the values one ought to be storing in an SV (32-bits).

gmp_UV
scan0 (z, start)
    mpz_coerce   z
    ulong_coerce start
ALIAS:
    GMP::Mpz::scan1 = 1
PREINIT:
    static_functable const struct {
      unsigned long (*op) (mpz_srcptr, unsigned long);
    } table[] = {
      { mpz_scan0  }, /* 0 */
      { mpz_scan1  }, /* 1 */
    };
CODE:
    assert_table (ix);
    RETVAL = (*table[ix].op) (z, start);
    if (PERL_LT (5,6))
      RETVAL &= 0xFFFFFFFF;
OUTPUT:
    RETVAL


void
setbit (sv, bit)
    SV           *sv
    ulong_coerce bit
ALIAS:
    GMP::Mpz::clrbit = 1
    GMP::Mpz::combit = 2
PREINIT:
    static_functable const struct {
      void (*op) (mpz_ptr, unsigned long);
    } table[] = {
      { mpz_setbit }, /* 0 */
      { mpz_clrbit }, /* 1 */
      { mpz_combit }, /* 2 */
    };
    int  use;
    mpz  z;
CODE:
    use = use_sv (sv);
    if (use == USE_MPZ && SvREFCNT(SvRV(sv)) == 1 && ! SvSMAGICAL(sv))
      {
        /* our operand is a non-magical mpz with a reference count of 1, so
           we can just modify it */
        (*table[ix].op) (SvMPZ(sv)->m, bit);
      }
    else
      {
        /* otherwise we need to make a new mpz, from whatever we have, and
           operate on that, possibly invoking magic when storing back */
        SV   *new_sv;
        mpz  z = new_mpz ();
        mpz_ptr  coerce_ptr = coerce_mpz_using (z->m, sv, use);
        if (coerce_ptr != z->m)
          mpz_set (z->m, coerce_ptr);
        (*table[ix].op) (z->m, bit);
        new_sv = sv_bless (sv_setref_pv (sv_newmortal(), NULL, z),
                           mpz_class_hv);
        SvSetMagicSV (sv, new_sv);
      }


void
sqrtrem (z)
    mpz_coerce z
PREINIT:
    SV  *sv;
    mpz root;
    mpz rem;
PPCODE:
    root = new_mpz();
    rem = new_mpz();
    mpz_sqrtrem (root->m, rem->m, z);
    EXTEND (SP, 2);
    PUSHs (MPX_NEWMORTAL (root, mpz_class_hv));
    PUSHs (MPX_NEWMORTAL (rem,  mpz_class_hv));


size_t
sizeinbase (z, base)
    mpz_coerce z
    int        base
CODE:
    RETVAL = mpz_sizeinbase (z, base);
OUTPUT:
    RETVAL


int
tstbit (z, bit)
    mpz_coerce   z
    ulong_coerce bit
CODE:
    RETVAL = mpz_tstbit (z, bit);
OUTPUT:
    RETVAL



#------------------------------------------------------------------------------

MODULE = GMP         PACKAGE = GMP::Mpq


mpq
mpq (...)
ALIAS:
    GMP::Mpq::new = 1
CODE:
    TRACE (printf ("%s new, ix=%ld, items=%d\n", mpq_class, ix, (int) items));
    RETVAL = new_mpq();
    switch (items) {
    case 0:
      mpq_set_ui (RETVAL->m, 0L, 1L);
      break;
    case 1:
      {
        mpq_ptr rp = RETVAL->m;
        mpq_ptr cp = coerce_mpq (rp, ST(0));
        if (cp != rp)
          mpq_set (rp, cp);
      }
      break;
    case 2:
      {
        mpz_ptr rp, cp;
        rp = mpq_numref (RETVAL->m);
        cp = coerce_mpz (rp, ST(0));
        if (cp != rp)
          mpz_set (rp, cp);
        rp = mpq_denref (RETVAL->m);
        cp = coerce_mpz (rp, ST(1));
        if (cp != rp)
          mpz_set (rp, cp);
      }
      break;
    default:
      croak ("%s new: invalid arguments", mpq_class);
    }
OUTPUT:
    RETVAL


void
overload_constant (str, pv, d1, ...)
    const_string_assume str
    SV                  *pv
    dummy               d1
PREINIT:
    SV  *sv;
    mpq q;
PPCODE:
    TRACE (printf ("%s constant: %s\n", mpq_class, str));
    q = new_mpq();
    if (mpq_set_str (q->m, str, 0) == 0)
      { sv = sv_bless (sv_setref_pv (sv_newmortal(), NULL, q), mpq_class_hv); }
    else
      { free_mpq (q); sv = pv; }
    XPUSHs(sv);


mpq
overload_copy (q, d1, d2)
    mpq_assume q
    dummy      d1
    dummy      d2
CODE:
    RETVAL = new_mpq();
    mpq_set (RETVAL->m, q->m);
OUTPUT:
    RETVAL


void
DESTROY (q)
    mpq_assume q
CODE:
    TRACE (printf ("%s DESTROY %p\n", mpq_class, q));
    free_mpq (q);


malloced_string
overload_string (q, d1, d2)
    mpq_assume q
    dummy      d1
    dummy      d2
CODE:
    TRACE (printf ("%s overload_string %p\n", mpq_class, q));
    RETVAL = mpq_get_str (NULL, 10, q->m);
OUTPUT:
    RETVAL


mpq
overload_add (xv, yv, order)
    SV *xv
    SV *yv
    SV *order
ALIAS:
    GMP::Mpq::overload_sub   = 1
    GMP::Mpq::overload_mul   = 2
    GMP::Mpq::overload_div   = 3
PREINIT:
    static_functable const struct {
      void (*op) (mpq_ptr, mpq_srcptr, mpq_srcptr);
    } table[] = {
      { mpq_add }, /* 0 */
      { mpq_sub }, /* 1 */
      { mpq_mul }, /* 2 */
      { mpq_div }, /* 3 */
    };
CODE:
    TRACE (printf ("%s binary\n", mpf_class));
    assert_table (ix);
    if (order == &PL_sv_yes)
      SV_PTR_SWAP (xv, yv);
    RETVAL = new_mpq();
    (*table[ix].op) (RETVAL->m,
                     coerce_mpq (tmp_mpq_0, xv),
                     coerce_mpq (tmp_mpq_1, yv));
OUTPUT:
    RETVAL


void
overload_addeq (x, y, o)
    mpq_assume   x
    mpq_coerce   y
    order_noswap o
ALIAS:
    GMP::Mpq::overload_subeq = 1
    GMP::Mpq::overload_muleq = 2
    GMP::Mpq::overload_diveq = 3
PREINIT:
    static_functable const struct {
      void (*op) (mpq_ptr, mpq_srcptr, mpq_srcptr);
    } table[] = {
      { mpq_add    }, /* 0 */
      { mpq_sub    }, /* 1 */
      { mpq_mul    }, /* 2 */
      { mpq_div    }, /* 3 */
    };
PPCODE:
    assert_table (ix);
    (*table[ix].op) (x->m, x->m, y);
    XPUSHs(ST(0));


mpq
overload_lshift (qv, nv, order)
    SV *qv
    SV *nv
    SV *order
ALIAS:
    GMP::Mpq::overload_rshift   = 1
    GMP::Mpq::overload_pow      = 2
PREINIT:
    static_functable const struct {
      void (*op) (mpq_ptr, mpq_srcptr, unsigned long);
    } table[] = {
      { mpq_mul_2exp }, /* 0 */
      { mpq_div_2exp }, /* 1 */
      { x_mpq_pow_ui }, /* 2 */
    };
CODE:
    assert_table (ix);
    if (order == &PL_sv_yes)
      SV_PTR_SWAP (qv, nv);
    RETVAL = new_mpq();
    (*table[ix].op) (RETVAL->m, coerce_mpq (RETVAL->m, qv), coerce_ulong (nv));
OUTPUT:
    RETVAL


void
overload_lshifteq (q, n, o)
    mpq_assume   q
    ulong_coerce n
    order_noswap o
ALIAS:
    GMP::Mpq::overload_rshifteq   = 1
    GMP::Mpq::overload_poweq      = 2
PREINIT:
    static_functable const struct {
      void (*op) (mpq_ptr, mpq_srcptr, unsigned long);
    } table[] = {
      { mpq_mul_2exp }, /* 0 */
      { mpq_div_2exp }, /* 1 */
      { x_mpq_pow_ui }, /* 2 */
    };
PPCODE:
    assert_table (ix);
    (*table[ix].op) (q->m, q->m, n);
    XPUSHs(ST(0));


void
overload_inc (q, d1, d2)
    mpq_assume q
    dummy      d1
    dummy      d2
ALIAS:
    GMP::Mpq::overload_dec = 1
PREINIT:
    static_functable const struct {
      void (*op) (mpz_ptr, mpz_srcptr, mpz_srcptr);
    } table[] = {
      { mpz_add }, /* 0 */
      { mpz_sub }, /* 1 */
    };
CODE:
    assert_table (ix);
    (*table[ix].op) (mpq_numref(q->m), mpq_numref(q->m), mpq_denref(q->m));


mpq
overload_abs (q, d1, d2)
    mpq_assume q
    dummy      d1
    dummy      d2
ALIAS:
    GMP::Mpq::overload_neg = 1
PREINIT:
    static_functable const struct {
      void (*op) (mpq_ptr w, mpq_srcptr x);
    } table[] = {
      { mpq_abs }, /* 0 */
      { mpq_neg }, /* 1 */
    };
CODE:
    assert_table (ix);
    RETVAL = new_mpq();
    (*table[ix].op) (RETVAL->m, q->m);
OUTPUT:
    RETVAL


int
overload_spaceship (x, y, order)
    mpq_assume x
    mpq_coerce y
    SV         *order
CODE:
    RETVAL = mpq_cmp (x->m, y);
    RETVAL = SGN (RETVAL);
    if (order == &PL_sv_yes)
      RETVAL = -RETVAL;
OUTPUT:
    RETVAL


bool
overload_bool (q, d1, d2)
    mpq_assume q
    dummy      d1
    dummy      d2
ALIAS:
    GMP::Mpq::overload_not = 1
CODE:
    RETVAL = (mpq_sgn (q->m) != 0) ^ ix;
OUTPUT:
    RETVAL


bool
overload_eq (x, yv, d)
    mpq_assume x
    SV         *yv
    dummy      d
ALIAS:
    GMP::Mpq::overload_ne = 1
PREINIT:
    int  use;
CODE:
    use = use_sv (yv);
    switch (use) {
    case USE_IVX:
    case USE_UVX:
    case USE_MPZ:
      RETVAL = 0;
      if (x_mpq_integer_p (x->m))
        {
          switch (use) {
          case USE_IVX:
            RETVAL = (mpz_cmp_si (mpq_numref(x->m), SvIVX(yv)) == 0);
            break;
          case USE_UVX:
            RETVAL = (mpz_cmp_ui (mpq_numref(x->m), SvUVX(yv)) == 0);
            break;
          case USE_MPZ:
            RETVAL = (mpz_cmp (mpq_numref(x->m), SvMPZ(yv)->m) == 0);
            break;
          }
        }
      break;

    case USE_MPQ:
      RETVAL = (mpq_equal (x->m, SvMPQ(yv)->m) != 0);
      break;

    default:
      RETVAL = (mpq_equal (x->m, coerce_mpq_using (tmp_mpq_0, yv, use)) != 0);
      break;
    }
    RETVAL ^= ix;
OUTPUT:
    RETVAL


void
canonicalize (q)
    mpq q
CODE:
    mpq_canonicalize (q->m);


mpq
inv (q)
    mpq_coerce q
CODE:
    RETVAL = new_mpq();
    mpq_inv (RETVAL->m, q);
OUTPUT:
    RETVAL


mpz
num (q)
    mpq q
ALIAS:
    GMP::Mpq::den = 1
CODE:
    RETVAL = new_mpz();
    mpz_set (RETVAL->m, (ix == 0 ? mpq_numref(q->m) : mpq_denref(q->m)));
OUTPUT:
    RETVAL



#------------------------------------------------------------------------------

MODULE = GMP         PACKAGE = GMP::Mpf


mpf
mpf (...)
ALIAS:
    GMP::Mpf::new = 1
PREINIT:
    unsigned long  prec;
CODE:
    TRACE (printf ("%s new\n", mpf_class));
    if (items > 2)
      croak ("%s new: invalid arguments", mpf_class);
    prec = (items == 2 ? coerce_ulong (ST(1)) : mpf_get_default_prec());
    RETVAL = new_mpf (prec);
    if (items >= 1)
      {
        SV *sv = ST(0);
        my_mpf_set_sv_using (RETVAL, sv, use_sv(sv));
      }
OUTPUT:
    RETVAL


mpf
overload_constant (sv, d1, d2, ...)
    SV     *sv
    dummy  d1
    dummy  d2
CODE:
    assert (SvPOK (sv));
    TRACE (printf ("%s constant: %s\n", mpq_class, SvPVX(sv)));
    RETVAL = new_mpf (mpf_get_default_prec());
    my_mpf_set_svstr (RETVAL, sv);
OUTPUT:
    RETVAL


mpf
overload_copy (f, d1, d2)
    mpf_assume f
    dummy      d1
    dummy      d2
CODE:
    TRACE (printf ("%s copy\n", mpf_class));
    RETVAL = new_mpf (mpf_get_prec (f));
    mpf_set (RETVAL, f);
OUTPUT:
    RETVAL


void
DESTROY (f)
    mpf_assume f
CODE:
    TRACE (printf ("%s DESTROY %p\n", mpf_class, f));
    mpf_clear (f);
    Safefree (f);
    assert_support (mpf_count--);
    TRACE_ACTIVE ();


mpf
overload_add (x, y, order)
    mpf_assume     x
    mpf_coerce_st0 y
    SV             *order
ALIAS:
    GMP::Mpf::overload_sub   = 1
    GMP::Mpf::overload_mul   = 2
    GMP::Mpf::overload_div   = 3
PREINIT:
    static_functable const struct {
      void (*op) (mpf_ptr, mpf_srcptr, mpf_srcptr);
    } table[] = {
      { mpf_add }, /* 0 */
      { mpf_sub }, /* 1 */
      { mpf_mul }, /* 2 */
      { mpf_div }, /* 3 */
    };
CODE:
    assert_table (ix);
    RETVAL = new_mpf (mpf_get_prec (x));
    if (order == &PL_sv_yes)
      MPF_PTR_SWAP (x, y);
    (*table[ix].op) (RETVAL, x, y);
OUTPUT:
    RETVAL


void
overload_addeq (x, y, o)
    mpf_assume     x
    mpf_coerce_st0 y
    order_noswap   o
ALIAS:
    GMP::Mpf::overload_subeq = 1
    GMP::Mpf::overload_muleq = 2
    GMP::Mpf::overload_diveq = 3
PREINIT:
    static_functable const struct {
      void (*op) (mpf_ptr, mpf_srcptr, mpf_srcptr);
    } table[] = {
      { mpf_add }, /* 0 */
      { mpf_sub }, /* 1 */
      { mpf_mul }, /* 2 */
      { mpf_div }, /* 3 */
    };
PPCODE:
    assert_table (ix);
    (*table[ix].op) (x, x, y);
    XPUSHs(ST(0));


mpf
overload_lshift (fv, nv, order)
    SV *fv
    SV *nv
    SV *order
ALIAS:
    GMP::Mpf::overload_rshift = 1
    GMP::Mpf::overload_pow    = 2
PREINIT:
    static_functable const struct {
      void (*op) (mpf_ptr, mpf_srcptr, unsigned long);
    } table[] = {
      { mpf_mul_2exp }, /* 0 */
      { mpf_div_2exp }, /* 1 */
      { mpf_pow_ui   }, /* 2 */
    };
    mpf f;
    unsigned long prec;
CODE:
    assert_table (ix);
    MPF_ASSUME (f, fv);
    prec = mpf_get_prec (f);
    if (order == &PL_sv_yes)
      SV_PTR_SWAP (fv, nv);
    f = coerce_mpf (tmp_mpf_0, fv, prec);
    RETVAL = new_mpf (prec);
    (*table[ix].op) (RETVAL, f, coerce_ulong (nv));
OUTPUT:
    RETVAL


void
overload_lshifteq (f, n, o)
    mpf_assume   f
    ulong_coerce n
    order_noswap o
ALIAS:
    GMP::Mpf::overload_rshifteq   = 1
    GMP::Mpf::overload_poweq      = 2
PREINIT:
    static_functable const struct {
      void (*op) (mpf_ptr, mpf_srcptr, unsigned long);
    } table[] = {
      { mpf_mul_2exp }, /* 0 */
      { mpf_div_2exp }, /* 1 */
      { mpf_pow_ui   }, /* 2 */
    };
PPCODE:
    assert_table (ix);
    (*table[ix].op) (f, f, n);
    XPUSHs(ST(0));


mpf
overload_abs (f, d1, d2)
    mpf_assume f
    dummy      d1
    dummy      d2
ALIAS:
    GMP::Mpf::overload_neg   = 1
    GMP::Mpf::overload_sqrt  = 2
PREINIT:
    static_functable const struct {
      void (*op) (mpf_ptr w, mpf_srcptr x);
    } table[] = {
      { mpf_abs  }, /* 0 */
      { mpf_neg  }, /* 1 */
      { mpf_sqrt }, /* 2 */
    };
CODE:
    assert_table (ix);
    RETVAL = new_mpf (mpf_get_prec (f));
    (*table[ix].op) (RETVAL, f);
OUTPUT:
    RETVAL


void
overload_inc (f, d1, d2)
    mpf_assume f
    dummy      d1
    dummy      d2
ALIAS:
    GMP::Mpf::overload_dec = 1
PREINIT:
    static_functable const struct {
      void (*op) (mpf_ptr w, mpf_srcptr x, unsigned long y);
    } table[] = {
      { mpf_add_ui }, /* 0 */
      { mpf_sub_ui }, /* 1 */
    };
CODE:
    assert_table (ix);
    (*table[ix].op) (f, f, 1L);


int
overload_spaceship (xv, yv, order)
    SV *xv
    SV *yv
    SV *order
PREINIT:
    mpf x;
CODE:
    MPF_ASSUME (x, xv);
    switch (use_sv (yv)) {
    case USE_IVX:
      RETVAL = mpf_cmp_si (x, SvIVX(yv));
      break;
    case USE_UVX:
      RETVAL = mpf_cmp_ui (x, SvUVX(yv));
      break;
    case USE_NVX:
      RETVAL = mpf_cmp_d (x, SvNVX(yv));
      break;
    case USE_PVX:
      {
        STRLEN len;
        const char *str = SvPV (yv, len);
        /* enough for all digits of the string */
        tmp_mpf_set_prec (tmp_mpf_0, strlen(str)+64);
        if (mpf_set_str (tmp_mpf_0->m, str, 10) != 0)
          croak ("%s <=>: invalid string format", mpf_class);
        RETVAL = mpf_cmp (x, tmp_mpf_0->m);
      }
      break;
    case USE_MPZ:
      RETVAL = - x_mpz_cmp_f (SvMPZ(yv)->m, x);
      break;
    case USE_MPF:
      RETVAL = mpf_cmp (x, SvMPF(yv));
      break;
    default:
      RETVAL = mpq_cmp (coerce_mpq (tmp_mpq_0, xv),
                        coerce_mpq (tmp_mpq_1, yv));
      break;
    }
    RETVAL = SGN (RETVAL);
    if (order == &PL_sv_yes)
      RETVAL = -RETVAL;
OUTPUT:
    RETVAL


bool
overload_bool (f, d1, d2)
    mpf_assume f
    dummy      d1
    dummy      d2
ALIAS:
    GMP::Mpf::overload_not = 1
CODE:
    RETVAL = (mpf_sgn (f) != 0) ^ ix;
OUTPUT:
    RETVAL


mpf
ceil (f)
    mpf_coerce_def f
ALIAS:
    GMP::Mpf::floor = 1
    GMP::Mpf::trunc = 2
PREINIT:
    static_functable const struct {
      void (*op) (mpf_ptr w, mpf_srcptr x);
    } table[] = {
      { mpf_ceil  }, /* 0 */
      { mpf_floor }, /* 1 */
      { mpf_trunc }, /* 2 */
    };
CODE:
    assert_table (ix);
    RETVAL = new_mpf (mpf_get_prec (f));
    (*table[ix].op) (RETVAL, f);
OUTPUT:
    RETVAL


unsigned long
get_default_prec ()
CODE:
    RETVAL = mpf_get_default_prec();
OUTPUT:
    RETVAL


unsigned long
get_prec (f)
    mpf_coerce_def f
CODE:
    RETVAL = mpf_get_prec (f);
OUTPUT:
    RETVAL


bool
mpf_eq (xv, yv, bits)
    SV           *xv
    SV           *yv
    ulong_coerce bits
PREINIT:
    mpf  x, y;
CODE:
    TRACE (printf ("%s eq\n", mpf_class));
    coerce_mpf_pair (&x,xv, &y,yv);
    RETVAL = mpf_eq (x, y, bits);
OUTPUT:
    RETVAL


mpf
reldiff (xv, yv)
    SV *xv
    SV *yv
PREINIT:
    mpf  x, y;
    unsigned long prec;
CODE:
    TRACE (printf ("%s reldiff\n", mpf_class));
    prec = coerce_mpf_pair (&x,xv, &y,yv);
    RETVAL = new_mpf (prec);
    mpf_reldiff (RETVAL, x, y);
OUTPUT:
    RETVAL


void
set_default_prec (prec)
    ulong_coerce prec
CODE:
    TRACE (printf ("%s set_default_prec %lu\n", mpf_class, prec));
    mpf_set_default_prec (prec);


void
set_prec (sv, prec)
    SV           *sv
    ulong_coerce prec
PREINIT:
    mpf_ptr  old_f, new_f;
    int      use;
CODE:
    TRACE (printf ("%s set_prec to %lu\n", mpf_class, prec));
    use = use_sv (sv);
    if (use == USE_MPF)
      {
        old_f = SvMPF(sv);
        if (SvREFCNT(SvRV(sv)) == 1)
          mpf_set_prec (old_f, prec);
        else
          {
            TRACE (printf ("  fork new mpf\n"));
            new_f = new_mpf (prec);
            mpf_set (new_f, old_f);
            goto setref;
          }
      }
    else
      {
        TRACE (printf ("  coerce to mpf\n"));
        new_f = new_mpf (prec);
        my_mpf_set_sv_using (new_f, sv, use);
      setref:
        sv_bless (sv_setref_pv (sv, NULL, new_f), mpf_class_hv);
      }



#------------------------------------------------------------------------------

MODULE = GMP         PACKAGE = GMP::Rand

randstate
new (...)
ALIAS:
    GMP::Rand::randstate = 1
CODE:
    TRACE (printf ("%s new\n", rand_class));
    New (GMP_MALLOC_ID, RETVAL, 1, __gmp_randstate_struct);
    TRACE (printf ("  RETVAL %p\n", RETVAL));
    assert_support (rand_count++);
    TRACE_ACTIVE ();

    if (items == 0)
      {
        gmp_randinit_default (RETVAL);
      }
    else
      {
        if (SvROK (ST(0)) && sv_derived_from (ST(0), rand_class))
          {
            if (items != 1)
              goto invalid;
            gmp_randinit_set (RETVAL, SvRANDSTATE (ST(0)));
          }
        else
          {
            STRLEN      len;
            const char  *method = SvPV (ST(0), len);
            assert (len == strlen (method));
            if (strcmp (method, "lc_2exp") == 0)
              {
                if (items != 4)
                  goto invalid;
                gmp_randinit_lc_2exp (RETVAL,
                                      coerce_mpz (tmp_mpz_0, ST(1)),
                                      coerce_ulong (ST(2)),
                                      coerce_ulong (ST(3)));
              }
            else if (strcmp (method, "lc_2exp_size") == 0)
              {
                if (items != 2)
                  goto invalid;
                if (! gmp_randinit_lc_2exp_size (RETVAL, coerce_ulong (ST(1))))
                  {
                    Safefree (RETVAL);
                    XSRETURN_UNDEF;
                  }
              }
            else if (strcmp (method, "mt") == 0)
              {
                if (items != 1)
                  goto invalid;
                gmp_randinit_mt (RETVAL);
              }
            else
              {
              invalid:
                croak ("%s new: invalid arguments", rand_class);
              }
          }
      }
OUTPUT:
    RETVAL


void
DESTROY (r)
    randstate r
CODE:
    TRACE (printf ("%s DESTROY\n", rand_class));
    gmp_randclear (r);
    Safefree (r);
    assert_support (rand_count--);
    TRACE_ACTIVE ();


void
seed (r, z)
    randstate  r
    mpz_coerce z
CODE:
    gmp_randseed (r, z);


mpz
mpz_urandomb (r, bits)
    randstate    r
    ulong_coerce bits
ALIAS:
    GMP::Rand::mpz_rrandomb = 1
PREINIT:
    static_functable const struct {
      void (*fun) (mpz_ptr, gmp_randstate_t r, unsigned long bits);
    } table[] = {
      { mpz_urandomb }, /* 0 */
      { mpz_rrandomb }, /* 1 */
    };
CODE:
    assert_table (ix);
    RETVAL = new_mpz();
    (*table[ix].fun) (RETVAL->m, r, bits);
OUTPUT:
    RETVAL


mpz
mpz_urandomm (r, m)
    randstate  r
    mpz_coerce m
CODE:
    RETVAL = new_mpz();
    mpz_urandomm (RETVAL->m, r, m);
OUTPUT:
    RETVAL


mpf
mpf_urandomb (r, bits)
    randstate    r
    ulong_coerce bits
CODE:
    RETVAL = new_mpf (bits);
    mpf_urandomb (RETVAL, r, bits);
OUTPUT:
    RETVAL


unsigned long
gmp_urandomb_ui (r, bits)
    randstate    r
    ulong_coerce bits
ALIAS:
    GMP::Rand::gmp_urandomm_ui = 1
PREINIT:
    static_functable const struct {
      unsigned long (*fun) (gmp_randstate_t r, unsigned long bits);
    } table[] = {
      { gmp_urandomb_ui }, /* 0 */
      { gmp_urandomm_ui }, /* 1 */
    };
CODE:
    assert_table (ix);
    RETVAL = (*table[ix].fun) (r, bits);
OUTPUT:
    RETVAL
