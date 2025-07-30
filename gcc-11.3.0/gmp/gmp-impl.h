/* Include file for internal GNU MP types and definitions.

   THE CONTENTS OF THIS FILE ARE FOR INTERNAL USE AND ARE ALMOST CERTAIN TO
   BE SUBJECT TO INCOMPATIBLE CHANGES IN FUTURE GNU MP RELEASES.

Copyright 1991, 1993-1997, 1999-2015 Free Software Foundation, Inc.

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
see https://www.gnu.org/licenses/.  */


/* __GMP_DECLSPEC must be given on any global data that will be accessed
   from outside libgmp, meaning from the test or development programs, or
   from libgmpxx.  Failing to do this will result in an incorrect address
   being used for the accesses.  On functions __GMP_DECLSPEC makes calls
   from outside libgmp more efficient, but they'll still work fine without
   it.  */


#ifndef __GMP_IMPL_H__
#define __GMP_IMPL_H__

#if defined _CRAY
#include <intrinsics.h>  /* for _popcnt */
#endif

/* For INT_MAX, etc. We used to avoid it because of a bug (on solaris,
   gcc 2.95 under -mcpu=ultrasparc in ABI=32 ends up getting wrong
   values (the ABI=64 values)), but it should be safe now.

   On Cray vector systems, however, we need the system limits.h since sizes
   of signed and unsigned types can differ there, depending on compiler
   options (eg. -hnofastmd), making our SHRT_MAX etc expressions fail.  For
   reference, int can be 46 or 64 bits, whereas uint is always 64 bits; and
   short can be 24, 32, 46 or 64 bits, and different for ushort.  */

#include <limits.h>

/* For fat.h and other fat binary stuff.
   No need for __GMP_ATTRIBUTE_PURE or __GMP_NOTHROW, since functions
   declared this way are only used to set function pointers in __gmpn_cpuvec,
   they're not called directly.  */
#define DECL_add_n(name) \
  __GMP_DECLSPEC mp_limb_t name (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t)
#define DECL_addlsh1_n(name) \
  DECL_add_n (name)
#define DECL_addlsh2_n(name) \
  DECL_add_n (name)
#define DECL_addmul_1(name) \
  __GMP_DECLSPEC mp_limb_t name (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t)
#define DECL_addmul_2(name) \
  __GMP_DECLSPEC mp_limb_t name (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr)
#define DECL_bdiv_dbm1c(name) \
  __GMP_DECLSPEC mp_limb_t name (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t)
#define DECL_cnd_add_n(name) \
  __GMP_DECLSPEC mp_limb_t name (mp_limb_t, mp_ptr, mp_srcptr, mp_srcptr, mp_size_t)
#define DECL_cnd_sub_n(name) \
  __GMP_DECLSPEC mp_limb_t name (mp_limb_t, mp_ptr, mp_srcptr, mp_srcptr, mp_size_t)
#define DECL_com(name) \
  __GMP_DECLSPEC void name (mp_ptr, mp_srcptr, mp_size_t)
#define DECL_copyd(name) \
  __GMP_DECLSPEC void name (mp_ptr, mp_srcptr, mp_size_t)
#define DECL_copyi(name) \
  DECL_copyd (name)
#define DECL_divexact_1(name) \
  __GMP_DECLSPEC void name (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t)
#define DECL_divexact_by3c(name) \
  __GMP_DECLSPEC mp_limb_t name (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t)
#define DECL_divrem_1(name) \
  __GMP_DECLSPEC mp_limb_t name (mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t)
#define DECL_gcd_1(name) \
  __GMP_DECLSPEC mp_limb_t name (mp_srcptr, mp_size_t, mp_limb_t)
#define DECL_lshift(name) \
  __GMP_DECLSPEC mp_limb_t name (mp_ptr, mp_srcptr, mp_size_t, unsigned)
#define DECL_lshiftc(name) \
  DECL_lshift (name)
#define DECL_mod_1(name) \
  __GMP_DECLSPEC mp_limb_t name (mp_srcptr, mp_size_t, mp_limb_t)
#define DECL_mod_1_1p(name) \
  __GMP_DECLSPEC mp_limb_t name (mp_srcptr, mp_size_t, mp_limb_t, const mp_limb_t [])
#define DECL_mod_1_1p_cps(name) \
  __GMP_DECLSPEC void name (mp_limb_t cps[], mp_limb_t b)
#define DECL_mod_1s_2p(name) \
  DECL_mod_1_1p (name)
#define DECL_mod_1s_2p_cps(name) \
  DECL_mod_1_1p_cps (name)
#define DECL_mod_1s_4p(name) \
  DECL_mod_1_1p (name)
#define DECL_mod_1s_4p_cps(name) \
  DECL_mod_1_1p_cps (name)
#define DECL_mod_34lsub1(name) \
  __GMP_DECLSPEC mp_limb_t name (mp_srcptr, mp_size_t)
#define DECL_modexact_1c_odd(name) \
  __GMP_DECLSPEC mp_limb_t name (mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t)
#define DECL_mul_1(name) \
  DECL_addmul_1 (name)
#define DECL_mul_basecase(name) \
  __GMP_DECLSPEC void name (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t)
#define DECL_mullo_basecase(name) \
  __GMP_DECLSPEC void name (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t)
#define DECL_preinv_divrem_1(name) \
  __GMP_DECLSPEC mp_limb_t name (mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t, int)
#define DECL_preinv_mod_1(name) \
  __GMP_DECLSPEC mp_limb_t name (mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t)
#define DECL_redc_1(name) \
  __GMP_DECLSPEC mp_limb_t name (mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_limb_t)
#define DECL_redc_2(name) \
  __GMP_DECLSPEC mp_limb_t name (mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_srcptr)
#define DECL_rshift(name) \
  DECL_lshift (name)
#define DECL_sqr_basecase(name) \
  __GMP_DECLSPEC void name (mp_ptr, mp_srcptr, mp_size_t)
#define DECL_sub_n(name) \
  DECL_add_n (name)
#define DECL_sublsh1_n(name) \
  DECL_add_n (name)
#define DECL_submul_1(name) \
  DECL_addmul_1 (name)

#if ! defined (__GMP_WITHIN_CONFIGURE)
#include "config.h"
#include "gmp-mparam.h"
#include "fib_table.h"
#include "fac_table.h"
#include "mp_bases.h"
#if WANT_FAT_BINARY
#include "fat.h"
#endif
#endif

#if HAVE_INTTYPES_H      /* for uint_least32_t */
# include <inttypes.h>
#else
# if HAVE_STDINT_H
#  include <stdint.h>
# endif
#endif

#ifdef __cplusplus
#include <cstring>  /* for strlen */
#include <string>   /* for std::string */
#endif


#ifndef WANT_TMP_DEBUG  /* for TMP_ALLOC_LIMBS_2 and others */
#define WANT_TMP_DEBUG 0
#endif

/* The following tries to get a good version of alloca.  The tests are
   adapted from autoconf AC_FUNC_ALLOCA, with a couple of additions.
   Whether this succeeds is tested by GMP_FUNC_ALLOCA and HAVE_ALLOCA will
   be setup appropriately.

   ifndef alloca - a cpp define might already exist.
       glibc <stdlib.h> includes <alloca.h> which uses GCC __builtin_alloca.
       HP cc +Olibcalls adds a #define of alloca to __builtin_alloca.

   GCC __builtin_alloca - preferred whenever available.

   _AIX pragma - IBM compilers need a #pragma in "each module that needs to
       use alloca".  Pragma indented to protect pre-ANSI cpp's.  _IBMR2 was
       used in past versions of GMP, retained still in case it matters.

       The autoconf manual says this pragma needs to be at the start of a C
       file, apart from comments and preprocessor directives.  Is that true?
       xlc on aix 4.xxx doesn't seem to mind it being after prototypes etc
       from gmp.h.
*/

#ifndef alloca
# ifdef __GNUC__
#  define alloca __builtin_alloca
# else
#  ifdef __DECC
#   define alloca(x) __ALLOCA(x)
#  else
#   ifdef _MSC_VER
#    include <malloc.h>
#    define alloca _alloca
#   else
#    if HAVE_ALLOCA_H
#     include <alloca.h>
#    else
#     if defined (_AIX) || defined (_IBMR2)
 #pragma alloca
#     else
       char *alloca ();
#     endif
#    endif
#   endif
#  endif
# endif
#endif


/* if not provided by gmp-mparam.h */
#ifndef GMP_LIMB_BYTES
#define GMP_LIMB_BYTES  SIZEOF_MP_LIMB_T
#endif
#ifndef GMP_LIMB_BITS
#define GMP_LIMB_BITS  (8 * SIZEOF_MP_LIMB_T)
#endif

#define BITS_PER_ULONG  (8 * SIZEOF_UNSIGNED_LONG)


/* gmp_uint_least32_t is an unsigned integer type with at least 32 bits. */
#if HAVE_UINT_LEAST32_T
typedef uint_least32_t      gmp_uint_least32_t;
#else
#if SIZEOF_UNSIGNED_SHORT >= 4
typedef unsigned short      gmp_uint_least32_t;
#else
#if SIZEOF_UNSIGNED >= 4
typedef unsigned            gmp_uint_least32_t;
#else
typedef unsigned long       gmp_uint_least32_t;
#endif
#endif
#endif


/* gmp_intptr_t, for pointer to integer casts */
#if HAVE_INTPTR_T
typedef intptr_t            gmp_intptr_t;
#else /* fallback */
typedef size_t              gmp_intptr_t;
#endif


/* pre-inverse types for truncating division and modulo */
typedef struct {mp_limb_t inv32;} gmp_pi1_t;
typedef struct {mp_limb_t inv21, inv32, inv53;} gmp_pi2_t;


/* "const" basically means a function does nothing but examine its arguments
   and give a return value, it doesn't read or write any memory (neither
   global nor pointed to by arguments), and has no other side-effects.  This
   is more restrictive than "pure".  See info node "(gcc)Function
   Attributes".  __GMP_NO_ATTRIBUTE_CONST_PURE lets tune/common.c etc turn
   this off when trying to write timing loops.  */
#if HAVE_ATTRIBUTE_CONST && ! defined (__GMP_NO_ATTRIBUTE_CONST_PURE)
#define ATTRIBUTE_CONST  __attribute__ ((const))
#else
#define ATTRIBUTE_CONST
#endif

#if HAVE_ATTRIBUTE_NORETURN
#define ATTRIBUTE_NORETURN  __attribute__ ((noreturn))
#else
#define ATTRIBUTE_NORETURN
#endif

/* "malloc" means a function behaves like malloc in that the pointer it
   returns doesn't alias anything.  */
#if HAVE_ATTRIBUTE_MALLOC
#define ATTRIBUTE_MALLOC  __attribute__ ((malloc))
#else
#define ATTRIBUTE_MALLOC
#endif


#if ! HAVE_STRCHR
#define strchr(s,c)  index(s,c)
#endif

#if ! HAVE_MEMSET
#define memset(p, c, n)			\
  do {					\
    ASSERT ((n) >= 0);			\
    char *__memset__p = (p);		\
    int	 __i;				\
    for (__i = 0; __i < (n); __i++)	\
      __memset__p[__i] = (c);		\
  } while (0)
#endif

/* va_copy is standard in C99, and gcc provides __va_copy when in strict C89
   mode.  Falling back to a memcpy will give maximum portability, since it
   works no matter whether va_list is a pointer, struct or array.  */
#if ! defined (va_copy) && defined (__va_copy)
#define va_copy(dst,src)  __va_copy(dst,src)
#endif
#if ! defined (va_copy)
#define va_copy(dst,src) \
  do { memcpy (&(dst), &(src), sizeof (va_list)); } while (0)
#endif


/* HAVE_HOST_CPU_alpha_CIX is 1 on an alpha with the CIX instructions
   (ie. ctlz, ctpop, cttz).  */
#if HAVE_HOST_CPU_alphaev67 || HAVE_HOST_CPU_alphaev68  \
  || HAVE_HOST_CPU_alphaev7
#define HAVE_HOST_CPU_alpha_CIX 1
#endif


#if defined (__cplusplus)
extern "C" {
#endif


/* Usage: TMP_DECL;
	  TMP_MARK;
	  ptr = TMP_ALLOC (bytes);
	  TMP_FREE;

   Small allocations should use TMP_SALLOC, big allocations should use
   TMP_BALLOC.  Allocations that might be small or big should use TMP_ALLOC.

   Functions that use just TMP_SALLOC should use TMP_SDECL, TMP_SMARK, and
   TMP_SFREE.

   TMP_DECL just declares a variable, but might be empty and so must be last
   in a list of variables.  TMP_MARK must be done before any TMP_ALLOC.
   TMP_ALLOC(0) is not allowed.  TMP_FREE doesn't need to be done if a
   TMP_MARK was made, but then no TMP_ALLOCs.  */

/* The alignment in bytes, used for TMP_ALLOCed blocks, when alloca or
   __gmp_allocate_func doesn't already determine it.  Currently TMP_ALLOC
   isn't used for "double"s, so that's not in the union.  */
union tmp_align_t {
  mp_limb_t  l;
  char       *p;
};
#define __TMP_ALIGN  sizeof (union tmp_align_t)

/* Return "a" rounded upwards to a multiple of "m", if it isn't already.
   "a" must be an unsigned type.
   This is designed for use with a compile-time constant "m".
   The POW2 case is expected to be usual, and gcc 3.0 and up recognises
   "(-(8*n))%8" or the like is always zero, which means the rounding up in
   the WANT_TMP_NOTREENTRANT version of TMP_ALLOC below will be a noop.  */
#define ROUND_UP_MULTIPLE(a,m)          \
  (POW2_P(m) ? (a) + (-(a))%(m)         \
   : (a)+(m)-1 - (((a)+(m)-1) % (m)))

#if defined (WANT_TMP_ALLOCA) || defined (WANT_TMP_REENTRANT)
struct tmp_reentrant_t {
  struct tmp_reentrant_t  *next;
  size_t		  size;	  /* bytes, including header */
};
__GMP_DECLSPEC void *__gmp_tmp_reentrant_alloc (struct tmp_reentrant_t **, size_t) ATTRIBUTE_MALLOC;
__GMP_DECLSPEC void  __gmp_tmp_reentrant_free (struct tmp_reentrant_t *);
#endif

#if WANT_TMP_ALLOCA
#define TMP_SDECL
#define TMP_DECL		struct tmp_reentrant_t *__tmp_marker
#define TMP_SMARK
#define TMP_MARK		__tmp_marker = 0
#define TMP_SALLOC(n)		alloca(n)
#define TMP_BALLOC(n)		__gmp_tmp_reentrant_alloc (&__tmp_marker, n)
/* The peculiar stack allocation limit here is chosen for efficient asm.  */
#define TMP_ALLOC(n)							\
  (LIKELY ((n) <= 0x7f00) ? TMP_SALLOC(n) : TMP_BALLOC(n))
#define TMP_SFREE
#define TMP_FREE							\
  do {									\
    if (UNLIKELY (__tmp_marker != 0))					\
      __gmp_tmp_reentrant_free (__tmp_marker);				\
  } while (0)
#endif

#if WANT_TMP_REENTRANT
#define TMP_SDECL		TMP_DECL
#define TMP_DECL		struct tmp_reentrant_t *__tmp_marker
#define TMP_SMARK		TMP_MARK
#define TMP_MARK		__tmp_marker = 0
#define TMP_SALLOC(n)		TMP_ALLOC(n)
#define TMP_BALLOC(n)		TMP_ALLOC(n)
#define TMP_ALLOC(n)		__gmp_tmp_reentrant_alloc (&__tmp_marker, n)
#define TMP_SFREE		TMP_FREE
#define TMP_FREE		__gmp_tmp_reentrant_free (__tmp_marker)
#endif

#if WANT_TMP_NOTREENTRANT
struct tmp_marker
{
  struct tmp_stack *which_chunk;
  void *alloc_point;
};
__GMP_DECLSPEC void *__gmp_tmp_alloc (unsigned long) ATTRIBUTE_MALLOC;
__GMP_DECLSPEC void __gmp_tmp_mark (struct tmp_marker *);
__GMP_DECLSPEC void __gmp_tmp_free (struct tmp_marker *);
#define TMP_SDECL		TMP_DECL
#define TMP_DECL		struct tmp_marker __tmp_marker
#define TMP_SMARK		TMP_MARK
#define TMP_MARK		__gmp_tmp_mark (&__tmp_marker)
#define TMP_SALLOC(n)		TMP_ALLOC(n)
#define TMP_BALLOC(n)		TMP_ALLOC(n)
#define TMP_ALLOC(n)							\
  __gmp_tmp_alloc (ROUND_UP_MULTIPLE ((unsigned long) (n), __TMP_ALIGN))
#define TMP_SFREE		TMP_FREE
#define TMP_FREE		__gmp_tmp_free (&__tmp_marker)
#endif

#if WANT_TMP_DEBUG
/* See tal-debug.c for some comments. */
struct tmp_debug_t {
  struct tmp_debug_entry_t  *list;
  const char                *file;
  int                       line;
};
struct tmp_debug_entry_t {
  struct tmp_debug_entry_t  *next;
  void                      *block;
  size_t                    size;
};
__GMP_DECLSPEC void  __gmp_tmp_debug_mark (const char *, int, struct tmp_debug_t **,
					   struct tmp_debug_t *,
					   const char *, const char *);
__GMP_DECLSPEC void *__gmp_tmp_debug_alloc (const char *, int, int,
					    struct tmp_debug_t **, const char *,
					    size_t) ATTRIBUTE_MALLOC;
__GMP_DECLSPEC void  __gmp_tmp_debug_free (const char *, int, int,
					   struct tmp_debug_t **,
					   const char *, const char *);
#define TMP_SDECL TMP_DECL_NAME(__tmp_xmarker, "__tmp_marker")
#define TMP_DECL TMP_DECL_NAME(__tmp_xmarker, "__tmp_marker")
#define TMP_SMARK TMP_MARK_NAME(__tmp_xmarker, "__tmp_marker")
#define TMP_MARK TMP_MARK_NAME(__tmp_xmarker, "__tmp_marker")
#define TMP_SFREE TMP_FREE_NAME(__tmp_xmarker, "__tmp_marker")
#define TMP_FREE TMP_FREE_NAME(__tmp_xmarker, "__tmp_marker")
/* The marker variable is designed to provoke an uninitialized variable
   warning from the compiler if TMP_FREE is used without a TMP_MARK.
   __tmp_marker_inscope does the same for TMP_ALLOC.  Runtime tests pick
   these things up too.  */
#define TMP_DECL_NAME(marker, marker_name)				\
  int marker;								\
  int __tmp_marker_inscope;						\
  const char *__tmp_marker_name = marker_name;				\
  struct tmp_debug_t  __tmp_marker_struct;				\
  /* don't demand NULL, just cast a zero */				\
  struct tmp_debug_t  *__tmp_marker = (struct tmp_debug_t *) 0
#define TMP_MARK_NAME(marker, marker_name)				\
  do {									\
    marker = 1;								\
    __tmp_marker_inscope = 1;						\
    __gmp_tmp_debug_mark  (ASSERT_FILE, ASSERT_LINE,			\
			   &__tmp_marker, &__tmp_marker_struct,		\
			   __tmp_marker_name, marker_name);		\
  } while (0)
#define TMP_SALLOC(n)		TMP_ALLOC(n)
#define TMP_BALLOC(n)		TMP_ALLOC(n)
#define TMP_ALLOC(size)							\
  __gmp_tmp_debug_alloc (ASSERT_FILE, ASSERT_LINE,			\
			 __tmp_marker_inscope,				\
			 &__tmp_marker, __tmp_marker_name, size)
#define TMP_FREE_NAME(marker, marker_name)				\
  do {									\
    __gmp_tmp_debug_free  (ASSERT_FILE, ASSERT_LINE,			\
			   marker, &__tmp_marker,			\
			   __tmp_marker_name, marker_name);		\
  } while (0)
#endif /* WANT_TMP_DEBUG */


/* Allocating various types. */
#define TMP_ALLOC_TYPE(n,type)  ((type *) TMP_ALLOC ((n) * sizeof (type)))
#define TMP_SALLOC_TYPE(n,type) ((type *) TMP_SALLOC ((n) * sizeof (type)))
#define TMP_BALLOC_TYPE(n,type) ((type *) TMP_BALLOC ((n) * sizeof (type)))
#define TMP_ALLOC_LIMBS(n)      TMP_ALLOC_TYPE(n,mp_limb_t)
#define TMP_SALLOC_LIMBS(n)     TMP_SALLOC_TYPE(n,mp_limb_t)
#define TMP_BALLOC_LIMBS(n)     TMP_BALLOC_TYPE(n,mp_limb_t)
#define TMP_ALLOC_MP_PTRS(n)    TMP_ALLOC_TYPE(n,mp_ptr)
#define TMP_SALLOC_MP_PTRS(n)   TMP_SALLOC_TYPE(n,mp_ptr)
#define TMP_BALLOC_MP_PTRS(n)   TMP_BALLOC_TYPE(n,mp_ptr)

/* It's more efficient to allocate one block than many.  This is certainly
   true of the malloc methods, but it can even be true of alloca if that
   involves copying a chunk of stack (various RISCs), or a call to a stack
   bounds check (mingw).  In any case, when debugging keep separate blocks
   so a redzoning malloc debugger can protect each individually.  */
#define TMP_ALLOC_LIMBS_2(xp,xsize, yp,ysize)				\
  do {									\
    if (WANT_TMP_DEBUG)							\
      {									\
	(xp) = TMP_ALLOC_LIMBS (xsize);					\
	(yp) = TMP_ALLOC_LIMBS (ysize);					\
      }									\
    else								\
      {									\
	(xp) = TMP_ALLOC_LIMBS ((xsize) + (ysize));			\
	(yp) = (xp) + (xsize);						\
      }									\
  } while (0)
#define TMP_ALLOC_LIMBS_3(xp,xsize, yp,ysize, zp,zsize)			\
  do {									\
    if (WANT_TMP_DEBUG)							\
      {									\
	(xp) = TMP_ALLOC_LIMBS (xsize);					\
	(yp) = TMP_ALLOC_LIMBS (ysize);					\
	(zp) = TMP_ALLOC_LIMBS (zsize);					\
      }									\
    else								\
      {									\
	(xp) = TMP_ALLOC_LIMBS ((xsize) + (ysize) + (zsize));		\
	(yp) = (xp) + (xsize);						\
	(zp) = (yp) + (ysize);						\
      }									\
  } while (0)

/* From gmp.h, nicer names for internal use. */
#define CRAY_Pragma(str)               __GMP_CRAY_Pragma(str)
#define MPN_CMP(result, xp, yp, size)  __GMPN_CMP(result, xp, yp, size)
#define LIKELY(cond)                   __GMP_LIKELY(cond)
#define UNLIKELY(cond)                 __GMP_UNLIKELY(cond)

#define ABS(x) ((x) >= 0 ? (x) : -(x))
#define NEG_CAST(T,x) (- (__GMP_CAST (T, (x) + 1) - 1))
#define ABS_CAST(T,x) ((x) >= 0 ? __GMP_CAST (T, x) : NEG_CAST (T,x))
#undef MIN
#define MIN(l,o) ((l) < (o) ? (l) : (o))
#undef MAX
#define MAX(h,i) ((h) > (i) ? (h) : (i))
#define numberof(x)  (sizeof (x) / sizeof ((x)[0]))

/* Field access macros.  */
#define SIZ(x) ((x)->_mp_size)
#define ABSIZ(x) ABS (SIZ (x))
#define PTR(x) ((x)->_mp_d)
#define EXP(x) ((x)->_mp_exp)
#define PREC(x) ((x)->_mp_prec)
#define ALLOC(x) ((x)->_mp_alloc)
#define NUM(x) mpq_numref(x)
#define DEN(x) mpq_denref(x)

/* n-1 inverts any low zeros and the lowest one bit.  If n&(n-1) leaves zero
   then that lowest one bit must have been the only bit set.  n==0 will
   return true though, so avoid that.  */
#define POW2_P(n)  (((n) & ((n) - 1)) == 0)

/* This is intended for constant THRESHOLDs only, where the compiler
   can completely fold the result.  */
#define LOG2C(n) \
 (((n) >=    0x1) + ((n) >=    0x2) + ((n) >=    0x4) + ((n) >=    0x8) + \
  ((n) >=   0x10) + ((n) >=   0x20) + ((n) >=   0x40) + ((n) >=   0x80) + \
  ((n) >=  0x100) + ((n) >=  0x200) + ((n) >=  0x400) + ((n) >=  0x800) + \
  ((n) >= 0x1000) + ((n) >= 0x2000) + ((n) >= 0x4000) + ((n) >= 0x8000))

#define MP_LIMB_T_MAX      (~ (mp_limb_t) 0)

/* Must cast ULONG_MAX etc to unsigned long etc, since they might not be
   unsigned on a K&R compiler.  In particular the HP-UX 10 bundled K&R cc
   treats the plain decimal values in <limits.h> as signed.  */
#define ULONG_HIGHBIT      (ULONG_MAX ^ ((unsigned long) ULONG_MAX >> 1))
#define UINT_HIGHBIT       (UINT_MAX ^ ((unsigned) UINT_MAX >> 1))
#define USHRT_HIGHBIT      (USHRT_MAX ^ ((unsigned short) USHRT_MAX >> 1))
#define GMP_LIMB_HIGHBIT  (MP_LIMB_T_MAX ^ (MP_LIMB_T_MAX >> 1))

#if __GMP_MP_SIZE_T_INT
#define MP_SIZE_T_MAX      INT_MAX
#define MP_SIZE_T_MIN      INT_MIN
#else
#define MP_SIZE_T_MAX      LONG_MAX
#define MP_SIZE_T_MIN      LONG_MIN
#endif

/* mp_exp_t is the same as mp_size_t */
#define MP_EXP_T_MAX   MP_SIZE_T_MAX
#define MP_EXP_T_MIN   MP_SIZE_T_MIN

#define LONG_HIGHBIT       LONG_MIN
#define INT_HIGHBIT        INT_MIN
#define SHRT_HIGHBIT       SHRT_MIN


#define GMP_NUMB_HIGHBIT  (CNST_LIMB(1) << (GMP_NUMB_BITS-1))

#if GMP_NAIL_BITS == 0
#define GMP_NAIL_LOWBIT   CNST_LIMB(0)
#else
#define GMP_NAIL_LOWBIT   (CNST_LIMB(1) << GMP_NUMB_BITS)
#endif

#if GMP_NAIL_BITS != 0
/* Set various *_THRESHOLD values to be used for nails.  Thus we avoid using
   code that has not yet been qualified.  */

#undef  DC_DIV_QR_THRESHOLD
#define DC_DIV_QR_THRESHOLD              50

#undef DIVREM_1_NORM_THRESHOLD
#undef DIVREM_1_UNNORM_THRESHOLD
#undef MOD_1_NORM_THRESHOLD
#undef MOD_1_UNNORM_THRESHOLD
#undef USE_PREINV_DIVREM_1
#undef DIVREM_2_THRESHOLD
#undef DIVEXACT_1_THRESHOLD
#define DIVREM_1_NORM_THRESHOLD           MP_SIZE_T_MAX  /* no preinv */
#define DIVREM_1_UNNORM_THRESHOLD         MP_SIZE_T_MAX  /* no preinv */
#define MOD_1_NORM_THRESHOLD              MP_SIZE_T_MAX  /* no preinv */
#define MOD_1_UNNORM_THRESHOLD            MP_SIZE_T_MAX  /* no preinv */
#define USE_PREINV_DIVREM_1               0  /* no preinv */
#define DIVREM_2_THRESHOLD                MP_SIZE_T_MAX  /* no preinv */

/* mpn/generic/mul_fft.c is not nails-capable. */
#undef  MUL_FFT_THRESHOLD
#undef  SQR_FFT_THRESHOLD
#define MUL_FFT_THRESHOLD                MP_SIZE_T_MAX
#define SQR_FFT_THRESHOLD                MP_SIZE_T_MAX
#endif

/* Swap macros. */

#define MP_LIMB_T_SWAP(x, y)						\
  do {									\
    mp_limb_t __mp_limb_t_swap__tmp = (x);				\
    (x) = (y);								\
    (y) = __mp_limb_t_swap__tmp;					\
  } while (0)
#define MP_SIZE_T_SWAP(x, y)						\
  do {									\
    mp_size_t __mp_size_t_swap__tmp = (x);				\
    (x) = (y);								\
    (y) = __mp_size_t_swap__tmp;					\
  } while (0)

#define MP_PTR_SWAP(x, y)						\
  do {									\
    mp_ptr __mp_ptr_swap__tmp = (x);					\
    (x) = (y);								\
    (y) = __mp_ptr_swap__tmp;						\
  } while (0)
#define MP_SRCPTR_SWAP(x, y)						\
  do {									\
    mp_srcptr __mp_srcptr_swap__tmp = (x);				\
    (x) = (y);								\
    (y) = __mp_srcptr_swap__tmp;					\
  } while (0)

#define MPN_PTR_SWAP(xp,xs, yp,ys)					\
  do {									\
    MP_PTR_SWAP (xp, yp);						\
    MP_SIZE_T_SWAP (xs, ys);						\
  } while(0)
#define MPN_SRCPTR_SWAP(xp,xs, yp,ys)					\
  do {									\
    MP_SRCPTR_SWAP (xp, yp);						\
    MP_SIZE_T_SWAP (xs, ys);						\
  } while(0)

#define MPZ_PTR_SWAP(x, y)						\
  do {									\
    mpz_ptr __mpz_ptr_swap__tmp = (x);					\
    (x) = (y);								\
    (y) = __mpz_ptr_swap__tmp;						\
  } while (0)
#define MPZ_SRCPTR_SWAP(x, y)						\
  do {									\
    mpz_srcptr __mpz_srcptr_swap__tmp = (x);				\
    (x) = (y);								\
    (y) = __mpz_srcptr_swap__tmp;					\
  } while (0)

#define MPQ_PTR_SWAP(x, y)						\
  do {                                                                  \
    mpq_ptr __mpq_ptr_swap__tmp = (x);					\
    (x) = (y);                                                          \
    (y) = __mpq_ptr_swap__tmp;						\
  } while (0)
#define MPQ_SRCPTR_SWAP(x, y)                                           \
  do {                                                                  \
    mpq_srcptr __mpq_srcptr_swap__tmp = (x);                            \
    (x) = (y);                                                          \
    (y) = __mpq_srcptr_swap__tmp;                                       \
  } while (0)


/* Enhancement: __gmp_allocate_func could have "__attribute__ ((malloc))",
   but current gcc (3.0) doesn't seem to support that.  */
__GMP_DECLSPEC extern void * (*__gmp_allocate_func) (size_t);
__GMP_DECLSPEC extern void * (*__gmp_reallocate_func) (void *, size_t, size_t);
__GMP_DECLSPEC extern void   (*__gmp_free_func) (void *, size_t);

__GMP_DECLSPEC void *__gmp_default_allocate (size_t);
__GMP_DECLSPEC void *__gmp_default_reallocate (void *, size_t, size_t);
__GMP_DECLSPEC void __gmp_default_free (void *, size_t);

#define __GMP_ALLOCATE_FUNC_TYPE(n,type) \
  ((type *) (*__gmp_allocate_func) ((n) * sizeof (type)))
#define __GMP_ALLOCATE_FUNC_LIMBS(n)   __GMP_ALLOCATE_FUNC_TYPE (n, mp_limb_t)

#define __GMP_REALLOCATE_FUNC_TYPE(p, old_size, new_size, type)		\
  ((type *) (*__gmp_reallocate_func)					\
   (p, (old_size) * sizeof (type), (new_size) * sizeof (type)))
#define __GMP_REALLOCATE_FUNC_LIMBS(p, old_size, new_size)		\
  __GMP_REALLOCATE_FUNC_TYPE(p, old_size, new_size, mp_limb_t)

#define __GMP_FREE_FUNC_TYPE(p,n,type) (*__gmp_free_func) (p, (n) * sizeof (type))
#define __GMP_FREE_FUNC_LIMBS(p,n)     __GMP_FREE_FUNC_TYPE (p, n, mp_limb_t)

#define __GMP_REALLOCATE_FUNC_MAYBE(ptr, oldsize, newsize)		\
  do {									\
    if ((oldsize) != (newsize))						\
      (ptr) = (*__gmp_reallocate_func) (ptr, oldsize, newsize);		\
  } while (0)

#define __GMP_REALLOCATE_FUNC_MAYBE_TYPE(ptr, oldsize, newsize, type)	\
  do {									\
    if ((oldsize) != (newsize))						\
      (ptr) = (type *) (*__gmp_reallocate_func)				\
	(ptr, (oldsize) * sizeof (type), (newsize) * sizeof (type));	\
  } while (0)


/* Dummy for non-gcc, code involving it will go dead. */
#if ! defined (__GNUC__) || __GNUC__ < 2
#define __builtin_constant_p(x)   0
#endif


/* In gcc 2.96 and up on i386, tail calls are optimized to jumps if the
   stack usage is compatible.  __attribute__ ((regparm (N))) helps by
   putting leading parameters in registers, avoiding extra stack.

   regparm cannot be used with calls going through the PLT, because the
   binding code there may clobber the registers (%eax, %edx, %ecx) used for
   the regparm parameters.  Calls to local (ie. static) functions could
   still use this, if we cared to differentiate locals and globals.

   On athlon-unknown-freebsd4.9 with gcc 3.3.3, regparm cannot be used with
   -p or -pg profiling, since that version of gcc doesn't realize the
   .mcount calls will clobber the parameter registers.  Other systems are
   ok, like debian with glibc 2.3.2 (mcount doesn't clobber), but we don't
   bother to try to detect this.  regparm is only an optimization so we just
   disable it when profiling (profiling being a slowdown anyway).  */

#if HAVE_HOST_CPU_FAMILY_x86 && __GMP_GNUC_PREREQ (2,96) && ! defined (PIC) \
  && ! WANT_PROFILING_PROF && ! WANT_PROFILING_GPROF
#define USE_LEADING_REGPARM 1
#else
#define USE_LEADING_REGPARM 0
#endif

/* Macros for altering parameter order according to regparm usage. */
#if USE_LEADING_REGPARM
#define REGPARM_2_1(a,b,x)    x,a,b
#define REGPARM_3_1(a,b,c,x)  x,a,b,c
#define REGPARM_ATTR(n) __attribute__ ((regparm (n)))
#else
#define REGPARM_2_1(a,b,x)    a,b,x
#define REGPARM_3_1(a,b,c,x)  a,b,c,x
#define REGPARM_ATTR(n)
#endif


/* ASM_L gives a local label for a gcc asm block, for use when temporary
   local labels like "1:" might not be available, which is the case for
   instance on the x86s (the SCO assembler doesn't support them).

   The label generated is made unique by including "%=" which is a unique
   number for each insn.  This ensures the same name can be used in multiple
   asm blocks, perhaps via a macro.  Since jumps between asm blocks are not
   allowed there's no need for a label to be usable outside a single
   block.  */

#define ASM_L(name)  LSYM_PREFIX "asm_%=_" #name


#if defined (__GNUC__) && HAVE_HOST_CPU_FAMILY_x86
#if 0
/* FIXME: Check that these actually improve things.
   FIXME: Need a cld after each std.
   FIXME: Can't have inputs in clobbered registers, must describe them as
   dummy outputs, and add volatile. */
#define MPN_COPY_INCR(DST, SRC, N)					\
  __asm__ ("cld\n\trep\n\tmovsl" : :					\
	   "D" (DST), "S" (SRC), "c" (N) :				\
	   "cx", "di", "si", "memory")
#define MPN_COPY_DECR(DST, SRC, N)					\
  __asm__ ("std\n\trep\n\tmovsl" : :					\
	   "D" ((DST) + (N) - 1), "S" ((SRC) + (N) - 1), "c" (N) :	\
	   "cx", "di", "si", "memory")
#endif
#endif


__GMP_DECLSPEC void __gmpz_aorsmul_1 (REGPARM_3_1 (mpz_ptr, mpz_srcptr, mp_limb_t, mp_size_t)) REGPARM_ATTR(1);
#define mpz_aorsmul_1(w,u,v,sub)  __gmpz_aorsmul_1 (REGPARM_3_1 (w, u, v, sub))

#define mpz_n_pow_ui __gmpz_n_pow_ui
__GMP_DECLSPEC void    mpz_n_pow_ui (mpz_ptr, mp_srcptr, mp_size_t, unsigned long);


#define mpn_addmul_1c __MPN(addmul_1c)
__GMP_DECLSPEC mp_limb_t mpn_addmul_1c (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t);

#ifndef mpn_addmul_2  /* if not done with cpuvec in a fat binary */
#define mpn_addmul_2 __MPN(addmul_2)
__GMP_DECLSPEC mp_limb_t mpn_addmul_2 (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);
#endif

#define mpn_addmul_3 __MPN(addmul_3)
__GMP_DECLSPEC mp_limb_t mpn_addmul_3 (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

#define mpn_addmul_4 __MPN(addmul_4)
__GMP_DECLSPEC mp_limb_t mpn_addmul_4 (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

#define mpn_addmul_5 __MPN(addmul_5)
__GMP_DECLSPEC mp_limb_t mpn_addmul_5 (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

#define mpn_addmul_6 __MPN(addmul_6)
__GMP_DECLSPEC mp_limb_t mpn_addmul_6 (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

#define mpn_addmul_7 __MPN(addmul_7)
__GMP_DECLSPEC mp_limb_t mpn_addmul_7 (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

#define mpn_addmul_8 __MPN(addmul_8)
__GMP_DECLSPEC mp_limb_t mpn_addmul_8 (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

/* Alternative entry point in mpn_addmul_2 for the benefit of mpn_sqr_basecase.  */
#define mpn_addmul_2s __MPN(addmul_2s)
__GMP_DECLSPEC mp_limb_t mpn_addmul_2s (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

/* mpn_addlsh1_n(c,a,b,n), when it exists, sets {c,n} to {a,n}+2*{b,n}, and
   returns the carry out (0, 1 or 2). Use _ip1 when a=c. */
#ifndef mpn_addlsh1_n  /* if not done with cpuvec in a fat binary */
#define mpn_addlsh1_n __MPN(addlsh1_n)
__GMP_DECLSPEC mp_limb_t mpn_addlsh1_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
#endif
#define mpn_addlsh1_nc __MPN(addlsh1_nc)
__GMP_DECLSPEC mp_limb_t mpn_addlsh1_nc (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);
#if HAVE_NATIVE_mpn_addlsh1_n && ! HAVE_NATIVE_mpn_addlsh1_n_ip1
#define mpn_addlsh1_n_ip1(dst,src,n) mpn_addlsh1_n(dst,dst,src,n)
#define HAVE_NATIVE_mpn_addlsh1_n_ip1 1
#else
#define mpn_addlsh1_n_ip1 __MPN(addlsh1_n_ip1)
__GMP_DECLSPEC mp_limb_t mpn_addlsh1_n_ip1 (mp_ptr, mp_srcptr, mp_size_t);
#endif
#if HAVE_NATIVE_mpn_addlsh1_nc && ! HAVE_NATIVE_mpn_addlsh1_nc_ip1
#define mpn_addlsh1_nc_ip1(dst,src,n,c) mpn_addlsh1_nc(dst,dst,src,n,c)
#define HAVE_NATIVE_mpn_addlsh1_nc_ip1 1
#else
#define mpn_addlsh1_nc_ip1 __MPN(addlsh1_nc_ip1)
__GMP_DECLSPEC mp_limb_t mpn_addlsh1_nc_ip1 (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);
#endif

#ifndef mpn_addlsh2_n  /* if not done with cpuvec in a fat binary */
/* mpn_addlsh2_n(c,a,b,n), when it exists, sets {c,n} to {a,n}+4*{b,n}, and
   returns the carry out (0, ..., 4). Use _ip1 when a=c. */
#define mpn_addlsh2_n __MPN(addlsh2_n)
__GMP_DECLSPEC mp_limb_t mpn_addlsh2_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
#endif
#define mpn_addlsh2_nc __MPN(addlsh2_nc)
__GMP_DECLSPEC mp_limb_t mpn_addlsh2_nc (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);
#if HAVE_NATIVE_mpn_addlsh2_n && ! HAVE_NATIVE_mpn_addlsh2_n_ip1
#define mpn_addlsh2_n_ip1(dst,src,n) mpn_addlsh2_n(dst,dst,src,n)
#define HAVE_NATIVE_mpn_addlsh2_n_ip1 1
#else
#define mpn_addlsh2_n_ip1 __MPN(addlsh2_n_ip1)
__GMP_DECLSPEC mp_limb_t mpn_addlsh2_n_ip1 (mp_ptr, mp_srcptr, mp_size_t);
#endif
#if HAVE_NATIVE_mpn_addlsh2_nc && ! HAVE_NATIVE_mpn_addlsh2_nc_ip1
#define mpn_addlsh2_nc_ip1(dst,src,n,c) mpn_addlsh2_nc(dst,dst,src,n,c)
#define HAVE_NATIVE_mpn_addlsh2_nc_ip1 1
#else
#define mpn_addlsh2_nc_ip1 __MPN(addlsh2_nc_ip1)
__GMP_DECLSPEC mp_limb_t mpn_addlsh2_nc_ip1 (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);
#endif

/* mpn_addlsh_n(c,a,b,n,k), when it exists, sets {c,n} to {a,n}+2^k*{b,n}, and
   returns the carry out (0, ..., 2^k). Use _ip1 when a=c. */
#define mpn_addlsh_n __MPN(addlsh_n)
__GMP_DECLSPEC mp_limb_t mpn_addlsh_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, unsigned int);
#define mpn_addlsh_nc __MPN(addlsh_nc)
__GMP_DECLSPEC mp_limb_t mpn_addlsh_nc (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, unsigned int, mp_limb_t);
#if HAVE_NATIVE_mpn_addlsh_n && ! HAVE_NATIVE_mpn_addlsh_n_ip1
#define mpn_addlsh_n_ip1(dst,src,n,s) mpn_addlsh_n(dst,dst,src,n,s)
#define HAVE_NATIVE_mpn_addlsh_n_ip1 1
#else
#define mpn_addlsh_n_ip1 __MPN(addlsh_n_ip1)
  __GMP_DECLSPEC mp_limb_t mpn_addlsh_n_ip1 (mp_ptr, mp_srcptr, mp_size_t, unsigned int);
#endif
#if HAVE_NATIVE_mpn_addlsh_nc && ! HAVE_NATIVE_mpn_addlsh_nc_ip1
#define mpn_addlsh_nc_ip1(dst,src,n,s,c) mpn_addlsh_nc(dst,dst,src,n,s,c)
#define HAVE_NATIVE_mpn_addlsh_nc_ip1 1
#else
#define mpn_addlsh_nc_ip1 __MPN(addlsh_nc_ip1)
__GMP_DECLSPEC mp_limb_t mpn_addlsh_nc_ip1 (mp_ptr, mp_srcptr, mp_size_t, unsigned int, mp_limb_t);
#endif

#ifndef mpn_sublsh1_n  /* if not done with cpuvec in a fat binary */
/* mpn_sublsh1_n(c,a,b,n), when it exists, sets {c,n} to {a,n}-2*{b,n}, and
   returns the borrow out (0, 1 or 2). Use _ip1 when a=c. */
#define mpn_sublsh1_n __MPN(sublsh1_n)
__GMP_DECLSPEC mp_limb_t mpn_sublsh1_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
#endif
#define mpn_sublsh1_nc __MPN(sublsh1_nc)
__GMP_DECLSPEC mp_limb_t mpn_sublsh1_nc (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);
#if HAVE_NATIVE_mpn_sublsh1_n && ! HAVE_NATIVE_mpn_sublsh1_n_ip1
#define mpn_sublsh1_n_ip1(dst,src,n) mpn_sublsh1_n(dst,dst,src,n)
#define HAVE_NATIVE_mpn_sublsh1_n_ip1 1
#else
#define mpn_sublsh1_n_ip1 __MPN(sublsh1_n_ip1)
__GMP_DECLSPEC mp_limb_t mpn_sublsh1_n_ip1 (mp_ptr, mp_srcptr, mp_size_t);
#endif
#if HAVE_NATIVE_mpn_sublsh1_nc && ! HAVE_NATIVE_mpn_sublsh1_nc_ip1
#define mpn_sublsh1_nc_ip1(dst,src,n,c) mpn_sublsh1_nc(dst,dst,src,n,c)
#define HAVE_NATIVE_mpn_sublsh1_nc_ip1 1
#else
#define mpn_sublsh1_nc_ip1 __MPN(sublsh1_nc_ip1)
__GMP_DECLSPEC mp_limb_t mpn_sublsh1_nc_ip1 (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);
#endif

/* mpn_rsblsh1_n(c,a,b,n), when it exists, sets {c,n} to 2*{b,n}-{a,n}, and
   returns the carry out (-1, 0, 1).  */
#define mpn_rsblsh1_n __MPN(rsblsh1_n)
__GMP_DECLSPEC mp_limb_signed_t mpn_rsblsh1_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
#define mpn_rsblsh1_nc __MPN(rsblsh1_nc)
__GMP_DECLSPEC mp_limb_signed_t mpn_rsblsh1_nc (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);

/* mpn_sublsh2_n(c,a,b,n), when it exists, sets {c,n} to {a,n}-4*{b,n}, and
   returns the borrow out (0, ..., 4). Use _ip1 when a=c. */
#define mpn_sublsh2_n __MPN(sublsh2_n)
__GMP_DECLSPEC mp_limb_t mpn_sublsh2_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
#define mpn_sublsh2_nc __MPN(sublsh2_nc)
__GMP_DECLSPEC mp_limb_t mpn_sublsh2_nc (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);
#if HAVE_NATIVE_mpn_sublsh2_n && ! HAVE_NATIVE_mpn_sublsh2_n_ip1
#define mpn_sublsh2_n_ip1(dst,src,n) mpn_sublsh2_n(dst,dst,src,n)
#define HAVE_NATIVE_mpn_sublsh2_n_ip1 1
#else
#define mpn_sublsh2_n_ip1 __MPN(sublsh2_n_ip1)
__GMP_DECLSPEC mp_limb_t mpn_sublsh2_n_ip1 (mp_ptr, mp_srcptr, mp_size_t);
#endif
#if HAVE_NATIVE_mpn_sublsh2_nc && ! HAVE_NATIVE_mpn_sublsh2_nc_ip1
#define mpn_sublsh2_nc_ip1(dst,src,n,c) mpn_sublsh2_nc(dst,dst,src,n,c)
#define HAVE_NATIVE_mpn_sublsh2_nc_ip1 1
#else
#define mpn_sublsh2_nc_ip1 __MPN(sublsh2_nc_ip1)
__GMP_DECLSPEC mp_limb_t mpn_sublsh2_nc_ip1 (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);
#endif

/* mpn_sublsh_n(c,a,b,n,k), when it exists, sets {c,n} to {a,n}-2^k*{b,n}, and
   returns the carry out (0, ..., 2^k). Use _ip1 when a=c. */
#define mpn_sublsh_n __MPN(sublsh_n)
__GMP_DECLSPEC mp_limb_t mpn_sublsh_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, unsigned int);
#if HAVE_NATIVE_mpn_sublsh_n && ! HAVE_NATIVE_mpn_sublsh_n_ip1
#define mpn_sublsh_n_ip1(dst,src,n,s) mpn_sublsh_n(dst,dst,src,n,s)
#define HAVE_NATIVE_mpn_sublsh_n_ip1 1
#else
#define mpn_sublsh_n_ip1 __MPN(sublsh_n_ip1)
__GMP_DECLSPEC mp_limb_t mpn_sublsh_n_ip1 (mp_ptr, mp_srcptr, mp_size_t, unsigned int);
#endif
#if HAVE_NATIVE_mpn_sublsh_nc && ! HAVE_NATIVE_mpn_sublsh_nc_ip1
#define mpn_sublsh_nc_ip1(dst,src,n,s,c) mpn_sublsh_nc(dst,dst,src,n,s,c)
#define HAVE_NATIVE_mpn_sublsh_nc_ip1 1
#else
#define mpn_sublsh_nc_ip1 __MPN(sublsh_nc_ip1)
__GMP_DECLSPEC mp_limb_t mpn_sublsh_nc_ip1 (mp_ptr, mp_srcptr, mp_size_t, unsigned int, mp_limb_t);
#endif

/* mpn_rsblsh2_n(c,a,b,n), when it exists, sets {c,n} to 4*{b,n}-{a,n}, and
   returns the carry out (-1, ..., 3).  */
#define mpn_rsblsh2_n __MPN(rsblsh2_n)
__GMP_DECLSPEC mp_limb_signed_t mpn_rsblsh2_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
#define mpn_rsblsh2_nc __MPN(rsblsh2_nc)
__GMP_DECLSPEC mp_limb_signed_t mpn_rsblsh2_nc (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);

/* mpn_rsblsh_n(c,a,b,n,k), when it exists, sets {c,n} to 2^k*{b,n}-{a,n}, and
   returns the carry out (-1, 0, ..., 2^k-1).  */
#define mpn_rsblsh_n __MPN(rsblsh_n)
__GMP_DECLSPEC mp_limb_signed_t mpn_rsblsh_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, unsigned int);
#define mpn_rsblsh_nc __MPN(rsblsh_nc)
__GMP_DECLSPEC mp_limb_signed_t mpn_rsblsh_nc (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, unsigned int, mp_limb_t);

/* mpn_rsh1add_n(c,a,b,n), when it exists, sets {c,n} to ({a,n} + {b,n}) >> 1,
   and returns the bit rshifted out (0 or 1).  */
#define mpn_rsh1add_n __MPN(rsh1add_n)
__GMP_DECLSPEC mp_limb_t mpn_rsh1add_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
#define mpn_rsh1add_nc __MPN(rsh1add_nc)
__GMP_DECLSPEC mp_limb_t mpn_rsh1add_nc (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);

/* mpn_rsh1sub_n(c,a,b,n), when it exists, sets {c,n} to ({a,n} - {b,n}) >> 1,
   and returns the bit rshifted out (0 or 1).  If there's a borrow from the
   subtract, it's stored as a 1 in the high bit of c[n-1], like a twos
   complement negative.  */
#define mpn_rsh1sub_n __MPN(rsh1sub_n)
__GMP_DECLSPEC mp_limb_t mpn_rsh1sub_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
#define mpn_rsh1sub_nc __MPN(rsh1sub_nc)
__GMP_DECLSPEC mp_limb_t mpn_rsh1sub_nc (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);

#ifndef mpn_lshiftc  /* if not done with cpuvec in a fat binary */
#define mpn_lshiftc __MPN(lshiftc)
__GMP_DECLSPEC mp_limb_t mpn_lshiftc (mp_ptr, mp_srcptr, mp_size_t, unsigned int);
#endif

#define mpn_add_err1_n  __MPN(add_err1_n)
__GMP_DECLSPEC mp_limb_t mpn_add_err1_n (mp_ptr, mp_srcptr, mp_srcptr, mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);

#define mpn_add_err2_n  __MPN(add_err2_n)
__GMP_DECLSPEC mp_limb_t mpn_add_err2_n (mp_ptr, mp_srcptr, mp_srcptr, mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);

#define mpn_add_err3_n  __MPN(add_err3_n)
__GMP_DECLSPEC mp_limb_t mpn_add_err3_n (mp_ptr, mp_srcptr, mp_srcptr, mp_ptr, mp_srcptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);

#define mpn_sub_err1_n  __MPN(sub_err1_n)
__GMP_DECLSPEC mp_limb_t mpn_sub_err1_n (mp_ptr, mp_srcptr, mp_srcptr, mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);

#define mpn_sub_err2_n  __MPN(sub_err2_n)
__GMP_DECLSPEC mp_limb_t mpn_sub_err2_n (mp_ptr, mp_srcptr, mp_srcptr, mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);

#define mpn_sub_err3_n  __MPN(sub_err3_n)
__GMP_DECLSPEC mp_limb_t mpn_sub_err3_n (mp_ptr, mp_srcptr, mp_srcptr, mp_ptr, mp_srcptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);

#define mpn_add_n_sub_n __MPN(add_n_sub_n)
__GMP_DECLSPEC mp_limb_t mpn_add_n_sub_n (mp_ptr, mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

#define mpn_add_n_sub_nc __MPN(add_n_sub_nc)
__GMP_DECLSPEC mp_limb_t mpn_add_n_sub_nc (mp_ptr, mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);

#define mpn_addaddmul_1msb0 __MPN(addaddmul_1msb0)
__GMP_DECLSPEC mp_limb_t mpn_addaddmul_1msb0 (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t);

#define mpn_divrem_1c __MPN(divrem_1c)
__GMP_DECLSPEC mp_limb_t mpn_divrem_1c (mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t);

#define mpn_dump __MPN(dump)
__GMP_DECLSPEC void mpn_dump (mp_srcptr, mp_size_t);

#define mpn_fib2_ui __MPN(fib2_ui)
__GMP_DECLSPEC mp_size_t mpn_fib2_ui (mp_ptr, mp_ptr, unsigned long);

/* Remap names of internal mpn functions.  */
#define __clz_tab               __MPN(clz_tab)
#define mpn_udiv_w_sdiv		__MPN(udiv_w_sdiv)

#define mpn_jacobi_base __MPN(jacobi_base)
__GMP_DECLSPEC int mpn_jacobi_base (mp_limb_t, mp_limb_t, int) ATTRIBUTE_CONST;

#define mpn_jacobi_2 __MPN(jacobi_2)
__GMP_DECLSPEC int mpn_jacobi_2 (mp_srcptr, mp_srcptr, unsigned);

#define mpn_jacobi_n __MPN(jacobi_n)
__GMP_DECLSPEC int mpn_jacobi_n (mp_ptr, mp_ptr, mp_size_t, unsigned);

#define mpn_mod_1c __MPN(mod_1c)
__GMP_DECLSPEC mp_limb_t mpn_mod_1c (mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t) __GMP_ATTRIBUTE_PURE;

#define mpn_mul_1c __MPN(mul_1c)
__GMP_DECLSPEC mp_limb_t mpn_mul_1c (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t);

#define mpn_mul_2 __MPN(mul_2)
__GMP_DECLSPEC mp_limb_t mpn_mul_2 (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

#define mpn_mul_3 __MPN(mul_3)
__GMP_DECLSPEC mp_limb_t mpn_mul_3 (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

#define mpn_mul_4 __MPN(mul_4)
__GMP_DECLSPEC mp_limb_t mpn_mul_4 (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

#define mpn_mul_5 __MPN(mul_5)
__GMP_DECLSPEC mp_limb_t mpn_mul_5 (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

#define mpn_mul_6 __MPN(mul_6)
__GMP_DECLSPEC mp_limb_t mpn_mul_6 (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

#ifndef mpn_mul_basecase  /* if not done with cpuvec in a fat binary */
#define mpn_mul_basecase __MPN(mul_basecase)
__GMP_DECLSPEC void mpn_mul_basecase (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);
#endif

#define mpn_mullo_n __MPN(mullo_n)
__GMP_DECLSPEC void mpn_mullo_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

#ifndef mpn_mullo_basecase  /* if not done with cpuvec in a fat binary */
#define mpn_mullo_basecase __MPN(mullo_basecase)
__GMP_DECLSPEC void mpn_mullo_basecase (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
#endif

#ifndef mpn_sqr_basecase  /* if not done with cpuvec in a fat binary */
#define mpn_sqr_basecase __MPN(sqr_basecase)
__GMP_DECLSPEC void mpn_sqr_basecase (mp_ptr, mp_srcptr, mp_size_t);
#endif

#define mpn_sqrlo __MPN(sqrlo)
__GMP_DECLSPEC void mpn_sqrlo (mp_ptr, mp_srcptr, mp_size_t);

#define mpn_sqrlo_basecase __MPN(sqrlo_basecase)
__GMP_DECLSPEC void mpn_sqrlo_basecase (mp_ptr, mp_srcptr, mp_size_t);

#define mpn_mulmid_basecase __MPN(mulmid_basecase)
__GMP_DECLSPEC void mpn_mulmid_basecase (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);

#define mpn_mulmid_n __MPN(mulmid_n)
__GMP_DECLSPEC void mpn_mulmid_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

#define mpn_mulmid __MPN(mulmid)
__GMP_DECLSPEC void mpn_mulmid (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);

#define mpn_submul_1c __MPN(submul_1c)
__GMP_DECLSPEC mp_limb_t mpn_submul_1c (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t);

#ifndef mpn_redc_1  /* if not done with cpuvec in a fat binary */
#define mpn_redc_1 __MPN(redc_1)
__GMP_DECLSPEC mp_limb_t mpn_redc_1 (mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);
#endif

#ifndef mpn_redc_2  /* if not done with cpuvec in a fat binary */
#define mpn_redc_2 __MPN(redc_2)
__GMP_DECLSPEC mp_limb_t mpn_redc_2 (mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);
#endif

#define mpn_redc_n __MPN(redc_n)
__GMP_DECLSPEC void mpn_redc_n (mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);


#ifndef mpn_mod_1_1p_cps  /* if not done with cpuvec in a fat binary */
#define mpn_mod_1_1p_cps __MPN(mod_1_1p_cps)
__GMP_DECLSPEC void mpn_mod_1_1p_cps (mp_limb_t [4], mp_limb_t);
#endif
#ifndef mpn_mod_1_1p  /* if not done with cpuvec in a fat binary */
#define mpn_mod_1_1p __MPN(mod_1_1p)
__GMP_DECLSPEC mp_limb_t mpn_mod_1_1p (mp_srcptr, mp_size_t, mp_limb_t, const mp_limb_t [4]) __GMP_ATTRIBUTE_PURE;
#endif

#ifndef mpn_mod_1s_2p_cps  /* if not done with cpuvec in a fat binary */
#define mpn_mod_1s_2p_cps __MPN(mod_1s_2p_cps)
__GMP_DECLSPEC void mpn_mod_1s_2p_cps (mp_limb_t [5], mp_limb_t);
#endif
#ifndef mpn_mod_1s_2p  /* if not done with cpuvec in a fat binary */
#define mpn_mod_1s_2p __MPN(mod_1s_2p)
__GMP_DECLSPEC mp_limb_t mpn_mod_1s_2p (mp_srcptr, mp_size_t, mp_limb_t, const mp_limb_t [5]) __GMP_ATTRIBUTE_PURE;
#endif

#ifndef mpn_mod_1s_3p_cps  /* if not done with cpuvec in a fat binary */
#define mpn_mod_1s_3p_cps __MPN(mod_1s_3p_cps)
__GMP_DECLSPEC void mpn_mod_1s_3p_cps (mp_limb_t [6], mp_limb_t);
#endif
#ifndef mpn_mod_1s_3p  /* if not done with cpuvec in a fat binary */
#define mpn_mod_1s_3p __MPN(mod_1s_3p)
__GMP_DECLSPEC mp_limb_t mpn_mod_1s_3p (mp_srcptr, mp_size_t, mp_limb_t, const mp_limb_t [6]) __GMP_ATTRIBUTE_PURE;
#endif

#ifndef mpn_mod_1s_4p_cps  /* if not done with cpuvec in a fat binary */
#define mpn_mod_1s_4p_cps __MPN(mod_1s_4p_cps)
__GMP_DECLSPEC void mpn_mod_1s_4p_cps (mp_limb_t [7], mp_limb_t);
#endif
#ifndef mpn_mod_1s_4p  /* if not done with cpuvec in a fat binary */
#define mpn_mod_1s_4p __MPN(mod_1s_4p)
__GMP_DECLSPEC mp_limb_t mpn_mod_1s_4p (mp_srcptr, mp_size_t, mp_limb_t, const mp_limb_t [7]) __GMP_ATTRIBUTE_PURE;
#endif

#define mpn_bc_mulmod_bnm1 __MPN(bc_mulmod_bnm1)
__GMP_DECLSPEC void mpn_bc_mulmod_bnm1 (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_ptr);
#define mpn_mulmod_bnm1 __MPN(mulmod_bnm1)
__GMP_DECLSPEC void mpn_mulmod_bnm1 (mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);
#define mpn_mulmod_bnm1_next_size __MPN(mulmod_bnm1_next_size)
__GMP_DECLSPEC mp_size_t mpn_mulmod_bnm1_next_size (mp_size_t) ATTRIBUTE_CONST;
static inline mp_size_t
mpn_mulmod_bnm1_itch (mp_size_t rn, mp_size_t an, mp_size_t bn) {
  mp_size_t n, itch;
  n = rn >> 1;
  itch = rn + 4 +
    (an > n ? (bn > n ? rn : n) : 0);
  return itch;
}

#define mpn_sqrmod_bnm1 __MPN(sqrmod_bnm1)
__GMP_DECLSPEC void mpn_sqrmod_bnm1 (mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);
#define mpn_sqrmod_bnm1_next_size __MPN(sqrmod_bnm1_next_size)
__GMP_DECLSPEC mp_size_t mpn_sqrmod_bnm1_next_size (mp_size_t) ATTRIBUTE_CONST;
static inline mp_size_t
mpn_sqrmod_bnm1_itch (mp_size_t rn, mp_size_t an) {
  mp_size_t n, itch;
  n = rn >> 1;
  itch = rn + 3 +
    (an > n ? an : 0);
  return itch;
}

typedef __gmp_randstate_struct *gmp_randstate_ptr;
typedef const __gmp_randstate_struct *gmp_randstate_srcptr;

/* Pseudo-random number generator function pointers structure.  */
typedef struct {
  void (*randseed_fn) (gmp_randstate_t, mpz_srcptr);
  void (*randget_fn) (gmp_randstate_t, mp_ptr, unsigned long int);
  void (*randclear_fn) (gmp_randstate_t);
  void (*randiset_fn) (gmp_randstate_ptr, gmp_randstate_srcptr);
} gmp_randfnptr_t;

/* Macro to obtain a void pointer to the function pointers structure.  */
#define RNG_FNPTR(rstate) ((rstate)->_mp_algdata._mp_lc)

/* Macro to obtain a pointer to the generator's state.
   When used as a lvalue the rvalue needs to be cast to mp_ptr.  */
#define RNG_STATE(rstate) ((rstate)->_mp_seed->_mp_d)

/* Write a given number of random bits to rp.  */
#define _gmp_rand(rp, state, bits)					\
  do {									\
    gmp_randstate_ptr  __rstate = (state);				\
    (*((gmp_randfnptr_t *) RNG_FNPTR (__rstate))->randget_fn)		\
      (__rstate, rp, bits);						\
  } while (0)

__GMP_DECLSPEC void __gmp_randinit_mt_noseed (gmp_randstate_t);


/* __gmp_rands is the global state for the old-style random functions, and
   is also used in the test programs (hence the __GMP_DECLSPEC).

   There's no seeding here, so mpz_random etc will generate the same
   sequence every time.  This is not unlike the C library random functions
   if you don't seed them, so perhaps it's acceptable.  Digging up a seed
   from /dev/random or the like would work on many systems, but might
   encourage a false confidence, since it'd be pretty much impossible to do
   something that would work reliably everywhere.  In any case the new style
   functions are recommended to applications which care about randomness, so
   the old functions aren't too important.  */

__GMP_DECLSPEC extern char             __gmp_rands_initialized;
__GMP_DECLSPEC extern gmp_randstate_t  __gmp_rands;

#define RANDS								\
  ((__gmp_rands_initialized ? 0						\
    : (__gmp_rands_initialized = 1,					\
       __gmp_randinit_mt_noseed (__gmp_rands), 0)),			\
   __gmp_rands)

/* this is used by the test programs, to free memory */
#define RANDS_CLEAR()							\
  do {									\
    if (__gmp_rands_initialized)					\
      {									\
	__gmp_rands_initialized = 0;					\
	gmp_randclear (__gmp_rands);					\
      }									\
  } while (0)


/* For a threshold between algorithms A and B, size>=thresh is where B
   should be used.  Special value MP_SIZE_T_MAX means only ever use A, or
   value 0 means only ever use B.  The tests for these special values will
   be compile-time constants, so the compiler should be able to eliminate
   the code for the unwanted algorithm.  */

#if ! defined (__GNUC__) || __GNUC__ < 2
#define ABOVE_THRESHOLD(size,thresh)					\
  ((thresh) == 0							\
   || ((thresh) != MP_SIZE_T_MAX					\
       && (size) >= (thresh)))
#else
#define ABOVE_THRESHOLD(size,thresh)					\
  ((__builtin_constant_p (thresh) && (thresh) == 0)			\
   || (!(__builtin_constant_p (thresh) && (thresh) == MP_SIZE_T_MAX)	\
       && (size) >= (thresh)))
#endif
#define BELOW_THRESHOLD(size,thresh)  (! ABOVE_THRESHOLD (size, thresh))

#define MPN_TOOM22_MUL_MINSIZE    4
#define MPN_TOOM2_SQR_MINSIZE     4

#define MPN_TOOM33_MUL_MINSIZE   17
#define MPN_TOOM3_SQR_MINSIZE    17

#define MPN_TOOM44_MUL_MINSIZE   30
#define MPN_TOOM4_SQR_MINSIZE    30

#define MPN_TOOM6H_MUL_MINSIZE   46
#define MPN_TOOM6_SQR_MINSIZE    46

#define MPN_TOOM8H_MUL_MINSIZE   86
#define MPN_TOOM8_SQR_MINSIZE    86

#define MPN_TOOM32_MUL_MINSIZE   10
#define MPN_TOOM42_MUL_MINSIZE   10
#define MPN_TOOM43_MUL_MINSIZE   25
#define MPN_TOOM53_MUL_MINSIZE   17
#define MPN_TOOM54_MUL_MINSIZE   31
#define MPN_TOOM63_MUL_MINSIZE   49

#define MPN_TOOM42_MULMID_MINSIZE    4

#define   mpn_sqr_diagonal __MPN(sqr_diagonal)
__GMP_DECLSPEC void      mpn_sqr_diagonal (mp_ptr, mp_srcptr, mp_size_t);

#define mpn_sqr_diag_addlsh1 __MPN(sqr_diag_addlsh1)
__GMP_DECLSPEC void      mpn_sqr_diag_addlsh1 (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

#define   mpn_toom_interpolate_5pts __MPN(toom_interpolate_5pts)
__GMP_DECLSPEC void      mpn_toom_interpolate_5pts (mp_ptr, mp_ptr, mp_ptr, mp_size_t, mp_size_t, int, mp_limb_t);

enum toom6_flags {toom6_all_pos = 0, toom6_vm1_neg = 1, toom6_vm2_neg = 2};
#define   mpn_toom_interpolate_6pts __MPN(toom_interpolate_6pts)
__GMP_DECLSPEC void      mpn_toom_interpolate_6pts (mp_ptr, mp_size_t, enum toom6_flags, mp_ptr, mp_ptr, mp_ptr, mp_size_t);

enum toom7_flags { toom7_w1_neg = 1, toom7_w3_neg = 2 };
#define   mpn_toom_interpolate_7pts __MPN(toom_interpolate_7pts)
__GMP_DECLSPEC void      mpn_toom_interpolate_7pts (mp_ptr, mp_size_t, enum toom7_flags, mp_ptr, mp_ptr, mp_ptr, mp_ptr, mp_size_t, mp_ptr);

#define mpn_toom_interpolate_8pts __MPN(toom_interpolate_8pts)
__GMP_DECLSPEC void      mpn_toom_interpolate_8pts (mp_ptr, mp_size_t, mp_ptr, mp_ptr, mp_size_t, mp_ptr);

#define mpn_toom_interpolate_12pts __MPN(toom_interpolate_12pts)
__GMP_DECLSPEC void      mpn_toom_interpolate_12pts (mp_ptr, mp_ptr, mp_ptr, mp_ptr, mp_size_t, mp_size_t, int, mp_ptr);

#define mpn_toom_interpolate_16pts __MPN(toom_interpolate_16pts)
__GMP_DECLSPEC void      mpn_toom_interpolate_16pts (mp_ptr, mp_ptr, mp_ptr, mp_ptr, mp_ptr, mp_size_t, mp_size_t, int, mp_ptr);

#define   mpn_toom_couple_handling __MPN(toom_couple_handling)
__GMP_DECLSPEC void mpn_toom_couple_handling (mp_ptr, mp_size_t, mp_ptr, int, mp_size_t, int, int);

#define   mpn_toom_eval_dgr3_pm1 __MPN(toom_eval_dgr3_pm1)
__GMP_DECLSPEC int mpn_toom_eval_dgr3_pm1 (mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_size_t, mp_ptr);

#define   mpn_toom_eval_dgr3_pm2 __MPN(toom_eval_dgr3_pm2)
__GMP_DECLSPEC int mpn_toom_eval_dgr3_pm2 (mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_size_t, mp_ptr);

#define   mpn_toom_eval_pm1 __MPN(toom_eval_pm1)
__GMP_DECLSPEC int mpn_toom_eval_pm1 (mp_ptr, mp_ptr, unsigned, mp_srcptr, mp_size_t, mp_size_t, mp_ptr);

#define   mpn_toom_eval_pm2 __MPN(toom_eval_pm2)
__GMP_DECLSPEC int mpn_toom_eval_pm2 (mp_ptr, mp_ptr, unsigned, mp_srcptr, mp_size_t, mp_size_t, mp_ptr);

#define   mpn_toom_eval_pm2exp __MPN(toom_eval_pm2exp)
__GMP_DECLSPEC int mpn_toom_eval_pm2exp (mp_ptr, mp_ptr, unsigned, mp_srcptr, mp_size_t, mp_size_t, unsigned, mp_ptr);

#define   mpn_toom_eval_pm2rexp __MPN(toom_eval_pm2rexp)
__GMP_DECLSPEC int mpn_toom_eval_pm2rexp (mp_ptr, mp_ptr, unsigned, mp_srcptr, mp_size_t, mp_size_t, unsigned, mp_ptr);

#define   mpn_toom22_mul __MPN(toom22_mul)
__GMP_DECLSPEC void      mpn_toom22_mul (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

#define   mpn_toom32_mul __MPN(toom32_mul)
__GMP_DECLSPEC void      mpn_toom32_mul (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

#define   mpn_toom42_mul __MPN(toom42_mul)
__GMP_DECLSPEC void      mpn_toom42_mul (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

#define   mpn_toom52_mul __MPN(toom52_mul)
__GMP_DECLSPEC void      mpn_toom52_mul (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

#define   mpn_toom62_mul __MPN(toom62_mul)
__GMP_DECLSPEC void      mpn_toom62_mul (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

#define   mpn_toom2_sqr __MPN(toom2_sqr)
__GMP_DECLSPEC void      mpn_toom2_sqr (mp_ptr, mp_srcptr, mp_size_t, mp_ptr);

#define   mpn_toom33_mul __MPN(toom33_mul)
__GMP_DECLSPEC void      mpn_toom33_mul (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

#define   mpn_toom43_mul __MPN(toom43_mul)
__GMP_DECLSPEC void      mpn_toom43_mul (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

#define   mpn_toom53_mul __MPN(toom53_mul)
__GMP_DECLSPEC void      mpn_toom53_mul (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

#define   mpn_toom54_mul __MPN(toom54_mul)
__GMP_DECLSPEC void      mpn_toom54_mul (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

#define   mpn_toom63_mul __MPN(toom63_mul)
__GMP_DECLSPEC void      mpn_toom63_mul (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

#define   mpn_toom3_sqr __MPN(toom3_sqr)
__GMP_DECLSPEC void      mpn_toom3_sqr (mp_ptr, mp_srcptr, mp_size_t, mp_ptr);

#define   mpn_toom44_mul __MPN(toom44_mul)
__GMP_DECLSPEC void      mpn_toom44_mul (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

#define   mpn_toom4_sqr __MPN(toom4_sqr)
__GMP_DECLSPEC void      mpn_toom4_sqr (mp_ptr, mp_srcptr, mp_size_t, mp_ptr);

#define   mpn_toom6h_mul __MPN(toom6h_mul)
__GMP_DECLSPEC void      mpn_toom6h_mul (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

#define   mpn_toom6_sqr __MPN(toom6_sqr)
__GMP_DECLSPEC void      mpn_toom6_sqr (mp_ptr, mp_srcptr, mp_size_t, mp_ptr);

#define   mpn_toom8h_mul __MPN(toom8h_mul)
__GMP_DECLSPEC void      mpn_toom8h_mul (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

#define   mpn_toom8_sqr __MPN(toom8_sqr)
__GMP_DECLSPEC void      mpn_toom8_sqr (mp_ptr, mp_srcptr, mp_size_t, mp_ptr);

#define   mpn_toom42_mulmid __MPN(toom42_mulmid)
__GMP_DECLSPEC void      mpn_toom42_mulmid (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_ptr);

#define   mpn_fft_best_k __MPN(fft_best_k)
__GMP_DECLSPEC int       mpn_fft_best_k (mp_size_t, int) ATTRIBUTE_CONST;

#define   mpn_mul_fft __MPN(mul_fft)
__GMP_DECLSPEC mp_limb_t mpn_mul_fft (mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, int);

#define   mpn_mul_fft_full __MPN(mul_fft_full)
__GMP_DECLSPEC void      mpn_mul_fft_full (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);

#define   mpn_nussbaumer_mul __MPN(nussbaumer_mul)
__GMP_DECLSPEC void      mpn_nussbaumer_mul (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);

#define   mpn_fft_next_size __MPN(fft_next_size)
__GMP_DECLSPEC mp_size_t mpn_fft_next_size (mp_size_t, int) ATTRIBUTE_CONST;

#define   mpn_div_qr_1n_pi1 __MPN(div_qr_1n_pi1)
  __GMP_DECLSPEC mp_limb_t mpn_div_qr_1n_pi1 (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t, mp_limb_t);

#define   mpn_div_qr_2n_pi1 __MPN(div_qr_2n_pi1)
  __GMP_DECLSPEC mp_limb_t mpn_div_qr_2n_pi1 (mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t, mp_limb_t);

#define   mpn_div_qr_2u_pi1 __MPN(div_qr_2u_pi1)
  __GMP_DECLSPEC mp_limb_t mpn_div_qr_2u_pi1 (mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t, int, mp_limb_t);

#define   mpn_sbpi1_div_qr __MPN(sbpi1_div_qr)
__GMP_DECLSPEC mp_limb_t mpn_sbpi1_div_qr (mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t);

#define   mpn_sbpi1_div_q __MPN(sbpi1_div_q)
__GMP_DECLSPEC mp_limb_t mpn_sbpi1_div_q (mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t);

#define   mpn_sbpi1_divappr_q __MPN(sbpi1_divappr_q)
__GMP_DECLSPEC mp_limb_t mpn_sbpi1_divappr_q (mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t);

#define   mpn_dcpi1_div_qr __MPN(dcpi1_div_qr)
__GMP_DECLSPEC mp_limb_t mpn_dcpi1_div_qr (mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t, gmp_pi1_t *);
#define   mpn_dcpi1_div_qr_n __MPN(dcpi1_div_qr_n)
__GMP_DECLSPEC mp_limb_t mpn_dcpi1_div_qr_n (mp_ptr, mp_ptr, mp_srcptr, mp_size_t, gmp_pi1_t *, mp_ptr);

#define   mpn_dcpi1_div_q __MPN(dcpi1_div_q)
__GMP_DECLSPEC mp_limb_t mpn_dcpi1_div_q (mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t, gmp_pi1_t *);

#define   mpn_dcpi1_divappr_q __MPN(dcpi1_divappr_q)
__GMP_DECLSPEC mp_limb_t mpn_dcpi1_divappr_q (mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t, gmp_pi1_t *);
#define   mpn_dcpi1_divappr_q_n __MPN(dcpi1_divappr_q_n)
__GMP_DECLSPEC mp_limb_t mpn_dcpi1_divappr_q_n (mp_ptr, mp_ptr, mp_srcptr, mp_size_t, gmp_pi1_t *, mp_ptr);

#define   mpn_mu_div_qr __MPN(mu_div_qr)
__GMP_DECLSPEC mp_limb_t mpn_mu_div_qr (mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);
#define   mpn_mu_div_qr_itch __MPN(mu_div_qr_itch)
__GMP_DECLSPEC mp_size_t mpn_mu_div_qr_itch (mp_size_t, mp_size_t, int) ATTRIBUTE_CONST;
#define   mpn_mu_div_qr_choose_in __MPN(mu_div_qr_choose_in)
__GMP_DECLSPEC mp_size_t mpn_mu_div_qr_choose_in (mp_size_t, mp_size_t, int);

#define   mpn_preinv_mu_div_qr __MPN(preinv_mu_div_qr)
__GMP_DECLSPEC mp_limb_t mpn_preinv_mu_div_qr (mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);
#define   mpn_preinv_mu_div_qr_itch __MPN(preinv_mu_div_qr_itch)
__GMP_DECLSPEC mp_size_t mpn_preinv_mu_div_qr_itch (mp_size_t, mp_size_t, mp_size_t) ATTRIBUTE_CONST;

#define   mpn_mu_divappr_q __MPN(mu_divappr_q)
__GMP_DECLSPEC mp_limb_t mpn_mu_divappr_q (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);
#define   mpn_mu_divappr_q_itch __MPN(mu_divappr_q_itch)
__GMP_DECLSPEC mp_size_t mpn_mu_divappr_q_itch (mp_size_t, mp_size_t, int) ATTRIBUTE_CONST;
#define   mpn_mu_divappr_q_choose_in __MPN(mu_divappr_q_choose_in)
__GMP_DECLSPEC mp_size_t mpn_mu_divappr_q_choose_in (mp_size_t, mp_size_t, int);

#define   mpn_preinv_mu_divappr_q __MPN(preinv_mu_divappr_q)
__GMP_DECLSPEC mp_limb_t mpn_preinv_mu_divappr_q (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

#define   mpn_mu_div_q __MPN(mu_div_q)
__GMP_DECLSPEC mp_limb_t mpn_mu_div_q (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);
#define   mpn_mu_div_q_itch __MPN(mu_div_q_itch)
__GMP_DECLSPEC mp_size_t mpn_mu_div_q_itch (mp_size_t, mp_size_t, int) ATTRIBUTE_CONST;

#define  mpn_div_q __MPN(div_q)
__GMP_DECLSPEC void mpn_div_q (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);

#define   mpn_invert __MPN(invert)
__GMP_DECLSPEC void      mpn_invert (mp_ptr, mp_srcptr, mp_size_t, mp_ptr);
#define mpn_invert_itch(n)  mpn_invertappr_itch(n)

#define   mpn_ni_invertappr __MPN(ni_invertappr)
__GMP_DECLSPEC mp_limb_t mpn_ni_invertappr (mp_ptr, mp_srcptr, mp_size_t, mp_ptr);
#define   mpn_invertappr __MPN(invertappr)
__GMP_DECLSPEC mp_limb_t mpn_invertappr (mp_ptr, mp_srcptr, mp_size_t, mp_ptr);
#define mpn_invertappr_itch(n)  (2 * (n))

#define   mpn_binvert __MPN(binvert)
__GMP_DECLSPEC void      mpn_binvert (mp_ptr, mp_srcptr, mp_size_t, mp_ptr);
#define   mpn_binvert_itch __MPN(binvert_itch)
__GMP_DECLSPEC mp_size_t mpn_binvert_itch (mp_size_t) ATTRIBUTE_CONST;

#define mpn_bdiv_q_1 __MPN(bdiv_q_1)
__GMP_DECLSPEC mp_limb_t mpn_bdiv_q_1 (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);

#define mpn_pi1_bdiv_q_1 __MPN(pi1_bdiv_q_1)
__GMP_DECLSPEC mp_limb_t mpn_pi1_bdiv_q_1 (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t, int);

#define   mpn_sbpi1_bdiv_qr __MPN(sbpi1_bdiv_qr)
__GMP_DECLSPEC mp_limb_t mpn_sbpi1_bdiv_qr (mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t);

#define   mpn_sbpi1_bdiv_q __MPN(sbpi1_bdiv_q)
__GMP_DECLSPEC void      mpn_sbpi1_bdiv_q (mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t);

#define   mpn_dcpi1_bdiv_qr __MPN(dcpi1_bdiv_qr)
__GMP_DECLSPEC mp_limb_t mpn_dcpi1_bdiv_qr (mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t);
#define   mpn_dcpi1_bdiv_qr_n_itch __MPN(dcpi1_bdiv_qr_n_itch)
__GMP_DECLSPEC mp_size_t mpn_dcpi1_bdiv_qr_n_itch (mp_size_t) ATTRIBUTE_CONST;

#define   mpn_dcpi1_bdiv_qr_n __MPN(dcpi1_bdiv_qr_n)
__GMP_DECLSPEC mp_limb_t mpn_dcpi1_bdiv_qr_n (mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_ptr);
#define   mpn_dcpi1_bdiv_q __MPN(dcpi1_bdiv_q)
__GMP_DECLSPEC void      mpn_dcpi1_bdiv_q (mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t);

#define   mpn_dcpi1_bdiv_q_n __MPN(dcpi1_bdiv_q_n)
__GMP_DECLSPEC void      mpn_dcpi1_bdiv_q_n (mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_ptr);
#define   mpn_dcpi1_bdiv_q_n_itch __MPN(dcpi1_bdiv_q_n_itch)
__GMP_DECLSPEC mp_size_t mpn_dcpi1_bdiv_q_n_itch (mp_size_t) ATTRIBUTE_CONST;

#define   mpn_mu_bdiv_qr __MPN(mu_bdiv_qr)
__GMP_DECLSPEC mp_limb_t mpn_mu_bdiv_qr (mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);
#define   mpn_mu_bdiv_qr_itch __MPN(mu_bdiv_qr_itch)
__GMP_DECLSPEC mp_size_t mpn_mu_bdiv_qr_itch (mp_size_t, mp_size_t) ATTRIBUTE_CONST;

#define   mpn_mu_bdiv_q __MPN(mu_bdiv_q)
__GMP_DECLSPEC void      mpn_mu_bdiv_q (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);
#define   mpn_mu_bdiv_q_itch __MPN(mu_bdiv_q_itch)
__GMP_DECLSPEC mp_size_t mpn_mu_bdiv_q_itch (mp_size_t, mp_size_t) ATTRIBUTE_CONST;

#define   mpn_bdiv_qr __MPN(bdiv_qr)
__GMP_DECLSPEC mp_limb_t mpn_bdiv_qr (mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);
#define   mpn_bdiv_qr_itch __MPN(bdiv_qr_itch)
__GMP_DECLSPEC mp_size_t mpn_bdiv_qr_itch (mp_size_t, mp_size_t) ATTRIBUTE_CONST;

#define   mpn_bdiv_q __MPN(bdiv_q)
__GMP_DECLSPEC void      mpn_bdiv_q (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);
#define   mpn_bdiv_q_itch __MPN(bdiv_q_itch)
__GMP_DECLSPEC mp_size_t mpn_bdiv_q_itch (mp_size_t, mp_size_t) ATTRIBUTE_CONST;

#define   mpn_divexact __MPN(divexact)
__GMP_DECLSPEC void      mpn_divexact (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);
#define   mpn_divexact_itch __MPN(divexact_itch)
__GMP_DECLSPEC mp_size_t mpn_divexact_itch (mp_size_t, mp_size_t) ATTRIBUTE_CONST;

#ifndef mpn_bdiv_dbm1c  /* if not done with cpuvec in a fat binary */
#define   mpn_bdiv_dbm1c __MPN(bdiv_dbm1c)
__GMP_DECLSPEC mp_limb_t mpn_bdiv_dbm1c (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t);
#endif

#define   mpn_bdiv_dbm1(dst, src, size, divisor) \
  mpn_bdiv_dbm1c (dst, src, size, divisor, __GMP_CAST (mp_limb_t, 0))

#define   mpn_powm __MPN(powm)
__GMP_DECLSPEC void      mpn_powm (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_ptr);
#define   mpn_powlo __MPN(powlo)
__GMP_DECLSPEC void      mpn_powlo (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_size_t, mp_ptr);

#define mpn_sec_pi1_div_qr __MPN(sec_pi1_div_qr)
__GMP_DECLSPEC mp_limb_t mpn_sec_pi1_div_qr (mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t, mp_ptr);
#define mpn_sec_pi1_div_r __MPN(sec_pi1_div_r)
__GMP_DECLSPEC void mpn_sec_pi1_div_r (mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t, mp_ptr);


/* Override mpn_addlsh1_n, mpn_addlsh2_n, mpn_sublsh1_n, etc with mpn_addlsh_n,
   etc when !HAVE_NATIVE the former but HAVE_NATIVE_ the latter.  We then lie
   and say these macros represent native functions, but leave a trace by using
   the value 2 rather than 1.  */

#if HAVE_NATIVE_mpn_addlsh_n && ! HAVE_NATIVE_mpn_addlsh1_n
#undef mpn_addlsh1_n
#define mpn_addlsh1_n(a,b,c,d) mpn_addlsh_n(a,b,c,d,1)
#define HAVE_NATIVE_mpn_addlsh1_n 2
#endif

#if HAVE_NATIVE_mpn_addlsh_n && ! HAVE_NATIVE_mpn_addlsh2_n
#undef mpn_addlsh2_n
#define mpn_addlsh2_n(a,b,c,d) mpn_addlsh_n(a,b,c,d,2)
#define HAVE_NATIVE_mpn_addlsh2_n 2
#endif

#if HAVE_NATIVE_mpn_sublsh_n && ! HAVE_NATIVE_mpn_sublsh1_n
#undef mpn_sublsh1_n
#define mpn_sublsh1_n(a,b,c,d) mpn_sublsh_n(a,b,c,d,1)
#define HAVE_NATIVE_mpn_sublsh1_n 2
#endif

#if HAVE_NATIVE_mpn_sublsh_n && ! HAVE_NATIVE_mpn_sublsh2_n
#undef mpn_sublsh2_n
#define mpn_sublsh2_n(a,b,c,d) mpn_sublsh_n(a,b,c,d,2)
#define HAVE_NATIVE_mpn_sublsh2_n 2
#endif

#if HAVE_NATIVE_mpn_rsblsh_n && ! HAVE_NATIVE_mpn_rsblsh1_n
#undef mpn_rsblsh1_n
#define mpn_rsblsh1_n(a,b,c,d) mpn_rsblsh_n(a,b,c,d,1)
#define HAVE_NATIVE_mpn_rsblsh1_n 2
#endif

#if HAVE_NATIVE_mpn_rsblsh_n && ! HAVE_NATIVE_mpn_rsblsh2_n
#undef mpn_rsblsh2_n
#define mpn_rsblsh2_n(a,b,c,d) mpn_rsblsh_n(a,b,c,d,2)
#define HAVE_NATIVE_mpn_rsblsh2_n 2
#endif


#ifndef DIVEXACT_BY3_METHOD
#if GMP_NUMB_BITS % 2 == 0 && ! defined (HAVE_NATIVE_mpn_divexact_by3c)
#define DIVEXACT_BY3_METHOD 0	/* default to using mpn_bdiv_dbm1c */
#else
#define DIVEXACT_BY3_METHOD 1
#endif
#endif

#if DIVEXACT_BY3_METHOD == 0
#undef mpn_divexact_by3
#define mpn_divexact_by3(dst,src,size) \
  (3 & mpn_bdiv_dbm1 (dst, src, size, __GMP_CAST (mp_limb_t, GMP_NUMB_MASK / 3)))
/* override mpn_divexact_by3c defined in gmp.h */
/*
#undef mpn_divexact_by3c
#define mpn_divexact_by3c(dst,src,size,cy) \
  (3 & mpn_bdiv_dbm1c (dst, src, size, __GMP_CAST (mp_limb_t, GMP_NUMB_MASK / 3, GMP_NUMB_MASK / 3 * cy)))
*/
#endif

#if GMP_NUMB_BITS % 4 == 0
#define mpn_divexact_by5(dst,src,size) \
  (7 & 3 * mpn_bdiv_dbm1 (dst, src, size, __GMP_CAST (mp_limb_t, GMP_NUMB_MASK / 5)))
#endif

#if GMP_NUMB_BITS % 3 == 0
#define mpn_divexact_by7(dst,src,size) \
  (7 & 1 * mpn_bdiv_dbm1 (dst, src, size, __GMP_CAST (mp_limb_t, GMP_NUMB_MASK / 7)))
#endif

#if GMP_NUMB_BITS % 6 == 0
#define mpn_divexact_by9(dst,src,size) \
  (15 & 7 * mpn_bdiv_dbm1 (dst, src, size, __GMP_CAST (mp_limb_t, GMP_NUMB_MASK / 9)))
#endif

#if GMP_NUMB_BITS % 10 == 0
#define mpn_divexact_by11(dst,src,size) \
  (15 & 5 * mpn_bdiv_dbm1 (dst, src, size, __GMP_CAST (mp_limb_t, GMP_NUMB_MASK / 11)))
#endif

#if GMP_NUMB_BITS % 12 == 0
#define mpn_divexact_by13(dst,src,size) \
  (15 & 3 * mpn_bdiv_dbm1 (dst, src, size, __GMP_CAST (mp_limb_t, GMP_NUMB_MASK / 13)))
#endif

#if GMP_NUMB_BITS % 4 == 0
#define mpn_divexact_by15(dst,src,size) \
  (15 & 1 * mpn_bdiv_dbm1 (dst, src, size, __GMP_CAST (mp_limb_t, GMP_NUMB_MASK / 15)))
#endif

#define mpz_divexact_gcd  __gmpz_divexact_gcd
__GMP_DECLSPEC void    mpz_divexact_gcd (mpz_ptr, mpz_srcptr, mpz_srcptr);

#define mpz_prodlimbs  __gmpz_prodlimbs
__GMP_DECLSPEC mp_size_t mpz_prodlimbs (mpz_ptr, mp_ptr, mp_size_t);

#define mpz_oddfac_1  __gmpz_oddfac_1
__GMP_DECLSPEC void mpz_oddfac_1 (mpz_ptr, mp_limb_t, unsigned);

#define mpz_inp_str_nowhite __gmpz_inp_str_nowhite
#ifdef _GMP_H_HAVE_FILE
__GMP_DECLSPEC size_t  mpz_inp_str_nowhite (mpz_ptr, FILE *, int, int, size_t);
#endif

#define mpn_divisible_p __MPN(divisible_p)
__GMP_DECLSPEC int     mpn_divisible_p (mp_srcptr, mp_size_t, mp_srcptr, mp_size_t) __GMP_ATTRIBUTE_PURE;

#define   mpn_rootrem __MPN(rootrem)
__GMP_DECLSPEC mp_size_t mpn_rootrem (mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);

#define mpn_broot __MPN(broot)
__GMP_DECLSPEC void mpn_broot (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);

#define mpn_broot_invm1 __MPN(broot_invm1)
__GMP_DECLSPEC void mpn_broot_invm1 (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);

#define mpn_brootinv __MPN(brootinv)
__GMP_DECLSPEC void mpn_brootinv (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_ptr);

#define mpn_bsqrt __MPN(bsqrt)
__GMP_DECLSPEC void mpn_bsqrt (mp_ptr, mp_srcptr, mp_bitcnt_t, mp_ptr);

#define mpn_bsqrtinv __MPN(bsqrtinv)
__GMP_DECLSPEC int mpn_bsqrtinv (mp_ptr, mp_srcptr, mp_bitcnt_t, mp_ptr);

#if defined (_CRAY)
#define MPN_COPY_INCR(dst, src, n)					\
  do {									\
    int __i;		/* Faster on some Crays with plain int */	\
    _Pragma ("_CRI ivdep");						\
    for (__i = 0; __i < (n); __i++)					\
      (dst)[__i] = (src)[__i];						\
  } while (0)
#endif

/* used by test programs, hence __GMP_DECLSPEC */
#ifndef mpn_copyi  /* if not done with cpuvec in a fat binary */
#define mpn_copyi __MPN(copyi)
__GMP_DECLSPEC void mpn_copyi (mp_ptr, mp_srcptr, mp_size_t);
#endif

#if ! defined (MPN_COPY_INCR) && HAVE_NATIVE_mpn_copyi
#define MPN_COPY_INCR(dst, src, size)					\
  do {									\
    ASSERT ((size) >= 0);						\
    ASSERT (MPN_SAME_OR_INCR_P (dst, src, size));			\
    mpn_copyi (dst, src, size);						\
  } while (0)
#endif

/* Copy N limbs from SRC to DST incrementing, N==0 allowed.  */
#if ! defined (MPN_COPY_INCR)
#define MPN_COPY_INCR(dst, src, n)					\
  do {									\
    ASSERT ((n) >= 0);							\
    ASSERT (MPN_SAME_OR_INCR_P (dst, src, n));				\
    if ((n) != 0)							\
      {									\
	mp_size_t __n = (n) - 1;					\
	mp_ptr __dst = (dst);						\
	mp_srcptr __src = (src);					\
	mp_limb_t __x;							\
	__x = *__src++;							\
	if (__n != 0)							\
	  {								\
	    do								\
	      {								\
		*__dst++ = __x;						\
		__x = *__src++;						\
	      }								\
	    while (--__n);						\
	  }								\
	*__dst++ = __x;							\
      }									\
  } while (0)
#endif


#if defined (_CRAY)
#define MPN_COPY_DECR(dst, src, n)					\
  do {									\
    int __i;		/* Faster on some Crays with plain int */	\
    _Pragma ("_CRI ivdep");						\
    for (__i = (n) - 1; __i >= 0; __i--)				\
      (dst)[__i] = (src)[__i];						\
  } while (0)
#endif

/* used by test programs, hence __GMP_DECLSPEC */
#ifndef mpn_copyd  /* if not done with cpuvec in a fat binary */
#define mpn_copyd __MPN(copyd)
__GMP_DECLSPEC void mpn_copyd (mp_ptr, mp_srcptr, mp_size_t);
#endif

#if ! defined (MPN_COPY_DECR) && HAVE_NATIVE_mpn_copyd
#define MPN_COPY_DECR(dst, src, size)					\
  do {									\
    ASSERT ((size) >= 0);						\
    ASSERT (MPN_SAME_OR_DECR_P (dst, src, size));			\
    mpn_copyd (dst, src, size);						\
  } while (0)
#endif

/* Copy N limbs from SRC to DST decrementing, N==0 allowed.  */
#if ! defined (MPN_COPY_DECR)
#define MPN_COPY_DECR(dst, src, n)					\
  do {									\
    ASSERT ((n) >= 0);							\
    ASSERT (MPN_SAME_OR_DECR_P (dst, src, n));				\
    if ((n) != 0)							\
      {									\
	mp_size_t __n = (n) - 1;					\
	mp_ptr __dst = (dst) + __n;					\
	mp_srcptr __src = (src) + __n;					\
	mp_limb_t __x;							\
	__x = *__src--;							\
	if (__n != 0)							\
	  {								\
	    do								\
	      {								\
		*__dst-- = __x;						\
		__x = *__src--;						\
	      }								\
	    while (--__n);						\
	  }								\
	*__dst-- = __x;							\
      }									\
  } while (0)
#endif


#ifndef MPN_COPY
#define MPN_COPY(d,s,n)							\
  do {									\
    ASSERT (MPN_SAME_OR_SEPARATE_P (d, s, n));				\
    MPN_COPY_INCR (d, s, n);						\
  } while (0)
#endif


/* Set {dst,size} to the limbs of {src,size} in reverse order. */
#define MPN_REVERSE(dst, src, size)					\
  do {									\
    mp_ptr     __dst = (dst);						\
    mp_size_t  __size = (size);						\
    mp_srcptr  __src = (src) + __size - 1;				\
    mp_size_t  __i;							\
    ASSERT ((size) >= 0);						\
    ASSERT (! MPN_OVERLAP_P (dst, size, src, size));			\
    CRAY_Pragma ("_CRI ivdep");						\
    for (__i = 0; __i < __size; __i++)					\
      {									\
	*__dst = *__src;						\
	__dst++;							\
	__src--;							\
      }									\
  } while (0)


/* Zero n limbs at dst.

   For power and powerpc we want an inline stu/bdnz loop for zeroing.  On
   ppc630 for instance this is optimal since it can sustain only 1 store per
   cycle.

   gcc 2.95.x (for powerpc64 -maix64, or powerpc32) doesn't recognise the
   "for" loop in the generic code below can become stu/bdnz.  The do/while
   here helps it get to that.  The same caveat about plain -mpowerpc64 mode
   applies here as to __GMPN_COPY_INCR in gmp.h.

   xlc 3.1 already generates stu/bdnz from the generic C, and does so from
   this loop too.

   Enhancement: GLIBC does some trickery with dcbz to zero whole cache lines
   at a time.  MPN_ZERO isn't all that important in GMP, so it might be more
   trouble than it's worth to do the same, though perhaps a call to memset
   would be good when on a GNU system.  */

#if HAVE_HOST_CPU_FAMILY_power || HAVE_HOST_CPU_FAMILY_powerpc
#define MPN_FILL(dst, n, f)						\
  do {									\
    mp_ptr __dst = (dst) - 1;						\
    mp_size_t __n = (n);						\
    ASSERT (__n > 0);							\
    do									\
      *++__dst = (f);							\
    while (--__n);							\
  } while (0)
#endif

#ifndef MPN_FILL
#define MPN_FILL(dst, n, f)						\
  do {									\
    mp_ptr __dst = (dst);						\
    mp_size_t __n = (n);						\
    ASSERT (__n > 0);							\
    do									\
      *__dst++ = (f);							\
    while (--__n);							\
  } while (0)
#endif

#define MPN_ZERO(dst, n)						\
  do {									\
    ASSERT ((n) >= 0);							\
    if ((n) != 0)							\
      MPN_FILL (dst, n, CNST_LIMB (0));					\
  } while (0)

/* On the x86s repe/scasl doesn't seem useful, since it takes many cycles to
   start up and would need to strip a lot of zeros before it'd be faster
   than a simple cmpl loop.  Here are some times in cycles for
   std/repe/scasl/cld and cld/repe/scasl (the latter would be for stripping
   low zeros).

		std   cld
	   P5    18    16
	   P6    46    38
	   K6    36    13
	   K7    21    20
*/
#ifndef MPN_NORMALIZE
#define MPN_NORMALIZE(DST, NLIMBS) \
  do {									\
    while ((NLIMBS) > 0)						\
      {									\
	if ((DST)[(NLIMBS) - 1] != 0)					\
	  break;							\
	(NLIMBS)--;							\
      }									\
  } while (0)
#endif
#ifndef MPN_NORMALIZE_NOT_ZERO
#define MPN_NORMALIZE_NOT_ZERO(DST, NLIMBS)				\
  do {									\
    while (1)								\
      {									\
	ASSERT ((NLIMBS) >= 1);						\
	if ((DST)[(NLIMBS) - 1] != 0)					\
	  break;							\
	(NLIMBS)--;							\
      }									\
  } while (0)
#endif

/* Strip least significant zero limbs from {ptr,size} by incrementing ptr
   and decrementing size.  low should be ptr[0], and will be the new ptr[0]
   on returning.  The number in {ptr,size} must be non-zero, ie. size!=0 and
   somewhere a non-zero limb.  */
#define MPN_STRIP_LOW_ZEROS_NOT_ZERO(ptr, size, low)			\
  do {									\
    ASSERT ((size) >= 1);						\
    ASSERT ((low) == (ptr)[0]);						\
									\
    while ((low) == 0)							\
      {									\
	(size)--;							\
	ASSERT ((size) >= 1);						\
	(ptr)++;							\
	(low) = *(ptr);							\
      }									\
  } while (0)

/* Initialize X of type mpz_t with space for NLIMBS limbs.  X should be a
   temporary variable; it will be automatically cleared out at function
   return.  We use __x here to make it possible to accept both mpz_ptr and
   mpz_t arguments.  */
#define MPZ_TMP_INIT(X, NLIMBS)						\
  do {									\
    mpz_ptr __x = (X);							\
    ASSERT ((NLIMBS) >= 1);						\
    __x->_mp_alloc = (NLIMBS);						\
    __x->_mp_d = TMP_ALLOC_LIMBS (NLIMBS);				\
  } while (0)

#if WANT_ASSERT
static inline void *
_mpz_newalloc (mpz_ptr z, mp_size_t n)
{
  void * res = _mpz_realloc(z,n);
  /* If we are checking the code, force a random change to limbs. */
  ((mp_ptr) res)[0] = ~ ((mp_ptr) res)[ALLOC (z) - 1];
  return res;
}
#else
#define _mpz_newalloc _mpz_realloc
#endif
/* Realloc for an mpz_t WHAT if it has less than NEEDED limbs.  */
#define MPZ_REALLOC(z,n) (UNLIKELY ((n) > ALLOC(z))			\
			  ? (mp_ptr) _mpz_realloc(z,n)			\
			  : PTR(z))
#define MPZ_NEWALLOC(z,n) (UNLIKELY ((n) > ALLOC(z))			\
			   ? (mp_ptr) _mpz_newalloc(z,n)		\
			   : PTR(z))

#define MPZ_EQUAL_1_P(z)  (SIZ(z)==1 && PTR(z)[0] == 1)


/* MPN_FIB2_SIZE(n) is the size in limbs required by mpn_fib2_ui for fp and
   f1p.

   From Knuth vol 1 section 1.2.8, F[n] = phi^n/sqrt(5) rounded to the
   nearest integer, where phi=(1+sqrt(5))/2 is the golden ratio.  So the
   number of bits required is n*log_2((1+sqrt(5))/2) = n*0.6942419.

   The multiplier used is 23/32=0.71875 for efficient calculation on CPUs
   without good floating point.  There's +2 for rounding up, and a further
   +2 since at the last step x limbs are doubled into a 2x+1 limb region
   whereas the actual F[2k] value might be only 2x-1 limbs.

   Note that a division is done first, since on a 32-bit system it's at
   least conceivable to go right up to n==ULONG_MAX.  (F[2^32-1] would be
   about 380Mbytes, plus temporary workspace of about 1.2Gbytes here and
   whatever a multiply of two 190Mbyte numbers takes.)

   Enhancement: When GMP_NUMB_BITS is not a power of 2 the division could be
   worked into the multiplier.  */

#define MPN_FIB2_SIZE(n) \
  ((mp_size_t) ((n) / 32 * 23 / GMP_NUMB_BITS) + 4)


/* FIB_TABLE(n) returns the Fibonacci number F[n].  Must have n in the range
   -1 <= n <= FIB_TABLE_LIMIT (that constant in fib_table.h).

   FIB_TABLE_LUCNUM_LIMIT (in fib_table.h) is the largest n for which L[n] =
   F[n] + 2*F[n-1] fits in a limb.  */

__GMP_DECLSPEC extern const mp_limb_t __gmp_fib_table[];
#define FIB_TABLE(n)  (__gmp_fib_table[(n)+1])

extern const mp_limb_t __gmp_oddfac_table[];
extern const mp_limb_t __gmp_odd2fac_table[];
extern const unsigned char __gmp_fac2cnt_table[];
extern const mp_limb_t __gmp_limbroots_table[];

/* n^log <= GMP_NUMB_MAX, a limb can store log factors less than n */
static inline unsigned
log_n_max (mp_limb_t n)
{
  unsigned log;
  for (log = 8; n > __gmp_limbroots_table[log - 1]; log--);
  return log;
}

#define SIEVESIZE 512		/* FIXME: Allow gmp_init_primesieve to choose */
typedef struct
{
  unsigned long d;		   /* current index in s[] */
  unsigned long s0;		   /* number corresponding to s[0] */
  unsigned long sqrt_s0;	   /* misnomer for sqrt(s[SIEVESIZE-1]) */
  unsigned char s[SIEVESIZE + 1];  /* sieve table */
} gmp_primesieve_t;

#define gmp_init_primesieve __gmp_init_primesieve
__GMP_DECLSPEC void gmp_init_primesieve (gmp_primesieve_t *);

#define gmp_nextprime __gmp_nextprime
__GMP_DECLSPEC unsigned long int gmp_nextprime (gmp_primesieve_t *);

#define gmp_primesieve __gmp_primesieve
__GMP_DECLSPEC mp_limb_t gmp_primesieve (mp_ptr, mp_limb_t);


#ifndef MUL_TOOM22_THRESHOLD
#define MUL_TOOM22_THRESHOLD             30
#endif

#ifndef MUL_TOOM33_THRESHOLD
#define MUL_TOOM33_THRESHOLD            100
#endif

#ifndef MUL_TOOM44_THRESHOLD
#define MUL_TOOM44_THRESHOLD            300
#endif

#ifndef MUL_TOOM6H_THRESHOLD
#define MUL_TOOM6H_THRESHOLD            350
#endif

#ifndef SQR_TOOM6_THRESHOLD
#define SQR_TOOM6_THRESHOLD MUL_TOOM6H_THRESHOLD
#endif

#ifndef MUL_TOOM8H_THRESHOLD
#define MUL_TOOM8H_THRESHOLD            450
#endif

#ifndef SQR_TOOM8_THRESHOLD
#define SQR_TOOM8_THRESHOLD MUL_TOOM8H_THRESHOLD
#endif

#ifndef MUL_TOOM32_TO_TOOM43_THRESHOLD
#define MUL_TOOM32_TO_TOOM43_THRESHOLD  100
#endif

#ifndef MUL_TOOM32_TO_TOOM53_THRESHOLD
#define MUL_TOOM32_TO_TOOM53_THRESHOLD  110
#endif

#ifndef MUL_TOOM42_TO_TOOM53_THRESHOLD
#define MUL_TOOM42_TO_TOOM53_THRESHOLD  100
#endif

#ifndef MUL_TOOM42_TO_TOOM63_THRESHOLD
#define MUL_TOOM42_TO_TOOM63_THRESHOLD  110
#endif

#ifndef MUL_TOOM43_TO_TOOM54_THRESHOLD
#define MUL_TOOM43_TO_TOOM54_THRESHOLD  150
#endif

/* MUL_TOOM22_THRESHOLD_LIMIT is the maximum for MUL_TOOM22_THRESHOLD.  In a
   normal build MUL_TOOM22_THRESHOLD is a constant and we use that.  In a fat
   binary or tune program build MUL_TOOM22_THRESHOLD is a variable and a
   separate hard limit will have been defined.  Similarly for TOOM3.  */
#ifndef MUL_TOOM22_THRESHOLD_LIMIT
#define MUL_TOOM22_THRESHOLD_LIMIT  MUL_TOOM22_THRESHOLD
#endif
#ifndef MUL_TOOM33_THRESHOLD_LIMIT
#define MUL_TOOM33_THRESHOLD_LIMIT  MUL_TOOM33_THRESHOLD
#endif
#ifndef MULLO_BASECASE_THRESHOLD_LIMIT
#define MULLO_BASECASE_THRESHOLD_LIMIT  MULLO_BASECASE_THRESHOLD
#endif
#ifndef SQRLO_BASECASE_THRESHOLD_LIMIT
#define SQRLO_BASECASE_THRESHOLD_LIMIT  SQRLO_BASECASE_THRESHOLD
#endif
#ifndef SQRLO_DC_THRESHOLD_LIMIT
#define SQRLO_DC_THRESHOLD_LIMIT  SQRLO_DC_THRESHOLD
#endif

/* SQR_BASECASE_THRESHOLD is where mpn_sqr_basecase should take over from
   mpn_mul_basecase.  Default is to use mpn_sqr_basecase from 0.  (Note that we
   certainly always want it if there's a native assembler mpn_sqr_basecase.)

   If it turns out that mpn_toom2_sqr becomes faster than mpn_mul_basecase
   before mpn_sqr_basecase does, then SQR_BASECASE_THRESHOLD is the toom2
   threshold and SQR_TOOM2_THRESHOLD is 0.  This oddity arises more or less
   because SQR_TOOM2_THRESHOLD represents the size up to which mpn_sqr_basecase
   should be used, and that may be never.  */

#ifndef SQR_BASECASE_THRESHOLD
#define SQR_BASECASE_THRESHOLD            0  /* never use mpn_mul_basecase */
#endif

#ifndef SQR_TOOM2_THRESHOLD
#define SQR_TOOM2_THRESHOLD              50
#endif

#ifndef SQR_TOOM3_THRESHOLD
#define SQR_TOOM3_THRESHOLD             120
#endif

#ifndef SQR_TOOM4_THRESHOLD
#define SQR_TOOM4_THRESHOLD             400
#endif

/* See comments above about MUL_TOOM33_THRESHOLD_LIMIT.  */
#ifndef SQR_TOOM3_THRESHOLD_LIMIT
#define SQR_TOOM3_THRESHOLD_LIMIT  SQR_TOOM3_THRESHOLD
#endif

#ifndef MULMID_TOOM42_THRESHOLD
#define MULMID_TOOM42_THRESHOLD     MUL_TOOM22_THRESHOLD
#endif

#ifndef MULLO_BASECASE_THRESHOLD
#define MULLO_BASECASE_THRESHOLD          0  /* never use mpn_mul_basecase */
#endif

#ifndef MULLO_DC_THRESHOLD
#define MULLO_DC_THRESHOLD         (2*MUL_TOOM22_THRESHOLD)
#endif

#ifndef MULLO_MUL_N_THRESHOLD
#define MULLO_MUL_N_THRESHOLD      (2*MUL_FFT_THRESHOLD)
#endif

#ifndef SQRLO_BASECASE_THRESHOLD
#define SQRLO_BASECASE_THRESHOLD          0  /* never use mpn_sqr_basecase */
#endif

#ifndef SQRLO_DC_THRESHOLD
#define SQRLO_DC_THRESHOLD         (MULLO_DC_THRESHOLD)
#endif

#ifndef SQRLO_SQR_THRESHOLD
#define SQRLO_SQR_THRESHOLD        (MULLO_MUL_N_THRESHOLD)
#endif

#ifndef DC_DIV_QR_THRESHOLD
#define DC_DIV_QR_THRESHOLD        (2*MUL_TOOM22_THRESHOLD)
#endif

#ifndef DC_DIVAPPR_Q_THRESHOLD
#define DC_DIVAPPR_Q_THRESHOLD          200
#endif

#ifndef DC_BDIV_QR_THRESHOLD
#define DC_BDIV_QR_THRESHOLD       (2*MUL_TOOM22_THRESHOLD)
#endif

#ifndef DC_BDIV_Q_THRESHOLD
#define DC_BDIV_Q_THRESHOLD             180
#endif

#ifndef DIVEXACT_JEB_THRESHOLD
#define DIVEXACT_JEB_THRESHOLD           25
#endif

#ifndef INV_MULMOD_BNM1_THRESHOLD
#define INV_MULMOD_BNM1_THRESHOLD  (4*MULMOD_BNM1_THRESHOLD)
#endif

#ifndef INV_APPR_THRESHOLD
#define INV_APPR_THRESHOLD         INV_NEWTON_THRESHOLD
#endif

#ifndef INV_NEWTON_THRESHOLD
#define INV_NEWTON_THRESHOLD            200
#endif

#ifndef BINV_NEWTON_THRESHOLD
#define BINV_NEWTON_THRESHOLD           300
#endif

#ifndef MU_DIVAPPR_Q_THRESHOLD
#define MU_DIVAPPR_Q_THRESHOLD         2000
#endif

#ifndef MU_DIV_QR_THRESHOLD
#define MU_DIV_QR_THRESHOLD            2000
#endif

#ifndef MUPI_DIV_QR_THRESHOLD
#define MUPI_DIV_QR_THRESHOLD           200
#endif

#ifndef MU_BDIV_Q_THRESHOLD
#define MU_BDIV_Q_THRESHOLD            2000
#endif

#ifndef MU_BDIV_QR_THRESHOLD
#define MU_BDIV_QR_THRESHOLD           2000
#endif

#ifndef MULMOD_BNM1_THRESHOLD
#define MULMOD_BNM1_THRESHOLD            16
#endif

#ifndef SQRMOD_BNM1_THRESHOLD
#define SQRMOD_BNM1_THRESHOLD            16
#endif

#ifndef MUL_TO_MULMOD_BNM1_FOR_2NXN_THRESHOLD
#define MUL_TO_MULMOD_BNM1_FOR_2NXN_THRESHOLD  (INV_MULMOD_BNM1_THRESHOLD/2)
#endif

#if HAVE_NATIVE_mpn_addmul_2 || HAVE_NATIVE_mpn_redc_2

#ifndef REDC_1_TO_REDC_2_THRESHOLD
#define REDC_1_TO_REDC_2_THRESHOLD       15
#endif
#ifndef REDC_2_TO_REDC_N_THRESHOLD
#define REDC_2_TO_REDC_N_THRESHOLD      100
#endif

#else

#ifndef REDC_1_TO_REDC_N_THRESHOLD
#define REDC_1_TO_REDC_N_THRESHOLD      100
#endif

#endif /* HAVE_NATIVE_mpn_addmul_2 || HAVE_NATIVE_mpn_redc_2 */


/* First k to use for an FFT modF multiply.  A modF FFT is an order
   log(2^k)/log(2^(k-1)) algorithm, so k=3 is merely 1.5 like karatsuba,
   whereas k=4 is 1.33 which is faster than toom3 at 1.485.    */
#define FFT_FIRST_K  4

/* Threshold at which FFT should be used to do a modF NxN -> N multiply. */
#ifndef MUL_FFT_MODF_THRESHOLD
#define MUL_FFT_MODF_THRESHOLD   (MUL_TOOM33_THRESHOLD * 3)
#endif
#ifndef SQR_FFT_MODF_THRESHOLD
#define SQR_FFT_MODF_THRESHOLD   (SQR_TOOM3_THRESHOLD * 3)
#endif

/* Threshold at which FFT should be used to do an NxN -> 2N multiply.  This
   will be a size where FFT is using k=7 or k=8, since an FFT-k used for an
   NxN->2N multiply and not recursing into itself is an order
   log(2^k)/log(2^(k-2)) algorithm, so it'll be at least k=7 at 1.39 which
   is the first better than toom3.  */
#ifndef MUL_FFT_THRESHOLD
#define MUL_FFT_THRESHOLD   (MUL_FFT_MODF_THRESHOLD * 10)
#endif
#ifndef SQR_FFT_THRESHOLD
#define SQR_FFT_THRESHOLD   (SQR_FFT_MODF_THRESHOLD * 10)
#endif

/* Table of thresholds for successive modF FFT "k"s.  The first entry is
   where FFT_FIRST_K+1 should be used, the second FFT_FIRST_K+2,
   etc.  See mpn_fft_best_k(). */
#ifndef MUL_FFT_TABLE
#define MUL_FFT_TABLE							\
  { MUL_TOOM33_THRESHOLD * 4,   /* k=5 */				\
    MUL_TOOM33_THRESHOLD * 8,   /* k=6 */				\
    MUL_TOOM33_THRESHOLD * 16,  /* k=7 */				\
    MUL_TOOM33_THRESHOLD * 32,  /* k=8 */				\
    MUL_TOOM33_THRESHOLD * 96,  /* k=9 */				\
    MUL_TOOM33_THRESHOLD * 288, /* k=10 */				\
    0 }
#endif
#ifndef SQR_FFT_TABLE
#define SQR_FFT_TABLE							\
  { SQR_TOOM3_THRESHOLD * 4,   /* k=5 */				\
    SQR_TOOM3_THRESHOLD * 8,   /* k=6 */				\
    SQR_TOOM3_THRESHOLD * 16,  /* k=7 */				\
    SQR_TOOM3_THRESHOLD * 32,  /* k=8 */				\
    SQR_TOOM3_THRESHOLD * 96,  /* k=9 */				\
    SQR_TOOM3_THRESHOLD * 288, /* k=10 */				\
    0 }
#endif

struct fft_table_nk
{
  unsigned int n:27;
  unsigned int k:5;
};

#ifndef FFT_TABLE_ATTRS
#define FFT_TABLE_ATTRS   static const
#endif

#define MPN_FFT_TABLE_SIZE  16


#ifndef DC_DIV_QR_THRESHOLD
#define DC_DIV_QR_THRESHOLD    (3 * MUL_TOOM22_THRESHOLD)
#endif

#ifndef GET_STR_DC_THRESHOLD
#define GET_STR_DC_THRESHOLD             18
#endif

#ifndef GET_STR_PRECOMPUTE_THRESHOLD
#define GET_STR_PRECOMPUTE_THRESHOLD     35
#endif

#ifndef SET_STR_DC_THRESHOLD
#define SET_STR_DC_THRESHOLD            750
#endif

#ifndef SET_STR_PRECOMPUTE_THRESHOLD
#define SET_STR_PRECOMPUTE_THRESHOLD   2000
#endif

#ifndef FAC_ODD_THRESHOLD
#define FAC_ODD_THRESHOLD    35
#endif

#ifndef FAC_DSC_THRESHOLD
#define FAC_DSC_THRESHOLD   400
#endif

/* Return non-zero if xp,xsize and yp,ysize overlap.
   If xp+xsize<=yp there's no overlap, or if yp+ysize<=xp there's no
   overlap.  If both these are false, there's an overlap. */
#define MPN_OVERLAP_P(xp, xsize, yp, ysize)				\
  ((xp) + (xsize) > (yp) && (yp) + (ysize) > (xp))
#define MEM_OVERLAP_P(xp, xsize, yp, ysize)				\
  (   (char *) (xp) + (xsize) > (char *) (yp)				\
   && (char *) (yp) + (ysize) > (char *) (xp))

/* Return non-zero if xp,xsize and yp,ysize are either identical or not
   overlapping.  Return zero if they're partially overlapping. */
#define MPN_SAME_OR_SEPARATE_P(xp, yp, size)				\
  MPN_SAME_OR_SEPARATE2_P(xp, size, yp, size)
#define MPN_SAME_OR_SEPARATE2_P(xp, xsize, yp, ysize)			\
  ((xp) == (yp) || ! MPN_OVERLAP_P (xp, xsize, yp, ysize))

/* Return non-zero if dst,dsize and src,ssize are either identical or
   overlapping in a way suitable for an incrementing/decrementing algorithm.
   Return zero if they're partially overlapping in an unsuitable fashion. */
#define MPN_SAME_OR_INCR2_P(dst, dsize, src, ssize)			\
  ((dst) <= (src) || ! MPN_OVERLAP_P (dst, dsize, src, ssize))
#define MPN_SAME_OR_INCR_P(dst, src, size)				\
  MPN_SAME_OR_INCR2_P(dst, size, src, size)
#define MPN_SAME_OR_DECR2_P(dst, dsize, src, ssize)			\
  ((dst) >= (src) || ! MPN_OVERLAP_P (dst, dsize, src, ssize))
#define MPN_SAME_OR_DECR_P(dst, src, size)				\
  MPN_SAME_OR_DECR2_P(dst, size, src, size)


/* ASSERT() is a private assertion checking scheme, similar to <assert.h>.
   ASSERT() does the check only if WANT_ASSERT is selected, ASSERT_ALWAYS()
   does it always.  Generally assertions are meant for development, but
   might help when looking for a problem later too.  */

#ifdef __LINE__
#define ASSERT_LINE  __LINE__
#else
#define ASSERT_LINE  -1
#endif

#ifdef __FILE__
#define ASSERT_FILE  __FILE__
#else
#define ASSERT_FILE  ""
#endif

__GMP_DECLSPEC void __gmp_assert_header (const char *, int);
__GMP_DECLSPEC void __gmp_assert_fail (const char *, int, const char *) ATTRIBUTE_NORETURN;

#define ASSERT_FAIL(expr)  __gmp_assert_fail (ASSERT_FILE, ASSERT_LINE, #expr)

#define ASSERT_ALWAYS(expr)						\
  do {									\
    if (UNLIKELY (!(expr)))						\
      ASSERT_FAIL (expr);						\
  } while (0)

#if WANT_ASSERT
#define ASSERT(expr)   ASSERT_ALWAYS (expr)
#else
#define ASSERT(expr)   do {} while (0)
#endif


/* ASSERT_CARRY checks the expression is non-zero, and ASSERT_NOCARRY checks
   that it's zero.  In both cases if assertion checking is disabled the
   expression is still evaluated.  These macros are meant for use with
   routines like mpn_add_n() where the return value represents a carry or
   whatever that should or shouldn't occur in some context.  For example,
   ASSERT_NOCARRY (mpn_add_n (rp, s1p, s2p, size)); */
#if WANT_ASSERT
#define ASSERT_CARRY(expr)     ASSERT_ALWAYS ((expr) != 0)
#define ASSERT_NOCARRY(expr)   ASSERT_ALWAYS ((expr) == 0)
#else
#define ASSERT_CARRY(expr)     (expr)
#define ASSERT_NOCARRY(expr)   (expr)
#endif


/* ASSERT_CODE includes code when assertion checking is wanted.  This is the
   same as writing "#if WANT_ASSERT", but more compact.  */
#if WANT_ASSERT
#define ASSERT_CODE(expr)  expr
#else
#define ASSERT_CODE(expr)
#endif


/* Test that an mpq_t is in fully canonical form.  This can be used as
   protection on routines like mpq_equal which give wrong results on
   non-canonical inputs.  */
#if WANT_ASSERT
#define ASSERT_MPQ_CANONICAL(q)						\
  do {									\
    ASSERT (q->_mp_den._mp_size > 0);					\
    if (q->_mp_num._mp_size == 0)					\
      {									\
	/* zero should be 0/1 */					\
	ASSERT (mpz_cmp_ui (mpq_denref(q), 1L) == 0);			\
      }									\
    else								\
      {									\
	/* no common factors */						\
	mpz_t  __g;							\
	mpz_init (__g);							\
	mpz_gcd (__g, mpq_numref(q), mpq_denref(q));			\
	ASSERT (mpz_cmp_ui (__g, 1) == 0);				\
	mpz_clear (__g);						\
      }									\
  } while (0)
#else
#define ASSERT_MPQ_CANONICAL(q)	 do {} while (0)
#endif

/* Check that the nail parts are zero. */
#define ASSERT_ALWAYS_LIMB(limb)					\
  do {									\
    mp_limb_t  __nail = (limb) & GMP_NAIL_MASK;				\
    ASSERT_ALWAYS (__nail == 0);					\
  } while (0)
#define ASSERT_ALWAYS_MPN(ptr, size)					\
  do {									\
    /* let whole loop go dead when no nails */				\
    if (GMP_NAIL_BITS != 0)						\
      {									\
	mp_size_t  __i;							\
	for (__i = 0; __i < (size); __i++)				\
	  ASSERT_ALWAYS_LIMB ((ptr)[__i]);				\
      }									\
  } while (0)
#if WANT_ASSERT
#define ASSERT_LIMB(limb)       ASSERT_ALWAYS_LIMB (limb)
#define ASSERT_MPN(ptr, size)   ASSERT_ALWAYS_MPN (ptr, size)
#else
#define ASSERT_LIMB(limb)       do {} while (0)
#define ASSERT_MPN(ptr, size)   do {} while (0)
#endif


/* Assert that an mpn region {ptr,size} is zero, or non-zero.
   size==0 is allowed, and in that case {ptr,size} considered to be zero.  */
#if WANT_ASSERT
#define ASSERT_MPN_ZERO_P(ptr,size)					\
  do {									\
    mp_size_t  __i;							\
    ASSERT ((size) >= 0);						\
    for (__i = 0; __i < (size); __i++)					\
      ASSERT ((ptr)[__i] == 0);						\
  } while (0)
#define ASSERT_MPN_NONZERO_P(ptr,size)					\
  do {									\
    mp_size_t  __i;							\
    int	       __nonzero = 0;						\
    ASSERT ((size) >= 0);						\
    for (__i = 0; __i < (size); __i++)					\
      if ((ptr)[__i] != 0)						\
	{								\
	  __nonzero = 1;						\
	  break;							\
	}								\
    ASSERT (__nonzero);							\
  } while (0)
#else
#define ASSERT_MPN_ZERO_P(ptr,size)     do {} while (0)
#define ASSERT_MPN_NONZERO_P(ptr,size)  do {} while (0)
#endif


#if ! HAVE_NATIVE_mpn_com
#undef mpn_com
#define mpn_com(d,s,n)							\
  do {									\
    mp_ptr     __d = (d);						\
    mp_srcptr  __s = (s);						\
    mp_size_t  __n = (n);						\
    ASSERT (__n >= 1);							\
    ASSERT (MPN_SAME_OR_SEPARATE_P (__d, __s, __n));			\
    do									\
      *__d++ = (~ *__s++) & GMP_NUMB_MASK;				\
    while (--__n);							\
  } while (0)
#endif

#define MPN_LOGOPS_N_INLINE(rp, up, vp, n, operation)			\
  do {									\
    mp_srcptr	__up = (up);						\
    mp_srcptr	__vp = (vp);						\
    mp_ptr	__rp = (rp);						\
    mp_size_t	__n = (n);						\
    mp_limb_t __a, __b;							\
    ASSERT (__n > 0);							\
    ASSERT (MPN_SAME_OR_SEPARATE_P (__rp, __up, __n));			\
    ASSERT (MPN_SAME_OR_SEPARATE_P (__rp, __vp, __n));			\
    __up += __n;							\
    __vp += __n;							\
    __rp += __n;							\
    __n = -__n;								\
    do {								\
      __a = __up[__n];							\
      __b = __vp[__n];							\
      __rp[__n] = operation;						\
    } while (++__n);							\
  } while (0)


#if ! HAVE_NATIVE_mpn_and_n
#undef mpn_and_n
#define mpn_and_n(rp, up, vp, n) \
  MPN_LOGOPS_N_INLINE (rp, up, vp, n, __a & __b)
#endif

#if ! HAVE_NATIVE_mpn_andn_n
#undef mpn_andn_n
#define mpn_andn_n(rp, up, vp, n) \
  MPN_LOGOPS_N_INLINE (rp, up, vp, n, __a & ~__b)
#endif

#if ! HAVE_NATIVE_mpn_nand_n
#undef mpn_nand_n
#define mpn_nand_n(rp, up, vp, n) \
  MPN_LOGOPS_N_INLINE (rp, up, vp, n, ~(__a & __b) & GMP_NUMB_MASK)
#endif

#if ! HAVE_NATIVE_mpn_ior_n
#undef mpn_ior_n
#define mpn_ior_n(rp, up, vp, n) \
  MPN_LOGOPS_N_INLINE (rp, up, vp, n, __a | __b)
#endif

#if ! HAVE_NATIVE_mpn_iorn_n
#undef mpn_iorn_n
#define mpn_iorn_n(rp, up, vp, n) \
  MPN_LOGOPS_N_INLINE (rp, up, vp, n, (__a | ~__b) & GMP_NUMB_MASK)
#endif

#if ! HAVE_NATIVE_mpn_nior_n
#undef mpn_nior_n
#define mpn_nior_n(rp, up, vp, n) \
  MPN_LOGOPS_N_INLINE (rp, up, vp, n, ~(__a | __b) & GMP_NUMB_MASK)
#endif

#if ! HAVE_NATIVE_mpn_xor_n
#undef mpn_xor_n
#define mpn_xor_n(rp, up, vp, n) \
  MPN_LOGOPS_N_INLINE (rp, up, vp, n, __a ^ __b)
#endif

#if ! HAVE_NATIVE_mpn_xnor_n
#undef mpn_xnor_n
#define mpn_xnor_n(rp, up, vp, n) \
  MPN_LOGOPS_N_INLINE (rp, up, vp, n, ~(__a ^ __b) & GMP_NUMB_MASK)
#endif

#define mpn_trialdiv __MPN(trialdiv)
__GMP_DECLSPEC mp_limb_t mpn_trialdiv (mp_srcptr, mp_size_t, mp_size_t, int *);

#define mpn_remove __MPN(remove)
__GMP_DECLSPEC mp_bitcnt_t mpn_remove (mp_ptr, mp_size_t *, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, mp_bitcnt_t);


/* ADDC_LIMB sets w=x+y and cout to 0 or 1 for a carry from that addition. */
#if GMP_NAIL_BITS == 0
#define ADDC_LIMB(cout, w, x, y)					\
  do {									\
    mp_limb_t  __x = (x);						\
    mp_limb_t  __y = (y);						\
    mp_limb_t  __w = __x + __y;						\
    (w) = __w;								\
    (cout) = __w < __x;							\
  } while (0)
#else
#define ADDC_LIMB(cout, w, x, y)					\
  do {									\
    mp_limb_t  __w;							\
    ASSERT_LIMB (x);							\
    ASSERT_LIMB (y);							\
    __w = (x) + (y);							\
    (w) = __w & GMP_NUMB_MASK;						\
    (cout) = __w >> GMP_NUMB_BITS;					\
  } while (0)
#endif

/* SUBC_LIMB sets w=x-y and cout to 0 or 1 for a borrow from that
   subtract.  */
#if GMP_NAIL_BITS == 0
#define SUBC_LIMB(cout, w, x, y)					\
  do {									\
    mp_limb_t  __x = (x);						\
    mp_limb_t  __y = (y);						\
    mp_limb_t  __w = __x - __y;						\
    (w) = __w;								\
    (cout) = __w > __x;							\
  } while (0)
#else
#define SUBC_LIMB(cout, w, x, y)					\
  do {									\
    mp_limb_t  __w = (x) - (y);						\
    (w) = __w & GMP_NUMB_MASK;						\
    (cout) = __w >> (GMP_LIMB_BITS-1);					\
  } while (0)
#endif


/* MPN_INCR_U does {ptr,size} += n, MPN_DECR_U does {ptr,size} -= n, both
   expecting no carry (or borrow) from that.

   The size parameter is only for the benefit of assertion checking.  In a
   normal build it's unused and the carry/borrow is just propagated as far
   as it needs to go.

   On random data, usually only one or two limbs of {ptr,size} get updated,
   so there's no need for any sophisticated looping, just something compact
   and sensible.

   FIXME: Switch all code from mpn_{incr,decr}_u to MPN_{INCR,DECR}_U,
   declaring their operand sizes, then remove the former.  This is purely
   for the benefit of assertion checking.  */

#if defined (__GNUC__) && GMP_NAIL_BITS == 0 && ! defined (NO_ASM)	\
  && (defined(HAVE_HOST_CPU_FAMILY_x86) || defined(HAVE_HOST_CPU_FAMILY_x86_64)) \
  && ! WANT_ASSERT
/* Better flags handling than the generic C gives on i386, saving a few
   bytes of code and maybe a cycle or two.  */

#define MPN_IORD_U(ptr, incr, aors)					\
  do {									\
    mp_ptr  __ptr_dummy;						\
    if (__builtin_constant_p (incr) && (incr) == 0)			\
      {									\
      }									\
    else if (__builtin_constant_p (incr) && (incr) == 1)		\
      {									\
	__asm__ __volatile__						\
	  ("\n" ASM_L(top) ":\n"					\
	   "\t" aors "\t$1, (%0)\n"					\
	   "\tlea\t%c2(%0), %0\n"					\
	   "\tjc\t" ASM_L(top)						\
	   : "=r" (__ptr_dummy)						\
	   : "0"  (ptr), "n" (sizeof(mp_limb_t))			\
	   : "memory");							\
      }									\
    else								\
      {									\
	__asm__ __volatile__						\
	  (   aors  "\t%2, (%0)\n"					\
	   "\tjnc\t" ASM_L(done) "\n"					\
	   ASM_L(top) ":\n"						\
	   "\t" aors "\t$1, %c3(%0)\n"					\
	   "\tlea\t%c3(%0), %0\n"					\
	   "\tjc\t" ASM_L(top) "\n"					\
	   ASM_L(done) ":\n"						\
	   : "=r" (__ptr_dummy)						\
	   : "0"  (ptr),						\
	     "ri" ((mp_limb_t) (incr)), "n" (sizeof(mp_limb_t))		\
	   : "memory");							\
      }									\
  } while (0)

#if GMP_LIMB_BITS == 32
#define MPN_INCR_U(ptr, size, incr)  MPN_IORD_U (ptr, incr, "addl")
#define MPN_DECR_U(ptr, size, incr)  MPN_IORD_U (ptr, incr, "subl")
#endif
#if GMP_LIMB_BITS == 64
#define MPN_INCR_U(ptr, size, incr)  MPN_IORD_U (ptr, incr, "addq")
#define MPN_DECR_U(ptr, size, incr)  MPN_IORD_U (ptr, incr, "subq")
#endif
#define mpn_incr_u(ptr, incr)  MPN_INCR_U (ptr, 0, incr)
#define mpn_decr_u(ptr, incr)  MPN_DECR_U (ptr, 0, incr)
#endif

#if GMP_NAIL_BITS == 0
#ifndef mpn_incr_u
#define mpn_incr_u(p,incr)						\
  do {									\
    mp_limb_t __x;							\
    mp_ptr __p = (p);							\
    if (__builtin_constant_p (incr) && (incr) == 1)			\
      {									\
	while (++(*(__p++)) == 0)					\
	  ;								\
      }									\
    else								\
      {									\
	__x = *__p + (incr);						\
	*__p = __x;							\
	if (__x < (incr))						\
	  while (++(*(++__p)) == 0)					\
	    ;								\
      }									\
  } while (0)
#endif
#ifndef mpn_decr_u
#define mpn_decr_u(p,incr)						\
  do {									\
    mp_limb_t __x;							\
    mp_ptr __p = (p);							\
    if (__builtin_constant_p (incr) && (incr) == 1)			\
      {									\
	while ((*(__p++))-- == 0)					\
	  ;								\
      }									\
    else								\
      {									\
	__x = *__p;							\
	*__p = __x - (incr);						\
	if (__x < (incr))						\
	  while ((*(++__p))-- == 0)					\
	    ;								\
      }									\
  } while (0)
#endif
#endif

#if GMP_NAIL_BITS >= 1
#ifndef mpn_incr_u
#define mpn_incr_u(p,incr)						\
  do {									\
    mp_limb_t __x;							\
    mp_ptr __p = (p);							\
    if (__builtin_constant_p (incr) && (incr) == 1)			\
      {									\
	do								\
	  {								\
	    __x = (*__p + 1) & GMP_NUMB_MASK;				\
	    *__p++ = __x;						\
	  }								\
	while (__x == 0);						\
      }									\
    else								\
      {									\
	__x = (*__p + (incr));						\
	*__p++ = __x & GMP_NUMB_MASK;					\
	if (__x >> GMP_NUMB_BITS != 0)					\
	  {								\
	    do								\
	      {								\
		__x = (*__p + 1) & GMP_NUMB_MASK;			\
		*__p++ = __x;						\
	      }								\
	    while (__x == 0);						\
	  }								\
      }									\
  } while (0)
#endif
#ifndef mpn_decr_u
#define mpn_decr_u(p,incr)						\
  do {									\
    mp_limb_t __x;							\
    mp_ptr __p = (p);							\
    if (__builtin_constant_p (incr) && (incr) == 1)			\
      {									\
	do								\
	  {								\
	    __x = *__p;							\
	    *__p++ = (__x - 1) & GMP_NUMB_MASK;				\
	  }								\
	while (__x == 0);						\
      }									\
    else								\
      {									\
	__x = *__p - (incr);						\
	*__p++ = __x & GMP_NUMB_MASK;					\
	if (__x >> GMP_NUMB_BITS != 0)					\
	  {								\
	    do								\
	      {								\
		__x = *__p;						\
		*__p++ = (__x - 1) & GMP_NUMB_MASK;			\
	      }								\
	    while (__x == 0);						\
	  }								\
      }									\
  } while (0)
#endif
#endif

#ifndef MPN_INCR_U
#if WANT_ASSERT
#define MPN_INCR_U(ptr, size, n)					\
  do {									\
    ASSERT ((size) >= 1);						\
    ASSERT_NOCARRY (mpn_add_1 (ptr, ptr, size, n));			\
  } while (0)
#else
#define MPN_INCR_U(ptr, size, n)   mpn_incr_u (ptr, n)
#endif
#endif

#ifndef MPN_DECR_U
#if WANT_ASSERT
#define MPN_DECR_U(ptr, size, n)					\
  do {									\
    ASSERT ((size) >= 1);						\
    ASSERT_NOCARRY (mpn_sub_1 (ptr, ptr, size, n));			\
  } while (0)
#else
#define MPN_DECR_U(ptr, size, n)   mpn_decr_u (ptr, n)
#endif
#endif


/* Structure for conversion between internal binary format and strings.  */
struct bases
{
  /* Number of digits in the conversion base that always fits in an mp_limb_t.
     For example, for base 10 on a machine where an mp_limb_t has 32 bits this
     is 9, since 10**9 is the largest number that fits into an mp_limb_t.  */
  int chars_per_limb;

  /* log(2)/log(conversion_base) */
  mp_limb_t logb2;

  /* log(conversion_base)/log(2) */
  mp_limb_t log2b;

  /* base**chars_per_limb, i.e. the biggest number that fits a word, built by
     factors of base.  Exception: For 2, 4, 8, etc, big_base is log2(base),
     i.e. the number of bits used to represent each digit in the base.  */
  mp_limb_t big_base;

  /* A GMP_LIMB_BITS bit approximation to 1/big_base, represented as a
     fixed-point number.  Instead of dividing by big_base an application can
     choose to multiply by big_base_inverted.  */
  mp_limb_t big_base_inverted;
};

#define   mp_bases __MPN(bases)
__GMP_DECLSPEC extern const struct bases mp_bases[257];


/* Compute the number of digits in base for nbits bits, making sure the result
   is never too small.  The two variants of the macro implement the same
   function; the GT2 variant below works just for bases > 2.  */
#define DIGITS_IN_BASE_FROM_BITS(res, nbits, b)				\
  do {									\
    mp_limb_t _ph, _dummy;						\
    size_t _nbits = (nbits);						\
    umul_ppmm (_ph, _dummy, mp_bases[b].logb2, _nbits);			\
    _ph += (_dummy + _nbits < _dummy);					\
    res = _ph + 1;							\
  } while (0)
#define DIGITS_IN_BASEGT2_FROM_BITS(res, nbits, b)			\
  do {									\
    mp_limb_t _ph, _dummy;						\
    size_t _nbits = (nbits);						\
    umul_ppmm (_ph, _dummy, mp_bases[b].logb2 + 1, _nbits);		\
    res = _ph + 1;							\
  } while (0)

/* For power of 2 bases this is exact.  For other bases the result is either
   exact or one too big.

   To be exact always it'd be necessary to examine all the limbs of the
   operand, since numbers like 100..000 and 99...999 generally differ only
   in the lowest limb.  It'd be possible to examine just a couple of high
   limbs to increase the probability of being exact, but that doesn't seem
   worth bothering with.  */

#define MPN_SIZEINBASE(result, ptr, size, base)				\
  do {									\
    int	   __lb_base, __cnt;						\
    size_t __totbits;							\
									\
    ASSERT ((size) >= 0);						\
    ASSERT ((base) >= 2);						\
    ASSERT ((base) < numberof (mp_bases));				\
									\
    /* Special case for X == 0.  */					\
    if ((size) == 0)							\
      (result) = 1;							\
    else								\
      {									\
	/* Calculate the total number of significant bits of X.  */	\
	count_leading_zeros (__cnt, (ptr)[(size)-1]);			\
	__totbits = (size_t) (size) * GMP_NUMB_BITS - (__cnt - GMP_NAIL_BITS);\
									\
	if (POW2_P (base))						\
	  {								\
	    __lb_base = mp_bases[base].big_base;			\
	    (result) = (__totbits + __lb_base - 1) / __lb_base;		\
	  }								\
	else								\
	  {								\
	    DIGITS_IN_BASEGT2_FROM_BITS (result, __totbits, base);	\
	  }								\
      }									\
  } while (0)

#define MPN_SIZEINBASE_2EXP(result, ptr, size, base2exp)			\
  do {										\
    int          __cnt;								\
    mp_bitcnt_t  __totbits;							\
    ASSERT ((size) > 0);							\
    ASSERT ((ptr)[(size)-1] != 0);						\
    count_leading_zeros (__cnt, (ptr)[(size)-1]);				\
    __totbits = (mp_bitcnt_t) (size) * GMP_NUMB_BITS - (__cnt - GMP_NAIL_BITS);	\
    (result) = (__totbits + (base2exp)-1) / (base2exp);				\
  } while (0)


/* bit count to limb count, rounding up */
#define BITS_TO_LIMBS(n)  (((n) + (GMP_NUMB_BITS - 1)) / GMP_NUMB_BITS)

/* MPN_SET_UI sets an mpn (ptr, cnt) to given ui.  MPZ_FAKE_UI creates fake
   mpz_t from ui.  The zp argument must have room for LIMBS_PER_ULONG limbs
   in both cases (LIMBS_PER_ULONG is also defined here.) */
#if BITS_PER_ULONG <= GMP_NUMB_BITS /* need one limb per ulong */

#define LIMBS_PER_ULONG 1
#define MPN_SET_UI(zp, zn, u)						\
  (zp)[0] = (u);							\
  (zn) = ((zp)[0] != 0);
#define MPZ_FAKE_UI(z, zp, u)						\
  (zp)[0] = (u);							\
  PTR (z) = (zp);							\
  SIZ (z) = ((zp)[0] != 0);						\
  ASSERT_CODE (ALLOC (z) = 1);

#else /* need two limbs per ulong */

#define LIMBS_PER_ULONG 2
#define MPN_SET_UI(zp, zn, u)						\
  (zp)[0] = (u) & GMP_NUMB_MASK;					\
  (zp)[1] = (u) >> GMP_NUMB_BITS;					\
  (zn) = ((zp)[1] != 0 ? 2 : (zp)[0] != 0 ? 1 : 0);
#define MPZ_FAKE_UI(z, zp, u)						\
  (zp)[0] = (u) & GMP_NUMB_MASK;					\
  (zp)[1] = (u) >> GMP_NUMB_BITS;					\
  SIZ (z) = ((zp)[1] != 0 ? 2 : (zp)[0] != 0 ? 1 : 0);			\
  PTR (z) = (zp);							\
  ASSERT_CODE (ALLOC (z) = 2);

#endif


#if HAVE_HOST_CPU_FAMILY_x86
#define TARGET_REGISTER_STARVED 1
#else
#define TARGET_REGISTER_STARVED 0
#endif


/* LIMB_HIGHBIT_TO_MASK(n) examines the high bit of a limb value and turns 1
   or 0 there into a limb 0xFF..FF or 0 respectively.

   On most CPUs this is just an arithmetic right shift by GMP_LIMB_BITS-1,
   but C99 doesn't guarantee signed right shifts are arithmetic, so we have
   a little compile-time test and a fallback to a "? :" form.  The latter is
   necessary for instance on Cray vector systems.

   Recent versions of gcc (eg. 3.3) will in fact optimize a "? :" like this
   to an arithmetic right shift anyway, but it's good to get the desired
   shift on past versions too (in particular since an important use of
   LIMB_HIGHBIT_TO_MASK is in udiv_qrnnd_preinv).  */

#define LIMB_HIGHBIT_TO_MASK(n)						\
  (((mp_limb_signed_t) -1 >> 1) < 0					\
   ? (mp_limb_signed_t) (n) >> (GMP_LIMB_BITS - 1)			\
   : (n) & GMP_LIMB_HIGHBIT ? MP_LIMB_T_MAX : CNST_LIMB(0))


/* Use a library function for invert_limb, if available. */
#define  mpn_invert_limb __MPN(invert_limb)
__GMP_DECLSPEC mp_limb_t mpn_invert_limb (mp_limb_t) ATTRIBUTE_CONST;
#if ! defined (invert_limb) && HAVE_NATIVE_mpn_invert_limb
#define invert_limb(invxl,xl)						\
  do {									\
    (invxl) = mpn_invert_limb (xl);					\
  } while (0)
#endif

#ifndef invert_limb
#define invert_limb(invxl,xl)						\
  do {									\
    mp_limb_t _dummy;							\
    ASSERT ((xl) != 0);							\
    udiv_qrnnd (invxl, _dummy, ~(xl), ~CNST_LIMB(0), xl);		\
  } while (0)
#endif

#define invert_pi1(dinv, d1, d0)					\
  do {									\
    mp_limb_t _v, _p, _t1, _t0, _mask;					\
    invert_limb (_v, d1);						\
    _p = (d1) * _v;							\
    _p += (d0);								\
    if (_p < (d0))							\
      {									\
	_v--;								\
	_mask = -(mp_limb_t) (_p >= (d1));				\
	_p -= (d1);							\
	_v += _mask;							\
	_p -= _mask & (d1);						\
      }									\
    umul_ppmm (_t1, _t0, d0, _v);					\
    _p += _t1;								\
    if (_p < _t1)							\
      {									\
	_v--;								\
	if (UNLIKELY (_p >= (d1)))					\
	  {								\
	    if (_p > (d1) || _t0 >= (d0))				\
	      _v--;							\
	  }								\
      }									\
    (dinv).inv32 = _v;							\
  } while (0)


/* udiv_qrnnd_preinv -- Based on work by Niels Möller and Torbjörn Granlund.
   We write things strangely below, to help gcc.  A more straightforward
   version:
	_r = (nl) - _qh * (d);
	_t = _r + (d);
	if (_r >= _ql)
	  {
	    _qh--;
	    _r = _t;
	  }
   For one operation shorter critical path, one may want to use this form:
	_p = _qh * (d)
	_s = (nl) + (d);
	_r = (nl) - _p;
	_t = _s - _p;
	if (_r >= _ql)
	  {
	    _qh--;
	    _r = _t;
	  }
*/
#define udiv_qrnnd_preinv(q, r, nh, nl, d, di)				\
  do {									\
    mp_limb_t _qh, _ql, _r, _mask;					\
    umul_ppmm (_qh, _ql, (nh), (di));					\
    if (__builtin_constant_p (nl) && (nl) == 0)				\
      {									\
	_qh += (nh) + 1;						\
	_r = - _qh * (d);						\
	_mask = -(mp_limb_t) (_r > _ql); /* both > and >= are OK */	\
	_qh += _mask;							\
	_r += _mask & (d);						\
      }									\
    else								\
      {									\
	add_ssaaaa (_qh, _ql, _qh, _ql, (nh) + 1, (nl));		\
	_r = (nl) - _qh * (d);						\
	_mask = -(mp_limb_t) (_r > _ql); /* both > and >= are OK */	\
	_qh += _mask;							\
	_r += _mask & (d);						\
	if (UNLIKELY (_r >= (d)))					\
	  {								\
	    _r -= (d);							\
	    _qh++;							\
	  }								\
      }									\
    (r) = _r;								\
    (q) = _qh;								\
  } while (0)

/* Dividing (NH, NL) by D, returning the remainder only. Unlike
   udiv_qrnnd_preinv, works also for the case NH == D, where the
   quotient doesn't quite fit in a single limb. */
#define udiv_rnnd_preinv(r, nh, nl, d, di)				\
  do {									\
    mp_limb_t _qh, _ql, _r, _mask;					\
    umul_ppmm (_qh, _ql, (nh), (di));					\
    if (__builtin_constant_p (nl) && (nl) == 0)				\
      {									\
	_r = ~(_qh + (nh)) * (d);					\
	_mask = -(mp_limb_t) (_r > _ql); /* both > and >= are OK */	\
	_r += _mask & (d);						\
      }									\
    else								\
      {									\
	add_ssaaaa (_qh, _ql, _qh, _ql, (nh) + 1, (nl));		\
	_r = (nl) - _qh * (d);						\
	_mask = -(mp_limb_t) (_r > _ql); /* both > and >= are OK */	\
	_r += _mask & (d);						\
	if (UNLIKELY (_r >= (d)))					\
	  _r -= (d);							\
      }									\
    (r) = _r;								\
  } while (0)

/* Compute quotient the quotient and remainder for n / d. Requires d
   >= B^2 / 2 and n < d B. di is the inverse

     floor ((B^3 - 1) / (d0 + d1 B)) - B.

   NOTE: Output variables are updated multiple times. Only some inputs
   and outputs may overlap.
*/
#define udiv_qr_3by2(q, r1, r0, n2, n1, n0, d1, d0, dinv)		\
  do {									\
    mp_limb_t _q0, _t1, _t0, _mask;					\
    umul_ppmm ((q), _q0, (n2), (dinv));					\
    add_ssaaaa ((q), _q0, (q), _q0, (n2), (n1));			\
									\
    /* Compute the two most significant limbs of n - q'd */		\
    (r1) = (n1) - (d1) * (q);						\
    sub_ddmmss ((r1), (r0), (r1), (n0), (d1), (d0));			\
    umul_ppmm (_t1, _t0, (d0), (q));					\
    sub_ddmmss ((r1), (r0), (r1), (r0), _t1, _t0);			\
    (q)++;								\
									\
    /* Conditionally adjust q and the remainders */			\
    _mask = - (mp_limb_t) ((r1) >= _q0);				\
    (q) += _mask;							\
    add_ssaaaa ((r1), (r0), (r1), (r0), _mask & (d1), _mask & (d0));	\
    if (UNLIKELY ((r1) >= (d1)))					\
      {									\
	if ((r1) > (d1) || (r0) >= (d0))				\
	  {								\
	    (q)++;							\
	    sub_ddmmss ((r1), (r0), (r1), (r0), (d1), (d0));		\
	  }								\
      }									\
  } while (0)

#ifndef mpn_preinv_divrem_1  /* if not done with cpuvec in a fat binary */
#define   mpn_preinv_divrem_1 __MPN(preinv_divrem_1)
__GMP_DECLSPEC mp_limb_t mpn_preinv_divrem_1 (mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t, int);
#endif


/* USE_PREINV_DIVREM_1 is whether to use mpn_preinv_divrem_1, as opposed to the
   plain mpn_divrem_1.  The default is yes, since the few CISC chips where
   preinv is not good have defines saying so.  */
#ifndef USE_PREINV_DIVREM_1
#define USE_PREINV_DIVREM_1   1
#endif

#if USE_PREINV_DIVREM_1
#define MPN_DIVREM_OR_PREINV_DIVREM_1(qp,xsize,ap,size,d,dinv,shift)    \
  mpn_preinv_divrem_1 (qp, xsize, ap, size, d, dinv, shift)
#else
#define MPN_DIVREM_OR_PREINV_DIVREM_1(qp,xsize,ap,size,d,dinv,shift)    \
  mpn_divrem_1 (qp, xsize, ap, size, d)
#endif

#ifndef PREINV_MOD_1_TO_MOD_1_THRESHOLD
#define PREINV_MOD_1_TO_MOD_1_THRESHOLD 10
#endif

/* This selection may seem backwards.  The reason mpn_mod_1 typically takes
   over for larger sizes is that it uses the mod_1_1 function.  */
#define MPN_MOD_OR_PREINV_MOD_1(src,size,divisor,inverse)		\
  (BELOW_THRESHOLD (size, PREINV_MOD_1_TO_MOD_1_THRESHOLD)		\
   ? mpn_preinv_mod_1 (src, size, divisor, inverse)			\
   : mpn_mod_1 (src, size, divisor))


#ifndef mpn_mod_34lsub1  /* if not done with cpuvec in a fat binary */
#define mpn_mod_34lsub1 __MPN(mod_34lsub1)
__GMP_DECLSPEC mp_limb_t mpn_mod_34lsub1 (mp_srcptr, mp_size_t) __GMP_ATTRIBUTE_PURE;
#endif


/* DIVEXACT_1_THRESHOLD is at what size to use mpn_divexact_1, as opposed to
   plain mpn_divrem_1.  Likewise BMOD_1_TO_MOD_1_THRESHOLD for
   mpn_modexact_1_odd against plain mpn_mod_1.  On most CPUs divexact and
   modexact are faster at all sizes, so the defaults are 0.  Those CPUs
   where this is not right have a tuned threshold.  */
#ifndef DIVEXACT_1_THRESHOLD
#define DIVEXACT_1_THRESHOLD  0
#endif
#ifndef BMOD_1_TO_MOD_1_THRESHOLD
#define BMOD_1_TO_MOD_1_THRESHOLD  10
#endif

#define MPN_DIVREM_OR_DIVEXACT_1(rp, up, n, d)				\
  do {									\
    if (BELOW_THRESHOLD (n, DIVEXACT_1_THRESHOLD))			\
      ASSERT_NOCARRY (mpn_divrem_1 (rp, (mp_size_t) 0, up, n, d));	\
    else								\
      {									\
	ASSERT (mpn_mod_1 (up, n, d) == 0);				\
	mpn_divexact_1 (rp, up, n, d);					\
      }									\
  } while (0)

#ifndef mpn_modexact_1c_odd  /* if not done with cpuvec in a fat binary */
#define mpn_modexact_1c_odd __MPN(modexact_1c_odd)
__GMP_DECLSPEC mp_limb_t mpn_modexact_1c_odd (mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t) __GMP_ATTRIBUTE_PURE;
#endif

#if HAVE_NATIVE_mpn_modexact_1_odd
#define   mpn_modexact_1_odd  __MPN(modexact_1_odd)
__GMP_DECLSPEC mp_limb_t mpn_modexact_1_odd (mp_srcptr, mp_size_t, mp_limb_t) __GMP_ATTRIBUTE_PURE;
#else
#define mpn_modexact_1_odd(src,size,divisor) \
  mpn_modexact_1c_odd (src, size, divisor, CNST_LIMB(0))
#endif

#define MPN_MOD_OR_MODEXACT_1_ODD(src,size,divisor)			\
  (BELOW_THRESHOLD (size, BMOD_1_TO_MOD_1_THRESHOLD)			\
   ? mpn_modexact_1_odd (src, size, divisor)				\
   : mpn_mod_1 (src, size, divisor))

/* binvert_limb() sets inv to the multiplicative inverse of n modulo
   2^GMP_NUMB_BITS, ie. satisfying inv*n == 1 mod 2^GMP_NUMB_BITS.
   n must be odd (otherwise such an inverse doesn't exist).

   This is not to be confused with invert_limb(), which is completely
   different.

   The table lookup gives an inverse with the low 8 bits valid, and each
   multiply step doubles the number of bits.  See Jebelean "An algorithm for
   exact division" end of section 4 (reference in gmp.texi).

   Possible enhancement: Could use UHWtype until the last step, if half-size
   multiplies are faster (might help under _LONG_LONG_LIMB).

   Alternative: As noted in Granlund and Montgomery "Division by Invariant
   Integers using Multiplication" (reference in gmp.texi), n itself gives a
   3-bit inverse immediately, and could be used instead of a table lookup.
   A 4-bit inverse can be obtained effectively from xoring bits 1 and 2 into
   bit 3, for instance with (((n + 2) & 4) << 1) ^ n.  */

#define binvert_limb_table  __gmp_binvert_limb_table
__GMP_DECLSPEC extern const unsigned char  binvert_limb_table[128];

#define binvert_limb(inv,n)						\
  do {									\
    mp_limb_t  __n = (n);						\
    mp_limb_t  __inv;							\
    ASSERT ((__n & 1) == 1);						\
									\
    __inv = binvert_limb_table[(__n/2) & 0x7F]; /*  8 */		\
    if (GMP_NUMB_BITS > 8)   __inv = 2 * __inv - __inv * __inv * __n;	\
    if (GMP_NUMB_BITS > 16)  __inv = 2 * __inv - __inv * __inv * __n;	\
    if (GMP_NUMB_BITS > 32)  __inv = 2 * __inv - __inv * __inv * __n;	\
									\
    if (GMP_NUMB_BITS > 64)						\
      {									\
	int  __invbits = 64;						\
	do {								\
	  __inv = 2 * __inv - __inv * __inv * __n;			\
	  __invbits *= 2;						\
	} while (__invbits < GMP_NUMB_BITS);				\
      }									\
									\
    ASSERT ((__inv * __n & GMP_NUMB_MASK) == 1);			\
    (inv) = __inv & GMP_NUMB_MASK;					\
  } while (0)
#define modlimb_invert binvert_limb  /* backward compatibility */

/* Multiplicative inverse of 3, modulo 2^GMP_NUMB_BITS.
   Eg. 0xAAAAAAAB for 32 bits, 0xAAAAAAAAAAAAAAAB for 64 bits.
   GMP_NUMB_MAX/3*2+1 is right when GMP_NUMB_BITS is even, but when it's odd
   we need to start from GMP_NUMB_MAX>>1. */
#define MODLIMB_INVERSE_3 (((GMP_NUMB_MAX >> (GMP_NUMB_BITS % 2)) / 3) * 2 + 1)

/* ceil(GMP_NUMB_MAX/3) and ceil(2*GMP_NUMB_MAX/3).
   These expressions work because GMP_NUMB_MAX%3 != 0 for all GMP_NUMB_BITS. */
#define GMP_NUMB_CEIL_MAX_DIV3   (GMP_NUMB_MAX / 3 + 1)
#define GMP_NUMB_CEIL_2MAX_DIV3  ((GMP_NUMB_MAX>>1) / 3 + 1 + GMP_NUMB_HIGHBIT)


/* Set r to -a mod d.  a>=d is allowed.  Can give r>d.  All should be limbs.

   It's not clear whether this is the best way to do this calculation.
   Anything congruent to -a would be fine for the one limb congruence
   tests.  */

#define NEG_MOD(r, a, d)						\
  do {									\
    ASSERT ((d) != 0);							\
    ASSERT_LIMB (a);							\
    ASSERT_LIMB (d);							\
									\
    if ((a) <= (d))							\
      {									\
	/* small a is reasonably likely */				\
	(r) = (d) - (a);						\
      }									\
    else								\
      {									\
	unsigned   __twos;						\
	mp_limb_t  __dnorm;						\
	count_leading_zeros (__twos, d);				\
	__twos -= GMP_NAIL_BITS;					\
	__dnorm = (d) << __twos;					\
	(r) = ((a) <= __dnorm ? __dnorm : 2*__dnorm) - (a);		\
      }									\
									\
    ASSERT_LIMB (r);							\
  } while (0)

/* A bit mask of all the least significant zero bits of n, or -1 if n==0. */
#define LOW_ZEROS_MASK(n)  (((n) & -(n)) - 1)


/* ULONG_PARITY sets "p" to 1 if there's an odd number of 1 bits in "n", or
   to 0 if there's an even number.  "n" should be an unsigned long and "p"
   an int.  */

#if defined (__GNUC__) && ! defined (NO_ASM) && HAVE_HOST_CPU_alpha_CIX
#define ULONG_PARITY(p, n)						\
  do {									\
    int __p;								\
    __asm__ ("ctpop %1, %0" : "=r" (__p) : "r" (n));			\
    (p) = __p & 1;							\
  } while (0)
#endif

/* Cray intrinsic _popcnt. */
#ifdef _CRAY
#define ULONG_PARITY(p, n)      \
  do {                          \
    (p) = _popcnt (n) & 1;      \
  } while (0)
#endif

#if defined (__GNUC__) && ! defined (__INTEL_COMPILER)			\
    && ! defined (NO_ASM) && defined (__ia64)
/* unsigned long is either 32 or 64 bits depending on the ABI, zero extend
   to a 64 bit unsigned long long for popcnt */
#define ULONG_PARITY(p, n)						\
  do {									\
    unsigned long long  __n = (unsigned long) (n);			\
    int  __p;								\
    __asm__ ("popcnt %0 = %1" : "=r" (__p) : "r" (__n));		\
    (p) = __p & 1;							\
  } while (0)
#endif

#if defined (__GNUC__) && ! defined (__INTEL_COMPILER)			\
    && ! defined (NO_ASM) && HAVE_HOST_CPU_FAMILY_x86
#if __GMP_GNUC_PREREQ (3,1)
#define __GMP_qm "=Qm"
#define __GMP_q "=Q"
#else
#define __GMP_qm "=qm"
#define __GMP_q "=q"
#endif
#define ULONG_PARITY(p, n)						\
  do {									\
    char	   __p;							\
    unsigned long  __n = (n);						\
    __n ^= (__n >> 16);							\
    __asm__ ("xorb %h1, %b1\n\t"					\
	     "setpo %0"							\
	 : __GMP_qm (__p), __GMP_q (__n)				\
	 : "1" (__n));							\
    (p) = __p;								\
  } while (0)
#endif

#if ! defined (ULONG_PARITY)
#define ULONG_PARITY(p, n)						\
  do {									\
    unsigned long  __n = (n);						\
    int  __p = 0;							\
    do									\
      {									\
	__p ^= 0x96696996L >> (__n & 0x1F);				\
	__n >>= 5;							\
      }									\
    while (__n != 0);							\
									\
    (p) = __p & 1;							\
  } while (0)
#endif


/* 3 cycles on 604 or 750 since shifts and rlwimi's can pair.  gcc (as of
   version 3.1 at least) doesn't seem to know how to generate rlwimi for
   anything other than bit-fields, so use "asm".  */
#if defined (__GNUC__) && ! defined (NO_ASM)                    \
  && HAVE_HOST_CPU_FAMILY_powerpc && GMP_LIMB_BITS == 32
#define BSWAP_LIMB(dst, src)						\
  do {									\
    mp_limb_t  __bswapl_src = (src);					\
    mp_limb_t  __tmp1 = __bswapl_src >> 24;		/* low byte */	\
    mp_limb_t  __tmp2 = __bswapl_src << 24;		/* high byte */	\
    __asm__ ("rlwimi %0, %2, 24, 16, 23"		/* 2nd low */	\
	 : "=r" (__tmp1) : "0" (__tmp1), "r" (__bswapl_src));		\
    __asm__ ("rlwimi %0, %2,  8,  8, 15"		/* 3nd high */	\
	 : "=r" (__tmp2) : "0" (__tmp2), "r" (__bswapl_src));		\
    (dst) = __tmp1 | __tmp2;				/* whole */	\
  } while (0)
#endif

/* bswap is available on i486 and up and is fast.  A combination rorw $8 /
   roll $16 / rorw $8 is used in glibc for plain i386 (and in the linux
   kernel with xchgb instead of rorw), but this is not done here, because
   i386 means generic x86 and mixing word and dword operations will cause
   partial register stalls on P6 chips.  */
#if defined (__GNUC__) && ! defined (NO_ASM)            \
  && HAVE_HOST_CPU_FAMILY_x86 && ! HAVE_HOST_CPU_i386   \
  && GMP_LIMB_BITS == 32
#define BSWAP_LIMB(dst, src)						\
  do {									\
    __asm__ ("bswap %0" : "=r" (dst) : "0" (src));			\
  } while (0)
#endif

#if defined (__GNUC__) && ! defined (NO_ASM)            \
  && defined (__amd64__) && GMP_LIMB_BITS == 64
#define BSWAP_LIMB(dst, src)						\
  do {									\
    __asm__ ("bswap %q0" : "=r" (dst) : "0" (src));			\
  } while (0)
#endif

#if defined (__GNUC__) && ! defined (__INTEL_COMPILER)			\
    && ! defined (NO_ASM) && defined (__ia64) && GMP_LIMB_BITS == 64
#define BSWAP_LIMB(dst, src)						\
  do {									\
    __asm__ ("mux1 %0 = %1, @rev" : "=r" (dst) :  "r" (src));		\
  } while (0)
#endif

/* As per glibc. */
#if defined (__GNUC__) && ! defined (NO_ASM)                    \
  && HAVE_HOST_CPU_FAMILY_m68k && GMP_LIMB_BITS == 32
#define BSWAP_LIMB(dst, src)						\
  do {									\
    mp_limb_t  __bswapl_src = (src);					\
    __asm__ ("ror%.w %#8, %0\n\t"					\
	     "swap   %0\n\t"						\
	     "ror%.w %#8, %0"						\
	     : "=d" (dst)						\
	     : "0" (__bswapl_src));					\
  } while (0)
#endif

#if ! defined (BSWAP_LIMB)
#if GMP_LIMB_BITS == 8
#define BSWAP_LIMB(dst, src)				\
  do { (dst) = (src); } while (0)
#endif
#if GMP_LIMB_BITS == 16
#define BSWAP_LIMB(dst, src)						\
  do {									\
    (dst) = ((src) << 8) + ((src) >> 8);				\
  } while (0)
#endif
#if GMP_LIMB_BITS == 32
#define BSWAP_LIMB(dst, src)						\
  do {									\
    (dst) =								\
      ((src) << 24)							\
      + (((src) & 0xFF00) << 8)						\
      + (((src) >> 8) & 0xFF00)						\
      + ((src) >> 24);							\
  } while (0)
#endif
#if GMP_LIMB_BITS == 64
#define BSWAP_LIMB(dst, src)						\
  do {									\
    (dst) =								\
      ((src) << 56)							\
      + (((src) & 0xFF00) << 40)					\
      + (((src) & 0xFF0000) << 24)					\
      + (((src) & 0xFF000000) << 8)					\
      + (((src) >> 8) & 0xFF000000)					\
      + (((src) >> 24) & 0xFF0000)					\
      + (((src) >> 40) & 0xFF00)					\
      + ((src) >> 56);							\
  } while (0)
#endif
#endif

#if ! defined (BSWAP_LIMB)
#define BSWAP_LIMB(dst, src)						\
  do {									\
    mp_limb_t  __bswapl_src = (src);					\
    mp_limb_t  __dstl = 0;						\
    int	       __i;							\
    for (__i = 0; __i < GMP_LIMB_BYTES; __i++)			\
      {									\
	__dstl = (__dstl << 8) | (__bswapl_src & 0xFF);			\
	__bswapl_src >>= 8;						\
      }									\
    (dst) = __dstl;							\
  } while (0)
#endif


/* Apparently lwbrx might be slow on some PowerPC chips, so restrict it to
   those we know are fast.  */
#if defined (__GNUC__) && ! defined (NO_ASM)				\
  && GMP_LIMB_BITS == 32 && HAVE_LIMB_BIG_ENDIAN			\
  && (HAVE_HOST_CPU_powerpc604						\
      || HAVE_HOST_CPU_powerpc604e					\
      || HAVE_HOST_CPU_powerpc750					\
      || HAVE_HOST_CPU_powerpc7400)
#define BSWAP_LIMB_FETCH(limb, src)					\
  do {									\
    mp_srcptr  __blf_src = (src);					\
    mp_limb_t  __limb;							\
    __asm__ ("lwbrx %0, 0, %1"						\
	     : "=r" (__limb)						\
	     : "r" (__blf_src),						\
	       "m" (*__blf_src));					\
    (limb) = __limb;							\
  } while (0)
#endif

#if ! defined (BSWAP_LIMB_FETCH)
#define BSWAP_LIMB_FETCH(limb, src)  BSWAP_LIMB (limb, *(src))
#endif


/* On the same basis that lwbrx might be slow, restrict stwbrx to those we
   know are fast.  FIXME: Is this necessary?  */
#if defined (__GNUC__) && ! defined (NO_ASM)				\
  && GMP_LIMB_BITS == 32 && HAVE_LIMB_BIG_ENDIAN			\
  && (HAVE_HOST_CPU_powerpc604						\
      || HAVE_HOST_CPU_powerpc604e					\
      || HAVE_HOST_CPU_powerpc750					\
      || HAVE_HOST_CPU_powerpc7400)
#define BSWAP_LIMB_STORE(dst, limb)					\
  do {									\
    mp_ptr     __dst = (dst);						\
    mp_limb_t  __limb = (limb);						\
    __asm__ ("stwbrx %1, 0, %2"						\
	     : "=m" (*__dst)						\
	     : "r" (__limb),						\
	       "r" (__dst));						\
  } while (0)
#endif

#if ! defined (BSWAP_LIMB_STORE)
#define BSWAP_LIMB_STORE(dst, limb)  BSWAP_LIMB (*(dst), limb)
#endif


/* Byte swap limbs from {src,size} and store at {dst,size}. */
#define MPN_BSWAP(dst, src, size)					\
  do {									\
    mp_ptr     __dst = (dst);						\
    mp_srcptr  __src = (src);						\
    mp_size_t  __size = (size);						\
    mp_size_t  __i;							\
    ASSERT ((size) >= 0);						\
    ASSERT (MPN_SAME_OR_SEPARATE_P (dst, src, size));			\
    CRAY_Pragma ("_CRI ivdep");						\
    for (__i = 0; __i < __size; __i++)					\
      {									\
	BSWAP_LIMB_FETCH (*__dst, __src);				\
	__dst++;							\
	__src++;							\
      }									\
  } while (0)

/* Byte swap limbs from {dst,size} and store in reverse order at {src,size}. */
#define MPN_BSWAP_REVERSE(dst, src, size)				\
  do {									\
    mp_ptr     __dst = (dst);						\
    mp_size_t  __size = (size);						\
    mp_srcptr  __src = (src) + __size - 1;				\
    mp_size_t  __i;							\
    ASSERT ((size) >= 0);						\
    ASSERT (! MPN_OVERLAP_P (dst, size, src, size));			\
    CRAY_Pragma ("_CRI ivdep");						\
    for (__i = 0; __i < __size; __i++)					\
      {									\
	BSWAP_LIMB_FETCH (*__dst, __src);				\
	__dst++;							\
	__src--;							\
      }									\
  } while (0)


/* No processor claiming to be SPARC v9 compliant seems to
   implement the POPC instruction.  Disable pattern for now.  */
#if 0
#if defined __GNUC__ && defined __sparc_v9__ && GMP_LIMB_BITS == 64
#define popc_limb(result, input)					\
  do {									\
    DItype __res;							\
    __asm__ ("popc %1,%0" : "=r" (result) : "rI" (input));		\
  } while (0)
#endif
#endif

#if defined (__GNUC__) && ! defined (NO_ASM) && HAVE_HOST_CPU_alpha_CIX
#define popc_limb(result, input)					\
  do {									\
    __asm__ ("ctpop %1, %0" : "=r" (result) : "r" (input));		\
  } while (0)
#endif

/* Cray intrinsic. */
#ifdef _CRAY
#define popc_limb(result, input)					\
  do {									\
    (result) = _popcnt (input);						\
  } while (0)
#endif

#if defined (__GNUC__) && ! defined (__INTEL_COMPILER)			\
    && ! defined (NO_ASM) && defined (__ia64) && GMP_LIMB_BITS == 64
#define popc_limb(result, input)					\
  do {									\
    __asm__ ("popcnt %0 = %1" : "=r" (result) : "r" (input));		\
  } while (0)
#endif

/* Cool population count of an mp_limb_t.
   You have to figure out how this works, We won't tell you!

   The constants could also be expressed as:
     0x55... = [2^N / 3]     = [(2^N-1)/3]
     0x33... = [2^N / 5]     = [(2^N-1)/5]
     0x0f... = [2^N / 17]    = [(2^N-1)/17]
     (N is GMP_LIMB_BITS, [] denotes truncation.) */

#if ! defined (popc_limb) && GMP_LIMB_BITS == 8
#define popc_limb(result, input)					\
  do {									\
    mp_limb_t  __x = (input);						\
    __x -= (__x >> 1) & MP_LIMB_T_MAX/3;				\
    __x = ((__x >> 2) & MP_LIMB_T_MAX/5) + (__x & MP_LIMB_T_MAX/5);	\
    __x = ((__x >> 4) + __x);						\
    (result) = __x & 0x0f;						\
  } while (0)
#endif

#if ! defined (popc_limb) && GMP_LIMB_BITS == 16
#define popc_limb(result, input)					\
  do {									\
    mp_limb_t  __x = (input);						\
    __x -= (__x >> 1) & MP_LIMB_T_MAX/3;				\
    __x = ((__x >> 2) & MP_LIMB_T_MAX/5) + (__x & MP_LIMB_T_MAX/5);	\
    __x = ((__x >> 4) + __x) & MP_LIMB_T_MAX/17;			\
    __x = ((__x >> 8) + __x);						\
    (result) = __x & 0xff;						\
  } while (0)
#endif

#if ! defined (popc_limb) && GMP_LIMB_BITS == 32
#define popc_limb(result, input)					\
  do {									\
    mp_limb_t  __x = (input);						\
    __x -= (__x >> 1) & MP_LIMB_T_MAX/3;				\
    __x = ((__x >> 2) & MP_LIMB_T_MAX/5) + (__x & MP_LIMB_T_MAX/5);	\
    __x = ((__x >> 4) + __x) & MP_LIMB_T_MAX/17;			\
    __x = ((__x >> 8) + __x);						\
    __x = ((__x >> 16) + __x);						\
    (result) = __x & 0xff;						\
  } while (0)
#endif

#if ! defined (popc_limb) && GMP_LIMB_BITS == 64
#define popc_limb(result, input)					\
  do {									\
    mp_limb_t  __x = (input);						\
    __x -= (__x >> 1) & MP_LIMB_T_MAX/3;				\
    __x = ((__x >> 2) & MP_LIMB_T_MAX/5) + (__x & MP_LIMB_T_MAX/5);	\
    __x = ((__x >> 4) + __x) & MP_LIMB_T_MAX/17;			\
    __x = ((__x >> 8) + __x);						\
    __x = ((__x >> 16) + __x);						\
    __x = ((__x >> 32) + __x);						\
    (result) = __x & 0xff;						\
  } while (0)
#endif


/* Define stuff for longlong.h.  */
#if HAVE_ATTRIBUTE_MODE
typedef unsigned int UQItype	__attribute__ ((mode (QI)));
typedef		 int SItype	__attribute__ ((mode (SI)));
typedef unsigned int USItype	__attribute__ ((mode (SI)));
typedef		 int DItype	__attribute__ ((mode (DI)));
typedef unsigned int UDItype	__attribute__ ((mode (DI)));
#else
typedef unsigned char UQItype;
typedef		 long SItype;
typedef unsigned long USItype;
#if HAVE_LONG_LONG
typedef	long long int DItype;
typedef unsigned long long int UDItype;
#else /* Assume `long' gives us a wide enough type.  Needed for hppa2.0w.  */
typedef long int DItype;
typedef unsigned long int UDItype;
#endif
#endif

typedef mp_limb_t UWtype;
typedef unsigned int UHWtype;
#define W_TYPE_SIZE GMP_LIMB_BITS

/* Define ieee_double_extract and _GMP_IEEE_FLOATS.

   Bit field packing is "implementation defined" according to C99, which
   leaves us at the compiler's mercy here.  For some systems packing is
   defined in the ABI (eg. x86).  In any case so far it seems universal that
   little endian systems pack from low to high, and big endian from high to
   low within the given type.

   Within the fields we rely on the integer endianness being the same as the
   float endianness, this is true everywhere we know of and it'd be a fairly
   strange system that did anything else.  */

#if HAVE_DOUBLE_IEEE_LITTLE_SWAPPED
#define _GMP_IEEE_FLOATS 1
union ieee_double_extract
{
  struct
    {
      gmp_uint_least32_t manh:20;
      gmp_uint_least32_t exp:11;
      gmp_uint_least32_t sig:1;
      gmp_uint_least32_t manl:32;
    } s;
  double d;
};
#endif

#if HAVE_DOUBLE_IEEE_LITTLE_ENDIAN
#define _GMP_IEEE_FLOATS 1
union ieee_double_extract
{
  struct
    {
      gmp_uint_least32_t manl:32;
      gmp_uint_least32_t manh:20;
      gmp_uint_least32_t exp:11;
      gmp_uint_least32_t sig:1;
    } s;
  double d;
};
#endif

#if HAVE_DOUBLE_IEEE_BIG_ENDIAN
#define _GMP_IEEE_FLOATS 1
union ieee_double_extract
{
  struct
    {
      gmp_uint_least32_t sig:1;
      gmp_uint_least32_t exp:11;
      gmp_uint_least32_t manh:20;
      gmp_uint_least32_t manl:32;
    } s;
  double d;
};
#endif

#if HAVE_DOUBLE_VAX_D
union double_extract
{
  struct
    {
      gmp_uint_least32_t man3:7;	/* highest 7 bits */
      gmp_uint_least32_t exp:8;		/* excess-128 exponent */
      gmp_uint_least32_t sig:1;
      gmp_uint_least32_t man2:16;
      gmp_uint_least32_t man1:16;
      gmp_uint_least32_t man0:16;	/* lowest 16 bits */
    } s;
  double d;
};
#endif

/* Use (4.0 * ...) instead of (2.0 * ...) to work around buggy compilers
   that don't convert ulong->double correctly (eg. SunOS 4 native cc).  */
#define MP_BASE_AS_DOUBLE (4.0 * ((mp_limb_t) 1 << (GMP_NUMB_BITS - 2)))
/* Maximum number of limbs it will take to store any `double'.
   We assume doubles have 53 mantissa bits.  */
#define LIMBS_PER_DOUBLE ((53 + GMP_NUMB_BITS - 2) / GMP_NUMB_BITS + 1)

__GMP_DECLSPEC int __gmp_extract_double (mp_ptr, double);

#define mpn_get_d __gmpn_get_d
__GMP_DECLSPEC double mpn_get_d (mp_srcptr, mp_size_t, mp_size_t, long) __GMP_ATTRIBUTE_PURE;


/* DOUBLE_NAN_INF_ACTION executes code a_nan if x is a NaN, or executes
   a_inf if x is an infinity.  Both are considered unlikely values, for
   branch prediction.  */

#if _GMP_IEEE_FLOATS
#define DOUBLE_NAN_INF_ACTION(x, a_nan, a_inf)				\
  do {									\
    union ieee_double_extract  u;					\
    u.d = (x);								\
    if (UNLIKELY (u.s.exp == 0x7FF))					\
      {									\
	if (u.s.manl == 0 && u.s.manh == 0)				\
	  { a_inf; }							\
	else								\
	  { a_nan; }							\
      }									\
  } while (0)
#endif

#if HAVE_DOUBLE_VAX_D || HAVE_DOUBLE_VAX_G || HAVE_DOUBLE_CRAY_CFP
/* no nans or infs in these formats */
#define DOUBLE_NAN_INF_ACTION(x, a_nan, a_inf)  \
  do { } while (0)
#endif

#ifndef DOUBLE_NAN_INF_ACTION
/* Unknown format, try something generic.
   NaN should be "unordered", so x!=x.
   Inf should be bigger than DBL_MAX.  */
#define DOUBLE_NAN_INF_ACTION(x, a_nan, a_inf)				\
  do {									\
    {									\
      if (UNLIKELY ((x) != (x)))					\
	{ a_nan; }							\
      else if (UNLIKELY ((x) > DBL_MAX || (x) < -DBL_MAX))		\
	{ a_inf; }							\
    }									\
  } while (0)
#endif

/* On m68k, x86 and amd64, gcc (and maybe other compilers) can hold doubles
   in the coprocessor, which means a bigger exponent range than normal, and
   depending on the rounding mode, a bigger mantissa than normal.  (See
   "Disappointments" in the gcc manual.)  FORCE_DOUBLE stores and fetches
   "d" through memory to force any rounding and overflows to occur.

   On amd64, and on x86s with SSE2, gcc (depending on options) uses the xmm
   registers, where there's no such extra precision and no need for the
   FORCE_DOUBLE.  We don't bother to detect this since the present uses for
   FORCE_DOUBLE are only in test programs and default generic C code.

   Not quite sure that an "automatic volatile" will use memory, but it does
   in gcc.  An asm("":"=m"(d):"0"(d)) can't be used to trick gcc, since
   apparently matching operands like "0" are only allowed on a register
   output.  gcc 3.4 warns about this, though in fact it and past versions
   seem to put the operand through memory as hoped.  */

#if (HAVE_HOST_CPU_FAMILY_m68k || HAVE_HOST_CPU_FAMILY_x86      \
     || defined (__amd64__))
#define FORCE_DOUBLE(d) \
  do { volatile double __gmp_force = (d); (d) = __gmp_force; } while (0)
#else
#define FORCE_DOUBLE(d)  do { } while (0)
#endif


__GMP_DECLSPEC extern const unsigned char __gmp_digit_value_tab[];

__GMP_DECLSPEC extern int __gmp_junk;
__GMP_DECLSPEC extern const int __gmp_0;
__GMP_DECLSPEC void __gmp_exception (int) ATTRIBUTE_NORETURN;
__GMP_DECLSPEC void __gmp_divide_by_zero (void) ATTRIBUTE_NORETURN;
__GMP_DECLSPEC void __gmp_sqrt_of_negative (void) ATTRIBUTE_NORETURN;
__GMP_DECLSPEC void __gmp_invalid_operation (void) ATTRIBUTE_NORETURN;
#define GMP_ERROR(code)   __gmp_exception (code)
#define DIVIDE_BY_ZERO    __gmp_divide_by_zero ()
#define SQRT_OF_NEGATIVE  __gmp_sqrt_of_negative ()

#if defined _LONG_LONG_LIMB
#define CNST_LIMB(C) ((mp_limb_t) C##LL)
#else /* not _LONG_LONG_LIMB */
#define CNST_LIMB(C) ((mp_limb_t) C##L)
#endif /* _LONG_LONG_LIMB */

/* Stuff used by mpn/generic/perfsqr.c and mpz/prime_p.c */
#if GMP_NUMB_BITS == 2
#define PP 0x3					/* 3 */
#define PP_FIRST_OMITTED 5
#endif
#if GMP_NUMB_BITS == 4
#define PP 0xF					/* 3 x 5 */
#define PP_FIRST_OMITTED 7
#endif
#if GMP_NUMB_BITS == 8
#define PP 0x69					/* 3 x 5 x 7 */
#define PP_FIRST_OMITTED 11
#endif
#if GMP_NUMB_BITS == 16
#define PP 0x3AA7				/* 3 x 5 x 7 x 11 x 13 */
#define PP_FIRST_OMITTED 17
#endif
#if GMP_NUMB_BITS == 32
#define PP 0xC0CFD797L				/* 3 x 5 x 7 x 11 x ... x 29 */
#define PP_INVERTED 0x53E5645CL
#define PP_FIRST_OMITTED 31
#endif
#if GMP_NUMB_BITS == 64
#define PP CNST_LIMB(0xE221F97C30E94E1D)	/* 3 x 5 x 7 x 11 x ... x 53 */
#define PP_INVERTED CNST_LIMB(0x21CFE6CFC938B36B)
#define PP_FIRST_OMITTED 59
#endif
#ifndef PP_FIRST_OMITTED
#define PP_FIRST_OMITTED 3
#endif

/* BIT1 means a result value in bit 1 (second least significant bit), with a
   zero bit representing +1 and a one bit representing -1.  Bits other than
   bit 1 are garbage.  These are meant to be kept in "int"s, and casts are
   used to ensure the expressions are "int"s even if a and/or b might be
   other types.

   JACOBI_TWOS_U_BIT1 and JACOBI_RECIP_UU_BIT1 are used in mpn_jacobi_base
   and their speed is important.  Expressions are used rather than
   conditionals to accumulate sign changes, which effectively means XORs
   instead of conditional JUMPs. */

/* (a/0), with a signed; is 1 if a=+/-1, 0 otherwise */
#define JACOBI_S0(a)   (((a) == 1) | ((a) == -1))

/* (a/0), with a unsigned; is 1 if a=+/-1, 0 otherwise */
#define JACOBI_U0(a)   ((a) == 1)

/* FIXME: JACOBI_LS0 and JACOBI_0LS are the same, so delete one and
   come up with a better name. */

/* (a/0), with a given by low and size;
   is 1 if a=+/-1, 0 otherwise */
#define JACOBI_LS0(alow,asize) \
  (((asize) == 1 || (asize) == -1) && (alow) == 1)

/* (a/0), with a an mpz_t;
   fetch of low limb always valid, even if size is zero */
#define JACOBI_Z0(a)   JACOBI_LS0 (PTR(a)[0], SIZ(a))

/* (0/b), with b unsigned; is 1 if b=1, 0 otherwise */
#define JACOBI_0U(b)   ((b) == 1)

/* (0/b), with b unsigned; is 1 if b=+/-1, 0 otherwise */
#define JACOBI_0S(b)   ((b) == 1 || (b) == -1)

/* (0/b), with b given by low and size; is 1 if b=+/-1, 0 otherwise */
#define JACOBI_0LS(blow,bsize) \
  (((bsize) == 1 || (bsize) == -1) && (blow) == 1)

/* Convert a bit1 to +1 or -1. */
#define JACOBI_BIT1_TO_PN(result_bit1) \
  (1 - ((int) (result_bit1) & 2))

/* (2/b), with b unsigned and odd;
   is (-1)^((b^2-1)/8) which is 1 if b==1,7mod8 or -1 if b==3,5mod8 and
   hence obtained from (b>>1)^b */
#define JACOBI_TWO_U_BIT1(b) \
  ((int) (((b) >> 1) ^ (b)))

/* (2/b)^twos, with b unsigned and odd */
#define JACOBI_TWOS_U_BIT1(twos, b) \
  ((int) ((twos) << 1) & JACOBI_TWO_U_BIT1 (b))

/* (2/b)^twos, with b unsigned and odd */
#define JACOBI_TWOS_U(twos, b) \
  (JACOBI_BIT1_TO_PN (JACOBI_TWOS_U_BIT1 (twos, b)))

/* (-1/b), with b odd (signed or unsigned);
   is (-1)^((b-1)/2) */
#define JACOBI_N1B_BIT1(b) \
  ((int) (b))

/* (a/b) effect due to sign of a: signed/unsigned, b odd;
   is (-1/b) if a<0, or +1 if a>=0 */
#define JACOBI_ASGN_SU_BIT1(a, b) \
  ((((a) < 0) << 1) & JACOBI_N1B_BIT1(b))

/* (a/b) effect due to sign of b: signed/signed;
   is -1 if a and b both negative, +1 otherwise */
#define JACOBI_BSGN_SS_BIT1(a, b) \
  ((((a)<0) & ((b)<0)) << 1)

/* (a/b) effect due to sign of b: signed/mpz;
   is -1 if a and b both negative, +1 otherwise */
#define JACOBI_BSGN_SZ_BIT1(a, b) \
  JACOBI_BSGN_SS_BIT1 (a, SIZ(b))

/* (a/b) effect due to sign of b: mpz/signed;
   is -1 if a and b both negative, +1 otherwise */
#define JACOBI_BSGN_ZS_BIT1(a, b) \
  JACOBI_BSGN_SZ_BIT1 (b, a)

/* (a/b) reciprocity to switch to (b/a), a,b both unsigned and odd;
   is (-1)^((a-1)*(b-1)/4), which means +1 if either a,b==1mod4, or -1 if
   both a,b==3mod4, achieved in bit 1 by a&b.  No ASSERT()s about a,b odd
   because this is used in a couple of places with only bit 1 of a or b
   valid. */
#define JACOBI_RECIP_UU_BIT1(a, b) \
  ((int) ((a) & (b)))

/* Strip low zero limbs from {b_ptr,b_size} by incrementing b_ptr and
   decrementing b_size.  b_low should be b_ptr[0] on entry, and will be
   updated for the new b_ptr.  result_bit1 is updated according to the
   factors of 2 stripped, as per (a/2).  */
#define JACOBI_STRIP_LOW_ZEROS(result_bit1, a, b_ptr, b_size, b_low)	\
  do {									\
    ASSERT ((b_size) >= 1);						\
    ASSERT ((b_low) == (b_ptr)[0]);					\
									\
    while (UNLIKELY ((b_low) == 0))					\
      {									\
	(b_size)--;							\
	ASSERT ((b_size) >= 1);						\
	(b_ptr)++;							\
	(b_low) = *(b_ptr);						\
									\
	ASSERT (((a) & 1) != 0);					\
	if ((GMP_NUMB_BITS % 2) == 1)					\
	  (result_bit1) ^= JACOBI_TWO_U_BIT1(a);			\
      }									\
  } while (0)

/* Set a_rem to {a_ptr,a_size} reduced modulo b, either using mod_1 or
   modexact_1_odd, but in either case leaving a_rem<b.  b must be odd and
   unsigned.  modexact_1_odd effectively calculates -a mod b, and
   result_bit1 is adjusted for the factor of -1.

   The way mpn_modexact_1_odd sometimes bases its remainder on a_size and
   sometimes on a_size-1 means if GMP_NUMB_BITS is odd we can't know what
   factor to introduce into result_bit1, so for that case use mpn_mod_1
   unconditionally.

   FIXME: mpn_modexact_1_odd is more efficient, so some way to get it used
   for odd GMP_NUMB_BITS would be good.  Perhaps it could mung its result,
   or not skip a divide step, or something. */

#define JACOBI_MOD_OR_MODEXACT_1_ODD(result_bit1, a_rem, a_ptr, a_size, b) \
  do {									   \
    mp_srcptr  __a_ptr	= (a_ptr);					   \
    mp_size_t  __a_size = (a_size);					   \
    mp_limb_t  __b	= (b);						   \
									   \
    ASSERT (__a_size >= 1);						   \
    ASSERT (__b & 1);							   \
									   \
    if ((GMP_NUMB_BITS % 2) != 0					   \
	|| ABOVE_THRESHOLD (__a_size, BMOD_1_TO_MOD_1_THRESHOLD))	   \
      {									   \
	(a_rem) = mpn_mod_1 (__a_ptr, __a_size, __b);			   \
      }									   \
    else								   \
      {									   \
	(result_bit1) ^= JACOBI_N1B_BIT1 (__b);				   \
	(a_rem) = mpn_modexact_1_odd (__a_ptr, __a_size, __b);		   \
      }									   \
  } while (0)

/* State for the Jacobi computation using Lehmer. */
#define jacobi_table __gmp_jacobi_table
__GMP_DECLSPEC extern const unsigned char jacobi_table[208];

/* Bit layout for the initial state. b must be odd.

      3  2  1 0
   +--+--+--+--+
   |a1|a0|b1| s|
   +--+--+--+--+

 */
static inline unsigned
mpn_jacobi_init (unsigned a, unsigned b, unsigned s)
{
  ASSERT (b & 1);
  ASSERT (s <= 1);
  return ((a & 3) << 2) + (b & 2) + s;
}

static inline int
mpn_jacobi_finish (unsigned bits)
{
  /* (a, b) = (1,0) or (0,1) */
  ASSERT ( (bits & 14) == 0);

  return 1-2*(bits & 1);
}

static inline unsigned
mpn_jacobi_update (unsigned bits, unsigned denominator, unsigned q)
{
  /* FIXME: Could halve table size by not including the e bit in the
   * index, and instead xor when updating. Then the lookup would be
   * like
   *
   *   bits ^= table[((bits & 30) << 2) + (denominator << 2) + q];
   */

  ASSERT (bits < 26);
  ASSERT (denominator < 2);
  ASSERT (q < 4);

  /* For almost all calls, denominator is constant and quite often q
     is constant too. So use addition rather than or, so the compiler
     can put the constant part can into the offset of an indexed
     addressing instruction.

     With constant denominator, the below table lookup is compiled to

       C Constant q = 1, constant denominator = 1
       movzbl table+5(%eax,8), %eax

     or

       C q in %edx, constant denominator = 1
       movzbl table+4(%edx,%eax,8), %eax

     One could maintain the state preshifted 3 bits, to save a shift
     here, but at least on x86, that's no real saving.
  */
  return bits = jacobi_table[(bits << 3) + (denominator << 2) + q];
}

/* Matrix multiplication */
#define   mpn_matrix22_mul __MPN(matrix22_mul)
__GMP_DECLSPEC void      mpn_matrix22_mul (mp_ptr, mp_ptr, mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_srcptr, mp_srcptr, mp_srcptr, mp_size_t, mp_ptr);
#define   mpn_matrix22_mul_strassen __MPN(matrix22_mul_strassen)
__GMP_DECLSPEC void      mpn_matrix22_mul_strassen (mp_ptr, mp_ptr, mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_srcptr, mp_srcptr, mp_srcptr, mp_size_t, mp_ptr);
#define   mpn_matrix22_mul_itch __MPN(matrix22_mul_itch)
__GMP_DECLSPEC mp_size_t mpn_matrix22_mul_itch (mp_size_t, mp_size_t) ATTRIBUTE_CONST;

#ifndef MATRIX22_STRASSEN_THRESHOLD
#define MATRIX22_STRASSEN_THRESHOLD 30
#endif

/* HGCD definitions */

/* Extract one numb, shifting count bits left
    ________  ________
   |___xh___||___xl___|
	  |____r____|
   >count <

   The count includes any nail bits, so it should work fine if count
   is computed using count_leading_zeros. If GMP_NAIL_BITS > 0, all of
   xh, xl and r include nail bits. Must have 0 < count < GMP_LIMB_BITS.

   FIXME: Omit masking with GMP_NUMB_MASK, and let callers do that for
   those calls where the count high bits of xh may be non-zero.
*/

#define MPN_EXTRACT_NUMB(count, xh, xl)				\
  ((((xh) << ((count) - GMP_NAIL_BITS)) & GMP_NUMB_MASK) |	\
   ((xl) >> (GMP_LIMB_BITS - (count))))


/* The matrix non-negative M = (u, u'; v,v') keeps track of the
   reduction (a;b) = M (alpha; beta) where alpha, beta are smaller
   than a, b. The determinant must always be one, so that M has an
   inverse (v', -u'; -v, u). Elements always fit in GMP_NUMB_BITS - 1
   bits. */
struct hgcd_matrix1
{
  mp_limb_t u[2][2];
};

#define mpn_hgcd2 __MPN (hgcd2)
__GMP_DECLSPEC int mpn_hgcd2 (mp_limb_t, mp_limb_t, mp_limb_t, mp_limb_t,	struct hgcd_matrix1 *);

#define mpn_hgcd_mul_matrix1_vector __MPN (hgcd_mul_matrix1_vector)
__GMP_DECLSPEC mp_size_t mpn_hgcd_mul_matrix1_vector (const struct hgcd_matrix1 *, mp_ptr, mp_srcptr, mp_ptr, mp_size_t);

#define mpn_matrix22_mul1_inverse_vector __MPN (matrix22_mul1_inverse_vector)
__GMP_DECLSPEC mp_size_t mpn_matrix22_mul1_inverse_vector (const struct hgcd_matrix1 *, mp_ptr, mp_srcptr, mp_ptr, mp_size_t);

#define mpn_hgcd2_jacobi __MPN (hgcd2_jacobi)
__GMP_DECLSPEC int mpn_hgcd2_jacobi (mp_limb_t, mp_limb_t, mp_limb_t, mp_limb_t, struct hgcd_matrix1 *, unsigned *);

struct hgcd_matrix
{
  mp_size_t alloc;		/* for sanity checking only */
  mp_size_t n;
  mp_ptr p[2][2];
};

#define MPN_HGCD_MATRIX_INIT_ITCH(n) (4 * ((n+1)/2 + 1))

#define mpn_hgcd_matrix_init __MPN (hgcd_matrix_init)
__GMP_DECLSPEC void mpn_hgcd_matrix_init (struct hgcd_matrix *, mp_size_t, mp_ptr);

#define mpn_hgcd_matrix_update_q __MPN (hgcd_matrix_update_q)
__GMP_DECLSPEC void mpn_hgcd_matrix_update_q (struct hgcd_matrix *, mp_srcptr, mp_size_t, unsigned, mp_ptr);

#define mpn_hgcd_matrix_mul_1 __MPN (hgcd_matrix_mul_1)
__GMP_DECLSPEC void mpn_hgcd_matrix_mul_1 (struct hgcd_matrix *, const struct hgcd_matrix1 *, mp_ptr);

#define mpn_hgcd_matrix_mul __MPN (hgcd_matrix_mul)
__GMP_DECLSPEC void mpn_hgcd_matrix_mul (struct hgcd_matrix *, const struct hgcd_matrix *, mp_ptr);

#define mpn_hgcd_matrix_adjust __MPN (hgcd_matrix_adjust)
__GMP_DECLSPEC mp_size_t mpn_hgcd_matrix_adjust (const struct hgcd_matrix *, mp_size_t, mp_ptr, mp_ptr, mp_size_t, mp_ptr);

#define mpn_hgcd_step __MPN(hgcd_step)
__GMP_DECLSPEC mp_size_t mpn_hgcd_step (mp_size_t, mp_ptr, mp_ptr, mp_size_t, struct hgcd_matrix *, mp_ptr);

#define mpn_hgcd_reduce __MPN(hgcd_reduce)
__GMP_DECLSPEC mp_size_t mpn_hgcd_reduce (struct hgcd_matrix *, mp_ptr, mp_ptr, mp_size_t, mp_size_t, mp_ptr);

#define mpn_hgcd_reduce_itch __MPN(hgcd_reduce_itch)
__GMP_DECLSPEC mp_size_t mpn_hgcd_reduce_itch (mp_size_t, mp_size_t) ATTRIBUTE_CONST;

#define mpn_hgcd_itch __MPN (hgcd_itch)
__GMP_DECLSPEC mp_size_t mpn_hgcd_itch (mp_size_t) ATTRIBUTE_CONST;

#define mpn_hgcd __MPN (hgcd)
__GMP_DECLSPEC mp_size_t mpn_hgcd (mp_ptr, mp_ptr, mp_size_t, struct hgcd_matrix *, mp_ptr);

#define mpn_hgcd_appr_itch __MPN (hgcd_appr_itch)
__GMP_DECLSPEC mp_size_t mpn_hgcd_appr_itch (mp_size_t) ATTRIBUTE_CONST;

#define mpn_hgcd_appr __MPN (hgcd_appr)
__GMP_DECLSPEC int mpn_hgcd_appr (mp_ptr, mp_ptr, mp_size_t, struct hgcd_matrix *, mp_ptr);

#define mpn_hgcd_jacobi __MPN (hgcd_jacobi)
__GMP_DECLSPEC mp_size_t mpn_hgcd_jacobi (mp_ptr, mp_ptr, mp_size_t, struct hgcd_matrix *, unsigned *, mp_ptr);

typedef void gcd_subdiv_step_hook(void *, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t, int);

/* Needs storage for the quotient */
#define MPN_GCD_SUBDIV_STEP_ITCH(n) (n)

#define mpn_gcd_subdiv_step __MPN(gcd_subdiv_step)
__GMP_DECLSPEC mp_size_t mpn_gcd_subdiv_step (mp_ptr, mp_ptr, mp_size_t, mp_size_t, gcd_subdiv_step_hook *, void *, mp_ptr);

struct gcdext_ctx
{
  /* Result parameters. */
  mp_ptr gp;
  mp_size_t gn;
  mp_ptr up;
  mp_size_t *usize;

  /* Cofactors updated in each step. */
  mp_size_t un;
  mp_ptr u0, u1, tp;
};

#define mpn_gcdext_hook __MPN (gcdext_hook)
gcd_subdiv_step_hook mpn_gcdext_hook;

#define MPN_GCDEXT_LEHMER_N_ITCH(n) (4*(n) + 3)

#define mpn_gcdext_lehmer_n __MPN(gcdext_lehmer_n)
__GMP_DECLSPEC mp_size_t mpn_gcdext_lehmer_n (mp_ptr, mp_ptr, mp_size_t *, mp_ptr, mp_ptr, mp_size_t, mp_ptr);

/* 4*(an + 1) + 4*(bn + 1) + an */
#define MPN_GCDEXT_LEHMER_ITCH(an, bn) (5*(an) + 4*(bn) + 8)

#ifndef HGCD_THRESHOLD
#define HGCD_THRESHOLD 400
#endif

#ifndef HGCD_APPR_THRESHOLD
#define HGCD_APPR_THRESHOLD 400
#endif

#ifndef HGCD_REDUCE_THRESHOLD
#define HGCD_REDUCE_THRESHOLD 1000
#endif

#ifndef GCD_DC_THRESHOLD
#define GCD_DC_THRESHOLD 1000
#endif

#ifndef GCDEXT_DC_THRESHOLD
#define GCDEXT_DC_THRESHOLD 600
#endif

/* Definitions for mpn_set_str and mpn_get_str */
struct powers
{
  mp_ptr p;			/* actual power value */
  mp_size_t n;			/* # of limbs at p */
  mp_size_t shift;		/* weight of lowest limb, in limb base B */
  size_t digits_in_base;	/* number of corresponding digits */
  int base;
};
typedef struct powers powers_t;
#define mpn_dc_set_str_powtab_alloc(n) ((n) + GMP_LIMB_BITS)
#define mpn_dc_set_str_itch(n) ((n) + GMP_LIMB_BITS)
#define mpn_dc_get_str_powtab_alloc(n) ((n) + 2 * GMP_LIMB_BITS)
#define mpn_dc_get_str_itch(n) ((n) + GMP_LIMB_BITS)

#define   mpn_dc_set_str __MPN(dc_set_str)
__GMP_DECLSPEC mp_size_t mpn_dc_set_str (mp_ptr, const unsigned char *, size_t, const powers_t *, mp_ptr);
#define   mpn_bc_set_str __MPN(bc_set_str)
__GMP_DECLSPEC mp_size_t mpn_bc_set_str (mp_ptr, const unsigned char *, size_t, int);
#define   mpn_set_str_compute_powtab __MPN(set_str_compute_powtab)
__GMP_DECLSPEC void      mpn_set_str_compute_powtab (powers_t *, mp_ptr, mp_size_t, int);


/* __GMPF_BITS_TO_PREC applies a minimum 53 bits, rounds upwards to a whole
   limb and adds an extra limb.  __GMPF_PREC_TO_BITS drops that extra limb,
   hence giving back the user's size in bits rounded up.  Notice that
   converting prec->bits->prec gives an unchanged value.  */
#define __GMPF_BITS_TO_PREC(n)						\
  ((mp_size_t) ((__GMP_MAX (53, n) + 2 * GMP_NUMB_BITS - 1) / GMP_NUMB_BITS))
#define __GMPF_PREC_TO_BITS(n) \
  ((mp_bitcnt_t) (n) * GMP_NUMB_BITS - GMP_NUMB_BITS)

__GMP_DECLSPEC extern mp_size_t __gmp_default_fp_limb_precision;

/* Compute the number of base-b digits corresponding to nlimbs limbs, rounding
   down.  */
#define DIGITS_IN_BASE_PER_LIMB(res, nlimbs, b)				\
  do {									\
    mp_limb_t _ph, _dummy;						\
    umul_ppmm (_ph, _dummy,						\
	       mp_bases[b].logb2, GMP_NUMB_BITS * (mp_limb_t) (nlimbs));\
    res = _ph;								\
  } while (0)

/* Compute the number of limbs corresponding to ndigits base-b digits, rounding
   up.  */
#define LIMBS_PER_DIGIT_IN_BASE(res, ndigits, b)			\
  do {									\
    mp_limb_t _ph, _dummy;						\
    umul_ppmm (_ph, _dummy, mp_bases[b].log2b, (mp_limb_t) (ndigits));	\
    res = 8 * _ph / GMP_NUMB_BITS + 2;					\
  } while (0)


/* Set n to the number of significant digits an mpf of the given _mp_prec
   field, in the given base.  This is a rounded up value, designed to ensure
   there's enough digits to reproduce all the guaranteed part of the value.

   There are prec many limbs, but the high might be only "1" so forget it
   and just count prec-1 limbs into chars.  +1 rounds that upwards, and a
   further +1 is because the limbs usually won't fall on digit boundaries.

   FIXME: If base is a power of 2 and the bits per digit divides
   GMP_LIMB_BITS then the +2 is unnecessary.  This happens always for
   base==2, and in base==16 with the current 32 or 64 bit limb sizes. */

#define MPF_SIGNIFICANT_DIGITS(n, base, prec)				\
  do {									\
    size_t rawn;							\
    ASSERT (base >= 2 && base < numberof (mp_bases));			\
    DIGITS_IN_BASE_PER_LIMB (rawn, (prec) - 1, base);			\
    n = rawn + 2;							\
  } while (0)


/* Decimal point string, from the current C locale.  Needs <langinfo.h> for
   nl_langinfo and constants, preferably with _GNU_SOURCE defined to get
   DECIMAL_POINT from glibc, and needs <locale.h> for localeconv, each under
   their respective #if HAVE_FOO_H.

   GLIBC recommends nl_langinfo because getting only one facet can be
   faster, apparently. */

/* DECIMAL_POINT seems to need _GNU_SOURCE defined to get it from glibc. */
#if HAVE_NL_LANGINFO && defined (DECIMAL_POINT)
#define GMP_DECIMAL_POINT  (nl_langinfo (DECIMAL_POINT))
#endif
/* RADIXCHAR is deprecated, still in unix98 or some such. */
#if HAVE_NL_LANGINFO && defined (RADIXCHAR) && ! defined (GMP_DECIMAL_POINT)
#define GMP_DECIMAL_POINT  (nl_langinfo (RADIXCHAR))
#endif
/* localeconv is slower since it returns all locale stuff */
#if HAVE_LOCALECONV && ! defined (GMP_DECIMAL_POINT)
#define GMP_DECIMAL_POINT  (localeconv()->decimal_point)
#endif
#if ! defined (GMP_DECIMAL_POINT)
#define GMP_DECIMAL_POINT  (".")
#endif


#define DOPRNT_CONV_FIXED        1
#define DOPRNT_CONV_SCIENTIFIC   2
#define DOPRNT_CONV_GENERAL      3

#define DOPRNT_JUSTIFY_NONE      0
#define DOPRNT_JUSTIFY_LEFT      1
#define DOPRNT_JUSTIFY_RIGHT     2
#define DOPRNT_JUSTIFY_INTERNAL  3

#define DOPRNT_SHOWBASE_YES      1
#define DOPRNT_SHOWBASE_NO       2
#define DOPRNT_SHOWBASE_NONZERO  3

struct doprnt_params_t {
  int         base;          /* negative for upper case */
  int         conv;          /* choices above */
  const char  *expfmt;       /* exponent format */
  int         exptimes4;     /* exponent multiply by 4 */
  char        fill;          /* character */
  int         justify;       /* choices above */
  int         prec;          /* prec field, or -1 for all digits */
  int         showbase;      /* choices above */
  int         showpoint;     /* if radix point always shown */
  int         showtrailing;  /* if trailing zeros wanted */
  char        sign;          /* '+', ' ', or '\0' */
  int         width;         /* width field */
};

#if _GMP_H_HAVE_VA_LIST

typedef int (*doprnt_format_t) (void *, const char *, va_list);
typedef int (*doprnt_memory_t) (void *, const char *, size_t);
typedef int (*doprnt_reps_t)   (void *, int, int);
typedef int (*doprnt_final_t)  (void *);

struct doprnt_funs_t {
  doprnt_format_t  format;
  doprnt_memory_t  memory;
  doprnt_reps_t    reps;
  doprnt_final_t   final;   /* NULL if not required */
};

extern const struct doprnt_funs_t  __gmp_fprintf_funs;
extern const struct doprnt_funs_t  __gmp_sprintf_funs;
extern const struct doprnt_funs_t  __gmp_snprintf_funs;
extern const struct doprnt_funs_t  __gmp_obstack_printf_funs;
extern const struct doprnt_funs_t  __gmp_ostream_funs;

/* "buf" is a __gmp_allocate_func block of "alloc" many bytes.  The first
   "size" of these have been written.  "alloc > size" is maintained, so
   there's room to store a '\0' at the end.  "result" is where the
   application wants the final block pointer.  */
struct gmp_asprintf_t {
  char    **result;
  char    *buf;
  size_t  size;
  size_t  alloc;
};

#define GMP_ASPRINTF_T_INIT(d, output)					\
  do {									\
    (d).result = (output);						\
    (d).alloc = 256;							\
    (d).buf = (char *) (*__gmp_allocate_func) ((d).alloc);		\
    (d).size = 0;							\
  } while (0)

/* If a realloc is necessary, use twice the size actually required, so as to
   avoid repeated small reallocs.  */
#define GMP_ASPRINTF_T_NEED(d, n)					\
  do {									\
    size_t  alloc, newsize, newalloc;					\
    ASSERT ((d)->alloc >= (d)->size + 1);				\
									\
    alloc = (d)->alloc;							\
    newsize = (d)->size + (n);						\
    if (alloc <= newsize)						\
      {									\
	newalloc = 2*newsize;						\
	(d)->alloc = newalloc;						\
	(d)->buf = __GMP_REALLOCATE_FUNC_TYPE ((d)->buf,		\
					       alloc, newalloc, char);	\
      }									\
  } while (0)

__GMP_DECLSPEC int __gmp_asprintf_memory (struct gmp_asprintf_t *, const char *, size_t);
__GMP_DECLSPEC int __gmp_asprintf_reps (struct gmp_asprintf_t *, int, int);
__GMP_DECLSPEC int __gmp_asprintf_final (struct gmp_asprintf_t *);

/* buf is where to write the next output, and size is how much space is left
   there.  If the application passed size==0 then that's what we'll have
   here, and nothing at all should be written.  */
struct gmp_snprintf_t {
  char    *buf;
  size_t  size;
};

/* Add the bytes printed by the call to the total retval, or bail out on an
   error.  */
#define DOPRNT_ACCUMULATE(call)						\
  do {									\
    int  __ret;								\
    __ret = call;							\
    if (__ret == -1)							\
      goto error;							\
    retval += __ret;							\
  } while (0)
#define DOPRNT_ACCUMULATE_FUN(fun, params)				\
  do {									\
    ASSERT ((fun) != NULL);						\
    DOPRNT_ACCUMULATE ((*(fun)) params);				\
  } while (0)

#define DOPRNT_FORMAT(fmt, ap)						\
  DOPRNT_ACCUMULATE_FUN (funs->format, (data, fmt, ap))
#define DOPRNT_MEMORY(ptr, len)						\
  DOPRNT_ACCUMULATE_FUN (funs->memory, (data, ptr, len))
#define DOPRNT_REPS(c, n)						\
  DOPRNT_ACCUMULATE_FUN (funs->reps, (data, c, n))

#define DOPRNT_STRING(str)      DOPRNT_MEMORY (str, strlen (str))

#define DOPRNT_REPS_MAYBE(c, n)						\
  do {									\
    if ((n) != 0)							\
      DOPRNT_REPS (c, n);						\
  } while (0)
#define DOPRNT_MEMORY_MAYBE(ptr, len)					\
  do {									\
    if ((len) != 0)							\
      DOPRNT_MEMORY (ptr, len);						\
  } while (0)

__GMP_DECLSPEC int __gmp_doprnt (const struct doprnt_funs_t *, void *, const char *, va_list);
__GMP_DECLSPEC int __gmp_doprnt_integer (const struct doprnt_funs_t *, void *, const struct doprnt_params_t *, const char *);

#define __gmp_doprnt_mpf __gmp_doprnt_mpf2
__GMP_DECLSPEC int __gmp_doprnt_mpf (const struct doprnt_funs_t *, void *, const struct doprnt_params_t *, const char *, mpf_srcptr);

__GMP_DECLSPEC int __gmp_replacement_vsnprintf (char *, size_t, const char *, va_list);
#endif /* _GMP_H_HAVE_VA_LIST */


typedef int (*gmp_doscan_scan_t)  (void *, const char *, ...);
typedef void *(*gmp_doscan_step_t) (void *, int);
typedef int (*gmp_doscan_get_t)   (void *);
typedef int (*gmp_doscan_unget_t) (int, void *);

struct gmp_doscan_funs_t {
  gmp_doscan_scan_t   scan;
  gmp_doscan_step_t   step;
  gmp_doscan_get_t    get;
  gmp_doscan_unget_t  unget;
};
extern const struct gmp_doscan_funs_t  __gmp_fscanf_funs;
extern const struct gmp_doscan_funs_t  __gmp_sscanf_funs;

#if _GMP_H_HAVE_VA_LIST
__GMP_DECLSPEC int __gmp_doscan (const struct gmp_doscan_funs_t *, void *, const char *, va_list);
#endif


/* For testing and debugging.  */
#define MPZ_CHECK_FORMAT(z)						\
  do {									\
    ASSERT_ALWAYS (SIZ(z) == 0 || PTR(z)[ABSIZ(z) - 1] != 0);		\
    ASSERT_ALWAYS (ALLOC(z) >= ABSIZ(z));				\
    ASSERT_ALWAYS_MPN (PTR(z), ABSIZ(z));				\
  } while (0)

#define MPQ_CHECK_FORMAT(q)						\
  do {									\
    MPZ_CHECK_FORMAT (mpq_numref (q));					\
    MPZ_CHECK_FORMAT (mpq_denref (q));					\
    ASSERT_ALWAYS (SIZ(mpq_denref(q)) >= 1);				\
									\
    if (SIZ(mpq_numref(q)) == 0)					\
      {									\
	/* should have zero as 0/1 */					\
	ASSERT_ALWAYS (SIZ(mpq_denref(q)) == 1				\
		       && PTR(mpq_denref(q))[0] == 1);			\
      }									\
    else								\
      {									\
	/* should have no common factors */				\
	mpz_t  g;							\
	mpz_init (g);							\
	mpz_gcd (g, mpq_numref(q), mpq_denref(q));			\
	ASSERT_ALWAYS (mpz_cmp_ui (g, 1) == 0);				\
	mpz_clear (g);							\
      }									\
  } while (0)

#define MPF_CHECK_FORMAT(f)						\
  do {									\
    ASSERT_ALWAYS (PREC(f) >= __GMPF_BITS_TO_PREC(53));			\
    ASSERT_ALWAYS (ABSIZ(f) <= PREC(f)+1);				\
    if (SIZ(f) == 0)							\
      ASSERT_ALWAYS (EXP(f) == 0);					\
    if (SIZ(f) != 0)							\
      ASSERT_ALWAYS (PTR(f)[ABSIZ(f) - 1] != 0);			\
  } while (0)


/* Enhancement: The "mod" and "gcd_1" functions below could have
   __GMP_ATTRIBUTE_PURE, but currently (gcc 3.3) that's not supported on
   function pointers, only actual functions.  It probably doesn't make much
   difference to the gmp code, since hopefully we arrange calls so there's
   no great need for the compiler to move things around.  */

#if WANT_FAT_BINARY && (HAVE_HOST_CPU_FAMILY_x86 || HAVE_HOST_CPU_FAMILY_x86_64)
/* NOTE: The function pointers in this struct are also in CPUVEC_FUNCS_LIST
   in mpn/x86/x86-defs.m4 and mpn/x86_64/x86_64-defs.m4.  Be sure to update
   those when changing here.  */
struct cpuvec_t {
  DECL_add_n           ((*add_n));
  DECL_addlsh1_n       ((*addlsh1_n));
  DECL_addlsh2_n       ((*addlsh2_n));
  DECL_addmul_1        ((*addmul_1));
  DECL_addmul_2        ((*addmul_2));
  DECL_bdiv_dbm1c      ((*bdiv_dbm1c));
  DECL_cnd_add_n       ((*cnd_add_n));
  DECL_cnd_sub_n       ((*cnd_sub_n));
  DECL_com             ((*com));
  DECL_copyd           ((*copyd));
  DECL_copyi           ((*copyi));
  DECL_divexact_1      ((*divexact_1));
  DECL_divrem_1        ((*divrem_1));
  DECL_gcd_1           ((*gcd_1));
  DECL_lshift          ((*lshift));
  DECL_lshiftc         ((*lshiftc));
  DECL_mod_1           ((*mod_1));
  DECL_mod_1_1p        ((*mod_1_1p));
  DECL_mod_1_1p_cps    ((*mod_1_1p_cps));
  DECL_mod_1s_2p       ((*mod_1s_2p));
  DECL_mod_1s_2p_cps   ((*mod_1s_2p_cps));
  DECL_mod_1s_4p       ((*mod_1s_4p));
  DECL_mod_1s_4p_cps   ((*mod_1s_4p_cps));
  DECL_mod_34lsub1     ((*mod_34lsub1));
  DECL_modexact_1c_odd ((*modexact_1c_odd));
  DECL_mul_1           ((*mul_1));
  DECL_mul_basecase    ((*mul_basecase));
  DECL_mullo_basecase  ((*mullo_basecase));
  DECL_preinv_divrem_1 ((*preinv_divrem_1));
  DECL_preinv_mod_1    ((*preinv_mod_1));
  DECL_redc_1          ((*redc_1));
  DECL_redc_2          ((*redc_2));
  DECL_rshift          ((*rshift));
  DECL_sqr_basecase    ((*sqr_basecase));
  DECL_sub_n           ((*sub_n));
  DECL_sublsh1_n       ((*sublsh1_n));
  DECL_submul_1        ((*submul_1));
  mp_size_t            mul_toom22_threshold;
  mp_size_t            mul_toom33_threshold;
  mp_size_t            sqr_toom2_threshold;
  mp_size_t            sqr_toom3_threshold;
  mp_size_t            bmod_1_to_mod_1_threshold;
};
__GMP_DECLSPEC extern struct cpuvec_t __gmpn_cpuvec;
__GMP_DECLSPEC extern int __gmpn_cpuvec_initialized;
#endif /* x86 fat binary */

__GMP_DECLSPEC void __gmpn_cpuvec_init (void);

/* Get a threshold "field" from __gmpn_cpuvec, running __gmpn_cpuvec_init()
   if that hasn't yet been done (to establish the right values).  */
#define CPUVEC_THRESHOLD(field)						      \
  ((LIKELY (__gmpn_cpuvec_initialized) ? 0 : (__gmpn_cpuvec_init (), 0)),     \
   __gmpn_cpuvec.field)


#if HAVE_NATIVE_mpn_add_nc
#define mpn_add_nc __MPN(add_nc)
__GMP_DECLSPEC mp_limb_t mpn_add_nc (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);
#else
static inline
mp_limb_t
mpn_add_nc (mp_ptr rp, mp_srcptr up, mp_srcptr vp, mp_size_t n, mp_limb_t ci)
{
  mp_limb_t co;
  co = mpn_add_n (rp, up, vp, n);
  co += mpn_add_1 (rp, rp, n, ci);
  return co;
}
#endif

#if HAVE_NATIVE_mpn_sub_nc
#define mpn_sub_nc __MPN(sub_nc)
__GMP_DECLSPEC mp_limb_t mpn_sub_nc (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);
#else
static inline mp_limb_t
mpn_sub_nc (mp_ptr rp, mp_srcptr up, mp_srcptr vp, mp_size_t n, mp_limb_t ci)
{
  mp_limb_t co;
  co = mpn_sub_n (rp, up, vp, n);
  co += mpn_sub_1 (rp, rp, n, ci);
  return co;
}
#endif

#if TUNE_PROGRAM_BUILD
/* Some extras wanted when recompiling some .c files for use by the tune
   program.  Not part of a normal build.

   It's necessary to keep these thresholds as #defines (just to an
   identically named variable), since various defaults are established based
   on #ifdef in the .c files.  For some this is not so (the defaults are
   instead established above), but all are done this way for consistency. */

#undef	MUL_TOOM22_THRESHOLD
#define MUL_TOOM22_THRESHOLD		mul_toom22_threshold
extern mp_size_t			mul_toom22_threshold;

#undef	MUL_TOOM33_THRESHOLD
#define MUL_TOOM33_THRESHOLD		mul_toom33_threshold
extern mp_size_t			mul_toom33_threshold;

#undef	MUL_TOOM44_THRESHOLD
#define MUL_TOOM44_THRESHOLD		mul_toom44_threshold
extern mp_size_t			mul_toom44_threshold;

#undef	MUL_TOOM6H_THRESHOLD
#define MUL_TOOM6H_THRESHOLD		mul_toom6h_threshold
extern mp_size_t			mul_toom6h_threshold;

#undef	MUL_TOOM8H_THRESHOLD
#define MUL_TOOM8H_THRESHOLD		mul_toom8h_threshold
extern mp_size_t			mul_toom8h_threshold;

#undef	MUL_TOOM32_TO_TOOM43_THRESHOLD
#define MUL_TOOM32_TO_TOOM43_THRESHOLD	mul_toom32_to_toom43_threshold
extern mp_size_t			mul_toom32_to_toom43_threshold;

#undef	MUL_TOOM32_TO_TOOM53_THRESHOLD
#define MUL_TOOM32_TO_TOOM53_THRESHOLD	mul_toom32_to_toom53_threshold
extern mp_size_t			mul_toom32_to_toom53_threshold;

#undef	MUL_TOOM42_TO_TOOM53_THRESHOLD
#define MUL_TOOM42_TO_TOOM53_THRESHOLD	mul_toom42_to_toom53_threshold
extern mp_size_t			mul_toom42_to_toom53_threshold;

#undef	MUL_TOOM42_TO_TOOM63_THRESHOLD
#define MUL_TOOM42_TO_TOOM63_THRESHOLD	mul_toom42_to_toom63_threshold
extern mp_size_t			mul_toom42_to_toom63_threshold;

#undef  MUL_TOOM43_TO_TOOM54_THRESHOLD
#define MUL_TOOM43_TO_TOOM54_THRESHOLD	mul_toom43_to_toom54_threshold;
extern mp_size_t			mul_toom43_to_toom54_threshold;

#undef	MUL_FFT_THRESHOLD
#define MUL_FFT_THRESHOLD		mul_fft_threshold
extern mp_size_t			mul_fft_threshold;

#undef	MUL_FFT_MODF_THRESHOLD
#define MUL_FFT_MODF_THRESHOLD		mul_fft_modf_threshold
extern mp_size_t			mul_fft_modf_threshold;

#undef	MUL_FFT_TABLE
#define MUL_FFT_TABLE			{ 0 }

#undef	MUL_FFT_TABLE3
#define MUL_FFT_TABLE3			{ {0,0} }

/* A native mpn_sqr_basecase is not tuned and SQR_BASECASE_THRESHOLD should
   remain as zero (always use it). */
#if ! HAVE_NATIVE_mpn_sqr_basecase
#undef	SQR_BASECASE_THRESHOLD
#define SQR_BASECASE_THRESHOLD		sqr_basecase_threshold
extern mp_size_t			sqr_basecase_threshold;
#endif

#if TUNE_PROGRAM_BUILD_SQR
#undef	SQR_TOOM2_THRESHOLD
#define SQR_TOOM2_THRESHOLD		SQR_TOOM2_MAX_GENERIC
#else
#undef	SQR_TOOM2_THRESHOLD
#define SQR_TOOM2_THRESHOLD		sqr_toom2_threshold
extern mp_size_t			sqr_toom2_threshold;
#endif

#undef	SQR_TOOM3_THRESHOLD
#define SQR_TOOM3_THRESHOLD		sqr_toom3_threshold
extern mp_size_t			sqr_toom3_threshold;

#undef	SQR_TOOM4_THRESHOLD
#define SQR_TOOM4_THRESHOLD		sqr_toom4_threshold
extern mp_size_t			sqr_toom4_threshold;

#undef	SQR_TOOM6_THRESHOLD
#define SQR_TOOM6_THRESHOLD		sqr_toom6_threshold
extern mp_size_t			sqr_toom6_threshold;

#undef	SQR_TOOM8_THRESHOLD
#define SQR_TOOM8_THRESHOLD		sqr_toom8_threshold
extern mp_size_t			sqr_toom8_threshold;

#undef  SQR_FFT_THRESHOLD
#define SQR_FFT_THRESHOLD		sqr_fft_threshold
extern mp_size_t			sqr_fft_threshold;

#undef  SQR_FFT_MODF_THRESHOLD
#define SQR_FFT_MODF_THRESHOLD		sqr_fft_modf_threshold
extern mp_size_t			sqr_fft_modf_threshold;

#undef	SQR_FFT_TABLE
#define SQR_FFT_TABLE			{ 0 }

#undef	SQR_FFT_TABLE3
#define SQR_FFT_TABLE3			{ {0,0} }

#undef	MULLO_BASECASE_THRESHOLD
#define MULLO_BASECASE_THRESHOLD	mullo_basecase_threshold
extern mp_size_t			mullo_basecase_threshold;

#undef	MULLO_DC_THRESHOLD
#define MULLO_DC_THRESHOLD		mullo_dc_threshold
extern mp_size_t			mullo_dc_threshold;

#undef	MULLO_MUL_N_THRESHOLD
#define MULLO_MUL_N_THRESHOLD		mullo_mul_n_threshold
extern mp_size_t			mullo_mul_n_threshold;

#undef	SQRLO_BASECASE_THRESHOLD
#define SQRLO_BASECASE_THRESHOLD	sqrlo_basecase_threshold
extern mp_size_t			sqrlo_basecase_threshold;

#undef	SQRLO_DC_THRESHOLD
#define SQRLO_DC_THRESHOLD		sqrlo_dc_threshold
extern mp_size_t			sqrlo_dc_threshold;

#undef	SQRLO_SQR_THRESHOLD
#define SQRLO_SQR_THRESHOLD		sqrlo_sqr_threshold
extern mp_size_t			sqrlo_sqr_threshold;

#undef	MULMID_TOOM42_THRESHOLD
#define MULMID_TOOM42_THRESHOLD		mulmid_toom42_threshold
extern mp_size_t			mulmid_toom42_threshold;

#undef	DIV_QR_2_PI2_THRESHOLD
#define DIV_QR_2_PI2_THRESHOLD		div_qr_2_pi2_threshold
extern mp_size_t			div_qr_2_pi2_threshold;

#undef	DC_DIV_QR_THRESHOLD
#define DC_DIV_QR_THRESHOLD		dc_div_qr_threshold
extern mp_size_t			dc_div_qr_threshold;

#undef	DC_DIVAPPR_Q_THRESHOLD
#define DC_DIVAPPR_Q_THRESHOLD		dc_divappr_q_threshold
extern mp_size_t			dc_divappr_q_threshold;

#undef	DC_BDIV_Q_THRESHOLD
#define DC_BDIV_Q_THRESHOLD		dc_bdiv_q_threshold
extern mp_size_t			dc_bdiv_q_threshold;

#undef	DC_BDIV_QR_THRESHOLD
#define DC_BDIV_QR_THRESHOLD		dc_bdiv_qr_threshold
extern mp_size_t			dc_bdiv_qr_threshold;

#undef	MU_DIV_QR_THRESHOLD
#define MU_DIV_QR_THRESHOLD		mu_div_qr_threshold
extern mp_size_t			mu_div_qr_threshold;

#undef	MU_DIVAPPR_Q_THRESHOLD
#define MU_DIVAPPR_Q_THRESHOLD		mu_divappr_q_threshold
extern mp_size_t			mu_divappr_q_threshold;

#undef	MUPI_DIV_QR_THRESHOLD
#define MUPI_DIV_QR_THRESHOLD		mupi_div_qr_threshold
extern mp_size_t			mupi_div_qr_threshold;

#undef	MU_BDIV_QR_THRESHOLD
#define MU_BDIV_QR_THRESHOLD		mu_bdiv_qr_threshold
extern mp_size_t			mu_bdiv_qr_threshold;

#undef	MU_BDIV_Q_THRESHOLD
#define MU_BDIV_Q_THRESHOLD		mu_bdiv_q_threshold
extern mp_size_t			mu_bdiv_q_threshold;

#undef	INV_MULMOD_BNM1_THRESHOLD
#define INV_MULMOD_BNM1_THRESHOLD	inv_mulmod_bnm1_threshold
extern mp_size_t			inv_mulmod_bnm1_threshold;

#undef	INV_NEWTON_THRESHOLD
#define INV_NEWTON_THRESHOLD		inv_newton_threshold
extern mp_size_t			inv_newton_threshold;

#undef	INV_APPR_THRESHOLD
#define INV_APPR_THRESHOLD		inv_appr_threshold
extern mp_size_t			inv_appr_threshold;

#undef	BINV_NEWTON_THRESHOLD
#define BINV_NEWTON_THRESHOLD		binv_newton_threshold
extern mp_size_t			binv_newton_threshold;

#undef	REDC_1_TO_REDC_2_THRESHOLD
#define REDC_1_TO_REDC_2_THRESHOLD	redc_1_to_redc_2_threshold
extern mp_size_t			redc_1_to_redc_2_threshold;

#undef	REDC_2_TO_REDC_N_THRESHOLD
#define REDC_2_TO_REDC_N_THRESHOLD	redc_2_to_redc_n_threshold
extern mp_size_t			redc_2_to_redc_n_threshold;

#undef	REDC_1_TO_REDC_N_THRESHOLD
#define REDC_1_TO_REDC_N_THRESHOLD	redc_1_to_redc_n_threshold
extern mp_size_t			redc_1_to_redc_n_threshold;

#undef	MATRIX22_STRASSEN_THRESHOLD
#define MATRIX22_STRASSEN_THRESHOLD	matrix22_strassen_threshold
extern mp_size_t			matrix22_strassen_threshold;

#undef	HGCD_THRESHOLD
#define HGCD_THRESHOLD			hgcd_threshold
extern mp_size_t			hgcd_threshold;

#undef	HGCD_APPR_THRESHOLD
#define HGCD_APPR_THRESHOLD		hgcd_appr_threshold
extern mp_size_t			hgcd_appr_threshold;

#undef	HGCD_REDUCE_THRESHOLD
#define HGCD_REDUCE_THRESHOLD		hgcd_reduce_threshold
extern mp_size_t			hgcd_reduce_threshold;

#undef	GCD_DC_THRESHOLD
#define GCD_DC_THRESHOLD		gcd_dc_threshold
extern mp_size_t			gcd_dc_threshold;

#undef  GCDEXT_DC_THRESHOLD
#define GCDEXT_DC_THRESHOLD		gcdext_dc_threshold
extern mp_size_t			gcdext_dc_threshold;

#undef  DIV_QR_1N_PI1_METHOD
#define DIV_QR_1N_PI1_METHOD		div_qr_1n_pi1_method
extern int				div_qr_1n_pi1_method;

#undef  DIV_QR_1_NORM_THRESHOLD
#define DIV_QR_1_NORM_THRESHOLD		div_qr_1_norm_threshold
extern mp_size_t			div_qr_1_norm_threshold;

#undef  DIV_QR_1_UNNORM_THRESHOLD
#define DIV_QR_1_UNNORM_THRESHOLD	div_qr_1_unnorm_threshold
extern mp_size_t			div_qr_1_unnorm_threshold;

#undef  DIVREM_1_NORM_THRESHOLD
#define DIVREM_1_NORM_THRESHOLD		divrem_1_norm_threshold
extern mp_size_t			divrem_1_norm_threshold;

#undef  DIVREM_1_UNNORM_THRESHOLD
#define DIVREM_1_UNNORM_THRESHOLD	divrem_1_unnorm_threshold
extern mp_size_t			divrem_1_unnorm_threshold;

#undef	MOD_1_NORM_THRESHOLD
#define MOD_1_NORM_THRESHOLD		mod_1_norm_threshold
extern mp_size_t			mod_1_norm_threshold;

#undef	MOD_1_UNNORM_THRESHOLD
#define MOD_1_UNNORM_THRESHOLD		mod_1_unnorm_threshold
extern mp_size_t			mod_1_unnorm_threshold;

#undef  MOD_1_1P_METHOD
#define MOD_1_1P_METHOD			mod_1_1p_method
extern int				mod_1_1p_method;

#undef	MOD_1N_TO_MOD_1_1_THRESHOLD
#define MOD_1N_TO_MOD_1_1_THRESHOLD	mod_1n_to_mod_1_1_threshold
extern mp_size_t			mod_1n_to_mod_1_1_threshold;

#undef	MOD_1U_TO_MOD_1_1_THRESHOLD
#define MOD_1U_TO_MOD_1_1_THRESHOLD	mod_1u_to_mod_1_1_threshold
extern mp_size_t			mod_1u_to_mod_1_1_threshold;

#undef	MOD_1_1_TO_MOD_1_2_THRESHOLD
#define MOD_1_1_TO_MOD_1_2_THRESHOLD	mod_1_1_to_mod_1_2_threshold
extern mp_size_t			mod_1_1_to_mod_1_2_threshold;

#undef	MOD_1_2_TO_MOD_1_4_THRESHOLD
#define MOD_1_2_TO_MOD_1_4_THRESHOLD	mod_1_2_to_mod_1_4_threshold
extern mp_size_t			mod_1_2_to_mod_1_4_threshold;

#undef	PREINV_MOD_1_TO_MOD_1_THRESHOLD
#define PREINV_MOD_1_TO_MOD_1_THRESHOLD	preinv_mod_1_to_mod_1_threshold
extern mp_size_t			preinv_mod_1_to_mod_1_threshold;

#if ! UDIV_PREINV_ALWAYS
#undef	DIVREM_2_THRESHOLD
#define DIVREM_2_THRESHOLD		divrem_2_threshold
extern mp_size_t			divrem_2_threshold;
#endif

#undef	MULMOD_BNM1_THRESHOLD
#define MULMOD_BNM1_THRESHOLD		mulmod_bnm1_threshold
extern mp_size_t			mulmod_bnm1_threshold;

#undef	SQRMOD_BNM1_THRESHOLD
#define SQRMOD_BNM1_THRESHOLD		sqrmod_bnm1_threshold
extern mp_size_t			sqrmod_bnm1_threshold;

#undef	GET_STR_DC_THRESHOLD
#define GET_STR_DC_THRESHOLD		get_str_dc_threshold
extern mp_size_t			get_str_dc_threshold;

#undef  GET_STR_PRECOMPUTE_THRESHOLD
#define GET_STR_PRECOMPUTE_THRESHOLD	get_str_precompute_threshold
extern mp_size_t			get_str_precompute_threshold;

#undef	SET_STR_DC_THRESHOLD
#define SET_STR_DC_THRESHOLD		set_str_dc_threshold
extern mp_size_t			set_str_dc_threshold;

#undef  SET_STR_PRECOMPUTE_THRESHOLD
#define SET_STR_PRECOMPUTE_THRESHOLD	set_str_precompute_threshold
extern mp_size_t			set_str_precompute_threshold;

#undef  FAC_ODD_THRESHOLD
#define FAC_ODD_THRESHOLD		fac_odd_threshold
extern  mp_size_t			fac_odd_threshold;

#undef  FAC_DSC_THRESHOLD
#define FAC_DSC_THRESHOLD		fac_dsc_threshold
extern  mp_size_t			fac_dsc_threshold;

#undef  FFT_TABLE_ATTRS
#define FFT_TABLE_ATTRS
extern mp_size_t  mpn_fft_table[2][MPN_FFT_TABLE_SIZE];
#define FFT_TABLE3_SIZE 2000	/* generous space for tuning */
extern struct fft_table_nk mpn_fft_table3[2][FFT_TABLE3_SIZE];

/* Sizes the tune program tests up to, used in a couple of recompilations. */
#undef MUL_TOOM22_THRESHOLD_LIMIT
#undef MUL_TOOM33_THRESHOLD_LIMIT
#undef MULLO_BASECASE_THRESHOLD_LIMIT
#undef SQRLO_BASECASE_THRESHOLD_LIMIT
#undef SQRLO_DC_THRESHOLD_LIMIT
#undef SQR_TOOM3_THRESHOLD_LIMIT
#define SQR_TOOM2_MAX_GENERIC           200
#define MUL_TOOM22_THRESHOLD_LIMIT      700
#define MUL_TOOM33_THRESHOLD_LIMIT      700
#define SQR_TOOM3_THRESHOLD_LIMIT       400
#define MUL_TOOM44_THRESHOLD_LIMIT     1000
#define SQR_TOOM4_THRESHOLD_LIMIT      1000
#define MUL_TOOM6H_THRESHOLD_LIMIT     1100
#define SQR_TOOM6_THRESHOLD_LIMIT      1100
#define MUL_TOOM8H_THRESHOLD_LIMIT     1200
#define SQR_TOOM8_THRESHOLD_LIMIT      1200
#define MULLO_BASECASE_THRESHOLD_LIMIT  200
#define SQRLO_BASECASE_THRESHOLD_LIMIT  200
#define SQRLO_DC_THRESHOLD_LIMIT        400
#define GET_STR_THRESHOLD_LIMIT         150
#define FAC_DSC_THRESHOLD_LIMIT        2048

#endif /* TUNE_PROGRAM_BUILD */

#if defined (__cplusplus)
}
#endif

/* FIXME: Make these itch functions less conservative.  Also consider making
   them dependent on just 'an', and compute the allocation directly from 'an'
   instead of via n.  */

/* toom22/toom2: Scratch need is 2*(an + k), k is the recursion depth.
   k is ths smallest k such that
     ceil(an/2^k) < MUL_TOOM22_THRESHOLD.
   which implies that
     k = bitsize of floor ((an-1)/(MUL_TOOM22_THRESHOLD-1))
       = 1 + floor (log_2 (floor ((an-1)/(MUL_TOOM22_THRESHOLD-1))))
*/
#define mpn_toom22_mul_itch(an, bn) \
  (2 * ((an) + GMP_NUMB_BITS))
#define mpn_toom2_sqr_itch(an) \
  (2 * ((an) + GMP_NUMB_BITS))

/* toom33/toom3: Scratch need is 5an/2 + 10k, k is the recursion depth.
   We use 3an + C, so that we can use a smaller constant.
 */
#define mpn_toom33_mul_itch(an, bn) \
  (3 * (an) + GMP_NUMB_BITS)
#define mpn_toom3_sqr_itch(an) \
  (3 * (an) + GMP_NUMB_BITS)

/* toom33/toom3: Scratch need is 8an/3 + 13k, k is the recursion depth.
   We use 3an + C, so that we can use a smaller constant.
 */
#define mpn_toom44_mul_itch(an, bn) \
  (3 * (an) + GMP_NUMB_BITS)
#define mpn_toom4_sqr_itch(an) \
  (3 * (an) + GMP_NUMB_BITS)

#define mpn_toom6_sqr_itch(n)						\
  (((n) - SQR_TOOM6_THRESHOLD)*2 +					\
   MAX(SQR_TOOM6_THRESHOLD*2 + GMP_NUMB_BITS*6,				\
       mpn_toom4_sqr_itch(SQR_TOOM6_THRESHOLD)))

#define MUL_TOOM6H_MIN							\
  ((MUL_TOOM6H_THRESHOLD > MUL_TOOM44_THRESHOLD) ?			\
    MUL_TOOM6H_THRESHOLD : MUL_TOOM44_THRESHOLD)
#define mpn_toom6_mul_n_itch(n)						\
  (((n) - MUL_TOOM6H_MIN)*2 +						\
   MAX(MUL_TOOM6H_MIN*2 + GMP_NUMB_BITS*6,				\
       mpn_toom44_mul_itch(MUL_TOOM6H_MIN,MUL_TOOM6H_MIN)))

static inline mp_size_t
mpn_toom6h_mul_itch (mp_size_t an, mp_size_t bn) {
  mp_size_t estimatedN;
  estimatedN = (an + bn) / (size_t) 10 + 1;
  return mpn_toom6_mul_n_itch (estimatedN * 6);
}

#define mpn_toom8_sqr_itch(n)						\
  ((((n)*15)>>3) - ((SQR_TOOM8_THRESHOLD*15)>>3) +			\
   MAX(((SQR_TOOM8_THRESHOLD*15)>>3) + GMP_NUMB_BITS*6,			\
       mpn_toom6_sqr_itch(SQR_TOOM8_THRESHOLD)))

#define MUL_TOOM8H_MIN							\
  ((MUL_TOOM8H_THRESHOLD > MUL_TOOM6H_MIN) ?				\
    MUL_TOOM8H_THRESHOLD : MUL_TOOM6H_MIN)
#define mpn_toom8_mul_n_itch(n)						\
  ((((n)*15)>>3) - ((MUL_TOOM8H_MIN*15)>>3) +				\
   MAX(((MUL_TOOM8H_MIN*15)>>3) + GMP_NUMB_BITS*6,			\
       mpn_toom6_mul_n_itch(MUL_TOOM8H_MIN)))

static inline mp_size_t
mpn_toom8h_mul_itch (mp_size_t an, mp_size_t bn) {
  mp_size_t estimatedN;
  estimatedN = (an + bn) / (size_t) 14 + 1;
  return mpn_toom8_mul_n_itch (estimatedN * 8);
}

static inline mp_size_t
mpn_toom32_mul_itch (mp_size_t an, mp_size_t bn)
{
  mp_size_t n = 1 + (2 * an >= 3 * bn ? (an - 1) / (size_t) 3 : (bn - 1) >> 1);
  mp_size_t itch = 2 * n + 1;

  return itch;
}

static inline mp_size_t
mpn_toom42_mul_itch (mp_size_t an, mp_size_t bn)
{
  mp_size_t n = an >= 2 * bn ? (an + 3) >> 2 : (bn + 1) >> 1;
  return 6 * n + 3;
}

static inline mp_size_t
mpn_toom43_mul_itch (mp_size_t an, mp_size_t bn)
{
  mp_size_t n = 1 + (3 * an >= 4 * bn ? (an - 1) >> 2 : (bn - 1) / (size_t) 3);

  return 6*n + 4;
}

static inline mp_size_t
mpn_toom52_mul_itch (mp_size_t an, mp_size_t bn)
{
  mp_size_t n = 1 + (2 * an >= 5 * bn ? (an - 1) / (size_t) 5 : (bn - 1) >> 1);
  return 6*n + 4;
}

static inline mp_size_t
mpn_toom53_mul_itch (mp_size_t an, mp_size_t bn)
{
  mp_size_t n = 1 + (3 * an >= 5 * bn ? (an - 1) / (size_t) 5 : (bn - 1) / (size_t) 3);
  return 10 * n + 10;
}

static inline mp_size_t
mpn_toom62_mul_itch (mp_size_t an, mp_size_t bn)
{
  mp_size_t n = 1 + (an >= 3 * bn ? (an - 1) / (size_t) 6 : (bn - 1) >> 1);
  return 10 * n + 10;
}

static inline mp_size_t
mpn_toom63_mul_itch (mp_size_t an, mp_size_t bn)
{
  mp_size_t n = 1 + (an >= 2 * bn ? (an - 1) / (size_t) 6 : (bn - 1) / (size_t) 3);
  return 9 * n + 3;
}

static inline mp_size_t
mpn_toom54_mul_itch (mp_size_t an, mp_size_t bn)
{
  mp_size_t n = 1 + (4 * an >= 5 * bn ? (an - 1) / (size_t) 5 : (bn - 1) / (size_t) 4);
  return 9 * n + 3;
}

/* let S(n) = space required for input size n,
   then S(n) = 3 floor(n/2) + 1 + S(floor(n/2)).   */
#define mpn_toom42_mulmid_itch(n) \
  (3 * (n) + GMP_NUMB_BITS)

#if 0
#define mpn_fft_mul mpn_mul_fft_full
#else
#define mpn_fft_mul mpn_nussbaumer_mul
#endif

#ifdef __cplusplus

/* A little helper for a null-terminated __gmp_allocate_func string.
   The destructor ensures it's freed even if an exception is thrown.
   The len field is needed by the destructor, and can be used by anyone else
   to avoid a second strlen pass over the data.

   Since our input is a C string, using strlen is correct.  Perhaps it'd be
   more C++-ish style to use std::char_traits<char>::length, but char_traits
   isn't available in gcc 2.95.4.  */

class gmp_allocated_string {
 public:
  char *str;
  size_t len;
  gmp_allocated_string(char *arg)
  {
    str = arg;
    len = std::strlen (str);
  }
  ~gmp_allocated_string()
  {
    (*__gmp_free_func) (str, len+1);
  }
};

std::istream &__gmpz_operator_in_nowhite (std::istream &, mpz_ptr, char);
int __gmp_istream_set_base (std::istream &, char &, bool &, bool &);
void __gmp_istream_set_digits (std::string &, std::istream &, char &, bool &, int);
void __gmp_doprnt_params_from_ios (struct doprnt_params_t *, std::ios &);
std::ostream& __gmp_doprnt_integer_ostream (std::ostream &, struct doprnt_params_t *, char *);
extern const struct doprnt_funs_t  __gmp_asprintf_funs_noformat;

#endif /* __cplusplus */

#endif /* __GMP_IMPL_H__ */
