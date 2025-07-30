/* mpc.h -- Include file for mpc.

Copyright (C) 2002, 2003, 2004, 2005, 2007, 2008, 2009, 2010, 2011, 2012, 2014, 2015 INRIA

This file is part of GNU MPC.

GNU MPC is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the
Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

GNU MPC is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
more details.

You should have received a copy of the GNU Lesser General Public License
along with this program. If not, see http://www.gnu.org/licenses/ .
*/

#ifndef __MPC_H
#define __MPC_H

#include "gmp.h"
#include "mpfr.h"

/* Backwards compatibility with mpfr<3.0.0 */
#ifndef mpfr_exp_t
#define mpfr_exp_t mp_exp_t
#endif

/* Define MPC version number */
#define MPC_VERSION_MAJOR 1
#define MPC_VERSION_MINOR 0
#define MPC_VERSION_PATCHLEVEL 3
#define MPC_VERSION_STRING "1.0.3"

/* Macros dealing with MPC VERSION */
#define MPC_VERSION_NUM(a,b,c) (((a) << 16L) | ((b) << 8) | (c))
#define MPC_VERSION                                                     \
  MPC_VERSION_NUM(MPC_VERSION_MAJOR,MPC_VERSION_MINOR,MPC_VERSION_PATCHLEVEL)

/* Check if stdint.h/inttypes.h is included */
#if defined (INTMAX_C) && defined (UINTMAX_C)
#define _MPC_H_HAVE_INTMAX_T 1
#endif

/* Return values */

/* Transform negative to 2, positive to 1, leave 0 unchanged */
#define MPC_INEX_POS(inex) (((inex) < 0) ? 2 : ((inex) == 0) ? 0 : 1)
/* Transform 2 to negative, 1 to positive, leave 0 unchanged */
#define MPC_INEX_NEG(inex) (((inex) == 2) ? -1 : ((inex) == 0) ? 0 : 1)

/* The global inexact flag is made of (real flag) + 4 * (imaginary flag), where
   each of the real and imaginary inexact flag are:
   0 when the result is exact (no rounding error)
   1 when the result is larger than the exact value
   2 when the result is smaller than the exact value */
#define MPC_INEX(inex_re, inex_im) \
        (MPC_INEX_POS(inex_re) | (MPC_INEX_POS(inex_im) << 2))
#define MPC_INEX_RE(inex) MPC_INEX_NEG((inex) & 3)
#define MPC_INEX_IM(inex) MPC_INEX_NEG((inex) >> 2)

/* For functions computing two results, the return value is
   inexact1+16*inexact2, which is 0 iif both results are exact. */
#define MPC_INEX12(inex1, inex2) (inex1 | (inex2 << 4))
#define MPC_INEX1(inex) (inex & 15)
#define MPC_INEX2(inex) (inex >> 4)

/* Definition of rounding modes */

/* a complex rounding mode is just a pair of two real rounding modes
   we reserve four bits for a real rounding mode.  */
typedef int mpc_rnd_t;

#define MPC_RND(r1,r2) (((int)(r1)) + ((int)(r2) << 4))
#define MPC_RND_RE(x) ((mpfr_rnd_t)((x) & 0x0F))
#define MPC_RND_IM(x) ((mpfr_rnd_t)((x) >> 4))

#define MPC_RNDNN MPC_RND (GMP_RNDN,GMP_RNDN)
#define MPC_RNDNZ MPC_RND (GMP_RNDN,GMP_RNDZ)
#define MPC_RNDNU MPC_RND (GMP_RNDN,GMP_RNDU)
#define MPC_RNDND MPC_RND (GMP_RNDN,GMP_RNDD)

#define MPC_RNDZN MPC_RND (GMP_RNDZ,GMP_RNDN)
#define MPC_RNDZZ MPC_RND (GMP_RNDZ,GMP_RNDZ)
#define MPC_RNDZU MPC_RND (GMP_RNDZ,GMP_RNDU)
#define MPC_RNDZD MPC_RND (GMP_RNDZ,GMP_RNDD)

#define MPC_RNDUN MPC_RND (GMP_RNDU,GMP_RNDN)
#define MPC_RNDUZ MPC_RND (GMP_RNDU,GMP_RNDZ)
#define MPC_RNDUU MPC_RND (GMP_RNDU,GMP_RNDU)
#define MPC_RNDUD MPC_RND (GMP_RNDU,GMP_RNDD)

#define MPC_RNDDN MPC_RND (GMP_RNDD,GMP_RNDN)
#define MPC_RNDDZ MPC_RND (GMP_RNDD,GMP_RNDZ)
#define MPC_RNDDU MPC_RND (GMP_RNDD,GMP_RNDU)
#define MPC_RNDDD MPC_RND (GMP_RNDD,GMP_RNDD)


/* Definitions of types and their semantics */

typedef struct {
  mpfr_t re;
  mpfr_t im;
}
__mpc_struct;

typedef __mpc_struct mpc_t[1];
typedef __mpc_struct *mpc_ptr;
typedef const __mpc_struct *mpc_srcptr;

/* Support for WINDOWS DLL, see
   http://lists.gforge.inria.fr/pipermail/mpc-discuss/2011-November/000990.html;
   when building the DLL, export symbols, otherwise behave as GMP           */
#if defined (__MPC_LIBRARY_BUILD) && __GMP_LIBGMP_DLL
#define __MPC_DECLSPEC __GMP_DECLSPEC_EXPORT
#else
#define __MPC_DECLSPEC __GMP_DECLSPEC
#endif

#if defined (__cplusplus)
extern "C" {
#endif

__MPC_DECLSPEC int  mpc_add       (mpc_ptr, mpc_srcptr, mpc_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_add_fr    (mpc_ptr, mpc_srcptr, mpfr_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_add_si    (mpc_ptr, mpc_srcptr, long int, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_add_ui    (mpc_ptr, mpc_srcptr, unsigned long int, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_sub       (mpc_ptr, mpc_srcptr, mpc_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_sub_fr    (mpc_ptr, mpc_srcptr, mpfr_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_fr_sub    (mpc_ptr, mpfr_srcptr, mpc_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_sub_ui    (mpc_ptr, mpc_srcptr, unsigned long int, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_ui_ui_sub (mpc_ptr, unsigned long int, unsigned long int, mpc_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_mul       (mpc_ptr, mpc_srcptr, mpc_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_mul_fr    (mpc_ptr, mpc_srcptr, mpfr_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_mul_ui    (mpc_ptr, mpc_srcptr, unsigned long int, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_mul_si    (mpc_ptr, mpc_srcptr, long int, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_mul_i     (mpc_ptr, mpc_srcptr, int, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_sqr       (mpc_ptr, mpc_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_div       (mpc_ptr, mpc_srcptr, mpc_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_pow       (mpc_ptr, mpc_srcptr, mpc_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_pow_fr    (mpc_ptr, mpc_srcptr, mpfr_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_pow_ld    (mpc_ptr, mpc_srcptr, long double, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_pow_d     (mpc_ptr, mpc_srcptr, double, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_pow_si    (mpc_ptr, mpc_srcptr, long, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_pow_ui    (mpc_ptr, mpc_srcptr, unsigned long, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_pow_z     (mpc_ptr, mpc_srcptr, mpz_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_div_fr    (mpc_ptr, mpc_srcptr, mpfr_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_fr_div    (mpc_ptr, mpfr_srcptr, mpc_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_div_ui    (mpc_ptr, mpc_srcptr, unsigned long int, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_ui_div    (mpc_ptr, unsigned long int, mpc_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_div_2ui   (mpc_ptr, mpc_srcptr, unsigned long int, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_mul_2ui   (mpc_ptr, mpc_srcptr, unsigned long int, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_div_2si   (mpc_ptr, mpc_srcptr, long int, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_mul_2si   (mpc_ptr, mpc_srcptr, long int, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_conj      (mpc_ptr, mpc_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_neg       (mpc_ptr, mpc_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_norm      (mpfr_ptr, mpc_srcptr, mpfr_rnd_t);
__MPC_DECLSPEC int  mpc_abs       (mpfr_ptr, mpc_srcptr, mpfr_rnd_t);
__MPC_DECLSPEC int  mpc_sqrt      (mpc_ptr, mpc_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_set       (mpc_ptr, mpc_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_set_d     (mpc_ptr, double, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_set_d_d   (mpc_ptr, double, double, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_set_ld    (mpc_ptr, long double, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_set_ld_ld (mpc_ptr, long double, long double, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_set_f     (mpc_ptr, mpf_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_set_f_f   (mpc_ptr, mpf_srcptr, mpf_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_set_fr    (mpc_ptr, mpfr_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_set_fr_fr (mpc_ptr, mpfr_srcptr, mpfr_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_set_q     (mpc_ptr, mpq_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_set_q_q   (mpc_ptr, mpq_srcptr, mpq_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_set_si    (mpc_ptr, long int, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_set_si_si (mpc_ptr, long int, long int, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_set_ui    (mpc_ptr, unsigned long int, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_set_ui_ui (mpc_ptr, unsigned long int, unsigned long int, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_set_z     (mpc_ptr, mpz_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_set_z_z   (mpc_ptr, mpz_srcptr, mpz_srcptr, mpc_rnd_t);
__MPC_DECLSPEC void mpc_swap      (mpc_ptr, mpc_ptr);
__MPC_DECLSPEC int  mpc_fma       (mpc_ptr, mpc_srcptr, mpc_srcptr, mpc_srcptr, mpc_rnd_t);

__MPC_DECLSPEC void mpc_set_nan   (mpc_ptr);

__MPC_DECLSPEC int  mpc_real      (mpfr_ptr, mpc_srcptr, mpfr_rnd_t);
__MPC_DECLSPEC int  mpc_imag      (mpfr_ptr, mpc_srcptr, mpfr_rnd_t);
__MPC_DECLSPEC int  mpc_arg       (mpfr_ptr, mpc_srcptr, mpfr_rnd_t);
__MPC_DECLSPEC int  mpc_proj      (mpc_ptr, mpc_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_cmp       (mpc_srcptr, mpc_srcptr);
__MPC_DECLSPEC int  mpc_cmp_si_si (mpc_srcptr, long int, long int);
__MPC_DECLSPEC int  mpc_exp       (mpc_ptr, mpc_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_log       (mpc_ptr, mpc_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_log10     (mpc_ptr, mpc_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_sin       (mpc_ptr, mpc_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_cos       (mpc_ptr, mpc_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_sin_cos   (mpc_ptr, mpc_ptr, mpc_srcptr, mpc_rnd_t, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_tan       (mpc_ptr, mpc_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_sinh      (mpc_ptr, mpc_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_cosh      (mpc_ptr, mpc_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_tanh      (mpc_ptr, mpc_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_asin      (mpc_ptr, mpc_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_acos      (mpc_ptr, mpc_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_atan      (mpc_ptr, mpc_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_asinh     (mpc_ptr, mpc_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_acosh     (mpc_ptr, mpc_srcptr, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_atanh     (mpc_ptr, mpc_srcptr, mpc_rnd_t);
__MPC_DECLSPEC void mpc_clear     (mpc_ptr);
__MPC_DECLSPEC int  mpc_urandom   (mpc_ptr, gmp_randstate_t);
__MPC_DECLSPEC void mpc_init2     (mpc_ptr, mpfr_prec_t);
__MPC_DECLSPEC void mpc_init3     (mpc_ptr, mpfr_prec_t, mpfr_prec_t);
__MPC_DECLSPEC mpfr_prec_t mpc_get_prec (mpc_srcptr x);
__MPC_DECLSPEC void mpc_get_prec2 (mpfr_prec_t *pr, mpfr_prec_t *pi, mpc_srcptr x);
__MPC_DECLSPEC void mpc_set_prec  (mpc_ptr, mpfr_prec_t);
__MPC_DECLSPEC const char * mpc_get_version (void);

__MPC_DECLSPEC int  mpc_strtoc    (mpc_ptr, const char *, char **, int, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_set_str   (mpc_ptr, const char *, int, mpc_rnd_t);
__MPC_DECLSPEC char * mpc_get_str (int, size_t, mpc_srcptr, mpc_rnd_t);
__MPC_DECLSPEC void mpc_free_str  (char *);

/* declare certain functions only if appropriate headers have been included */
#ifdef _MPC_H_HAVE_INTMAX_T
__MPC_DECLSPEC int  mpc_set_sj    (mpc_ptr, intmax_t, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_set_uj    (mpc_ptr, uintmax_t,  mpc_rnd_t);
__MPC_DECLSPEC int  mpc_set_sj_sj (mpc_ptr, intmax_t, intmax_t, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_set_uj_uj (mpc_ptr, uintmax_t, uintmax_t, mpc_rnd_t);
#endif

#ifdef _Complex_I
__MPC_DECLSPEC int  mpc_set_dc    (mpc_ptr, double _Complex, mpc_rnd_t);
__MPC_DECLSPEC int  mpc_set_ldc   (mpc_ptr, long double _Complex, mpc_rnd_t);
__MPC_DECLSPEC double _Complex mpc_get_dc (mpc_srcptr, mpc_rnd_t);
__MPC_DECLSPEC long double _Complex mpc_get_ldc (mpc_srcptr, mpc_rnd_t);
#endif

#ifdef _GMP_H_HAVE_FILE
__MPC_DECLSPEC int mpc_inp_str    (mpc_ptr, FILE *, size_t *, int, mpc_rnd_t);
__MPC_DECLSPEC size_t mpc_out_str (FILE *, int, size_t, mpc_srcptr, mpc_rnd_t);
#endif

#if defined (__cplusplus)
}
#endif

#define mpc_realref(x) ((x)->re)
#define mpc_imagref(x) ((x)->im)

#define mpc_cmp_si(x, y) \
 ( mpc_cmp_si_si ((x), (y), 0l) )
#define mpc_ui_sub(x, y, z, r) mpc_ui_ui_sub (x, y, 0ul, z, r)

/*
   Define a fake mpfr_set_fr so that, for instance, mpc_set_fr_z would
   be defined as follows:
   mpc_set_fr_z (mpc_t rop, mpfr_t x, mpz_t y, mpc_rnd_t rnd)
       MPC_SET_X_Y (fr, z, rop, x, y, rnd)
*/
#ifndef mpfr_set_fr
#define mpfr_set_fr mpfr_set
#endif
#define MPC_SET_X_Y(real_t, imag_t, z, real_value, imag_value, rnd)     \
  {                                                                     \
    int _inex_re, _inex_im;                                             \
    _inex_re = (mpfr_set_ ## real_t) (mpc_realref (z), (real_value), MPC_RND_RE (rnd)); \
    _inex_im = (mpfr_set_ ## imag_t) (mpc_imagref (z), (imag_value), MPC_RND_IM (rnd)); \
    return MPC_INEX (_inex_re, _inex_im);                               \
  }

#endif /* ifndef __MPC_H */
