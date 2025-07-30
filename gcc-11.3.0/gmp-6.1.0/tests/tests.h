/* Tests support prototypes etc.

Copyright 2000-2004, 2008-2013 Free Software Foundation, Inc.

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


#ifndef __TESTS_H__
#define __TESTS_H__

#include "config.h"

#include <setjmp.h>  /* for jmp_buf */

#if defined (__cplusplus)
extern "C" {
#endif


#ifdef __cplusplus
#define ANYARGS  ...
#else
#define ANYARGS
#endif


void tests_start (void);
void tests_end (void);

void tests_memory_start (void);
void tests_memory_end (void);
void *tests_allocate (size_t);
void *tests_reallocate (void *, size_t, size_t);
void tests_free (void *, size_t);
void tests_free_nosize (void *);
int tests_memory_valid (void *);

void tests_rand_start (void);
void tests_rand_end (void);

double tests_infinity_d ();
int tests_hardware_getround (void);
int tests_hardware_setround (int);
int tests_isinf (double);
int tests_dbl_mant_bits (void);

void x86_fldcw (unsigned short);
unsigned short x86_fstcw (void);


/* tests_setjmp_sigfpe is like a setjmp, establishing a trap for SIGFPE.
   The initial return is 0, if SIGFPE is trapped execution goes back there
   with return value 1.

   tests_sigfpe_done puts SIGFPE back to SIG_DFL, which should be used once
   the setjmp point is out of scope, so a later SIGFPE won't try to go back
   there.  */

#define tests_setjmp_sigfpe()                   \
  (signal (SIGFPE, tests_sigfpe_handler),       \
   setjmp (tests_sigfpe_target))

RETSIGTYPE tests_sigfpe_handler (int);
void tests_sigfpe_done (void);
extern jmp_buf  tests_sigfpe_target;


#if HAVE_CALLING_CONVENTIONS
extern mp_limb_t (*calling_conventions_function) (ANYARGS);
mp_limb_t calling_conventions (ANYARGS);
int calling_conventions_check (void);
#define CALLING_CONVENTIONS(function) \
  (calling_conventions_function = (function), calling_conventions)
#define CALLING_CONVENTIONS_CHECK()    (calling_conventions_check())
#else
#define CALLING_CONVENTIONS(function)  (function)
#define CALLING_CONVENTIONS_CHECK()    1 /* always ok */
#endif


extern int mp_trace_base;
void mp_limb_trace (const char *, mp_limb_t);
void mpn_trace (const char *, mp_srcptr, mp_size_t);
void mpn_tracea (const char *, const mp_ptr *, int, mp_size_t);
void mpn_tracen (const char *, int, mp_srcptr, mp_size_t);
void mpn_trace_file (const char *, mp_srcptr, mp_size_t);
void mpn_tracea_file (const char *, const mp_ptr *, int, mp_size_t);
void mpf_trace (const char *, mpf_srcptr);
void mpq_trace (const char *, mpq_srcptr);
void mpz_trace (const char *, mpz_srcptr);
void mpz_tracen (const char *, int, mpz_srcptr);
void byte_trace (const char *, const void *, mp_size_t);
void byte_tracen (const char *, int, const void *, mp_size_t);
void d_trace (const char *, double);


void spinner (void);
extern unsigned long  spinner_count;
extern int  spinner_wanted;
extern int  spinner_tick;


void *align_pointer (void *, size_t);
void *__gmp_allocate_func_aligned (size_t, size_t);
void *__gmp_allocate_or_reallocate (void *, size_t, size_t);
char *__gmp_allocate_strdup (const char *);
char *strtoupper (char *);
mp_limb_t urandom (void);
void call_rand_algs (void (*func) (const char *, gmp_randstate_t));


void mpf_set_str_or_abort (mpf_ptr, const char *, int);


void mpq_set_str_or_abort (mpq_ptr, const char *, int);


void mpz_erandomb (mpz_ptr, gmp_randstate_t, unsigned long);
void mpz_erandomb_nonzero (mpz_ptr, gmp_randstate_t, unsigned long);
void mpz_errandomb (mpz_ptr, gmp_randstate_t, unsigned long);
void mpz_errandomb_nonzero (mpz_ptr, gmp_randstate_t, unsigned long);
void mpz_init_set_n (mpz_ptr, mp_srcptr, mp_size_t);
void mpz_negrandom (mpz_ptr, gmp_randstate_t);
int mpz_pow2abs_p (mpz_srcptr) __GMP_ATTRIBUTE_PURE;
void mpz_set_n (mpz_ptr, mp_srcptr, mp_size_t);
void mpz_set_str_or_abort (mpz_ptr, const char *, int);

mp_size_t mpn_diff_highest (mp_srcptr, mp_srcptr, mp_size_t) __GMP_ATTRIBUTE_PURE;
mp_size_t mpn_diff_lowest (mp_srcptr, mp_srcptr, mp_size_t) __GMP_ATTRIBUTE_PURE;
mp_size_t byte_diff_highest (const void *, const void *, mp_size_t) __GMP_ATTRIBUTE_PURE;
mp_size_t byte_diff_lowest (const void *, const void *, mp_size_t) __GMP_ATTRIBUTE_PURE;


mp_limb_t ref_addc_limb (mp_limb_t *, mp_limb_t, mp_limb_t);
mp_limb_t ref_bswap_limb (mp_limb_t);
unsigned long ref_popc_limb (mp_limb_t);
mp_limb_t ref_subc_limb (mp_limb_t *, mp_limb_t, mp_limb_t);


void refmpf_add (mpf_ptr, mpf_srcptr, mpf_srcptr);
void refmpf_add_ulp (mpf_ptr );
void refmpf_fill (mpf_ptr, mp_size_t, mp_limb_t);
void refmpf_normalize (mpf_ptr);
void refmpf_set_prec_limbs (mpf_ptr, unsigned long);
unsigned long refmpf_set_overlap (mpf_ptr, mpf_srcptr);
void refmpf_sub (mpf_ptr, mpf_srcptr, mpf_srcptr);
int refmpf_validate (const char *, mpf_srcptr, mpf_srcptr);
int refmpf_validate_division (const char *, mpf_srcptr, mpf_srcptr, mpf_srcptr);


mp_limb_t refmpn_cnd_add_n (mp_limb_t, mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
mp_limb_t refmpn_cnd_sub_n (mp_limb_t, mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

mp_limb_t refmpn_add (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);
mp_limb_t refmpn_add_1 (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);
mp_limb_t refmpn_add_err1_n (mp_ptr, mp_srcptr, mp_srcptr, mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);
mp_limb_t refmpn_add_err2_n (mp_ptr, mp_srcptr, mp_srcptr, mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);
mp_limb_t refmpn_add_err3_n (mp_ptr, mp_srcptr, mp_srcptr, mp_ptr, mp_srcptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);
mp_limb_t refmpn_add_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
mp_limb_t refmpn_add_nc (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);
mp_limb_t refmpn_addlsh1_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
mp_limb_t refmpn_addlsh2_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
mp_limb_t refmpn_addlsh_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, unsigned);
mp_limb_t refmpn_addlsh1_n_ip1 (mp_ptr, mp_srcptr, mp_size_t);
mp_limb_t refmpn_addlsh2_n_ip1 (mp_ptr, mp_srcptr, mp_size_t);
mp_limb_t refmpn_addlsh_n_ip1 (mp_ptr, mp_srcptr, mp_size_t, unsigned);
mp_limb_t refmpn_addlsh1_n_ip2 (mp_ptr, mp_srcptr, mp_size_t);
mp_limb_t refmpn_addlsh2_n_ip2 (mp_ptr, mp_srcptr, mp_size_t);
mp_limb_t refmpn_addlsh_n_ip2 (mp_ptr, mp_srcptr, mp_size_t, unsigned);
mp_limb_t refmpn_addlsh1_nc (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);
mp_limb_t refmpn_addlsh2_nc (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);
mp_limb_t refmpn_addlsh_nc (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, unsigned, mp_limb_t);
mp_limb_t refmpn_addmul_1 (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);
mp_limb_t refmpn_addmul_1c (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t);
mp_limb_t refmpn_addmul_2 (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);
mp_limb_t refmpn_addmul_3 (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);
mp_limb_t refmpn_addmul_4 (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);
mp_limb_t refmpn_addmul_5 (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);
mp_limb_t refmpn_addmul_6 (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);
mp_limb_t refmpn_addmul_7 (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);
mp_limb_t refmpn_addmul_8 (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

mp_limb_t refmpn_add_n_sub_n (mp_ptr, mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
mp_limb_t refmpn_add_n_sub_nc (mp_ptr, mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);

void refmpn_and_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
void refmpn_andn_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

mp_limb_t refmpn_big_base (int);

int refmpn_chars_per_limb (int);
void refmpn_clrbit (mp_ptr, unsigned long);
int refmpn_cmp (mp_srcptr, mp_srcptr, mp_size_t);
int refmpn_cmp_allowzero (mp_srcptr, mp_srcptr, mp_size_t);
int refmpn_cmp_twosizes (mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);

void refmpn_com (mp_ptr, mp_srcptr, mp_size_t);
void refmpn_copy (mp_ptr, mp_srcptr, mp_size_t);
void refmpn_copyi (mp_ptr, mp_srcptr, mp_size_t);
void refmpn_copyd (mp_ptr, mp_srcptr, mp_size_t);
void refmpn_copy_extend (mp_ptr, mp_size_t, mp_srcptr, mp_size_t);

unsigned refmpn_count_leading_zeros (mp_limb_t);
unsigned refmpn_count_trailing_zeros (mp_limb_t);

mp_limb_t refmpn_divexact_by3 (mp_ptr, mp_srcptr, mp_size_t);
mp_limb_t refmpn_divexact_by3c (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);

mp_limb_t refmpn_divmod_1 (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);
mp_limb_t refmpn_divmod_1c (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t);
mp_limb_t refmpn_divrem_1 (mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t);
mp_limb_t refmpn_divrem_1c (mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t);
mp_limb_t refmpn_divrem_2 (mp_ptr, mp_size_t, mp_ptr, mp_size_t, mp_srcptr);

int refmpn_equal_anynail (mp_srcptr, mp_srcptr, mp_size_t);

void refmpn_fill (mp_ptr, mp_size_t, mp_limb_t);

mp_limb_t refmpn_gcd_1 (mp_srcptr, mp_size_t, mp_limb_t);
mp_limb_t refmpn_gcd (mp_ptr, mp_ptr, mp_size_t, mp_ptr, mp_size_t);

size_t refmpn_get_str (unsigned char *, int, mp_ptr, mp_size_t);

unsigned long refmpn_hamdist (mp_srcptr, mp_srcptr, mp_size_t);

mp_limb_t refmpn_invert_limb (mp_limb_t);
void refmpn_ior_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
void refmpn_iorn_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

mp_limb_t refmpn_lshift (mp_ptr, mp_srcptr, mp_size_t, unsigned);
mp_limb_t refmpn_lshift_or_copy (mp_ptr, mp_srcptr, mp_size_t, unsigned);
mp_limb_t refmpn_lshift_or_copy_any (mp_ptr, mp_srcptr, mp_size_t, unsigned);
mp_limb_t refmpn_lshiftc (mp_ptr, mp_srcptr, mp_size_t, unsigned);
void refmpn_com (mp_ptr, mp_srcptr, mp_size_t);

mp_ptr refmpn_malloc_limbs (mp_size_t);
mp_ptr refmpn_malloc_limbs_aligned (mp_size_t, size_t);
void refmpn_free_limbs (mp_ptr);
mp_limb_t refmpn_msbone (mp_limb_t);
mp_limb_t refmpn_msbone_mask (mp_limb_t);
mp_ptr refmpn_memdup_limbs (mp_srcptr, mp_size_t);

mp_limb_t refmpn_mod_1 (mp_srcptr, mp_size_t, mp_limb_t);
mp_limb_t refmpn_mod_1c (mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t);
mp_limb_t refmpn_mod_34lsub1 (mp_srcptr, mp_size_t);

mp_limb_t refmpn_mul_1 (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);
mp_limb_t refmpn_mul_1c (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t);
mp_limb_t refmpn_mul_2 (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);
mp_limb_t refmpn_mul_3 (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);
mp_limb_t refmpn_mul_4 (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);
mp_limb_t refmpn_mul_5 (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);
mp_limb_t refmpn_mul_6 (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);

void refmpn_mul_basecase (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);
void refmpn_mulmid_basecase (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);
void refmpn_toom42_mulmid (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_ptr);
void refmpn_mulmid_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
void refmpn_mulmid (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);
void refmpn_mullo_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
void refmpn_mul_any (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);
void refmpn_mul_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
void refmpn_mul (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);

void refmpn_nand_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
void refmpn_nior_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
mp_limb_t refmpn_neg (mp_ptr, mp_srcptr, mp_size_t);
mp_size_t refmpn_normalize (mp_srcptr, mp_size_t);

unsigned long refmpn_popcount (mp_srcptr, mp_size_t);
mp_limb_t refmpn_preinv_divrem_1 (mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t, unsigned);
mp_limb_t refmpn_preinv_mod_1 (mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t);

void refmpn_random (mp_ptr, mp_size_t);
void refmpn_random2 (mp_ptr, mp_size_t);
mp_limb_t refmpn_random_limb (void);

mp_limb_t refmpn_rsh1add_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
mp_limb_t refmpn_rsh1sub_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
mp_limb_t refmpn_rshift (mp_ptr, mp_srcptr, mp_size_t, unsigned);
mp_limb_t refmpn_rshift_or_copy (mp_ptr, mp_srcptr, mp_size_t, unsigned);
mp_limb_t refmpn_rshift_or_copy_any (mp_ptr, mp_srcptr, mp_size_t, unsigned);

mp_limb_t refmpn_sb_div_qr (mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t);
unsigned long refmpn_scan0 (mp_srcptr, unsigned long);
unsigned long refmpn_scan1 (mp_srcptr, unsigned long);
void refmpn_setbit (mp_ptr, unsigned long);
void refmpn_sqr (mp_ptr, mp_srcptr, mp_size_t);
void refmpn_sqrlo (mp_ptr, mp_srcptr, mp_size_t);
mp_size_t refmpn_sqrtrem (mp_ptr, mp_ptr, mp_srcptr, mp_size_t);

void refmpn_sub_ddmmss (mp_limb_t *, mp_limb_t *, mp_limb_t, mp_limb_t, mp_limb_t, mp_limb_t);
mp_limb_t refmpn_sub (mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);
mp_limb_t refmpn_sub_1 (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);
mp_limb_t refmpn_sub_err1_n (mp_ptr, mp_srcptr, mp_srcptr, mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);
mp_limb_t refmpn_sub_err2_n (mp_ptr, mp_srcptr, mp_srcptr, mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);
mp_limb_t refmpn_sub_err3_n (mp_ptr, mp_srcptr, mp_srcptr, mp_ptr, mp_srcptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);
mp_limb_t refmpn_sub_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
mp_limb_t refmpn_sub_nc (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);
mp_limb_t refmpn_sublsh1_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
mp_limb_t refmpn_sublsh2_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
mp_limb_t refmpn_sublsh_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, unsigned int);
mp_limb_t refmpn_sublsh1_n_ip1 (mp_ptr, mp_srcptr, mp_size_t);
mp_limb_t refmpn_sublsh2_n_ip1 (mp_ptr, mp_srcptr, mp_size_t);
mp_limb_t refmpn_sublsh_n_ip1 (mp_ptr, mp_srcptr, mp_size_t, unsigned int);
mp_limb_t refmpn_sublsh1_nc (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);
mp_limb_t refmpn_sublsh2_nc (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_t);
mp_limb_t refmpn_sublsh_nc (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, unsigned int, mp_limb_t);
mp_limb_t refmpn_submul_1 (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);
mp_limb_t refmpn_submul_1c (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t);

mp_limb_signed_t refmpn_rsblsh1_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
mp_limb_signed_t refmpn_rsblsh2_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
mp_limb_signed_t refmpn_rsblsh_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, unsigned int);
mp_limb_signed_t refmpn_rsblsh1_nc (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_signed_t);
mp_limb_signed_t refmpn_rsblsh2_nc (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, mp_limb_signed_t);
mp_limb_signed_t refmpn_rsblsh_nc (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t, unsigned int, mp_limb_signed_t);

void refmpn_tdiv_qr (mp_ptr, mp_ptr, mp_size_t, mp_ptr, mp_size_t, mp_srcptr, mp_size_t);
int refmpn_tstbit (mp_srcptr, unsigned long);

mp_limb_t refmpn_udiv_qrnnd (mp_limb_t *, mp_limb_t, mp_limb_t, mp_limb_t);
mp_limb_t refmpn_udiv_qrnnd_r (mp_limb_t, mp_limb_t, mp_limb_t, mp_limb_t *);
mp_limb_t refmpn_umul_ppmm (mp_limb_t *, mp_limb_t, mp_limb_t);
mp_limb_t refmpn_umul_ppmm_r (mp_limb_t, mp_limb_t, mp_limb_t *);

void refmpn_xnor_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
void refmpn_xor_n (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

void refmpn_zero (mp_ptr, mp_size_t);
void refmpn_zero_extend (mp_ptr, mp_size_t, mp_size_t);
int refmpn_zero_p (mp_srcptr, mp_size_t);

void refmpn_binvert (mp_ptr, mp_srcptr, mp_size_t, mp_ptr);
void refmpn_invert (mp_ptr, mp_srcptr, mp_size_t, mp_ptr);


void refmpq_add (mpq_ptr, mpq_srcptr, mpq_srcptr);
void refmpq_sub (mpq_ptr, mpq_srcptr, mpq_srcptr);


void refmpz_combit (mpz_ptr, unsigned long);
unsigned long refmpz_hamdist (mpz_srcptr, mpz_srcptr);
int refmpz_kronecker (mpz_srcptr, mpz_srcptr);
int refmpz_jacobi (mpz_srcptr, mpz_srcptr);
int refmpz_legendre (mpz_srcptr, mpz_srcptr);
int refmpz_kronecker_si (mpz_srcptr, long);
int refmpz_kronecker_ui (mpz_srcptr, unsigned long);
int refmpz_si_kronecker (long, mpz_srcptr);
int refmpz_ui_kronecker (unsigned long, mpz_srcptr);

void refmpz_pow_ui (mpz_ptr, mpz_srcptr, unsigned long);


#if defined (__cplusplus)
}
#endif


/* Establish ostringstream and istringstream.  Do this here so as to hide
   the conditionals, rather than putting stuff in each test program.

   Oldish versions of g++, like 2.95.2, don't have <sstream>, only
   <strstream>.  Fake up ostringstream and istringstream classes, but not a
   full implementation, just enough for our purposes.  */

#ifdef __cplusplus
#if 1 || HAVE_SSTREAM
#include <sstream>
#else /* ! HAVE_SSTREAM */
#include <string>
#include <strstream>
class
ostringstream : public std::ostrstream {
 public:
  string str() {
    int  pcount = ostrstream::pcount ();
    char *s = (char *) (*__gmp_allocate_func) (pcount + 1);
    memcpy (s, ostrstream::str(), pcount);
    s[pcount] = '\0';
    string ret = string(s);
    (*__gmp_free_func) (s, pcount + 1);
    return ret; }
};
class
istringstream : public std::istrstream {
 public:
  istringstream (const char *s) : istrstream (s) { };
};
#endif /* ! HAVE_SSTREAM */
#endif /* __cplusplus */


#define TESTS_REPS(count, argv, argc)					\
  do {									\
  char *envval, *end;							\
  double repfactor;							\
  int reps_nondefault = 0;						\
  if (argc > 1)								\
    {									\
      count = strtol (argv[1], &end, 0);				\
      if (*end || count <= 0)						\
	{								\
	  fprintf (stderr, "Invalid test count: %s.\n", argv[1]);	\
	  exit (1);							\
	}								\
      argv++;								\
      argc--;								\
      reps_nondefault = 1;						\
    }									\
  envval = getenv ("GMP_CHECK_REPFACTOR");				\
  if (envval != NULL)							\
    {									\
      repfactor = strtod (envval, &end);				\
      if (*end || repfactor <= 0)					\
	{								\
	  fprintf (stderr, "Invalid repfactor: %f.\n", repfactor);	\
	  exit (1);							\
	}								\
      count *= repfactor;						\
      count = MAX (count, 1);						\
      reps_nondefault = 1;						\
    }									\
  if (reps_nondefault)							\
    printf ("Running test with %ld repetitions (include this in bug reports)\n",\
	    (long) count);						\
  } while (0)


#endif /* __TESTS_H__ */
