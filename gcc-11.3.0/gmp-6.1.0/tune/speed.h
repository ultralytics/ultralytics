/* Header for speed and threshold things.

Copyright 1999-2003, 2005, 2006, 2008-2015 Free Software Foundation, Inc.

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

#ifndef __SPEED_H__
#define __SPEED_H__


/* Pad ptr,oldsize with zero limbs (at the most significant end) to make it
   newsize long. */
#define MPN_ZERO_EXTEND(ptr, oldsize, newsize)		\
  do {							\
    ASSERT ((newsize) >= (oldsize));			\
    MPN_ZERO ((ptr)+(oldsize), (newsize)-(oldsize));	\
  } while (0)

/* A mask of the least significant n bits.  Note 1<<32 doesn't give zero on
   x86 family CPUs, hence the separate case for GMP_LIMB_BITS. */
#define MP_LIMB_T_LOWBITMASK(n)	\
  ((n) == GMP_LIMB_BITS ? MP_LIMB_T_MAX : ((mp_limb_t) 1 << (n)) - 1)


/* align must be a power of 2 here, usually CACHE_LINE_SIZE is a good choice */

#define TMP_ALLOC_ALIGNED(bytes, align)	\
  align_pointer (TMP_ALLOC ((bytes) + (align)-1), (align))
#define TMP_ALLOC_LIMBS_ALIGNED(limbs, align)	\
  ((mp_ptr) TMP_ALLOC_ALIGNED ((limbs)*sizeof(mp_limb_t), align))

/* CACHE_LINE_SIZE is our default alignment for speed operands, and the
   limit on what s->align_xp etc and then request for off-alignment.  Maybe
   this should be an option of some sort, but in any case here are some line
   sizes,

       bytes
	 32   pentium
	 64   athlon
	 64   itanium-2 L1
	128   itanium-2 L2
*/
#define CACHE_LINE_SIZE   64 /* bytes */

#define SPEED_TMP_ALLOC_ADJUST_MASK  (CACHE_LINE_SIZE/GMP_LIMB_BYTES - 1)

/* Set ptr to a TMP_ALLOC block of the given limbs, with the given limb
   alignment.  */
#define SPEED_TMP_ALLOC_LIMBS(ptr, limbs, align)			\
  do {									\
    mp_ptr     __ptr;							\
    mp_size_t  __ptr_align, __ptr_add;					\
									\
    ASSERT ((CACHE_LINE_SIZE % GMP_LIMB_BYTES) == 0);		\
    __ptr = TMP_ALLOC_LIMBS ((limbs) + SPEED_TMP_ALLOC_ADJUST_MASK);	\
    __ptr_align = (__ptr - (mp_ptr) NULL);				\
    __ptr_add = ((align) - __ptr_align) & SPEED_TMP_ALLOC_ADJUST_MASK;	\
    (ptr) = __ptr + __ptr_add;						\
  } while (0)


/* This is the size for s->xp_block and s->yp_block, used in certain
   routines that want to run across many different data values and use
   s->size for a different purpose, eg. SPEED_ROUTINE_MPN_GCD_1.

   512 means 2kbytes of data for each of xp_block and yp_block, making 4k
   total, which should fit easily in any L1 data cache. */

#define SPEED_BLOCK_SIZE   512 /* limbs */


extern double  speed_unittime;
extern double  speed_cycletime;
extern int     speed_precision;
extern char    speed_time_string[];
void speed_time_init (void);
void speed_cycletime_fail (const char *str);
void speed_cycletime_init (void);
void speed_cycletime_need_cycles (void);
void speed_cycletime_need_seconds (void);
void speed_starttime (void);
double speed_endtime (void);


struct speed_params {
  unsigned   reps;	/* how many times to run the routine */
  mp_ptr     xp;	/* first argument */
  mp_ptr     yp;	/* second argument */
  mp_size_t  size;	/* size of both arguments */
  mp_limb_t  r;		/* user supplied parameter */
  mp_size_t  align_xp;	/* alignment of xp */
  mp_size_t  align_yp;	/* alignment of yp */
  mp_size_t  align_wp;	/* intended alignment of wp */
  mp_size_t  align_wp2; /* intended alignment of wp2 */
  mp_ptr     xp_block;	/* first special SPEED_BLOCK_SIZE block */
  mp_ptr     yp_block;	/* second special SPEED_BLOCK_SIZE block */

  double     time_divisor; /* optionally set by the speed routine */

  /* used by the cache priming things */
  int	     cache;
  unsigned   src_num, dst_num;
  struct {
    mp_ptr    ptr;
    mp_size_t size;
  } src[5], dst[4];
};

typedef double (*speed_function_t) (struct speed_params *);

double speed_measure (speed_function_t fun, struct speed_params *);

/* Prototypes for speed measuring routines */

double speed_back_to_back (struct speed_params *);
double speed_count_leading_zeros (struct speed_params *);
double speed_count_trailing_zeros (struct speed_params *);
double speed_find_a (struct speed_params *);
double speed_gmp_allocate_free (struct speed_params *);
double speed_gmp_allocate_reallocate_free (struct speed_params *);
double speed_invert_limb (struct speed_params *);
double speed_malloc_free (struct speed_params *);
double speed_malloc_realloc_free (struct speed_params *);
double speed_memcpy (struct speed_params *);
double speed_binvert_limb (struct speed_params *);
double speed_binvert_limb_mul1 (struct speed_params *);
double speed_binvert_limb_loop (struct speed_params *);
double speed_binvert_limb_cond (struct speed_params *);
double speed_binvert_limb_arith (struct speed_params *);

double speed_mpf_init_clear (struct speed_params *);

double speed_mpn_add_n (struct speed_params *);
double speed_mpn_add_1 (struct speed_params *);
double speed_mpn_add_1_inplace (struct speed_params *);
double speed_mpn_add_err1_n (struct speed_params *);
double speed_mpn_add_err2_n (struct speed_params *);
double speed_mpn_add_err3_n (struct speed_params *);
double speed_mpn_addlsh_n (struct speed_params *);
double speed_mpn_addlsh1_n (struct speed_params *);
double speed_mpn_addlsh2_n (struct speed_params *);
double speed_mpn_addlsh_n_ip1 (struct speed_params *);
double speed_mpn_addlsh1_n_ip1 (struct speed_params *);
double speed_mpn_addlsh2_n_ip1 (struct speed_params *);
double speed_mpn_addlsh_n_ip2 (struct speed_params *);
double speed_mpn_addlsh1_n_ip2 (struct speed_params *);
double speed_mpn_addlsh2_n_ip2 (struct speed_params *);
double speed_mpn_add_n_sub_n (struct speed_params *);
double speed_mpn_and_n (struct speed_params *);
double speed_mpn_andn_n (struct speed_params *);
double speed_mpn_addmul_1 (struct speed_params *);
double speed_mpn_addmul_2 (struct speed_params *);
double speed_mpn_addmul_3 (struct speed_params *);
double speed_mpn_addmul_4 (struct speed_params *);
double speed_mpn_addmul_5 (struct speed_params *);
double speed_mpn_addmul_6 (struct speed_params *);
double speed_mpn_addmul_7 (struct speed_params *);
double speed_mpn_addmul_8 (struct speed_params *);
double speed_mpn_cnd_add_n (struct speed_params *);
double speed_mpn_cnd_sub_n (struct speed_params *);
double speed_mpn_com (struct speed_params *);
double speed_mpn_neg (struct speed_params *);
double speed_mpn_copyd (struct speed_params *);
double speed_mpn_copyi (struct speed_params *);
double speed_MPN_COPY (struct speed_params *);
double speed_MPN_COPY_DECR (struct speed_params *);
double speed_MPN_COPY_INCR (struct speed_params *);
double speed_mpn_sec_tabselect (struct speed_params *);
double speed_mpn_divexact_1 (struct speed_params *);
double speed_mpn_divexact_by3 (struct speed_params *);
double speed_mpn_bdiv_q_1 (struct speed_params *);
double speed_mpn_pi1_bdiv_q_1 (struct speed_params *);
double speed_mpn_bdiv_dbm1c (struct speed_params *);
double speed_mpn_divrem_1 (struct speed_params *);
double speed_mpn_divrem_1f (struct speed_params *);
double speed_mpn_divrem_1c (struct speed_params *);
double speed_mpn_divrem_1cf (struct speed_params *);
double speed_mpn_divrem_1_div (struct speed_params *);
double speed_mpn_divrem_1f_div (struct speed_params *);
double speed_mpn_divrem_1_inv (struct speed_params *);
double speed_mpn_divrem_1f_inv (struct speed_params *);
double speed_mpn_divrem_2 (struct speed_params *);
double speed_mpn_divrem_2_div (struct speed_params *);
double speed_mpn_divrem_2_inv (struct speed_params *);
double speed_mpn_div_qr_1n_pi1 (struct speed_params *);
double speed_mpn_div_qr_1n_pi1_1 (struct speed_params *);
double speed_mpn_div_qr_1n_pi1_2 (struct speed_params *);
double speed_mpn_div_qr_1 (struct speed_params *);
double speed_mpn_div_qr_2n (struct speed_params *);
double speed_mpn_div_qr_2u (struct speed_params *);
double speed_mpn_fib2_ui (struct speed_params *);
double speed_mpn_matrix22_mul (struct speed_params *);
double speed_mpn_hgcd (struct speed_params *);
double speed_mpn_hgcd_lehmer (struct speed_params *);
double speed_mpn_hgcd_appr (struct speed_params *);
double speed_mpn_hgcd_appr_lehmer (struct speed_params *);
double speed_mpn_hgcd_reduce (struct speed_params *);
double speed_mpn_hgcd_reduce_1 (struct speed_params *);
double speed_mpn_hgcd_reduce_2 (struct speed_params *);
double speed_mpn_gcd (struct speed_params *);
double speed_mpn_gcd_1 (struct speed_params *);
double speed_mpn_gcd_1N (struct speed_params *);
double speed_mpn_gcdext (struct speed_params *);
double speed_mpn_gcdext_double (struct speed_params *);
double speed_mpn_gcdext_one_double (struct speed_params *);
double speed_mpn_gcdext_one_single (struct speed_params *);
double speed_mpn_gcdext_single (struct speed_params *);
double speed_mpn_get_str (struct speed_params *);
double speed_mpn_hamdist (struct speed_params *);
double speed_mpn_ior_n (struct speed_params *);
double speed_mpn_iorn_n (struct speed_params *);
double speed_mpn_jacobi_base (struct speed_params *);
double speed_mpn_jacobi_base_1 (struct speed_params *);
double speed_mpn_jacobi_base_2 (struct speed_params *);
double speed_mpn_jacobi_base_3 (struct speed_params *);
double speed_mpn_jacobi_base_4 (struct speed_params *);
double speed_mpn_lshift (struct speed_params *);
double speed_mpn_lshiftc (struct speed_params *);
double speed_mpn_mod_1 (struct speed_params *);
double speed_mpn_mod_1c (struct speed_params *);
double speed_mpn_mod_1_div (struct speed_params *);
double speed_mpn_mod_1_inv (struct speed_params *);
double speed_mpn_mod_1_1 (struct speed_params *);
double speed_mpn_mod_1_1_1 (struct speed_params *);
double speed_mpn_mod_1_1_2 (struct speed_params *);
double speed_mpn_mod_1_2 (struct speed_params *);
double speed_mpn_mod_1_3 (struct speed_params *);
double speed_mpn_mod_1_4 (struct speed_params *);
double speed_mpn_mod_34lsub1 (struct speed_params *);
double speed_mpn_modexact_1_odd (struct speed_params *);
double speed_mpn_modexact_1c_odd (struct speed_params *);
double speed_mpn_mul_1 (struct speed_params *);
double speed_mpn_mul_1_inplace (struct speed_params *);
double speed_mpn_mul_2 (struct speed_params *);
double speed_mpn_mul_3 (struct speed_params *);
double speed_mpn_mul_4 (struct speed_params *);
double speed_mpn_mul_5 (struct speed_params *);
double speed_mpn_mul_6 (struct speed_params *);
double speed_mpn_mul (struct speed_params *);
double speed_mpn_mul_basecase (struct speed_params *);
double speed_mpn_mulmid (struct speed_params *);
double speed_mpn_mulmid_basecase (struct speed_params *);
double speed_mpn_mul_fft (struct speed_params *);
double speed_mpn_mul_fft_sqr (struct speed_params *);
double speed_mpn_fft_mul (struct speed_params *);
double speed_mpn_fft_sqr (struct speed_params *);
#if WANT_OLD_FFT_FULL
double speed_mpn_mul_fft_full (struct speed_params *);
double speed_mpn_mul_fft_full_sqr (struct speed_params *);
#endif
double speed_mpn_nussbaumer_mul (struct speed_params *);
double speed_mpn_nussbaumer_mul_sqr (struct speed_params *);
double speed_mpn_mul_n (struct speed_params *);
double speed_mpn_mul_n_sqr (struct speed_params *);
double speed_mpn_mulmid_n (struct speed_params *);
double speed_mpn_sqrlo (struct speed_params *);
double speed_mpn_sqrlo_basecase (struct speed_params *);
double speed_mpn_mullo_n (struct speed_params *);
double speed_mpn_mullo_basecase (struct speed_params *);
double speed_mpn_nand_n (struct speed_params *);
double speed_mpn_nior_n (struct speed_params *);
double speed_mpn_popcount (struct speed_params *);
double speed_mpn_preinv_divrem_1 (struct speed_params *);
double speed_mpn_preinv_divrem_1f (struct speed_params *);
double speed_mpn_preinv_mod_1 (struct speed_params *);
double speed_mpn_sbpi1_div_qr (struct speed_params *);
double speed_mpn_dcpi1_div_qr (struct speed_params *);
double speed_mpn_sbpi1_divappr_q (struct speed_params *);
double speed_mpn_dcpi1_divappr_q (struct speed_params *);
double speed_mpn_mu_div_qr (struct speed_params *);
double speed_mpn_mu_divappr_q (struct speed_params *);
double speed_mpn_mupi_div_qr (struct speed_params *);
double speed_mpn_mu_div_q (struct speed_params *);
double speed_mpn_sbpi1_bdiv_qr (struct speed_params *);
double speed_mpn_dcpi1_bdiv_qr (struct speed_params *);
double speed_mpn_sbpi1_bdiv_q (struct speed_params *);
double speed_mpn_dcpi1_bdiv_q (struct speed_params *);
double speed_mpn_mu_bdiv_q (struct speed_params *);
double speed_mpn_mu_bdiv_qr (struct speed_params *);
double speed_mpn_broot (struct speed_params *);
double speed_mpn_broot_invm1 (struct speed_params *);
double speed_mpn_brootinv (struct speed_params *);
double speed_mpn_invert (struct speed_params *);
double speed_mpn_invertappr (struct speed_params *);
double speed_mpn_ni_invertappr (struct speed_params *);
double speed_mpn_sec_invert (struct speed_params *s);
double speed_mpn_binvert (struct speed_params *);
double speed_mpn_redc_1 (struct speed_params *);
double speed_mpn_redc_2 (struct speed_params *);
double speed_mpn_redc_n (struct speed_params *);
double speed_mpn_rsblsh_n (struct speed_params *);
double speed_mpn_rsblsh1_n (struct speed_params *);
double speed_mpn_rsblsh2_n (struct speed_params *);
double speed_mpn_rsh1add_n (struct speed_params *);
double speed_mpn_rsh1sub_n (struct speed_params *);
double speed_mpn_rshift (struct speed_params *);
double speed_mpn_sb_divrem_m3 (struct speed_params *);
double speed_mpn_sb_divrem_m3_div (struct speed_params *);
double speed_mpn_sb_divrem_m3_inv (struct speed_params *);
double speed_mpn_set_str (struct speed_params *);
double speed_mpn_bc_set_str (struct speed_params *);
double speed_mpn_dc_set_str (struct speed_params *);
double speed_mpn_set_str_pre (struct speed_params *);
double speed_mpn_sqr_basecase (struct speed_params *);
double speed_mpn_sqr_diag_addlsh1 (struct speed_params *);
double speed_mpn_sqr_diagonal (struct speed_params *);
double speed_mpn_sqr (struct speed_params *);
double speed_mpn_sqrtrem (struct speed_params *);
double speed_mpn_rootrem (struct speed_params *);
double speed_mpn_sqrt (struct speed_params *);
double speed_mpn_root (struct speed_params *);
double speed_mpn_sub_n (struct speed_params *);
double speed_mpn_sub_1 (struct speed_params *);
double speed_mpn_sub_1_inplace (struct speed_params *);
double speed_mpn_sub_err1_n (struct speed_params *);
double speed_mpn_sub_err2_n (struct speed_params *);
double speed_mpn_sub_err3_n (struct speed_params *);
double speed_mpn_sublsh_n (struct speed_params *);
double speed_mpn_sublsh1_n (struct speed_params *);
double speed_mpn_sublsh2_n (struct speed_params *);
double speed_mpn_sublsh_n_ip1 (struct speed_params *);
double speed_mpn_sublsh1_n_ip1 (struct speed_params *);
double speed_mpn_sublsh2_n_ip1 (struct speed_params *);
double speed_mpn_submul_1 (struct speed_params *);
double speed_mpn_toom2_sqr (struct speed_params *);
double speed_mpn_toom3_sqr (struct speed_params *);
double speed_mpn_toom4_sqr (struct speed_params *);
double speed_mpn_toom6_sqr (struct speed_params *);
double speed_mpn_toom8_sqr (struct speed_params *);
double speed_mpn_toom22_mul (struct speed_params *);
double speed_mpn_toom33_mul (struct speed_params *);
double speed_mpn_toom44_mul (struct speed_params *);
double speed_mpn_toom6h_mul (struct speed_params *);
double speed_mpn_toom8h_mul (struct speed_params *);
double speed_mpn_toom32_mul (struct speed_params *);
double speed_mpn_toom42_mul (struct speed_params *);
double speed_mpn_toom43_mul (struct speed_params *);
double speed_mpn_toom63_mul (struct speed_params *);
double speed_mpn_toom32_for_toom43_mul (struct speed_params *);
double speed_mpn_toom43_for_toom32_mul (struct speed_params *);
double speed_mpn_toom32_for_toom53_mul (struct speed_params *);
double speed_mpn_toom53_for_toom32_mul (struct speed_params *);
double speed_mpn_toom42_for_toom53_mul (struct speed_params *);
double speed_mpn_toom53_for_toom42_mul (struct speed_params *);
double speed_mpn_toom43_for_toom54_mul (struct speed_params *);
double speed_mpn_toom54_for_toom43_mul (struct speed_params *);
double speed_mpn_toom42_mulmid (struct speed_params *);
double speed_mpn_mulmod_bnm1 (struct speed_params *);
double speed_mpn_bc_mulmod_bnm1 (struct speed_params *);
double speed_mpn_mulmod_bnm1_rounded (struct speed_params *);
double speed_mpn_sqrmod_bnm1 (struct speed_params *);
double speed_mpn_udiv_qrnnd (struct speed_params *);
double speed_mpn_udiv_qrnnd_r (struct speed_params *);
double speed_mpn_umul_ppmm (struct speed_params *);
double speed_mpn_umul_ppmm_r (struct speed_params *);
double speed_mpn_xnor_n (struct speed_params *);
double speed_mpn_xor_n (struct speed_params *);
double speed_MPN_ZERO (struct speed_params *);

double speed_mpq_init_clear (struct speed_params *);

double speed_mpz_add (struct speed_params *);
double speed_mpz_bin_uiui (struct speed_params *);
double speed_mpz_bin_ui (struct speed_params *);
double speed_mpz_fac_ui (struct speed_params *);
double speed_mpz_2fac_ui (struct speed_params *);
double speed_mpz_fib_ui (struct speed_params *);
double speed_mpz_fib2_ui (struct speed_params *);
double speed_mpz_init_clear (struct speed_params *);
double speed_mpz_init_realloc_clear (struct speed_params *);
double speed_mpz_jacobi (struct speed_params *);
double speed_mpz_lucnum_ui (struct speed_params *);
double speed_mpz_lucnum2_ui (struct speed_params *);
double speed_mpz_mod (struct speed_params *);
double speed_mpz_powm (struct speed_params *);
double speed_mpz_powm_mod (struct speed_params *);
double speed_mpz_powm_redc (struct speed_params *);
double speed_mpz_powm_sec (struct speed_params *);
double speed_mpz_powm_ui (struct speed_params *);
double speed_mpz_urandomb (struct speed_params *);

double speed_gmp_randseed (struct speed_params *);
double speed_gmp_randseed_ui (struct speed_params *);

double speed_noop (struct speed_params *);
double speed_noop_wxs (struct speed_params *);
double speed_noop_wxys (struct speed_params *);

double speed_operator_div (struct speed_params *);
double speed_operator_mod (struct speed_params *);

double speed_udiv_qrnnd (struct speed_params *);
double speed_udiv_qrnnd_preinv1 (struct speed_params *);
double speed_udiv_qrnnd_preinv2 (struct speed_params *);
double speed_udiv_qrnnd_preinv3 (struct speed_params *);
double speed_udiv_qrnnd_c (struct speed_params *);
double speed_umul_ppmm (struct speed_params *);

/* Prototypes for other routines */

#if defined (__cplusplus)
extern "C" {
#endif

/* low 32-bits in p[0], high 32-bits in p[1] */
void speed_cyclecounter (unsigned p[2]);

#if defined (__cplusplus)
}
#endif

void mftb_function (unsigned p[2]);

double speed_cyclecounter_diff (const unsigned [2], const unsigned [2]);
int gettimeofday_microseconds_p (void);
int getrusage_microseconds_p (void);
int cycles_works_p (void);
long clk_tck (void);
double freq_measure (const char *, double (*)(void));

int double_cmp_ptr (const double *, const double *);
void pentium_wbinvd (void);
typedef int (*qsort_function_t) (const void *, const void *);

void noop (void);
void noop_1 (mp_limb_t);
void noop_wxs (mp_ptr, mp_srcptr, mp_size_t);
void noop_wxys (mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);
void mpn_cache_fill (mp_srcptr, mp_size_t);
void mpn_cache_fill_dummy (mp_limb_t);
void speed_cache_fill (struct speed_params *);
void speed_operand_src (struct speed_params *, mp_ptr, mp_size_t);
void speed_operand_dst (struct speed_params *, mp_ptr, mp_size_t);

extern int  speed_option_addrs;
extern int  speed_option_verbose;
extern int  speed_option_cycles_broken;
void speed_option_set (const char *);

mp_limb_t mpn_div_qr_1n_pi1_1 (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t, mp_limb_t);
mp_limb_t mpn_div_qr_1n_pi1_2 (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t, mp_limb_t);

mp_limb_t mpn_divrem_1_div (mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t);
mp_limb_t mpn_divrem_1_inv (mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t);
mp_limb_t mpn_divrem_2_div (mp_ptr, mp_size_t, mp_ptr, mp_size_t, mp_srcptr);
mp_limb_t mpn_divrem_2_inv (mp_ptr, mp_size_t, mp_ptr, mp_size_t, mp_srcptr);

int mpn_jacobi_base_1 (mp_limb_t, mp_limb_t, int);
int mpn_jacobi_base_2 (mp_limb_t, mp_limb_t, int);
int mpn_jacobi_base_3 (mp_limb_t, mp_limb_t, int);
int mpn_jacobi_base_4 (mp_limb_t, mp_limb_t, int);

mp_limb_t mpn_mod_1_div (mp_srcptr, mp_size_t, mp_limb_t);
mp_limb_t mpn_mod_1_inv (mp_srcptr, mp_size_t, mp_limb_t);

mp_limb_t mpn_mod_1_1p_1 (mp_srcptr, mp_size_t, mp_limb_t, const mp_limb_t [4]);
mp_limb_t mpn_mod_1_1p_2 (mp_srcptr, mp_size_t, mp_limb_t, const mp_limb_t [4]);

void mpn_mod_1_1p_cps_1 (mp_limb_t [4], mp_limb_t);
void mpn_mod_1_1p_cps_2 (mp_limb_t [4], mp_limb_t);

mp_size_t mpn_gcdext_one_double (mp_ptr, mp_ptr, mp_size_t *, mp_ptr, mp_size_t, mp_ptr, mp_size_t);
mp_size_t mpn_gcdext_one_single (mp_ptr, mp_ptr, mp_size_t *, mp_ptr, mp_size_t, mp_ptr, mp_size_t);
mp_size_t mpn_gcdext_single (mp_ptr, mp_ptr, mp_size_t *, mp_ptr, mp_size_t, mp_ptr, mp_size_t);
mp_size_t mpn_gcdext_double (mp_ptr, mp_ptr, mp_size_t *, mp_ptr, mp_size_t, mp_ptr, mp_size_t);
mp_size_t mpn_hgcd_lehmer (mp_ptr, mp_ptr, mp_size_t, struct hgcd_matrix *, mp_ptr);
mp_size_t mpn_hgcd_lehmer_itch (mp_size_t);

mp_size_t mpn_hgcd_appr_lehmer (mp_ptr, mp_ptr, mp_size_t, struct hgcd_matrix *, mp_ptr);
mp_size_t mpn_hgcd_appr_lehmer_itch (mp_size_t);

mp_size_t mpn_hgcd_reduce_1 (struct hgcd_matrix *, mp_ptr, mp_ptr, mp_size_t, mp_size_t, mp_ptr);
mp_size_t mpn_hgcd_reduce_1_itch (mp_size_t, mp_size_t);

mp_size_t mpn_hgcd_reduce_2 (struct hgcd_matrix *, mp_ptr, mp_ptr, mp_size_t, mp_size_t, mp_ptr);
mp_size_t mpn_hgcd_reduce_2_itch (mp_size_t, mp_size_t);

mp_limb_t mpn_sb_divrem_mn_div (mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t);
mp_limb_t mpn_sb_divrem_mn_inv (mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t);

mp_size_t mpn_set_str_basecase (mp_ptr, const unsigned char *, size_t, int);
void mpn_pre_set_str (mp_ptr, unsigned char *, size_t, powers_t *, mp_ptr);

void mpz_powm_mod (mpz_ptr, mpz_srcptr, mpz_srcptr, mpz_srcptr);
void mpz_powm_redc (mpz_ptr, mpz_srcptr, mpz_srcptr, mpz_srcptr);

int speed_routine_count_zeros_setup (struct speed_params *, mp_ptr, int, int);


/* "get" is called repeatedly until it ticks over, just in case on a fast
   processor it takes less than a microsecond, though this is probably
   unlikely if it's a system call.

   speed_cyclecounter is called on the same side of the "get" for the start
   and end measurements.  It doesn't matter how long it takes from the "get"
   sample to the cycles sample, since that period will cancel out in the
   difference calculation (assuming it's the same each time).

   Letting the test run for more than a process time slice is probably only
   going to reduce accuracy, especially for getrusage when the cycle counter
   is real time, or for gettimeofday if the cycle counter is in fact process
   time.  Use CLK_TCK/2 as a reasonable stop.

   It'd be desirable to be quite accurate here.  The default speed_precision
   for a cycle counter is 10000 cycles, so to mix that with getrusage or
   gettimeofday the frequency should be at least that accurate.  But running
   measurements for 10000 microseconds (or more) is too long.  Be satisfied
   with just a half clock tick (5000 microseconds usually).  */

#define FREQ_MEASURE_ONE(name, type, get, getc, sec, usec)		\
  do {									\
    type      st1, st, et1, et;						\
    unsigned  sc[2], ec[2];						\
    long      dt, half_tick;						\
    double    dc, cyc;							\
									\
    half_tick = (1000000L / clk_tck()) / 2;				\
									\
    get (st1);								\
    do {								\
      get (st);								\
    } while (usec(st) == usec(st1) && sec(st) == sec(st1));		\
									\
    getc (sc);								\
									\
    for (;;)								\
      {									\
	get (et1);							\
	do {								\
	  get (et);							\
	} while (usec(et) == usec(et1) && sec(et) == sec(et1));		\
									\
	getc (ec);							\
									\
	dc = speed_cyclecounter_diff (ec, sc);				\
									\
	/* allow secs to cancel before multiplying */			\
	dt = sec(et) - sec(st);						\
	dt = dt * 1000000L + (usec(et) - usec(st));			\
									\
	if (dt >= half_tick)						\
	  break;							\
      }									\
									\
    cyc = dt * 1e-6 / dc;						\
									\
    if (speed_option_verbose >= 2)					\
      printf ("freq_measure_%s_one() dc=%.6g dt=%ld cyc=%.6g\n",	\
	      name, dc, dt, cyc);					\
									\
    return dt * 1e-6 / dc;						\
									\
  } while (0)




/* The measuring routines use these big macros to save duplication for
   similar forms.  They also get used for some automatically generated
   measuring of new implementations of functions.

   Having something like SPEED_ROUTINE_BINARY_N as a subroutine accepting a
   function pointer is considered undesirable since it's not the way a
   normal application will be calling, and some processors might do
   different things with an indirect call, like not branch predicting, or
   doing a full pipe flush.  At least some of the "functions" measured are
   actually macros too.

   The net effect is to bloat the object code, possibly in a big way, but
   only what's being measured is being run, so that doesn't matter.

   The loop forms don't try to cope with __GMP_ATTRIBUTE_PURE or
   ATTRIBUTE_CONST on the called functions.  Adding a cast to a non-pure
   function pointer doesn't work in gcc 3.2.  Using an actual non-pure
   function pointer variable works, but stands a real risk of a
   non-optimizing compiler generating unnecessary overheads in the call.
   Currently the best idea is not to use those attributes for a timing
   program build.  __GMP_NO_ATTRIBUTE_CONST_PURE will tell gmp.h and
   gmp-impl.h to omit them from routines there.  */

#define SPEED_RESTRICT_COND(cond)   if (!(cond)) return -1.0;

/* For mpn_copy or similar. */
#define SPEED_ROUTINE_MPN_COPY_CALL(call)				\
  {									\
    mp_ptr    wp;							\
    unsigned  i;							\
    double    t;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 0);					\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (wp, s->size, s->align_wp);			\
									\
    speed_operand_src (s, s->xp, s->size);				\
    speed_operand_dst (s, wp, s->size);					\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      call;								\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }
#define SPEED_ROUTINE_MPN_COPY(function)				\
  SPEED_ROUTINE_MPN_COPY_CALL (function (wp, s->xp, s->size))

#define SPEED_ROUTINE_MPN_TABSELECT(function)				\
  {									\
    mp_ptr    xp, wp;							\
    unsigned  i;							\
    double    t;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 0);					\
									\
    if (s->r == 0)							\
      s->r = s->size;	/* default to a quadratic shape */		\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (xp, s->size * s->r, s->align_xp);		\
    SPEED_TMP_ALLOC_LIMBS (wp, s->size, s->align_wp);			\
									\
    speed_operand_src (s, xp, s->size * s->r);				\
    speed_operand_dst (s, wp, s->size);					\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      function (wp, xp, s->size, s->r, (s->r) / 2);			\
    while (--i != 0);							\
    t = speed_endtime () / s->r;					\
									\
    TMP_FREE;								\
    return t;								\
  }


#define SPEED_ROUTINE_MPN_COPYC(function)				\
  {									\
    mp_ptr    wp;							\
    unsigned  i;							\
    double    t;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 0);					\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (wp, s->size, s->align_wp);			\
									\
    speed_operand_src (s, s->xp, s->size);				\
    speed_operand_dst (s, wp, s->size);					\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      function (wp, s->xp, s->size, 0);					\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }

/* s->size is still in limbs, and it's limbs which are copied, but
   "function" takes a size in bytes not limbs.  */
#define SPEED_ROUTINE_MPN_COPY_BYTES(function)				\
  {									\
    mp_ptr    wp;							\
    unsigned  i;							\
    double    t;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 0);					\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (wp, s->size, s->align_wp);			\
									\
    speed_operand_src (s, s->xp, s->size);				\
    speed_operand_dst (s, wp, s->size);					\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      function (wp, s->xp, s->size * GMP_LIMB_BYTES);		\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }


/* For mpn_add_n, mpn_sub_n, or similar. */
#define SPEED_ROUTINE_MPN_BINARY_N_CALL(call)				\
  {									\
    mp_ptr     wp;							\
    mp_ptr     xp, yp;							\
    unsigned   i;							\
    double     t;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 1);					\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (wp, s->size, s->align_wp);			\
									\
    xp = s->xp;								\
    yp = s->yp;								\
									\
    if (s->r == 0)	;						\
    else if (s->r == 1) { xp = wp;	    }				\
    else if (s->r == 2) {	   yp = wp; }				\
    else if (s->r == 3) { xp = wp; yp = wp; }				\
    else if (s->r == 4) {     yp = xp;	    }				\
    else		{						\
      TMP_FREE;								\
      return -1.0;							\
    }									\
									\
    /* initialize wp if operand overlap */				\
    if (xp == wp || yp == wp)						\
      MPN_COPY (wp, s->xp, s->size);					\
									\
    speed_operand_src (s, xp, s->size);					\
    speed_operand_src (s, yp, s->size);					\
    speed_operand_dst (s, wp, s->size);					\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      call;								\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }


/* For mpn_aors_errK_n, where 1 <= K <= 3. */
#define SPEED_ROUTINE_MPN_BINARY_ERR_N_CALL(call, K)			\
  {									\
    mp_ptr     wp;							\
    mp_ptr     xp, yp;							\
    mp_ptr     zp[K];							\
    mp_limb_t  ep[2*K];							\
    unsigned   i;							\
    double     t;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 1);					\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (wp, s->size, s->align_wp);			\
									\
    /* (don't have a mechanism to specify zp alignments) */		\
    for (i = 0; i < K; i++)						\
      SPEED_TMP_ALLOC_LIMBS (zp[i], s->size, 0);			\
									\
    xp = s->xp;								\
    yp = s->yp;								\
									\
    if (s->r == 0)	;						\
    else if (s->r == 1) { xp = wp;	    }				\
    else if (s->r == 2) {	   yp = wp; }				\
    else if (s->r == 3) { xp = wp; yp = wp; }				\
    else if (s->r == 4) {     yp = xp;	    }				\
    else		{						\
      TMP_FREE;								\
      return -1.0;							\
    }									\
									\
    /* initialize wp if operand overlap */				\
    if (xp == wp || yp == wp)						\
      MPN_COPY (wp, s->xp, s->size);					\
									\
    speed_operand_src (s, xp, s->size);					\
    speed_operand_src (s, yp, s->size);					\
    for (i = 0; i < K; i++)						\
      speed_operand_src (s, zp[i], s->size);				\
    speed_operand_dst (s, wp, s->size);					\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      call;								\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }

#define SPEED_ROUTINE_MPN_BINARY_ERR1_N(function)			\
  SPEED_ROUTINE_MPN_BINARY_ERR_N_CALL ((*function) (wp, xp, yp, ep, zp[0], s->size, 0), 1)

#define SPEED_ROUTINE_MPN_BINARY_ERR2_N(function)			\
  SPEED_ROUTINE_MPN_BINARY_ERR_N_CALL ((*function) (wp, xp, yp, ep, zp[0], zp[1], s->size, 0), 2)

#define SPEED_ROUTINE_MPN_BINARY_ERR3_N(function)			\
  SPEED_ROUTINE_MPN_BINARY_ERR_N_CALL ((*function) (wp, xp, yp, ep, zp[0], zp[1], zp[2], s->size, 0), 3)


/* For mpn_add_n, mpn_sub_n, or similar. */
#define SPEED_ROUTINE_MPN_ADDSUB_N_CALL(call)				\
  {									\
    mp_ptr     ap, sp;							\
    mp_ptr     xp, yp;							\
    unsigned   i;							\
    double     t;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 1);					\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (ap, s->size, s->align_wp);			\
    SPEED_TMP_ALLOC_LIMBS (sp, s->size, s->align_wp);			\
									\
    xp = s->xp;								\
    yp = s->yp;								\
									\
    if ((s->r & 1) != 0) { xp = ap; }					\
    if ((s->r & 2) != 0) { yp = ap; }					\
    if ((s->r & 4) != 0) { xp = sp; }					\
    if ((s->r & 8) != 0) { yp = sp; }					\
    if ((s->r & 3) == 3  ||  (s->r & 12) == 12)				\
      {									\
	TMP_FREE;							\
	return -1.0;							\
      }									\
									\
    /* initialize ap if operand overlap */				\
    if (xp == ap || yp == ap)						\
      MPN_COPY (ap, s->xp, s->size);					\
    /* initialize sp if operand overlap */				\
    if (xp == sp || yp == sp)						\
      MPN_COPY (sp, s->xp, s->size);					\
									\
    speed_operand_src (s, xp, s->size);					\
    speed_operand_src (s, yp, s->size);					\
    speed_operand_dst (s, ap, s->size);					\
    speed_operand_dst (s, sp, s->size);					\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      call;								\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }

#define SPEED_ROUTINE_MPN_BINARY_N(function)				\
   SPEED_ROUTINE_MPN_BINARY_N_CALL ((*function) (wp, xp, yp, s->size))

#define SPEED_ROUTINE_MPN_BINARY_NC(function)				\
   SPEED_ROUTINE_MPN_BINARY_N_CALL ((*function) (wp, xp, yp, s->size, 0))


/* For mpn_lshift, mpn_rshift, mpn_mul_1, with r, or similar. */
#define SPEED_ROUTINE_MPN_UNARY_1_CALL(call)				\
  {									\
    mp_ptr    wp;							\
    unsigned  i;							\
    double    t;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 1);					\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (wp, s->size, s->align_wp);			\
									\
    speed_operand_src (s, s->xp, s->size);				\
    speed_operand_dst (s, wp, s->size);					\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      call;								\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }

#define SPEED_ROUTINE_MPN_UNARY_1(function)				\
  SPEED_ROUTINE_MPN_UNARY_1_CALL ((*function) (wp, s->xp, s->size, s->r))

#define SPEED_ROUTINE_MPN_UNARY_1C(function)				\
  SPEED_ROUTINE_MPN_UNARY_1_CALL ((*function) (wp, s->xp, s->size, s->r, 0))

/* FIXME: wp is uninitialized here, should start it off from xp */
#define SPEED_ROUTINE_MPN_UNARY_1_INPLACE(function)			\
  SPEED_ROUTINE_MPN_UNARY_1_CALL ((*function) (wp, wp, s->size, s->r))

#define SPEED_ROUTINE_MPN_DIVEXACT_1(function)				\
  SPEED_ROUTINE_MPN_UNARY_1_CALL ((*function) (wp, s->xp, s->size, s->r))

#define SPEED_ROUTINE_MPN_BDIV_Q_1(function)				\
    SPEED_ROUTINE_MPN_UNARY_1_CALL ((*function) (wp, s->xp, s->size, s->r))

#define SPEED_ROUTINE_MPN_PI1_BDIV_Q_1_CALL(call)			\
  {									\
    unsigned   shift;							\
    mp_limb_t  dinv;							\
									\
    SPEED_RESTRICT_COND (s->size > 0);					\
    SPEED_RESTRICT_COND (s->r != 0);					\
									\
    count_trailing_zeros (shift, s->r);					\
    binvert_limb (dinv, s->r >> shift);					\
									\
    SPEED_ROUTINE_MPN_UNARY_1_CALL (call);				\
  }
#define SPEED_ROUTINE_MPN_PI1_BDIV_Q_1(function)			\
  SPEED_ROUTINE_MPN_PI1_BDIV_Q_1_CALL					\
  ((*function) (wp, s->xp, s->size, s->r, dinv, shift))

#define SPEED_ROUTINE_MPN_BDIV_DBM1C(function)				\
  SPEED_ROUTINE_MPN_UNARY_1_CALL ((*function) (wp, s->xp, s->size, s->r, 0))

#define SPEED_ROUTINE_MPN_DIVREM_1(function)				\
  SPEED_ROUTINE_MPN_UNARY_1_CALL ((*function) (wp, 0, s->xp, s->size, s->r))

#define SPEED_ROUTINE_MPN_DIVREM_1C(function)				\
  SPEED_ROUTINE_MPN_UNARY_1_CALL ((*function) (wp, 0, s->xp, s->size, s->r, 0))

#define SPEED_ROUTINE_MPN_DIVREM_1F(function)				\
  SPEED_ROUTINE_MPN_UNARY_1_CALL ((*function) (wp, s->size, s->xp, 0, s->r))

#define SPEED_ROUTINE_MPN_DIVREM_1CF(function)				\
  SPEED_ROUTINE_MPN_UNARY_1_CALL ((*function) (wp, s->size, s->xp, 0, s->r, 0))


#define SPEED_ROUTINE_MPN_PREINV_DIVREM_1_CALL(call)			\
  {									\
    unsigned   shift;							\
    mp_limb_t  dinv;							\
									\
    SPEED_RESTRICT_COND (s->size >= 0);					\
    SPEED_RESTRICT_COND (s->r != 0);					\
									\
    count_leading_zeros (shift, s->r);					\
    invert_limb (dinv, s->r << shift);					\
									\
    SPEED_ROUTINE_MPN_UNARY_1_CALL (call);				\
  }									\

#define SPEED_ROUTINE_MPN_PREINV_DIVREM_1(function)			\
  SPEED_ROUTINE_MPN_PREINV_DIVREM_1_CALL				\
  ((*function) (wp, 0, s->xp, s->size, s->r, dinv, shift))

/* s->size limbs worth of fraction part */
#define SPEED_ROUTINE_MPN_PREINV_DIVREM_1F(function)			\
  SPEED_ROUTINE_MPN_PREINV_DIVREM_1_CALL				\
  ((*function) (wp, s->size, s->xp, 0, s->r, dinv, shift))


/* s->r is duplicated to form the multiplier, defaulting to
   MP_BASES_BIG_BASE_10.  Not sure if that's particularly useful, but at
   least it provides some control.  */
#define SPEED_ROUTINE_MPN_UNARY_N(function,N)				\
  {									\
    mp_ptr     wp;							\
    mp_size_t  wn;							\
    unsigned   i;							\
    double     t;							\
    mp_limb_t  yp[N];							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= N);					\
									\
    TMP_MARK;								\
    wn = s->size + N-1;							\
    SPEED_TMP_ALLOC_LIMBS (wp, wn, s->align_wp);			\
    for (i = 0; i < N; i++)						\
      yp[i] = (s->r != 0 ? s->r : MP_BASES_BIG_BASE_10);		\
									\
    speed_operand_src (s, s->xp, s->size);				\
    speed_operand_src (s, yp, (mp_size_t) N);				\
    speed_operand_dst (s, wp, wn);					\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      function (wp, s->xp, s->size, yp);				\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }

#define SPEED_ROUTINE_MPN_UNARY_2(function)				\
  SPEED_ROUTINE_MPN_UNARY_N (function, 2)
#define SPEED_ROUTINE_MPN_UNARY_3(function)				\
  SPEED_ROUTINE_MPN_UNARY_N (function, 3)
#define SPEED_ROUTINE_MPN_UNARY_4(function)				\
  SPEED_ROUTINE_MPN_UNARY_N (function, 4)
#define SPEED_ROUTINE_MPN_UNARY_5(function)				\
  SPEED_ROUTINE_MPN_UNARY_N (function, 5)
#define SPEED_ROUTINE_MPN_UNARY_6(function)				\
  SPEED_ROUTINE_MPN_UNARY_N (function, 6)
#define SPEED_ROUTINE_MPN_UNARY_7(function)				\
  SPEED_ROUTINE_MPN_UNARY_N (function, 7)
#define SPEED_ROUTINE_MPN_UNARY_8(function)				\
  SPEED_ROUTINE_MPN_UNARY_N (function, 8)


/* For mpn_mul, mpn_mul_basecase, xsize=r, ysize=s->size. */
#define SPEED_ROUTINE_MPN_MUL(function)					\
  {									\
    mp_ptr    wp;							\
    mp_size_t size1;							\
    unsigned  i;							\
    double    t;							\
    TMP_DECL;								\
									\
    size1 = (s->r == 0 ? s->size : s->r);				\
    if (size1 < 0) size1 = -size1 - s->size;				\
									\
    SPEED_RESTRICT_COND (size1 >= 1);					\
    SPEED_RESTRICT_COND (s->size >= size1);				\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (wp, size1 + s->size, s->align_wp);		\
									\
    speed_operand_src (s, s->xp, s->size);				\
    speed_operand_src (s, s->yp, size1);				\
    speed_operand_dst (s, wp, size1 + s->size);				\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      function (wp, s->xp, s->size, s->yp, size1);			\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }


#define SPEED_ROUTINE_MPN_MUL_N_CALL(call)				\
  {									\
    mp_ptr    wp;							\
    unsigned  i;							\
    double    t;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 1);					\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (wp, 2*s->size, s->align_wp);			\
									\
    speed_operand_src (s, s->xp, s->size);				\
    speed_operand_src (s, s->yp, s->size);				\
    speed_operand_dst (s, wp, 2*s->size);				\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      call;								\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }

#define SPEED_ROUTINE_MPN_MUL_N(function)				\
  SPEED_ROUTINE_MPN_MUL_N_CALL (function (wp, s->xp, s->yp, s->size));

#define SPEED_ROUTINE_MPN_MULLO_N_CALL(call)				\
  {									\
    mp_ptr    wp;							\
    unsigned  i;							\
    double    t;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 1);					\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (wp, s->size, s->align_wp);			\
									\
    speed_operand_src (s, s->xp, s->size);				\
    speed_operand_src (s, s->yp, s->size);				\
    speed_operand_dst (s, wp, s->size);					\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      call;								\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }

#define SPEED_ROUTINE_MPN_MULLO_N(function)				\
  SPEED_ROUTINE_MPN_MULLO_N_CALL (function (wp, s->xp, s->yp, s->size));

#define SPEED_ROUTINE_MPN_MULLO_BASECASE(function)			\
  SPEED_ROUTINE_MPN_MULLO_N_CALL (function (wp, s->xp, s->yp, s->size));

#define SPEED_ROUTINE_MPN_SQRLO(function)				\
  {									\
    mp_ptr    wp;							\
    unsigned  i;							\
    double    t;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 1);					\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (wp, s->size, s->align_wp);			\
									\
    speed_operand_src (s, s->xp, s->size);				\
    speed_operand_dst (s, wp, s->size);					\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      function (wp, s->xp, s->size);					\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }

/* For mpn_mulmid, mpn_mulmid_basecase, xsize=r, ysize=s->size. */
#define SPEED_ROUTINE_MPN_MULMID(function)				\
  {									\
    mp_ptr    wp, xp;							\
    mp_size_t size1;							\
    unsigned  i;							\
    double    t;							\
    TMP_DECL;								\
									\
    size1 = (s->r == 0 ? (2 * s->size - 1) : s->r);			\
									\
    SPEED_RESTRICT_COND (s->size >= 1);					\
    SPEED_RESTRICT_COND (size1 >= s->size);				\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (wp, size1 - s->size + 3, s->align_wp);	\
    SPEED_TMP_ALLOC_LIMBS (xp, size1, s->align_xp);			\
									\
    speed_operand_src (s, xp, size1);					\
    speed_operand_src (s, s->yp, s->size);				\
    speed_operand_dst (s, wp, size1 - s->size + 3);			\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      function (wp, xp, size1, s->yp, s->size);				\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }

#define SPEED_ROUTINE_MPN_MULMID_N(function)				\
  {									\
    mp_ptr    wp, xp;							\
    mp_size_t size1;							\
    unsigned  i;							\
    double    t;							\
    TMP_DECL;								\
									\
    size1 = 2 * s->size - 1;						\
									\
    SPEED_RESTRICT_COND (s->size >= 1);					\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (wp, size1 - s->size + 3, s->align_wp);	\
    SPEED_TMP_ALLOC_LIMBS (xp, size1, s->align_xp);			\
									\
    speed_operand_src (s, xp, size1);					\
    speed_operand_src (s, s->yp, s->size);				\
    speed_operand_dst (s, wp, size1 - s->size + 3);			\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      function (wp, xp, s->yp, s->size);				\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }

#define SPEED_ROUTINE_MPN_TOOM42_MULMID(function)			\
  {									\
    mp_ptr    wp, xp, scratch;						\
    mp_size_t size1, scratch_size;					\
    unsigned  i;							\
    double    t;							\
    TMP_DECL;								\
									\
    size1 = 2 * s->size - 1;						\
									\
    SPEED_RESTRICT_COND (s->size >= 1);					\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (wp, size1 - s->size + 3, s->align_wp);	\
    SPEED_TMP_ALLOC_LIMBS (xp, size1, s->align_xp);			\
    scratch_size = mpn_toom42_mulmid_itch (s->size);			\
    SPEED_TMP_ALLOC_LIMBS (scratch, scratch_size, 0);			\
									\
    speed_operand_src (s, xp, size1);					\
    speed_operand_src (s, s->yp, s->size);				\
    speed_operand_dst (s, wp, size1 - s->size + 3);			\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      function (wp, xp, s->yp, s->size, scratch);			\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }

#define SPEED_ROUTINE_MPN_MULMOD_BNM1_CALL(call)			\
  {									\
    mp_ptr    wp, tp;							\
    unsigned  i;							\
    double    t;							\
    mp_size_t itch;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 1);					\
									\
    itch = mpn_mulmod_bnm1_itch (s->size, s->size, s->size);		\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (wp, 2 * s->size, s->align_wp);		\
    SPEED_TMP_ALLOC_LIMBS (tp, itch, s->align_wp2);			\
									\
    speed_operand_src (s, s->xp, s->size);				\
    speed_operand_src (s, s->yp, s->size);				\
    speed_operand_dst (s, wp, 2 * s->size);				\
    speed_operand_dst (s, tp, itch);					\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      call;								\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }
#define SPEED_ROUTINE_MPN_MULMOD_BNM1_ROUNDED(function)			\
  {									\
    mp_ptr    wp, tp;							\
    unsigned  i;							\
    double    t;							\
    mp_size_t size, itch;						\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 1);					\
									\
    size = mpn_mulmod_bnm1_next_size (s->size);				\
    itch = mpn_mulmod_bnm1_itch (size, size, size);			\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (wp, size, s->align_wp);			\
    SPEED_TMP_ALLOC_LIMBS (tp, itch, s->align_wp2);			\
									\
    speed_operand_src (s, s->xp, s->size);				\
    speed_operand_src (s, s->yp, s->size);				\
    speed_operand_dst (s, wp, size);					\
    speed_operand_dst (s, tp, itch);					\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      function (wp, size, s->xp, s->size, s->yp, s->size, tp);		\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }

#define SPEED_ROUTINE_MPN_MUL_N_TSPACE(call, tsize, minsize)		\
  {									\
    mp_ptr    wp, tspace;						\
    unsigned  i;							\
    double    t;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= minsize);				\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (wp, 2*s->size, s->align_wp);			\
    SPEED_TMP_ALLOC_LIMBS (tspace, tsize, s->align_wp2);		\
									\
    speed_operand_src (s, s->xp, s->size);				\
    speed_operand_src (s, s->yp, s->size);				\
    speed_operand_dst (s, wp, 2*s->size);				\
    speed_operand_dst (s, tspace, tsize);				\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      call;								\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }

#define SPEED_ROUTINE_MPN_TOOM22_MUL_N(function)			\
  SPEED_ROUTINE_MPN_MUL_N_TSPACE					\
    (function (wp, s->xp, s->size, s->yp, s->size, tspace),		\
     mpn_toom22_mul_itch (s->size, s->size),				\
     MPN_TOOM22_MUL_MINSIZE)

#define SPEED_ROUTINE_MPN_TOOM33_MUL_N(function)			\
  SPEED_ROUTINE_MPN_MUL_N_TSPACE					\
    (function (wp, s->xp, s->size, s->yp, s->size, tspace),		\
     mpn_toom33_mul_itch (s->size, s->size),				\
     MPN_TOOM33_MUL_MINSIZE)

#define SPEED_ROUTINE_MPN_TOOM44_MUL_N(function)			\
  SPEED_ROUTINE_MPN_MUL_N_TSPACE					\
    (function (wp, s->xp, s->size, s->yp, s->size, tspace),		\
     mpn_toom44_mul_itch (s->size, s->size),				\
     MPN_TOOM44_MUL_MINSIZE)

#define SPEED_ROUTINE_MPN_TOOM6H_MUL_N(function)			\
  SPEED_ROUTINE_MPN_MUL_N_TSPACE					\
    (function (wp, s->xp, s->size, s->yp, s->size, tspace),		\
     mpn_toom6h_mul_itch (s->size, s->size),				\
     MPN_TOOM6H_MUL_MINSIZE)

#define SPEED_ROUTINE_MPN_TOOM8H_MUL_N(function)			\
  SPEED_ROUTINE_MPN_MUL_N_TSPACE					\
    (function (wp, s->xp, s->size, s->yp, s->size, tspace),		\
     mpn_toom8h_mul_itch (s->size, s->size),				\
     MPN_TOOM8H_MUL_MINSIZE)

#define SPEED_ROUTINE_MPN_TOOM32_MUL(function)				\
  SPEED_ROUTINE_MPN_MUL_N_TSPACE					\
    (function (wp, s->xp, s->size, s->yp, 2*s->size/3, tspace),		\
     mpn_toom32_mul_itch (s->size, 2*s->size/3),			\
     MPN_TOOM32_MUL_MINSIZE)

#define SPEED_ROUTINE_MPN_TOOM42_MUL(function)				\
  SPEED_ROUTINE_MPN_MUL_N_TSPACE					\
    (function (wp, s->xp, s->size, s->yp, s->size/2, tspace),		\
     mpn_toom42_mul_itch (s->size, s->size/2),				\
     MPN_TOOM42_MUL_MINSIZE)

#define SPEED_ROUTINE_MPN_TOOM43_MUL(function)				\
  SPEED_ROUTINE_MPN_MUL_N_TSPACE					\
    (function (wp, s->xp, s->size, s->yp, s->size*3/4, tspace),		\
     mpn_toom43_mul_itch (s->size, s->size*3/4),			\
     MPN_TOOM43_MUL_MINSIZE)

#define SPEED_ROUTINE_MPN_TOOM63_MUL(function)				\
  SPEED_ROUTINE_MPN_MUL_N_TSPACE					\
    (function (wp, s->xp, s->size, s->yp, s->size/2, tspace),		\
     mpn_toom63_mul_itch (s->size, s->size/2),				\
     MPN_TOOM63_MUL_MINSIZE)

#define SPEED_ROUTINE_MPN_TOOM32_FOR_TOOM43_MUL(function)		\
  SPEED_ROUTINE_MPN_MUL_N_TSPACE					\
    (function (wp, s->xp, s->size, s->yp, 17*s->size/24, tspace),	\
     mpn_toom32_mul_itch (s->size, 17*s->size/24),			\
     MPN_TOOM32_MUL_MINSIZE)
#define SPEED_ROUTINE_MPN_TOOM43_FOR_TOOM32_MUL(function)		\
  SPEED_ROUTINE_MPN_MUL_N_TSPACE					\
    (function (wp, s->xp, s->size, s->yp, 17*s->size/24, tspace),	\
     mpn_toom43_mul_itch (s->size, 17*s->size/24),			\
     MPN_TOOM43_MUL_MINSIZE)

#define SPEED_ROUTINE_MPN_TOOM32_FOR_TOOM53_MUL(function)		\
  SPEED_ROUTINE_MPN_MUL_N_TSPACE					\
    (function (wp, s->xp, s->size, s->yp, 19*s->size/30, tspace),	\
     mpn_toom32_mul_itch (s->size, 19*s->size/30),			\
     MPN_TOOM32_MUL_MINSIZE)
#define SPEED_ROUTINE_MPN_TOOM53_FOR_TOOM32_MUL(function)		\
  SPEED_ROUTINE_MPN_MUL_N_TSPACE					\
    (function (wp, s->xp, s->size, s->yp, 19*s->size/30, tspace),	\
     mpn_toom53_mul_itch (s->size, 19*s->size/30),			\
     MPN_TOOM53_MUL_MINSIZE)

#define SPEED_ROUTINE_MPN_TOOM42_FOR_TOOM53_MUL(function)		\
  SPEED_ROUTINE_MPN_MUL_N_TSPACE					\
    (function (wp, s->xp, s->size, s->yp, 11*s->size/20, tspace),	\
     mpn_toom42_mul_itch (s->size, 11*s->size/20),			\
     MPN_TOOM42_MUL_MINSIZE)
#define SPEED_ROUTINE_MPN_TOOM53_FOR_TOOM42_MUL(function)		\
  SPEED_ROUTINE_MPN_MUL_N_TSPACE					\
    (function (wp, s->xp, s->size, s->yp, 11*s->size/20, tspace),	\
     mpn_toom53_mul_itch (s->size, 11*s->size/20),			\
     MPN_TOOM53_MUL_MINSIZE)

#define SPEED_ROUTINE_MPN_TOOM43_FOR_TOOM54_MUL(function)		\
  SPEED_ROUTINE_MPN_MUL_N_TSPACE					\
    (function (wp, s->xp, s->size, s->yp, 5*s->size/6, tspace),	\
     mpn_toom42_mul_itch (s->size, 5*s->size/6),			\
     MPN_TOOM54_MUL_MINSIZE)
#define SPEED_ROUTINE_MPN_TOOM54_FOR_TOOM43_MUL(function)		\
  SPEED_ROUTINE_MPN_MUL_N_TSPACE					\
    (function (wp, s->xp, s->size, s->yp, 5*s->size/6, tspace),	\
     mpn_toom54_mul_itch (s->size, 5*s->size/6),			\
     MPN_TOOM54_MUL_MINSIZE)



#define SPEED_ROUTINE_MPN_SQR_CALL(call)				\
  {									\
    mp_ptr    wp;							\
    unsigned  i;							\
    double    t;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 1);					\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (wp, 2*s->size, s->align_wp);			\
									\
    speed_operand_src (s, s->xp, s->size);				\
    speed_operand_dst (s, wp, 2*s->size);				\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      call;								\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }

#define SPEED_ROUTINE_MPN_SQR(function)					\
  SPEED_ROUTINE_MPN_SQR_CALL (function (wp, s->xp, s->size))

#define SPEED_ROUTINE_MPN_SQR_DIAG_ADDLSH1_CALL(call)			\
  {									\
    mp_ptr    wp, tp;							\
    unsigned  i;							\
    double    t;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 2);					\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (tp, 2 * s->size, s->align_wp);		\
    SPEED_TMP_ALLOC_LIMBS (wp, 2 * s->size, s->align_wp);		\
									\
    speed_operand_src (s, s->xp, s->size);				\
    speed_operand_src (s, tp, 2 * s->size);				\
    speed_operand_dst (s, wp, 2 * s->size);				\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      call;								\
    while (--i != 0);							\
    t = speed_endtime () / 2;						\
									\
    TMP_FREE;								\
    return t;								\
  }

#define SPEED_ROUTINE_MPN_SQR_TSPACE(call, tsize, minsize)		\
  {									\
    mp_ptr    wp, tspace;						\
    unsigned  i;							\
    double    t;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= minsize);				\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (wp, 2*s->size, s->align_wp);			\
    SPEED_TMP_ALLOC_LIMBS (tspace, tsize, s->align_wp2);		\
									\
    speed_operand_src (s, s->xp, s->size);				\
    speed_operand_dst (s, wp, 2*s->size);				\
    speed_operand_dst (s, tspace, tsize);				\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      call;								\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }

#define SPEED_ROUTINE_MPN_TOOM2_SQR(function)				\
  SPEED_ROUTINE_MPN_SQR_TSPACE (function (wp, s->xp, s->size, tspace),	\
				mpn_toom2_sqr_itch (s->size),		\
				MPN_TOOM2_SQR_MINSIZE)

#define SPEED_ROUTINE_MPN_TOOM3_SQR(function)				\
  SPEED_ROUTINE_MPN_SQR_TSPACE (function (wp, s->xp, s->size, tspace),	\
				mpn_toom3_sqr_itch (s->size),		\
				MPN_TOOM3_SQR_MINSIZE)


#define SPEED_ROUTINE_MPN_TOOM4_SQR(function)				\
  SPEED_ROUTINE_MPN_SQR_TSPACE (function (wp, s->xp, s->size, tspace),	\
				mpn_toom4_sqr_itch (s->size),		\
				MPN_TOOM4_SQR_MINSIZE)

#define SPEED_ROUTINE_MPN_TOOM6_SQR(function)				\
  SPEED_ROUTINE_MPN_SQR_TSPACE (function (wp, s->xp, s->size, tspace),	\
				mpn_toom6_sqr_itch (s->size),		\
				MPN_TOOM6_SQR_MINSIZE)

#define SPEED_ROUTINE_MPN_TOOM8_SQR(function)				\
  SPEED_ROUTINE_MPN_SQR_TSPACE (function (wp, s->xp, s->size, tspace),	\
				mpn_toom8_sqr_itch (s->size),		\
				MPN_TOOM8_SQR_MINSIZE)

#define SPEED_ROUTINE_MPN_MOD_CALL(call)				\
  {									\
    unsigned   i;							\
									\
    SPEED_RESTRICT_COND (s->size >= 0);					\
									\
    speed_operand_src (s, s->xp, s->size);				\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      call;								\
    while (--i != 0);							\
									\
    return speed_endtime ();						\
  }

#define SPEED_ROUTINE_MPN_MOD_1(function)				\
   SPEED_ROUTINE_MPN_MOD_CALL ((*function) (s->xp, s->size, s->r))

#define SPEED_ROUTINE_MPN_MOD_1C(function)				\
   SPEED_ROUTINE_MPN_MOD_CALL ((*function)(s->xp, s->size, s->r, CNST_LIMB(0)))

#define SPEED_ROUTINE_MPN_MODEXACT_1_ODD(function)			\
  SPEED_ROUTINE_MPN_MOD_CALL (function (s->xp, s->size, s->r));

#define SPEED_ROUTINE_MPN_MODEXACT_1C_ODD(function)			\
  SPEED_ROUTINE_MPN_MOD_CALL (function (s->xp, s->size, s->r, CNST_LIMB(0)));

#define SPEED_ROUTINE_MPN_MOD_34LSUB1(function)				\
   SPEED_ROUTINE_MPN_MOD_CALL ((*function) (s->xp, s->size))

#define SPEED_ROUTINE_MPN_PREINV_MOD_1(function)			\
  {									\
    unsigned   i;							\
    mp_limb_t  inv;							\
									\
    SPEED_RESTRICT_COND (s->size >= 0);					\
    SPEED_RESTRICT_COND (s->r & GMP_LIMB_HIGHBIT);			\
									\
    invert_limb (inv, s->r);						\
    speed_operand_src (s, s->xp, s->size);				\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      (*function) (s->xp, s->size, s->r, inv);				\
    while (--i != 0);							\
									\
    return speed_endtime ();						\
  }

#define SPEED_ROUTINE_MPN_MOD_1_1(function,pfunc)			\
  {									\
    unsigned   i;							\
    mp_limb_t  inv[4];							\
									\
    SPEED_RESTRICT_COND (s->size >= 2);					\
									\
    mpn_mod_1_1p_cps (inv, s->r);					\
    speed_operand_src (s, s->xp, s->size);				\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do {								\
      pfunc (inv, s->r);						\
      function (s->xp, s->size, s->r << inv[1], inv);				\
    } while (--i != 0);							\
									\
    return speed_endtime ();						\
  }
#define SPEED_ROUTINE_MPN_MOD_1_N(function,pfunc,N)			\
  {									\
    unsigned   i;							\
    mp_limb_t  inv[N+3];						\
									\
    SPEED_RESTRICT_COND (s->size >= 1);					\
    SPEED_RESTRICT_COND (s->r <= ~(mp_limb_t)0 / N);			\
									\
    speed_operand_src (s, s->xp, s->size);				\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do {								\
      pfunc (inv, s->r);						\
      function (s->xp, s->size, s->r, inv);				\
    } while (--i != 0);							\
									\
    return speed_endtime ();						\
  }


/* A division of 2*s->size by s->size limbs */

#define SPEED_ROUTINE_MPN_DC_DIVREM_CALL(call)				\
  {									\
    unsigned  i;							\
    mp_ptr    a, d, q, r;						\
    double    t;							\
    gmp_pi1_t dinv;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 1);					\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (a, 2*s->size, s->align_xp);			\
    SPEED_TMP_ALLOC_LIMBS (d, s->size,   s->align_yp);			\
    SPEED_TMP_ALLOC_LIMBS (q, s->size+1, s->align_wp);			\
    SPEED_TMP_ALLOC_LIMBS (r, s->size,   s->align_wp2);			\
									\
    MPN_COPY (a, s->xp, s->size);					\
    MPN_COPY (a+s->size, s->xp, s->size);				\
									\
    MPN_COPY (d, s->yp, s->size);					\
									\
    /* normalize the data */						\
    d[s->size-1] |= GMP_NUMB_HIGHBIT;					\
    a[2*s->size-1] = d[s->size-1] - 1;					\
									\
    invert_pi1 (dinv, d[s->size-1], d[s->size-2]);			\
									\
    speed_operand_src (s, a, 2*s->size);				\
    speed_operand_src (s, d, s->size);					\
    speed_operand_dst (s, q, s->size+1);				\
    speed_operand_dst (s, r, s->size);					\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      call;								\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }


/* A remainder 2*s->size by s->size limbs */

#define SPEED_ROUTINE_MPZ_MOD(function)					\
  {									\
    unsigned   i;							\
    mpz_t      a, d, r;							\
									\
    SPEED_RESTRICT_COND (s->size >= 1);					\
									\
    mpz_init_set_n (d, s->yp, s->size);					\
									\
    /* high part less than d, low part a duplicate copied in */		\
    mpz_init_set_n (a, s->xp, s->size);					\
    mpz_mod (a, a, d);							\
    mpz_mul_2exp (a, a, GMP_LIMB_BITS * s->size);			\
    MPN_COPY (PTR(a), s->xp, s->size);					\
									\
    mpz_init (r);							\
									\
    speed_operand_src (s, PTR(a), SIZ(a));				\
    speed_operand_src (s, PTR(d), SIZ(d));				\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      function (r, a, d);						\
    while (--i != 0);							\
    return speed_endtime ();						\
  }

#define SPEED_ROUTINE_MPN_PI1_DIV(function, INV, DMIN, QMIN)		\
  {									\
    unsigned   i;							\
    mp_ptr     dp, tp, ap, qp;						\
    gmp_pi1_t  inv;							\
    double     t;							\
    mp_size_t size1;							\
    TMP_DECL;								\
									\
    size1 = (s->r == 0 ? 2 * s->size : s->r);				\
									\
    SPEED_RESTRICT_COND (s->size >= DMIN);				\
    SPEED_RESTRICT_COND (size1 - s->size >= QMIN);			\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (ap, size1, s->align_xp);			\
    SPEED_TMP_ALLOC_LIMBS (dp, s->size, s->align_yp);			\
    SPEED_TMP_ALLOC_LIMBS (qp, size1 - s->size, s->align_wp);		\
    SPEED_TMP_ALLOC_LIMBS (tp, size1, s->align_wp2);			\
									\
    /* we don't fill in dividend completely when size1 > s->size */	\
    MPN_COPY (ap,         s->xp, s->size);				\
    MPN_COPY (ap + size1 - s->size, s->xp, s->size);			\
									\
    MPN_COPY (dp,         s->yp, s->size);				\
									\
    /* normalize the data */						\
    dp[s->size-1] |= GMP_NUMB_HIGHBIT;					\
    ap[size1 - 1] = dp[s->size - 1] - 1;				\
									\
    invert_pi1 (inv, dp[s->size-1], dp[s->size-2]);			\
									\
    speed_operand_src (s, ap, size1);					\
    speed_operand_dst (s, tp, size1);					\
    speed_operand_src (s, dp, s->size);					\
    speed_operand_dst (s, qp, size1 - s->size);				\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do {								\
      MPN_COPY (tp, ap, size1);						\
      function (qp, tp, size1, dp, s->size, INV);			\
    } while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }
#define SPEED_ROUTINE_MPN_MU_DIV_Q(function,itchfn)			\
  {									\
    unsigned   i;							\
    mp_ptr     dp, tp, qp, scratch;					\
    double     t;							\
    mp_size_t itch;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 2);					\
									\
    itch = itchfn (2 * s->size, s->size, 0);				\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (dp, s->size, s->align_yp);			\
    SPEED_TMP_ALLOC_LIMBS (qp, s->size, s->align_wp);			\
    SPEED_TMP_ALLOC_LIMBS (tp, 2 * s->size, s->align_xp);		\
    SPEED_TMP_ALLOC_LIMBS (scratch, itch, s->align_wp2);		\
									\
    MPN_COPY (tp,         s->xp, s->size);				\
    MPN_COPY (tp+s->size, s->xp, s->size);				\
									\
    /* normalize the data */						\
    dp[s->size-1] |= GMP_NUMB_HIGHBIT;					\
    tp[2*s->size-1] = dp[s->size-1] - 1;				\
									\
    speed_operand_dst (s, qp, s->size);					\
    speed_operand_src (s, tp, 2 * s->size);				\
    speed_operand_src (s, dp, s->size);					\
    speed_operand_dst (s, scratch, itch);				\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do {								\
      function (qp, tp, 2 * s->size, dp, s->size, scratch);		\
    } while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }
#define SPEED_ROUTINE_MPN_MU_DIV_QR(function,itchfn)			\
  {									\
    unsigned   i;							\
    mp_ptr     dp, tp, qp, rp, scratch;					\
    double     t;							\
    mp_size_t size1, itch;						\
    TMP_DECL;								\
									\
    size1 = (s->r == 0 ? 2 * s->size : s->r);				\
									\
    SPEED_RESTRICT_COND (s->size >= 2);					\
    SPEED_RESTRICT_COND (size1 >= s->size);				\
									\
    itch = itchfn (size1, s->size, 0);					\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (dp, s->size, s->align_yp);			\
    SPEED_TMP_ALLOC_LIMBS (qp, size1 - s->size, s->align_wp);		\
    SPEED_TMP_ALLOC_LIMBS (tp, size1, s->align_xp);			\
    SPEED_TMP_ALLOC_LIMBS (scratch, itch, s->align_wp2);		\
    SPEED_TMP_ALLOC_LIMBS (rp, s->size, s->align_wp2); /* alignment? */	\
									\
    /* we don't fill in dividend completely when size1 > s->size */	\
    MPN_COPY (tp,         s->xp, s->size);				\
    MPN_COPY (tp + size1 - s->size, s->xp, s->size);			\
									\
    MPN_COPY (dp,         s->yp, s->size);				\
									\
    /* normalize the data */						\
    dp[s->size-1] |= GMP_NUMB_HIGHBIT;					\
    tp[size1 - 1] = dp[s->size - 1] - 1;				\
									\
    speed_operand_dst (s, qp, size1 - s->size);				\
    speed_operand_dst (s, rp, s->size);					\
    speed_operand_src (s, tp, size1);					\
    speed_operand_src (s, dp, s->size);					\
    speed_operand_dst (s, scratch, itch);				\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do {								\
      function (qp, rp, tp, size1, dp, s->size, scratch);		\
    } while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }
#define SPEED_ROUTINE_MPN_MUPI_DIV_QR(function,itchfn)			\
  {									\
    unsigned   i;							\
    mp_ptr     dp, tp, qp, rp, ip, scratch, tmp;			\
    double     t;							\
    mp_size_t  size1, itch;						\
    TMP_DECL;								\
									\
    size1 = (s->r == 0 ? 2 * s->size : s->r);				\
									\
    SPEED_RESTRICT_COND (s->size >= 2);					\
    SPEED_RESTRICT_COND (size1 >= s->size);				\
									\
    itch = itchfn (size1, s->size, s->size);				\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (dp, s->size, s->align_yp);			\
    SPEED_TMP_ALLOC_LIMBS (qp, size1 - s->size, s->align_wp);		\
    SPEED_TMP_ALLOC_LIMBS (tp, size1, s->align_xp);			\
    SPEED_TMP_ALLOC_LIMBS (scratch, itch, s->align_wp2);		\
    SPEED_TMP_ALLOC_LIMBS (rp, s->size, s->align_wp2); /* alignment? */	\
    SPEED_TMP_ALLOC_LIMBS (ip, s->size, s->align_wp2); /* alignment? */	\
									\
    /* we don't fill in dividend completely when size1 > s->size */	\
    MPN_COPY (tp,         s->xp, s->size);				\
    MPN_COPY (tp + size1 - s->size, s->xp, s->size);			\
									\
    MPN_COPY (dp,         s->yp, s->size);				\
									\
    /* normalize the data */						\
    dp[s->size-1] |= GMP_NUMB_HIGHBIT;					\
    tp[size1 - 1] = dp[s->size-1] - 1;					\
									\
    tmp = TMP_ALLOC_LIMBS (mpn_invert_itch (s->size));			\
    mpn_invert (ip, dp, s->size, tmp);					\
									\
    speed_operand_dst (s, qp, size1 - s->size);				\
    speed_operand_dst (s, rp, s->size);					\
    speed_operand_src (s, tp, size1);					\
    speed_operand_src (s, dp, s->size);					\
    speed_operand_src (s, ip, s->size);					\
    speed_operand_dst (s, scratch, itch);				\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do {								\
      function (qp, rp, tp, size1, dp, s->size, ip, s->size, scratch);	\
    } while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }

#define SPEED_ROUTINE_MPN_PI1_BDIV_QR(function)				\
  {									\
    unsigned   i;							\
    mp_ptr     dp, tp, ap, qp;						\
    mp_limb_t  inv;							\
    double     t;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 1);					\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (ap, 2*s->size, s->align_xp);			\
    SPEED_TMP_ALLOC_LIMBS (dp, s->size, s->align_yp);			\
    SPEED_TMP_ALLOC_LIMBS (qp, s->size, s->align_wp);			\
    SPEED_TMP_ALLOC_LIMBS (tp, 2*s->size, s->align_wp2);		\
									\
    MPN_COPY (ap,         s->xp, s->size);				\
    MPN_COPY (ap+s->size, s->xp, s->size);				\
									\
    /* divisor must be odd */						\
    MPN_COPY (dp, s->yp, s->size);					\
    dp[0] |= 1;								\
    binvert_limb (inv, dp[0]);						\
    inv = -inv;								\
									\
    speed_operand_src (s, ap, 2*s->size);				\
    speed_operand_dst (s, tp, 2*s->size);				\
    speed_operand_src (s, dp, s->size);					\
    speed_operand_dst (s, qp, s->size);					\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do {								\
      MPN_COPY (tp, ap, 2*s->size);					\
      function (qp, tp, 2*s->size, dp, s->size, inv);			\
    } while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }
#define SPEED_ROUTINE_MPN_PI1_BDIV_Q(function)				\
  {									\
    unsigned   i;							\
    mp_ptr     dp, tp, qp;						\
    mp_limb_t  inv;							\
    double     t;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 1);					\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (dp, s->size, s->align_yp);			\
    SPEED_TMP_ALLOC_LIMBS (qp, s->size, s->align_wp);			\
    SPEED_TMP_ALLOC_LIMBS (tp, s->size, s->align_wp2);			\
									\
    /* divisor must be odd */						\
    MPN_COPY (dp, s->yp, s->size);					\
    dp[0] |= 1;								\
    binvert_limb (inv, dp[0]);						\
    inv = -inv;								\
									\
    speed_operand_src (s, s->xp, s->size);				\
    speed_operand_dst (s, tp, s->size);					\
    speed_operand_src (s, dp, s->size);					\
    speed_operand_dst (s, qp, s->size);					\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do {								\
      MPN_COPY (tp, s->xp, s->size);					\
      function (qp, tp, s->size, dp, s->size, inv);			\
    } while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }
#define SPEED_ROUTINE_MPN_MU_BDIV_Q(function,itchfn)			\
  {									\
    unsigned   i;							\
    mp_ptr     dp, qp, scratch;						\
    double     t;							\
    mp_size_t itch;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 2);					\
									\
    itch = itchfn (s->size, s->size);					\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (dp, s->size, s->align_yp);			\
    SPEED_TMP_ALLOC_LIMBS (qp, s->size, s->align_wp);			\
    SPEED_TMP_ALLOC_LIMBS (scratch, itch, s->align_wp2);		\
									\
    /* divisor must be odd */						\
    MPN_COPY (dp, s->yp, s->size);					\
    dp[0] |= 1;								\
									\
    speed_operand_dst (s, qp, s->size);					\
    speed_operand_src (s, s->xp, s->size);				\
    speed_operand_src (s, dp, s->size);					\
    speed_operand_dst (s, scratch, itch);				\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do {								\
      function (qp, s->xp, s->size, dp, s->size, scratch);		\
    } while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }
#define SPEED_ROUTINE_MPN_MU_BDIV_QR(function,itchfn)			\
  {									\
    unsigned   i;							\
    mp_ptr     dp, tp, qp, rp, scratch;					\
    double     t;							\
    mp_size_t itch;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 2);					\
									\
    itch = itchfn (2 * s->size, s->size);				\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (dp, s->size, s->align_yp);			\
    SPEED_TMP_ALLOC_LIMBS (qp, s->size, s->align_wp);			\
    SPEED_TMP_ALLOC_LIMBS (tp, 2 * s->size, s->align_xp);		\
    SPEED_TMP_ALLOC_LIMBS (scratch, itch, s->align_wp2);		\
    SPEED_TMP_ALLOC_LIMBS (rp, s->size, s->align_wp2); /* alignment? */	\
									\
    MPN_COPY (tp,         s->xp, s->size);				\
    MPN_COPY (tp+s->size, s->xp, s->size);				\
									\
    /* divisor must be odd */						\
    MPN_COPY (dp, s->yp, s->size);					\
    dp[0] |= 1;								\
									\
    speed_operand_dst (s, qp, s->size);					\
    speed_operand_dst (s, rp, s->size);					\
    speed_operand_src (s, tp, 2 * s->size);				\
    speed_operand_src (s, dp, s->size);					\
    speed_operand_dst (s, scratch, itch);				\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do {								\
      function (qp, rp, tp, 2 * s->size, dp, s->size, scratch);		\
    } while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }

#define SPEED_ROUTINE_MPN_BROOT(function)	\
  {						\
    SPEED_RESTRICT_COND (s->r & 1);		\
    s->xp[0] |= 1;				\
    SPEED_ROUTINE_MPN_UNARY_1_CALL		\
      ((*function) (wp, s->xp, s->size, s->r));	\
  }

#define SPEED_ROUTINE_MPN_BROOTINV(function, itch)	\
  {							\
    mp_ptr    wp, tp;					\
    unsigned  i;					\
    double    t;					\
    TMP_DECL;						\
    TMP_MARK;						\
    SPEED_RESTRICT_COND (s->size >= 1);			\
    SPEED_RESTRICT_COND (s->r & 1);			\
    wp = TMP_ALLOC_LIMBS (s->size);			\
    tp = TMP_ALLOC_LIMBS ( (itch));			\
    s->xp[0] |= 1;					\
							\
    speed_operand_src (s, s->xp, s->size);		\
    speed_operand_dst (s, wp, s->size);			\
    speed_cache_fill (s);				\
							\
    speed_starttime ();					\
    i = s->reps;					\
    do							\
      (*function) (wp, s->xp, s->size, s->r, tp);	\
    while (--i != 0);					\
    t = speed_endtime ();				\
							\
    TMP_FREE;						\
    return t;						\
  }

#define SPEED_ROUTINE_MPN_INVERT(function,itchfn)			\
  {									\
    long  i;								\
    mp_ptr    up, tp, ip;						\
    double    t;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 1);					\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (ip, s->size, s->align_xp);			\
    SPEED_TMP_ALLOC_LIMBS (up, s->size,   s->align_yp);			\
    SPEED_TMP_ALLOC_LIMBS (tp, itchfn (s->size), s->align_wp);		\
									\
    MPN_COPY (up, s->xp, s->size);					\
									\
    /* normalize the data */						\
    up[s->size-1] |= GMP_NUMB_HIGHBIT;					\
									\
    speed_operand_src (s, up, s->size);					\
    speed_operand_dst (s, tp, s->size);					\
    speed_operand_dst (s, ip, s->size);					\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      function (ip, up, s->size, tp);					\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }

#define SPEED_ROUTINE_MPN_INVERTAPPR(function,itchfn)			\
  {									\
    long  i;								\
    mp_ptr    up, tp, ip;						\
    double    t;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 1);					\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (ip, s->size, s->align_xp);			\
    SPEED_TMP_ALLOC_LIMBS (up, s->size, s->align_yp);			\
    SPEED_TMP_ALLOC_LIMBS (tp, itchfn (s->size), s->align_wp);		\
									\
    MPN_COPY (up, s->xp, s->size);					\
									\
    /* normalize the data */						\
    up[s->size-1] |= GMP_NUMB_HIGHBIT;					\
									\
    speed_operand_src (s, up, s->size);					\
    speed_operand_dst (s, tp, s->size);					\
    speed_operand_dst (s, ip, s->size);					\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      function (ip, up, s->size, tp);					\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }

#define SPEED_ROUTINE_MPN_NI_INVERTAPPR(function,itchfn)		\
  {									\
    long  i;								\
    mp_ptr    up, tp, ip;						\
    double    t;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 3);					\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (ip, s->size, s->align_xp);			\
    SPEED_TMP_ALLOC_LIMBS (up, s->size, s->align_yp);			\
    SPEED_TMP_ALLOC_LIMBS (tp, itchfn (s->size), s->align_wp);		\
									\
    MPN_COPY (up, s->xp, s->size);					\
									\
    /* normalize the data */						\
    up[s->size-1] |= GMP_NUMB_HIGHBIT;					\
									\
    speed_operand_src (s, up, s->size);					\
    speed_operand_dst (s, tp, s->size);					\
    speed_operand_dst (s, ip, s->size);					\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      function (ip, up, s->size, tp);					\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }

#define SPEED_ROUTINE_MPN_BINVERT(function,itchfn)			\
  {									\
    long  i;								\
    mp_ptr    up, tp, ip;						\
    double    t;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 1);					\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (ip, s->size, s->align_xp);			\
    SPEED_TMP_ALLOC_LIMBS (up, s->size,   s->align_yp);			\
    SPEED_TMP_ALLOC_LIMBS (tp, itchfn (s->size), s->align_wp);		\
									\
    MPN_COPY (up, s->xp, s->size);					\
									\
    /* normalize the data */						\
    up[0] |= 1;								\
									\
    speed_operand_src (s, up, s->size);					\
    speed_operand_dst (s, tp, s->size);					\
    speed_operand_dst (s, ip, s->size);					\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      function (ip, up, s->size, tp);					\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }

#define SPEED_ROUTINE_MPN_SEC_INVERT(function,itchfn)			\
  {									\
    long  i;								\
    mp_ptr    up, mp, tp, ip;						\
    double    t;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 1);					\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (ip, s->size, s->align_xp);			\
    SPEED_TMP_ALLOC_LIMBS (up, s->size, s->align_yp);			\
    SPEED_TMP_ALLOC_LIMBS (mp, s->size, s->align_yp);			\
    SPEED_TMP_ALLOC_LIMBS (tp, itchfn (s->size), s->align_wp);		\
									\
    speed_operand_src (s, up, s->size);					\
    speed_operand_dst (s, tp, s->size);					\
    speed_operand_dst (s, ip, s->size);					\
    speed_cache_fill (s);						\
									\
    MPN_COPY (mp, s->yp, s->size);					\
    /* Must be odd */							\
    mp[0] |= 1;								\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      {									\
	MPN_COPY (up, s->xp, s->size);					\
	function (ip, up, mp, s->size, 2*s->size*GMP_NUMB_BITS, tp);	\
      }									\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }

#define SPEED_ROUTINE_REDC_1(function)					\
  {									\
    unsigned   i;							\
    mp_ptr     cp, mp, tp, ap;						\
    mp_limb_t  inv;							\
    double     t;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 1);					\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (ap, 2*s->size+1, s->align_xp);		\
    SPEED_TMP_ALLOC_LIMBS (mp, s->size,     s->align_yp);		\
    SPEED_TMP_ALLOC_LIMBS (cp, s->size,     s->align_wp);		\
    SPEED_TMP_ALLOC_LIMBS (tp, 2*s->size+1, s->align_wp2);		\
									\
    MPN_COPY (ap,         s->xp, s->size);				\
    MPN_COPY (ap+s->size, s->xp, s->size);				\
									\
    /* modulus must be odd */						\
    MPN_COPY (mp, s->yp, s->size);					\
    mp[0] |= 1;								\
    binvert_limb (inv, mp[0]);						\
    inv = -inv;								\
									\
    speed_operand_src (s, ap, 2*s->size+1);				\
    speed_operand_dst (s, tp, 2*s->size+1);				\
    speed_operand_src (s, mp, s->size);					\
    speed_operand_dst (s, cp, s->size);					\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do {								\
      MPN_COPY (tp, ap, 2*s->size);					\
      function (cp, tp, mp, s->size, inv);				\
    } while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }
#define SPEED_ROUTINE_REDC_2(function)					\
  {									\
    unsigned   i;							\
    mp_ptr     cp, mp, tp, ap;						\
    mp_limb_t  invp[2];							\
    double     t;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 1);					\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (ap, 2*s->size+1, s->align_xp);		\
    SPEED_TMP_ALLOC_LIMBS (mp, s->size,     s->align_yp);		\
    SPEED_TMP_ALLOC_LIMBS (cp, s->size,     s->align_wp);		\
    SPEED_TMP_ALLOC_LIMBS (tp, 2*s->size+1, s->align_wp2);		\
									\
    MPN_COPY (ap,         s->xp, s->size);				\
    MPN_COPY (ap+s->size, s->xp, s->size);				\
									\
    /* modulus must be odd */						\
    MPN_COPY (mp, s->yp, s->size);					\
    mp[0] |= 1;								\
    mpn_binvert (invp, mp, 2, tp);					\
    invp[0] = -invp[0]; invp[1] = ~invp[1];				\
									\
    speed_operand_src (s, ap, 2*s->size+1);				\
    speed_operand_dst (s, tp, 2*s->size+1);				\
    speed_operand_src (s, mp, s->size);					\
    speed_operand_dst (s, cp, s->size);					\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do {								\
      MPN_COPY (tp, ap, 2*s->size);					\
      function (cp, tp, mp, s->size, invp);				\
    } while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }
#define SPEED_ROUTINE_REDC_N(function)					\
  {									\
    unsigned   i;							\
    mp_ptr     cp, mp, tp, ap, invp;					\
    double     t;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size > 8);					\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (ap, 2*s->size+1, s->align_xp);		\
    SPEED_TMP_ALLOC_LIMBS (mp, s->size,     s->align_yp);		\
    SPEED_TMP_ALLOC_LIMBS (cp, s->size,     s->align_wp);		\
    SPEED_TMP_ALLOC_LIMBS (tp, 2*s->size+1, s->align_wp2);		\
    SPEED_TMP_ALLOC_LIMBS (invp, s->size,   s->align_wp2); /* align? */	\
									\
    MPN_COPY (ap,         s->xp, s->size);				\
    MPN_COPY (ap+s->size, s->xp, s->size);				\
									\
    /* modulus must be odd */						\
    MPN_COPY (mp, s->yp, s->size);					\
    mp[0] |= 1;								\
    mpn_binvert (invp, mp, s->size, tp);				\
									\
    speed_operand_src (s, ap, 2*s->size+1);				\
    speed_operand_dst (s, tp, 2*s->size+1);				\
    speed_operand_src (s, mp, s->size);					\
    speed_operand_dst (s, cp, s->size);					\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do {								\
      MPN_COPY (tp, ap, 2*s->size);					\
      function (cp, tp, mp, s->size, invp);				\
    } while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }


#define SPEED_ROUTINE_MPN_POPCOUNT(function)				\
  {									\
    unsigned i;								\
									\
    SPEED_RESTRICT_COND (s->size >= 1);					\
									\
    speed_operand_src (s, s->xp, s->size);				\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      function (s->xp, s->size);					\
    while (--i != 0);							\
									\
    return speed_endtime ();						\
  }

#define SPEED_ROUTINE_MPN_HAMDIST(function)				\
  {									\
    unsigned i;								\
									\
    SPEED_RESTRICT_COND (s->size >= 1);					\
									\
    speed_operand_src (s, s->xp, s->size);				\
    speed_operand_src (s, s->yp, s->size);				\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      function (s->xp, s->yp, s->size);					\
    while (--i != 0);							\
									\
    return speed_endtime ();						\
  }


#define SPEED_ROUTINE_MPZ_UI(function)					\
  {									\
    mpz_t     z;							\
    unsigned  i;							\
    double    t;							\
									\
    SPEED_RESTRICT_COND (s->size >= 0);					\
									\
    mpz_init (z);							\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      function (z, s->size);						\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    mpz_clear (z);							\
    return t;								\
  }

#define SPEED_ROUTINE_MPZ_FAC_UI(function)    SPEED_ROUTINE_MPZ_UI(function)
#define SPEED_ROUTINE_MPZ_FIB_UI(function)    SPEED_ROUTINE_MPZ_UI(function)
#define SPEED_ROUTINE_MPZ_LUCNUM_UI(function) SPEED_ROUTINE_MPZ_UI(function)


#define SPEED_ROUTINE_MPZ_2_UI(function)				\
  {									\
    mpz_t     z, z2;							\
    unsigned  i;							\
    double    t;							\
									\
    SPEED_RESTRICT_COND (s->size >= 0);					\
									\
    mpz_init (z);							\
    mpz_init (z2);							\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      function (z, z2, s->size);					\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    mpz_clear (z);							\
    mpz_clear (z2);							\
    return t;								\
  }

#define SPEED_ROUTINE_MPZ_FIB2_UI(function)    SPEED_ROUTINE_MPZ_2_UI(function)
#define SPEED_ROUTINE_MPZ_LUCNUM2_UI(function) SPEED_ROUTINE_MPZ_2_UI(function)


#define SPEED_ROUTINE_MPN_FIB2_UI(function)				\
  {									\
    mp_ptr     fp, f1p;							\
    mp_size_t  alloc;							\
    unsigned   i;							\
    double     t;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 0);					\
									\
    TMP_MARK;								\
    alloc = MPN_FIB2_SIZE (s->size);					\
    SPEED_TMP_ALLOC_LIMBS (fp,	alloc, s->align_xp);			\
    SPEED_TMP_ALLOC_LIMBS (f1p, alloc, s->align_yp);			\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      function (fp, f1p, s->size);					\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }



/* Calculate b^e mod m for random b and m of s->size limbs and random e of 6
   limbs.  m is forced to odd so that redc can be used.  e is limited in
   size so the calculation doesn't take too long. */
#define SPEED_ROUTINE_MPZ_POWM(function)				\
  {									\
    mpz_t     r, b, e, m;						\
    unsigned  i;							\
    double    t;							\
									\
    SPEED_RESTRICT_COND (s->size >= 1);					\
									\
    mpz_init (r);							\
    mpz_init_set_n (b, s->xp, s->size);					\
    mpz_init_set_n (m, s->yp, s->size);					\
    mpz_setbit (m, 0);	/* force m to odd */				\
    mpz_init_set_n (e, s->xp_block, 6);					\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      function (r, b, e, m);						\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    mpz_clear (r);							\
    mpz_clear (b);							\
    mpz_clear (e);							\
    mpz_clear (m);							\
    return t;								\
  }

/* (m-2)^0xAAAAAAAA mod m */
#define SPEED_ROUTINE_MPZ_POWM_UI(function)				\
  {									\
    mpz_t     r, b, m;							\
    unsigned  long  e;							\
    unsigned  i;							\
    double    t;							\
									\
    SPEED_RESTRICT_COND (s->size >= 1);					\
									\
    mpz_init (r);							\
									\
    /* force m to odd */						\
    mpz_init (m);							\
    mpz_set_n (m, s->xp, s->size);					\
    PTR(m)[0] |= 1;							\
									\
    e = (~ (unsigned long) 0) / 3;					\
    if (s->r != 0)							\
      e = s->r;								\
									\
    mpz_init_set (b, m);						\
    mpz_sub_ui (b, b, 2);						\
/* printf ("%X\n", mpz_get_ui(m)); */					\
    i = s->reps;							\
    speed_starttime ();							\
    do									\
      function (r, b, e, m);						\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    mpz_clear (r);							\
    mpz_clear (b);							\
    mpz_clear (m);							\
    return t;								\
  }


#define SPEED_ROUTINE_MPN_ADDSUB_CALL(call)				\
  {									\
    mp_ptr    wp, wp2, xp, yp;						\
    unsigned  i;							\
    double    t;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 0);					\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (wp,	s->size, s->align_wp);			\
    SPEED_TMP_ALLOC_LIMBS (wp2, s->size, s->align_wp2);			\
    xp = s->xp;								\
    yp = s->yp;								\
									\
    if (s->r == 0)	;						\
    else if (s->r == 1) { xp = wp;	      }				\
    else if (s->r == 2) {	    yp = wp2; }				\
    else if (s->r == 3) { xp = wp;  yp = wp2; }				\
    else if (s->r == 4) { xp = wp2; yp = wp;  }				\
    else {								\
      TMP_FREE;								\
      return -1.0;							\
    }									\
    if (xp != s->xp) MPN_COPY (xp, s->xp, s->size);			\
    if (yp != s->yp) MPN_COPY (yp, s->yp, s->size);			\
									\
    speed_operand_src (s, xp, s->size);					\
    speed_operand_src (s, yp, s->size);					\
    speed_operand_dst (s, wp, s->size);					\
    speed_operand_dst (s, wp2, s->size);				\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      call;								\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }

#define SPEED_ROUTINE_MPN_ADDSUB_N(function)				\
  SPEED_ROUTINE_MPN_ADDSUB_CALL						\
    (function (wp, wp2, xp, yp, s->size));

#define SPEED_ROUTINE_MPN_ADDSUB_NC(function)				\
  SPEED_ROUTINE_MPN_ADDSUB_CALL						\
    (function (wp, wp2, xp, yp, s->size, 0));


/* Doing an Nx1 gcd with the given r. */
#define SPEED_ROUTINE_MPN_GCD_1N(function)				\
  {									\
    mp_ptr    xp;							\
    unsigned  i;							\
    double    t;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 1);					\
    SPEED_RESTRICT_COND (s->r != 0);					\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (xp, s->size, s->align_xp);			\
    MPN_COPY (xp, s->xp, s->size);					\
    xp[0] |= refmpn_zero_p (xp, s->size);				\
									\
    speed_operand_src (s, s->xp, s->size);				\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      function (xp, s->size, s->r);					\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }


/* SPEED_BLOCK_SIZE many one GCDs of s->size bits each. */

#define SPEED_ROUTINE_MPN_GCD_1_CALL(setup, call)			\
  {									\
    unsigned  i, j;							\
    mp_ptr    px, py;							\
    mp_limb_t x_mask, y_mask;						\
    double    t;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 1);					\
    SPEED_RESTRICT_COND (s->size <= mp_bits_per_limb);			\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (px, SPEED_BLOCK_SIZE, s->align_xp);		\
    SPEED_TMP_ALLOC_LIMBS (py, SPEED_BLOCK_SIZE, s->align_yp);		\
    MPN_COPY (px, s->xp_block, SPEED_BLOCK_SIZE);			\
    MPN_COPY (py, s->yp_block, SPEED_BLOCK_SIZE);			\
									\
    x_mask = MP_LIMB_T_LOWBITMASK (s->size);				\
    y_mask = MP_LIMB_T_LOWBITMASK (s->r != 0 ? s->r : s->size);		\
    for (i = 0; i < SPEED_BLOCK_SIZE; i++)				\
      {									\
	px[i] &= x_mask; px[i] += (px[i] == 0);				\
	py[i] &= y_mask; py[i] += (py[i] == 0);				\
	setup;								\
      }									\
									\
    speed_operand_src (s, px, SPEED_BLOCK_SIZE);			\
    speed_operand_src (s, py, SPEED_BLOCK_SIZE);			\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      {									\
	j = SPEED_BLOCK_SIZE;						\
	do								\
	  {								\
	    call;							\
	  }								\
	while (--j != 0);						\
      }									\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
									\
    s->time_divisor = SPEED_BLOCK_SIZE;					\
    return t;								\
  }

#define SPEED_ROUTINE_MPN_GCD_1(function)				\
  SPEED_ROUTINE_MPN_GCD_1_CALL( , function (&px[j-1], 1, py[j-1]))

#define SPEED_ROUTINE_MPN_JACBASE(function)				\
  SPEED_ROUTINE_MPN_GCD_1_CALL						\
    ({									\
       /* require x<y, y odd, y!=1 */					\
       px[i] %= py[i];							\
       px[i] |= 1;							\
       py[i] |= 1;							\
       if (py[i]==1) py[i]=3;						\
     },									\
     function (px[j-1], py[j-1], 0))


#define SPEED_ROUTINE_MPN_HGCD_CALL(func, itchfunc)			\
  {									\
    mp_size_t hgcd_init_itch, hgcd_itch;				\
    mp_ptr ap, bp, wp, tmp1;						\
    struct hgcd_matrix hgcd;						\
    int res;								\
    unsigned i;								\
    double t;								\
    TMP_DECL;								\
									\
    if (s->size < 2)							\
      return -1;							\
									\
    TMP_MARK;								\
									\
    SPEED_TMP_ALLOC_LIMBS (ap, s->size + 1, s->align_xp);		\
    SPEED_TMP_ALLOC_LIMBS (bp, s->size + 1, s->align_yp);		\
									\
    s->xp[s->size - 1] |= 1;						\
    s->yp[s->size - 1] |= 1;						\
									\
    hgcd_init_itch = MPN_HGCD_MATRIX_INIT_ITCH (s->size);		\
    hgcd_itch = itchfunc (s->size);					\
									\
    SPEED_TMP_ALLOC_LIMBS (tmp1, hgcd_init_itch, s->align_wp);		\
    SPEED_TMP_ALLOC_LIMBS (wp, hgcd_itch, s->align_wp);			\
									\
    speed_operand_src (s, s->xp, s->size);				\
    speed_operand_src (s, s->yp, s->size);				\
    speed_operand_dst (s, ap, s->size + 1);				\
    speed_operand_dst (s, bp, s->size + 1);				\
    speed_operand_dst (s, wp, hgcd_itch);				\
    speed_operand_dst (s, tmp1, hgcd_init_itch);			\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      {									\
	MPN_COPY (ap, s->xp, s->size);					\
	MPN_COPY (bp, s->yp, s->size);					\
	mpn_hgcd_matrix_init (&hgcd, s->size, tmp1);			\
	res = func (ap, bp, s->size, &hgcd, wp);			\
      }									\
    while (--i != 0);							\
    t = speed_endtime ();						\
    TMP_FREE;								\
    return t;								\
  }

#define SPEED_ROUTINE_MPN_HGCD_REDUCE_CALL(func, itchfunc)		\
  {									\
    mp_size_t hgcd_init_itch, hgcd_step_itch;				\
    mp_ptr ap, bp, wp, tmp1;						\
    struct hgcd_matrix hgcd;						\
    mp_size_t p = s->size/2;						\
    int res;								\
    unsigned i;								\
    double t;								\
    TMP_DECL;								\
									\
    if (s->size < 2)							\
      return -1;							\
									\
    TMP_MARK;								\
									\
    SPEED_TMP_ALLOC_LIMBS (ap, s->size + 1, s->align_xp);		\
    SPEED_TMP_ALLOC_LIMBS (bp, s->size + 1, s->align_yp);		\
									\
    s->xp[s->size - 1] |= 1;						\
    s->yp[s->size - 1] |= 1;						\
									\
    hgcd_init_itch = MPN_HGCD_MATRIX_INIT_ITCH (s->size);		\
    hgcd_step_itch = itchfunc (s->size, p);				\
									\
    SPEED_TMP_ALLOC_LIMBS (tmp1, hgcd_init_itch, s->align_wp);		\
    SPEED_TMP_ALLOC_LIMBS (wp, hgcd_step_itch, s->align_wp);			\
									\
    speed_operand_src (s, s->xp, s->size);				\
    speed_operand_src (s, s->yp, s->size);				\
    speed_operand_dst (s, ap, s->size + 1);				\
    speed_operand_dst (s, bp, s->size + 1);				\
    speed_operand_dst (s, wp, hgcd_step_itch);				\
    speed_operand_dst (s, tmp1, hgcd_init_itch);			\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      {									\
	MPN_COPY (ap, s->xp, s->size);					\
	MPN_COPY (bp, s->yp, s->size);					\
	mpn_hgcd_matrix_init (&hgcd, s->size, tmp1);			\
	res = func (&hgcd, ap, bp, s->size, p, wp);			\
      }									\
    while (--i != 0);							\
    t = speed_endtime ();						\
    TMP_FREE;								\
    return t;								\
  }

/* Run some GCDs of s->size limbs each.  The number of different data values
   is decreased as s->size**2, since GCD is a quadratic algorithm.
   SPEED_ROUTINE_MPN_GCD runs more times than SPEED_ROUTINE_MPN_GCDEXT
   though, because the plain gcd is about twice as fast as gcdext.  */

#define SPEED_ROUTINE_MPN_GCD_CALL(datafactor, call)			\
  {									\
    unsigned  i;							\
    mp_size_t j, pieces, psize;						\
    mp_ptr    wp, wp2, xtmp, ytmp, px, py;				\
    double    t;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 1);					\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (xtmp, s->size+1, s->align_xp);		\
    SPEED_TMP_ALLOC_LIMBS (ytmp, s->size+1, s->align_yp);		\
    SPEED_TMP_ALLOC_LIMBS (wp,   s->size+1, s->align_wp);		\
    SPEED_TMP_ALLOC_LIMBS (wp2,  s->size+1, s->align_wp2);		\
									\
    pieces = SPEED_BLOCK_SIZE * datafactor / s->size / s->size;		\
    pieces = MIN (pieces, SPEED_BLOCK_SIZE / s->size);			\
    pieces = MAX (pieces, 1);						\
									\
    psize = pieces * s->size;						\
    px = TMP_ALLOC_LIMBS (psize);					\
    py = TMP_ALLOC_LIMBS (psize);					\
    MPN_COPY (px, pieces==1 ? s->xp : s->xp_block, psize);		\
    MPN_COPY (py, pieces==1 ? s->yp : s->yp_block, psize);		\
									\
    /* Requirements: x >= y, y must be odd, high limbs != 0.		\
       No need to ensure random numbers are really great.  */		\
    for (j = 0; j < pieces; j++)					\
      {									\
	mp_ptr	x = px + j * s->size;					\
	mp_ptr	y = py + j * s->size;					\
	if (x[s->size - 1] == 0) x[s->size - 1] = 1;			\
	if (y[s->size - 1] == 0) y[s->size - 1] = 1;			\
									\
	if (x[s->size - 1] < y[s->size - 1])				\
	  MP_LIMB_T_SWAP (x[s->size - 1], y[s->size - 1]);		\
	else if (x[s->size - 1] == y[s->size - 1])			\
	  {								\
	    x[s->size - 1] = 2;						\
	    y[s->size - 1] = 1;						\
	  }								\
	y[0] |= 1;							\
      }									\
									\
    speed_operand_src (s, px, psize);					\
    speed_operand_src (s, py, psize);					\
    speed_operand_dst (s, xtmp, s->size);				\
    speed_operand_dst (s, ytmp, s->size);				\
    speed_operand_dst (s, wp, s->size);					\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      {									\
	j = pieces;							\
	do								\
	  {								\
	    MPN_COPY (xtmp, px+(j - 1)*s->size, s->size);		\
	    MPN_COPY (ytmp, py+(j - 1)*s->size, s->size);		\
	    call;							\
	  }								\
	while (--j != 0);						\
      }									\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
									\
    s->time_divisor = pieces;						\
    return t;								\
  }

#define SPEED_ROUTINE_MPN_GCD(function)	\
  SPEED_ROUTINE_MPN_GCD_CALL (8, function (wp, xtmp, s->size, ytmp, s->size))

#define SPEED_ROUTINE_MPN_GCDEXT(function)				\
  SPEED_ROUTINE_MPN_GCD_CALL						\
    (4, { mp_size_t  wp2size;						\
	  function (wp, wp2, &wp2size, xtmp, s->size, ytmp, s->size); })


#define SPEED_ROUTINE_MPN_GCDEXT_ONE(function)				\
  {									\
    unsigned  i;							\
    mp_size_t j, pieces, psize, wp2size;				\
    mp_ptr    wp, wp2, xtmp, ytmp, px, py;				\
    double    t;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 1);					\
									\
    TMP_MARK;								\
									\
    SPEED_TMP_ALLOC_LIMBS (xtmp, s->size+1, s->align_xp);		\
    SPEED_TMP_ALLOC_LIMBS (ytmp, s->size+1, s->align_yp);		\
    MPN_COPY (xtmp, s->xp, s->size);					\
    MPN_COPY (ytmp, s->yp, s->size);					\
									\
    SPEED_TMP_ALLOC_LIMBS (wp,	s->size+1, s->align_wp);		\
    SPEED_TMP_ALLOC_LIMBS (wp2, s->size+1, s->align_wp2);		\
									\
    pieces = SPEED_BLOCK_SIZE / 3;					\
    psize = 3 * pieces;							\
    px = TMP_ALLOC_LIMBS (psize);					\
    py = TMP_ALLOC_LIMBS (psize);					\
    MPN_COPY (px, s->xp_block, psize);					\
    MPN_COPY (py, s->yp_block, psize);					\
									\
    /* x must have at least as many bits as y,				\
       high limbs must be non-zero */					\
    for (j = 0; j < pieces; j++)					\
      {									\
	mp_ptr	x = px+3*j;						\
	mp_ptr	y = py+3*j;						\
	x[2] += (x[2] == 0);						\
	y[2] += (y[2] == 0);						\
	if (x[2] < y[2])						\
	  MP_LIMB_T_SWAP (x[2], y[2]);					\
      }									\
									\
    speed_operand_src (s, px, psize);					\
    speed_operand_src (s, py, psize);					\
    speed_operand_dst (s, xtmp, s->size);				\
    speed_operand_dst (s, ytmp, s->size);				\
    speed_operand_dst (s, wp, s->size);					\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      {									\
	mp_ptr	x = px;							\
	mp_ptr	y = py;							\
	mp_ptr	xth = &xtmp[s->size-3];					\
	mp_ptr	yth = &ytmp[s->size-3];					\
	j = pieces;							\
	do								\
	  {								\
	    xth[0] = x[0], xth[1] = x[1], xth[2] = x[2];		\
	    yth[0] = y[0], yth[1] = y[1], yth[2] = y[2];		\
									\
	    ytmp[0] |= 1; /* y must be odd, */				\
									\
	    function (wp, wp2, &wp2size, xtmp, s->size, ytmp, s->size);	\
									\
	    x += 3;							\
	    y += 3;							\
	  }								\
	while (--j != 0);						\
      }									\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
									\
    s->time_divisor = pieces;						\
    return t;								\
  }

#define SPEED_ROUTINE_MPZ_JACOBI(function)				\
  {									\
    mpz_t     a, b;							\
    unsigned  i;							\
    mp_size_t j, pieces, psize;						\
    mp_ptr    px, py;							\
    double    t;							\
    TMP_DECL;								\
									\
    TMP_MARK;								\
    pieces = SPEED_BLOCK_SIZE / MAX (s->size, 1);			\
    pieces = MAX (pieces, 1);						\
    s->time_divisor = pieces;						\
									\
    psize = pieces * s->size;						\
    px = TMP_ALLOC_LIMBS (psize);					\
    py = TMP_ALLOC_LIMBS (psize);					\
    MPN_COPY (px, pieces==1 ? s->xp : s->xp_block, psize);		\
    MPN_COPY (py, pieces==1 ? s->yp : s->yp_block, psize);		\
									\
    for (j = 0; j < pieces; j++)					\
      {									\
	mp_ptr	x = px+j*s->size;					\
	mp_ptr	y = py+j*s->size;					\
									\
	/* y odd */							\
	y[0] |= 1;							\
									\
	/* high limbs non-zero */					\
	if (x[s->size-1] == 0) x[s->size-1] = 1;			\
	if (y[s->size-1] == 0) y[s->size-1] = 1;			\
      }									\
									\
    SIZ(a) = s->size;							\
    SIZ(b) = s->size;							\
									\
    speed_operand_src (s, px, psize);					\
    speed_operand_src (s, py, psize);					\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      {									\
	j = pieces;							\
	do								\
	  {								\
	    PTR(a) = px+(j-1)*s->size;					\
	    PTR(b) = py+(j-1)*s->size;					\
	    function (a, b);						\
	  }								\
	while (--j != 0);						\
      }									\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }

#define SPEED_ROUTINE_MPN_DIVREM_2(function)				\
  {									\
    mp_ptr    wp, xp;							\
    mp_limb_t yp[2];							\
    unsigned  i;							\
    double    t;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 2);					\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (xp, s->size, s->align_xp);			\
    SPEED_TMP_ALLOC_LIMBS (wp, s->size, s->align_wp);			\
									\
    /* source is destroyed */						\
    MPN_COPY (xp, s->xp, s->size);					\
									\
    /* divisor must be normalized */					\
    MPN_COPY (yp, s->yp_block, 2);					\
    yp[1] |= GMP_NUMB_HIGHBIT;						\
									\
    speed_operand_src (s, xp, s->size);					\
    speed_operand_src (s, yp, 2);					\
    speed_operand_dst (s, wp, s->size);					\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      function (wp, 0, xp, s->size, yp);				\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }

#define SPEED_ROUTINE_MPN_DIV_QR_1(function)				\
  {									\
    mp_ptr    wp, xp;							\
    mp_limb_t d;							\
    mp_limb_t r;							\
    unsigned  i;							\
    double    t;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 1);					\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (wp, s->size, s->align_wp);			\
									\
    d = s->r;								\
    if (d == 0)								\
      d = 1;								\
    speed_operand_src (s, s->xp, s->size);				\
    speed_operand_dst (s, wp, s->size);					\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      r = function (wp, wp+s->size-1, s->xp, s->size, d);		\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }

#define SPEED_ROUTINE_MPN_DIV_QR_1N_PI1(function)			\
  {									\
    mp_ptr    wp, xp;							\
    mp_limb_t d, dinv;							\
    mp_limb_t r;							\
    unsigned  i;							\
    double    t;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 1);					\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (wp, s->size, s->align_wp);			\
									\
    d = s->r;								\
    /* divisor must be normalized */					\
    SPEED_RESTRICT_COND (d & GMP_NUMB_HIGHBIT);				\
    invert_limb (dinv, d);						\
    speed_operand_src (s, s->xp, s->size);				\
    speed_operand_dst (s, wp, s->size);					\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      r = function (wp, s->xp, s->size, 0, d, dinv);			\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }

#define SPEED_ROUTINE_MPN_DIV_QR_2(function, norm)			\
  {									\
    mp_ptr    wp, xp;							\
    mp_limb_t yp[2];							\
    mp_limb_t rp[2];							\
    unsigned  i;							\
    double    t;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 2);					\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (wp, s->size, s->align_wp);			\
									\
    /* divisor must be normalized */					\
    MPN_COPY (yp, s->yp_block, 2);					\
    if (norm)								\
      yp[1] |= GMP_NUMB_HIGHBIT;					\
    else								\
      {									\
	yp[1] &= ~GMP_NUMB_HIGHBIT;					\
	if (yp[1] == 0)							\
	  yp[1] = 1;							\
      }									\
    speed_operand_src (s, s->xp, s->size);				\
    speed_operand_src (s, yp, 2);					\
    speed_operand_dst (s, wp, s->size);					\
    speed_operand_dst (s, rp, 2);					\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      function (wp, rp, s->xp, s->size, yp);				\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }

#define SPEED_ROUTINE_MODLIMB_INVERT(function)				\
  {									\
    unsigned   i, j;							\
    mp_ptr     xp;							\
    mp_limb_t  n = 1;							\
    double     t;							\
									\
    xp = s->xp_block-1;							\
									\
    speed_operand_src (s, s->xp_block, SPEED_BLOCK_SIZE);		\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      {									\
	j = SPEED_BLOCK_SIZE;						\
	do								\
	  {								\
	    /* randomized but successively dependent */			\
	    n += (xp[j] << 1);						\
									\
	    function (n, n);						\
	  }								\
	while (--j != 0);						\
      }									\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    /* make sure the compiler won't optimize away n */			\
    noop_1 (n);								\
									\
    s->time_divisor = SPEED_BLOCK_SIZE;					\
    return t;								\
  }


#define SPEED_ROUTINE_MPN_SQRTROOT_CALL(call)				\
  {									\
    mp_ptr    wp, wp2;							\
    unsigned  i;							\
    double    t;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 1);					\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (wp,	s->size, s->align_wp);			\
    SPEED_TMP_ALLOC_LIMBS (wp2, s->size, s->align_wp2);			\
									\
    speed_operand_src (s, s->xp, s->size);				\
    speed_operand_dst (s, wp, s->size);					\
    speed_operand_dst (s, wp2, s->size);				\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      call;								\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }


/* s->size controls the number of limbs in the input, s->r is the base, or
   decimal by default. */
#define SPEED_ROUTINE_MPN_GET_STR(function)				\
  {									\
    unsigned char *wp;							\
    mp_size_t wn;							\
    mp_ptr xp;								\
    int base;								\
    unsigned i;								\
    double t;								\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 1);					\
									\
    base = s->r == 0 ? 10 : s->r;					\
    SPEED_RESTRICT_COND (base >= 2 && base <= 256);			\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (xp, s->size + 1, s->align_xp);		\
									\
    MPN_SIZEINBASE (wn, s->xp, s->size, base);				\
    wp = (unsigned char *) TMP_ALLOC (wn);				\
									\
    /* use this during development to guard against overflowing wp */	\
    /*									\
    MPN_COPY (xp, s->xp, s->size);					\
    ASSERT_ALWAYS (mpn_get_str (wp, base, xp, s->size) <= wn);		\
    */									\
									\
    speed_operand_src (s, s->xp, s->size);				\
    speed_operand_dst (s, xp, s->size);					\
    speed_operand_dst (s, (mp_ptr) wp, wn/GMP_LIMB_BYTES);		\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      {									\
	MPN_COPY (xp, s->xp, s->size);					\
	function (wp, base, xp, s->size);				\
      }									\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }

/* s->size controls the number of digits in the input, s->r is the base, or
   decimal by default. */
#define SPEED_ROUTINE_MPN_SET_STR_CALL(call)				\
  {									\
    unsigned char *xp;							\
    mp_ptr     wp;							\
    mp_size_t  wn;							\
    unsigned   i;							\
    int        base;							\
    double     t;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 1);					\
									\
    base = s->r == 0 ? 10 : s->r;					\
    SPEED_RESTRICT_COND (base >= 2 && base <= 256);			\
									\
    TMP_MARK;								\
									\
    xp = (unsigned char *) TMP_ALLOC (s->size);				\
    for (i = 0; i < s->size; i++)					\
      xp[i] = s->xp[i] % base;						\
									\
    LIMBS_PER_DIGIT_IN_BASE (wn, s->size, base);			\
    SPEED_TMP_ALLOC_LIMBS (wp, wn, s->align_wp);			\
									\
    /* use this during development to check wn is big enough */		\
    /*									\
    ASSERT_ALWAYS (mpn_set_str (wp, xp, s->size, base) <= wn);		\
    */									\
									\
    speed_operand_src (s, (mp_ptr) xp, s->size/GMP_LIMB_BYTES);	\
    speed_operand_dst (s, wp, wn);					\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      call;								\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }


/* Run an accel gcd find_a() function over various data values.  A set of
   values is used in case some run particularly fast or slow.  The size
   parameter is ignored, the amount of data tested is fixed.  */

#define SPEED_ROUTINE_MPN_GCD_FINDA(function)				\
  {									\
    unsigned  i, j;							\
    mp_limb_t cp[SPEED_BLOCK_SIZE][2];					\
    double    t;							\
    TMP_DECL;								\
									\
    TMP_MARK;								\
									\
    /* low must be odd, high must be non-zero */			\
    for (i = 0; i < SPEED_BLOCK_SIZE; i++)				\
      {									\
	cp[i][0] = s->xp_block[i] | 1;					\
	cp[i][1] = s->yp_block[i] + (s->yp_block[i] == 0);		\
      }									\
									\
    speed_operand_src (s, &cp[0][0], 2*SPEED_BLOCK_SIZE);		\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      {									\
	j = SPEED_BLOCK_SIZE;						\
	do								\
	  {								\
	    function (cp[j-1]);						\
	  }								\
	while (--j != 0);						\
      }									\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
									\
    s->time_divisor = SPEED_BLOCK_SIZE;					\
    return t;								\
  }


/* "call" should do "count_foo_zeros(c,n)".
   Give leading=1 if foo is leading zeros, leading=0 for trailing.
   Give zero=1 if n=0 is allowed in the call, zero=0 if not.  */

#define SPEED_ROUTINE_COUNT_ZEROS_A(leading, zero)			\
  {									\
    mp_ptr     xp;							\
    int        i, c;							\
    unsigned   j;							\
    mp_limb_t  n;							\
    double     t;							\
    TMP_DECL;								\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (xp, SPEED_BLOCK_SIZE, s->align_xp);		\
									\
    if (! speed_routine_count_zeros_setup (s, xp, leading, zero))	\
      return -1.0;							\
    speed_operand_src (s, xp, SPEED_BLOCK_SIZE);			\
    speed_cache_fill (s);						\
									\
    c = 0;								\
    speed_starttime ();							\
    j = s->reps;							\
    do {								\
      for (i = 0; i < SPEED_BLOCK_SIZE; i++)				\
	{								\
	  n = xp[i];							\
	  n ^= c;							\

#define SPEED_ROUTINE_COUNT_ZEROS_B()					\
	}								\
    } while (--j != 0);							\
    t = speed_endtime ();						\
									\
    /* don't let c go dead */						\
    noop_1 (c);								\
									\
    s->time_divisor = SPEED_BLOCK_SIZE;					\
									\
    TMP_FREE;								\
    return t;								\
  }									\

#define SPEED_ROUTINE_COUNT_ZEROS_C(call, leading, zero)		\
  do {									\
    SPEED_ROUTINE_COUNT_ZEROS_A (leading, zero);			\
    call;								\
    SPEED_ROUTINE_COUNT_ZEROS_B ();					\
  } while (0)								\

#define SPEED_ROUTINE_COUNT_LEADING_ZEROS_C(call,zero)			\
  SPEED_ROUTINE_COUNT_ZEROS_C (call, 1, zero)
#define SPEED_ROUTINE_COUNT_LEADING_ZEROS(fun)				\
  SPEED_ROUTINE_COUNT_ZEROS_C (fun (c, n), 1, 0)

#define SPEED_ROUTINE_COUNT_TRAILING_ZEROS_C(call,zero)			\
  SPEED_ROUTINE_COUNT_ZEROS_C (call, 0, zero)
#define SPEED_ROUTINE_COUNT_TRAILING_ZEROS(call)			\
  SPEED_ROUTINE_COUNT_ZEROS_C (fun (c, n), 0, 0)


#define SPEED_ROUTINE_INVERT_LIMB_CALL(call)				\
  {									\
    unsigned   i, j;							\
    mp_limb_t  d, dinv=0;						\
    mp_ptr     xp = s->xp_block - 1;					\
									\
    s->time_divisor = SPEED_BLOCK_SIZE;					\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      {									\
	j = SPEED_BLOCK_SIZE;						\
	do								\
	  {								\
	    d = dinv ^ xp[j];						\
	    d |= GMP_LIMB_HIGHBIT;					\
	    do { call; } while (0);					\
	  }								\
	while (--j != 0);						\
      }									\
    while (--i != 0);							\
									\
    /* don't let the compiler optimize everything away */		\
    noop_1 (dinv);							\
									\
    return speed_endtime();						\
  }


#define SPEED_ROUTINE_MPN_BACK_TO_BACK(function)			\
  {									\
    unsigned  i;							\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      function ();							\
    while (--i != 0);							\
    return speed_endtime ();						\
  }


#define SPEED_ROUTINE_MPN_ZERO_CALL(call)				\
  {									\
    mp_ptr    wp;							\
    unsigned  i;							\
    double    t;							\
    TMP_DECL;								\
									\
    SPEED_RESTRICT_COND (s->size >= 0);					\
									\
    TMP_MARK;								\
    SPEED_TMP_ALLOC_LIMBS (wp, s->size, s->align_wp);			\
    speed_operand_dst (s, wp, s->size);					\
    speed_cache_fill (s);						\
									\
    speed_starttime ();							\
    i = s->reps;							\
    do									\
      call;								\
    while (--i != 0);							\
    t = speed_endtime ();						\
									\
    TMP_FREE;								\
    return t;								\
  }

#define SPEED_ROUTINE_MPN_ZERO(function)				\
  SPEED_ROUTINE_MPN_ZERO_CALL (function (wp, s->size))


#endif
