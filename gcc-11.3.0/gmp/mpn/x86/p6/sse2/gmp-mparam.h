/* Intel P6/sse2 gmp-mparam.h -- Compiler/machine parameter header file.

Copyright 1991, 1993, 1994, 1999-2003, 2008-2010 Free Software Foundation, Inc.

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


#define GMP_LIMB_BITS 32
#define GMP_LIMB_BYTES 4


/* NOTE: In a fat binary build SQR_TOOM2_THRESHOLD here cannot be more than the
   value in mpn/x86/p6/gmp-mparam.h.  The latter is used as a hard limit in
   mpn/x86/p6/sqr_basecase.asm.  */


/* 1867 MHz P6 model 13 */

#define MOD_1_NORM_THRESHOLD                 4
#define MOD_1_UNNORM_THRESHOLD               4
#define MOD_1N_TO_MOD_1_1_THRESHOLD          5
#define MOD_1U_TO_MOD_1_1_THRESHOLD          4
#define MOD_1_1_TO_MOD_1_2_THRESHOLD        11
#define MOD_1_2_TO_MOD_1_4_THRESHOLD         0  /* never mpn_mod_1s_2p */
#define PREINV_MOD_1_TO_MOD_1_THRESHOLD      8
#define USE_PREINV_DIVREM_1                  1  /* native */
#define DIV_QR_2_PI2_THRESHOLD           MP_SIZE_T_MAX  /* never */
#define DIVEXACT_1_THRESHOLD                 0  /* always (native) */
#define BMOD_1_TO_MOD_1_THRESHOLD           21

#define MUL_TOOM22_THRESHOLD                20
#define MUL_TOOM33_THRESHOLD                77
#define MUL_TOOM44_THRESHOLD               169
#define MUL_TOOM6H_THRESHOLD               246
#define MUL_TOOM8H_THRESHOLD               381

#define MUL_TOOM32_TO_TOOM43_THRESHOLD      73
#define MUL_TOOM32_TO_TOOM53_THRESHOLD     114
#define MUL_TOOM42_TO_TOOM53_THRESHOLD      97
#define MUL_TOOM42_TO_TOOM63_THRESHOLD      80
#define MUL_TOOM43_TO_TOOM54_THRESHOLD     106

#define SQR_BASECASE_THRESHOLD               0  /* always (native) */
#define SQR_TOOM2_THRESHOLD                 30
#define SQR_TOOM3_THRESHOLD                101
#define SQR_TOOM4_THRESHOLD                154
#define SQR_TOOM6_THRESHOLD                222
#define SQR_TOOM8_THRESHOLD                527

#define MULMID_TOOM42_THRESHOLD             58

#define MULMOD_BNM1_THRESHOLD               13
#define SQRMOD_BNM1_THRESHOLD               17

#define MUL_FFT_MODF_THRESHOLD             690  /* k = 5 */
#define MUL_FFT_TABLE3                                      \
  { {    565, 5}, {     25, 6}, {     13, 5}, {     27, 6}, \
    {     25, 7}, {     13, 6}, {     28, 7}, {     15, 6}, \
    {     31, 7}, {     17, 6}, {     35, 7}, {     27, 8}, \
    {     15, 7}, {     35, 8}, {     19, 7}, {     41, 8}, \
    {     23, 7}, {     47, 8}, {     27, 9}, {     15, 8}, \
    {     31, 7}, {     63, 8}, {     39, 9}, {     23, 5}, \
    {    383, 4}, {    991, 5}, {    511, 6}, {    267, 7}, \
    {    157, 8}, {     91, 9}, {     47, 8}, {    111, 9}, \
    {     63, 8}, {    127, 9}, {     79,10}, {     47, 9}, \
    {     95,11}, {     31,10}, {     63, 9}, {    135,10}, \
    {     79, 9}, {    159,10}, {     95,11}, {     63,10}, \
    {    143, 9}, {    287,10}, {    159,11}, {     95,10}, \
    {    191,12}, {     63,11}, {    127,10}, {    255, 9}, \
    {    511,10}, {    271, 9}, {    543,10}, {    287,11}, \
    {    159,10}, {    335, 9}, {    671,11}, {    191,10}, \
    {    383, 9}, {    767,10}, {    399, 9}, {    799,10}, \
    {    415,11}, {    223,12}, {    127,11}, {    255,10}, \
    {    543, 9}, {   1087,11}, {    287,10}, {    607,11}, \
    {    319,10}, {    671,12}, {    191,11}, {    383,10}, \
    {    799,11}, {    415,10}, {    831,13}, {    127,12}, \
    {    255,11}, {    543,10}, {   1087,11}, {    607,10}, \
    {   1215,12}, {    319,11}, {    671,10}, {   1343,11}, \
    {    735,10}, {   1471,12}, {    383,11}, {    799,10}, \
    {   1599,11}, {    863,12}, {    447,11}, {    959,13}, \
    {    255,12}, {    511,11}, {   1087,12}, {    575,11}, \
    {   1215,12}, {    639,11}, {   1343,12}, {    703,11}, \
    {   1471,13}, {    383,12}, {    831,11}, {   1727,12}, \
    {    959,14}, {    255,13}, {    511,12}, {   1215,13}, \
    {    639,12}, {   1471,11}, {   2943,13}, {    767,12}, \
    {   1727,13}, {    895,12}, {   1919,14}, {    511,13}, \
    {   1023,12}, {   2111,13}, {   1151,12}, {   2431,13}, \
    {   1407,12}, {   2815,14}, {    767,13}, {   1663,12}, \
    {   3455,13}, {   8192,14}, {  16384,15}, {  32768,16} }
#define MUL_FFT_TABLE3_SIZE 132
#define MUL_FFT_THRESHOLD                 7424

#define SQR_FFT_MODF_THRESHOLD             565  /* k = 5 */
#define SQR_FFT_TABLE3                                      \
  { {    472, 5}, {     25, 6}, {     13, 5}, {     27, 6}, \
    {     25, 7}, {     13, 6}, {     27, 7}, {     15, 6}, \
    {     31, 7}, {     17, 6}, {     35, 7}, {     27, 8}, \
    {     15, 7}, {     35, 8}, {     19, 7}, {     41, 8}, \
    {     23, 7}, {     49, 8}, {     27, 9}, {     15, 8}, \
    {     39, 9}, {     23, 8}, {     51,10}, {     15, 9}, \
    {     31, 8}, {     63, 4}, {   1023, 8}, {     67, 9}, \
    {     39, 5}, {    639, 4}, {   1471, 6}, {    383, 7}, \
    {    209, 8}, {    119, 9}, {     63, 7}, {    255, 8}, \
    {    139, 9}, {     71, 8}, {    143, 9}, {     79,10}, \
    {     47, 9}, {     95,11}, {     31,10}, {     63, 9}, \
    {    135,10}, {     79, 9}, {    159, 8}, {    319, 9}, \
    {    167,10}, {     95,11}, {     63,10}, {    143, 9}, \
    {    287,10}, {    159,11}, {     95,10}, {    191,12}, \
    {     63,11}, {    127,10}, {    255, 9}, {    543, 8}, \
    {   1087,10}, {    287, 9}, {    575,11}, {    159,10}, \
    {    319, 9}, {    639,10}, {    335, 9}, {    671,10}, \
    {    351, 9}, {    703,11}, {    191,10}, {    383, 9}, \
    {    767,10}, {    399, 9}, {    799,10}, {    415, 9}, \
    {    831,11}, {    223,12}, {    127,11}, {    255,10}, \
    {    543, 9}, {   1087,11}, {    287,10}, {    607, 9}, \
    {   1215,11}, {    319,10}, {    671, 9}, {   1343,11}, \
    {    351,10}, {    703,12}, {    191,11}, {    383,10}, \
    {    799,11}, {    415,10}, {    831,13}, {    127,12}, \
    {    255,11}, {    543,10}, {   1087,11}, {    607,12}, \
    {    319,11}, {    671,10}, {   1343,11}, {    735,12}, \
    {    383,11}, {    799,10}, {   1599,11}, {    863,12}, \
    {    447,11}, {    959,13}, {    255,12}, {    511,11}, \
    {   1087,12}, {    575,11}, {   1215,12}, {    639,11}, \
    {   1343,12}, {    703,11}, {   1471,13}, {    383,12}, \
    {    767,11}, {   1599,12}, {    831,11}, {   1727,12}, \
    {    959,14}, {    255,13}, {    511,12}, {   1215,13}, \
    {    639,12}, {   1471,13}, {    767,12}, {   1727,13}, \
    {    895,12}, {   1919,14}, {    511,13}, {   1023,12}, \
    {   2111,13}, {   1151,12}, {   2431,13}, {   1407,14}, \
    {    767,13}, {   1663,12}, {   3455,13}, {   8192,14}, \
    {  16384,15}, {  32768,16} }
#define SQR_FFT_TABLE3_SIZE 146
#define SQR_FFT_THRESHOLD                 5760

#define MULLO_BASECASE_THRESHOLD             0  /* always */
#define MULLO_DC_THRESHOLD                  31
#define MULLO_MUL_N_THRESHOLD            13463
#define SQRLO_BASECASE_THRESHOLD             0  /* always */
#define SQRLO_DC_THRESHOLD                 100
#define SQRLO_SQR_THRESHOLD               9236

#define DC_DIV_QR_THRESHOLD                 25
#define DC_DIVAPPR_Q_THRESHOLD              55
#define DC_BDIV_QR_THRESHOLD                60
#define DC_BDIV_Q_THRESHOLD                132

#define INV_MULMOD_BNM1_THRESHOLD           38
#define INV_NEWTON_THRESHOLD                65
#define INV_APPR_THRESHOLD                  65

#define BINV_NEWTON_THRESHOLD              252
#define REDC_1_TO_REDC_N_THRESHOLD          62

#define MU_DIV_QR_THRESHOLD               1164
#define MU_DIVAPPR_Q_THRESHOLD             748
#define MUPI_DIV_QR_THRESHOLD               38
#define MU_BDIV_QR_THRESHOLD              1360
#define MU_BDIV_Q_THRESHOLD               1470

#define POWM_SEC_TABLE  2,23,258,879,2246

#define GET_STR_DC_THRESHOLD                13
#define GET_STR_PRECOMPUTE_THRESHOLD        25
#define SET_STR_DC_THRESHOLD               582
#define SET_STR_PRECOMPUTE_THRESHOLD      1118

#define FAC_DSC_THRESHOLD                  178
#define FAC_ODD_THRESHOLD                   34

#define MATRIX22_STRASSEN_THRESHOLD         17
#define HGCD_THRESHOLD                      69
#define HGCD_APPR_THRESHOLD                112
#define HGCD_REDUCE_THRESHOLD             3389
#define GCD_DC_THRESHOLD                   386
#define GCDEXT_DC_THRESHOLD                303
#define JACOBI_BASE_METHOD                   1
