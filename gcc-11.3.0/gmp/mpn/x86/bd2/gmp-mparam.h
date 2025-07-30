/* AMD bd2 gmp-mparam.h -- Compiler/machine parameter header file.

Copyright 1991, 1993, 1994, 2000-2005, 2008-2010, 2014, 2015 Free
Software Foundation, Inc.

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

/* 4000 MHz Piledriver Vishera  */
/* FFT tuning limit = 40000000 */

#define MOD_1_NORM_THRESHOLD                 0  /* always */
#define MOD_1_UNNORM_THRESHOLD               3
#define MOD_1N_TO_MOD_1_1_THRESHOLD          7
#define MOD_1U_TO_MOD_1_1_THRESHOLD          5
#define MOD_1_1_TO_MOD_1_2_THRESHOLD        19
#define MOD_1_2_TO_MOD_1_4_THRESHOLD         0  /* never mpn_mod_1s_2p */
#define PREINV_MOD_1_TO_MOD_1_THRESHOLD     10
#define USE_PREINV_DIVREM_1                  1  /* native */
#define DIV_QR_1N_PI1_METHOD                 1
#define DIV_QR_1_NORM_THRESHOLD              3
#define DIV_QR_1_UNNORM_THRESHOLD        MP_SIZE_T_MAX  /* never */
#define DIV_QR_2_PI2_THRESHOLD           MP_SIZE_T_MAX  /* never */
#define DIVEXACT_1_THRESHOLD                 0  /* always (native) */
#define BMOD_1_TO_MOD_1_THRESHOLD           24

#define MUL_TOOM22_THRESHOLD                30
#define MUL_TOOM33_THRESHOLD                81
#define MUL_TOOM44_THRESHOLD               153
#define MUL_TOOM6H_THRESHOLD               222
#define MUL_TOOM8H_THRESHOLD               357

#define MUL_TOOM32_TO_TOOM43_THRESHOLD      89
#define MUL_TOOM32_TO_TOOM53_THRESHOLD     114
#define MUL_TOOM42_TO_TOOM53_THRESHOLD      99
#define MUL_TOOM42_TO_TOOM63_THRESHOLD      96
#define MUL_TOOM43_TO_TOOM54_THRESHOLD     130

#define SQR_BASECASE_THRESHOLD               0  /* always (native) */
#define SQR_TOOM2_THRESHOLD                 38
#define SQR_TOOM3_THRESHOLD                 89
#define SQR_TOOM4_THRESHOLD                196
#define SQR_TOOM6_THRESHOLD                290
#define SQR_TOOM8_THRESHOLD                454

#define MULMID_TOOM42_THRESHOLD             68

#define MULMOD_BNM1_THRESHOLD               19
#define SQRMOD_BNM1_THRESHOLD               22

#define MUL_FFT_MODF_THRESHOLD             636  /* k = 5 */
#define MUL_FFT_TABLE3                                      \
  { {    636, 5}, {     27, 6}, {     27, 7}, {     15, 6}, \
    {     33, 7}, {     17, 6}, {     35, 7}, {     19, 6}, \
    {     39, 7}, {     23, 6}, {     47, 7}, {     29, 8}, \
    {     15, 7}, {     35, 8}, {     19, 7}, {     41, 8}, \
    {     23, 7}, {     49, 8}, {     27, 7}, {     55, 9}, \
    {     15, 8}, {     31, 7}, {     63, 8}, {     43, 9}, \
    {     23, 8}, {     55, 9}, {     31, 8}, {     67, 9}, \
    {     39, 8}, {     79, 9}, {     47, 8}, {     95, 9}, \
    {     55,10}, {     31, 9}, {     63, 8}, {    127, 9}, \
    {     79,10}, {     47, 9}, {     95,11}, {     31,10}, \
    {     63, 9}, {    135,10}, {     79, 9}, {    159,10}, \
    {     95, 9}, {    191,11}, {     63,10}, {    127, 6}, \
    {   2111, 5}, {   4351, 6}, {   2239, 7}, {   1215, 9}, \
    {    311, 8}, {    639,10}, {    175, 8}, {    703,10}, \
    {    191,12}, {     63,11}, {    127,10}, {    255, 9}, \
    {    511,10}, {    271, 9}, {    543,10}, {    287,11}, \
    {    159, 9}, {    671,11}, {    191,10}, {    383, 9}, \
    {    799,11}, {    223,12}, {    127,11}, {    255,10}, \
    {    543, 9}, {   1087,11}, {    287,10}, {    607,11}, \
    {    319,10}, {    671,12}, {    191,11}, {    383,10}, \
    {    799,11}, {    415,13}, {    127,12}, {    255,11}, \
    {    543,10}, {   1087,11}, {    607,10}, {   1215,12}, \
    {    319,11}, {    671,10}, {   1343,11}, {    735,10}, \
    {   1471,12}, {    383,11}, {    799,10}, {   1599,11}, \
    {    863,12}, {    447,11}, {    895,13}, {    255,12}, \
    {    511,11}, {   1087,12}, {    575,11}, {   1215,10}, \
    {   2431,12}, {    639,11}, {   1343,12}, {    703,11}, \
    {   1471,13}, {    383,12}, {    767,11}, {   1599,12}, \
    {    831,11}, {   1727,10}, {   3455,12}, {    895,14}, \
    {    255,13}, {    511,12}, {   1023,11}, {   2047,12}, \
    {   1087,11}, {   2239,10}, {   4479,12}, {   1215,11}, \
    {   2431,13}, {    639,12}, {   1471,11}, {   2943,13}, \
    {    767,12}, {   1727,11}, {   3455,13}, {    895,12}, \
    {   1919,14}, {    511,13}, {   1023,12}, {   2239,11}, \
    {   4479,13}, {   1151,12}, {   2495,11}, {   4991,13}, \
    {   1279,12}, {   2623,13}, {   1407,12}, {   2943,14}, \
    {    767,13}, {   1535,12}, {   3071,13}, {   1663,12}, \
    {   3455,13}, {   1919,15}, {    511,14}, {   1023,13}, \
    {   2175,12}, {   4479,13}, {   2431,12}, {   4991,14}, \
    {   1279,13}, {   2943,12}, {   5887,14}, {   1535,13}, \
    {   3455,14}, {   1791,13}, {   3967,12}, {   7935,11}, \
    {  15871,15}, {   1023,14}, {   2047,13}, {   4479,14}, \
    {   2303,13}, {   8192,14}, {  16384,15}, {  32768,16} }
#define MUL_FFT_TABLE3_SIZE 172
#define MUL_FFT_THRESHOLD                 6784

#define SQR_FFT_MODF_THRESHOLD             606  /* k = 5 */
#define SQR_FFT_TABLE3                                      \
  { {    606, 5}, {     28, 6}, {     15, 5}, {     31, 6}, \
    {     29, 7}, {     15, 6}, {     32, 7}, {     17, 6}, \
    {     35, 7}, {     19, 6}, {     39, 7}, {     23, 6}, \
    {     47, 7}, {     29, 8}, {     15, 7}, {     35, 8}, \
    {     19, 7}, {     41, 8}, {     23, 7}, {     49, 8}, \
    {     31, 7}, {     63, 8}, {     43, 9}, {     23, 8}, \
    {     51, 9}, {     31, 8}, {     67, 9}, {     39, 8}, \
    {     79, 9}, {     47, 8}, {     95,10}, {     31, 9}, \
    {     79,10}, {     47, 9}, {     95,11}, {     31,10}, \
    {     63, 9}, {    135,10}, {     79, 9}, {    159,10}, \
    {     95, 9}, {    191,11}, {     63,10}, {    159,11}, \
    {     95,10}, {    191, 6}, {   3135, 5}, {   6399, 6}, \
    {   3455, 8}, {    895, 9}, {    479, 8}, {    991,10}, \
    {    255, 9}, {    575,11}, {    159, 9}, {    639,10}, \
    {    335, 8}, {   1343,10}, {    351,11}, {    191, 9}, \
    {    799,11}, {    223,12}, {    127,11}, {    255,10}, \
    {    543,11}, {    287,10}, {    607, 9}, {   1215,10}, \
    {    671,12}, {    191,11}, {    383,10}, {    767, 9}, \
    {   1535,10}, {    799,11}, {    415,10}, {    863,13}, \
    {    127,12}, {    255,11}, {    511,10}, {   1023,11}, \
    {    543,10}, {   1087,11}, {    607,12}, {    319,11}, \
    {    671,10}, {   1343,11}, {    735,12}, {    383,11}, \
    {    799,10}, {   1599,11}, {    863,12}, {    447,11}, \
    {    927,13}, {    255,12}, {    511,11}, {   1087,12}, \
    {    575,11}, {   1215,12}, {    639,11}, {   1343,12}, \
    {    703,11}, {   1471,13}, {    383,12}, {    767,11}, \
    {   1599,12}, {    831,11}, {   1727,12}, {    895,11}, \
    {   1791,12}, {    959,14}, {    255,13}, {    511,12}, \
    {   1087,11}, {   2239,10}, {   4479,12}, {   1215,13}, \
    {    639,12}, {   1471,11}, {   2943,13}, {    767,12}, \
    {   1727,13}, {    895,12}, {   1919,14}, {    511,13}, \
    {   1023,12}, {   2239,11}, {   4479,13}, {   1151,12}, \
    {   2495,11}, {   4991,13}, {   1279,12}, {   2623,13}, \
    {   1407,12}, {   2943,14}, {    767,13}, {   1663,12}, \
    {   3455,13}, {   1791,12}, {   3583,13}, {   1919,15}, \
    {    511,14}, {   1023,13}, {   2175,12}, {   4479,13}, \
    {   2431,12}, {   4991,14}, {   1279,13}, {   2943,12}, \
    {   5887,14}, {   1535,13}, {   3455,14}, {   1791,13}, \
    {   3967,15}, {   1023,14}, {   2047,13}, {   4479,14}, \
    {   2303,13}, {   8192,14}, {  16384,15}, {  32768,16} }
#define SQR_FFT_TABLE3_SIZE 160
#define SQR_FFT_THRESHOLD                 5760

#define MULLO_BASECASE_THRESHOLD             3
#define MULLO_DC_THRESHOLD                  34
#define MULLO_MUL_N_THRESHOLD            13463
#define SQRLO_BASECASE_THRESHOLD             7
#define SQRLO_DC_THRESHOLD                  43
#define SQRLO_SQR_THRESHOLD              11278

#define DC_DIV_QR_THRESHOLD                 67
#define DC_DIVAPPR_Q_THRESHOLD             196
#define DC_BDIV_QR_THRESHOLD                67
#define DC_BDIV_Q_THRESHOLD                112

#define INV_MULMOD_BNM1_THRESHOLD           70
#define INV_NEWTON_THRESHOLD               262
#define INV_APPR_THRESHOLD                 222

#define BINV_NEWTON_THRESHOLD              288
#define REDC_1_TO_REDC_N_THRESHOLD          67

#define MU_DIV_QR_THRESHOLD               1718
#define MU_DIVAPPR_Q_THRESHOLD            1652
#define MUPI_DIV_QR_THRESHOLD              122
#define MU_BDIV_QR_THRESHOLD              1387
#define MU_BDIV_Q_THRESHOLD               1528

#define POWM_SEC_TABLE  1,16,69,508,1378,2657,2825

#define MATRIX22_STRASSEN_THRESHOLD         19
#define HGCD_THRESHOLD                      61
#define HGCD_APPR_THRESHOLD                 50
#define HGCD_REDUCE_THRESHOLD             3389
#define GCD_DC_THRESHOLD                   492
#define GCDEXT_DC_THRESHOLD                345
#define JACOBI_BASE_METHOD                   4

#define GET_STR_DC_THRESHOLD                 9
#define GET_STR_PRECOMPUTE_THRESHOLD        21
#define SET_STR_DC_THRESHOLD               189
#define SET_STR_PRECOMPUTE_THRESHOLD       541

#define FAC_DSC_THRESHOLD                  141
#define FAC_ODD_THRESHOLD                   29
