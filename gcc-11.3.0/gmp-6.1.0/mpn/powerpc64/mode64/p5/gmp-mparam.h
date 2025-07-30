/* POWER5 gmp-mparam.h -- Compiler/machine parameter header file.

Copyright 1991, 1993, 1994, 1999-2003, 2009, 2010 Free Software Foundation,
Inc.

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

#define GMP_LIMB_BITS 64
#define GMP_LIMB_BYTES 8

/* POWER5 (friggms.hpc.ntnu.no) */

#define MOD_1_NORM_THRESHOLD                 0  /* always */
#define MOD_1_UNNORM_THRESHOLD               0  /* always */
#define MOD_1N_TO_MOD_1_1_THRESHOLD          6
#define MOD_1U_TO_MOD_1_1_THRESHOLD          5
#define MOD_1_1_TO_MOD_1_2_THRESHOLD        15
#define MOD_1_2_TO_MOD_1_4_THRESHOLD         0  /* never mpn_mod_1s_2p */
#define PREINV_MOD_1_TO_MOD_1_THRESHOLD     11
#define USE_PREINV_DIVREM_1                  0
#define DIV_QR_2_PI2_THRESHOLD           MP_SIZE_T_MAX  /* never */
#define DIVEXACT_1_THRESHOLD                 0  /* always (native) */
#define BMOD_1_TO_MOD_1_THRESHOLD           40

#define MUL_TOOM22_THRESHOLD                21
#define MUL_TOOM33_THRESHOLD                24
#define MUL_TOOM44_THRESHOLD                70
#define MUL_TOOM6H_THRESHOLD               262
#define MUL_TOOM8H_THRESHOLD               393

#define MUL_TOOM32_TO_TOOM43_THRESHOLD      49
#define MUL_TOOM32_TO_TOOM53_THRESHOLD     126
#define MUL_TOOM42_TO_TOOM53_THRESHOLD      85
#define MUL_TOOM42_TO_TOOM63_THRESHOLD      94
#define MUL_TOOM43_TO_TOOM54_THRESHOLD      70

#define SQR_BASECASE_THRESHOLD               0  /* always (native) */
#define SQR_TOOM2_THRESHOLD                 24
#define SQR_TOOM3_THRESHOLD                 81
#define SQR_TOOM4_THRESHOLD                142
#define SQR_TOOM6_THRESHOLD                189
#define SQR_TOOM8_THRESHOLD                284

#define MULMID_TOOM42_THRESHOLD             36

#define MULMOD_BNM1_THRESHOLD               12
#define SQRMOD_BNM1_THRESHOLD               15

#define MUL_FFT_MODF_THRESHOLD             304  /* k = 5 */
#define MUL_FFT_TABLE3                                      \
  { {    348, 5}, {     17, 6}, {      9, 5}, {     19, 6}, \
    {     10, 5}, {     21, 6}, {     21, 7}, {     11, 6}, \
    {     23, 7}, {     12, 6}, {     25, 7}, {     21, 8}, \
    {     11, 7}, {     25, 8}, {     13, 7}, {     27, 8}, \
    {     15, 7}, {     31, 8}, {     21, 9}, {     11, 8}, \
    {     27, 9}, {     15, 8}, {     33, 9}, {     19, 8}, \
    {     39, 9}, {     23, 8}, {     47, 9}, {     27,10}, \
    {     15, 9}, {     39,10}, {     23, 9}, {     51,11}, \
    {     15,10}, {     31, 9}, {     67,10}, {     39, 9}, \
    {     79,10}, {     47, 9}, {     95,10}, {     55,11}, \
    {     31,10}, {     79,11}, {     47,10}, {     95,12}, \
    {     31,11}, {     63,10}, {    127, 9}, {    255,10}, \
    {    135,11}, {     79,10}, {    159, 9}, {    319,11}, \
    {     95,10}, {    191, 9}, {    383,12}, {     63,11}, \
    {    127,10}, {    255, 9}, {    511,10}, {    271,11}, \
    {    143,10}, {    287, 9}, {    575,10}, {    319,12}, \
    {     95,11}, {    191,10}, {    383,13}, {     63,12}, \
    {    127,11}, {    255,10}, {    511,11}, {    271,10}, \
    {    543,11}, {    287,10}, {    575, 9}, {   1151,11}, \
    {    319,10}, {    639,11}, {    351,10}, {    703,12}, \
    {    191,11}, {    383,10}, {    767,11}, {    415,12}, \
    {    223,11}, {    447,13}, {    127,12}, {    255,11}, \
    {    511,10}, {   1023,11}, {    543,10}, {   1087,12}, \
    {    287,11}, {    575,10}, {   1151,12}, {    319,11}, \
    {    639,12}, {    351,11}, {    703,13}, {    191,12}, \
    {    383,11}, {    767,12}, {    415,11}, {    831,12}, \
    {    447,11}, {    895,14}, {    127,13}, {    255,12}, \
    {    511,11}, {   1023,12}, {    543,11}, {   1087,10}, \
    {   2175,12}, {    575,11}, {   1151,12}, {    607,13}, \
    {    319,12}, {    639,11}, {   1279,12}, {    671,11}, \
    {   1343,12}, {    703,11}, {   1407,13}, {    383,12}, \
    {    767,11}, {   1535,12}, {    831,13}, {    447,12}, \
    {    959,11}, {   1919,14}, {    255,13}, {    511,12}, \
    {   1087,11}, {   2175,13}, {    575,12}, {   1215,11}, \
    {   2431,10}, {   4863,13}, {    639,12}, {   1343,13}, \
    {    703,12}, {   1407,14}, {    383,13}, {    767,12}, \
    {   1535,13}, {    831,12}, {   1663,13}, {    959,12}, \
    {   1919,11}, {   3839,15}, {    255,14}, {    511,13}, \
    {   1087,12}, {   2175,13}, {   1215,12}, {   2431,11}, \
    {   4863,14}, {    639,13}, {   1343,12}, {   2687,13}, \
    {   1407,12}, {   2815,13}, {   1471,12}, {   2943,14}, \
    {    767,13}, {   1599,12}, {   3199,13}, {   1663,14}, \
    {    895,13}, {   1919,12}, {   3839,15}, {    511,14}, \
    {   1023,13}, {   2175,14}, {   1151,13}, {   2431,12}, \
    {   4863,14}, {   1279,13}, {   2687,14}, {   1407,13}, \
    {   2943,15}, {    767,14}, {   1535,13}, {   3199,14}, \
    {   1663,13}, {   3327,14}, {   1919,13}, {   3839,16}, \
    {    511,15}, {   1023,14}, {   2431,13}, {   4863,15}, \
    {   1279,14}, {   2943,12}, {  11775,15}, {   1535,14}, \
    {   3327,15}, {   1791,14}, {  16384,15}, {  32768,16}, \
    {  65536,17}, { 131072,18}, { 262144,19}, { 524288,20}, \
    {1048576,21}, {2097152,22}, {4194304,23}, {8388608,24} }
#define MUL_FFT_TABLE3_SIZE 208
#define MUL_FFT_THRESHOLD                 4224

#define SQR_FFT_MODF_THRESHOLD             284  /* k = 5 */
#define SQR_FFT_TABLE3                                      \
  { {    272, 5}, {     15, 6}, {      8, 5}, {     17, 6}, \
    {     19, 7}, {     17, 8}, {      9, 7}, {     21, 8}, \
    {     11, 7}, {     24, 8}, {     13, 7}, {     27, 8}, \
    {     15, 7}, {     31, 8}, {     21, 9}, {     11, 8}, \
    {     27, 9}, {     15, 8}, {     33, 9}, {     19, 8}, \
    {     39, 9}, {     23, 8}, {     47, 9}, {     27,10}, \
    {     15, 9}, {     39,10}, {     23, 9}, {     47,11}, \
    {     15,10}, {     31, 9}, {     63,10}, {     47,11}, \
    {     31,10}, {     71, 9}, {    143,10}, {     79,11}, \
    {     47,10}, {     95,12}, {     31,11}, {     63,10}, \
    {    127, 9}, {    255, 8}, {    511, 9}, {    271,10}, \
    {    143,11}, {     79,10}, {    159, 9}, {    319,10}, \
    {    175, 9}, {    351,11}, {     95,10}, {    191, 9}, \
    {    383,10}, {    207, 9}, {    415,12}, {     63,11}, \
    {    127,10}, {    255, 9}, {    511,10}, {    271,11}, \
    {    143,10}, {    287, 9}, {    575,11}, {    159,10}, \
    {    319,11}, {    175,10}, {    351,12}, {     95,11}, \
    {    191,10}, {    383,11}, {    207,10}, {    415,13}, \
    {     63,12}, {    127,11}, {    255,10}, {    511,11}, \
    {    271,10}, {    543,11}, {    287,10}, {    575,12}, \
    {    159,11}, {    319,10}, {    639,11}, {    351,10}, \
    {    703,12}, {    191,11}, {    383,10}, {    767,11}, \
    {    415,12}, {    223,11}, {    447,10}, {    895,11}, \
    {    479,10}, {    959,12}, {    255,11}, {    511,10}, \
    {   1023,11}, {    543,12}, {    287,11}, {    575,12}, \
    {    319,11}, {    639,12}, {    351,11}, {    703,13}, \
    {    191,12}, {    383,11}, {    767,12}, {    415,11}, \
    {    831,12}, {    447,11}, {    895,12}, {    479,11}, \
    {    959,13}, {    255,12}, {    511,11}, {   1023,12}, \
    {    543,11}, {   1087,12}, {    575,13}, {    319,12}, \
    {    639,11}, {   1279,12}, {    703,11}, {   1407,13}, \
    {    383,12}, {    831,13}, {    447,12}, {    959,14}, \
    {    255,13}, {    511,12}, {   1087,13}, {    575,12}, \
    {   1215,13}, {    639,12}, {   1279,13}, {    703,12}, \
    {   1407,14}, {    383,13}, {    831,12}, {   1663,13}, \
    {    959,12}, {   1919,15}, {    255,14}, {    511,13}, \
    {   1023,12}, {   2047,13}, {   1087,12}, {   2175,13}, \
    {   1215,14}, {    639,13}, {   1407,12}, {   2815,14}, \
    {    767,13}, {   1663,14}, {    895,13}, {   1919,15}, \
    {    511,14}, {   1023,13}, {   2175,14}, {   1151,13}, \
    {   2431,12}, {   4863,14}, {   1407,13}, {   2815,15}, \
    {    767,14}, {   1663,13}, {   3327,14}, {   1919,13}, \
    {   3839,16}, {    511,15}, {   1023,14}, {   2431,13}, \
    {   4863,15}, {   1279,14}, {   2943,13}, {   5887,12}, \
    {  11775,15}, {   1535,14}, {   3327,15}, {   1791,14}, \
    {  16384,15}, {  32768,16}, {  65536,17}, { 131072,18}, \
    { 262144,19}, { 524288,20}, {1048576,21}, {2097152,22}, \
    {4194304,23}, {8388608,24} }
#define SQR_FFT_TABLE3_SIZE 190
#define SQR_FFT_THRESHOLD                 3264

#define MULLO_BASECASE_THRESHOLD             6
#define MULLO_DC_THRESHOLD                  60
#define MULLO_MUL_N_THRESHOLD             7463

#define DC_DIV_QR_THRESHOLD                 58
#define DC_DIVAPPR_Q_THRESHOLD             232
#define DC_BDIV_QR_THRESHOLD                78
#define DC_BDIV_Q_THRESHOLD                238

#define INV_MULMOD_BNM1_THRESHOLD           92
#define INV_NEWTON_THRESHOLD               155
#define INV_APPR_THRESHOLD                 157

#define BINV_NEWTON_THRESHOLD              155
#define REDC_1_TO_REDC_N_THRESHOLD          61

#define MU_DIV_QR_THRESHOLD                998
#define MU_DIVAPPR_Q_THRESHOLD             979
#define MUPI_DIV_QR_THRESHOLD               79
#define MU_BDIV_QR_THRESHOLD               823
#define MU_BDIV_Q_THRESHOLD                942

#define MATRIX22_STRASSEN_THRESHOLD         14
#define HGCD_THRESHOLD                      74
#define HGCD_APPR_THRESHOLD                155
#define HGCD_REDUCE_THRESHOLD             2479
#define GCD_DC_THRESHOLD                   351
#define GCDEXT_DC_THRESHOLD                288
#define JACOBI_BASE_METHOD                   4

#define GET_STR_DC_THRESHOLD                12
#define GET_STR_PRECOMPUTE_THRESHOLD        21
#define SET_STR_DC_THRESHOLD               650
#define SET_STR_PRECOMPUTE_THRESHOLD      1585

#define FAC_DSC_THRESHOLD                  662
#define FAC_ODD_THRESHOLD                   28
