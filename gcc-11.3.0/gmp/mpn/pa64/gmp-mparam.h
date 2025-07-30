/* gmp-mparam.h -- Compiler/machine parameter header file.

Copyright 1991, 1993, 1994, 1999-2004, 2008-2010 Free Software Foundation, Inc.

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

/* 440MHz PA8200 */

#define DIVREM_1_NORM_THRESHOLD              0  /* always */
#define DIVREM_1_UNNORM_THRESHOLD            0  /* always */
#define MOD_1_1P_METHOD                      2
#define MOD_1_NORM_THRESHOLD                 0  /* always */
#define MOD_1_UNNORM_THRESHOLD               0  /* always */
#define MOD_1N_TO_MOD_1_1_THRESHOLD          4
#define MOD_1U_TO_MOD_1_1_THRESHOLD         10
#define MOD_1_1_TO_MOD_1_2_THRESHOLD         0  /* never mpn_mod_1_1p */
#define MOD_1_2_TO_MOD_1_4_THRESHOLD        14
#define PREINV_MOD_1_TO_MOD_1_THRESHOLD     11
#define USE_PREINV_DIVREM_1                  1
#define DIV_QR_2_PI2_THRESHOLD              21
#define DIVEXACT_1_THRESHOLD                 0  /* always */
#define BMOD_1_TO_MOD_1_THRESHOLD        MP_SIZE_T_MAX  /* never */

#define MUL_TOOM22_THRESHOLD                31
#define MUL_TOOM33_THRESHOLD               114
#define MUL_TOOM44_THRESHOLD               179
#define MUL_TOOM6H_THRESHOLD               222
#define MUL_TOOM8H_THRESHOLD               296

#define MUL_TOOM32_TO_TOOM43_THRESHOLD     130
#define MUL_TOOM32_TO_TOOM53_THRESHOLD     229
#define MUL_TOOM42_TO_TOOM53_THRESHOLD     129
#define MUL_TOOM42_TO_TOOM63_THRESHOLD      54

#define SQR_BASECASE_THRESHOLD               5
#define SQR_TOOM2_THRESHOLD                 58
#define SQR_TOOM3_THRESHOLD                153
#define SQR_TOOM4_THRESHOLD                278
#define SQR_TOOM6_THRESHOLD                  0  /* always */
#define SQR_TOOM8_THRESHOLD                  0  /* always */

#define MULMID_TOOM42_THRESHOLD             56

#define MULMOD_BNM1_THRESHOLD               15
#define SQRMOD_BNM1_THRESHOLD               19

#define POWM_SEC_TABLE  2,23,228,1084

#define MUL_FFT_MODF_THRESHOLD             336  /* k = 5 */
#define MUL_FFT_TABLE3                                      \
  { {    336, 5}, {     11, 4}, {     23, 5}, {     21, 6}, \
    {     11, 5}, {     23, 6}, {     21, 7}, {     11, 6}, \
    {     23, 7}, {     15, 6}, {     31, 7}, {     21, 8}, \
    {     11, 7}, {     24, 8}, {     13, 7}, {     27, 8}, \
    {     15, 7}, {     31, 8}, {     19, 7}, {     39, 8}, \
    {     27, 9}, {     15, 8}, {     33, 9}, {     19, 8}, \
    {     39, 9}, {     23, 8}, {     47, 9}, {     27,10}, \
    {     15, 9}, {     39,10}, {     23, 9}, {     51,10}, \
    {     31, 9}, {     67,10}, {     39, 9}, {     79,10}, \
    {     47, 9}, {     95,10}, {     55,11}, {     31,10}, \
    {     63, 9}, {    127,10}, {     71, 8}, {    287,10}, \
    {     79,11}, {     47,10}, {     95, 9}, {    191, 8}, \
    {    383, 7}, {    767,10}, {    103, 9}, {    207, 8}, \
    {    415, 7}, {    831,12}, {     31,11}, {     63,10}, \
    {    127, 9}, {    255, 8}, {    543, 7}, {   1087, 6}, \
    {   2175,10}, {    143, 9}, {    287, 8}, {    575,11}, \
    {     79, 9}, {    319, 8}, {    639, 7}, {   1279, 9}, \
    {    335, 8}, {    671,10}, {    175, 9}, {    351, 8}, \
    {    703,11}, {     95,10}, {    191, 9}, {    383, 8}, \
    {    767,10}, {    207, 9}, {    415, 8}, {    831, 7}, \
    {   1663,11}, {    111,10}, {    223, 9}, {    447, 8}, \
    {    895,12}, {     63,11}, {    127,10}, {    255, 9}, \
    {    543, 8}, {   1087, 7}, {   2175,10}, {    287, 9}, \
    {    575, 8}, {   1215, 7}, {   2431,10}, {    319, 9}, \
    {    639, 8}, {   1279,10}, {    335, 9}, {    671, 8}, \
    {   1343, 9}, {    703, 8}, {   1407,12}, {     95,11}, \
    {    191,10}, {    383,11}, {    207, 9}, {    831, 8}, \
    {   1663,11}, {    223,10}, {    447, 9}, {    959,13}, \
    {     63,12}, {    127,11}, {    255, 8}, {   2047,11}, \
    {    271,10}, {    543, 9}, {   1087, 8}, {   2175,11}, \
    {    287,10}, {    575, 9}, {   1215, 8}, {   2431,11}, \
    {    319,10}, {    671, 9}, {   1343, 8}, {   2687,11}, \
    {    351,10}, {    703, 9}, {   1471, 8}, {   2943,12}, \
    {    191,11}, {    383, 8}, {   3071,11}, {    415,10}, \
    {    831, 9}, {   1663,11}, {    479,10}, {    959, 9}, \
    {   1919, 8}, {   3839,13}, {    127,12}, {    255,11}, \
    {    543,10}, {   1087, 9}, {   2175,12}, {    287,11}, \
    {    607,10}, {   1215, 9}, {   2431, 8}, {   4863,12}, \
    {    319,11}, {    671,10}, {   1343,13}, {    191, 9}, \
    {   3071,12}, {    415,11}, {    831,10}, {   1663, 8}, \
    {   6655, 9}, {   3455,12}, {    447, 9}, {   3583,13}, \
    {    255,12}, {    511,11}, {   1023,10}, {   2175,13}, \
    {    319,11}, {   1279,12}, {    671,10}, {   2815,12}, \
    {    735,10}, {   2943, 9}, {   5887,13}, {    383,12}, \
    {    767,11}, {   1535,10}, {   3071,13}, {    447,10}, \
    {   3583,12}, {    959,13}, {    511,12}, {   1087,13}, \
    {    639,12}, {   1343,13}, {    767,11}, {   3071,13}, \
    {    831,12}, {   1663,11}, {   3455,10}, {   6911,13}, \
    {    895,14}, {    511,13}, {   1023,12}, {   2047,13}, \
    {   1087,12}, {   2303,13}, {   1215,12}, {   2431,14}, \
    {    639,13}, {   1279,12}, {   2559,13}, {   1343,12}, \
    {   2687,11}, {   5375,13}, {   1407,12}, {   2815,11}, \
    {   5631,12}, {   2943,13}, {   1535,12}, {   3199,13}, \
    {   1663,12}, {   3327,13}, {   1727,14}, {    895,13}, \
    {   1791,12}, {   3583,13}, {   1919,15}, {    511,14}, \
    {   1023,13}, {   2047,12}, {   4095,14}, {   1151,13}, \
    {   2431,14}, {   1279,13}, {   2687,14}, {   1407,13}, \
    {   2815,12}, {   5631,15}, {    767,14}, {   1535,13}, \
    {   3071,14}, {   1663,13}, {   3327,14}, {   1791,13}, \
    {   3583,14}, {   1919,15}, {   1023,14}, {   2303,13}, \
    {   4607,14}, {   2431,13}, {   4863,15}, {  32768,16}, \
    {  65536,17}, { 131072,18}, { 262144,19}, { 524288,20}, \
    {1048576,21}, {2097152,22}, {4194304,23}, {8388608,24} }
#define MUL_FFT_TABLE3_SIZE 252
#define MUL_FFT_THRESHOLD                 2368

#define SQR_FFT_MODF_THRESHOLD             284  /* k = 5 */
#define SQR_FFT_TABLE3                                      \
  { {    284, 5}, {      9, 4}, {     21, 5}, {     21, 6}, \
    {     11, 5}, {     23, 6}, {     25, 7}, {     25, 8}, \
    {     13, 7}, {     27, 8}, {     15, 7}, {     31, 8}, \
    {     27, 9}, {     15, 8}, {     33, 9}, {     19, 8}, \
    {     39, 9}, {     23, 8}, {     47, 9}, {     27,10}, \
    {     15, 9}, {     39,10}, {     23, 9}, {     51,11}, \
    {     15,10}, {     31, 9}, {     67,10}, {     39, 9}, \
    {     79,10}, {     47, 9}, {     95,10}, {     55,11}, \
    {     31,10}, {     63, 8}, {    255, 7}, {    511,10}, \
    {     71, 8}, {    287, 7}, {    575,10}, {     79,11}, \
    {     47,10}, {     95, 9}, {    191, 8}, {    383, 7}, \
    {    767,10}, {    103, 9}, {    207, 8}, {    415,12}, \
    {     31,11}, {     63,10}, {    127, 9}, {    255, 8}, \
    {    543, 7}, {   1087, 8}, {    575, 7}, {   1151,11}, \
    {     79, 8}, {    639, 7}, {   1279, 9}, {    335, 8}, \
    {    671, 7}, {   1343,10}, {    175, 8}, {    703, 7}, \
    {   1407,11}, {     95,10}, {    191, 9}, {    383, 8}, \
    {    767,10}, {    207, 9}, {    415, 8}, {    831, 7}, \
    {   1663, 9}, {    447, 8}, {    895,12}, {     63,11}, \
    {    127,10}, {    255, 9}, {    543, 8}, {   1087, 7}, \
    {   2175, 9}, {    575, 8}, {   1151,10}, {    303, 9}, \
    {    607, 8}, {   1215, 7}, {   2431,10}, {    319, 9}, \
    {    639, 8}, {   1279, 9}, {    671, 8}, {   1343, 7}, \
    {   2687,10}, {    351, 9}, {    703, 8}, {   1407,12}, \
    {     95,11}, {    191,10}, {    383, 9}, {    767,11}, \
    {    207,10}, {    415, 9}, {    831, 8}, {   1663,11}, \
    {    223,10}, {    447, 9}, {    895,13}, {     63,11}, \
    {    255,10}, {    543, 8}, {   2175,11}, {    287,10}, \
    {    575, 9}, {   1151,10}, {    607, 9}, {   1215, 8}, \
    {   2431,11}, {    319, 9}, {   1279,10}, {    671, 9}, \
    {   1343, 8}, {   2687,11}, {    351,10}, {    703, 9}, \
    {   1407,10}, {    735,12}, {    191,11}, {    383,10}, \
    {    831, 9}, {   1663,12}, {    223,11}, {    447,10}, \
    {    895,11}, {    479, 9}, {   1919, 8}, {   3839,12}, \
    {    255,11}, {    511,10}, {   1023,11}, {    543,10}, \
    {   1087, 9}, {   2175,12}, {    287,11}, {    575,10}, \
    {   1151,11}, {    607,10}, {   1215, 9}, {   2431, 8}, \
    {   4863,10}, {   1279,11}, {    671,10}, {   1343, 9}, \
    {   2687,12}, {    351,11}, {    703,10}, {   1407,11}, \
    {    735,13}, {    191, 9}, {   3071, 7}, {  12287,11}, \
    {    799,12}, {    415,11}, {    831,10}, {   1663,12}, \
    {    447, 8}, {   7167,12}, {    479, 9}, {   3839,14}, \
    {    127,13}, {    255,12}, {    511,11}, {   1023,12}, \
    {    543,10}, {   2175, 9}, {   4607,11}, {   1215,10}, \
    {   2431,11}, {   1279,10}, {   2559,13}, {    383,12}, \
    {    767,11}, {   1535,12}, {    799,10}, {   3199, 9}, \
    {   6399,12}, {    895,13}, {    511,12}, {   1023,11}, \
    {   2047,12}, {   1087,13}, {    575,12}, {   1151,10}, \
    {   4607,13}, {    639,12}, {   1279,11}, {   2687,14}, \
    {    383,13}, {    767,11}, {   3071,12}, {   1599,13}, \
    {    895,12}, {   1791,11}, {   3583,13}, {    959,15}, \
    {    255,12}, {   2175,13}, {   1215,14}, {    639,13}, \
    {   1279,12}, {   2559,13}, {   1343,12}, {   2687,13}, \
    {   1471,11}, {   5887,14}, {    767,13}, {   1535,12}, \
    {   3071,13}, {   1599,12}, {   3199,13}, {   1663,12}, \
    {   3327,13}, {   1727,14}, {    895,13}, {   1791,12}, \
    {   3583,15}, {    511,14}, {   1023,13}, {   2175,14}, \
    {   1151,12}, {   4607,13}, {   2431,14}, {   1279,13}, \
    {   2687,14}, {   1407,13}, {   2815,15}, {    767,13}, \
    {   3199,14}, {   1663,13}, {   3327,14}, {   1791,13}, \
    {   3583,14}, {   1919,15}, {   1023,14}, {   2047,13}, \
    {   4095,14}, {   2303,13}, {   4607,14}, {   2431,15}, \
    {  32768,16}, {  65536,17}, { 131072,18}, { 262144,19}, \
    { 524288,20}, {1048576,21}, {2097152,22}, {4194304,23}, \
    {8388608,24} }
#define SQR_FFT_TABLE3_SIZE 257
#define SQR_FFT_THRESHOLD                 1856

#define MULLO_BASECASE_THRESHOLD             0  /* always */
#define MULLO_DC_THRESHOLD                 113
#define MULLO_MUL_N_THRESHOLD             4658

#define DC_DIV_QR_THRESHOLD                123
#define DC_DIVAPPR_Q_THRESHOLD             372
#define DC_BDIV_QR_THRESHOLD               142
#define DC_BDIV_Q_THRESHOLD                312

#define INV_MULMOD_BNM1_THRESHOLD           58
#define INV_NEWTON_THRESHOLD               315
#define INV_APPR_THRESHOLD                 315

#define BINV_NEWTON_THRESHOLD              360
#define REDC_1_TO_REDC_N_THRESHOLD         101

#define MU_DIV_QR_THRESHOLD                979
#define MU_DIVAPPR_Q_THRESHOLD            1142
#define MUPI_DIV_QR_THRESHOLD               93
#define MU_BDIV_QR_THRESHOLD               889
#define MU_BDIV_Q_THRESHOLD               1187

#define MATRIX22_STRASSEN_THRESHOLD          9
#define HGCD_THRESHOLD                     234
#define HGCD_APPR_THRESHOLD                300
#define HGCD_REDUCE_THRESHOLD             1553
#define GCD_DC_THRESHOLD                   684
#define GCDEXT_DC_THRESHOLD                525
#define JACOBI_BASE_METHOD                   2

#define GET_STR_DC_THRESHOLD                21
#define GET_STR_PRECOMPUTE_THRESHOLD        24
#define SET_STR_DC_THRESHOLD              1951
#define SET_STR_PRECOMPUTE_THRESHOLD      4034
