/* PowerPC-32 gmp-mparam.h -- Compiler/machine parameter header file.

Copyright 2002, 2004, 2009, 2010 Free Software Foundation, Inc.

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


/* This file is used for 75x (G3) and for 7400/7410 (G4), both which have
   much slow multiply instructions.  */

/* 450 MHz PPC 7400 */

#define DIVREM_1_NORM_THRESHOLD              0  /* always */
#define DIVREM_1_UNNORM_THRESHOLD            0  /* always */
#define MOD_1_NORM_THRESHOLD                 3
#define MOD_1_UNNORM_THRESHOLD               0  /* always */
#define MOD_1N_TO_MOD_1_1_THRESHOLD         11
#define MOD_1U_TO_MOD_1_1_THRESHOLD          7
#define MOD_1_1_TO_MOD_1_2_THRESHOLD        11
#define MOD_1_2_TO_MOD_1_4_THRESHOLD        18
#define PREINV_MOD_1_TO_MOD_1_THRESHOLD     38
#define USE_PREINV_DIVREM_1                  1
#define DIVEXACT_1_THRESHOLD                 0  /* always */
#define BMOD_1_TO_MOD_1_THRESHOLD        MP_SIZE_T_MAX  /* never */

#define MUL_TOOM22_THRESHOLD                10
#define MUL_TOOM33_THRESHOLD                38
#define MUL_TOOM44_THRESHOLD                99
#define MUL_TOOM6H_THRESHOLD               141
#define MUL_TOOM8H_THRESHOLD               212

#define MUL_TOOM32_TO_TOOM43_THRESHOLD      65
#define MUL_TOOM32_TO_TOOM53_THRESHOLD      69
#define MUL_TOOM42_TO_TOOM53_THRESHOLD      65
#define MUL_TOOM42_TO_TOOM63_THRESHOLD      66

#define SQR_BASECASE_THRESHOLD               4
#define SQR_TOOM2_THRESHOLD                 18
#define SQR_TOOM3_THRESHOLD                 57
#define SQR_TOOM4_THRESHOLD                142
#define SQR_TOOM6_THRESHOLD                173
#define SQR_TOOM8_THRESHOLD                309

#define MULMOD_BNM1_THRESHOLD                9
#define SQRMOD_BNM1_THRESHOLD               11

#define MUL_FFT_MODF_THRESHOLD             220  /* k = 5 */
#define MUL_FFT_TABLE3                                      \
  { {    220, 5}, {     13, 6}, {      7, 5}, {     15, 6}, \
    {      8, 5}, {     17, 6}, {      9, 5}, {     19, 6}, \
    {     13, 7}, {      7, 6}, {     17, 7}, {      9, 6}, \
    {     19, 7}, {     11, 6}, {     23, 7}, {     13, 8}, \
    {      7, 7}, {     19, 8}, {     11, 7}, {     23, 9}, \
    {      7, 8}, {     15, 7}, {     33, 8}, {     19, 7}, \
    {     39, 8}, {     23, 9}, {     15, 8}, {     39, 9}, \
    {     23, 8}, {     47,10}, {     15, 9}, {     31, 8}, \
    {     67, 9}, {     55,10}, {     31, 9}, {     63, 8}, \
    {    127, 7}, {    255, 9}, {     71, 8}, {    143, 7}, \
    {    287, 9}, {     79,10}, {     47, 9}, {     95,11}, \
    {     31,10}, {     63, 9}, {    127, 8}, {    255, 9}, \
    {    143, 8}, {    287,10}, {     79, 9}, {    159, 8}, \
    {    319, 9}, {    175, 8}, {    351, 7}, {    703,10}, \
    {     95, 9}, {    191, 8}, {    383, 9}, {    207,10}, \
    {    111,11}, {     63,10}, {    127, 9}, {    255,10}, \
    {    143, 9}, {    287, 8}, {    575,10}, {    159, 9}, \
    {    319,10}, {    175, 9}, {    351, 8}, {    703,11}, \
    {     95,10}, {    191, 9}, {    383,10}, {    207, 9}, \
    {    415, 8}, {    831,12}, {     63,11}, {    127,10}, \
    {    255, 9}, {    511,10}, {    271, 9}, {    543,10}, \
    {    287, 9}, {    575,11}, {    159,10}, {    351, 9}, \
    {    703, 8}, {   1407,11}, {    191,10}, {    415, 9}, \
    {    831,11}, {    223,10}, {    447, 9}, {    895,12}, \
    {    127,11}, {    255,10}, {    543,11}, {    287,10}, \
    {    575,11}, {    351,10}, {    703, 9}, {   1407,12}, \
    {    191,11}, {    415,10}, {    831,11}, {    447,10}, \
    {    895,13}, {    127,12}, {    255,11}, {    543,10}, \
    {   1087,11}, {    575,12}, {    319,11}, {    703,10}, \
    {   1407,12}, {    383,11}, {    831,12}, {    447,11}, \
    {    895,10}, {   1791,11}, {    959,13}, {    255,12}, \
    {    511,11}, {   1087,12}, {    575,11}, {   1215,12}, \
    {    703,11}, {   1407,13}, {    383,12}, {    895,11}, \
    {   1791,12}, {    959,14}, {    255,13}, {    511,12}, \
    {   1215,13}, {    639,12}, {   1407,13}, {    895,12}, \
    {   1919,14}, {    511,13}, {   1023,12}, {   2047,13}, \
    {   1151,12}, {   2303,13}, {   1407,14}, {    767,13}, \
    {   1919,10}, {  15359,12}, {   4096,13}, {   8192,14}, \
    {  16384,15}, {  32768,16} }
#define MUL_FFT_TABLE3_SIZE 154
#define MUL_FFT_THRESHOLD                 2688

#define SQR_FFT_MODF_THRESHOLD             184  /* k = 5 */
#define SQR_FFT_TABLE3                                      \
  { {    184, 5}, {      6, 4}, {     13, 5}, {     13, 6}, \
    {      7, 5}, {     15, 6}, {     13, 7}, {      7, 6}, \
    {     16, 7}, {      9, 6}, {     19, 7}, {     11, 6}, \
    {     23, 7}, {     13, 8}, {      7, 7}, {     19, 8}, \
    {     11, 7}, {     25, 9}, {      7, 8}, {     15, 7}, \
    {     31, 8}, {     19, 7}, {     39, 8}, {     27, 9}, \
    {     15, 8}, {     39, 9}, {     23,10}, {     15, 9}, \
    {     31, 8}, {     63, 9}, {     39, 8}, {     79, 9}, \
    {     47, 8}, {     95,10}, {     31, 9}, {     63, 8}, \
    {    127, 7}, {    255, 9}, {     71, 8}, {    143, 7}, \
    {    287, 9}, {     79, 8}, {    159,10}, {     47, 9}, \
    {     95,11}, {     31,10}, {     63, 9}, {    127, 8}, \
    {    255, 9}, {    143, 8}, {    287, 7}, {    575,10}, \
    {     79, 9}, {    159, 8}, {    319, 9}, {    175, 8}, \
    {    351,10}, {     95, 9}, {    191, 8}, {    383, 9}, \
    {    207,10}, {    111,11}, {     63,10}, {    127, 9}, \
    {    255,10}, {    143, 9}, {    287, 8}, {    575,10}, \
    {    159, 9}, {    319,10}, {    175, 9}, {    351,11}, \
    {     95,10}, {    191, 9}, {    383,10}, {    207, 9}, \
    {    415, 8}, {    831,10}, {    223,12}, {     63,11}, \
    {    127,10}, {    255, 9}, {    511,10}, {    287, 9}, \
    {    575,11}, {    159,10}, {    351, 9}, {    703,11}, \
    {    191,10}, {    415, 9}, {    831,11}, {    223,10}, \
    {    447, 9}, {    895,12}, {    127,11}, {    255,10}, \
    {    511,11}, {    287,10}, {    575,11}, {    319,10}, \
    {    639,11}, {    351,10}, {    703, 9}, {   1407,12}, \
    {    191,11}, {    383,10}, {    767,11}, {    415,10}, \
    {    831,11}, {    447,10}, {    895,13}, {    127,12}, \
    {    255,11}, {    511,10}, {   1023,11}, {    575,12}, \
    {    319,11}, {    703,10}, {   1407,12}, {    383,11}, \
    {    831,12}, {    447,11}, {    895,10}, {   1791,11}, \
    {    959,13}, {    255,12}, {    511,11}, {   1023,12}, \
    {    575,11}, {   1215,12}, {    703,11}, {   1407,13}, \
    {    383,12}, {    895,11}, {   1791,12}, {    959,14}, \
    {    255,13}, {    511,12}, {   1215,13}, {    639,12}, \
    {   1471,13}, {    767,12}, {   1535,13}, {    895,12}, \
    {   1919,14}, {    511,13}, {   1151,12}, {   2431,13}, \
    {   1407,14}, {    767,13}, {   1919,15}, {  32768,16} }
#define SQR_FFT_TABLE3_SIZE 152
#define SQR_FFT_THRESHOLD                 1728

#define MULLO_BASECASE_THRESHOLD             0  /* always */
#define MULLO_DC_THRESHOLD                  33
#define MULLO_MUL_N_THRESHOLD             5240

#define DC_DIV_QR_THRESHOLD                 31
#define DC_DIVAPPR_Q_THRESHOLD             108
#define DC_BDIV_QR_THRESHOLD                35
#define DC_BDIV_Q_THRESHOLD                 88

#define INV_MULMOD_BNM1_THRESHOLD           42
#define INV_NEWTON_THRESHOLD               149
#define INV_APPR_THRESHOLD                 125

#define BINV_NEWTON_THRESHOLD              156
#define REDC_1_TO_REDC_N_THRESHOLD          39

#define MU_DIV_QR_THRESHOLD                807
#define MU_DIVAPPR_Q_THRESHOLD             807
#define MUPI_DIV_QR_THRESHOLD               66
#define MU_BDIV_QR_THRESHOLD               667
#define MU_BDIV_Q_THRESHOLD                807

#define MATRIX22_STRASSEN_THRESHOLD         11
#define HGCD_THRESHOLD                      87
#define GCD_DC_THRESHOLD                   233
#define GCDEXT_DC_THRESHOLD                198
#define JACOBI_BASE_METHOD                   1

#define GET_STR_DC_THRESHOLD                12
#define GET_STR_PRECOMPUTE_THRESHOLD        28
#define SET_STR_DC_THRESHOLD               390
#define SET_STR_PRECOMPUTE_THRESHOLD       814
