/* PowerPC-32 gmp-mparam.h -- Compiler/machine parameter header file.

Copyright 1991, 1993, 1994, 1999-2004, 2008-2011 Free Software Foundation, Inc.

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

/* 3500 MHz POWER6 */

#define DIVREM_1_NORM_THRESHOLD              0  /* always */
#define DIVREM_1_UNNORM_THRESHOLD            0  /* always */
#define MOD_1_1P_METHOD                      2
#define MOD_1_NORM_THRESHOLD                 3
#define MOD_1_UNNORM_THRESHOLD               0  /* always */
#define MOD_1N_TO_MOD_1_1_THRESHOLD          3
#define MOD_1U_TO_MOD_1_1_THRESHOLD          3
#define MOD_1_1_TO_MOD_1_2_THRESHOLD     MP_SIZE_T_MAX
#define MOD_1_2_TO_MOD_1_4_THRESHOLD         0  /* never mpn_mod_1s_2p */
#define PREINV_MOD_1_TO_MOD_1_THRESHOLD      8
#define USE_PREINV_DIVREM_1                  1
#define DIV_QR_2_PI2_THRESHOLD           MP_SIZE_T_MAX  /* never */
#define DIVEXACT_1_THRESHOLD                 0  /* always */
#define BMOD_1_TO_MOD_1_THRESHOLD        MP_SIZE_T_MAX  /* never */

#define MUL_TOOM22_THRESHOLD                19
#define MUL_TOOM33_THRESHOLD                55
#define MUL_TOOM44_THRESHOLD                88
#define MUL_TOOM6H_THRESHOLD               137
#define MUL_TOOM8H_THRESHOLD               181

#define MUL_TOOM32_TO_TOOM43_THRESHOLD      57
#define MUL_TOOM32_TO_TOOM53_THRESHOLD      56
#define MUL_TOOM42_TO_TOOM53_THRESHOLD      57
#define MUL_TOOM42_TO_TOOM63_THRESHOLD      56

#define SQR_BASECASE_THRESHOLD               0  /* always */
#define SQR_TOOM2_THRESHOLD                 30
#define SQR_TOOM3_THRESHOLD                 56
#define SQR_TOOM4_THRESHOLD                130
#define SQR_TOOM6_THRESHOLD                189
#define SQR_TOOM8_THRESHOLD                296

#define MULMID_TOOM42_THRESHOLD             26

#define MULMOD_BNM1_THRESHOLD                7
#define SQRMOD_BNM1_THRESHOLD               12

#define POWM_SEC_TABLE  2,26,127,453,1068

#define MUL_FFT_MODF_THRESHOLD             212  /* k = 5 */
#define MUL_FFT_TABLE3                                      \
  { {    212, 5}, {     13, 6}, {      7, 5}, {     15, 6}, \
    {     13, 7}, {      7, 6}, {     16, 7}, {      9, 6}, \
    {     19, 7}, {     13, 8}, {      7, 7}, {     19, 8}, \
    {     11, 7}, {     25, 9}, {      7, 8}, {     15, 7}, \
    {     31, 8}, {     19, 7}, {     39, 8}, {     23, 9}, \
    {     15, 8}, {     39, 9}, {     23, 8}, {     47,10}, \
    {     15, 9}, {     31, 8}, {     63, 9}, {     39, 8}, \
    {     79, 9}, {     47,10}, {     31, 9}, {     63, 8}, \
    {    127, 9}, {     71, 8}, {    143, 7}, {    287, 9}, \
    {     79,10}, {     47,11}, {     31,10}, {     63, 9}, \
    {    127, 8}, {    255, 7}, {    511, 9}, {    143, 8}, \
    {    287,10}, {     79, 9}, {    159, 8}, {    319, 9}, \
    {    175, 8}, {    351,10}, {     95, 9}, {    191, 8}, \
    {    383, 9}, {    207,10}, {    111,11}, {     63,10}, \
    {    127, 9}, {    255, 8}, {    511,10}, {    143, 9}, \
    {    287, 8}, {    575,10}, {    159, 9}, {    319,10}, \
    {    175, 9}, {    351,11}, {     95,10}, {    191, 9}, \
    {    383,10}, {    207, 9}, {    415,12}, {     63,11}, \
    {    127,10}, {    255, 9}, {    511,10}, {    287, 9}, \
    {    575,11}, {    159,10}, {    351, 9}, {    703,11}, \
    {    191,10}, {    415, 9}, {    831,11}, {    223,10}, \
    {    447,12}, {   4096,13}, {   8192,14}, {  16384,15}, \
    {  32768,16} }
#define MUL_FFT_TABLE3_SIZE 89
#define MUL_FFT_THRESHOLD                 1728

#define SQR_FFT_MODF_THRESHOLD             184  /* k = 5 */
#define SQR_FFT_TABLE3                                      \
  { {    184, 5}, {      6, 4}, {     13, 5}, {     13, 6}, \
    {      7, 5}, {     15, 6}, {     13, 7}, {      7, 6}, \
    {     16, 7}, {      9, 6}, {     19, 7}, {     11, 6}, \
    {     23, 7}, {     13, 8}, {      7, 7}, {     19, 8}, \
    {     11, 7}, {     23, 9}, {      7, 8}, {     23, 9}, \
    {     15, 8}, {     39, 9}, {     23,10}, {     15, 9}, \
    {     31, 8}, {     63, 9}, {     39, 8}, {     79, 9}, \
    {     47,10}, {     31, 9}, {     63, 8}, {    127, 7}, \
    {    255, 9}, {     71, 8}, {    143, 7}, {    287, 6}, \
    {    575, 9}, {     79,10}, {     47,11}, {     31,10}, \
    {     63, 9}, {    127, 8}, {    255, 9}, {    143, 8}, \
    {    287, 7}, {    575,10}, {     79, 9}, {    159, 8}, \
    {    319, 9}, {    175, 8}, {    351,10}, {     95, 9}, \
    {    191, 8}, {    383, 9}, {    207,10}, {    111, 9}, \
    {    223,11}, {     63,10}, {    127, 9}, {    255,10}, \
    {    143, 9}, {    287, 8}, {    575,10}, {    159, 9}, \
    {    319,10}, {    175, 9}, {    351,11}, {     95,10}, \
    {    191, 9}, {    383,10}, {    207, 9}, {    415,10}, \
    {    223,12}, {     63,11}, {    127,10}, {    255, 9}, \
    {    511,10}, {    287, 9}, {    575,11}, {    159,10}, \
    {    351, 9}, {    703, 8}, {   1407,11}, {    191,10}, \
    {    415,11}, {    223,10}, {    447, 9}, {    895,12}, \
    {   4096,13}, {   8192,14}, {  16384,15}, {  32768,16} }
#define SQR_FFT_TABLE3_SIZE 92
#define SQR_FFT_THRESHOLD                 1600

#define MULLO_BASECASE_THRESHOLD             2
#define MULLO_DC_THRESHOLD                  57
#define MULLO_MUL_N_THRESHOLD             3176

#define DC_DIV_QR_THRESHOLD                 52
#define DC_DIVAPPR_Q_THRESHOLD             187
#define DC_BDIV_QR_THRESHOLD                64
#define DC_BDIV_Q_THRESHOLD                146

#define INV_MULMOD_BNM1_THRESHOLD           68
#define INV_NEWTON_THRESHOLD               182
#define INV_APPR_THRESHOLD                 182

#define BINV_NEWTON_THRESHOLD              186
#define REDC_1_TO_REDC_N_THRESHOLD          60

#define MU_DIV_QR_THRESHOLD                924
#define MU_DIVAPPR_Q_THRESHOLD             807
#define MUPI_DIV_QR_THRESHOLD               73
#define MU_BDIV_QR_THRESHOLD               667
#define MU_BDIV_Q_THRESHOLD                823

#define MATRIX22_STRASSEN_THRESHOLD          8
#define HGCD_THRESHOLD                      61
#define HGCD_APPR_THRESHOLD                 50
#define HGCD_REDUCE_THRESHOLD              974
#define GCD_DC_THRESHOLD                   195
#define GCDEXT_DC_THRESHOLD                134
#define JACOBI_BASE_METHOD                   4

#define GET_STR_DC_THRESHOLD                 9
#define GET_STR_PRECOMPUTE_THRESHOLD        21
#define SET_STR_DC_THRESHOLD               190
#define SET_STR_PRECOMPUTE_THRESHOLD       411
