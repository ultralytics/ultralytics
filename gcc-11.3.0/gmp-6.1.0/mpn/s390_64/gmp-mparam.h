/* S/390-64 gmp-mparam.h -- Compiler/machine parameter header file.

Copyright 1991, 1993, 1994, 2000-2011 Free Software Foundation, Inc.

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

/* 1200 MHz z990 */

#define DIVREM_1_NORM_THRESHOLD              0  /* always */
#define DIVREM_1_UNNORM_THRESHOLD            0  /* always */
#define MOD_1_1P_METHOD                      2
#define MOD_1_NORM_THRESHOLD                 0  /* always */
#define MOD_1_UNNORM_THRESHOLD               0  /* always */
#define MOD_1N_TO_MOD_1_1_THRESHOLD          9
#define MOD_1U_TO_MOD_1_1_THRESHOLD          6
#define MOD_1_1_TO_MOD_1_2_THRESHOLD        62
#define MOD_1_2_TO_MOD_1_4_THRESHOLD         0  /* never mpn_mod_1s_2p */
#define PREINV_MOD_1_TO_MOD_1_THRESHOLD     17
#define USE_PREINV_DIVREM_1                  1
#define DIV_QR_1N_PI1_METHOD                 1
#define DIV_QR_1_NORM_THRESHOLD              1
#define DIV_QR_1_UNNORM_THRESHOLD        MP_SIZE_T_MAX  /* never */
#define DIV_QR_2_PI2_THRESHOLD           MP_SIZE_T_MAX  /* never */
#define DIVEXACT_1_THRESHOLD                 0  /* always */
#define BMOD_1_TO_MOD_1_THRESHOLD           98

#define MUL_TOOM22_THRESHOLD                10
#define MUL_TOOM33_THRESHOLD                41
#define MUL_TOOM44_THRESHOLD               105
#define MUL_TOOM6H_THRESHOLD               149
#define MUL_TOOM8H_THRESHOLD               212

#define MUL_TOOM32_TO_TOOM43_THRESHOLD      65
#define MUL_TOOM32_TO_TOOM53_THRESHOLD      69
#define MUL_TOOM42_TO_TOOM53_THRESHOLD      72
#define MUL_TOOM42_TO_TOOM63_THRESHOLD      64
#define MUL_TOOM43_TO_TOOM54_THRESHOLD      55

#define SQR_BASECASE_THRESHOLD               0  /* always (native) */
#define SQR_TOOM2_THRESHOLD                 16
#define SQR_TOOM3_THRESHOLD                 57
#define SQR_TOOM4_THRESHOLD                153
#define SQR_TOOM6_THRESHOLD                204
#define SQR_TOOM8_THRESHOLD                309

#define MULMID_TOOM42_THRESHOLD             20

#define MULMOD_BNM1_THRESHOLD               10
#define SQRMOD_BNM1_THRESHOLD               11

#define MUL_FFT_MODF_THRESHOLD             220  /* k = 5 */
#define MUL_FFT_TABLE3                                      \
  { {    220, 5}, {     11, 6}, {      6, 5}, {     13, 6}, \
    {      7, 5}, {     15, 6}, {     13, 7}, {      7, 6}, \
    {     15, 7}, {      8, 6}, {     17, 7}, {      9, 6}, \
    {     19, 7}, {     13, 8}, {      7, 7}, {     17, 8}, \
    {      9, 7}, {     19, 8}, {     13, 9}, {      7, 8}, \
    {     19, 9}, {     11, 8}, {     23,10}, {      7, 9}, \
    {     15, 8}, {     33, 9}, {     19, 8}, {     39, 9}, \
    {     23,10}, {     15, 9}, {     39,10}, {     23,11}, \
    {     15,10}, {     31, 9}, {     63,10}, {     39, 9}, \
    {     79,10}, {     47,11}, {     31,10}, {     63, 9}, \
    {    127, 8}, {    255,10}, {     71, 9}, {    143, 8}, \
    {    287,10}, {     79,11}, {     47,12}, {     31,11}, \
    {     63,10}, {    127, 9}, {    255, 8}, {    511,10}, \
    {    143, 9}, {    287,11}, {     79,10}, {    159, 9}, \
    {    319,10}, {    175, 9}, {    351, 8}, {    703, 7}, \
    {   1407,11}, {     95,10}, {    191, 9}, {    383,10}, \
    {    207,11}, {    111,10}, {    223,12}, {     63,11}, \
    {    127,10}, {    255, 9}, {    511,11}, {    143,10}, \
    {    287, 9}, {    575, 8}, {   1151,10}, {    319,11}, \
    {    175,10}, {    351, 9}, {    703,12}, {     95,11}, \
    {    191,10}, {    383, 9}, {    767,11}, {    207,10}, \
    {    415,11}, {    223,13}, {   8192,14}, {  16384,15}, \
    {  32768,16}, {  65536,17}, { 131072,18}, { 262144,19}, \
    { 524288,20}, {1048576,21}, {2097152,22}, {4194304,23}, \
    {8388608,24} }
#define MUL_FFT_TABLE3_SIZE 97
#define MUL_FFT_THRESHOLD                 1728

#define SQR_FFT_MODF_THRESHOLD             212  /* k = 5 */
#define SQR_FFT_TABLE3                                      \
  { {    212, 5}, {      7, 4}, {     15, 5}, {     13, 6}, \
    {      7, 5}, {     15, 6}, {     15, 7}, {      8, 6}, \
    {     17, 7}, {     13, 8}, {      7, 7}, {     17, 8}, \
    {      9, 7}, {     19, 8}, {     11, 7}, {     23, 8}, \
    {     13, 9}, {      7, 8}, {     19, 9}, {     11, 8}, \
    {     25,10}, {      7, 9}, {     15, 8}, {     31, 9}, \
    {     19, 8}, {     39, 9}, {     23,10}, {     15, 9}, \
    {     39,10}, {     23,11}, {     15,10}, {     31, 9}, \
    {     63,10}, {     47,11}, {     31,10}, {     63, 9}, \
    {    127, 8}, {    255,10}, {     71, 9}, {    143, 8}, \
    {    287,10}, {     79,11}, {     47,12}, {     31,11}, \
    {     63,10}, {    127, 9}, {    255, 8}, {    511,10}, \
    {    143, 9}, {    287,11}, {     79,10}, {    159, 9}, \
    {    319, 8}, {    639,10}, {    175, 9}, {    351, 8}, \
    {    703,10}, {    191, 9}, {    383, 8}, {    767,10}, \
    {    207, 9}, {    415,11}, {    111,10}, {    223,12}, \
    {     63,11}, {    127,10}, {    255, 9}, {    511,11}, \
    {    143,10}, {    287, 9}, {    575, 8}, {   1151,11}, \
    {    159,10}, {    319,11}, {    175,10}, {    351, 9}, \
    {    703,11}, {    191,10}, {    383,11}, {    207,10}, \
    {    415,11}, {    223,13}, {   8192,14}, {  16384,15}, \
    {  32768,16}, {  65536,17}, { 131072,18}, { 262144,19}, \
    { 524288,20}, {1048576,21}, {2097152,22}, {4194304,23}, \
    {8388608,24} }
#define SQR_FFT_TABLE3_SIZE 93
#define SQR_FFT_THRESHOLD                 1600

#define MULLO_BASECASE_THRESHOLD             2
#define MULLO_DC_THRESHOLD                  33
#define MULLO_MUL_N_THRESHOLD             3176

#define DC_DIV_QR_THRESHOLD                 28
#define DC_DIVAPPR_Q_THRESHOLD             107
#define DC_BDIV_QR_THRESHOLD                31
#define DC_BDIV_Q_THRESHOLD                 78

#define INV_MULMOD_BNM1_THRESHOLD           43
#define INV_NEWTON_THRESHOLD               129
#define INV_APPR_THRESHOLD                 117

#define BINV_NEWTON_THRESHOLD              149
#define REDC_1_TO_REDC_N_THRESHOLD          38

#define MU_DIV_QR_THRESHOLD                748
#define MU_DIVAPPR_Q_THRESHOLD             748
#define MUPI_DIV_QR_THRESHOLD               65
#define MU_BDIV_QR_THRESHOLD               562
#define MU_BDIV_Q_THRESHOLD                734

#define POWM_SEC_TABLE  4,23,274,961,2783

#define MATRIX22_STRASSEN_THRESHOLD         11
#define HGCD_THRESHOLD                      79
#define HGCD_APPR_THRESHOLD                 70
#define HGCD_REDUCE_THRESHOLD             1094
#define GCD_DC_THRESHOLD                   183
#define GCDEXT_DC_THRESHOLD                148
#define JACOBI_BASE_METHOD                   4

#define GET_STR_DC_THRESHOLD                30
#define GET_STR_PRECOMPUTE_THRESHOLD        41
#define SET_STR_DC_THRESHOLD               402
#define SET_STR_PRECOMPUTE_THRESHOLD      1104

#define FAC_DSC_THRESHOLD                  842
#define FAC_ODD_THRESHOLD                    0  /* always */
