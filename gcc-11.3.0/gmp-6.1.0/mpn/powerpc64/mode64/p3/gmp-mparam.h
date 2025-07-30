/* POWER3/PowerPC630 gmp-mparam.h -- Compiler/machine parameter header file.

Copyright 2008-2010 Free Software Foundation, Inc.

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


#define MOD_1_NORM_THRESHOLD                 0  /* always */
#define MOD_1_UNNORM_THRESHOLD               0  /* always */
#define MOD_1N_TO_MOD_1_1_THRESHOLD          7
#define MOD_1U_TO_MOD_1_1_THRESHOLD          3
#define MOD_1_1_TO_MOD_1_2_THRESHOLD        18
#define MOD_1_2_TO_MOD_1_4_THRESHOLD         0  /* never mpn_mod_1s_2p */
#define PREINV_MOD_1_TO_MOD_1_THRESHOLD     17
#define USE_PREINV_DIVREM_1                  0
#define DIV_QR_2_PI2_THRESHOLD           MP_SIZE_T_MAX  /* never */
#define DIVEXACT_1_THRESHOLD                 0  /* always (native) */
#define BMOD_1_TO_MOD_1_THRESHOLD        MP_SIZE_T_MAX  /* never */

#define MUL_TOOM22_THRESHOLD                10
#define MUL_TOOM33_THRESHOLD                33
#define MUL_TOOM44_THRESHOLD                46
#define MUL_TOOM6H_THRESHOLD                77
#define MUL_TOOM8H_THRESHOLD               139

#define MUL_TOOM32_TO_TOOM43_THRESHOLD      49
#define MUL_TOOM32_TO_TOOM53_THRESHOLD      47
#define MUL_TOOM42_TO_TOOM53_THRESHOLD      49
#define MUL_TOOM42_TO_TOOM63_THRESHOLD      49
#define MUL_TOOM43_TO_TOOM54_THRESHOLD      34

#define SQR_BASECASE_THRESHOLD               0  /* always (native) */
#define SQR_TOOM2_THRESHOLD                 14
#define SQR_TOOM3_THRESHOLD                 45
#define SQR_TOOM4_THRESHOLD                 64
#define SQR_TOOM6_THRESHOLD                 85
#define SQR_TOOM8_THRESHOLD                139

#define MULMID_TOOM42_THRESHOLD             22

#define MULMOD_BNM1_THRESHOLD                8
#define SQRMOD_BNM1_THRESHOLD               10

#define MUL_FFT_MODF_THRESHOLD             220  /* k = 5 */
#define MUL_FFT_TABLE3                                      \
  { {    220, 5}, {      9, 6}, {      5, 5}, {     11, 6}, \
    {     13, 7}, {      7, 6}, {     15, 7}, {     13, 8}, \
    {      7, 7}, {     15, 8}, {     13, 9}, {      7, 8}, \
    {     19, 9}, {     11, 8}, {     23,10}, {      7, 9}, \
    {     15, 8}, {     33, 9}, {     23,10}, {     15, 9}, \
    {     35, 8}, {     71,10}, {     23, 9}, {     47,11}, \
    {     15,10}, {     31, 9}, {     71,10}, {     39, 9}, \
    {     79,10}, {     55,11}, {     31,10}, {     63, 9}, \
    {    127,10}, {     71, 9}, {    143, 8}, {    287,10}, \
    {     79,11}, {     47,10}, {     95, 9}, {    191,12}, \
    {     31,11}, {     63,10}, {    127, 9}, {    255, 8}, \
    {    511,10}, {    143, 9}, {    287,11}, {     79,10}, \
    {    159, 9}, {    319, 8}, {    639,10}, {    175, 9}, \
    {    351,11}, {     95,10}, {    191, 9}, {    383,11}, \
    {    111,10}, {    223,12}, {     63,11}, {    127,10}, \
    {    255, 9}, {    511,11}, {    143,10}, {    287, 9}, \
    {    575,11}, {    159,10}, {    319, 9}, {    639,11}, \
    {    175,10}, {    351,12}, {     95,11}, {    191,10}, \
    {    383, 9}, {    767,11}, {    223,13}, {     63,12}, \
    {    127,11}, {    255,10}, {    511,11}, {    287,10}, \
    {    575, 9}, {   1151,12}, {    159,11}, {    319,10}, \
    {    639,11}, {    351,12}, {    191,11}, {    383,10}, \
    {    767,12}, {    223,11}, {    447,10}, {    895,13}, \
    {    127,12}, {    255,11}, {    511,12}, {    287,11}, \
    {    575,10}, {   1151,12}, {    319,11}, {    639,12}, \
    {    351,11}, {    703,13}, {    191,12}, {    383,11}, \
    {    767,12}, {    415,11}, {    831,10}, {   1663,12}, \
    {    447,11}, {    895,14}, {  16384,15}, {  32768,16}, \
    {  65536,17}, { 131072,18}, { 262144,19}, { 524288,20}, \
    {1048576,21}, {2097152,22}, {4194304,23}, {8388608,24} }
#define MUL_FFT_TABLE3_SIZE 120
#define MUL_FFT_THRESHOLD                 2688

#define SQR_FFT_MODF_THRESHOLD             188  /* k = 5 */
#define SQR_FFT_TABLE3                                      \
  { {    188, 5}, {      9, 6}, {      5, 5}, {     11, 6}, \
    {     13, 7}, {     13, 8}, {      7, 7}, {     16, 8}, \
    {      9, 7}, {     19, 8}, {     13, 9}, {      7, 8}, \
    {     19, 9}, {     11, 8}, {     23,10}, {      7, 9}, \
    {     15, 8}, {     31, 9}, {     19, 8}, {     39, 9}, \
    {     23,10}, {     15, 9}, {     39,10}, {     23,11}, \
    {     15,10}, {     31, 9}, {     67,10}, {     39, 9}, \
    {     79, 8}, {    159,10}, {     47, 9}, {     95, 8}, \
    {    191,11}, {     31,10}, {     63, 9}, {    127, 8}, \
    {    255,10}, {     71, 9}, {    143, 8}, {    287,10}, \
    {     79, 9}, {    159,11}, {     47,10}, {     95, 9}, \
    {    191,12}, {     31,11}, {     63,10}, {    127, 9}, \
    {    255, 8}, {    511,10}, {    143, 9}, {    287,11}, \
    {     79,10}, {    159, 9}, {    319, 8}, {    639,10}, \
    {    175,11}, {     95,10}, {    191, 9}, {    383,11}, \
    {    111,10}, {    223,12}, {     63,11}, {    127,10}, \
    {    255, 9}, {    511,11}, {    143,10}, {    287, 9}, \
    {    575,11}, {    159,10}, {    319, 9}, {    639,11}, \
    {    175,12}, {     95,11}, {    191,10}, {    383, 9}, \
    {    767,11}, {    223,13}, {     63,12}, {    127,11}, \
    {    255,10}, {    511,11}, {    287,10}, {    575,12}, \
    {    159,11}, {    319,10}, {    639,11}, {    351,12}, \
    {    191,11}, {    383,10}, {    767,12}, {    223,11}, \
    {    447,10}, {    895,13}, {    127,12}, {    255,11}, \
    {    511,12}, {    287,11}, {    575,10}, {   1151,12}, \
    {    319,11}, {    639,12}, {    351,13}, {    191,12}, \
    {    383,11}, {    767,12}, {    447,11}, {    895,14}, \
    {  16384,15}, {  32768,16}, {  65536,17}, { 131072,18}, \
    { 262144,19}, { 524288,20}, {1048576,21}, {2097152,22}, \
    {4194304,23}, {8388608,24} }
#define SQR_FFT_TABLE3_SIZE 118
#define SQR_FFT_THRESHOLD                 1728

#define MULLO_BASECASE_THRESHOLD             2
#define MULLO_DC_THRESHOLD                  27
#define MULLO_MUL_N_THRESHOLD             2511

#define DC_DIV_QR_THRESHOLD                 23
#define DC_DIVAPPR_Q_THRESHOLD              87
#define DC_BDIV_QR_THRESHOLD                27
#define DC_BDIV_Q_THRESHOLD                 60

#define INV_MULMOD_BNM1_THRESHOLD           27
#define INV_NEWTON_THRESHOLD                91
#define INV_APPR_THRESHOLD                  91

#define BINV_NEWTON_THRESHOLD              115
#define REDC_1_TO_REDC_N_THRESHOLD          31

#define MU_DIV_QR_THRESHOLD                551
#define MU_DIVAPPR_Q_THRESHOLD             551
#define MUPI_DIV_QR_THRESHOLD               42
#define MU_BDIV_QR_THRESHOLD               483
#define MU_BDIV_Q_THRESHOLD                492

#define POWM_SEC_TABLE  2,23,140,556,713,746

#define MATRIX22_STRASSEN_THRESHOLD          8
#define HGCD_THRESHOLD                      56
#define HGCD_APPR_THRESHOLD                 51
#define HGCD_REDUCE_THRESHOLD              688
#define GCD_DC_THRESHOLD                   333
#define GCDEXT_DC_THRESHOLD                126
#define JACOBI_BASE_METHOD                   1

#define GET_STR_DC_THRESHOLD                17
#define GET_STR_PRECOMPUTE_THRESHOLD        28
#define SET_STR_DC_THRESHOLD               375
#define SET_STR_PRECOMPUTE_THRESHOLD       812

#define FAC_DSC_THRESHOLD                  351
#define FAC_ODD_THRESHOLD                    0  /* always */
