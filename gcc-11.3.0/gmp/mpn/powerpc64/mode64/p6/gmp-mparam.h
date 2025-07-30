/* POWER6 gmp-mparam.h -- Compiler/machine parameter header file.

Copyright 1991, 1993, 1994, 1999-2003, 2009-2011 Free Software Foundation, Inc.

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

/* 3500 MHz POWER6 (kolga.bibsys.no) */

#define MOD_1_NORM_THRESHOLD                 0  /* always */
#define MOD_1_UNNORM_THRESHOLD               0  /* always */
#define MOD_1N_TO_MOD_1_1_THRESHOLD          3
#define MOD_1U_TO_MOD_1_1_THRESHOLD          3
#define MOD_1_1_TO_MOD_1_2_THRESHOLD        12
#define MOD_1_2_TO_MOD_1_4_THRESHOLD         0  /* never mpn_mod_1s_2p */
#define PREINV_MOD_1_TO_MOD_1_THRESHOLD      6
#define USE_PREINV_DIVREM_1                  0
#define DIV_QR_2_PI2_THRESHOLD           MP_SIZE_T_MAX  /* never */
#define DIVEXACT_1_THRESHOLD                 0  /* always (native) */
#define BMOD_1_TO_MOD_1_THRESHOLD           21

#define MUL_TOOM22_THRESHOLD                20
#define MUL_TOOM33_THRESHOLD                50
#define MUL_TOOM44_THRESHOLD               106
#define MUL_TOOM6H_THRESHOLD               274
#define MUL_TOOM8H_THRESHOLD               339

#define MUL_TOOM32_TO_TOOM43_THRESHOLD      62
#define MUL_TOOM32_TO_TOOM53_THRESHOLD      76
#define MUL_TOOM42_TO_TOOM53_THRESHOLD      73
#define MUL_TOOM42_TO_TOOM63_THRESHOLD      66
#define MUL_TOOM43_TO_TOOM54_THRESHOLD      88

#define SQR_BASECASE_THRESHOLD               0  /* always (native) */
#define SQR_TOOM2_THRESHOLD                 24
#define SQR_TOOM3_THRESHOLD                 49
#define SQR_TOOM4_THRESHOLD                130
#define SQR_TOOM6_THRESHOLD                226
#define SQR_TOOM8_THRESHOLD                272

#define MULMID_TOOM42_THRESHOLD             36

#define MULMOD_BNM1_THRESHOLD               14
#define SQRMOD_BNM1_THRESHOLD               14

#define MUL_FFT_MODF_THRESHOLD             380  /* k = 5 */
#define MUL_FFT_TABLE3                                      \
  { {    340, 5}, {     19, 6}, {     10, 5}, {     21, 6}, \
    {     11, 5}, {     23, 6}, {     21, 7}, {     11, 6}, \
    {     23, 7}, {     12, 6}, {     25, 7}, {     21, 8}, \
    {     11, 7}, {     24, 8}, {     13, 7}, {     27, 8}, \
    {     21, 9}, {     11, 8}, {     25, 9}, {     15, 8}, \
    {     33, 9}, {     23, 8}, {     47, 9}, {     27,10}, \
    {     15, 9}, {     39,10}, {     23, 9}, {     47,11}, \
    {     15,10}, {     31, 9}, {     63,10}, {     47,11}, \
    {     31,10}, {     71,11}, {     47,12}, {     31,11}, \
    {     63,10}, {    127, 9}, {    255, 8}, {    511,10}, \
    {    135, 9}, {    271,11}, {     79, 9}, {    319, 8}, \
    {    639,10}, {    175,11}, {     95,10}, {    191, 9}, \
    {    383,10}, {    207,12}, {     63,10}, {    255, 9}, \
    {    511,10}, {    271, 9}, {    543,11}, {    143,10}, \
    {    287, 9}, {    575,10}, {    303, 9}, {    607,10}, \
    {    319, 9}, {    639,11}, {    175,12}, {     95,11}, \
    {    191,10}, {    383,11}, {    207,10}, {    415,13}, \
    {   8192,14}, {  16384,15}, {  32768,16}, {  65536,17}, \
    { 131072,18}, { 262144,19}, { 524288,20}, {1048576,21}, \
    {2097152,22}, {4194304,23}, {8388608,24} }
#define MUL_FFT_TABLE3_SIZE 79
#define MUL_FFT_THRESHOLD                 3520

#define SQR_FFT_MODF_THRESHOLD             308  /* k = 5 */
#define SQR_FFT_TABLE3                                      \
  { {    280, 5}, {     17, 6}, {      9, 5}, {     19, 6}, \
    {     21, 7}, {     11, 6}, {     23, 7}, {     21, 8}, \
    {     11, 7}, {     24, 8}, {     13, 7}, {     27, 8}, \
    {     21, 9}, {     11, 8}, {     25, 9}, {     15, 8}, \
    {     33, 9}, {     19, 8}, {     39, 9}, {     23, 8}, \
    {     47, 9}, {     27,10}, {     15, 9}, {     39,10}, \
    {     23, 9}, {     47,11}, {     15,10}, {     31, 9}, \
    {     63,10}, {     47,11}, {     31,10}, {     71, 9}, \
    {    143,11}, {     47,12}, {     31,11}, {     63, 9}, \
    {    255, 8}, {    511, 9}, {    271,10}, {    143,11}, \
    {     79,10}, {    159, 9}, {    319,10}, {    175, 9}, \
    {    351,11}, {     95,10}, {    191, 9}, {    383,10}, \
    {    207,12}, {     63,11}, {    127,10}, {    255, 9}, \
    {    511, 8}, {   1023,10}, {    271, 9}, {    543,11}, \
    {    143,10}, {    287, 9}, {    575,11}, {    159,10}, \
    {    319, 9}, {    639,11}, {    175,10}, {    351,12}, \
    {     95,11}, {    191,10}, {    383,11}, {    207,10}, \
    {    415,13}, {   8192,14}, {  16384,15}, {  32768,16}, \
    {  65536,17}, { 131072,18}, { 262144,19}, { 524288,20}, \
    {1048576,21}, {2097152,22}, {4194304,23}, {8388608,24} }
#define SQR_FFT_TABLE3_SIZE 80
#define SQR_FFT_THRESHOLD                 2752

#define MULLO_BASECASE_THRESHOLD             5
#define MULLO_DC_THRESHOLD                  62
#define MULLO_MUL_N_THRESHOLD             2995

#define DC_DIV_QR_THRESHOLD                 59
#define DC_DIVAPPR_Q_THRESHOLD             200
#define DC_BDIV_QR_THRESHOLD                70
#define DC_BDIV_Q_THRESHOLD                168

#define INV_MULMOD_BNM1_THRESHOLD           53
#define INV_NEWTON_THRESHOLD               170
#define INV_APPR_THRESHOLD                 166

#define BINV_NEWTON_THRESHOLD              220
#define REDC_1_TO_REDC_N_THRESHOLD          67

#define MU_DIV_QR_THRESHOLD                998
#define MU_DIVAPPR_Q_THRESHOLD             942
#define MUPI_DIV_QR_THRESHOLD               57
#define MU_BDIV_QR_THRESHOLD               889
#define MU_BDIV_Q_THRESHOLD               1078

#define POWM_SEC_TABLE  4,26,216,804,1731

#define MATRIX22_STRASSEN_THRESHOLD         13
#define HGCD_THRESHOLD                     106
#define HGCD_APPR_THRESHOLD                109
#define HGCD_REDUCE_THRESHOLD             2205
#define GCD_DC_THRESHOLD                   492
#define GCDEXT_DC_THRESHOLD                327
#define JACOBI_BASE_METHOD                   4

#define GET_STR_DC_THRESHOLD                16
#define GET_STR_PRECOMPUTE_THRESHOLD        28
#define SET_STR_DC_THRESHOLD               537
#define SET_STR_PRECOMPUTE_THRESHOLD      1576

#define FAC_DSC_THRESHOLD                  426
#define FAC_ODD_THRESHOLD                    0  /* always */
