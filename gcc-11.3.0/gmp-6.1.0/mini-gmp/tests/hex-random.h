/*

Copyright 2011, Free Software Foundation, Inc.

This file is part of the GNU MP Library test suite.

The GNU MP Library test suite is free software; you can redistribute it
and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 3 of the License,
or (at your option) any later version.

The GNU MP Library test suite is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
the GNU MP Library test suite.  If not, see https://www.gnu.org/licenses/.  */

enum hex_random_op
  {
    OP_ADD, OP_SUB, OP_MUL, OP_SQR,
    OP_CDIV, OP_FDIV, OP_TDIV,
    OP_CDIV_Q_2, OP_CDIV_R_2,
    OP_FDIV_Q_2, OP_FDIV_R_2,
    OP_TDIV_Q_2,  OP_TDIV_R_2,
    OP_GCD, OP_LCM, OP_POWM, OP_AND, OP_IOR, OP_XOR,
    OP_SETBIT, OP_CLRBIT, OP_COMBIT,
    OP_SCAN0, OP_SCAN1,
  };

void hex_random_init (void);
char *hex_urandomb (unsigned long bits);
char *hex_rrandomb (unsigned long bits);
char *hex_rrandomb_export (void *dst, size_t *countp,
			   int order, size_t size, int endian,
			   unsigned long bits);

void hex_random_op2 (enum hex_random_op op,  unsigned long maxbits,
		     char **ap, char **rp);
void hex_random_op3 (enum hex_random_op op,  unsigned long maxbits,
		     char **ap, char **bp, char **rp);
void hex_random_op4 (enum hex_random_op op,  unsigned long maxbits,
		     char **ap, char **bp, char **rp, char **qp);
void hex_random_bit_op (enum hex_random_op op, unsigned long maxbits,
			char **ap, unsigned long *b, char **rp);
void hex_random_scan_op (enum hex_random_op op, unsigned long maxbits,
			char **ap, unsigned long *b, unsigned long *r);
void hex_random_str_op (unsigned long maxbits,
			int base, char **ap, char **rp);
