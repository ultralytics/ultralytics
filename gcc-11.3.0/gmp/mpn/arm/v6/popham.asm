dnl  ARM mpn_popcount and mpn_hamdist.

dnl  Contributed to the GNU project by Torbjörn Granlund.

dnl  Copyright 2013 Free Software Foundation, Inc.

dnl  This file is part of the GNU MP Library.
dnl
dnl  The GNU MP Library is free software; you can redistribute it and/or modify
dnl  it under the terms of either:
dnl
dnl    * the GNU Lesser General Public License as published by the Free
dnl      Software Foundation; either version 3 of the License, or (at your
dnl      option) any later version.
dnl
dnl  or
dnl
dnl    * the GNU General Public License as published by the Free Software
dnl      Foundation; either version 2 of the License, or (at your option) any
dnl      later version.
dnl
dnl  or both in parallel, as here.
dnl
dnl  The GNU MP Library is distributed in the hope that it will be useful, but
dnl  WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
dnl  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
dnl  for more details.
dnl
dnl  You should have received copies of the GNU General Public License and the
dnl  GNU Lesser General Public License along with the GNU MP Library.  If not,
dnl  see https://www.gnu.org/licenses/.

include(`../config.m4')

C		     popcount	      hamdist
C		    cycles/limb	    cycles/limb
C StrongARM		 -
C XScale		 -
C Cortex-A7		 ?
C Cortex-A8		 ?
C Cortex-A9		 8.94		 9.47
C Cortex-A15		 5.67		 6.44

C Architecture requirements:
C v5	-
C v5t	-
C v5te	ldrd strd
C v6	usada8
C v6t2	-
C v7a	-

ifdef(`OPERATION_popcount',`
  define(`func',`mpn_popcount')
  define(`ap',		`r0')
  define(`n',		`r1')
  define(`a0',		`r2')
  define(`a1',		`r3')
  define(`s',		`r5')
  define(`b_01010101',	`r6')
  define(`b_00110011',	`r7')
  define(`b_00001111',	`r8')
  define(`zero',	`r9')
  define(`POPC',	`$1')
  define(`HAMD',	`dnl')
')
ifdef(`OPERATION_hamdist',`
  define(`func',`mpn_hamdist')
  define(`ap',		`r0')
  define(`bp',		`r1')
  define(`n',		`r2')
  define(`a0',		`r6')
  define(`a1',		`r7')
  define(`b0',		`r4')
  define(`b1',		`r5')
  define(`s',		`r11')
  define(`b_01010101',	`r8')
  define(`b_00110011',	`r9')
  define(`b_00001111',	`r10')
  define(`zero',	`r3')
  define(`POPC',	`dnl')
  define(`HAMD',	`$1')
')

MULFUNC_PROLOGUE(mpn_popcount mpn_hamdist)

ASM_START()
PROLOGUE(func)
POPC(`	push	{ r4-r9 }	')
HAMD(`	push	{ r4-r11 }	')

	ldr	b_01010101, =0x55555555
	mov	r12, #0
	ldr	b_00110011, =0x33333333
	mov	zero, #0
	ldr	b_00001111, =0x0f0f0f0f

	tst	n, #1
	beq	L(evn)

L(odd):	ldr	a1, [ap], #4		C 1 x 32 1-bit accumulators, 0-1
HAMD(`	ldr	b1, [bp], #4	')	C 1 x 32 1-bit accumulators, 0-1
HAMD(`	eor	a1, a1, b1	')
	and	r4, b_01010101, a1, lsr #1
	sub	a1, a1, r4
	and	r4, a1, b_00110011
	bic	r5, a1, b_00110011
	add	r5, r4, r5, lsr #2	C 8 4-bit accumulators, 0-4
	subs	n, n, #1
	b	L(mid)

L(evn):	mov	s, #0

L(top):	ldrd	a0, a1, [ap], #8	C 2 x 32 1-bit accumulators, 0-1
HAMD(`	ldrd	b0, b1, [bp], #8')
HAMD(`	eor	a0, a0, b0	')
HAMD(`	eor	a1, a1, b1	')
	subs	n, n, #2
	usada8	r12, s, zero, r12
	and	r4, b_01010101, a0, lsr #1
	sub	a0, a0, r4
	and	r4, b_01010101, a1, lsr #1
	sub	a1, a1, r4
	and	r4, a0, b_00110011
	bic	r5, a0, b_00110011
	add	a0, r4, r5, lsr #2	C 8 4-bit accumulators, 0-4
	and	r4, a1, b_00110011
	bic	r5, a1, b_00110011
	add	a1, r4, r5, lsr #2	C 8 4-bit accumulators, 0-4
	add	r5, a0, a1		C 8 4-bit accumulators, 0-8
L(mid):	and	r4, r5, b_00001111
	bic	r5, r5, b_00001111
	add	s, r4, r5, lsr #4	C 4 8-bit accumulators
	bne	L(top)

	usada8	r0, s, zero, r12
POPC(`	pop	{ r4-r9 }	')
HAMD(`	pop	{ r4-r11 }	')
	bx	r14
EPILOGUE()
