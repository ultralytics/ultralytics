dnl  ARM mpn_addlshC_n, mpn_sublshC_n, mpn_rsblshC_n

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


C	     cycles/limb
C StrongARM	 -
C XScale	 -
C Cortex-A7	 ?
C Cortex-A8	 ?
C Cortex-A9	 5.25
C Cortex-A15	 2.25

C TODO
C  * Consider using 4-way feed-in code.
C  * This is ad-hoc scheduled, perhaps unnecessarily so for A15, and perhaps
C    insufficiently for A7 and A8.

define(`rp', `r0')
define(`up', `r1')
define(`vp', `r2')
define(`n',  `r3')

ifdef(`DO_add', `
  define(`ADCSBCS',	`adcs	$1, $2, $3')
  define(`CLRCY',	`cmn	r13, #1')
  define(`RETVAL',	`adc	r0, $1, #0')
  define(`func',	mpn_addlsh`'LSH`'_n)')
ifdef(`DO_sub', `
  define(`ADCSBCS',	`sbcs	$1, $2, $3')
  define(`CLRCY',	`cmp	r13, #0')
  define(`RETVAL',	`sbc	$2, $2, $2
			cmn	$2, #1
			adc	 r0, $1, #0')
  define(`func',	mpn_sublsh`'LSH`'_n)')
ifdef(`DO_rsb', `
  define(`ADCSBCS',	`sbcs	$1, $3, $2')
  define(`CLRCY',	`cmp	r13, #0')
  define(`RETVAL',	`sbc	r0, $1, #0')
  define(`func',	mpn_rsblsh`'LSH`'_n)')


ASM_START()
PROLOGUE(func)
	push	 {r4-r10}
	vmov.i8	 d0, #0			C could feed carry through here
	CLRCY
	tst	n, #1
	beq	L(bb0)

L(bb1):	vld1.32	 {d3[0]}, [vp]!
	vsli.u32 d0, d3, #LSH
	ldr	 r12, [up], #4
	vmov.32	 r5, d0[0]
	vshr.u32 d0, d3, #32-LSH
	ADCSBCS( r12, r12, r5)
	str	 r12, [rp], #4
	bics	 n, n, #1
	beq	 L(rtn)

L(bb0):	tst	n, #2
	beq	L(b00)

L(b10):	vld1.32	 {d3}, [vp]!
	vsli.u64 d0, d3, #LSH
	ldmia	 up!, {r10,r12}
	vmov	 r4, r5, d0
	vshr.u64 d0, d3, #64-LSH
	ADCSBCS( r10, r10, r4)
	ADCSBCS( r12, r12, r5)
	stmia	 rp!, {r10,r12}
	bics	 n, n, #2
	beq	 L(rtn)

L(b00):	vld1.32	 {d2}, [vp]!
	vsli.u64 d0, d2, #LSH
	vshr.u64 d1, d2, #64-LSH
	vld1.32	 {d3}, [vp]!
	vsli.u64 d1, d3, #LSH
	vmov	 r6, r7, d0
	vshr.u64 d0, d3, #64-LSH
	sub	 n, n, #4
	tst	 n, n
	beq	 L(end)

	ALIGN(16)
L(top):	ldmia	 up!, {r8,r9,r10,r12}
	vld1.32	 {d2}, [vp]!
	vsli.u64 d0, d2, #LSH
	vmov	 r4, r5, d1
	vshr.u64 d1, d2, #64-LSH
	ADCSBCS( r8, r8, r6)
	ADCSBCS( r9, r9, r7)
	vld1.32	 {d3}, [vp]!
	vsli.u64 d1, d3, #LSH
	vmov	 r6, r7, d0
	vshr.u64 d0, d3, #64-LSH
	ADCSBCS( r10, r10, r4)
	ADCSBCS( r12, r12, r5)
	stmia	 rp!, {r8,r9,r10,r12}
	sub	 n, n, #4
	tst	 n, n
	bne	 L(top)

L(end):	ldmia	 up!, {r8,r9,r10,r12}
	vmov	 r4, r5, d1
	ADCSBCS( r8, r8, r6)
	ADCSBCS( r9, r9, r7)
	ADCSBCS( r10, r10, r4)
	ADCSBCS( r12, r12, r5)
	stmia	 rp!, {r8,r9,r10,r12}
L(rtn):	vmov.32	 r0, d0[0]
	RETVAL(	 r0, r1)
	pop	 {r4-r10}
	bx	 r14
EPILOGUE()
