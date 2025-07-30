dnl  ARM Neon mpn_rsh1add_n, mpn_rsh1sub_n.

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

C	     cycles/limb
C StrongARM	 -
C XScale	 -
C Cortex-A7	 ?
C Cortex-A8	 ?
C Cortex-A9	4-5
C Cortex-A15	 2.5

C TODO
C  * Try to make this smaller, its size (384 bytes) is excessive.
C  * Try to reach 2.25 c/l on A15, to match the addlsh_1 family.
C  * This is ad-hoc scheduled, perhaps unnecessarily so for A15, and perhaps
C    insufficiently for A7 and A8.

define(`rp', `r0')
define(`up', `r1')
define(`vp', `r2')
define(`n',  `r3')

ifdef(`OPERATION_rsh1add_n', `
  define(`ADDSUBS',	`adds	$1, $2, $3')
  define(`ADCSBCS',	`adcs	$1, $2, $3')
  define(`IFADD',	`$1')
  define(`IFSUB',	`')
  define(`func',	mpn_rsh1add_n)')
ifdef(`OPERATION_rsh1sub_n', `
  define(`ADDSUBS',	`subs	$1, $2, $3')
  define(`ADCSBCS',	`sbcs	$1, $2, $3')
  define(`IFADD',	`')
  define(`IFSUB',	`$1')
  define(`func',	mpn_rsh1sub_n)')

MULFUNC_PROLOGUE(mpn_rsh1add_n mpn_rsh1sub_n)

ASM_START()
PROLOGUE(func)
	push	 {r4-r10}

	ands	r4, n, #3
	beq	L(b00)
	cmp	r4, #2
	blo	L(b01)
	beq	L(b10)

L(b11):	ldmia	 up!, {r9,r10,r12}
	ldmia	 vp!, {r5,r6,r7}
	ADDSUBS( r9, r9, r5)
	vmov	 d4, r9, r9
	ADCSBCS( r10, r10, r6)
	ADCSBCS( r12, r12, r7)
	vshr.u64 d3, d4, #1
	vmov	 d1, r10, r12
	vsli.u64 d3, d1, #31
	vshr.u64 d2, d1, #1
	vst1.32	 d3[0], [rp]!
	bics	 n, n, #3
	beq	 L(wd2)
L(gt3):	ldmia	 up!, {r8,r9,r10,r12}
	ldmia	 vp!, {r4,r5,r6,r7}
	b	 L(mi0)

L(b10):	ldmia	 up!, {r10,r12}
	ldmia	 vp!, {r6,r7}
	ADDSUBS( r10, r10, r6)
	ADCSBCS( r12, r12, r7)
	vmov	 d4, r10, r12
	bics	 n, n, #2
	vshr.u64 d2, d4, #1
	beq	 L(wd2)
L(gt2):	ldmia	 up!, {r8,r9,r10,r12}
	ldmia	 vp!, {r4,r5,r6,r7}
	b	 L(mi0)

L(b01):	ldr	 r12, [up], #4
	ldr	 r7, [vp], #4
	ADDSUBS( r12, r12, r7)
	vmov	 d4, r12, r12
	bics	 n, n, #1
	bne	 L(gt1)
	mov	 r5, r12, lsr #1
IFADD(`	adc	 r1, n, #0')
IFSUB(`	adc	 r1, n, #1')
	bfi	 r5, r1, #31, #1
	str	 r5, [rp]
	and	 r0, r12, #1
	pop	 {r4-r10}
	bx	 r14
L(gt1):	ldmia	 up!, {r8,r9,r10,r12}
	ldmia	 vp!, {r4,r5,r6,r7}
	vshr.u64 d2, d4, #1
	ADCSBCS( r8, r8, r4)
	ADCSBCS( r9, r9, r5)
	vmov	 d0, r8, r9
	ADCSBCS( r10, r10, r6)
	ADCSBCS( r12, r12, r7)
	vsli.u64 d2, d0, #31
	vshr.u64 d3, d0, #1
	vst1.32	 d2[0], [rp]!
	b	 L(mi1)

L(b00):	ldmia	 up!, {r8,r9,r10,r12}
	ldmia	 vp!, {r4,r5,r6,r7}
	ADDSUBS( r8, r8, r4)
	ADCSBCS( r9, r9, r5)
	vmov	 d4, r8, r9
	ADCSBCS( r10, r10, r6)
	ADCSBCS( r12, r12, r7)
	vshr.u64 d3, d4, #1
	b	 L(mi1)

	ALIGN(16)
L(top):	ldmia	 up!, {r8,r9,r10,r12}
	ldmia	 vp!, {r4,r5,r6,r7}
	vsli.u64 d3, d1, #63
	vshr.u64 d2, d1, #1
	vst1.32	 d3, [rp]!
L(mi0):	ADCSBCS( r8, r8, r4)
	ADCSBCS( r9, r9, r5)
	vmov	 d0, r8, r9
	ADCSBCS( r10, r10, r6)
	ADCSBCS( r12, r12, r7)
	vsli.u64 d2, d0, #63
	vshr.u64 d3, d0, #1
	vst1.32	 d2, [rp]!
L(mi1):	vmov	 d1, r10, r12
	sub	 n, n, #4
	tst	 n, n
	bne	 L(top)

L(end):	vsli.u64 d3, d1, #63
	vshr.u64 d2, d1, #1
	vst1.32	 d3, [rp]!
L(wd2):	vmov	 r4, r5, d2
IFADD(`	adc	 r1, n, #0')
IFSUB(`	adc	 r1, n, #1')
	bfi	 r5, r1, #31, #1
	stm	 rp, {r4,r5}

L(rtn):	vmov.32	 r0, d4[0]
	and	 r0, r0, #1
	pop	 {r4-r10}
	bx	 r14
EPILOGUE()
