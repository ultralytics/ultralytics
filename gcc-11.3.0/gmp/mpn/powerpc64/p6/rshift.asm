dnl  PowerPC-64 mpn_rshift -- rp[] = up[] << cnt

dnl  Copyright 2003, 2005, 2010, 2013 Free Software Foundation, Inc.

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

C		    cycles/limb
C POWER3/PPC630		 ?
C POWER4/PPC970		 ?
C POWER5		 2
C POWER6		 3.5  (mysteriously 3.0 for cnt=1)

C TODO
C  * Micro-optimise header code
C  * Perhaps do 4-way unrolling, for 2.5 c/l on POWER6.  The code is 4248
C    bytes, 4-way code would become about 50% larger.

C INPUT PARAMETERS
define(`rp_param',  `r3')
define(`up',  `r4')
define(`n',   `r5')
define(`cnt', `r6')

define(`tnc',`r0')
define(`retval',`r3')
define(`rp',  `r7')

ASM_START()
PROLOGUE(mpn_rshift,toc)

ifdef(`HAVE_ABI_mode32',`
	rldicl	n, n, 0,32		C FIXME: avoid this zero extend
')
	mflr	r12
	LEAL(	r11, L(e1))		C address of L(e1) label in SHIFT(1)
	sldi	r10, cnt, 6		C multiply cnt by size of a SHIFT block
	add	r11, r11, r10		C address of L(oN) for N = cnt
	srdi	r10, n, 1
	mr	rp, rp_param
	subfic	tnc, cnt, 64
	rlwinm.	r8, n, 0,31,31		C extract bit 0
	mtctr	r10
	beq	L(evn)

L(odd):	ld	r9, 0(up)
	cmpdi	cr0, n, 1		C n = 1?
	beq	L(1)
	ld	r8, 8(up)
	addi	r11, r11, -84		C L(o1) - L(e1) - 64
	mtlr	r11
	sld	r3, r9, tnc		C retval
	addi	up, up, 8
	addi	rp, rp, 8
	blr				C branch to L(oN)

L(evn):	ld	r8, 0(up)
	ld	r9, 8(up)
	addi	r11, r11, -64
	mtlr	r11
	sld	r3, r8, tnc		C retval
	addi	up, up, 16
	blr				C branch to L(eN)

L(1):	sld	r3, r9, tnc		C retval
	srd	r8, r9, cnt
	std	r8, 0(rp)
	mtlr	r12
ifdef(`HAVE_ABI_mode32',
`	mr	r4, r3
	srdi	r3, r3, 32
')
	blr


define(SHIFT,`
L(lo$1):ld	r8, 0(up)
	std	r11, 0(rp)
	addi	rp, rp, 16
L(o$1):	srdi	r10, r9, $1
	rldimi	r10, r8, eval(64-$1), 0
	ld	r9, 8(up)
	addi	up, up, 16
	std	r10, -8(rp)
L(e$1):	srdi	r11, r8, $1
	rldimi	r11, r9, eval(64-$1), 0
	bdnz	L(lo$1)
	std	r11, 0(rp)
	srdi	r10, r9, $1
	b	L(com)
	nop
	nop
')

	ALIGN(64)
forloop(`i',1,63,`SHIFT(i)')

L(com):	std	r10, 8(rp)
	mtlr	r12
ifdef(`HAVE_ABI_mode32',
`	mr	r4, r3
	srdi	r3, r3, 32
')
	blr
EPILOGUE()
ASM_END()
