dnl  Alpha mpn_invert_limb -- Invert a normalized limb.

dnl  Copyright 1996, 2000-2003, 2007, 2011, 2013 Free Software Foundation, Inc.

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

C      cycles/limb
C EV4:     ?
C EV5:   137/140  (with BWX/without BWX)
C EV6:    71/72   (with BWX/without BWX)

C This was compiler generated, with minimal manual edits.  Surely several
C cycles could be cut with some thought.

ASM_START()
PROLOGUE(mpn_invert_limb,gp)
	LEA(	r2, approx_tab)
	srl	r16, 54, r1
	srl	r16, 24, r4
	and	r16, 1, r5
	bic	r1, 1, r7
	lda	r4, 1(r4)
	srl	r16, 1, r3
	addq	r7, r2, r1
ifelse(bwx_available_p,1,`
	ldwu	r0, -512(r1)
',`
	ldq_u	r0, -512(r1)
	extwl	r0, r7, r0
')
	addq	r3, r5, r3
	mull	r0, r0, r1
	sll	r0, 11, r0
	mulq	r1, r4, r1
	srl	r1, 40, r1
	subq	r0, r1, r0
	lda	r0, -1(r0)
	mulq	r0, r0, r2
	sll	r0, 60, r1
	sll	r0, 13, r0
	mulq	r2, r4, r2
	subq	r1, r2, r1
	srl	r1, 47, r1
	addq	r0, r1, r0
	mulq	r0, r3, r3
	srl	r0, 1, r1
	cmoveq	r5, 0, r1
	subq	r1, r3, r1
	umulh	r1, r0, r3
	sll	r0, 31, r0
	srl	r3, 1, r1
	addq	r0, r1, r0
	mulq	r0, r16, r2
	umulh	r0, r16, r3
	addq	r2, r16, r1
	addq	r3, r16, r16
	cmpult	r1, r2, r1
	addq	r16, r1, r3
	subq	r0, r3, r0
	ret	r31, (r26), 1
EPILOGUE()
DATASTART(approx_tab,8)
forloop(i,256,512-1,dnl
`	.word	eval(0x7fd00/i)
')dnl
	SIZE(approx_tab, 512)
	TYPE(approx_tab, object)
DATAEND()
ASM_END()
