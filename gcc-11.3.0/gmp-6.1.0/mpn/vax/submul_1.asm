dnl  VAX mpn_submul_1 -- Multiply a limb vector with a limb and subtract the
dnl  result from a second limb vector.

dnl  Copyright 1992, 1994, 1996, 2000, 2012 Free Software Foundation, Inc.

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

ASM_START()
PROLOGUE(mpn_submul_1)
	.word	0xfc0
	movl	12(ap), r4
	movl	8(ap), r8
	movl	4(ap), r9
	clrl	r3
	incl	r4
	ashl	$-1, r4, r7
	clrl	r11
	movl	16(ap), r6
	jlss	L(v0_big)
	jlbc	r4, L(1)

C Loop for v0 < 0x80000000
L(tp1):	movl	(r8)+, r1
	jlss	L(1n0)
	emul	r1, r6, $0, r2
	addl2	r11, r2
	adwc	$0, r3
	subl2	r2, (r9)+
	adwc	$0, r3
L(1):	movl	(r8)+, r1
	jlss	L(1n1)
L(1p1):	emul	r1, r6, $0, r10
	addl2	r3, r10
	adwc	$0, r11
	subl2	r10, (r9)+
	adwc	$0, r11

	sobgtr	r7, L(tp1)
	movl	r11, r0
	ret

L(1n0):	emul	r1, r6, $0, r2
	addl2	r11, r2
	adwc	r6, r3
	subl2	r2, (r9)+
	adwc	$0, r3
	movl	(r8)+, r1
	jgeq	L(1p1)
L(1n1):	emul	r1, r6, $0, r10
	addl2	r3, r10
	adwc	r6, r11
	subl2	r10, (r9)+
	adwc	$0, r11

	sobgtr	r7, L(tp1)
	movl	r11, r0
	ret

L(v0_big):
	jlbc	r4, L(2)

C Loop for v0 >= 0x80000000
L(tp2):	movl	(r8)+, r1
	jlss	L(2n0)
	emul	r1, r6, $0, r2
	addl2	r11, r2
	adwc	r1, r3
	subl2	r2, (r9)+
	adwc	$0, r3
L(2):	movl	(r8)+, r1
	jlss	L(2n1)
L(2p1):	emul	r1, r6, $0, r10
	addl2	r3, r10
	adwc	r1, r11
	subl2	r10, (r9)+
	adwc	$0, r11

	sobgtr	r7, L(tp2)
	movl	r11, r0
	ret

L(2n0):	emul	r1, r6, $0, r2
	addl2	r11, r2
	adwc	r6, r3
	subl2	r2, (r9)+
	adwc	r1, r3
	movl	(r8)+, r1
	jgeq	L(2p1)
L(2n1):	emul	r1, r6, $0, r10
	addl2	r3, r10
	adwc	r6, r11
	subl2	r10, (r9)+
	adwc	r1, r11

	sobgtr	r7, L(tp2)
	movl	r11, r0
	ret
EPILOGUE()
