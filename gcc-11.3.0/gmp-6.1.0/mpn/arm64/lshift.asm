dnl  ARM64 mpn_lshift.

dnl  Copyright 2013, 2014 Free Software Foundation, Inc.

dnl  This file is part of the GNU MP Library.

dnl  The GNU MP Library is free software; you can redistribute it and/or modify
dnl  it under the terms of the GNU Lesser General Public License as published
dnl  by the Free Software Foundation; either version 3 of the License, or (at
dnl  your option) any later version.

dnl  The GNU MP Library is distributed in the hope that it will be useful, but
dnl  WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
dnl  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
dnl  License for more details.

dnl  You should have received a copy of the GNU Lesser General Public License
dnl  along with the GNU MP Library.  If not, see http://www.gnu.org/licenses/.

include(`../config.m4')

C	     cycles/limb
C Cortex-A53	 ?
C Cortex-A57	 ?

changecom(@&*$)

define(`rp_arg', `x0')
define(`up',     `x1')
define(`n',      `x2')
define(`cnt',    `x3')

define(`rp',     `x16')

define(`tnc',`x8')

ASM_START()
PROLOGUE(mpn_lshift)
	add	rp, rp_arg, n, lsl #3
	add	up, up, n, lsl #3
	sub	tnc, xzr, cnt
	tbz	n, #0, L(bx0)

L(bx1):	ldr	x4, [up,#-8]
	tbnz	n, #1, L(b11)

L(b01):	lsr	x0, x4, tnc
	lsl	x18, x4, cnt
	sub	n, n, #1
	cbnz	n, L(gt1)
	str	x18, [rp,#-8]
	ret
L(gt1):	ldp	x4, x5, [up,#-24]
	sub	up, up, #8
	add	rp, rp, #16
	b	L(lo2)

L(b11):	lsr	x0, x4, tnc
	lsl	x9, x4, cnt
	ldp	x6, x7, [up,#-24]
	add	n, n, #1
	add	up, up, #8
	add	rp, rp, #32
	b	L(lo0)

L(bx0):	ldp	x4, x5, [up,#-16]
	tbz	n, #1, L(b00)

L(b10):	lsr	x0, x5, tnc
	lsl	x13, x5, cnt
	lsr	x10, x4, tnc
	lsl	x18, x4, cnt
	sub	n, n, #2
	cbnz	n, L(gt2)
	orr	x10, x10, x13
	stp	x18, x10, [rp,#-16]
	ret
L(gt2):	ldp	x4, x5, [up,#-32]
	orr	x10, x10, x13
	str	x10, [rp,#-8]
	sub	up, up, #16
	add	rp, rp, #8
	b	L(lo2)

L(b00):	lsr	x0, x5, tnc
	lsl	x13, x5, cnt
	lsr	x10, x4, tnc
	lsl	x9, x4, cnt
	ldp	x6, x7, [up,#-32]
	orr	x10, x10, x13
	str	x10, [rp,#-8]
	add	rp, rp, #24
	b	L(lo0)

	ALIGN(16)
L(top):	ldp	x4, x5, [up,#-48]
	sub	rp, rp, #32		C integrate with stp?
	sub	up, up, #32		C integrate with ldp?
	orr	x11, x11, x9
	orr	x10, x10, x13
	stp	x10, x11, [rp,#-16]
L(lo2):	lsr	x11, x5, tnc
	lsl	x13, x5, cnt
	lsr	x10, x4, tnc
	lsl	x9, x4, cnt
	ldp	x6, x7, [up,#-32]
	orr	x11, x11, x18
	orr	x10, x10, x13
	stp	x10, x11, [rp,#-32]
L(lo0):	sub	n, n, #4
	lsr	x11, x7, tnc
	lsl	x13, x7, cnt
	lsr	x10, x6, tnc
	lsl	x18, x6, cnt
	cbnz	n, L(top)

L(end):	orr	x11, x11, x9
	orr	x10, x10, x13
	stp	x10, x11, [rp,#-48]
	str	x18, [rp,#-56]
	ret
EPILOGUE()
