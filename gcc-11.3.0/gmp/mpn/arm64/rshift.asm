dnl  ARM64 mpn_rshift.

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
PROLOGUE(mpn_rshift)
	mov	rp, rp_arg
	sub	tnc, xzr, cnt
	tbz	n, #0, L(bx0)

L(bx1):	ldr	x4, [up,#0]
	tbnz	n, #1, L(b11)

L(b01):	lsl	x0, x4, tnc
	lsr	x18, x4, cnt
	sub	n, n, #1
	cbnz	n, L(gt1)
	str	x18, [rp,#0]
	ret
L(gt1):	ldp	x5, x4, [up,#8]
	sub	up, up, #8
	sub	rp, rp, #32
	b	L(lo2)

L(b11):	lsl	x0, x4, tnc
	lsr	x9, x4, cnt
	ldp	x7, x6, [up,#8]
	add	n, n, #1
	sub	up, up, #24
	sub	rp, rp, #48
	b	L(lo0)

L(bx0):	ldp	x5, x4, [up,#0]
	tbz	n, #1, L(b00)

L(b10):	lsl	x0, x5, tnc
	lsr	x13, x5, cnt
	lsl	x10, x4, tnc
	lsr	x18, x4, cnt
	sub	n, n, #2
	cbnz	n, L(gt2)
	orr	x10, x10, x13
	stp	x10, x18, [rp,#0]
	ret
L(gt2):	ldp	x5, x4, [up,#16]
	orr	x10, x10, x13
	str	x10, [rp,#0]
	sub	rp, rp, #24
	b	L(lo2)

L(b00):	lsl	x0, x5, tnc
	lsr	x13, x5, cnt
	lsl	x10, x4, tnc
	lsr	x9, x4, cnt
	ldp	x7, x6, [up,#16]
	orr	x10, x10, x13
	str	x10, [rp,#0]
	sub	up, up, #16
	sub	rp, rp, #40
	b	L(lo0)

	ALIGN(16)
L(top):	ldp	x5, x4, [up,#48]
	add	rp, rp, #32		C integrate with stp?
	add	up, up, #32		C integrate with ldp?
	orr	x11, x11, x9
	orr	x10, x10, x13
	stp	x11, x10, [rp,#16]
L(lo2):	lsl	x11, x5, tnc
	lsr	x13, x5, cnt
	lsl	x10, x4, tnc
	lsr	x9, x4, cnt
	ldp	x7, x6, [up,#32]
	orr	x11, x11, x18
	orr	x10, x10, x13
	stp	x11, x10, [rp,#32]
L(lo0):	sub	n, n, #4
	lsl	x11, x7, tnc
	lsr	x13, x7, cnt
	lsl	x10, x6, tnc
	lsr	x18, x6, cnt
	cbnz	n, L(top)

L(end):	orr	x11, x11, x9
	orr	x10, x10, x13
	stp	x11, x10, [rp,#48]
	str	x18, [rp,#64]
	ret
EPILOGUE()
