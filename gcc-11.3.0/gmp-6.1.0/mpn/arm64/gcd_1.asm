dnl  ARM v6t2 mpn_gcd_1.

dnl  Based on the K7 gcd_1.asm, by Kevin Ryde.  Rehacked for ARM by Torbjorn
dnl  Granlund.

dnl  Copyright 2000-2002, 2005, 2009, 2011-2013 Free Software Foundation, Inc.

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

changecom(@&*$)

C	     cycles/bit (approx)
C Cortex-A53	 ?
C Cortex-A57	 ?

C TODO
C  * Optimise inner-loop better.
C  * Push saving/restoring of callee-user regs into call code

C Threshold of when to call bmod when U is one limb.  Should be about
C (time_in_cycles(bmod_1,1) + call_overhead) / (cycles/bit).
define(`BMOD_THRES_LOG2', 7)

C INPUT PARAMETERS
define(`up',    `x0')
define(`n',     `x1')
define(`v0',    `x2')

ifdef(`BMOD_1_TO_MOD_1_THRESHOLD',,
  `define(`BMOD_1_TO_MOD_1_THRESHOLD',30)')

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_gcd_1)
	stp	x29, x30, [sp,#-32]!
	ldr	x3, [up]		C U low limb
	stp     x19, x20, [sp,#16]

	orr	x3, x3, v0
	rbit	x4, x3
	clz	x20, x4			C min(ctz(u0),ctz(v0))

	rbit	x12, v0
	clz	x12, x12
	lsr	v0, v0, x12

	mov	x19, v0

	cmp	n, #1
	b.ne	L(nby1)

C Both U and V are single limbs, reduce with bmod if u0 >> v0.
	ldr	x3, [up]
	cmp	v0, x3, lsr #BMOD_THRES_LOG2
	b.hi	L(red1)

L(bmod):mov	x3, #0			C carry argument
	bl	mpn_modexact_1c_odd
	b	L(red0)

L(nby1):cmp	n, #BMOD_1_TO_MOD_1_THRESHOLD
	b.lo	L(bmod)

	bl	mpn_mod_1

L(red0):mov	x3, x0
L(red1):cmp	x3, #0
	rbit	x12, x3
	clz	x12, x12
	b.ne	L(mid)
	b	L(end)

	ALIGN(8)
L(top):
ifelse(1,1,`
C This shorter variant makes full use of armv8 insns
	csneg	x3, x1, x1, cs		C if x-y < 0
	csel	x19, x4, x19, cs	C use x,y-x
L(mid):	lsr	x4, x3, x12		C
	subs	x1, x19, x4		C
',`
C This variant is akin to the 32-bit v6t2 code
	csel	x3, x1, x3, cs		C if x-y < 0
	csel	x19, x0, x19, cs	C use x,y-x
L(mid):	lsr	x3, x3, x12		C
	mov	x0, x3			C
	subs	x1, x19, x3		C
	sub	x3, x3, x19		C
')
	rbit	x12, x1
	clz	x12, x12		C
	b.ne	L(top)			C

L(end):	lsl	x0, x19, x20
	ldp     x19, x20, [sp,#16]
	ldp	x29, x30, [sp],#32
	ret
EPILOGUE()
