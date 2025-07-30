dnl  ARM mpn_lshiftc.

dnl  Contributed to the GNU project by Torbjörn Granlund.

dnl  Copyright 2012 Free Software Foundation, Inc.

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
C StrongARM	 ?
C XScale	 ?
C Cortex-A7	 ?
C Cortex-A8	 ?
C Cortex-A9	 4.0
C Cortex-A15	 ?

define(`rp',  `r0')
define(`up',  `r1')
define(`n',   `r2')
define(`cnt', `r3')
define(`tnc', `r12')

ASM_START()
PROLOGUE(mpn_lshiftc)
	add	up, up, n, lsl #2
	push	{r4, r6, r7, r8}
	ldr	r4, [up, #-4]!
	add	rp, rp, n, lsl #2
	rsb	tnc, cnt, #32
	mvn	r6, r4

	mov	r7, r6, lsl cnt
	tst	n, #1
	beq	L(evn)			C n even

L(odd):	subs	n, n, #2
	bcc	L(1)			C n = 1
	ldr	r8, [up, #-4]!
	mvn	r8, r8
	b	L(mid)

L(evn):	ldr	r6, [up, #-4]!
	mvn	r6, r6
	subs	n, n, #2
	beq	L(end)

L(top):	ldr	r8, [up, #-4]!
	orr	r7, r7, r6, lsr tnc
	str	r7, [rp, #-4]!
	mvn	r8, r8
	mov	r7, r6, lsl cnt
L(mid):	ldr	r6, [up, #-4]!
	orr	r7, r7, r8, lsr tnc
	str	r7, [rp, #-4]!
	mvn	r6, r6
	mov	r7, r8, lsl cnt
	subs	n, n, #2
	bgt	L(top)

L(end):	orr	r7, r7, r6, lsr tnc
	str	r7, [rp, #-4]!
	mov	r7, r6, lsl cnt
L(1):	mvn	r6, #0
	orr	r7, r7, r6, lsr tnc
	str	r7, [rp, #-4]
	mov	r0, r4, lsr tnc
	pop	{r4, r6, r7, r8}
	bx	r14
EPILOGUE()
