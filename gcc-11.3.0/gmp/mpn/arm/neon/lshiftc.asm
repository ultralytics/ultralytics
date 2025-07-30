dnl  ARM Neon mpn_lshiftc.

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

C	     cycles/limb     cycles/limb     cycles/limb      good
C              aligned	      unaligned	      best seen	     for cpu?
C StrongARM	 -		 -
C XScale	 -		 -
C Cortex-A7	 ?		 ?
C Cortex-A8	 ?		 ?
C Cortex-A9	 3.5		 3.5				Y
C Cortex-A15	 1.75		 1.75				Y


C We read 64 bits at a time at 32-bit aligned addresses, and except for the
C first and last store, we write using 64-bit aligned addresses.  All shifting
C is done on 64-bit words in 'extension' registers.
C
C It should be possible to read also using 64-bit alignment, by manipulating
C the shift count for unaligned operands.  Not done, since it does not seem to
C matter for A9 or A15.
C
C This will not work in big-endian mode.

C TODO
C  * Try using 128-bit operations.  Note that Neon lacks pure 128-bit shifts,
C    which might make it tricky.
C  * Clean up and simplify.
C  * Consider sharing most of the code for lshift and rshift, since the feed-in
C    code, the loop, and most of the wind-down code are identical.
C  * Replace the basecase code with code using 'extension' registers.
C  * Optimise.  It is not clear that this loop insn permutation is optimal for
C    either A9 or A15.

C INPUT PARAMETERS
define(`rp',  `r0')
define(`ap',  `r1')
define(`n',   `r2')
define(`cnt', `r3')

	define(`IFLSH', `$1')
	define(`IFRSH', `')
	define(`X',`0')
	define(`Y',`1')
	define(`func',`mpn_lshiftc')
define(`OPERATION_lshiftc',1)

ASM_START()
	TEXT
	ALIGN(64)
PROLOGUE(mpn_lshiftc)
IFLSH(`	mov	r12, n, lsl #2	')
IFLSH(`	add	rp, rp, r12	')
IFLSH(`	add	ap, ap, r12	')

	cmp	n, #4			C SIMD code n limit
	ble	L(base)

ifdef(`OPERATION_lshiftc',`
	vdup.32	d6, r3			C left shift count is positive
	sub	r3, r3, #64		C right shift count is negative
	vdup.32	d7, r3
	mov	r12, #-8')		C lshift pointer update offset
ifdef(`OPERATION_rshift',`
	rsb	r3, r3, #0		C right shift count is negative
	vdup.32	d6, r3
	add	r3, r3, #64		C left shift count is positive
	vdup.32	d7, r3
	mov	r12, #8')		C rshift pointer update offset

IFLSH(`	sub	ap, ap, #8	')
	vld1.32	{d19}, [ap], r12	C load initial 2 limbs
	vshl.u64 d18, d19, d7		C retval

	tst	rp, #4			C is rp 64-bit aligned already?
	beq	L(rp_aligned)		C yes, skip
	vmvn	 d19, d19
IFLSH(`	add	ap, ap, #4	')	C move back ap pointer
IFRSH(`	sub	ap, ap, #4	')	C move back ap pointer
	vshl.u64 d4, d19, d6
	sub	n, n, #1		C first limb handled
IFLSH(`	sub	 rp, rp, #4	')
	vst1.32	 {d4[Y]}, [rp]IFRSH(!)	C store first limb, rp gets aligned
	vld1.32	 {d19}, [ap], r12	C load ap[1] and ap[2]

L(rp_aligned):
IFLSH(`	sub	rp, rp, #8	')
	subs	n, n, #6
	vmvn	 d19, d19
	blt	L(two_or_three_more)
	tst	n, #2
	beq	L(2)

L(1):	vld1.32	 {d17}, [ap], r12
	vshl.u64 d5, d19, d6
	vmvn	 d17, d17
	vld1.32	 {d16}, [ap], r12
	vshl.u64 d0, d17, d7
	vshl.u64 d4, d17, d6
	sub	n, n, #2
	b	 L(mid)

L(2):	vld1.32	 {d16}, [ap], r12
	vshl.u64 d4, d19, d6
	vmvn	 d16, d16
	vld1.32	 {d17}, [ap], r12
	vshl.u64 d1, d16, d7
	vshl.u64 d5, d16, d6
	subs	n, n, #4
	blt	L(end)

L(top):	vmvn	 d17, d17
	vld1.32	 {d16}, [ap], r12
	vorr	 d2, d4, d1
	vshl.u64 d0, d17, d7
	vshl.u64 d4, d17, d6
	vst1.32	 {d2}, [rp:64], r12
L(mid):	vmvn	 d16, d16
	vld1.32	 {d17}, [ap], r12
	vorr	 d3, d5, d0
	vshl.u64 d1, d16, d7
	vshl.u64 d5, d16, d6
	vst1.32	 {d3}, [rp:64], r12
	subs	n, n, #4
	bge	L(top)

L(end):	tst	 n, #1
	beq	 L(evn)

	vorr	 d2, d4, d1
	vst1.32	 {d2}, [rp:64], r12
	b	 L(cj1)

L(evn):	vmvn	 d17, d17
	vorr	 d2, d4, d1
	vshl.u64 d0, d17, d7
	vshl.u64 d4, d17, d6
	vst1.32	 {d2}, [rp:64], r12
	vmov.u8	 d17, #255
	vorr	 d2, d5, d0
	vshl.u64 d0, d17, d7
	vorr	 d3, d4, d0
	b	 L(cj2)

C Load last 2 - 3 limbs, store last 4 - 5 limbs
L(two_or_three_more):
	tst	n, #1
	beq	L(l2)

L(l3):	vshl.u64 d5, d19, d6
	vld1.32	 {d17}, [ap], r12
L(cj1):	vmov.u8	 d16, #0
IFLSH(`	add	 ap, ap, #4	')
	vmvn	 d17, d17
	vld1.32	 {d16[Y]}, [ap], r12
	vshl.u64 d0, d17, d7
	vshl.u64 d4, d17, d6
	vmvn	 d16, d16
	vorr	 d3, d5, d0
	vshl.u64 d1, d16, d7
	vshl.u64 d5, d16, d6
	vst1.32	 {d3}, [rp:64], r12
	vorr	 d2, d4, d1
	vst1.32	 {d2}, [rp:64], r12
IFLSH(`	add	 rp, rp, #4	')
	vst1.32	 {d5[Y]}, [rp]
	vmov.32	 r0, d18[X]
	bx	lr

L(l2):	vld1.32	 {d16}, [ap], r12
	vshl.u64 d4, d19, d6
	vmvn	 d16, d16
	vshl.u64 d1, d16, d7
	vshl.u64 d5, d16, d6
	vmov.u8	 d17, #255
	vorr	 d2, d4, d1
	vshl.u64 d0, d17, d7
	vorr	 d3, d5, d0
L(cj2):	vst1.32	 {d2}, [rp:64], r12
	vst1.32	 {d3}, [rp]
	vmov.32	 r0, d18[X]
	bx	lr


define(`tnc', `r12')
L(base):
	push	{r4, r6, r7, r8}
	ldr	r4, [ap, #-4]!
	rsb	tnc, cnt, #32
	mvn	r6, r4

	mov	r7, r6, lsl cnt
	tst	n, #1
	beq	L(ev)			C n even

L(od):	subs	n, n, #2
	bcc	L(ed1)			C n = 1
	ldr	r8, [ap, #-4]!
	mvn	r8, r8
	b	L(md)			C n = 3

L(ev):	ldr	r6, [ap, #-4]!
	mvn	r6, r6
	subs	n, n, #2
	beq	L(ed)			C n = 3
					C n = 4
L(tp):	ldr	r8, [ap, #-4]!
	orr	r7, r7, r6, lsr tnc
	str	r7, [rp, #-4]!
	mvn	r8, r8
	mov	r7, r6, lsl cnt
L(md):	ldr	r6, [ap, #-4]!
	orr	r7, r7, r8, lsr tnc
	str	r7, [rp, #-4]!
	mvn	r6, r6
	mov	r7, r8, lsl cnt

L(ed):	orr	r7, r7, r6, lsr tnc
	str	r7, [rp, #-4]!
	mov	r7, r6, lsl cnt
L(ed1):	mvn	r6, #0
	orr	r7, r7, r6, lsr tnc
	str	r7, [rp, #-4]
	mov	r0, r4, lsr tnc
	pop	{r4, r6, r7, r8}
	bx	r14
EPILOGUE()
