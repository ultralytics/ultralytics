dnl  ARM Neon mpn_popcount -- mpn bit population count.

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
C StrongARM:	 -
C XScale	 -
C Cortex-A7	 ?
C Cortex-A8	 ?
C Cortex-A9	 1.125
C Cortex-A15	 0.56

C TODO
C  * Explore using vldr and vldm.  Does it help on A9?  (These loads do
C    64-bits-at-a-time, which will mess up in big-endian mode.  Except not for
C    popcount. Except perhaps also for popcount for the edge loads.)
C  * Arrange to align the pointer, if that helps performance.  Use the same
C    read-and-mask trick we use on PCs, for simplicity and performance.  (Sorry
C    valgrind!)
C  * Explore if explicit align directives, e.g., "[ptr:128]" help.
C  * See rth's gmp-devel 2013-02/03 messages about final summation tricks.

C INPUT PARAMETERS
define(`ap', r0)
define(`n',  r1)

C We sum into 16 16-bit counters in q8,q9, but at the end we sum them and end
C up with 8 16-bit counters.  Therefore, we can sum to 8(2^16-1) bits, or
C (8*2^16-1)/32 = 0x3fff limbs.  We use a chunksize close to that, but which
C can be represented as a 8-bit ARM constant.
C
define(`chunksize',0x3f80)

ASM_START()
PROLOGUE(mpn_popcount)

	cmp	n, #chunksize
	bhi	L(gt16k)

L(lt16k):
	vmov.i64   q8, #0		C clear summation register
	vmov.i64   q9, #0		C clear summation register

	tst	   n, #1
	beq	   L(xxx0)
	vmov.i64   d0, #0
	sub	   n, n, #1
	vld1.32   {d0[0]}, [ap]!	C load 1 limb
	vcnt.8	   d24, d0
	vpadal.u8  d16, d24		C d16/q8 = 0; could just splat

L(xxx0):tst	   n, #2
	beq	   L(xx00)
	sub	   n, n, #2
	vld1.32    {d0}, [ap]!		C load 2 limbs
	vcnt.8	   d24, d0
	vpadal.u8  d16, d24

L(xx00):tst	   n, #4
	beq	   L(x000)
	sub	   n, n, #4
	vld1.32    {q0}, [ap]!		C load 4 limbs
	vcnt.8	   q12, q0
	vpadal.u8  q8, q12

L(x000):tst	   n, #8
	beq	   L(0000)

	subs	   n, n, #8
	vld1.32    {q0,q1}, [ap]!	C load 8 limbs
	bls	   L(sum)

L(gt8):	vld1.32    {q2,q3}, [ap]!	C load 8 limbs
	sub	   n, n, #8
	vcnt.8	   q12, q0
	vcnt.8	   q13, q1
	b	   L(mid)

L(0000):subs	   n, n, #16
	blo	   L(e0)

	vld1.32    {q2,q3}, [ap]!	C load 8 limbs
	vld1.32    {q0,q1}, [ap]!	C load 8 limbs
	vcnt.8	   q12, q2
	vcnt.8	   q13, q3
	subs	   n, n, #16
	blo	   L(end)

L(top):	vld1.32    {q2,q3}, [ap]!	C load 8 limbs
	vpadal.u8  q8, q12
	vcnt.8	   q12, q0
	vpadal.u8  q9, q13
	vcnt.8	   q13, q1
L(mid):	vld1.32    {q0,q1}, [ap]!	C load 8 limbs
	subs	   n, n, #16
	vpadal.u8  q8, q12
	vcnt.8	   q12, q2
	vpadal.u8  q9, q13
	vcnt.8	   q13, q3
	bhs	   L(top)

L(end):	vpadal.u8  q8, q12
	vpadal.u8  q9, q13
L(sum):	vcnt.8	   q12, q0
	vcnt.8	   q13, q1
	vpadal.u8  q8, q12
	vpadal.u8  q9, q13
	vadd.i16   q8, q8, q9
					C we have 8 16-bit counts
L(e0):	vpaddl.u16 q8, q8		C we have 4 32-bit counts
	vpaddl.u32 q8, q8		C we have 2 64-bit counts
	vmov.32    r0, d16[0]
	vmov.32    r1, d17[0]
	add	   r0, r0, r1
	bx	lr

C Code for large count.  Splits operand and calls above code.
define(`ap2', r2)			C caller-saves reg not used above
L(gt16k):
	push	{r4,r14}
	mov	ap2, ap
	mov	r3, n			C full count
	mov	r4, #0			C total sum

1:	mov	n, #chunksize		C count for this invocation
	bl	L(lt16k)		C could jump deep inside code
	add	ap2, ap2, #chunksize*4	C point at next chunk
	add	r4, r4, r0
	mov	ap, ap2			C put chunk pointer in place for call
	sub	r3, r3, #chunksize
	cmp	r3, #chunksize
	bhi	1b

	mov	n, r3			C count for final invocation
	bl	L(lt16k)
	add	r0, r4, r0
	pop	{r4,pc}
EPILOGUE()
