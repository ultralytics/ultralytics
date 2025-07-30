dnl  ARM64 Neon mpn_popcount -- mpn bit population count.

dnl  Copyright 2013, 2014 Free Software Foundation, Inc.

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
C Cortex-A53	 ?
C Cortex-A57	 ?

C TODO
C  * Consider greater unrolling.
C  * Arrange to align the pointer, if that helps performance.  Use the same
C    read-and-mask trick we use on PCs, for simplicity and performance.  (Sorry
C    valgrind!)
C  * Explore if explicit align directives, e.g., "[ptr:128]" help.
C  * See rth's gmp-devel 2013-02/03 messages about final summation tricks.

changecom(@&*$)

C INPUT PARAMETERS
define(`ap', x0)
define(`n',  x1)

C We sum into 16 16-bit counters in v4,v5, but at the end we sum them and end
C up with 8 16-bit counters.  Therefore, we can sum to 8(2^16-1) bits, or
C (8*2^16-1)/64 = 0x1fff limbs.  We use a chunksize close to that, but which
C  allows the huge count code to jump deep into the code (at L(chu)).

define(`maxsize',  0x1fff)
define(`chunksize',0x1ff0)

ASM_START()
PROLOGUE(mpn_popcount)

	mov	x11, #maxsize
	cmp	n, x11
	b.hi	L(gt8k)

L(lt8k):
	movi	v4.16b, #0			C clear summation register
	movi	v5.16b, #0			C clear summation register

	tbz	n, #0, L(xx0)
	sub	n, n, #1
	ld1	{v0.1d}, [ap], #8		C load 1 limb
	cnt	v6.16b, v0.16b
	uadalp	v4.8h,  v6.16b			C could also splat

L(xx0):	tbz	n, #1, L(x00)
	sub	n, n, #2
	ld1	{v0.2d}, [ap], #16		C load 2 limbs
	cnt	v6.16b, v0.16b
	uadalp	v4.8h,  v6.16b

L(x00):	tbz	n, #2, L(000)
	subs	n, n, #4
	ld1	{v0.2d,v1.2d}, [ap], #32	C load 4 limbs
	b.ls	L(sum)

L(gt4):	ld1	{v2.2d,v3.2d}, [ap], #32	C load 4 limbs
	sub	n, n, #4
	cnt	v6.16b, v0.16b
	cnt	v7.16b, v1.16b
	b	L(mid)

L(000):	subs	n, n, #8
	b.lo	L(e0)

L(chu):	ld1	{v2.2d,v3.2d}, [ap], #32	C load 4 limbs
	ld1	{v0.2d,v1.2d}, [ap], #32	C load 4 limbs
	cnt	v6.16b, v2.16b
	cnt	v7.16b, v3.16b
	subs	n, n, #8
	b.lo	L(end)

L(top):	ld1	{v2.2d,v3.2d}, [ap], #32	C load 4 limbs
	uadalp	v4.8h,  v6.16b
	cnt	v6.16b, v0.16b
	uadalp	v5.8h,  v7.16b
	cnt	v7.16b, v1.16b
L(mid):	ld1	{v0.2d,v1.2d}, [ap], #32	C load 4 limbs
	subs	n, n, #8
	uadalp	v4.8h,  v6.16b
	cnt	v6.16b, v2.16b
	uadalp	v5.8h,  v7.16b
	cnt	v7.16b, v3.16b
	b.hs	L(top)

L(end):	uadalp	v4.8h,  v6.16b
	uadalp	v5.8h,  v7.16b
L(sum):	cnt	v6.16b, v0.16b
	cnt	v7.16b, v1.16b
	uadalp	v4.8h,  v6.16b
	uadalp	v5.8h,  v7.16b
	add	v4.8h, v4.8h, v5.8h
					C we have 8 16-bit counts
L(e0):	uaddlp	v4.4s,  v4.8h		C we have 4 32-bit counts
	uaddlp	v4.2d,  v4.4s		C we have 2 64-bit counts
	mov	x0, v4.d[0]
	mov	x1, v4.d[1]
	add	x0, x0, x1
	ret

C Code for count > maxsize.  Splits operand and calls above code.
define(`ap2', x5)			C caller-saves reg not used above
L(gt8k):
	mov	x8, x30
	mov	x7, n			C full count (caller-saves reg not used above)
	mov	x4, #0			C total sum  (caller-saves reg not used above)
	mov	x9, #chunksize*8	C caller-saves reg not used above
	mov	x10, #chunksize		C caller-saves reg not used above

1:	add	ap2, ap, x9		C point at subsequent block
	mov	n, #chunksize-8		C count for this invocation, adjusted for entry pt
	movi	v4.16b, #0		C clear chunk summation register
	movi	v5.16b, #0		C clear chunk summation register
	bl	L(chu)			C jump deep inside code
	add	x4, x4, x0
	mov	ap, ap2			C put chunk pointer in place for calls
	sub	x7, x7, x10
	cmp	x7, x11
	b.hi	1b

	mov	n, x7			C count for final invocation
	bl	L(lt8k)
	add	x0, x4, x0
	mov	x30, x8
	ret
EPILOGUE()
