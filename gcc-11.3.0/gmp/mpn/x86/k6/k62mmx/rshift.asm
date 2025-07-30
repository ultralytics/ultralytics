dnl  AMD K6-2 mpn_rshift -- mpn right shift.

dnl  Copyright 1999, 2000, 2002 Free Software Foundation, Inc.

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


C K6-2: 1.75 cycles/limb


C mp_limb_t mpn_rshift (mp_ptr dst, mp_srcptr src, mp_size_t size,
C                       unsigned shift);
C

defframe(PARAM_SHIFT,16)
defframe(PARAM_SIZE, 12)
defframe(PARAM_SRC,  8)
defframe(PARAM_DST,  4)
deflit(`FRAME',0)

dnl  Minimum 9, because the unrolled loop can't handle less.
dnl
deflit(UNROLL_THRESHOLD, 9)

	TEXT
	ALIGN(32)

PROLOGUE(mpn_rshift)
deflit(`FRAME',0)

	C The 1 limb case can be done without the push %ebx, but it's then
	C still the same speed.  The push is left as a free helping hand for
	C the two_or_more code.

	movl	PARAM_SIZE, %eax
	pushl	%ebx			FRAME_pushl()

	movl	PARAM_SRC, %ebx
	decl	%eax

	movl	PARAM_SHIFT, %ecx
	jnz	L(two_or_more)

	movl	(%ebx), %edx		C src limb
	movl	PARAM_DST, %ebx

	shrdl(	%cl, %edx, %eax)	C return value

	shrl	%cl, %edx

	movl	%edx, (%ebx)		C dst limb
	popl	%ebx

	ret


C -----------------------------------------------------------------------------
	ALIGN(16)	C avoid offset 0x1f
L(two_or_more):
	C eax	size-1
	C ebx	src
	C ecx	shift
	C edx

	movl	(%ebx), %edx	C src low limb
	negl	%ecx

	addl	$32, %ecx
	movd	PARAM_SHIFT, %mm6

	shll	%cl, %edx
	cmpl	$UNROLL_THRESHOLD-1, %eax

	jae	L(unroll)


	C eax	size-1
	C ebx	src
	C ecx	32-shift
	C edx	retval
	C
	C mm6	shift

	movl	PARAM_DST, %ecx
	leal	(%ebx,%eax,4), %ebx

	leal	-4(%ecx,%eax,4), %ecx
	negl	%eax

	C This loop runs at about 3 cycles/limb, which is the amount of
	C decoding, and this is despite every second access being unaligned.

L(simple):
	C eax	counter, -(size-1) to -1
	C ebx	&src[size-1]
	C ecx	&dst[size-1]
	C edx	retval
	C
	C mm0	scratch
	C mm6	shift

Zdisp(	movq,	0,(%ebx,%eax,4), %mm0)
	incl	%eax

	psrlq	%mm6, %mm0

Zdisp(	movd,	%mm0, 0,(%ecx,%eax,4))
	jnz	L(simple)


	movq	%mm0, (%ecx)
	movl	%edx, %eax

	popl	%ebx

	femms
	ret


C -----------------------------------------------------------------------------
	ALIGN(16)
L(unroll):
	C eax	size-1
	C ebx	src
	C ecx	32-shift
	C edx	retval
	C
	C mm6	shift

	addl	$32, %ecx
	subl	$7, %eax		C size-8

	movd	%ecx, %mm7
	movl	PARAM_DST, %ecx

	movq	(%ebx), %mm2		C src low qword
	leal	(%ebx,%eax,4), %ebx	C src end - 32

	testb	$4, %cl
	leal	(%ecx,%eax,4), %ecx	C dst end - 32

	notl	%eax			C -(size-7)
	jz	L(dst_aligned)

	psrlq	%mm6, %mm2
	incl	%eax

Zdisp(	movd,	%mm2, 0,(%ecx,%eax,4))	C dst low limb
	movq	4(%ebx,%eax,4), %mm2	C new src low qword
L(dst_aligned):

	movq	12(%ebx,%eax,4), %mm0	C src second lowest qword
	nop	C avoid bad cache line crossing


	C This loop is the important bit, the rest is just support for it.
	C Four src limbs are held at the start, and four more will be read.
	C Four dst limbs will be written.  This schedule seems necessary for
	C full speed.
	C
	C The use of -(size-7) lets the loop stop when %eax becomes >= 0 and
	C and leaves 0 to 3 which can be tested with test $1 and $2.

L(top):
	C eax	counter, -(size-7) step by +4 until >=0
	C ebx	src end - 32
	C ecx	dst end - 32
	C edx	retval
	C
	C mm0	src next qword
	C mm1	scratch
	C mm2	src prev qword
	C mm6	shift
	C mm7	64-shift

	psrlq	%mm6, %mm2
	addl	$4, %eax

	movq	%mm0, %mm1
	psllq	%mm7, %mm0

	por	%mm0, %mm2
	movq	4(%ebx,%eax,4), %mm0

	psrlq	%mm6, %mm1
	movq	%mm2, -12(%ecx,%eax,4)

	movq	%mm0, %mm2
	psllq	%mm7, %mm0

	por	%mm0, %mm1
	movq	12(%ebx,%eax,4), %mm0

	movq	%mm1, -4(%ecx,%eax,4)
	ja	L(top)		C jump if no carry and not zero



	C Now have the four limbs in mm2 (low) and mm0 (high), and %eax is 0
	C to 3 representing respectively 3 to 0 further limbs.

	testl	$2, %eax	C testl to avoid bad cache line crossings
	jnz	L(finish_nottwo)

	C Two or three extra limbs: rshift mm2, OR it with lshifted mm0, mm0
	C becomes new mm2 and a new mm0 is loaded.

	psrlq	%mm6, %mm2
	movq	%mm0, %mm1

	psllq	%mm7, %mm0
	addl	$2, %eax

	por	%mm0, %mm2
	movq	12(%ebx,%eax,4), %mm0

	movq	%mm2, -4(%ecx,%eax,4)
	movq	%mm1, %mm2
L(finish_nottwo):


	testb	$1, %al
	psrlq	%mm6, %mm2

	movq	%mm0, %mm1
	psllq	%mm7, %mm0

	por	%mm0, %mm2
	psrlq	%mm6, %mm1

	movq	%mm2, 4(%ecx,%eax,4)
	jnz	L(finish_even)


	C one further extra limb to process

	movd	32-4(%ebx), %mm0	C src[size-1], most significant limb
	popl	%ebx

	movq	%mm0, %mm2
	psllq	%mm7, %mm0

	por	%mm0, %mm1
	psrlq	%mm6, %mm2

	movq	%mm1, 32-12(%ecx)	C dst[size-3,size-2]
	movd	%mm2, 32-4(%ecx)	C dst[size-1]

	movl	%edx, %eax		C retval

	femms
	ret


	nop	C avoid bad cache line crossing
L(finish_even):
	C no further extra limbs

	movq	%mm1, 32-8(%ecx)	C dst[size-2,size-1]
	movl	%edx, %eax		C retval

	popl	%ebx

	femms
	ret

EPILOGUE()
