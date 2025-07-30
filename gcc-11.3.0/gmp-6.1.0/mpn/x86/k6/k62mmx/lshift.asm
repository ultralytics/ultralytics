dnl  AMD K6-2 mpn_lshift -- mpn left shift.

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


C mp_limb_t mpn_lshift (mp_ptr dst, mp_srcptr src, mp_size_t size,
C                       unsigned shift);
C

defframe(PARAM_SHIFT,16)
defframe(PARAM_SIZE, 12)
defframe(PARAM_SRC,  8)
defframe(PARAM_DST,  4)
deflit(`FRAME',0)

dnl  used after src has been fetched
define(VAR_RETVAL,`PARAM_SRC')

dnl  minimum 9, because unrolled loop can't handle less
deflit(UNROLL_THRESHOLD, 9)

	TEXT
	ALIGN(32)

PROLOGUE(mpn_lshift)
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

	shldl(	%cl, %edx, %eax)	C return value

	shll	%cl, %edx

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

	movl	(%ebx,%eax,4), %edx	C src high limb
	negl	%ecx

	movd	PARAM_SHIFT, %mm6
	addl	$32, %ecx		C 32-shift

	shrl	%cl, %edx
	cmpl	$UNROLL_THRESHOLD-1, %eax

	movl	%edx, VAR_RETVAL
	jae	L(unroll)


	movd	%ecx, %mm7
	movl	%eax, %ecx

	movl	PARAM_DST, %eax

L(simple):
	C eax	dst
	C ebx	src
	C ecx	counter, size-1 to 1
	C edx	retval
	C
	C mm0	scratch
	C mm6	shift
	C mm7	32-shift

	movq	-4(%ebx,%ecx,4), %mm0

	psrlq	%mm7, %mm0

Zdisp(	movd,	%mm0, 0,(%eax,%ecx,4))
	loop	L(simple)


	movd	(%ebx), %mm0
	popl	%ebx

	psllq	%mm6, %mm0

	movd	%mm0, (%eax)
	movl	%edx, %eax

	femms
	ret


C -----------------------------------------------------------------------------
	ALIGN(16)
L(unroll):
	C eax	size-1
	C ebx	src
	C ecx	32-shift
	C edx	retval (but instead VAR_RETVAL is used)
	C
	C mm6	shift

	addl	$32, %ecx
	movl	PARAM_DST, %edx

	movd	%ecx, %mm7
	subl	$7, %eax			C size-8

	leal	(%edx,%eax,4), %ecx		C alignment of dst

	movq	32-8(%ebx,%eax,4), %mm2		C src high qword
	testb	$4, %cl

	jz	L(dst_aligned)
	psllq	%mm6, %mm2

	psrlq	$32, %mm2
	decl	%eax

	movd	%mm2, 32(%edx,%eax,4)		C dst high limb
	movq	32-8(%ebx,%eax,4), %mm2		C new src high qword
L(dst_aligned):

	movq	32-16(%ebx,%eax,4), %mm0	C src second highest qword


	C This loop is the important bit, the rest is just support for it.
	C Four src limbs are held at the start, and four more will be read.
	C Four dst limbs will be written.  This schedule seems necessary for
	C full speed.
	C
	C The use of size-8 lets the loop stop when %eax goes negative and
	C leaves -4 to -1 which can be tested with test $1 and $2.

L(top):
	C eax	counter, size-8 step by -4 until <0
	C ebx	src
	C ecx
	C edx	dst
	C
	C mm0	src next qword
	C mm1	scratch
	C mm2	src prev qword
	C mm6	shift
	C mm7	64-shift

	psllq	%mm6, %mm2
	subl	$4, %eax

	movq	%mm0, %mm1
	psrlq	%mm7, %mm0

	por	%mm0, %mm2
	movq	24(%ebx,%eax,4), %mm0

	psllq	%mm6, %mm1
	movq	%mm2, 40(%edx,%eax,4)

	movq	%mm0, %mm2
	psrlq	%mm7, %mm0

	por	%mm0, %mm1
	movq	16(%ebx,%eax,4), %mm0

	movq	%mm1, 32(%edx,%eax,4)
	jnc	L(top)


	C Now have four limbs in mm2 (prev) and mm0 (next), plus eax mod 4.
	C
	C 8(%ebx) is the next source, and 24(%edx) is the next destination.
	C %eax is between -4 and -1, representing respectively 0 to 3 extra
	C limbs that must be read.


	testl	$2, %eax	C testl to avoid bad cache line crossing
	jz	L(finish_nottwo)

	C Two more limbs: lshift mm2, OR it with rshifted mm0, mm0 becomes
	C new mm2 and a new mm0 is loaded.

	psllq	%mm6, %mm2
	movq	%mm0, %mm1

	psrlq	%mm7, %mm0
	subl	$2, %eax

	por	%mm0, %mm2
	movq	16(%ebx,%eax,4), %mm0

	movq	%mm2, 32(%edx,%eax,4)
	movq	%mm1, %mm2
L(finish_nottwo):


	C lshift mm2, OR with rshifted mm0, mm1 becomes lshifted mm0

	testb	$1, %al
	psllq	%mm6, %mm2

	movq	%mm0, %mm1
	psrlq	%mm7, %mm0

	por	%mm0, %mm2
	psllq	%mm6, %mm1

	movq	%mm2, 24(%edx,%eax,4)
	jz	L(finish_even)


	C Size is odd, so mm1 and one extra limb to process.

	movd	(%ebx), %mm0		C src[0]
	popl	%ebx
deflit(`FRAME',0)

	movq	%mm0, %mm2
	psllq	$32, %mm0

	psrlq	%mm7, %mm0

	psllq	%mm6, %mm2
	por	%mm0, %mm1

	movq	%mm1, 4(%edx)		C dst[1,2]
	movd	%mm2, (%edx)		C dst[0]

	movl	VAR_RETVAL, %eax

	femms
	ret


	nop	C avoid bad cache line crossing
L(finish_even):
deflit(`FRAME',4)
	C Size is even, so only mm1 left to process.

	movq	%mm1, (%edx)		C dst[0,1]
	movl	VAR_RETVAL, %eax

	popl	%ebx
	femms
	ret

EPILOGUE()
