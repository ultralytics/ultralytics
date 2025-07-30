dnl  x86 generic mpn_sqr_basecase -- square an mpn number.

dnl  Copyright 1999, 2000, 2002, 2003 Free Software Foundation, Inc.

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


C     cycles/crossproduct  cycles/triangleproduct
C P5
C P6
C K6
C K7
C P4


C void mpn_sqr_basecase (mp_ptr dst, mp_srcptr src, mp_size_t size);
C
C The algorithm is basically the same as mpn/generic/sqr_basecase.c, but a
C lot of function call overheads are avoided, especially when the size is
C small.
C
C The mul1 loop is not unrolled like mul_1.asm, it doesn't seem worth the
C code size to do so here.
C
C Enhancements:
C
C The addmul loop here is also not unrolled like aorsmul_1.asm and
C mul_basecase.asm are.  Perhaps it should be done.  It'd add to the
C complexity, but if it's worth doing in the other places then it should be
C worthwhile here.
C
C A fully-unrolled style like other sqr_basecase.asm versions (k6, k7, p6)
C might be worth considering.  That'd add quite a bit to the code size, but
C only as much as is used would be dragged into L1 cache.

defframe(PARAM_SIZE,12)
defframe(PARAM_SRC, 8)
defframe(PARAM_DST, 4)

	TEXT
	ALIGN(8)
PROLOGUE(mpn_sqr_basecase)
deflit(`FRAME',0)

	movl	PARAM_SIZE, %edx

	movl	PARAM_SRC, %eax

	cmpl	$2, %edx
	movl	PARAM_DST, %ecx

	je	L(two_limbs)
	ja	L(three_or_more)


C -----------------------------------------------------------------------------
C one limb only
	C eax	src
	C ebx
	C ecx	dst
	C edx

	movl	(%eax), %eax
	mull	%eax
	movl	%eax, (%ecx)
	movl	%edx, 4(%ecx)
	ret


C -----------------------------------------------------------------------------
	ALIGN(8)
L(two_limbs):
	C eax	src
	C ebx
	C ecx	dst
	C edx

	pushl	%ebx
	pushl	%ebp

	movl	%eax, %ebx
	movl	(%eax), %eax

	mull	%eax		C src[0]^2

	pushl	%esi
	pushl	%edi

	movl	%edx, %esi	C dst[1]
	movl	%eax, (%ecx)	C dst[0]

	movl	4(%ebx), %eax
	mull	%eax		C src[1]^2

	movl	%eax, %edi	C dst[2]
	movl	%edx, %ebp	C dst[3]

	movl	(%ebx), %eax
	mull	4(%ebx)		C src[0]*src[1]

	addl	%eax, %esi

	adcl	%edx, %edi

	adcl	$0, %ebp
	addl	%esi, %eax

	adcl	%edi, %edx
	movl	%eax, 4(%ecx)

	adcl	$0, %ebp

	movl	%edx, 8(%ecx)
	movl	%ebp, 12(%ecx)

	popl	%edi
	popl	%esi

	popl	%ebp
	popl	%ebx

	ret


C -----------------------------------------------------------------------------
	ALIGN(8)
L(three_or_more):
deflit(`FRAME',0)
	C eax	src
	C ebx
	C ecx	dst
	C edx	size

	pushl	%ebx	FRAME_pushl()
	pushl	%edi	FRAME_pushl()

	pushl	%esi	FRAME_pushl()
	pushl	%ebp	FRAME_pushl()

	leal	(%ecx,%edx,4), %edi	C &dst[size], end of this mul1
	leal	(%eax,%edx,4), %esi	C &src[size]

C First multiply src[0]*src[1..size-1] and store at dst[1..size].

	movl	(%eax), %ebp		C src[0], multiplier
	movl	%edx, %ecx

	negl	%ecx			C -size
	xorl	%ebx, %ebx		C clear carry limb

	incl	%ecx			C -(size-1)

L(mul1):
	C eax	scratch
	C ebx	carry
	C ecx	counter, limbs, negative
	C edx	scratch
	C esi	&src[size]
	C edi	&dst[size]
	C ebp	multiplier

	movl	(%esi,%ecx,4), %eax
	mull	%ebp
	addl	%eax, %ebx
	adcl	$0, %edx
	movl	%ebx, (%edi,%ecx,4)
	movl	%edx, %ebx
	incl	%ecx
	jnz	L(mul1)

	movl	%ebx, (%edi)


	C Add products src[n]*src[n+1..size-1] at dst[2*n-1...], for
	C n=1..size-2.
	C
	C The last products src[size-2]*src[size-1], which is the end corner
	C of the product triangle, is handled separately at the end to save
	C looping overhead.  If size is 3 then it's only this that needs to
	C be done.
	C
	C In the outer loop %esi is a constant, and %edi just advances by 1
	C limb each time.  The size of the operation decreases by 1 limb
	C each time.

	C eax
	C ebx	carry (needing carry flag added)
	C ecx
	C edx
	C esi	&src[size]
	C edi	&dst[size]
	C ebp

	movl	PARAM_SIZE, %ecx
	subl	$3, %ecx
	jz	L(corner)

	negl	%ecx

dnl  re-use parameter space
define(VAR_OUTER,`PARAM_DST')

L(outer):
	C eax
	C ebx
	C ecx
	C edx	outer loop counter, -(size-3) to -1
	C esi	&src[size]
	C edi	dst, pointing at stored carry limb of previous loop
	C ebp

	movl	%ecx, VAR_OUTER
	addl	$4, %edi		C advance dst end

	movl	-8(%esi,%ecx,4), %ebp	C next multiplier
	subl	$1, %ecx

	xorl	%ebx, %ebx		C initial carry limb

L(inner):
	C eax	scratch
	C ebx	carry (needing carry flag added)
	C ecx	counter, -n-1 to -1
	C edx	scratch
	C esi	&src[size]
	C edi	dst end of this addmul
	C ebp	multiplier

	movl	(%esi,%ecx,4), %eax
	mull	%ebp
	addl	%ebx, %eax
	adcl	$0, %edx
	addl	%eax, (%edi,%ecx,4)
	adcl	$0, %edx
	movl	%edx, %ebx
	addl	$1, %ecx
	jl	L(inner)


	movl	%ebx, (%edi)
	movl	VAR_OUTER, %ecx
	incl	%ecx
	jnz	L(outer)


L(corner):
	C esi	&src[size]
	C edi	&dst[2*size-3]

	movl	-4(%esi), %eax
	mull	-8(%esi)		C src[size-1]*src[size-2]
	addl	%eax, 0(%edi)
	adcl	$0, %edx
	movl	%edx, 4(%edi)		C dst high limb


C -----------------------------------------------------------------------------
C Left shift of dst[1..2*size-2], high bit shifted out becomes dst[2*size-1].

	movl	PARAM_SIZE, %eax
	negl	%eax
	addl	$1, %eax		C -(size-1) and clear carry

L(lshift):
	C eax	counter, negative
	C ebx	next limb
	C ecx
	C edx
	C esi
	C edi	&dst[2*size-4]
	C ebp

	rcll	8(%edi,%eax,8)
	rcll	12(%edi,%eax,8)
	incl	%eax
	jnz	L(lshift)


	adcl	%eax, %eax		C high bit out
	movl	%eax, 8(%edi)		C dst most significant limb


C Now add in the squares on the diagonal, namely src[0]^2, src[1]^2, ...,
C src[size-1]^2.  dst[0] hasn't yet been set at all yet, and just gets the
C low limb of src[0]^2.

	movl	PARAM_SRC, %esi
	movl	(%esi), %eax		C src[0]
	mull	%eax			C src[0]^2

	movl	PARAM_SIZE, %ecx
	leal	(%esi,%ecx,4), %esi	C src end

	negl	%ecx			C -size
	movl	%edx, %ebx		C initial carry

	movl	%eax, 12(%edi,%ecx,8)	C dst[0]
	incl	%ecx			C -(size-1)

L(diag):
	C eax	scratch (low product)
	C ebx	carry limb
	C ecx	counter, -(size-1) to -1
	C edx	scratch (high product)
	C esi	&src[size]
	C edi	&dst[2*size-3]
	C ebp	scratch (fetched dst limbs)

	movl	(%esi,%ecx,4), %eax
	mull	%eax

	addl	%ebx, 8(%edi,%ecx,8)
	movl	%edx, %ebx

	adcl	%eax, 12(%edi,%ecx,8)
	adcl	$0, %ebx

	incl	%ecx
	jnz	L(diag)


	addl	%ebx, 8(%edi)		C dst most significant limb

	popl	%ebp
	popl	%esi

	popl	%edi
	popl	%ebx

	ret

EPILOGUE()
