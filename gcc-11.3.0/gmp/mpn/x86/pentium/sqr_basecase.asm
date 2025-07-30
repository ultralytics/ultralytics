dnl  Intel P5 mpn_sqr_basecase -- square an mpn number.

dnl  Copyright 1999-2002 Free Software Foundation, Inc.

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


C P5: approx 8 cycles per crossproduct, or 15.5 cycles per triangular
C product at around 20x20 limbs.


C void mpn_sqr_basecase (mp_ptr dst, mp_srcptr src, mp_size_t size);
C
C Calculate src,size squared, storing the result in dst,2*size.
C
C The algorithm is basically the same as mpn/generic/sqr_basecase.c, but a
C lot of function call overheads are avoided, especially when the size is
C small.

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

	movl	(%eax), %eax
	ja	L(three_or_more)

C -----------------------------------------------------------------------------
C one limb only
	C eax	src
	C ebx
	C ecx	dst
	C edx

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
	C edx	size

	pushl	%ebp
	pushl	%edi

	pushl	%esi
	pushl	%ebx

	movl	%eax, %ebx
	movl	(%eax), %eax

	mull	%eax		C src[0]^2

	movl	%eax, (%ecx)	C dst[0]
	movl	%edx, %esi	C dst[1]

	movl	4(%ebx), %eax

	mull	%eax		C src[1]^2

	movl	%eax, %edi	C dst[2]
	movl	%edx, %ebp	C dst[3]

	movl	(%ebx), %eax

	mull	4(%ebx)		C src[0]*src[1]

	addl	%eax, %esi
	popl	%ebx

	adcl	%edx, %edi

	adcl	$0, %ebp
	addl	%esi, %eax

	adcl	%edi, %edx
	movl	%eax, 4(%ecx)

	adcl	$0, %ebp
	popl	%esi

	movl	%edx, 8(%ecx)
	movl	%ebp, 12(%ecx)

	popl	%edi
	popl	%ebp

	ret


C -----------------------------------------------------------------------------
	ALIGN(8)
L(three_or_more):
	C eax	src low limb
	C ebx
	C ecx	dst
	C edx	size

	cmpl	$4, %edx
	pushl	%ebx
deflit(`FRAME',4)

	movl	PARAM_SRC, %ebx
	jae	L(four_or_more)


C -----------------------------------------------------------------------------
C three limbs
	C eax	src low limb
	C ebx	src
	C ecx	dst
	C edx	size

	pushl	%ebp
	pushl	%edi

	mull	%eax		C src[0] ^ 2

	movl	%eax, (%ecx)
	movl	%edx, 4(%ecx)

	movl	4(%ebx), %eax
	xorl	%ebp, %ebp

	mull	%eax		C src[1] ^ 2

	movl	%eax, 8(%ecx)
	movl	%edx, 12(%ecx)

	movl	8(%ebx), %eax
	pushl	%esi		C risk of cache bank clash

	mull	%eax		C src[2] ^ 2

	movl	%eax, 16(%ecx)
	movl	%edx, 20(%ecx)

	movl	(%ebx), %eax

	mull	4(%ebx)		C src[0] * src[1]

	movl	%eax, %esi
	movl	%edx, %edi

	movl	(%ebx), %eax

	mull	8(%ebx)		C src[0] * src[2]

	addl	%eax, %edi
	movl	%edx, %ebp

	adcl	$0, %ebp
	movl	4(%ebx), %eax

	mull	8(%ebx)		C src[1] * src[2]

	xorl	%ebx, %ebx
	addl	%eax, %ebp

	C eax
	C ebx	zero, will be dst[5]
	C ecx	dst
	C edx	dst[4]
	C esi	dst[1]
	C edi	dst[2]
	C ebp	dst[3]

	adcl	$0, %edx
	addl	%esi, %esi

	adcl	%edi, %edi

	adcl	%ebp, %ebp

	adcl	%edx, %edx
	movl	4(%ecx), %eax

	adcl	$0, %ebx
	addl	%esi, %eax

	movl	%eax, 4(%ecx)
	movl	8(%ecx), %eax

	adcl	%edi, %eax
	movl	12(%ecx), %esi

	adcl	%ebp, %esi
	movl	16(%ecx), %edi

	movl	%eax, 8(%ecx)
	movl	%esi, 12(%ecx)

	adcl	%edx, %edi
	popl	%esi

	movl	20(%ecx), %eax
	movl	%edi, 16(%ecx)

	popl	%edi
	popl	%ebp

	adcl	%ebx, %eax	C no carry out of this
	popl	%ebx

	movl	%eax, 20(%ecx)

	ret


C -----------------------------------------------------------------------------
	ALIGN(8)
L(four_or_more):
	C eax	src low limb
	C ebx	src
	C ecx	dst
	C edx	size
	C esi
	C edi
	C ebp
	C
	C First multiply src[0]*src[1..size-1] and store at dst[1..size].

deflit(`FRAME',4)

	pushl	%edi
FRAME_pushl()
	pushl	%esi
FRAME_pushl()

	pushl	%ebp
FRAME_pushl()
	leal	(%ecx,%edx,4), %edi	C dst end of this mul1

	leal	(%ebx,%edx,4), %esi	C src end
	movl	%ebx, %ebp		C src

	negl	%edx			C -size
	xorl	%ebx, %ebx		C clear carry limb and carry flag

	leal	1(%edx), %ecx		C -(size-1)

L(mul1):
	C eax	scratch
	C ebx	carry
	C ecx	counter, negative
	C edx	scratch
	C esi	&src[size]
	C edi	&dst[size]
	C ebp	src

	adcl	$0, %ebx
	movl	(%esi,%ecx,4), %eax

	mull	(%ebp)

	addl	%eax, %ebx

	movl	%ebx, (%edi,%ecx,4)
	incl	%ecx

	movl	%edx, %ebx
	jnz	L(mul1)


	C Add products src[n]*src[n+1..size-1] at dst[2*n-1...], for
	C n=1..size-2.
	C
	C The last two products, which are the end corner of the product
	C triangle, are handled separately to save looping overhead.  These
	C are src[size-3]*src[size-2,size-1] and src[size-2]*src[size-1].
	C If size is 4 then it's only these that need to be done.
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

	adcl	$0, %ebx
	movl	PARAM_SIZE, %edx

	movl	%ebx, (%edi)
	subl	$4, %edx

	negl	%edx
	jz	L(corner)


L(outer):
	C ebx	previous carry limb to store
	C edx	outer loop counter (negative)
	C esi	&src[size]
	C edi	dst, pointing at stored carry limb of previous loop

	pushl	%edx			C new outer loop counter
	leal	-2(%edx), %ecx

	movl	%ebx, (%edi)
	addl	$4, %edi

	addl	$4, %ebp
	xorl	%ebx, %ebx		C initial carry limb, clear carry flag

L(inner):
	C eax	scratch
	C ebx	carry (needing carry flag added)
	C ecx	counter, negative
	C edx	scratch
	C esi	&src[size]
	C edi	dst end of this addmul
	C ebp	&src[j]

	adcl	$0, %ebx
	movl	(%esi,%ecx,4), %eax

	mull	(%ebp)

	addl	%ebx, %eax
	movl	(%edi,%ecx,4), %ebx

	adcl	$0, %edx
	addl	%eax, %ebx

	movl	%ebx, (%edi,%ecx,4)
	incl	%ecx

	movl	%edx, %ebx
	jnz	L(inner)


	adcl	$0, %ebx
	popl	%edx		C outer loop counter

	incl	%edx
	jnz	L(outer)


	movl	%ebx, (%edi)

L(corner):
	C esi	&src[size]
	C edi	&dst[2*size-4]

	movl	-8(%esi), %eax
	movl	-4(%edi), %ebx		C risk of data cache bank clash here

	mull	-12(%esi)		C src[size-2]*src[size-3]

	addl	%eax, %ebx
	movl	%edx, %ecx

	adcl	$0, %ecx
	movl	-4(%esi), %eax

	mull	-12(%esi)		C src[size-1]*src[size-3]

	addl	%ecx, %eax
	movl	(%edi), %ecx

	adcl	$0, %edx
	movl	%ebx, -4(%edi)

	addl	%eax, %ecx
	movl	%edx, %ebx

	adcl	$0, %ebx
	movl	-4(%esi), %eax

	mull	-8(%esi)		C src[size-1]*src[size-2]

	movl	%ecx, (%edi)
	addl	%eax, %ebx

	adcl	$0, %edx
	movl	PARAM_SIZE, %eax

	negl	%eax
	movl	%ebx, 4(%edi)

	addl	$1, %eax		C -(size-1) and clear carry
	movl	%edx, 8(%edi)


C -----------------------------------------------------------------------------
C Left shift of dst[1..2*size-2], high bit shifted out becomes dst[2*size-1].

L(lshift):
	C eax	counter, negative
	C ebx	next limb
	C ecx
	C edx
	C esi
	C edi	&dst[2*size-4]
	C ebp

	movl	12(%edi,%eax,8), %ebx

	rcll	%ebx
	movl	16(%edi,%eax,8), %ecx

	rcll	%ecx
	movl	%ebx, 12(%edi,%eax,8)

	movl	%ecx, 16(%edi,%eax,8)
	incl	%eax

	jnz	L(lshift)


	adcl	%eax, %eax		C high bit out
	movl	PARAM_SRC, %esi

	movl	PARAM_SIZE, %ecx	C risk of cache bank clash
	movl	%eax, 12(%edi)		C dst most significant limb


C -----------------------------------------------------------------------------
C Now add in the squares on the diagonal, namely src[0]^2, src[1]^2, ...,
C src[size-1]^2.  dst[0] hasn't yet been set at all yet, and just gets the
C low limb of src[0]^2.

	movl	(%esi), %eax		C src[0]
	leal	(%esi,%ecx,4), %esi	C src end

	negl	%ecx

	mull	%eax

	movl	%eax, 16(%edi,%ecx,8)	C dst[0]
	movl	%edx, %ebx

	addl	$1, %ecx		C size-1 and clear carry

L(diag):
	C eax	scratch (low product)
	C ebx	carry limb
	C ecx	counter, negative
	C edx	scratch (high product)
	C esi	&src[size]
	C edi	&dst[2*size-4]
	C ebp	scratch (fetched dst limbs)

	movl	(%esi,%ecx,4), %eax
	adcl	$0, %ebx

	mull	%eax

	movl	16-4(%edi,%ecx,8), %ebp

	addl	%ebp, %ebx
	movl	16(%edi,%ecx,8), %ebp

	adcl	%eax, %ebp
	movl	%ebx, 16-4(%edi,%ecx,8)

	movl	%ebp, 16(%edi,%ecx,8)
	incl	%ecx

	movl	%edx, %ebx
	jnz	L(diag)


	adcl	$0, %edx
	movl	16-4(%edi), %eax	C dst most significant limb

	addl	%eax, %edx
	popl	%ebp

	movl	%edx, 16-4(%edi)
	popl	%esi		C risk of cache bank clash

	popl	%edi
	popl	%ebx

	ret

EPILOGUE()
