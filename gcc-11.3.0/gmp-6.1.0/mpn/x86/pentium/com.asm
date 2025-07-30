dnl  Intel Pentium mpn_com -- mpn ones complement.

dnl  Copyright 1996, 2001, 2002, 2006 Free Software Foundation, Inc.

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


C P5: 1.75 cycles/limb


NAILS_SUPPORT(0-31)


C void mpn_com (mp_ptr dst, mp_srcptr src, mp_size_t size);
C
C This code is similar to mpn_copyi, basically there's just some "xorl
C $GMP_NUMB_MASK"s inserted.
C
C Alternatives:
C
C On P55 some MMX code could be 1.25 c/l (8 limb unrolled) if src and dst
C are the same alignment mod 8, but it doesn't seem worth the trouble for
C just that case (there'd need to be some plain integer available too for
C the unaligned case).

defframe(PARAM_SIZE,12)
defframe(PARAM_SRC, 8)
defframe(PARAM_DST, 4)

	TEXT
	ALIGN(8)
PROLOGUE(mpn_com)
deflit(`FRAME',0)

	movl	PARAM_SRC, %eax
	movl	PARAM_SIZE, %ecx

	pushl	%esi	FRAME_pushl()
	pushl	%edi	FRAME_pushl()

	leal	(%eax,%ecx,4), %eax
	xorl	$-1, %ecx		C -size-1

	movl	PARAM_DST, %edx
	addl	$8, %ecx		C -size+7

	jns	L(end)

	movl	(%edx), %esi		C fetch destination cache line
	nop

L(top):
	C eax	&src[size]
	C ebx
	C ecx	counter, limbs, negative
	C edx	dst, incrementing
	C esi	scratch
	C edi	scratch
	C ebp

	movl	28(%edx), %esi		C destination prefetch
	addl	$32, %edx

	movl	-28(%eax,%ecx,4), %esi
	movl	-24(%eax,%ecx,4), %edi
	xorl	$GMP_NUMB_MASK, %esi
	xorl	$GMP_NUMB_MASK, %edi
	movl	%esi, -32(%edx)
	movl	%edi, -28(%edx)

	movl	-20(%eax,%ecx,4), %esi
	movl	-16(%eax,%ecx,4), %edi
	xorl	$GMP_NUMB_MASK, %esi
	xorl	$GMP_NUMB_MASK, %edi
	movl	%esi, -24(%edx)
	movl	%edi, -20(%edx)

	movl	-12(%eax,%ecx,4), %esi
	movl	-8(%eax,%ecx,4), %edi
	xorl	$GMP_NUMB_MASK, %esi
	xorl	$GMP_NUMB_MASK, %edi
	movl	%esi, -16(%edx)
	movl	%edi, -12(%edx)

	movl	-4(%eax,%ecx,4), %esi
	movl	(%eax,%ecx,4), %edi
	xorl	$GMP_NUMB_MASK, %esi
	xorl	$GMP_NUMB_MASK, %edi
	movl	%esi, -8(%edx)
	movl	%edi, -4(%edx)

	addl	$8, %ecx
	js	L(top)


L(end):
	C eax	&src[size]
	C ecx	0 to 7, representing respectively 7 to 0 limbs remaining
	C edx	dst, next location to store

	subl	$4, %ecx
	nop

	jns	L(no4)

	movl	-12(%eax,%ecx,4), %esi
	movl	-8(%eax,%ecx,4), %edi
	xorl	$GMP_NUMB_MASK, %esi
	xorl	$GMP_NUMB_MASK, %edi
	movl	%esi, (%edx)
	movl	%edi, 4(%edx)

	movl	-4(%eax,%ecx,4), %esi
	movl	(%eax,%ecx,4), %edi
	xorl	$GMP_NUMB_MASK, %esi
	xorl	$GMP_NUMB_MASK, %edi
	movl	%esi, 8(%edx)
	movl	%edi, 12(%edx)

	addl	$16, %edx
	addl	$4, %ecx
L(no4):

	subl	$2, %ecx
	nop

	jns	L(no2)

	movl	-4(%eax,%ecx,4), %esi
	movl	(%eax,%ecx,4), %edi
	xorl	$GMP_NUMB_MASK, %esi
	xorl	$GMP_NUMB_MASK, %edi
	movl	%esi, (%edx)
	movl	%edi, 4(%edx)

	addl	$8, %edx
	addl	$2, %ecx
L(no2):

	popl	%edi
	jnz	L(done)

	movl	-4(%eax), %ecx

	xorl	$GMP_NUMB_MASK, %ecx
	popl	%esi

	movl	%ecx, (%edx)
	ret

L(done):
	popl	%esi
	ret

EPILOGUE()
