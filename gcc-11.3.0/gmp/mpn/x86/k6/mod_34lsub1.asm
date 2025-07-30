dnl  AMD K6 mpn_mod_34lsub1 -- mpn remainder modulo 2**24-1.

dnl  Copyright 2000-2002 Free Software Foundation, Inc.

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


C K6: 2.66 cycles/limb


C mp_limb_t mpn_mod_34lsub1 (mp_srcptr src, mp_size_t size)
C
C An attempt was made to use a loop like
C
C L(top):
C	adcl	(%edx), %eax
C	adcl	4(%edx), %ebx
C	adcl	8(%edx), %esi
C	leal	12(%edx), %edx
C	loop	L(top)
C
C with %ecx starting from floor(size/3), but it still measured 2.66 c/l.
C The form used instead can save about 6 cycles by not dividing by 3.
C
C In the code used, putting the "leal"s at the top of the loop is necessary
C for the claimed speed, anywhere else costs an extra cycle per loop.
C Perhaps a tight loop like this needs short decode instructions at the
C branch target, which would explain the leal/loop form above taking 8
C cycles instead of 7 too.

defframe(PARAM_SIZE, 8)
defframe(PARAM_SRC,  4)

dnl  re-use parameter space
define(SAVE_EBX, `PARAM_SIZE')
define(SAVE_ESI, `PARAM_SRC')

	TEXT
	ALIGN(16)
PROLOGUE(mpn_mod_34lsub1)
deflit(`FRAME',0)

	movl	PARAM_SIZE, %eax
	movl	PARAM_SRC, %edx

	subl	$2, %eax
	ja	L(three_or_more)

Zdisp(	movl,	0,(%edx), %eax)		C avoid code cache line boundary
	jne	L(one)

	movl	%eax, %ecx
	movl	4(%edx), %edx

	shrl	$24, %eax		C src[0] high
	andl	$0x00FFFFFF, %ecx	C src[0] low

	addl	%ecx, %eax
	movl	%edx, %ecx

	shll	$8, %edx
	andl	$0x00FFFF00, %edx	C src[1] high

	shrl	$16, %ecx		C src[1] low
	addl	%ecx, %eax

	addl	%edx, %eax

L(one):
	ret


L(three_or_more):
	C eax	size-2
	C ebx
	C ecx
	C edx	src

	movl	%ebx, SAVE_EBX
	xorl	%ebx, %ebx

	movl	%esi, SAVE_ESI
	pushl	%edi	FRAME_pushl()

	xorl	%esi, %esi
	xorl	%edi, %edi		C and clear carry flag

L(top):
	C eax	counter, limbs
	C ebx	acc 0mod3
	C ecx
	C edx	src, incrementing
	C esi	acc 1mod3
	C edi	acc 2mod3
	C ebp

	leal	-2(%eax), %eax
	leal	12(%edx), %edx

	adcl	-12(%edx), %ebx
	adcl	-8(%edx), %esi
	adcl	-4(%edx), %edi

	decl	%eax
	jg	L(top)


	C ecx is -3, -2 or -1 representing 0, 1 or 2 more limbs, respectively

	movb	$0, %cl
	incl	%eax

	js	L(combine)		C 0 more

Zdisp(	adcl,	0,(%edx), %ebx)		C avoid code cache line crossings

	movb	$8, %cl
	decl	%eax

	js	L(combine)		C 1 more

	adcl	4(%edx), %esi

	movb	$16, %cl


L(combine):
	sbbl	%edx, %edx

	shll	%cl, %edx		C carry
	movl	%ebx, %eax		C 0mod3

	shrl	$24, %eax		C 0mod3 high
	andl	$0x00FFFFFF, %ebx	C 0mod3 low

	subl	%edx, %eax		C apply carry
	movl	%esi, %ecx		C 1mod3

	shrl	$16, %esi		C 1mod3 high
	addl	%ebx, %eax		C apply 0mod3 low

	andl	$0x0000FFFF, %ecx
	addl	%esi, %eax		C apply 1mod3 high

	shll	$8, %ecx		C 1mod3 low
	movl	%edi, %edx		C 2mod3

	shrl	$8, %edx		C 2mod3 high
	addl	%ecx, %eax		C apply 1mod3 low

	addl	%edx, %eax		C apply 2mod3 high
	andl	$0x000000FF, %edi

	shll	$16, %edi		C 2mod3 low
	movl	SAVE_EBX, %ebx

	addl	%edi, %eax		C apply 2mod3 low
	movl	SAVE_ESI, %esi

	popl	%edi

	ret

EPILOGUE()
