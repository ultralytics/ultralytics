dnl  Intel P5 mpn_mod_34lsub1 -- mpn remainder modulo 2**24-1.

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


C P5: 1.66 cycles/limb


C mp_limb_t mpn_mod_34lsub1 (mp_srcptr src, mp_size_t size)
C

defframe(PARAM_SIZE, 8)
defframe(PARAM_SRC,  4)

	TEXT
	ALIGN(16)
PROLOGUE(mpn_mod_34lsub1)
deflit(`FRAME',0)

	movl	PARAM_SIZE, %ecx
	movl	PARAM_SRC, %edx

	subl	$2, %ecx
	ja	L(three_or_more)

	movl	(%edx), %eax
	jne	L(one)


	movl	4(%edx), %ecx
	movl	%eax, %edx

	shrl	$24, %edx
	andl	$0xFFFFFF, %eax

	addl	%edx, %eax
	movl	%ecx, %edx

	shrl	$16, %ecx
	andl	$0xFFFF, %edx

	shll	$8, %edx
	addl	%ecx, %eax

	addl	%edx, %eax

L(one):
	ret


L(three_or_more):
	C eax
	C ebx
	C ecx	size-2
	C edx	src
	C esi
	C edi
	C ebp

	pushl	%ebx	FRAME_pushl()
	pushl	%esi	FRAME_pushl()

	pushl	%edi	FRAME_pushl()
	pushl	%ebp	FRAME_pushl()

	xorl	%esi, %esi		C 0mod3
	xorl	%edi, %edi		C 1mod3

	xorl	%ebp, %ebp		C 2mod3, and clear carry

L(top):
	C eax	scratch
	C ebx	scratch
	C ecx	counter, limbs
	C edx	src
	C esi	0mod3
	C edi	1mod3
	C ebp	2mod3

	movl	(%edx), %eax
	movl	4(%edx), %ebx

	adcl	%eax, %esi
	movl	8(%edx), %eax

	adcl	%ebx, %edi
	leal	12(%edx), %edx

	adcl	%eax, %ebp
	leal	-2(%ecx), %ecx

	decl	%ecx
	jg	L(top)


	C ecx is -2, -1 or 0, representing 0, 1 or 2 more limbs, respectively

	movl	$0xFFFFFFFF, %ebx	C mask
	incl	%ecx

	js	L(combine)		C 0 more

	movl	(%edx), %eax
	movl	$0xFFFFFF00, %ebx

	adcl	%eax, %esi
	decl	%ecx

	js	L(combine)		C 1 more

	movl	4(%edx), %eax
	movl	$0xFFFF0000, %ebx

	adcl	%eax, %edi



L(combine):
	C eax
	C ebx	mask
	C ecx
	C edx
	C esi	0mod3
	C edi	1mod3
	C ebp	2mod3

	sbbl	%ecx, %ecx		C carry
	movl	%esi, %eax		C 0mod3

	andl	%ebx, %ecx		C masked for position
	andl	$0xFFFFFF, %eax		C 0mod3 low

	shrl	$24, %esi		C 0mod3 high
	subl	%ecx, %eax		C apply carry

	addl	%esi, %eax		C apply 0mod3
	movl	%edi, %ebx		C 1mod3

	shrl	$16, %edi		C 1mod3 high
	andl	$0x0000FFFF, %ebx

	shll	$8, %ebx		C 1mod3 low
	addl	%edi, %eax		C apply 1mod3 high

	addl	%ebx, %eax		C apply 1mod3 low
	movl	%ebp, %ebx		C 2mod3

	shrl	$8, %ebp		C 2mod3 high
	andl	$0xFF, %ebx

	shll	$16, %ebx		C 2mod3 low
	addl	%ebp, %eax		C apply 2mod3 high

	addl	%ebx, %eax		C apply 2mod3 low

	popl	%ebp
	popl	%edi

	popl	%esi
	popl	%ebx

	ret

EPILOGUE()
