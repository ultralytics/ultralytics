dnl  Generic x86 mpn_mod_34lsub1 -- mpn remainder modulo 2^24-1.

dnl  Copyright 2000-2002, 2004 Free Software Foundation, Inc.

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


C      cycles/limb
C P5	  3.0
C P6	  3.66
C K6	  3.0
C K7	  1.3
C P4	  9


C mp_limb_t mpn_mod_34lsub1 (mp_srcptr src, mp_size_t size)
C

defframe(PARAM_SIZE, 8)
defframe(PARAM_SRC,  4)

dnl  re-use parameter space
define(SAVE_EBX, `PARAM_SRC')

	TEXT
	ALIGN(16)
PROLOGUE(mpn_mod_34lsub1)
deflit(`FRAME',0)

	movl	PARAM_SIZE, %ecx
	movl	PARAM_SRC, %edx

	subl	$2, %ecx
	ja	L(three_or_more)

	movl	(%edx), %eax
	jb	L(one)

	movl	4(%edx), %ecx
	movl	%eax, %edx
	shrl	$24, %eax		C src[0] low

	andl	$0xFFFFFF, %edx		C src[0] high
	addl	%edx, %eax
	movl	%ecx, %edx

	andl	$0xFFFF, %ecx
	shrl	$16, %edx		C src[1] high
	addl	%edx, %eax

	shll	$8, %ecx		C src[1] low
	addl	%ecx, %eax

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

	movl	%ebx, SAVE_EBX		C and arrange 16-byte loop alignment
	xorl	%ebx, %ebx

	pushl	%esi	FRAME_pushl()
	xorl	%esi, %esi

	pushl	%edi	FRAME_pushl()
	xorl	%eax, %eax		C and clear carry flag


	C offset 0x40 here
L(top):
	C eax	acc 0mod3
	C ebx	acc 1mod3
	C ecx	counter, limbs
	C edx	src
	C esi	acc 2mod3
	C edi
	C ebp

	leal	12(%edx), %edx
	leal	-2(%ecx), %ecx

	adcl	-12(%edx), %eax
	adcl	-8(%edx), %ebx
	adcl	-4(%edx), %esi

	decl	%ecx
	jg	L(top)


	C ecx is -2, -1 or 0 representing 0, 1 or 2 more limbs, respectively

	movl	$0xFFFFFFFF, %edi
	incl	%ecx
	js	L(combine)

	adcl	(%edx), %eax
	movl	$0xFFFFFF00, %edi
	decl	%ecx
	js	L(combine)

	adcl	4(%edx), %ebx
	movl	$0xFFFF0000, %edi


L(combine):
	C eax	acc 0mod3
	C ebx	acc 1mod3
	C ecx
	C edx
	C esi	acc 2mod3
	C edi	mask
	C ebp

	sbbl	%ecx, %ecx		C carry
	movl	%eax, %edx		C 0mod3

	shrl	$24, %eax		C 0mod3 high
	andl	%edi, %ecx		C carry masked

	subl	%ecx, %eax		C apply carry
	movl	%ebx, %edi		C 1mod3

	shrl	$16, %ebx		C 1mod3 high
	andl	$0x00FFFFFF, %edx	C 0mod3 low

	addl	%edx, %eax		C apply 0mod3 low
	andl	$0xFFFF, %edi

	shll	$8, %edi		C 1mod3 low
	addl	%ebx, %eax		C apply 1mod3 high

	addl	%edi, %eax		C apply 1mod3 low
	movl	%esi, %edx		C 2mod3

	shrl	$8, %esi		C 2mod3 high
	andl	$0xFF, %edx		C 2mod3 low

	shll	$16, %edx		C 2mod3 low
	addl	%esi, %eax		C apply 2mod3 high

	addl	%edx, %eax		C apply 2mod3 low
	popl	%edi	FRAME_popl()

	movl	SAVE_EBX, %ebx
	popl	%esi	FRAME_popl()

	ret

EPILOGUE()
