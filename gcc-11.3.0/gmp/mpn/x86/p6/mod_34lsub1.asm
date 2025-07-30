dnl  Intel P6 mpn_mod_34lsub1 -- remainder modulo 2^24-1.

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


C P6: 2.0 cycles/limb

C TODO
C  Experiments with more unrolling indicate that 1.5 c/l is possible on P6-13
C  with the current carry handling scheme.

C mp_limb_t mpn_mod_34lsub1 (mp_srcptr src, mp_size_t size)
C
C Groups of three limbs are handled, with carry bits from 0mod3 into 1mod3
C into 2mod3, but at that point going into a separate carries total so we
C don't keep the carry flag live across the loop control.  Avoiding decl
C lets us get to 2.0 c/l, as compared to the generic x86 code at 3.66.
C

defframe(PARAM_SIZE, 8)
defframe(PARAM_SRC,  4)

dnl  re-use parameter space
define(SAVE_EBX, `PARAM_SIZE')
define(SAVE_ESI, `PARAM_SRC')

	TEXT
	ALIGN(16)
PROLOGUE(mpn_mod_34lsub1)
deflit(`FRAME',0)

	movl	PARAM_SIZE, %ecx
	movl	PARAM_SRC, %edx

	subl	$2, %ecx		C size-2
	movl	(%edx), %eax		C src[0]
	ja	L(three_or_more)
	jb	L(one)

	C size==2

	movl	4(%edx), %ecx		C src[1]

	movl	%eax, %edx		C src[0]
	shrl	$24, %eax		C src[0] high

	andl	$0xFFFFFF, %edx		C src[0] low

	addl	%edx, %eax
	movl	%ecx, %edx		C src[1]
	shrl	$16, %ecx		C src[1] high

	andl	$0xFFFF, %edx
	addl	%ecx, %eax

	shll	$8, %edx		C src[1] low

	addl	%edx, %eax
L(one):
	ret


L(three_or_more):
	C eax	src[0], initial acc 0mod3
	C ebx
	C ecx	size-2
	C edx	src
	C esi
	C edi
	C ebp

	movl	%ebx, SAVE_EBX
	movl	4(%edx), %ebx		C src[1], initial 1mod3
	subl	$3, %ecx		C size-5

	movl	%esi, SAVE_ESI
	movl	8(%edx), %esi		C src[2], initial 2mod3

	pushl	%edi	FRAME_pushl()
	movl	$0, %edi		C initial carries 0mod3
	jng	L(done)			C if size < 6


L(top):
	C eax	acc 0mod3
	C ebx	acc 1mod3
	C ecx	counter, limbs
	C edx	src
	C esi	acc 2mod3
	C edi	carrys into 0mod3
	C ebp

	addl	12(%edx), %eax
	adcl	16(%edx), %ebx
	adcl	20(%edx), %esi
	leal	12(%edx), %edx
	adcl	$0, %edi

	subl	$3, %ecx
	jg	L(top)			C at least 3 more to process


L(done):
	C ecx is -2, -1 or 0 representing 0, 1 or 2 more limbs respectively
	cmpl	$-1, %ecx
	jl	L(done_0)		C if -2, meaning 0 more limbs

	C 1 or 2 more limbs
	movl	$0, %ecx
	je	L(done_1)		C if -1, meaning 1 more limb only
	movl	16(%edx), %ecx
L(done_1):
	addl	12(%edx), %eax		C 0mod3
	adcl	%ecx, %ebx		C 1mod3
	adcl	$0, %esi		C 2mod3
	adcl	$0, %edi		C carries 0mod3

L(done_0):
	C eax	acc 0mod3
	C ebx	acc 1mod3
	C ecx
	C edx
	C esi	acc 2mod3
	C edi	carries 0mod3
	C ebp

	movl	%eax, %ecx		C 0mod3
	shrl	$24, %eax		C 0mod3 high initial total

	andl	$0xFFFFFF, %ecx		C 0mod3 low
	movl	%edi, %edx		C carries
	shrl	$24, %edi		C carries high

	addl	%ecx, %eax		C add 0mod3 low
	andl	$0xFFFFFF, %edx		C carries 0mod3 low
	movl	%ebx, %ecx		C 1mod3

	shrl	$16, %ebx		C 1mod3 high
	addl	%edi, %eax		C add carries high
	addl	%edx, %eax		C add carries 0mod3 low

	andl	$0xFFFF, %ecx		C 1mod3 low mask
	addl	%ebx, %eax		C add 1mod3 high
	movl	SAVE_EBX, %ebx

	shll	$8, %ecx		C 1mod3 low
	movl	%esi, %edx		C 2mod3
	popl	%edi	FRAME_popl()

	shrl	$8, %esi		C 2mod3 high
	andl	$0xFF, %edx		C 2mod3 low mask
	addl	%ecx, %eax		C add 1mod3 low

	shll	$16, %edx		C 2mod3 low
	addl	%esi, %eax		C add 2mod3 high
	movl	SAVE_ESI, %esi

	addl	%edx, %eax		C add 2mod3 low

	ret

EPILOGUE()
