dnl  AMD K7 mpn_copyd -- copy limb vector, decrementing.

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


C    alignment dst/src, A=0mod8 N=4mod8
C       A/A   A/N   N/A   N/N
C K7    0.75  1.0   1.0   0.75


C void mpn_copyd (mp_ptr dst, mp_srcptr src, mp_size_t size);
C
C The various comments in mpn/x86/k7/copyi.asm apply here too.

defframe(PARAM_SIZE,12)
defframe(PARAM_SRC, 8)
defframe(PARAM_DST, 4)
deflit(`FRAME',0)

dnl  parameter space reused
define(SAVE_EBX,`PARAM_SIZE')
define(SAVE_ESI,`PARAM_SRC')

dnl  minimum 5 since the unrolled code can't handle less than 5
deflit(UNROLL_THRESHOLD, 5)

	TEXT
	ALIGN(32)
PROLOGUE(mpn_copyd)

	movl	PARAM_SIZE, %ecx
	movl	%ebx, SAVE_EBX

	movl	PARAM_SRC, %eax
	movl	PARAM_DST, %edx

	cmpl	$UNROLL_THRESHOLD, %ecx
	jae	L(unroll)

	orl	%ecx, %ecx
	jz	L(simple_done)

L(simple):
	C eax	src
	C ebx	scratch
	C ecx	counter
	C edx	dst
	C
	C this loop is 2 cycles/limb

	movl	-4(%eax,%ecx,4), %ebx
	movl	%ebx, -4(%edx,%ecx,4)
	decl	%ecx
	jnz	L(simple)

L(simple_done):
	movl	SAVE_EBX, %ebx
	ret


L(unroll):
	movl	%esi, SAVE_ESI
	leal	(%eax,%ecx,4), %ebx
	leal	(%edx,%ecx,4), %esi

	andl	%esi, %ebx
	movl	SAVE_ESI, %esi
	subl	$4, %ecx		C size-4

	testl	$4, %ebx   C testl to pad code closer to 16 bytes for L(top)
	jz	L(aligned)

	C both src and dst unaligned, process one limb to align them
	movl	12(%eax,%ecx,4), %ebx
	movl	%ebx, 12(%edx,%ecx,4)
	decl	%ecx
L(aligned):


	ALIGN(16)
L(top):
	C eax	src
	C ebx
	C ecx	counter, limbs
	C edx	dst

	movq	8(%eax,%ecx,4), %mm0
	movq	(%eax,%ecx,4), %mm1
	subl	$4, %ecx
	movq	%mm0, 16+8(%edx,%ecx,4)
	movq	%mm1, 16(%edx,%ecx,4)
	jns	L(top)


	C now %ecx is -4 to -1 representing respectively 0 to 3 limbs remaining

	testb	$2, %cl
	jz	L(finish_not_two)

	movq	8(%eax,%ecx,4), %mm0
	movq	%mm0, 8(%edx,%ecx,4)
L(finish_not_two):

	testb	$1, %cl
	jz	L(done)

	movl	(%eax), %ebx
	movl	%ebx, (%edx)

L(done):
	movl	SAVE_EBX, %ebx
	emms
	ret


EPILOGUE()
