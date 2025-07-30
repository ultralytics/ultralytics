dnl  Intel Pentium mpn_mul_1 -- mpn by limb multiplication.

dnl  Copyright 1992, 1994, 1996, 1999, 2000, 2002 Free Software Foundation,
dnl  Inc.

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


C P5: 12.0 cycles/limb


C mp_limb_t mpn_mul_1 (mp_ptr dst, mp_srcptr src, mp_size_t size,
C                      mp_limb_t multiplier);
C mp_limb_t mpn_mul_1c (mp_ptr dst, mp_srcptr src, mp_size_t size,
C                       mp_limb_t multiplier, mp_limb_t carry);
C

defframe(PARAM_CARRY,     20)
defframe(PARAM_MULTIPLIER,16)
defframe(PARAM_SIZE,      12)
defframe(PARAM_SRC,       8)
defframe(PARAM_DST,       4)

	TEXT
	ALIGN(8)
PROLOGUE(mpn_mul_1c)
deflit(`FRAME',0)

	movl	PARAM_CARRY, %ecx
	pushl	%esi		FRAME_pushl()

	jmp	L(start_1c)

EPILOGUE()


	ALIGN(8)
PROLOGUE(mpn_mul_1)
deflit(`FRAME',0)

	xorl	%ecx, %ecx
	pushl	%esi		FRAME_pushl()

L(start_1c):
	movl	PARAM_SRC, %esi
	movl	PARAM_SIZE, %eax

	shrl	%eax
	jnz	L(two_or_more)


	C one limb only

	movl	(%esi), %eax

	mull	PARAM_MULTIPLIER

	addl	%eax, %ecx
	movl	PARAM_DST, %eax

	adcl	$0, %edx
	popl	%esi

	movl	%ecx, (%eax)
	movl	%edx, %eax

	ret


L(two_or_more):
	C eax	size/2
	C ebx
	C ecx	carry
	C edx
	C esi	src
	C edi
	C ebp

	pushl	%edi		FRAME_pushl()
	pushl	%ebx		FRAME_pushl()

	movl	PARAM_DST, %edi
	leal	-1(%eax), %ebx		C size/2-1

	notl	%ebx			C -size, preserve carry

	leal	(%esi,%eax,8), %esi	C src end
	leal	(%edi,%eax,8), %edi	C dst end

	pushl	%ebp		FRAME_pushl()
	jnc	L(top)


	C size was odd, process one limb separately

	movl	(%esi,%ebx,8), %eax
	addl	$4, %esi

	mull	PARAM_MULTIPLIER

	addl	%ecx, %eax
	movl	%edx, %ecx

	movl	%eax, (%edi,%ebx,8)
	leal	4(%edi), %edi


L(top):
	C eax
	C ebx	counter, negative
	C ecx	carry
	C edx
	C esi	src end
	C edi	dst end
	C ebp

	adcl	$0, %ecx
	movl	(%esi,%ebx,8), %eax

	mull	PARAM_MULTIPLIER

	movl	%edx, %ebp
	addl	%eax, %ecx

	adcl	$0, %ebp
	movl	4(%esi,%ebx,8), %eax

	mull	PARAM_MULTIPLIER

	movl	%ecx, (%edi,%ebx,8)
	addl	%ebp, %eax

	movl	%eax, 4(%edi,%ebx,8)
	incl	%ebx

	movl	%edx, %ecx
	jnz	L(top)


	adcl	$0, %ecx
	popl	%ebp

	movl	%ecx, %eax
	popl	%ebx

	popl	%edi
	popl	%esi

	ret

EPILOGUE()
