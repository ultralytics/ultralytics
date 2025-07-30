dnl  AMD K6 mpn_lshift -- mpn left shift.

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


C K6: 3.0 cycles/limb


C mp_limb_t mpn_lshift (mp_ptr dst, mp_srcptr src, mp_size_t size,
C                       unsigned shift);
C
C The loop runs at 3 cycles/limb, limited by decoding and by having 3 mmx
C instructions.  This is despite every second fetch being unaligned.


defframe(PARAM_SHIFT,16)
defframe(PARAM_SIZE, 12)
defframe(PARAM_SRC,  8)
defframe(PARAM_DST,  4)

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


	ALIGN(16)	C avoid offset 0x1f
	nop		C avoid bad cache line crossing
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

	movd	%ecx, %mm7
	movl	PARAM_DST, %ecx

L(top):
	C eax	counter, size-1 to 1
	C ebx	src
	C ecx	dst
	C edx	retval
	C
	C mm0	scratch
	C mm6	shift
	C mm7	32-shift

	movq	-4(%ebx,%eax,4), %mm0
	decl	%eax

	psrlq	%mm7, %mm0

	movd	%mm0, 4(%ecx,%eax,4)
	jnz	L(top)


	movd	(%ebx), %mm0
	popl	%ebx

	psllq	%mm6, %mm0
	movl	%edx, %eax

	movd	%mm0, (%ecx)

	emms
	ret

EPILOGUE()
