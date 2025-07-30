dnl  Intel Pentium mpn_mul_2 -- mpn by 2-limb multiplication.

dnl  Copyright 2001, 2002 Free Software Foundation, Inc.

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


C P5: 24.0 cycles/limb


C mp_limb_t mpn_mul_2 (mp_ptr dst, mp_srcptr src, mp_size_t size,
C                      mp_srcptr mult);
C
C At 24 c/l this is only 2 cycles faster than a separate mul_1 and addmul_1,
C but has the advantage of making just one pass over the operands.
C
C There's not enough registers to use PARAM_MULT directly, so the multiplier
C limbs are transferred to local variables on the stack.

defframe(PARAM_MULT, 16)
defframe(PARAM_SIZE, 12)
defframe(PARAM_SRC,   8)
defframe(PARAM_DST,   4)

dnl  re-use parameter space
define(VAR_MULT_LOW, `PARAM_SRC')
define(VAR_MULT_HIGH,`PARAM_DST')

	TEXT
	ALIGN(8)
PROLOGUE(mpn_mul_2)
deflit(`FRAME',0)

	pushl	%esi		FRAME_pushl()
	pushl	%edi		FRAME_pushl()

	movl	PARAM_SRC, %esi
	movl	PARAM_DST, %edi

	movl	PARAM_MULT, %eax
	movl	PARAM_SIZE, %ecx

	movl	4(%eax), %edx		C mult high
	movl	(%eax), %eax		C mult low

	movl	%eax, VAR_MULT_LOW
	movl	%edx, VAR_MULT_HIGH

	pushl	%ebx		FRAME_pushl()
	pushl	%ebp		FRAME_pushl()

	mull	(%esi)			C src[0] * mult[0]

	movl	%eax, %ebp		C in case src==dst
	movl	(%esi), %eax		C src[0]

	movl	%ebp, (%edi)		C dst[0]
	movl	%edx, %ebx		C initial low carry

	xorl	%ebp, %ebp		C initial high carry
	leal	(%edi,%ecx,4), %edi	C dst end

	mull	VAR_MULT_HIGH		C src[0] * mult[1]

	subl	$2, %ecx		C size-2
	js	L(done)

	leal	8(%esi,%ecx,4), %esi	C &src[size]
	xorl	$-1, %ecx		C -(size-1)



L(top):
	C eax	low prod
	C ebx	low carry
	C ecx	counter, negative
	C edx	high prod
	C esi	src end
	C edi	dst end
	C ebp	high carry (0 or -1)

	andl	$1, %ebp		C 1 or 0
	addl	%eax, %ebx

	adcl	%edx, %ebp
	ASSERT(nc)
	movl	(%esi,%ecx,4), %eax

	mull	VAR_MULT_LOW

	addl	%eax, %ebx		C low carry
	movl	(%esi,%ecx,4), %eax

	adcl	%ebp, %edx		C high carry
	movl	%ebx, (%edi,%ecx,4)

	sbbl	%ebp, %ebp		C new high carry, -1 or 0
	movl	%edx, %ebx		C new low carry

	mull	VAR_MULT_HIGH

	incl	%ecx
	jnz	L(top)


L(done):
	andl	$1, %ebp		C 1 or 0
	addl	%ebx, %eax

	adcl	%ebp, %edx
	ASSERT(nc)
	movl	%eax, (%edi)		C store carry low

	movl	%edx, %eax		C return carry high

	popl	%ebp
	popl	%ebx

	popl	%edi
	popl	%esi

	ret

EPILOGUE()
