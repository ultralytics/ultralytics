dnl  AMD K6 mpn_mul_1 -- mpn by limb multiply.

dnl  Copyright 1999, 2000, 2002, 2005 Free Software Foundation, Inc.

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


C			    cycles/limb
C P5
C P6 model 0-8,10-12		 5.5
C P6 model 9  (Banias)
C P6 model 13 (Dothan)		 4.87
C P4 model 0  (Willamette)
C P4 model 1  (?)
C P4 model 2  (Northwood)
C P4 model 3  (Prescott)
C P4 model 4  (Nocona)
C AMD K6			 6.25
C AMD K7
C AMD K8


C mp_limb_t mpn_mul_1 (mp_ptr dst, mp_srcptr src, mp_size_t size,
C                      mp_limb_t multiplier);
C mp_limb_t mpn_mul_1c (mp_ptr dst, mp_srcptr src, mp_size_t size,
C                       mp_limb_t multiplier, mp_limb_t carry);
C
C Multiply src,size by mult and store the result in dst,size.
C Return the carry limb from the top of the result.
C
C mpn_mul_1c() accepts an initial carry for the calculation, it's added into
C the low limb of the result.

defframe(PARAM_CARRY,     20)
defframe(PARAM_MULTIPLIER,16)
defframe(PARAM_SIZE,      12)
defframe(PARAM_SRC,       8)
defframe(PARAM_DST,       4)

dnl  minimum 5 because the unrolled code can't handle less
deflit(UNROLL_THRESHOLD, 5)

	TEXT
	ALIGN(32)

PROLOGUE(mpn_mul_1c)
	pushl	%esi
deflit(`FRAME',4)
	movl	PARAM_CARRY, %esi
	jmp	L(start_nc)
EPILOGUE()


PROLOGUE(mpn_mul_1)
	push	%esi
deflit(`FRAME',4)
	xorl	%esi, %esi	C initial carry

L(start_nc):
	mov	PARAM_SIZE, %ecx
	push	%ebx
FRAME_pushl()

	movl	PARAM_SRC, %ebx
	push	%edi
FRAME_pushl()

	movl	PARAM_DST, %edi
	pushl	%ebp
FRAME_pushl()

	cmpl	$UNROLL_THRESHOLD, %ecx
	movl	PARAM_MULTIPLIER, %ebp

	jae	L(unroll)


	C code offset 0x22 here, close enough to aligned
L(simple):
	C eax	scratch
	C ebx	src
	C ecx	counter
	C edx	scratch
	C esi	carry
	C edi	dst
	C ebp	multiplier
	C
	C this loop 8 cycles/limb

	movl	(%ebx), %eax
	addl	$4, %ebx

	mull	%ebp

	addl	%esi, %eax
	movl	$0, %esi

	adcl	%edx, %esi

	movl	%eax, (%edi)
	addl	$4, %edi

	loop	L(simple)


	popl	%ebp

	popl	%edi
	popl	%ebx

	movl	%esi, %eax
	popl	%esi

	ret


C -----------------------------------------------------------------------------
C The code for each limb is 6 cycles, with instruction decoding being the
C limiting factor.  At 4 limbs/loop and 1 cycle/loop of overhead it's 6.25
C cycles/limb in total.
C
C The secret ingredient to get 6.25 is to start the loop with the mul and
C have the load/store pair at the end.  Rotating the load/store to the top
C is an 0.5 c/l slowdown.  (Some address generation effect probably.)
C
C The whole unrolled loop fits nicely in exactly 80 bytes.


	ALIGN(16)	C already aligned to 16 here actually
L(unroll):
	movl	(%ebx), %eax
	leal	-16(%ebx,%ecx,4), %ebx

	leal	-16(%edi,%ecx,4), %edi
	subl	$4, %ecx

	negl	%ecx


	ALIGN(16)	C one byte nop for this alignment
L(top):
	C eax	scratch
	C ebx	&src[size-4]
	C ecx	counter
	C edx	scratch
	C esi	carry
	C edi	&dst[size-4]
	C ebp	multiplier

	mull	%ebp

	addl	%esi, %eax
	movl	$0, %esi

	adcl	%edx, %esi

	movl	%eax, (%edi,%ecx,4)
	movl	4(%ebx,%ecx,4), %eax


	mull	%ebp

	addl	%esi, %eax
	movl	$0, %esi

	adcl	%edx, %esi

	movl	%eax, 4(%edi,%ecx,4)
	movl	8(%ebx,%ecx,4), %eax


	mull	%ebp

	addl	%esi, %eax
	movl	$0, %esi

	adcl	%edx, %esi

	movl	%eax, 8(%edi,%ecx,4)
	movl	12(%ebx,%ecx,4), %eax


	mull	%ebp

	addl	%esi, %eax
	movl	$0, %esi

	adcl	%edx, %esi

	movl	%eax, 12(%edi,%ecx,4)
	movl	16(%ebx,%ecx,4), %eax


	addl	$4, %ecx
	js	L(top)



	C eax	next src limb
	C ebx	&src[size-4]
	C ecx	0 to 3 representing respectively 4 to 1 further limbs
	C edx
	C esi	carry
	C edi	&dst[size-4]

	testb	$2, %cl
	jnz	L(finish_not_two)

	mull	%ebp

	addl	%esi, %eax
	movl	$0, %esi

	adcl	%edx, %esi

	movl	%eax, (%edi,%ecx,4)
	movl	4(%ebx,%ecx,4), %eax


	mull	%ebp

	addl	%esi, %eax
	movl	$0, %esi

	adcl	%edx, %esi

	movl	%eax, 4(%edi,%ecx,4)
	movl	8(%ebx,%ecx,4), %eax

	addl	$2, %ecx
L(finish_not_two):


	testb	$1, %cl
	jnz	L(finish_not_one)

	mull	%ebp

	addl	%esi, %eax
	movl	$0, %esi

	adcl	%edx, %esi

	movl	%eax, 8(%edi)
	movl	12(%ebx), %eax
L(finish_not_one):


	mull	%ebp

	addl	%esi, %eax
	popl	%ebp

	adcl	$0, %edx

	movl	%eax, 12(%edi)
	popl	%edi

	popl	%ebx
	movl	%edx, %eax

	popl	%esi

	ret

EPILOGUE()
