dnl  AMD K6 mpn_gcd_1 -- mpn by 1 gcd.

dnl  Copyright 2000-2002, 2004, 2014 Free Software Foundation, Inc.

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


C K6: 9.5 cycles/bit (approx)   1x1 gcd
C     11.0 cycles/limb          Nx1 reduction (modexact_1_odd)


C mp_limb_t mpn_gcd_1 (mp_srcptr src, mp_size_t size, mp_limb_t y);
C
C This code is nothing very special, but offers a speedup over what gcc 2.95
C can do with mpn/generic/gcd_1.c.
C
C Future:
C
C Using a lookup table to count trailing zeros seems a touch quicker, but
C after a slightly longer startup.  Might be worthwhile if an mpn_gcd_2 used
C it too.


dnl  If size==1 and x (the larger operand) is more than DIV_THRESHOLD bits
dnl  bigger than y, then a division x%y is done to reduce it.
dnl
dnl  A divl is 20 cycles and the loop runs at about 9.5 cycles/bitpair so
dnl  there should be an advantage in the divl at about 4 or 5 bits, which is
dnl  what's found.

deflit(DIV_THRESHOLD, 5)


defframe(PARAM_LIMB, 12)
defframe(PARAM_SIZE,  8)
defframe(PARAM_SRC,   4)

	TEXT
	ALIGN(16)

PROLOGUE(mpn_gcd_1)
deflit(`FRAME',0)

	ASSERT(ne, `cmpl $0, PARAM_LIMB')
	ASSERT(ae, `cmpl $1, PARAM_SIZE')


	movl	PARAM_SRC, %eax
	pushl	%ebx			FRAME_pushl()

	movl	PARAM_LIMB, %edx
	movl	$-1, %ecx

	movl	(%eax), %ebx		C src low limb

	movl	%ebx, %eax		C src low limb
	orl	%edx, %ebx

L(common_twos):
	shrl	%ebx
	incl	%ecx

	jnc	L(common_twos)		C 1/4 chance on random data
	shrl	%cl, %edx		C y

	cmpl	$1, PARAM_SIZE
	ja	L(size_two_or_more)


	ASSERT(nz, `orl	%eax, %eax')	C should have src limb != 0

	shrl	%cl, %eax		C x


	C Swap if necessary to make x>=y.  Measures a touch quicker as a
	C jump than a branch free calculation.
	C
	C eax	x
	C ebx
	C ecx	common twos
	C edx	y

	movl	%eax, %ebx
	cmpl	%eax, %edx

	jb	L(noswap)
	movl	%edx, %eax

	movl	%ebx, %edx
	movl	%eax, %ebx
L(noswap):


	C See if it's worth reducing x with a divl.
	C
	C eax	x
	C ebx	x
	C ecx	common twos
	C edx	y

	shrl	$DIV_THRESHOLD, %ebx

	cmpl	%ebx, %edx
	ja	L(nodiv)


	C Reduce x to x%y.
	C
	C eax	x
	C ebx
	C ecx	common twos
	C edx	y

	movl	%edx, %ebx
	xorl	%edx, %edx

	divl	%ebx

	orl	%edx, %edx	C y
	nop	C code alignment

	movl	%ebx, %eax	C x
	jz	L(done_shll)
L(nodiv):


	C eax	x
	C ebx
	C ecx	common twos
	C edx	y
	C esi
	C edi
	C ebp

L(strip_y):
	shrl	%edx
	jnc	L(strip_y)

	leal	1(%edx,%edx), %edx
	movl	%ecx, %ebx	C common twos

	leal	1(%eax), %ecx
	jmp	L(strip_x_and)


C Calculating a %cl shift based on the low bit 0 or 1 avoids doing a branch
C on a 50/50 chance of 0 or 1.  The chance of the next bit also being 0 is
C only 1/4.
C
C A second computed %cl shift was tried, but that measured a touch slower
C than branching back.
C
C A branch-free abs(x-y) and min(x,y) calculation was tried, but that
C measured about 1 cycle/bit slower.

	C eax	x
	C ebx	common twos
	C ecx	scratch
	C edx	y

	ALIGN(4)
L(swap):
	addl	%eax, %edx	C x-y+y = x
	negl	%eax		C -(x-y) = y-x

L(strip_x):
	shrl	%eax		C odd-odd = even, so always one to strip
	ASSERT(nz)

L(strip_x_leal):
	leal	1(%eax), %ecx

L(strip_x_and):
	andl	$1, %ecx	C (x^1)&1

	shrl	%cl, %eax	C shift if x even

	testb	$1, %al
	jz	L(strip_x)

	ASSERT(nz,`testl $1, %eax')	C x, y odd
	ASSERT(nz,`testl $1, %edx')

	subl	%edx, %eax
	jb	L(swap)
	ja	L(strip_x)


	movl	%edx, %eax
	movl	%ebx, %ecx

L(done_shll):
	shll	%cl, %eax
	popl	%ebx

	ret


C -----------------------------------------------------------------------------
C Two or more limbs.
C
C x={src,size} is reduced modulo y using either a plain mod_1 style
C remainder, or a modexact_1 style exact division.

deflit(MODEXACT_THRESHOLD, ifdef(`PIC', 4, 4))

	ALIGN(8)
L(size_two_or_more):
	C eax
	C ebx
	C ecx	common twos
	C edx	y, without common twos
	C esi
	C edi
	C ebp

deflit(FRAME_TWO_OR_MORE, FRAME)

	pushl	%edi		defframe_pushl(SAVE_EDI)
	movl	PARAM_SRC, %ebx

L(y_twos):
	shrl	%edx
	jnc	L(y_twos)

	movl	%ecx, %edi		C common twos
	movl	PARAM_SIZE, %ecx

	pushl	%esi		defframe_pushl(SAVE_ESI)
	leal	1(%edx,%edx), %esi	C y (odd)

	movl	-4(%ebx,%ecx,4), %eax	C src high limb

	cmpl	%edx, %eax		C carry if high<divisor

	sbbl	%edx, %edx		C -1 if high<divisor

	addl	%edx, %ecx		C skip one limb if high<divisor
	andl	%eax, %edx

	cmpl	$MODEXACT_THRESHOLD, %ecx
	jae	L(modexact)


L(divide_top):
	C eax	scratch (quotient)
	C ebx	src
	C ecx	counter, size-1 to 1
	C edx	carry (remainder)
	C esi	divisor (odd)
	C edi
	C ebp

	movl	-4(%ebx,%ecx,4), %eax
	divl	%esi
	loop	L(divide_top)


	movl	%edx, %eax	C x
	movl	%esi, %edx	C y (odd)

	movl	%edi, %ebx	C common twos
	popl	%esi

	popl	%edi
	leal	1(%eax), %ecx

	orl	%eax, %eax
	jnz	L(strip_x_and)


	movl	%ebx, %ecx
	movl	%edx, %eax

	shll	%cl, %eax
	popl	%ebx

	ret


	ALIGN(8)
L(modexact):
	C eax
	C ebx	src ptr
	C ecx	size or size-1
	C edx
	C esi	y odd
	C edi	common twos
	C ebp

	movl	PARAM_SIZE, %eax
	pushl	%esi		FRAME_pushl()

	pushl	%eax		FRAME_pushl()

	pushl	%ebx		FRAME_pushl()

ifdef(`PIC_WITH_EBX',`
	nop	C code alignment
	call	L(movl_eip_ebx)
	add	$_GLOBAL_OFFSET_TABLE_, %ebx
')
	CALL(	mpn_modexact_1_odd)

	movl	%esi, %edx		C y odd
	movl	SAVE_ESI, %esi

	movl	%edi, %ebx		C common twos
	movl	SAVE_EDI, %edi

	addl	$eval(FRAME - FRAME_TWO_OR_MORE), %esp
	orl	%eax, %eax

	leal	1(%eax), %ecx
	jnz	L(strip_x_and)


	movl	%ebx, %ecx
	movl	%edx, %eax

	shll	%cl, %eax
	popl	%ebx

	ret


ifdef(`PIC_WITH_EBX',`
L(movl_eip_ebx):
	movl	(%esp), %ebx
	ret_internal
')

EPILOGUE()
