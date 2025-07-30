dnl  AMD K6 mpn_divrem_1 -- mpn by limb division.

dnl  Copyright 1999-2003, 2007 Free Software Foundation, Inc.

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


C K6: 20 cycles/limb


C mp_limb_t mpn_divrem_1 (mp_ptr dst, mp_size_t xsize,
C                         mp_srcptr src, mp_size_t size, mp_limb_t divisor);
C mp_limb_t mpn_divrem_1c (mp_ptr dst, mp_size_t xsize,
C                          mp_srcptr src, mp_size_t size, mp_limb_t divisor,
C                          mp_limb_t carry);
C
C The code here is basically the same as mpn/x86/divrem_1.asm, but uses loop
C instead of decl+jnz, since it comes out 2 cycles/limb faster.
C
C A test is done to see if the high limb is less than the divisor, and if so
C one less div is done.  A div is 20 cycles, so assuming high<divisor about
C half the time, then this test saves half that amount.  The branch
C misprediction penalty is less than that.
C
C Back-to-back div instructions run at 20 cycles, the same as the loop here,
C so it seems there's nothing to gain by rearranging the loop.  Pairing the
C mov and loop instructions was found to gain nothing.
C
C Enhancements:
C
C The low-latency K6 multiply might be thought to suit a mul-by-inverse, but
C that algorithm has been found to suffer from the relatively poor carry
C handling on K6 and too many auxiliary instructions.  The fractional part
C however could be done at about 13 c/l, if it mattered enough.

defframe(PARAM_CARRY,  24)
defframe(PARAM_DIVISOR,20)
defframe(PARAM_SIZE,   16)
defframe(PARAM_SRC,    12)
defframe(PARAM_XSIZE,  8)
defframe(PARAM_DST,    4)

	TEXT

	ALIGN(32)
PROLOGUE(mpn_divrem_1c)
deflit(`FRAME',0)

	movl	PARAM_SIZE, %ecx
	pushl	%edi		FRAME_pushl()

	movl	PARAM_SRC, %edi
	pushl	%esi		FRAME_pushl()

	movl	PARAM_DIVISOR, %esi
	pushl	%ebx		FRAME_pushl()

	movl	PARAM_DST, %ebx
	pushl	%ebp		FRAME_pushl()

	movl	PARAM_XSIZE, %ebp
	orl	%ecx, %ecx		C size

	movl	PARAM_CARRY, %edx
	jz	L(fraction)		C if size==0

	leal	-4(%ebx,%ebp,4), %ebx	C dst one limb below integer part
	jmp	L(integer_top)

EPILOGUE()


	ALIGN(16)
PROLOGUE(mpn_divrem_1)
deflit(`FRAME',0)

	movl	PARAM_SIZE, %ecx
	pushl	%edi		FRAME_pushl()

	movl	PARAM_SRC, %edi
	pushl	%esi		FRAME_pushl()

	movl	PARAM_DIVISOR, %esi
	orl	%ecx,%ecx		C size

	jz	L(size_zero)
	pushl	%ebx		FRAME_pushl()

	movl	-4(%edi,%ecx,4), %eax	C src high limb
	xorl	%edx, %edx

	movl	PARAM_DST, %ebx
	pushl	%ebp		FRAME_pushl()

	movl	PARAM_XSIZE, %ebp
	cmpl	%esi, %eax

	leal	-4(%ebx,%ebp,4), %ebx	C dst one limb below integer part
	jae	L(integer_entry)


	C high<divisor, so high of dst is zero, and avoid one div

	movl	%edx, (%ebx,%ecx,4)
	decl	%ecx

	movl	%eax, %edx
	jz	L(fraction)


L(integer_top):
	C eax	scratch (quotient)
	C ebx	dst+4*xsize-4
	C ecx	counter
	C edx	scratch (remainder)
	C esi	divisor
	C edi	src
	C ebp	xsize

	movl	-4(%edi,%ecx,4), %eax
L(integer_entry):

	divl	%esi

	movl	%eax, (%ebx,%ecx,4)
	loop	L(integer_top)


L(fraction):
	orl	%ebp, %ecx
	jz	L(done)

	movl	PARAM_DST, %ebx


L(fraction_top):
	C eax	scratch (quotient)
	C ebx	dst
	C ecx	counter
	C edx	scratch (remainder)
	C esi	divisor
	C edi
	C ebp

	xorl	%eax, %eax

	divl	%esi

	movl	%eax, -4(%ebx,%ecx,4)
	loop	L(fraction_top)


L(done):
	popl	%ebp
	movl	%edx, %eax
	popl	%ebx
	popl	%esi
	popl	%edi
	ret


L(size_zero):
deflit(`FRAME',8)
	movl	PARAM_XSIZE, %ecx
	xorl	%eax, %eax

	movl	PARAM_DST, %edi

	cld	C better safe than sorry, see mpn/x86/README

	rep
	stosl

	popl	%esi
	popl	%edi
	ret
EPILOGUE()
