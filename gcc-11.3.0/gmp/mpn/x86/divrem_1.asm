dnl  x86 mpn_divrem_1 -- mpn by limb division extending to fractional quotient.

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


C       cycles/limb
C 486   approx 43 maybe
C P5        44
C P6        39
C P6MMX     39
C K6        22
C K7        42
C P4        58


C mp_limb_t mpn_divrem_1 (mp_ptr dst, mp_size_t xsize,
C                         mp_srcptr src, mp_size_t size, mp_limb_t divisor);
C mp_limb_t mpn_divrem_1c (mp_ptr dst, mp_size_t xsize,
C                          mp_srcptr src, mp_size_t size, mp_limb_t divisor,
C                          mp_limb_t carry);
C
C Divide src,size by divisor and store the quotient in dst+xsize,size.
C Extend the division to fractional quotient limbs in dst,xsize.  Return the
C remainder.  Either or both xsize and size can be 0.
C
C mpn_divrem_1c takes a carry parameter which is an initial high limb,
C effectively one extra limb at the top of src,size.  Must have
C carry<divisor.
C
C
C Essentially the code is the same as the division based part of
C mpn/generic/divrem_1.c, but has the advantage that we get the desired divl
C instruction even when gcc is not being used (when longlong.h only has the
C rather slow generic C udiv_qrnnd().
C
C A test is done to see if the high limb is less than the divisor, and if so
C one less div is done.  A div is between 20 and 40 cycles on the various
C x86s, so assuming high<divisor about half the time, then this test saves
C half that amount.  The branch misprediction penalty on each chip is less
C than half a div.
C
C
C Notes for P5:
C
C It might be thought that moving the load down to pair with the store would
C save 1 cycle, but that doesn't seem to happen in practice, and in any case
C would be a mere 2.2% saving, so it's hardly worth bothering about.
C
C A mul-by-inverse might be a possibility for P5, as done in
C mpn/x86/pentium/mod_1.asm.  The number of auxiliary instructions required
C is a hinderance, but there could be a 10-15% speedup available.
C
C
C Notes for K6:
C
C K6 has its own version of this code, using loop and paying attention to
C cache line boundary crossings.  The target 20 c/l can be had with the
C decl+jnz of the present code by pairing up the load and store in the
C loops.  But it's considered easier not to introduce complexity just for
C that, but instead let k6 have its own code.
C

defframe(PARAM_CARRY,  24)
defframe(PARAM_DIVISOR,20)
defframe(PARAM_SIZE,   16)
defframe(PARAM_SRC,    12)
defframe(PARAM_XSIZE,  8)
defframe(PARAM_DST,    4)

	TEXT
	ALIGN(16)

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
	orl	%ecx, %ecx

	movl	PARAM_CARRY, %edx
	jz	L(fraction)

	leal	-4(%ebx,%ebp,4), %ebx	C dst one limb below integer part
	jmp	L(integer_top)

EPILOGUE()


PROLOGUE(mpn_divrem_1)
deflit(`FRAME',0)

	movl	PARAM_SIZE, %ecx
	pushl	%edi		FRAME_pushl()

	movl	PARAM_SRC, %edi
	pushl	%esi		FRAME_pushl()

	movl	PARAM_DIVISOR, %esi
	orl	%ecx,%ecx

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
	decl	%ecx
	jnz	L(integer_top)


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
	decl	%ecx
	jnz	L(fraction_top)


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
