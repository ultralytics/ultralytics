dnl  Intel Pentium MMX mpn_mul_1 -- mpn by limb multiplication.

dnl  Copyright 2000-2002 Free Software Foundation, Inc.

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


C    cycles/limb
C P5:   12.0   for 32-bit multiplier
C        7.0   for 16-bit multiplier


C mp_limb_t mpn_mul_1 (mp_ptr dst, mp_srcptr src, mp_size_t size,
C                      mp_limb_t multiplier);
C
C When the multiplier is 16 bits some special case MMX code is used.  Small
C multipliers might arise reasonably often from mpz_mul_ui etc.  If the size
C is odd there's roughly a 5 cycle penalty, so times for say size==7 and
C size==8 end up being quite close.  If src isn't aligned to an 8 byte
C boundary then one limb is processed separately with roughly a 5 cycle
C penalty, so in that case it's say size==8 and size==9 which are close.
C
C Alternatives:
C
C MMX is not believed to be of any use for 32-bit multipliers, since for
C instance the current method would just have to be more or less duplicated
C for the high and low halves of the multiplier, and would probably
C therefore run at about 14 cycles, which is slower than the plain integer
C at 12.
C
C Adding the high and low MMX products using integer code seems best.  An
C attempt at using paddd and carry bit propagation with pcmpgtd didn't give
C any joy.  Perhaps something could be done keeping the values signed and
C thereby avoiding adjustments to make pcmpgtd into an unsigned compare, or
C perhaps not.
C
C Future:
C
C An mpn_mul_1c entrypoint would need a double carry out of the low result
C limb in the 16-bit code, unless it could be assumed the carry fits in 16
C bits, possibly as carry<multiplier, this being true of a big calculation
C done piece by piece.  But let's worry about that if/when mul_1c is
C actually used.

defframe(PARAM_MULTIPLIER,16)
defframe(PARAM_SIZE,      12)
defframe(PARAM_SRC,       8)
defframe(PARAM_DST,       4)

	TEXT

	ALIGN(8)
PROLOGUE(mpn_mul_1)
deflit(`FRAME',0)

	movl	PARAM_SIZE, %ecx
	movl	PARAM_SRC, %edx

	cmpl	$1, %ecx
	jne	L(two_or_more)

	C one limb only

	movl	PARAM_MULTIPLIER, %eax
	movl	PARAM_DST, %ecx

	mull	(%edx)

	movl	%eax, (%ecx)
	movl	%edx, %eax

	ret


L(two_or_more):
	C eax	size
	C ebx
	C ecx	carry
	C edx
	C esi	src
	C edi
	C ebp

	pushl	%esi		FRAME_pushl()
	pushl	%edi		FRAME_pushl()

	movl	%edx, %esi		C src
	movl	PARAM_DST, %edi

	movl	PARAM_MULTIPLIER, %eax
	pushl	%ebx		FRAME_pushl()

	leal	(%esi,%ecx,4), %esi	C src end
	leal	(%edi,%ecx,4), %edi	C dst end

	negl	%ecx			C -size

	pushl	%ebp		FRAME_pushl()
	cmpl	$65536, %eax

	jb	L(small)


L(big):
	xorl	%ebx, %ebx		C carry limb
	sarl	%ecx			C -size/2

	jnc	L(top)			C with carry flag clear


	C size was odd, process one limb separately

	mull	4(%esi,%ecx,8)		C m * src[0]

	movl	%eax, 4(%edi,%ecx,8)
	incl	%ecx

	orl	%edx, %ebx		C carry limb, and clear carry flag


L(top):
	C eax
	C ebx	carry
	C ecx	counter, negative
	C edx
	C esi	src end
	C edi	dst end
	C ebp	(scratch carry)

	adcl	$0, %ebx
	movl	(%esi,%ecx,8), %eax

	mull	PARAM_MULTIPLIER

	movl	%edx, %ebp
	addl	%eax, %ebx

	adcl	$0, %ebp
	movl	4(%esi,%ecx,8), %eax

	mull	PARAM_MULTIPLIER

	movl	%ebx, (%edi,%ecx,8)
	addl	%ebp, %eax

	movl	%eax, 4(%edi,%ecx,8)
	incl	%ecx

	movl	%edx, %ebx
	jnz	L(top)


	adcl	$0, %ebx
	popl	%ebp

	movl	%ebx, %eax
	popl	%ebx

	popl	%edi
	popl	%esi

	ret


L(small):
	C Special case for 16-bit multiplier.
	C
	C eax	multiplier
	C ebx
	C ecx	-size
	C edx	src
	C esi	src end
	C edi	dst end
	C ebp	multiplier

	C size<3 not supported here.  At size==3 we're already a couple of
	C cycles faster, so there's no threshold as such, just use the MMX
	C as soon as possible.

	cmpl	$-3, %ecx
	ja	L(big)

	movd	%eax, %mm7		C m
	pxor	%mm6, %mm6		C initial carry word

	punpcklwd %mm7, %mm7		C m replicated 2 times
	addl	$2, %ecx		C -size+2

	punpckldq %mm7, %mm7		C m replicated 4 times
	andl	$4, %edx		C test alignment, clear carry flag

	movq	%mm7, %mm0		C m
	jz	L(small_entry)


	C Source is unaligned, process one limb separately.
	C
	C Plain integer code is used here, since it's smaller and is about
	C the same 13 cycles as an mmx block would be.
	C
	C An "addl $1,%ecx" doesn't clear the carry flag when size==3, hence
	C the use of separate incl and orl.

	mull	-8(%esi,%ecx,4)		C m * src[0]

	movl	%eax, -8(%edi,%ecx,4)	C dst[0]
	incl	%ecx			C one limb processed

	movd	%edx, %mm6		C initial carry

	orl	%eax, %eax		C clear carry flag
	jmp	L(small_entry)


C The scheduling here is quite tricky, since so many instructions have
C pairing restrictions.  In particular the js won't pair with a movd, and
C can't be paired with an adc since it wants flags from the inc, so
C instructions are rotated to the top of the loop to find somewhere useful
C for it.
C
C Trouble has been taken to avoid overlapping successive loop iterations,
C since that would greatly increase the size of the startup and finishup
C code.  Actually there's probably not much advantage to be had from
C overlapping anyway, since the difficulties are mostly with pairing, not
C with latencies as such.
C
C In the comments x represents the src data and m the multiplier (16
C bits, but replicated 4 times).
C
C The m signs calculated in %mm3 are a loop invariant and could be held in
C say %mm5, but that would save only one instruction and hence be no faster.

L(small_top):
	C eax	l.low, then l.high
	C ebx	(h.low)
	C ecx	counter, -size+2 to 0 or 1
	C edx	(h.high)
	C esi	&src[size]
	C edi	&dst[size]
	C ebp
	C
	C %mm0	(high products)
	C %mm1	(low products)
	C %mm2	(adjust for m using x signs)
	C %mm3	(adjust for x using m signs)
	C %mm4
	C %mm5
	C %mm6	h.low, then carry
	C %mm7	m replicated 4 times

	movd	%mm6, %ebx		C h.low
	psrlq	$32, %mm1		C l.high

	movd	%mm0, %edx		C h.high
	movq	%mm0, %mm6		C new c

	adcl	%eax, %ebx
	incl	%ecx

	movd	%mm1, %eax		C l.high
	movq	%mm7, %mm0

	adcl	%eax, %edx
	movl	%ebx, -16(%edi,%ecx,4)

	movl	%edx, -12(%edi,%ecx,4)
	psrlq	$32, %mm6		C c

L(small_entry):
	pmulhw	-8(%esi,%ecx,4), %mm0	C h = (x*m).high
	movq	%mm7, %mm1

	pmullw	-8(%esi,%ecx,4), %mm1	C l = (x*m).low
	movq	%mm7, %mm3

	movq	-8(%esi,%ecx,4), %mm2	C x
	psraw	$15, %mm3		C m signs

	pand	-8(%esi,%ecx,4), %mm3	C x selected by m signs
	psraw	$15, %mm2		C x signs

	paddw	%mm3, %mm0		C add x to h if m neg
	pand	%mm7, %mm2		C m selected by x signs

	paddw	%mm2, %mm0		C add m to h if x neg
	incl	%ecx

	movd	%mm1, %eax		C l.low
	punpcklwd %mm0, %mm6		C c + h.low << 16

	psrlq	$16, %mm0		C h.high
	js	L(small_top)




	movd	%mm6, %ebx		C h.low
	psrlq	$32, %mm1		C l.high

	adcl	%eax, %ebx
	popl	%ebp		FRAME_popl()

	movd	%mm0, %edx		C h.high
	psrlq	$32, %mm0		C l.high

	movd	%mm1, %eax		C l.high

	adcl	%eax, %edx
	movl	%ebx, -12(%edi,%ecx,4)

	movd	%mm0, %eax		C c

	adcl	$0, %eax
	movl	%edx, -8(%edi,%ecx,4)

	orl	%ecx, %ecx
	jnz	L(small_done)		C final %ecx==1 means even, ==0 odd


	C Size odd, one extra limb to process.
	C Plain integer code is used here, since it's smaller and is about
	C the same speed as another mmx block would be.

	movl	%eax, %ecx
	movl	PARAM_MULTIPLIER, %eax

	mull	-4(%esi)

	addl	%ecx, %eax

	adcl	$0, %edx
	movl	%eax, -4(%edi)

	movl	%edx, %eax
L(small_done):
	popl	%ebx

	popl	%edi
	popl	%esi

	emms

	ret

EPILOGUE()
