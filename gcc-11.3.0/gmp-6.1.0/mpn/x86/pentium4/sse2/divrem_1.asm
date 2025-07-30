dnl  Intel Pentium-4 mpn_divrem_1 -- mpn by limb division.

dnl  Copyright 1999-2004 Free Software Foundation, Inc.

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


C P4: 32 cycles/limb integer part, 30 cycles/limb fraction part.


C mp_limb_t mpn_divrem_1 (mp_ptr dst, mp_size_t xsize,
C                         mp_srcptr src, mp_size_t size,
C                         mp_limb_t divisor);
C mp_limb_t mpn_divrem_1c (mp_ptr dst, mp_size_t xsize,
C                          mp_srcptr src, mp_size_t size,
C                          mp_limb_t divisor, mp_limb_t carry);
C mp_limb_t mpn_preinv_divrem_1 (mp_ptr dst, mp_size_t xsize,
C                                mp_srcptr src, mp_size_t size,
C                                mp_limb_t divisor, mp_limb_t inverse,
C                                unsigned shift);
C
C Algorithm:
C
C The method and nomenclature follow part 8 of "Division by Invariant
C Integers using Multiplication" by Granlund and Montgomery, reference in
C gmp.texi.
C
C "m" is written for what is m' in the paper, and "d" for d_norm, which
C won't cause any confusion since it's only the normalized divisor that's of
C any use in the code.  "b" is written for 2^N, the size of a limb, N being
C 32 here.
C
C The step "sdword dr = n - 2^N*d + (2^N-1-q1) * d" is instead done as
C "n-d - q1*d".  This rearrangement gives the same two-limb answer but lets
C us have just a psubq on the dependent chain.
C
C For reference, the way the k7 code uses "n-(q1+1)*d" would not suit here,
C detecting an overflow of q1+1 when q1=0xFFFFFFFF would cost too much.
C
C Notes:
C
C mpn_divrem_1 and mpn_preinv_divrem_1 avoid one division if the src high
C limb is less than the divisor.  mpn_divrem_1c doesn't check for a zero
C carry, since in normal circumstances that will be a very rare event.
C
C The test for skipping a division is branch free (once size>=1 is tested).
C The store to the destination high limb is 0 when a divide is skipped, or
C if it's not skipped then a copy of the src high limb is stored.  The
C latter is in case src==dst.
C
C There's a small bias towards expecting xsize==0, by having code for
C xsize==0 in a straight line and xsize!=0 under forward jumps.
C
C Enhancements:
C
C The loop measures 32 cycles, but the dependent chain would suggest it
C could be done with 30.  Not sure where to start looking for the extras.
C
C Alternatives:
C
C If the divisor is normalized (high bit set) then a division step can
C always be skipped, since the high destination limb is always 0 or 1 in
C that case.  It doesn't seem worth checking for this though, since it
C probably occurs infrequently.


dnl  MUL_THRESHOLD is the value of xsize+size at which the multiply by
dnl  inverse method is used, rather than plain "divl"s.  Minimum value 1.
dnl
dnl  The inverse takes about 80-90 cycles to calculate, but after that the
dnl  multiply is 32 c/l versus division at about 58 c/l.
dnl
dnl  At 4 limbs the div is a touch faster than the mul (and of course
dnl  simpler), so start the mul from 5 limbs.

deflit(MUL_THRESHOLD, 5)


defframe(PARAM_PREINV_SHIFT,   28)  dnl mpn_preinv_divrem_1
defframe(PARAM_PREINV_INVERSE, 24)  dnl mpn_preinv_divrem_1
defframe(PARAM_CARRY,  24)          dnl mpn_divrem_1c
defframe(PARAM_DIVISOR,20)
defframe(PARAM_SIZE,   16)
defframe(PARAM_SRC,    12)
defframe(PARAM_XSIZE,  8)
defframe(PARAM_DST,    4)

dnl  re-use parameter space
define(SAVE_ESI,`PARAM_SIZE')
define(SAVE_EBP,`PARAM_SRC')
define(SAVE_EDI,`PARAM_DIVISOR')
define(SAVE_EBX,`PARAM_DST')

	TEXT

	ALIGN(16)
PROLOGUE(mpn_preinv_divrem_1)
deflit(`FRAME',0)

	movl	PARAM_SIZE, %ecx
	xorl	%edx, %edx		C carry if can't skip a div

	movl	%esi, SAVE_ESI
	movl	PARAM_SRC, %esi

	movl	%ebp, SAVE_EBP
	movl	PARAM_DIVISOR, %ebp

	movl	%edi, SAVE_EDI
	movl	PARAM_DST, %edi

	movl	-4(%esi,%ecx,4), %eax	C src high limb

	movl	%ebx, SAVE_EBX
	movl	PARAM_XSIZE, %ebx

	movd	PARAM_PREINV_INVERSE, %mm4

	movd	PARAM_PREINV_SHIFT, %mm7  C l
	cmpl	%ebp, %eax		C high cmp divisor

	cmovc(	%eax, %edx)		C high is carry if high<divisor
	movd	%edx, %mm0		C carry

	movd	%edx, %mm1		C carry
	movl	$0, %edx

	movd	%ebp, %mm5		C d
	cmovnc(	%eax, %edx)		C 0 if skip div, src high if not
					C (the latter in case src==dst)
	leal	-4(%edi,%ebx,4), %edi	C &dst[xsize-1]

	movl	%edx, (%edi,%ecx,4)	C dst high limb
	sbbl	$0, %ecx		C skip one division if high<divisor
	movl	$32, %eax

	subl	PARAM_PREINV_SHIFT, %eax
	psllq	%mm7, %mm5		C d normalized
	leal	(%edi,%ecx,4), %edi	C &dst[xsize+size-1]
	leal	-4(%esi,%ecx,4), %esi	C &src[size-1]

	movd	%eax, %mm6		C 32-l
	jmp	L(start_preinv)

EPILOGUE()


	ALIGN(16)
PROLOGUE(mpn_divrem_1c)
deflit(`FRAME',0)

	movl	PARAM_CARRY, %edx

	movl	PARAM_SIZE, %ecx

	movl	%esi, SAVE_ESI
	movl	PARAM_SRC, %esi

	movl	%ebp, SAVE_EBP
	movl	PARAM_DIVISOR, %ebp

	movl	%edi, SAVE_EDI
	movl	PARAM_DST, %edi

	movl	%ebx, SAVE_EBX
	movl	PARAM_XSIZE, %ebx

	leal	-4(%edi,%ebx,4), %edi	C &dst[xsize-1]
	jmp	L(start_1c)

EPILOGUE()


	ALIGN(16)
PROLOGUE(mpn_divrem_1)
deflit(`FRAME',0)

	movl	PARAM_SIZE, %ecx
	xorl	%edx, %edx		C initial carry (if can't skip a div)

	movl	%esi, SAVE_ESI
	movl	PARAM_SRC, %esi

	movl	%ebp, SAVE_EBP
	movl	PARAM_DIVISOR, %ebp

	movl	%edi, SAVE_EDI
	movl	PARAM_DST, %edi

	movl	%ebx, SAVE_EBX
	movl	PARAM_XSIZE, %ebx
	leal	-4(%edi,%ebx,4), %edi	C &dst[xsize-1]

	orl	%ecx, %ecx		C size
	jz	L(no_skip_div)		C if size==0
	movl	-4(%esi,%ecx,4), %eax	C src high limb

	cmpl	%ebp, %eax		C high cmp divisor

	cmovnc(	%eax, %edx)		C 0 if skip div, src high if not
	movl	%edx, (%edi,%ecx,4)	C dst high limb

	movl	$0, %edx
	cmovc(	%eax, %edx)		C high is carry if high<divisor

	sbbl	$0, %ecx		C size-1 if high<divisor
L(no_skip_div):


L(start_1c):
	C eax
	C ebx	xsize
	C ecx	size
	C edx	carry
	C esi	src
	C edi	&dst[xsize-1]
	C ebp	divisor

	leal	(%ebx,%ecx), %eax	C size+xsize
	leal	-4(%esi,%ecx,4), %esi	C &src[size-1]
	leal	(%edi,%ecx,4), %edi	C &dst[size+xsize-1]

	cmpl	$MUL_THRESHOLD, %eax
	jae	L(mul_by_inverse)


	orl	%ecx, %ecx
	jz	L(divide_no_integer)	C if size==0

L(divide_integer):
	C eax	scratch (quotient)
	C ebx	xsize
	C ecx	counter
	C edx	carry
	C esi	src, decrementing
	C edi	dst, decrementing
	C ebp	divisor

	movl	(%esi), %eax
	subl	$4, %esi

	divl	%ebp

	movl	%eax, (%edi)
	subl	$4, %edi

	subl	$1, %ecx
	jnz	L(divide_integer)


L(divide_no_integer):
	orl	%ebx, %ebx
	jnz	L(divide_fraction)	C if xsize!=0

L(divide_done):
	movl	SAVE_ESI, %esi
	movl	SAVE_EDI, %edi
	movl	SAVE_EBX, %ebx
	movl	SAVE_EBP, %ebp
	movl	%edx, %eax
	ret


L(divide_fraction):
	C eax	scratch (quotient)
	C ebx	counter
	C ecx
	C edx	carry
	C esi
	C edi	dst, decrementing
	C ebp	divisor

	movl	$0, %eax

	divl	%ebp

	movl	%eax, (%edi)
	subl	$4, %edi

	subl	$1, %ebx
	jnz	L(divide_fraction)

	jmp	L(divide_done)



C -----------------------------------------------------------------------------

L(mul_by_inverse):
	C eax
	C ebx	xsize
	C ecx	size
	C edx	carry
	C esi	&src[size-1]
	C edi	&dst[size+xsize-1]
	C ebp	divisor

	bsrl	%ebp, %eax		C 31-l
	movd	%edx, %mm0		C carry
	movd	%edx, %mm1		C carry
	movl	%ecx, %edx		C size
	movl	$31, %ecx

	C

	xorl	%eax, %ecx		C l = leading zeros on d
	addl	$1, %eax

	shll	%cl, %ebp		C d normalized
	movd	%ecx, %mm7		C l
	movl	%edx, %ecx		C size

	movd	%eax, %mm6		C 32-l
	movl	$-1, %edx
	movl	$-1, %eax

	C

	subl	%ebp, %edx		C (b-d)-1 so  edx:eax = b*(b-d)-1

	divl	%ebp			C floor (b*(b-d)-1 / d)
	movd	%ebp, %mm5		C d

	C

	movd	%eax, %mm4		C m


L(start_preinv):
	C eax	inverse
	C ebx	xsize
	C ecx	size
	C edx
	C esi	&src[size-1]
	C edi	&dst[size+xsize-1]
	C ebp
	C
	C mm0	carry
	C mm1	carry
	C mm2
	C mm4	m
	C mm5	d
	C mm6	31-l
	C mm7	l

	psllq	%mm7, %mm0		C n2 = carry << l, for size==0

	subl	$1, %ecx
	jb	L(integer_none)

	movd	(%esi), %mm0		C src high limb
	punpckldq %mm1, %mm0
	psrlq	%mm6, %mm0		C n2 = high (carry:srchigh << l)
	jz	L(integer_last)


C The dependent chain here consists of
C
C	2   paddd    n1+n2
C	8   pmuludq  m*(n1+n2)
C	2   paddq    n2:nadj + m*(n1+n2)
C	2   psrlq    q1
C	8   pmuludq  d*q1
C	2   psubq    (n-d)-q1*d
C	2   psrlq    high n-(q1+1)*d mask
C	2   pand     d masked
C	2   paddd    n2+d addback
C	--
C	30
C
C But it seems to run at 32 cycles, so presumably there's something else
C going on.

	ALIGN(16)
L(integer_top):
	C eax
	C ebx
	C ecx	counter, size-1 to 0
	C edx
	C esi	src, decrementing
	C edi	dst, decrementing
	C
	C mm0	n2
	C mm4	m
	C mm5	d
	C mm6	32-l
	C mm7	l

	ASSERT(b,`C n2<d
	 movd	%mm0, %eax
	 movd	%mm5, %edx
	 cmpl	%edx, %eax')

	movd	-4(%esi), %mm1		C next src limbs
	movd	(%esi), %mm2
	leal	-4(%esi), %esi

	punpckldq %mm2, %mm1
	psrlq	%mm6, %mm1		C n10

	movq	%mm1, %mm2		C n10
	movq	%mm1, %mm3		C n10
	psrad	$31, %mm1		C -n1
	pand	%mm5, %mm1		C -n1 & d
	paddd	%mm2, %mm1		C nadj = n10+(-n1&d), ignore overflow

	psrld	$31, %mm2		C n1
	paddd	%mm0, %mm2		C n2+n1
	punpckldq %mm0, %mm1		C n2:nadj

	pmuludq	%mm4, %mm2		C m*(n2+n1)

	C

	paddq	%mm2, %mm1		C n2:nadj + m*(n2+n1)
	pxor	%mm2, %mm2		C break dependency, saves 4 cycles
	pcmpeqd	%mm2, %mm2		C FF...FF
	psrlq	$63, %mm2		C 1

	psrlq	$32, %mm1		C q1 = high(n2:nadj + m*(n2+n1))

	paddd	%mm1, %mm2		C q1+1
	pmuludq	%mm5, %mm1		C q1*d

	punpckldq %mm0, %mm3		C n = n2:n10
	pxor	%mm0, %mm0

	psubq	%mm5, %mm3		C n - d

	C

	psubq	%mm1, %mm3		C n - (q1+1)*d

	por	%mm3, %mm0		C copy remainder -> new n2
	psrlq	$32, %mm3		C high n - (q1+1)*d, 0 or -1

	ASSERT(be,`C 0 or -1
	 movd	%mm3, %eax
	 addl	$1, %eax
	 cmpl	$1, %eax')

	paddd	%mm3, %mm2		C q
	pand	%mm5, %mm3		C mask & d

	paddd	%mm3, %mm0		C addback if necessary
	movd	%mm2, (%edi)
	leal	-4(%edi), %edi

	subl	$1, %ecx
	ja	L(integer_top)


L(integer_last):
	C eax
	C ebx	xsize
	C ecx
	C edx
	C esi	&src[0]
	C edi	&dst[xsize]
	C
	C mm0	n2
	C mm4	m
	C mm5	d
	C mm6
	C mm7	l

	ASSERT(b,`C n2<d
	 movd	%mm0, %eax
	 movd	%mm5, %edx
	 cmpl	%edx, %eax')

	movd	(%esi), %mm1		C src[0]
	psllq	%mm7, %mm1		C n10

	movq	%mm1, %mm2		C n10
	movq	%mm1, %mm3		C n10
	psrad	$31, %mm1		C -n1
	pand	%mm5, %mm1		C -n1 & d
	paddd	%mm2, %mm1		C nadj = n10+(-n1&d), ignore overflow

	psrld	$31, %mm2		C n1
	paddd	%mm0, %mm2		C n2+n1
	punpckldq %mm0, %mm1		C n2:nadj

	pmuludq	%mm4, %mm2		C m*(n2+n1)

	C

	paddq	%mm2, %mm1		C n2:nadj + m*(n2+n1)
	pcmpeqd	%mm2, %mm2		C FF...FF
	psrlq	$63, %mm2		C 1

	psrlq	$32, %mm1		C q1 = high(n2:nadj + m*(n2+n1))
	paddd	%mm1, %mm2		C q1

	pmuludq	%mm5, %mm1		C q1*d
	punpckldq %mm0, %mm3		C n
	psubq	%mm5, %mm3		C n - d
	pxor	%mm0, %mm0

	C

	psubq	%mm1, %mm3		C n - (q1+1)*d

	por	%mm3, %mm0		C remainder -> n2
	psrlq	$32, %mm3		C high n - (q1+1)*d, 0 or -1

	ASSERT(be,`C 0 or -1
	 movd	%mm3, %eax
	 addl	$1, %eax
	 cmpl	$1, %eax')

	paddd	%mm3, %mm2		C q
	pand	%mm5, %mm3		C mask & d

	paddd	%mm3, %mm0		C addback if necessary
	movd	%mm2, (%edi)
	leal	-4(%edi), %edi


L(integer_none):
	C eax
	C ebx	xsize

	orl	%ebx, %ebx
	jnz	L(fraction_some)	C if xsize!=0


L(fraction_done):
	movl	SAVE_EBP, %ebp
	psrld	%mm7, %mm0		C remainder

	movl	SAVE_EDI, %edi
	movd	%mm0, %eax

	movl	SAVE_ESI, %esi
	movl	SAVE_EBX, %ebx
	emms
	ret



C -----------------------------------------------------------------------------
C

L(fraction_some):
	C eax
	C ebx	xsize
	C ecx
	C edx
	C esi
	C edi	&dst[xsize-1]
	C ebp


L(fraction_top):
	C eax
	C ebx	counter, xsize iterations
	C ecx
	C edx
	C esi	src, decrementing
	C edi	dst, decrementing
	C
	C mm0	n2
	C mm4	m
	C mm5	d
	C mm6	32-l
	C mm7	l

	ASSERT(b,`C n2<d
	 movd	%mm0, %eax
	 movd	%mm5, %edx
	 cmpl	%edx, %eax')

	movq	%mm0, %mm1		C n2
	pmuludq	%mm4, %mm0		C m*n2

	pcmpeqd	%mm2, %mm2
	psrlq	$63, %mm2

	C

	psrlq	$32, %mm0		C high(m*n2)

	paddd	%mm1, %mm0		C q1 = high(n2:0 + m*n2)

	paddd	%mm0, %mm2		C q1+1
	pmuludq	%mm5, %mm0		C q1*d

	psllq	$32, %mm1		C n = n2:0
	psubq	%mm5, %mm1		C n - d

	C

	psubq	%mm0, %mm1		C r = n - (q1+1)*d
	pxor	%mm0, %mm0

	por	%mm1, %mm0		C r -> n2
	psrlq	$32, %mm1		C high n - (q1+1)*d, 0 or -1

	ASSERT(be,`C 0 or -1
	 movd	%mm1, %eax
	 addl	$1, %eax
	 cmpl	$1, %eax')

	paddd	%mm1, %mm2		C q
	pand	%mm5, %mm1		C mask & d

	paddd	%mm1, %mm0		C addback if necessary
	movd	%mm2, (%edi)
	leal	-4(%edi), %edi

	subl	$1, %ebx
	jne	L(fraction_top)


	jmp	L(fraction_done)

EPILOGUE()
