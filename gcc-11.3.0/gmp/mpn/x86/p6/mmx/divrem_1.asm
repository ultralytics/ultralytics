dnl  Intel Pentium-II mpn_divrem_1 -- mpn by limb division.

dnl  Copyright 1999-2002 Free Software Foundation, Inc.

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


C P6MMX: 25.0 cycles/limb integer part, 17.5 cycles/limb fraction part.


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
C This code is a lightly reworked version of mpn/x86/k7/mmx/divrem_1.asm,
C see that file for some comments.  It's possible what's here can be improved.


dnl  MUL_THRESHOLD is the value of xsize+size at which the multiply by
dnl  inverse method is used, rather than plain "divl"s.  Minimum value 1.
dnl
dnl  The different speeds of the integer and fraction parts means that using
dnl  xsize+size isn't quite right.  The threshold wants to be a bit higher
dnl  for the integer part and a bit lower for the fraction part.  (Or what's
dnl  really wanted is to speed up the integer part!)
dnl
dnl  The threshold is set to make the integer part right.  At 4 limbs the
dnl  div and mul are about the same there, but on the fractional part the
dnl  mul is much faster.

deflit(MUL_THRESHOLD, 4)


defframe(PARAM_PREINV_SHIFT,   28)  dnl mpn_preinv_divrem_1
defframe(PARAM_PREINV_INVERSE, 24)  dnl mpn_preinv_divrem_1
defframe(PARAM_CARRY,  24)          dnl mpn_divrem_1c
defframe(PARAM_DIVISOR,20)
defframe(PARAM_SIZE,   16)
defframe(PARAM_SRC,    12)
defframe(PARAM_XSIZE,  8)
defframe(PARAM_DST,    4)

defframe(SAVE_EBX,    -4)
defframe(SAVE_ESI,    -8)
defframe(SAVE_EDI,    -12)
defframe(SAVE_EBP,    -16)

defframe(VAR_NORM,    -20)
defframe(VAR_INVERSE, -24)
defframe(VAR_SRC,     -28)
defframe(VAR_DST,     -32)
defframe(VAR_DST_STOP,-36)

deflit(STACK_SPACE, 36)

	TEXT
	ALIGN(16)

PROLOGUE(mpn_preinv_divrem_1)
deflit(`FRAME',0)
	movl	PARAM_XSIZE, %ecx
	subl	$STACK_SPACE, %esp	FRAME_subl_esp(STACK_SPACE)

	movl	%esi, SAVE_ESI
	movl	PARAM_SRC, %esi

	movl	%ebx, SAVE_EBX
	movl	PARAM_SIZE, %ebx

	movl	%ebp, SAVE_EBP
	movl	PARAM_DIVISOR, %ebp

	movl	%edi, SAVE_EDI
	movl	PARAM_DST, %edx

	movl	-4(%esi,%ebx,4), %eax	C src high limb
	xorl	%edi, %edi		C initial carry (if can't skip a div)

	C

	leal	8(%edx,%ecx,4), %edx	C &dst[xsize+2]
	xor	%ecx, %ecx

	movl	%edx, VAR_DST_STOP	C &dst[xsize+2]
	cmpl	%ebp, %eax		C high cmp divisor

	cmovc(	%eax, %edi)		C high is carry if high<divisor

	cmovnc(	%eax, %ecx)		C 0 if skip div, src high if not
					C (the latter in case src==dst)

	movl	%ecx, -12(%edx,%ebx,4)	C dst high limb

	sbbl	$0, %ebx		C skip one division if high<divisor
	movl	PARAM_PREINV_SHIFT, %ecx

	leal	-8(%edx,%ebx,4), %edx	C &dst[xsize+size]
	movl	$32, %eax

	movl	%edx, VAR_DST		C &dst[xsize+size]

	shll	%cl, %ebp		C d normalized
	subl	%ecx, %eax
	movl	%ecx, VAR_NORM

	movd	%eax, %mm7		C rshift
	movl	PARAM_PREINV_INVERSE, %eax
	jmp	L(start_preinv)

EPILOGUE()



	ALIGN(16)

PROLOGUE(mpn_divrem_1c)
deflit(`FRAME',0)
	movl	PARAM_CARRY, %edx

	movl	PARAM_SIZE, %ecx
	subl	$STACK_SPACE, %esp
deflit(`FRAME',STACK_SPACE)

	movl	%ebx, SAVE_EBX
	movl	PARAM_XSIZE, %ebx

	movl	%edi, SAVE_EDI
	movl	PARAM_DST, %edi

	movl	%ebp, SAVE_EBP
	movl	PARAM_DIVISOR, %ebp

	movl	%esi, SAVE_ESI
	movl	PARAM_SRC, %esi

	leal	-4(%edi,%ebx,4), %edi
	jmp	L(start_1c)

EPILOGUE()


	C offset 0x31, close enough to aligned
PROLOGUE(mpn_divrem_1)
deflit(`FRAME',0)

	movl	PARAM_SIZE, %ecx
	movl	$0, %edx		C initial carry (if can't skip a div)
	subl	$STACK_SPACE, %esp
deflit(`FRAME',STACK_SPACE)

	movl	%ebp, SAVE_EBP
	movl	PARAM_DIVISOR, %ebp

	movl	%ebx, SAVE_EBX
	movl	PARAM_XSIZE, %ebx

	movl	%esi, SAVE_ESI
	movl	PARAM_SRC, %esi
	orl	%ecx, %ecx		C size

	movl	%edi, SAVE_EDI
	movl	PARAM_DST, %edi

	leal	-4(%edi,%ebx,4), %edi	C &dst[xsize-1]
	jz	L(no_skip_div)		C if size==0

	movl	-4(%esi,%ecx,4), %eax	C src high limb
	xorl	%esi, %esi
	cmpl	%ebp, %eax		C high cmp divisor

	cmovc(	%eax, %edx)		C high is carry if high<divisor

	cmovnc(	%eax, %esi)		C 0 if skip div, src high if not
					C (the latter in case src==dst)

	movl	%esi, (%edi,%ecx,4)	C dst high limb

	sbbl	$0, %ecx		C size-1 if high<divisor
	movl	PARAM_SRC, %esi		C reload
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
	cmpl	$MUL_THRESHOLD, %eax
	jae	L(mul_by_inverse)

	orl	%ecx, %ecx
	jz	L(divide_no_integer)

L(divide_integer):
	C eax	scratch (quotient)
	C ebx	xsize
	C ecx	counter
	C edx	scratch (remainder)
	C esi	src
	C edi	&dst[xsize-1]
	C ebp	divisor

	movl	-4(%esi,%ecx,4), %eax

	divl	%ebp

	movl	%eax, (%edi,%ecx,4)
	decl	%ecx
	jnz	L(divide_integer)


L(divide_no_integer):
	movl	PARAM_DST, %edi
	orl	%ebx, %ebx
	jnz	L(divide_fraction)

L(divide_done):
	movl	SAVE_ESI, %esi

	movl	SAVE_EDI, %edi

	movl	SAVE_EBX, %ebx
	movl	%edx, %eax

	movl	SAVE_EBP, %ebp
	addl	$STACK_SPACE, %esp

	ret


L(divide_fraction):
	C eax	scratch (quotient)
	C ebx	counter
	C ecx
	C edx	scratch (remainder)
	C esi
	C edi	dst
	C ebp	divisor

	movl	$0, %eax

	divl	%ebp

	movl	%eax, -4(%edi,%ebx,4)
	decl	%ebx
	jnz	L(divide_fraction)

	jmp	L(divide_done)



C -----------------------------------------------------------------------------

L(mul_by_inverse):
	C eax
	C ebx	xsize
	C ecx	size
	C edx	carry
	C esi	src
	C edi	&dst[xsize-1]
	C ebp	divisor

	leal	12(%edi), %ebx		C &dst[xsize+2], loop dst stop

	movl	%ebx, VAR_DST_STOP
	leal	4(%edi,%ecx,4), %edi	C &dst[xsize+size]

	movl	%edi, VAR_DST
	movl	%ecx, %ebx		C size

	bsrl	%ebp, %ecx		C 31-l
	movl	%edx, %edi		C carry

	leal	1(%ecx), %eax		C 32-l
	xorl	$31, %ecx		C l

	movl	%ecx, VAR_NORM
	movl	$-1, %edx

	shll	%cl, %ebp		C d normalized
	movd	%eax, %mm7

	movl	$-1, %eax
	subl	%ebp, %edx		C (b-d)-1 giving edx:eax = b*(b-d)-1

	divl	%ebp			C floor (b*(b-d)-1) / d

L(start_preinv):
	C eax	inverse
	C ebx	size
	C ecx	shift
	C edx
	C esi	src
	C edi	carry
	C ebp	divisor
	C
	C mm7	rshift

	movl	%eax, VAR_INVERSE
	orl	%ebx, %ebx		C size
	leal	-12(%esi,%ebx,4), %eax	C &src[size-3]

	movl	%eax, VAR_SRC
	jz	L(start_zero)

	movl	8(%eax), %esi		C src high limb
	cmpl	$1, %ebx
	jz	L(start_one)

L(start_two_or_more):
	movl	4(%eax), %edx		C src second highest limb

	shldl(	%cl, %esi, %edi)	C n2 = carry,high << l

	shldl(	%cl, %edx, %esi)	C n10 = high,second << l

	cmpl	$2, %ebx
	je	L(integer_two_left)
	jmp	L(integer_top)


L(start_one):
	shldl(	%cl, %esi, %edi)	C n2 = carry,high << l

	shll	%cl, %esi		C n10 = high << l
	jmp	L(integer_one_left)


L(start_zero):
	C Can be here with xsize==0 if mpn_preinv_divrem_1 had size==1 and
	C skipped a division.

	shll	%cl, %edi		C n2 = carry << l
	movl	%edi, %eax		C return value for zero_done
	cmpl	$0, PARAM_XSIZE

	je	L(zero_done)
	jmp	L(fraction_some)



C -----------------------------------------------------------------------------
C
C This loop runs at about 25 cycles, which is probably sub-optimal, and
C certainly more than the dependent chain would suggest.  A better loop, or
C a better rough analysis of what's possible, would be welcomed.
C
C In the current implementation, the following successively dependent
C micro-ops seem to exist.
C
C		       uops
C		n2+n1	1   (addl)
C		mul	5
C		q1+1	3   (addl/adcl)
C		mul	5
C		sub	3   (subl/sbbl)
C		addback	2   (cmov)
C		       ---
C		       19
C
C Lack of registers hinders explicit scheduling and it might be that the
C normal out of order execution isn't able to hide enough under the mul
C latencies.
C
C Using sarl/negl to pick out n1 for the n2+n1 stage is a touch faster than
C cmov (and takes one uop off the dependent chain).  A sarl/andl/addl
C combination was tried for the addback (despite the fact it would lengthen
C the dependent chain) but found to be no faster.


	ALIGN(16)
L(integer_top):
	C eax	scratch
	C ebx	scratch (nadj, q1)
	C ecx	scratch (src, dst)
	C edx	scratch
	C esi	n10
	C edi	n2
	C ebp	d
	C
	C mm0	scratch (src qword)
	C mm7	rshift for normalization

	movl	%esi, %eax
	movl	%ebp, %ebx

	sarl	$31, %eax          C -n1
	movl	VAR_SRC, %ecx

	andl	%eax, %ebx         C -n1 & d
	negl	%eax               C n1

	addl	%esi, %ebx         C nadj = n10 + (-n1 & d), ignoring overflow
	addl	%edi, %eax         C n2+n1
	movq	(%ecx), %mm0       C next src limb and the one below it

	mull	VAR_INVERSE        C m*(n2+n1)

	subl	$4, %ecx

	movl	%ecx, VAR_SRC

	C

	C

	addl	%ebx, %eax         C m*(n2+n1) + nadj, low giving carry flag
	movl	%ebp, %eax	   C d
	leal	1(%edi), %ebx      C n2+1

	adcl	%edx, %ebx         C 1 + high(n2<<32 + m*(n2+n1) + nadj) = q1+1
	jz	L(q1_ff)

	mull	%ebx		   C (q1+1)*d

	movl	VAR_DST, %ecx
	psrlq	%mm7, %mm0

	C

	C

	C

	subl	%eax, %esi
	movl	VAR_DST_STOP, %eax

	sbbl	%edx, %edi	   C n - (q1+1)*d
	movl	%esi, %edi	   C remainder -> n2
	leal	(%ebp,%esi), %edx

	cmovc(	%edx, %edi)	   C n - q1*d if underflow from using q1+1
	movd	%mm0, %esi

	sbbl	$0, %ebx	   C q
	subl	$4, %ecx

	movl	%ebx, (%ecx)
	cmpl	%eax, %ecx

	movl	%ecx, VAR_DST
	jne	L(integer_top)


L(integer_loop_done):


C -----------------------------------------------------------------------------
C
C Here, and in integer_one_left below, an sbbl $0 is used rather than a jz
C q1_ff special case.  This make the code a bit smaller and simpler, and
C costs only 2 cycles (each).

L(integer_two_left):
	C eax	scratch
	C ebx	scratch (nadj, q1)
	C ecx	scratch (src, dst)
	C edx	scratch
	C esi	n10
	C edi	n2
	C ebp	divisor
	C
	C mm7	rshift


	movl	%esi, %eax
	movl	%ebp, %ebx

	sarl	$31, %eax          C -n1
	movl	PARAM_SRC, %ecx

	andl	%eax, %ebx         C -n1 & d
	negl	%eax               C n1

	addl	%esi, %ebx         C nadj = n10 + (-n1 & d), ignoring overflow
	addl	%edi, %eax         C n2+n1

	mull	VAR_INVERSE        C m*(n2+n1)

	movd	(%ecx), %mm0	   C src low limb

	movl	VAR_DST_STOP, %ecx

	C

	C

	addl	%ebx, %eax         C m*(n2+n1) + nadj, low giving carry flag
	leal	1(%edi), %ebx      C n2+1
	movl	%ebp, %eax	   C d

	adcl	%edx, %ebx         C 1 + high(n2<<32 + m*(n2+n1) + nadj) = q1+1

	sbbl	$0, %ebx

	mull	%ebx		   C (q1+1)*d

	psllq	$32, %mm0

	psrlq	%mm7, %mm0

	C

	C

	subl	%eax, %esi

	sbbl	%edx, %edi	   C n - (q1+1)*d
	movl	%esi, %edi	   C remainder -> n2
	leal	(%ebp,%esi), %edx

	cmovc(	%edx, %edi)	   C n - q1*d if underflow from using q1+1
	movd	%mm0, %esi

	sbbl	$0, %ebx	   C q

	movl	%ebx, -4(%ecx)


C -----------------------------------------------------------------------------
L(integer_one_left):
	C eax	scratch
	C ebx	scratch (nadj, q1)
	C ecx	scratch (dst)
	C edx	scratch
	C esi	n10
	C edi	n2
	C ebp	divisor
	C
	C mm7	rshift


	movl	%esi, %eax
	movl	%ebp, %ebx

	sarl	$31, %eax          C -n1
	movl	VAR_DST_STOP, %ecx

	andl	%eax, %ebx         C -n1 & d
	negl	%eax               C n1

	addl	%esi, %ebx         C nadj = n10 + (-n1 & d), ignoring overflow
	addl	%edi, %eax         C n2+n1

	mull	VAR_INVERSE        C m*(n2+n1)

	C

	C

	C

	addl	%ebx, %eax         C m*(n2+n1) + nadj, low giving carry flag
	leal	1(%edi), %ebx      C n2+1
	movl	%ebp, %eax	   C d

	C

	adcl	%edx, %ebx         C 1 + high(n2<<32 + m*(n2+n1) + nadj) = q1+1

	sbbl	$0, %ebx           C q1 if q1+1 overflowed

	mull	%ebx

	C

	C

	C

	C

	subl	%eax, %esi
	movl	PARAM_XSIZE, %eax

	sbbl	%edx, %edi	   C n - (q1+1)*d
	movl	%esi, %edi	   C remainder -> n2
	leal	(%ebp,%esi), %edx

	cmovc(	%edx, %edi)	   C n - q1*d if underflow from using q1+1

	sbbl	$0, %ebx	   C q

	movl	%ebx, -8(%ecx)
	subl	$8, %ecx



	orl	%eax, %eax         C xsize
	jnz	L(fraction_some)

	movl	%edi, %eax
L(fraction_done):
	movl	VAR_NORM, %ecx
L(zero_done):
	movl	SAVE_EBP, %ebp

	movl	SAVE_EDI, %edi

	movl	SAVE_ESI, %esi

	movl	SAVE_EBX, %ebx
	addl	$STACK_SPACE, %esp

	shrl	%cl, %eax
	emms

	ret


C -----------------------------------------------------------------------------
C
C Special case for q1=0xFFFFFFFF, giving q=0xFFFFFFFF meaning the low dword
C of q*d is simply -d and the remainder n-q*d = n10+d

L(q1_ff):
	C eax	(divisor)
	C ebx	(q1+1 == 0)
	C ecx
	C edx
	C esi	n10
	C edi	n2
	C ebp	divisor

	movl	VAR_DST, %ecx
	movl	VAR_DST_STOP, %edx
	subl	$4, %ecx

	movl	%ecx, VAR_DST
	psrlq	%mm7, %mm0
	leal	(%ebp,%esi), %edi	C n-q*d remainder -> next n2

	movl	$-1, (%ecx)
	movd	%mm0, %esi		C next n10

	cmpl	%ecx, %edx
	jne	L(integer_top)

	jmp	L(integer_loop_done)



C -----------------------------------------------------------------------------
C
C In the current implementation, the following successively dependent
C micro-ops seem to exist.
C
C		       uops
C		mul	5
C		q1+1	1   (addl)
C		mul	5
C		sub	3   (negl/sbbl)
C		addback	2   (cmov)
C		       ---
C		       16
C
C The loop in fact runs at about 17.5 cycles.  Using a sarl/andl/addl for
C the addback was found to be a touch slower.


	ALIGN(16)
L(fraction_some):
	C eax
	C ebx
	C ecx
	C edx
	C esi
	C edi	carry
	C ebp	divisor

	movl	PARAM_DST, %esi
	movl	VAR_DST_STOP, %ecx	C &dst[xsize+2]
	movl	%edi, %eax

	subl	$8, %ecx		C &dst[xsize]


	ALIGN(16)
L(fraction_top):
	C eax	n2, then scratch
	C ebx	scratch (nadj, q1)
	C ecx	dst, decrementing
	C edx	scratch
	C esi	dst stop point
	C edi	n2
	C ebp	divisor

	mull	VAR_INVERSE	C m*n2

	movl	%ebp, %eax	C d
	subl	$4, %ecx	C dst
	leal	1(%edi), %ebx

	C

	C

	C

	addl	%edx, %ebx	C 1 + high(n2<<32 + m*n2) = q1+1

	mull	%ebx		C (q1+1)*d

	C

	C

	C

	C

	negl	%eax		C low of n - (q1+1)*d

	sbbl	%edx, %edi	C high of n - (q1+1)*d, caring only about carry
	leal	(%ebp,%eax), %edx

	cmovc(	%edx, %eax)	C n - q1*d if underflow from using q1+1

	sbbl	$0, %ebx	C q
	movl	%eax, %edi	C remainder->n2
	cmpl	%esi, %ecx

	movl	%ebx, (%ecx)	C previous q
	jne	L(fraction_top)


	jmp	L(fraction_done)

EPILOGUE()
