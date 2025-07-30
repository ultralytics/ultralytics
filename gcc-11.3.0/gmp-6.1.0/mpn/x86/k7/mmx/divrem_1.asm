dnl  AMD K7 mpn_divrem_1, mpn_divrem_1c, mpn_preinv_divrem_1 -- mpn by limb
dnl  division.

dnl  Copyright 1999-2002, 2004 Free Software Foundation, Inc.

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


C K7: 17.0 cycles/limb integer part, 15.0 cycles/limb fraction part.


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
C The "and"s shown in the paper are done here with "cmov"s.  "m" is written
C for m', and "d" for d_norm, which won't cause any confusion since it's
C only the normalized divisor that's of any use in the code.  "b" is written
C for 2^N, the size of a limb, N being 32 here.
C
C The step "sdword dr = n - 2^N*d + (2^N-1-q1) * d" is instead done as
C "n-(q1+1)*d"; this rearrangement gives the same two-limb answer.  If
C q1==0xFFFFFFFF, then q1+1 would overflow.  We branch to a special case
C "q1_ff" if this occurs.  Since the true quotient is either q1 or q1+1 then
C if q1==0xFFFFFFFF that must be the right value.
C
C For the last and second last steps q1==0xFFFFFFFF is instead handled by an
C sbbl to go back to 0xFFFFFFFF if an overflow occurs when adding 1.  This
C then goes through as normal, and finding no addback required.  sbbl costs
C an extra cycle over what the main loop code does, but it keeps code size
C and complexity down.
C
C Notes:
C
C mpn_divrem_1 and mpn_preinv_divrem_1 avoid one division if the src high
C limb is less than the divisor.  mpn_divrem_1c doesn't check for a zero
C carry, since in normal circumstances that will be a very rare event.
C
C The test for skipping a division is branch free (once size>=1 is tested).
C The store to the destination high limb is 0 when a divide is skipped, or
C if it's not skipped then a copy of the src high limb is used.  The latter
C is in case src==dst.
C
C There's a small bias towards expecting xsize==0, by having code for
C xsize==0 in a straight line and xsize!=0 under forward jumps.
C
C Alternatives:
C
C If the divisor is normalized (high bit set) then a division step can
C always be skipped, since the high destination limb is always 0 or 1 in
C that case.  It doesn't seem worth checking for this though, since it
C probably occurs infrequently, in particular note that big_base for a
C decimal mpn_get_str is not normalized in a 32-bit limb.


dnl  MUL_THRESHOLD is the value of xsize+size at which the multiply by
dnl  inverse method is used, rather than plain "divl"s.  Minimum value 1.
dnl
dnl  The inverse takes about 50 cycles to calculate, but after that the
dnl  multiply is 17 c/l versus division at 42 c/l.
dnl
dnl  At 3 limbs the mul is a touch faster than div on the integer part, and
dnl  even more so on the fractional part.

deflit(MUL_THRESHOLD, 3)


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
	ALIGN(32)

PROLOGUE(mpn_preinv_divrem_1)
deflit(`FRAME',0)
	movl	PARAM_XSIZE, %ecx
	movl	PARAM_DST, %edx
	subl	$STACK_SPACE, %esp	FRAME_subl_esp(STACK_SPACE)

	movl	%esi, SAVE_ESI
	movl	PARAM_SRC, %esi

	movl	%ebx, SAVE_EBX
	movl	PARAM_SIZE, %ebx

	leal	8(%edx,%ecx,4), %edx	C &dst[xsize+2]
	movl	%ebp, SAVE_EBP
	movl	PARAM_DIVISOR, %ebp

	movl	%edx, VAR_DST_STOP	C &dst[xsize+2]
	movl	%edi, SAVE_EDI
	xorl	%edi, %edi		C carry

	movl	-4(%esi,%ebx,4), %eax	C src high limb
	xor	%ecx, %ecx

	C

	C

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

	leal	-4(%edi,%ebx,4), %edi	C &dst[xsize-1]
	jmp	L(start_1c)

EPILOGUE()


	C offset 0xa1, close enough to aligned
PROLOGUE(mpn_divrem_1)
deflit(`FRAME',0)

	movl	PARAM_SIZE, %ecx
	movl	$0, %edx		C initial carry (if can't skip a div)
	subl	$STACK_SPACE, %esp
deflit(`FRAME',STACK_SPACE)

	movl	%esi, SAVE_ESI
	movl	PARAM_SRC, %esi

	movl	%ebx, SAVE_EBX
	movl	PARAM_XSIZE, %ebx

	movl	%ebp, SAVE_EBP
	movl	PARAM_DIVISOR, %ebp
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


C With MUL_THRESHOLD set to 3, the simple loops here only do 0 to 2 limbs.
C It'd be possible to write them out without the looping, but no speedup
C would be expected.
C
C Using PARAM_DIVISOR instead of %ebp measures 1 cycle/loop faster on the
C integer part, but curiously not on the fractional part, where %ebp is a
C (fixed) couple of cycles faster.

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

	divl	PARAM_DIVISOR

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
	movl	%edx, %eax

	movl	SAVE_EBX, %ebx
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

	bsrl	%ebp, %eax		C 31-l

	leal	12(%edi), %ebx		C &dst[xsize+2], loop dst stop
	leal	4(%edi,%ecx,4), %edi	C &dst[xsize+size]

	movl	%edi, VAR_DST
	movl	%ebx, VAR_DST_STOP

	movl	%ecx, %ebx		C size
	movl	$31, %ecx

	movl	%edx, %edi		C carry
	movl	$-1, %edx

	C

	xorl	%eax, %ecx		C l
	incl	%eax			C 32-l

	shll	%cl, %ebp		C d normalized
	movl	%ecx, VAR_NORM

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

	orl	%ebx, %ebx		C size
	movl	%eax, VAR_INVERSE
	leal	-12(%esi,%ebx,4), %eax	C &src[size-3]

	jz	L(start_zero)
	movl	%eax, VAR_SRC
	cmpl	$1, %ebx

	movl	8(%eax), %esi		C src high limb
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
	movl	%eax, VAR_SRC
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
C The multiply by inverse loop is 17 cycles, and relies on some out-of-order
C execution.  The instruction scheduling is important, with various
C apparently equivalent forms running 1 to 5 cycles slower.
C
C A lower bound for the time would seem to be 16 cycles, based on the
C following successive dependencies.
C
C		      cycles
C		n2+n1	1
C		mul	6
C		q1+1	1
C		mul	6
C		sub	1
C		addback	1
C		       ---
C		       16
C
C This chain is what the loop has already, but 16 cycles isn't achieved.
C K7 has enough decode, and probably enough execute (depending maybe on what
C a mul actually consumes), but nothing running under 17 has been found.
C
C In theory n2+n1 could be done in the sub and addback stages (by
C calculating both n2 and n2+n1 there), but lack of registers makes this an
C unlikely proposition.
C
C The jz in the loop keeps the q1+1 stage to 1 cycle.  Handling an overflow
C from q1+1 with an "sbbl $0, %ebx" would add a cycle to the dependent
C chain, and nothing better than 18 cycles has been found when using it.
C The jump is taken only when q1 is 0xFFFFFFFF, and on random data this will
C be an extremely rare event.
C
C Branch mispredictions will hit random occurrences of q1==0xFFFFFFFF, but
C if some special data is coming out with this always, the q1_ff special
C case actually runs at 15 c/l.  0x2FFF...FFFD divided by 3 is a good way to
C induce the q1_ff case, for speed measurements or testing.  Note that
C 0xFFF...FFF divided by 1 or 2 doesn't induce it.
C
C The instruction groupings and empty comments show the cycles for a naive
C in-order view of the code (conveniently ignoring the load latency on
C VAR_INVERSE).  This shows some of where the time is going, but is nonsense
C to the extent that out-of-order execution rearranges it.  In this case
C there's 19 cycles shown, but it executes at 17.

	ALIGN(16)
L(integer_top):
	C eax	scratch
	C ebx	scratch (nadj, q1)
	C ecx	scratch (src, dst)
	C edx	scratch
	C esi	n10
	C edi	n2
	C ebp	divisor
	C
	C mm0	scratch (src qword)
	C mm7	rshift for normalization

	cmpl	$0x80000000, %esi  C n1 as 0=c, 1=nc
	movl	%edi, %eax         C n2
	movl	VAR_SRC, %ecx

	leal	(%ebp,%esi), %ebx
	cmovc(	%esi, %ebx)	   C nadj = n10 + (-n1 & d), ignoring overflow
	sbbl	$-1, %eax          C n2+n1

	mull	VAR_INVERSE        C m*(n2+n1)

	movq	(%ecx), %mm0       C next limb and the one below it
	subl	$4, %ecx

	movl	%ecx, VAR_SRC

	C

	addl	%ebx, %eax         C m*(n2+n1) + nadj, low giving carry flag
	leal	1(%edi), %ebx      C n2+1
	movl	%ebp, %eax	   C d

	C

	adcl	%edx, %ebx         C 1 + high(n2<<32 + m*(n2+n1) + nadj) = q1+1
	jz	L(q1_ff)
	movl	VAR_DST, %ecx

	mull	%ebx		   C (q1+1)*d

	psrlq	%mm7, %mm0

	leal	-4(%ecx), %ecx

	C

	subl	%eax, %esi
	movl	VAR_DST_STOP, %eax

	C

	sbbl	%edx, %edi	   C n - (q1+1)*d
	movl	%esi, %edi	   C remainder -> n2
	leal	(%ebp,%esi), %edx

	movd	%mm0, %esi

	cmovc(	%edx, %edi)	   C n - q1*d if underflow from using q1+1
	sbbl	$0, %ebx	   C q
	cmpl	%eax, %ecx

	movl	%ebx, (%ecx)
	movl	%ecx, VAR_DST
	jne	L(integer_top)


L(integer_loop_done):


C -----------------------------------------------------------------------------
C
C Here, and in integer_one_left below, an sbbl $0 is used rather than a jz
C q1_ff special case.  This make the code a bit smaller and simpler, and
C costs only 1 cycle (each).

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

	cmpl	$0x80000000, %esi  C n1 as 0=c, 1=nc
	movl	%edi, %eax         C n2
	movl	PARAM_SRC, %ecx

	leal	(%ebp,%esi), %ebx
	cmovc(	%esi, %ebx)	   C nadj = n10 + (-n1 & d), ignoring overflow
	sbbl	$-1, %eax          C n2+n1

	mull	VAR_INVERSE        C m*(n2+n1)

	movd	(%ecx), %mm0	   C src low limb

	movl	VAR_DST_STOP, %ecx

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

	subl	%eax, %esi

	C

	sbbl	%edx, %edi	   C n - (q1+1)*d
	movl	%esi, %edi	   C remainder -> n2
	leal	(%ebp,%esi), %edx

	movd	%mm0, %esi

	cmovc(	%edx, %edi)	   C n - q1*d if underflow from using q1+1
	sbbl	$0, %ebx	   C q

	movl	%ebx, -4(%ecx)


C -----------------------------------------------------------------------------
L(integer_one_left):
	C eax	scratch
	C ebx	scratch (nadj, q1)
	C ecx	dst
	C edx	scratch
	C esi	n10
	C edi	n2
	C ebp	divisor
	C
	C mm7	rshift

	movl	VAR_DST_STOP, %ecx
	cmpl	$0x80000000, %esi  C n1 as 0=c, 1=nc
	movl	%edi, %eax         C n2

	leal	(%ebp,%esi), %ebx
	cmovc(	%esi, %ebx)	   C nadj = n10 + (-n1 & d), ignoring overflow
	sbbl	$-1, %eax          C n2+n1

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

	subl	%eax, %esi

	C

	sbbl	%edx, %edi	   C n - (q1+1)*d
	movl	%esi, %edi	   C remainder -> n2
	leal	(%ebp,%esi), %edx

	cmovc(	%edx, %edi)	   C n - q1*d if underflow from using q1+1
	sbbl	$0, %ebx	   C q

	movl	%ebx, -8(%ecx)
	subl	$8, %ecx



L(integer_none):
	cmpl	$0, PARAM_XSIZE
	jne	L(fraction_some)

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

	psrlq	%mm7, %mm0
	leal	(%ebp,%esi), %edi	C n-q*d remainder -> next n2
	movl	%ecx, VAR_DST

	movd	%mm0, %esi		C next n10

	movl	$-1, (%ecx)
	cmpl	%ecx, %edx
	jne	L(integer_top)

	jmp	L(integer_loop_done)



C -----------------------------------------------------------------------------
C
C Being the fractional part, the "source" limbs are all zero, meaning
C n10=0, n1=0, and hence nadj=0, leading to many instructions eliminated.
C
C The loop runs at 15 cycles.  The dependent chain is the same as the
C general case above, but without the n2+n1 stage (due to n1==0), so 15
C would seem to be the lower bound.
C
C A not entirely obvious simplification is that q1+1 never overflows a limb,
C and so there's no need for the sbbl $0 or jz q1_ff from the general case.
C q1 is the high word of m*n2+b*n2 and the following shows q1<=b-2 always.
C rnd() means rounding down to a multiple of d.
C
C	m*n2 + b*n2 <= m*(d-1) + b*(d-1)
C		     = m*d + b*d - m - b
C		     = floor((b(b-d)-1)/d)*d + b*d - m - b
C		     = rnd(b(b-d)-1) + b*d - m - b
C		     = rnd(b(b-d)-1 + b*d) - m - b
C		     = rnd(b*b-1) - m - b
C		     <= (b-2)*b
C
C Unchanged from the general case is that the final quotient limb q can be
C either q1 or q1+1, and the q1+1 case occurs often.  This can be seen from
C equation 8.4 of the paper which simplifies as follows when n1==0 and
C n0==0.
C
C	n-q1*d = (n2*k+q0*d)/b <= d + (d*d-2d)/b
C
C As before, the instruction groupings and empty comments show a naive
C in-order view of the code, which is made a nonsense by out of order
C execution.  There's 17 cycles shown, but it executes at 15.
C
C Rotating the store q and remainder->n2 instructions up to the top of the
C loop gets the run time down from 16 to 15.

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
	jmp	L(fraction_entry)


	ALIGN(16)
L(fraction_top):
	C eax	n2 carry, then scratch
	C ebx	scratch (nadj, q1)
	C ecx	dst, decrementing
	C edx	scratch
	C esi	dst stop point
	C edi	(will be n2)
	C ebp	divisor

	movl	%ebx, (%ecx)	C previous q
	movl	%eax, %edi	C remainder->n2

L(fraction_entry):
	mull	VAR_INVERSE	C m*n2

	movl	%ebp, %eax	C d
	subl	$4, %ecx	C dst
	leal	1(%edi), %ebx

	C

	C

	C

	C

	addl	%edx, %ebx	C 1 + high(n2<<32 + m*n2) = q1+1

	mull	%ebx		C (q1+1)*d

	C

	C

	C

	negl	%eax		C low of n - (q1+1)*d

	C

	sbbl	%edx, %edi	C high of n - (q1+1)*d, caring only about carry
	leal	(%ebp,%eax), %edx

	cmovc(	%edx, %eax)	C n - q1*d if underflow from using q1+1
	sbbl	$0, %ebx	C q
	cmpl	%esi, %ecx

	jne	L(fraction_top)


	movl	%ebx, (%ecx)
	jmp	L(fraction_done)

EPILOGUE()
