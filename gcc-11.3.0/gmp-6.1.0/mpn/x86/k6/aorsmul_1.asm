dnl  AMD K6 mpn_addmul_1/mpn_submul_1 -- add or subtract mpn multiple.

dnl  Copyright 1999-2003, 2005 Free Software Foundation, Inc.

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
C P6 model 0-8,10-12		 5.94
C P6 model 9  (Banias)		 5.51
C P6 model 13 (Dothan)		 5.57
C P4 model 0  (Willamette)
C P4 model 1  (?)
C P4 model 2  (Northwood)
C P4 model 3  (Prescott)
C P4 model 4  (Nocona)
C AMD K6			7.65-8.5 (data dependent)
C AMD K7
C AMD K8


dnl  K6:           large multipliers  small multipliers
dnl  UNROLL_COUNT    cycles/limb       cycles/limb
dnl        4             9.5              7.78
dnl        8             9.0              7.78
dnl       16             8.4              7.65
dnl       32             8.4              8.2
dnl
dnl  Maximum possible unrolling with the current code is 32.
dnl
dnl  Unrolling to 16 limbs/loop makes the unrolled loop fit exactly in a 256
dnl  byte block, which might explain the good speed at that unrolling.

deflit(UNROLL_COUNT, 16)


ifdef(`OPERATION_addmul_1', `
	define(M4_inst,        addl)
	define(M4_function_1,  mpn_addmul_1)
	define(M4_function_1c, mpn_addmul_1c)
',`ifdef(`OPERATION_submul_1', `
	define(M4_inst,        subl)
	define(M4_function_1,  mpn_submul_1)
	define(M4_function_1c, mpn_submul_1c)
',`m4_error(`Need OPERATION_addmul_1 or OPERATION_submul_1
')')')

MULFUNC_PROLOGUE(mpn_addmul_1 mpn_addmul_1c mpn_submul_1 mpn_submul_1c)


C mp_limb_t mpn_addmul_1 (mp_ptr dst, mp_srcptr src, mp_size_t size,
C                         mp_limb_t mult);
C mp_limb_t mpn_addmul_1c (mp_ptr dst, mp_srcptr src, mp_size_t size,
C                          mp_limb_t mult, mp_limb_t carry);
C mp_limb_t mpn_submul_1 (mp_ptr dst, mp_srcptr src, mp_size_t size,
C                         mp_limb_t mult);
C mp_limb_t mpn_submul_1c (mp_ptr dst, mp_srcptr src, mp_size_t size,
C                          mp_limb_t mult, mp_limb_t carry);
C
C The jadcl0()s in the unrolled loop makes the speed data dependent.  Small
C multipliers (most significant few bits clear) result in few carry bits and
C speeds up to 7.65 cycles/limb are attained.  Large multipliers (most
C significant few bits set) make the carry bits 50/50 and lead to something
C more like 8.4 c/l.  With adcl's both of these would be 9.3 c/l.
C
C It's important that the gains for jadcl0 on small multipliers don't come
C at the cost of slowing down other data.  Tests on uniformly distributed
C random data, designed to confound branch prediction, show about a 7%
C speed-up using jadcl0 over adcl (8.93 versus 9.57 cycles/limb, with all
C overheads included).
C
C In the simple loop, jadcl0() measures slower than adcl (11.9-14.7 versus
C 11.0 cycles/limb), and hence isn't used.
C
C In the simple loop, note that running ecx from negative to zero and using
C it as an index in the two movs wouldn't help.  It would save one
C instruction (2*addl+loop becoming incl+jnz), but there's nothing unpaired
C that would be collapsed by this.
C
C Attempts at a simpler main loop, with less unrolling, haven't yielded much
C success, generally running over 9 c/l.
C
C
C jadcl0
C ------
C
C jadcl0() being faster than adcl $0 seems to be an artifact of two things,
C firstly the instruction decoding and secondly the fact that there's a
C carry bit for the jadcl0 only on average about 1/4 of the time.
C
C The code in the unrolled loop decodes something like the following.
C
C                                         decode cycles
C		mull	%ebp                    2
C		M4_inst	%esi, disp(%edi)        1
C		adcl	%eax, %ecx              2
C		movl	%edx, %esi            \ 1
C		jnc	1f                    /
C		incl	%esi                  \ 1
C	1:	movl	disp(%ebx), %eax      /
C                                              ---
C                                               7
C
C In a back-to-back style test this measures 7 with the jnc not taken, or 8
C with it taken (both when correctly predicted).  This is opposite to the
C measurements showing small multipliers running faster than large ones.
C Don't really know why.
C
C It's not clear how much branch misprediction might be costing.  The K6
C doco says it will be 1 to 4 cycles, but presumably it's near the low end
C of that range to get the measured results.
C
C
C In the code the two carries are more or less the preceding mul product and
C the calculation is roughly
C
C	x*y + u*b+v
C
C where b=2^32 is the size of a limb, x*y is the two carry limbs, and u and
C v are the two limbs it's added to (being the low of the next mul, and a
C limb from the destination).
C
C To get a carry requires x*y+u*b+v >= b^2, which is u*b+v >= b^2-x*y, and
C there are b^2-(b^2-x*y) = x*y many such values, giving a probability of
C x*y/b^2.  If x, y, u and v are random and uniformly distributed between 0
C and b-1, then the total probability can be summed over x and y,
C
C	 1    b-1 b-1 x*y    1    b*(b-1)   b*(b-1)
C	--- * sum sum --- = --- * ------- * ------- = 1/4
C       b^2   x=0 y=1 b^2   b^4      2         2
C
C Actually it's a very tiny bit less than 1/4 of course.  If y is fixed,
C then the probability is 1/2*y/b thus varying linearly between 0 and 1/2.


ifdef(`PIC',`
deflit(UNROLL_THRESHOLD, 9)
',`
deflit(UNROLL_THRESHOLD, 6)
')

defframe(PARAM_CARRY,     20)
defframe(PARAM_MULTIPLIER,16)
defframe(PARAM_SIZE,      12)
defframe(PARAM_SRC,       8)
defframe(PARAM_DST,       4)

	TEXT
	ALIGN(32)

PROLOGUE(M4_function_1c)
	pushl	%esi
deflit(`FRAME',4)
	movl	PARAM_CARRY, %esi
	jmp	L(start_nc)
EPILOGUE()

PROLOGUE(M4_function_1)
	push	%esi
deflit(`FRAME',4)
	xorl	%esi, %esi	C initial carry

L(start_nc):
	movl	PARAM_SIZE, %ecx
	pushl	%ebx
deflit(`FRAME',8)

	movl	PARAM_SRC, %ebx
	pushl	%edi
deflit(`FRAME',12)

	cmpl	$UNROLL_THRESHOLD, %ecx
	movl	PARAM_DST, %edi

	pushl	%ebp
deflit(`FRAME',16)
	jae	L(unroll)


	C simple loop

	movl	PARAM_MULTIPLIER, %ebp

L(simple):
	C eax	scratch
	C ebx	src
	C ecx	counter
	C edx	scratch
	C esi	carry
	C edi	dst
	C ebp	multiplier

	movl	(%ebx), %eax
	addl	$4, %ebx

	mull	%ebp

	addl	$4, %edi
	addl	%esi, %eax

	adcl	$0, %edx

	M4_inst	%eax, -4(%edi)

	adcl	$0, %edx

	movl	%edx, %esi
	loop	L(simple)


	popl	%ebp
	popl	%edi

	popl	%ebx
	movl	%esi, %eax

	popl	%esi
	ret



C -----------------------------------------------------------------------------
C The unrolled loop uses a "two carry limbs" scheme.  At the top of the loop
C the carries are ecx=lo, esi=hi, then they swap for each limb processed.
C For the computed jump an odd size means they start one way around, an even
C size the other.
C
C VAR_JUMP holds the computed jump temporarily because there's not enough
C registers at the point of doing the mul for the initial two carry limbs.
C
C The add/adc for the initial carry in %esi is necessary only for the
C mpn_addmul/submul_1c entry points.  Duplicating the startup code to
C eliminate this for the plain mpn_add/submul_1 doesn't seem like a good
C idea.

dnl  overlapping with parameters already fetched
define(VAR_COUNTER, `PARAM_SIZE')
define(VAR_JUMP,    `PARAM_DST')

L(unroll):
	C eax
	C ebx	src
	C ecx	size
	C edx
	C esi	initial carry
	C edi	dst
	C ebp

	movl	%ecx, %edx
	decl	%ecx

	subl	$2, %edx
	negl	%ecx

	shrl	$UNROLL_LOG2, %edx
	andl	$UNROLL_MASK, %ecx

	movl	%edx, VAR_COUNTER
	movl	%ecx, %edx

	shll	$4, %edx
	negl	%ecx

	C 15 code bytes per limb
ifdef(`PIC',`
	call	L(pic_calc)
L(here):
',`
	leal	L(entry) (%edx,%ecx,1), %edx
')
	movl	(%ebx), %eax		C src low limb

	movl	PARAM_MULTIPLIER, %ebp
	movl	%edx, VAR_JUMP

	mull	%ebp

	addl	%esi, %eax	C initial carry (from _1c)
	jadcl0(	%edx)


	leal	4(%ebx,%ecx,4), %ebx
	movl	%edx, %esi	C high carry

	movl	VAR_JUMP, %edx
	leal	(%edi,%ecx,4), %edi

	testl	$1, %ecx
	movl	%eax, %ecx	C low carry

	jz	L(noswap)
	movl	%esi, %ecx	C high,low carry other way around

	movl	%eax, %esi
L(noswap):

	jmp	*%edx


ifdef(`PIC',`
L(pic_calc):
	C See mpn/x86/README about old gas bugs
	leal	(%edx,%ecx,1), %edx
	addl	$L(entry)-L(here), %edx
	addl	(%esp), %edx
	ret_internal
')


C -----------------------------------------------------------
	ALIGN(32)
L(top):
deflit(`FRAME',16)
	C eax	scratch
	C ebx	src
	C ecx	carry lo
	C edx	scratch
	C esi	carry hi
	C edi	dst
	C ebp	multiplier
	C
	C 15 code bytes per limb

	leal	UNROLL_BYTES(%edi), %edi

L(entry):
forloop(`i', 0, UNROLL_COUNT/2-1, `
	deflit(`disp0', eval(2*i*4))
	deflit(`disp1', eval(disp0 + 4))

Zdisp(	movl,	disp0,(%ebx), %eax)
	mull	%ebp
Zdisp(	M4_inst,%ecx, disp0,(%edi))
	adcl	%eax, %esi
	movl	%edx, %ecx
	jadcl0(	%ecx)

	movl	disp1(%ebx), %eax
	mull	%ebp
	M4_inst	%esi, disp1(%edi)
	adcl	%eax, %ecx
	movl	%edx, %esi
	jadcl0(	%esi)
')

	decl	VAR_COUNTER

	leal	UNROLL_BYTES(%ebx), %ebx
	jns	L(top)


	popl	%ebp
	M4_inst	%ecx, UNROLL_BYTES(%edi)

	popl	%edi
	movl	%esi, %eax

	popl	%ebx
	jadcl0(	%eax)

	popl	%esi
	ret

EPILOGUE()
