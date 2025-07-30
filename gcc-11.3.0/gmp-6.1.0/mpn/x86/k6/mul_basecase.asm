dnl  AMD K6 mpn_mul_basecase -- multiply two mpn numbers.

dnl  Copyright 1999-2003 Free Software Foundation, Inc.

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


C K6: approx 9.0 cycles per cross product on 30x30 limbs (with 16 limbs/loop
C     unrolling).



dnl  K6: UNROLL_COUNT cycles/product (approx)
dnl           8           9.75
dnl          16           9.3
dnl          32           9.3
dnl  Maximum possible with the current code is 32.
dnl
dnl  With 16 the inner unrolled loop fits exactly in a 256 byte block, which
dnl  might explain it's good performance.

deflit(UNROLL_COUNT, 16)


C void mpn_mul_basecase (mp_ptr wp,
C                        mp_srcptr xp, mp_size_t xsize,
C                        mp_srcptr yp, mp_size_t ysize);
C
C Calculate xp,xsize multiplied by yp,ysize, storing the result in
C wp,xsize+ysize.
C
C This routine is essentially the same as mpn/generic/mul_basecase.c, but
C it's faster because it does most of the mpn_addmul_1() entry code only
C once.  The saving is about 10-20% on typical sizes coming from the
C Karatsuba multiply code.
C
C Enhancements:
C
C The mul_1 loop is about 8.5 c/l, which is slower than mpn_mul_1 at 6.25
C c/l.  Could call mpn_mul_1 when ysize is big enough to make it worthwhile.
C
C The main unrolled addmul loop could be shared by mpn_addmul_1, using some
C extra stack setups and maybe 2 or 3 wasted cycles at the end.  Code saving
C would be 256 bytes.

ifdef(`PIC',`
deflit(UNROLL_THRESHOLD, 8)
',`
deflit(UNROLL_THRESHOLD, 8)
')

defframe(PARAM_YSIZE,20)
defframe(PARAM_YP,   16)
defframe(PARAM_XSIZE,12)
defframe(PARAM_XP,   8)
defframe(PARAM_WP,   4)

	TEXT
	ALIGN(32)
PROLOGUE(mpn_mul_basecase)
deflit(`FRAME',0)

	movl	PARAM_XSIZE, %ecx
	movl	PARAM_YP, %eax

	movl	PARAM_XP, %edx
	movl	(%eax), %eax	C yp low limb

	cmpl	$2, %ecx
	ja	L(xsize_more_than_two_limbs)
	je	L(two_by_something)


	C one limb by one limb

	movl	(%edx), %edx	C xp low limb
	movl	PARAM_WP, %ecx

	mull	%edx

	movl	%eax, (%ecx)
	movl	%edx, 4(%ecx)
	ret


C -----------------------------------------------------------------------------
L(two_by_something):
	decl	PARAM_YSIZE
	pushl	%ebx
deflit(`FRAME',4)

	movl	PARAM_WP, %ebx
	pushl	%esi
deflit(`FRAME',8)

	movl	%eax, %ecx	C yp low limb
	movl	(%edx), %eax	C xp low limb

	movl	%edx, %esi	C xp
	jnz	L(two_by_two)


	C two limbs by one limb

	mull	%ecx

	movl	%eax, (%ebx)
	movl	4(%esi), %eax

	movl	%edx, %esi	C carry

	mull	%ecx

	addl	%eax, %esi
	movl	%esi, 4(%ebx)

	adcl	$0, %edx

	movl	%edx, 8(%ebx)
	popl	%esi

	popl	%ebx
	ret



C -----------------------------------------------------------------------------
	ALIGN(16)
L(two_by_two):
	C eax	xp low limb
	C ebx	wp
	C ecx	yp low limb
	C edx
	C esi	xp
	C edi
	C ebp
deflit(`FRAME',8)

	mull	%ecx		C xp[0] * yp[0]

	push	%edi
deflit(`FRAME',12)
	movl	%eax, (%ebx)

	movl	4(%esi), %eax
	movl	%edx, %edi	C carry, for wp[1]

	mull	%ecx		C xp[1] * yp[0]

	addl	%eax, %edi
	movl	PARAM_YP, %ecx

	adcl	$0, %edx

	movl	%edi, 4(%ebx)
	movl	4(%ecx), %ecx	C yp[1]

	movl	4(%esi), %eax	C xp[1]
	movl	%edx, %edi	C carry, for wp[2]

	mull	%ecx		C xp[1] * yp[1]

	addl	%eax, %edi

	adcl	$0, %edx

	movl	(%esi), %eax	C xp[0]
	movl	%edx, %esi	C carry, for wp[3]

	mull	%ecx		C xp[0] * yp[1]

	addl	%eax, 4(%ebx)
	adcl	%edx, %edi
	adcl	$0, %esi

	movl	%edi, 8(%ebx)
	popl	%edi

	movl	%esi, 12(%ebx)
	popl	%esi

	popl	%ebx
	ret


C -----------------------------------------------------------------------------
	ALIGN(16)
L(xsize_more_than_two_limbs):

C The first limb of yp is processed with a simple mpn_mul_1 style loop
C inline.  Unrolling this doesn't seem worthwhile since it's only run once
C (whereas the addmul below is run ysize-1 many times).  A call to the
C actual mpn_mul_1 will be slowed down by the call and parameter pushing and
C popping, and doesn't seem likely to be worthwhile on the typical 10-20
C limb operations the Karatsuba code calls here with.

	C eax	yp[0]
	C ebx
	C ecx	xsize
	C edx	xp
	C esi
	C edi
	C ebp
deflit(`FRAME',0)

	pushl	%edi		defframe_pushl(SAVE_EDI)
	pushl	%ebp		defframe_pushl(SAVE_EBP)

	movl	PARAM_WP, %edi
	pushl	%esi		defframe_pushl(SAVE_ESI)

	movl	%eax, %ebp
	pushl	%ebx		defframe_pushl(SAVE_EBX)

	leal	(%edx,%ecx,4), %ebx	C xp end
	xorl	%esi, %esi

	leal	(%edi,%ecx,4), %edi	C wp end of mul1
	negl	%ecx


L(mul1):
	C eax	scratch
	C ebx	xp end
	C ecx	counter, negative
	C edx	scratch
	C esi	carry
	C edi	wp end of mul1
	C ebp	multiplier

	movl	(%ebx,%ecx,4), %eax

	mull	%ebp

	addl	%esi, %eax
	movl	$0, %esi

	adcl	%edx, %esi

	movl	%eax, (%edi,%ecx,4)
	incl	%ecx

	jnz	L(mul1)


	movl	PARAM_YSIZE, %edx
	movl	%esi, (%edi)		C final carry

	movl	PARAM_XSIZE, %ecx
	decl	%edx

	jnz	L(ysize_more_than_one_limb)

	popl	%ebx
	popl	%esi
	popl	%ebp
	popl	%edi
	ret


L(ysize_more_than_one_limb):
	cmpl	$UNROLL_THRESHOLD, %ecx
	movl	PARAM_YP, %eax

	jae	L(unroll)


C -----------------------------------------------------------------------------
C Simple addmul loop.
C
C Using ebx and edi pointing at the ends of their respective locations saves
C a couple of instructions in the outer loop.  The inner loop is still 11
C cycles, the same as the simple loop in aorsmul_1.asm.

	C eax	yp
	C ebx	xp end
	C ecx	xsize
	C edx	ysize-1
	C esi
	C edi	wp end of mul1
	C ebp

	movl	4(%eax), %ebp		C multiplier
	negl	%ecx

	movl	%ecx, PARAM_XSIZE	C -xsize
	xorl	%esi, %esi		C initial carry

	leal	4(%eax,%edx,4), %eax	C yp end
	negl	%edx

	movl	%eax, PARAM_YP
	movl	%edx, PARAM_YSIZE

	jmp	L(simple_outer_entry)


	C aligning here saves a couple of cycles
	ALIGN(16)
L(simple_outer_top):
	C edx	ysize counter, negative

	movl	PARAM_YP, %eax		C yp end
	xorl	%esi, %esi		C carry

	movl	PARAM_XSIZE, %ecx	C -xsize
	movl	%edx, PARAM_YSIZE

	movl	(%eax,%edx,4), %ebp	C yp limb multiplier
L(simple_outer_entry):
	addl	$4, %edi


L(simple_inner):
	C eax	scratch
	C ebx	xp end
	C ecx	counter, negative
	C edx	scratch
	C esi	carry
	C edi	wp end of this addmul
	C ebp	multiplier

	movl	(%ebx,%ecx,4), %eax

	mull	%ebp

	addl	%esi, %eax
	movl	$0, %esi

	adcl	$0, %edx
	addl	%eax, (%edi,%ecx,4)
	adcl	%edx, %esi

	incl	%ecx
	jnz	L(simple_inner)


	movl	PARAM_YSIZE, %edx
	movl	%esi, (%edi)

	incl	%edx
	jnz	L(simple_outer_top)


	popl	%ebx
	popl	%esi
	popl	%ebp
	popl	%edi
	ret


C -----------------------------------------------------------------------------
C Unrolled loop.
C
C The unrolled inner loop is the same as in aorsmul_1.asm, see that code for
C some comments.
C
C VAR_COUNTER is for the inner loop, running from VAR_COUNTER_INIT down to
C 0, inclusive.
C
C VAR_JMP is the computed jump into the unrolled loop.
C
C PARAM_XP and PARAM_WP get offset appropriately for where the unrolled loop
C is entered.
C
C VAR_XP_LOW is the least significant limb of xp, which is needed at the
C start of the unrolled loop.  This can't just be fetched through the xp
C pointer because of the offset applied to it.
C
C PARAM_YSIZE is the outer loop counter, going from -(ysize-1) up to -1,
C inclusive.
C
C PARAM_YP is offset appropriately so that the PARAM_YSIZE counter can be
C added to give the location of the next limb of yp, which is the multiplier
C in the unrolled loop.
C
C PARAM_WP is similarly offset so that the PARAM_YSIZE counter can be added
C to give the starting point in the destination for each unrolled loop (this
C point is one limb upwards for each limb of yp processed).
C
C Having PARAM_YSIZE count negative to zero means it's not necessary to
C store new values of PARAM_YP and PARAM_WP on each loop.  Those values on
C the stack remain constant and on each loop an leal adjusts them with the
C PARAM_YSIZE counter value.


defframe(VAR_COUNTER,      -20)
defframe(VAR_COUNTER_INIT, -24)
defframe(VAR_JMP,          -28)
defframe(VAR_XP_LOW,       -32)
deflit(VAR_STACK_SPACE, 16)

dnl  For some strange reason using (%esp) instead of 0(%esp) is a touch
dnl  slower in this code, hence the defframe empty-if-zero feature is
dnl  disabled.
dnl
dnl  If VAR_COUNTER is at (%esp), the effect is worse.  In this case the
dnl  unrolled loop is 255 instead of 256 bytes, but quite how this affects
dnl  anything isn't clear.
dnl
define(`defframe_empty_if_zero_disabled',1)

L(unroll):
	C eax	yp (not used)
	C ebx	xp end (not used)
	C ecx	xsize
	C edx	ysize-1
	C esi
	C edi	wp end of mul1 (not used)
	C ebp
deflit(`FRAME', 16)

	leal	-2(%ecx), %ebp	C one limb processed at start,
	decl	%ecx		C and ebp is one less

	shrl	$UNROLL_LOG2, %ebp
	negl	%ecx

	subl	$VAR_STACK_SPACE, %esp
deflit(`FRAME', 16+VAR_STACK_SPACE)
	andl	$UNROLL_MASK, %ecx

	movl	%ecx, %esi
	shll	$4, %ecx

	movl	%ebp, VAR_COUNTER_INIT
	negl	%esi

	C 15 code bytes per limb
ifdef(`PIC',`
	call	L(pic_calc)
L(unroll_here):
',`
	leal	L(unroll_entry) (%ecx,%esi,1), %ecx
')

	movl	PARAM_XP, %ebx
	movl	%ebp, VAR_COUNTER

	movl	PARAM_WP, %edi
	movl	%ecx, VAR_JMP

	movl	(%ebx), %eax
	leal	4(%edi,%esi,4), %edi	C wp adjust for unrolling and mul1

	leal	(%ebx,%esi,4), %ebx	C xp adjust for unrolling

	movl	%eax, VAR_XP_LOW

	movl	%ebx, PARAM_XP
	movl	PARAM_YP, %ebx

	leal	(%edi,%edx,4), %ecx	C wp adjust for ysize indexing
	movl	4(%ebx), %ebp		C multiplier (yp second limb)

	leal	4(%ebx,%edx,4), %ebx	C yp adjust for ysize indexing

	movl	%ecx, PARAM_WP

	leal	1(%esi), %ecx	C adjust parity for decl %ecx above

	movl	%ebx, PARAM_YP
	negl	%edx

	movl	%edx, PARAM_YSIZE
	jmp	L(unroll_outer_entry)


ifdef(`PIC',`
L(pic_calc):
	C See mpn/x86/README about old gas bugs
	leal	(%ecx,%esi,1), %ecx
	addl	$L(unroll_entry)-L(unroll_here), %ecx
	addl	(%esp), %ecx
	ret_internal
')


C -----------------------------------------------------------------------------
	C Aligning here saves a couple of cycles per loop.  Using 32 doesn't
	C cost any extra space, since the inner unrolled loop below is
	C aligned to 32.
	ALIGN(32)
L(unroll_outer_top):
	C edx	ysize

	movl	PARAM_YP, %eax
	movl	%edx, PARAM_YSIZE	C incremented ysize counter

	movl	PARAM_WP, %edi

	movl	VAR_COUNTER_INIT, %ebx
	movl	(%eax,%edx,4), %ebp	C next multiplier

	movl	PARAM_XSIZE, %ecx
	leal	(%edi,%edx,4), %edi	C adjust wp for where we are in yp

	movl	VAR_XP_LOW, %eax
	movl	%ebx, VAR_COUNTER

L(unroll_outer_entry):
	mull	%ebp

	C using testb is a tiny bit faster than testl
	testb	$1, %cl

	movl	%eax, %ecx	C low carry
	movl	VAR_JMP, %eax

	movl	%edx, %esi	C high carry
	movl	PARAM_XP, %ebx

	jnz	L(unroll_noswap)
	movl	%ecx, %esi	C high,low carry other way around

	movl	%edx, %ecx
L(unroll_noswap):

	jmp	*%eax



C -----------------------------------------------------------------------------
	ALIGN(32)
L(unroll_top):
	C eax	scratch
	C ebx	xp
	C ecx	carry low
	C edx	scratch
	C esi	carry high
	C edi	wp
	C ebp	multiplier
	C VAR_COUNTER  loop counter
	C
	C 15 code bytes each limb

	leal	UNROLL_BYTES(%edi), %edi

L(unroll_entry):
deflit(CHUNK_COUNT,2)
forloop(`i', 0, UNROLL_COUNT/CHUNK_COUNT-1, `
	deflit(`disp0', eval(i*CHUNK_COUNT*4))
	deflit(`disp1', eval(disp0 + 4))
	deflit(`disp2', eval(disp1 + 4))

	movl	disp1(%ebx), %eax
	mull	%ebp
Zdisp(	addl,	%ecx, disp0,(%edi))
	adcl	%eax, %esi
	movl	%edx, %ecx
	jadcl0( %ecx)

	movl	disp2(%ebx), %eax
	mull	%ebp
	addl	%esi, disp1(%edi)
	adcl	%eax, %ecx
	movl	%edx, %esi
	jadcl0( %esi)
')

	decl	VAR_COUNTER
	leal	UNROLL_BYTES(%ebx), %ebx

	jns	L(unroll_top)


	movl	PARAM_YSIZE, %edx
	addl	%ecx, UNROLL_BYTES(%edi)

	adcl	$0, %esi

	incl	%edx
	movl	%esi, UNROLL_BYTES+4(%edi)

	jnz	L(unroll_outer_top)


	movl	SAVE_ESI, %esi
	movl	SAVE_EBP, %ebp
	movl	SAVE_EDI, %edi
	movl	SAVE_EBX, %ebx

	addl	$FRAME, %esp
	ret

EPILOGUE()
