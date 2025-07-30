dnl  AMD K6 mpn_modexact_1_odd -- exact division style remainder.

dnl  Copyright 2000-2003, 2007 Free Software Foundation, Inc.

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


C K6: 10.0 cycles/limb


C mp_limb_t mpn_modexact_1_odd (mp_srcptr src, mp_size_t size,
C                               mp_limb_t divisor);
C mp_limb_t mpn_modexact_1c_odd (mp_srcptr src, mp_size_t size,
C                                mp_limb_t divisor, mp_limb_t carry);
C
C A special case for high<divisor at the end measured only about 4 cycles
C faster, and so isn't used.
C
C A special case for size==1 using a divl rather than the inverse measured
C only about 5 cycles faster, and so isn't used.  When size==1 and
C high<divisor it can skip a division and be a full 24 cycles faster, but
C this isn't an important case.

defframe(PARAM_CARRY,  16)
defframe(PARAM_DIVISOR,12)
defframe(PARAM_SIZE,   8)
defframe(PARAM_SRC,    4)

	TEXT

	ALIGN(32)
PROLOGUE(mpn_modexact_1c_odd)
deflit(`FRAME',0)

	movl	PARAM_DIVISOR, %ecx
	pushl	%esi		FRAME_pushl()

	movl	PARAM_CARRY, %edx
	jmp	L(start_1c)

EPILOGUE()


	ALIGN(16)
PROLOGUE(mpn_modexact_1_odd)
deflit(`FRAME',0)

	movl	PARAM_DIVISOR, %ecx
	pushl	%esi		FRAME_pushl()

	xorl	%edx, %edx
L(start_1c):
	pushl	%edi		FRAME_pushl()

	shrl	%ecx			C d/2
	movl	PARAM_DIVISOR, %esi

	andl	$127, %ecx		C d/2, 7 bits
	pushl	%ebp		FRAME_pushl()

ifdef(`PIC',`
	LEA(	binvert_limb_table, %edi)
Zdisp(	movzbl,	0,(%ecx,%edi), %edi)		C inv 8 bits
',`
	movzbl	binvert_limb_table(%ecx), %edi	C inv 8 bits
')
	leal	(%edi,%edi), %ecx	C 2*inv

	imull	%edi, %edi		C inv*inv

	movl	PARAM_SRC, %eax
	movl	PARAM_SIZE, %ebp

	imull	%esi, %edi		C inv*inv*d

	pushl	%ebx		FRAME_pushl()
	leal	(%eax,%ebp,4), %ebx	C src end

	subl	%edi, %ecx		C inv = 2*inv - inv*inv*d
	leal	(%ecx,%ecx), %edi	C 2*inv

	imull	%ecx, %ecx		C inv*inv

	movl	(%eax), %eax		C src low limb
	negl	%ebp			C -size

	imull	%esi, %ecx		C inv*inv*d

	subl	%ecx, %edi		C inv = 2*inv - inv*inv*d

	ASSERT(e,`	C d*inv == 1 mod 2^GMP_LIMB_BITS
	pushl	%eax
	movl	%esi, %eax
	imull	%edi, %eax
	cmpl	$1, %eax
	popl	%eax')

	jmp	L(entry)


C Rotating the mul to the top of the loop saves 1 cycle, presumably by
C hiding the loop control under the imul latency.
C
C The run time is 10 cycles, but decoding is only 9 (and the dependent chain
C only 8).  It's not clear how to get down to 9 cycles.
C
C The xor and rcl to handle the carry bit could be an sbb instead, with the
C the carry bit add becoming a sub, but that doesn't save anything.

L(top):
	C eax	(low product)
	C ebx	src end
	C ecx	carry bit, 0 or 1
	C edx	(high product, being carry limb)
	C esi	divisor
	C edi	inverse
	C ebp	counter, limbs, negative

	mull	%esi

	movl	(%ebx,%ebp,4), %eax
	addl	%ecx, %edx		C apply carry bit to carry limb

L(entry):
	xorl	%ecx, %ecx
	subl	%edx, %eax		C apply carry limb

	rcll	%ecx

	imull	%edi, %eax

	incl	%ebp
	jnz	L(top)



	popl	%ebx
	popl	%ebp

	mull	%esi

	popl	%edi
	popl	%esi

	leal	(%ecx,%edx), %eax

	ret

EPILOGUE()
ASM_END()
