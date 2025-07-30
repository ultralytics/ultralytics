dnl  AMD K7 mpn_modexact_1_odd -- exact division style remainder.

dnl  Copyright 2000-2002, 2004, 2007 Free Software Foundation, Inc.

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


C          cycles/limb
C Athlon:     11.0
C Hammer:      7.0


C mp_limb_t mpn_modexact_1_odd (mp_srcptr src, mp_size_t size,
C                               mp_limb_t divisor);
C mp_limb_t mpn_modexact_1c_odd (mp_srcptr src, mp_size_t size,
C                                mp_limb_t divisor, mp_limb_t carry);
C
C With the loop running at just 11 cycles it doesn't seem worth bothering to
C check for high<divisor to save one step.
C
C Using a divl for size==1 measures slower than the modexact method, which
C is not too surprising since for the latter it's only about 24 cycles to
C calculate the modular inverse.

defframe(PARAM_CARRY,  16)
defframe(PARAM_DIVISOR,12)
defframe(PARAM_SIZE,   8)
defframe(PARAM_SRC,    4)

defframe(SAVE_EBX,     -4)
defframe(SAVE_ESI,     -8)
defframe(SAVE_EDI,    -12)
defframe(SAVE_EBP,    -16)

deflit(STACK_SPACE, 16)

	TEXT

	ALIGN(16)
PROLOGUE(mpn_modexact_1c_odd)
deflit(`FRAME',0)

	movl	PARAM_CARRY, %ecx
	jmp	L(start_1c)

EPILOGUE()


	ALIGN(16)
PROLOGUE(mpn_modexact_1_odd)
deflit(`FRAME',0)

	xorl	%ecx, %ecx
L(start_1c):
	movl	PARAM_DIVISOR, %eax
	subl	$STACK_SPACE, %esp	FRAME_subl_esp(STACK_SPACE)

	movl	%esi, SAVE_ESI
	movl	PARAM_DIVISOR, %esi

	movl	%edi, SAVE_EDI

	shrl	%eax			C d/2

	andl	$127, %eax

ifdef(`PIC',`
	LEA(	binvert_limb_table, %edi)
	movzbl	(%eax,%edi), %edi		C inv 8 bits
',`
	movzbl	binvert_limb_table(%eax), %edi	C inv 8 bits
')

	xorl	%edx, %edx		C initial extra carry
	leal	(%edi,%edi), %eax	C 2*inv

	imull	%edi, %edi		C inv*inv

	movl	%ebp, SAVE_EBP
	movl	PARAM_SIZE, %ebp

	movl	%ebx, SAVE_EBX
	movl	PARAM_SRC, %ebx

	imull	%esi, %edi		C inv*inv*d

	subl	%edi, %eax		C inv = 2*inv - inv*inv*d
	leal	(%eax,%eax), %edi	C 2*inv

	imull	%eax, %eax		C inv*inv

	imull	%esi, %eax		C inv*inv*d

	leal	(%ebx,%ebp,4), %ebx	C src end
	negl	%ebp			C -size

	subl	%eax, %edi		C inv = 2*inv - inv*inv*d

	ASSERT(e,`	C d*inv == 1 mod 2^GMP_LIMB_BITS
	movl	%esi, %eax
	imull	%edi, %eax
	cmpl	$1, %eax')


C The dependent chain here is
C
C                            cycles
C	subl	%edx, %eax	1
C	imull	%edi, %eax	4
C	mull	%esi		6  (high limb)
C			      ----
C       total		       11
C
C Out of order execution hides the load latency for the source data, so no
C special scheduling is required.

L(top):
	C eax	src limb
	C ebx	src end ptr
	C ecx	next carry bit, 0 or 1 (or initial carry param)
	C edx	carry limb, high of last product
	C esi	divisor
	C edi	inverse
	C ebp	counter, limbs, negative

	movl	(%ebx,%ebp,4), %eax

	subl	%ecx, %eax		C apply carry bit
	movl	$0, %ecx

	setc	%cl			C new carry bit

	subl	%edx, %eax		C apply carry limb
	adcl	$0, %ecx

	imull	%edi, %eax

	mull	%esi

	incl	%ebp
	jnz	L(top)


	movl	SAVE_ESI, %esi
	movl	SAVE_EDI, %edi
	leal	(%ecx,%edx), %eax

	movl	SAVE_EBX, %ebx
	movl	SAVE_EBP, %ebp
	addl	$STACK_SPACE, %esp

	ret

EPILOGUE()
ASM_END()
