dnl  Intel P6 mpn_modexact_1_odd -- exact division style remainder.

dnl  Copyright 2000-2002, 2007 Free Software Foundation, Inc.

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


C P6: 10.0 cycles/limb


C mp_limb_t mpn_modexact_1_odd (mp_srcptr src, mp_size_t size,
C                               mp_limb_t divisor);
C mp_limb_t mpn_modexact_1c_odd (mp_srcptr src, mp_size_t size,
C                                mp_limb_t divisor, mp_limb_t carry);
C
C It's not worth skipping a step at the end when high<divisor since the main
C loop is only 10 cycles.

defframe(PARAM_CARRY,  16)
defframe(PARAM_DIVISOR,12)
defframe(PARAM_SIZE,   8)
defframe(PARAM_SRC,    4)

dnl  Not enough room under modexact_1 to make these re-use the parameter
dnl  space, unfortunately.
defframe(SAVE_EBX,     -4)
defframe(SAVE_ESI,     -8)
defframe(SAVE_EDI,    -12)
deflit(STACK_SPACE, 12)

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
	movl	PARAM_SRC, %esi

	shrl	%eax			C d/2
	movl	%edi, SAVE_EDI

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

	movl	%ebx, SAVE_EBX
	movl	PARAM_SIZE, %ebx

	imull	PARAM_DIVISOR, %edi	C inv*inv*d

	subl	%edi, %eax		C inv = 2*inv - inv*inv*d
	leal	(%eax,%eax), %edi	C 2*inv

	imull	%eax, %eax		C inv*inv

	imull	PARAM_DIVISOR, %eax	C inv*inv*d

	leal	(%esi,%ebx,4), %esi	C src end
	negl	%ebx			C -size

	subl	%eax, %edi		C inv = 2*inv - inv*inv*d

	ASSERT(e,`	C d*inv == 1 mod 2^GMP_LIMB_BITS
	movl	PARAM_DIVISOR, %eax
	imull	%edi, %eax
	cmpl	$1, %eax')


C The dependent chain here is
C
C	subl	%edx, %eax       1
C	imull	%edi, %eax       4
C	mull	PARAM_DIVISOR    5
C			       ----
C	total			10
C
C and this is the measured speed.  No special scheduling is necessary, out
C of order execution hides the load latency.

L(top):
	C eax	scratch (src limb)
	C ebx	counter, limbs, negative
	C ecx	carry bit, 0 or 1
	C edx	carry limb, high of last product
	C esi	&src[size]
	C edi	inverse
	C ebp

	movl	(%esi,%ebx,4), %eax
	subl	%ecx, %eax

	sbbl	%ecx, %ecx
	subl	%edx, %eax

	sbbl	$0, %ecx

	imull	%edi, %eax

	negl	%ecx

	mull	PARAM_DIVISOR

	incl	%ebx
	jnz	L(top)


	movl	SAVE_ESI, %esi
	leal	(%ecx,%edx), %eax

	movl	SAVE_EDI, %edi

	movl	SAVE_EBX, %ebx
	addl	$STACK_SPACE, %esp

	ret

EPILOGUE()
ASM_END()
