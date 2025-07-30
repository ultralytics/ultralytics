dnl  Intel P6 mpn_modexact_1_odd -- exact division style remainder.

dnl  Rearranged from mpn/x86/p6/dive_1.asm by Marco Bodrato.

dnl  Copyright 2001, 2002, 2007, 2011 Free Software Foundation, Inc.

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


C       odd  even  divisor
C P6:  10.0  12.0  cycles/limb

C MULFUNC_PROLOGUE(mpn_bdiv_q_1 mpn_pi1_bdiv_q_1)

C The odd case is basically the same as mpn_modexact_1_odd, just with an
C extra store, and it runs at the same 10 cycles which is the dependent
C chain.
C
C The shifts for the even case aren't on the dependent chain so in principle
C it could run the same too, but nothing running at 10 has been found.
C Perhaps there's too many uops (an extra 4 over the odd case).

defframe(PARAM_SHIFT,  24)
defframe(PARAM_INVERSE,20)
defframe(PARAM_DIVISOR,16)
defframe(PARAM_SIZE,   12)
defframe(PARAM_SRC,     8)
defframe(PARAM_DST,     4)

defframe(SAVE_EBX,     -4)
defframe(SAVE_ESI,     -8)
defframe(SAVE_EDI,    -12)
defframe(SAVE_EBP,    -16)
deflit(STACK_SPACE, 16)

dnl  re-use parameter space
define(VAR_INVERSE,`PARAM_SRC')

	TEXT

C mp_limb_t
C mpn_pi1_bdiv_q_1 (mp_ptr dst, mp_srcptr src, mp_size_t size, mp_limb_t divisor,
C		    mp_limb_t inverse, int shift)

	ALIGN(16)
PROLOGUE(mpn_pi1_bdiv_q_1)
deflit(`FRAME',0)

	subl	$STACK_SPACE, %esp	FRAME_subl_esp(STACK_SPACE)

	movl	%esi, SAVE_ESI
	movl	PARAM_SRC, %esi

	movl	%ebx, SAVE_EBX
	movl	PARAM_SIZE, %ebx

	movl	%ebp, SAVE_EBP
	movl	PARAM_INVERSE, %ebp

	movl	PARAM_SHIFT, %ecx	C trailing twos

L(common):
	movl	%edi, SAVE_EDI
	movl	PARAM_DST, %edi

	leal	(%esi,%ebx,4), %esi	C src end

	leal	(%edi,%ebx,4), %edi	C dst end
	negl	%ebx			C -size

	movl	(%esi,%ebx,4), %eax	C src[0]

	orl	%ecx, %ecx
	jz	L(odd_entry)

	movl	%edi, PARAM_DST
	movl	%ebp, VAR_INVERSE

L(even):
	C eax	src[0]
	C ebx	counter, limbs, negative
	C ecx	shift
	C edx
	C esi
	C edi
	C ebp

	xorl	%ebp, %ebp		C initial carry bit
	xorl	%edx, %edx		C initial carry limb (for size==1)

	incl	%ebx
	jz	L(even_one)

	movl	(%esi,%ebx,4), %edi	C src[1]

	shrdl(	%cl, %edi, %eax)

	jmp	L(even_entry)


L(even_top):
	C eax	scratch
	C ebx	counter, limbs, negative
	C ecx	shift
	C edx	scratch
	C esi	&src[size]
	C edi	&dst[size] and scratch
	C ebp	carry bit

	movl	(%esi,%ebx,4), %edi

	mull	PARAM_DIVISOR

	movl	-4(%esi,%ebx,4), %eax
	shrdl(	%cl, %edi, %eax)

	subl	%ebp, %eax

	sbbl	%ebp, %ebp
	subl	%edx, %eax

	sbbl	$0, %ebp

L(even_entry):
	imull	VAR_INVERSE, %eax

	movl	PARAM_DST, %edi
	negl	%ebp

	movl	%eax, -4(%edi,%ebx,4)
	incl	%ebx
	jnz	L(even_top)

	mull	PARAM_DIVISOR

	movl	-4(%esi), %eax

L(even_one):
	shrl	%cl, %eax
	movl	SAVE_ESI, %esi

	subl	%ebp, %eax
	movl	SAVE_EBP, %ebp

	subl	%edx, %eax
	movl	SAVE_EBX, %ebx

	imull	VAR_INVERSE, %eax

	movl	%eax, -4(%edi)
	movl	SAVE_EDI, %edi
	addl	$STACK_SPACE, %esp

	ret

C The dependent chain here is
C
C	subl	%edx, %eax       1
C	imull	%ebp, %eax       4
C	mull	PARAM_DIVISOR    5
C			       ----
C	total			10
C
C and this is the measured speed.  No special scheduling is necessary, out
C of order execution hides the load latency.

L(odd_top):
	C eax	scratch (src limb)
	C ebx	counter, limbs, negative
	C ecx	carry bit
	C edx	carry limb, high of last product
	C esi	&src[size]
	C edi	&dst[size]
	C ebp	inverse

	mull	PARAM_DIVISOR

	movl	(%esi,%ebx,4), %eax
	subl	%ecx, %eax

	sbbl	%ecx, %ecx
	subl	%edx, %eax

	sbbl	$0, %ecx

L(odd_entry):
	imull	%ebp, %eax

	movl	%eax, (%edi,%ebx,4)
	negl	%ecx

	incl	%ebx
	jnz	L(odd_top)


	movl	SAVE_ESI, %esi

	movl	SAVE_EDI, %edi

	movl	SAVE_EBP, %ebp

	movl	SAVE_EBX, %ebx
	addl	$STACK_SPACE, %esp

	ret

EPILOGUE()

C mp_limb_t mpn_bdiv_q_1 (mp_ptr dst, mp_srcptr src, mp_size_t size,
C                           mp_limb_t divisor);
C

	ALIGN(16)
PROLOGUE(mpn_bdiv_q_1)
deflit(`FRAME',0)

	movl	PARAM_DIVISOR, %eax
	subl	$STACK_SPACE, %esp	FRAME_subl_esp(STACK_SPACE)

	movl	%esi, SAVE_ESI
	movl	PARAM_SRC, %esi

	movl	%ebx, SAVE_EBX
	movl	PARAM_SIZE, %ebx

	bsfl	%eax, %ecx		C trailing twos

	movl	%ebp, SAVE_EBP

	shrl	%cl, %eax		C d without twos

	movl	%eax, %edx
	shrl	%eax			C d/2 without twos

	movl	%edx, PARAM_DIVISOR
	andl	$127, %eax

ifdef(`PIC',`
	LEA(	binvert_limb_table, %ebp)
	movzbl	(%eax,%ebp), %ebp		C inv 8 bits
',`
	movzbl	binvert_limb_table(%eax), %ebp	C inv 8 bits
')

	leal	(%ebp,%ebp), %eax	C 2*inv

	imull	%ebp, %ebp		C inv*inv
	imull	%edx, %ebp	C inv*inv*d

	subl	%ebp, %eax		C inv = 2*inv - inv*inv*d
	leal	(%eax,%eax), %ebp	C 2*inv

	imull	%eax, %eax		C inv*inv
	imull	%edx, %eax	C inv*inv*d

	subl	%eax, %ebp		C inv = 2*inv - inv*inv*d

	jmp	L(common)

EPILOGUE()
ASM_END()
