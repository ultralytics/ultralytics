dnl  Intel Pentium mpn_divexact_1 -- mpn by limb exact division.

dnl  Rearranged from mpn/x86/pentium/dive_1.asm by Marco Bodrato.

dnl  Copyright 2001, 2002, 2011, 2014 Free Software Foundation, Inc.

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


C         divisor
C       odd   even
C P54:  24.5  30.5   cycles/limb
C P55:  23.0  28.0

MULFUNC_PROLOGUE(mpn_bdiv_q_1 mpn_pi1_bdiv_q_1)

C The P55 speeds noted above, 23 cycles odd or 28 cycles even, are as
C expected.  On P54 in the even case the shrdl pairing nonsense (see
C mpn/x86/pentium/README) costs 1 cycle, but it's not clear why there's a
C further 1.5 slowdown for both odd and even.

defframe(PARAM_SHIFT,  24)
defframe(PARAM_INVERSE,20)
defframe(PARAM_DIVISOR,16)
defframe(PARAM_SIZE,   12)
defframe(PARAM_SRC,    8)
defframe(PARAM_DST,    4)

dnl  re-use parameter space
define(VAR_INVERSE,`PARAM_DST')

	TEXT

	ALIGN(32)
C mp_limb_t mpn_bdiv_q_1 (mp_ptr dst, mp_srcptr src, mp_size_t size,
C                           mp_limb_t divisor);
C
PROLOGUE(mpn_bdiv_q_1)
deflit(`FRAME',0)

	movl	$-1, %ecx
	movl	PARAM_DIVISOR, %eax

L(strip_twos):
	ASSERT(nz, `orl %eax, %eax')
	shrl	%eax
	incl	%ecx			C shift count

	jnc	L(strip_twos)

	leal	1(%eax,%eax), %edx	C d
	andl	$127, %eax		C d/2, 7 bits

	pushl	%ebx		FRAME_pushl()
	pushl	%ebp		FRAME_pushl()

ifdef(`PIC',`
ifdef(`DARWIN',`
	LEA(	binvert_limb_table, %ebp)
	movzbl	(%eax,%ebp), %eax
',`
	call	L(here)
L(here):
	popl	%ebp			C eip

	addl	$_GLOBAL_OFFSET_TABLE_+[.-L(here)], %ebp
	C AGI
	movl	binvert_limb_table@GOT(%ebp), %ebp
	C AGI
	movzbl	(%eax,%ebp), %eax
')
',`

dnl non-PIC
	movzbl	binvert_limb_table(%eax), %eax	C inv 8 bits
')

	movl	%eax, %ebp		C inv
	addl	%eax, %eax		C 2*inv

	imull	%ebp, %ebp		C inv*inv

	imull	%edx, %ebp		C inv*inv*d

	subl	%ebp, %eax		C inv = 2*inv - inv*inv*d
	movl	PARAM_SIZE, %ebx

	movl	%eax, %ebp
	addl	%eax, %eax		C 2*inv

	imull	%ebp, %ebp		C inv*inv

	imull	%edx, %ebp		C inv*inv*d

	subl	%ebp, %eax		C inv = 2*inv - inv*inv*d
	movl	%edx, PARAM_DIVISOR	C d without twos

	ASSERT(e,`	C expect d*inv == 1 mod 2^GMP_LIMB_BITS
	pushl	%eax	FRAME_pushl()
	imull	PARAM_DIVISOR, %eax
	cmpl	$1, %eax
	popl	%eax	FRAME_popl()')

	jmp	L(common)
EPILOGUE()

C mp_limb_t
C mpn_pi1_bdiv_q_1 (mp_ptr dst, mp_srcptr src, mp_size_t size, mp_limb_t divisor,
C		    mp_limb_t inverse, int shift)
	ALIGN(32)
PROLOGUE(mpn_pi1_bdiv_q_1)
deflit(`FRAME',0)

	movl	PARAM_SHIFT, %ecx

	pushl	%ebx		FRAME_pushl()
	pushl	%ebp		FRAME_pushl()

	movl	PARAM_SIZE, %ebx
	movl	PARAM_INVERSE, %eax

L(common):
	pushl	%esi		FRAME_pushl()
	push	%edi		FRAME_pushl()

	movl	PARAM_SRC, %esi
	movl	PARAM_DST, %edi
	movl	%eax, VAR_INVERSE

	leal	(%esi,%ebx,4), %esi	C src end
	leal	(%edi,%ebx,4), %edi	C dst end

	negl	%ebx			C -size

	xorl	%ebp, %ebp		C initial carry bit

	orl	%ecx, %ecx		C shift
	movl	(%esi,%ebx,4), %eax	C src low limb
	jz	L(odd_entry)

	xorl	%edx, %edx		C initial carry limb (for even, if one)
	incl	%ebx
	jz	L(one)

	movl	(%esi,%ebx,4), %edx	C src second limb (for even)
	shrdl(	%cl, %edx, %eax)

	jmp	L(even_entry)


	ALIGN(8)
L(odd_top):
	C eax	scratch
	C ebx	counter, limbs, negative
	C ecx
	C edx
	C esi	src end
	C edi	dst end
	C ebp	carry bit, 0 or -1

	mull	PARAM_DIVISOR

	movl	(%esi,%ebx,4), %eax
	subl	%ebp, %edx

	subl	%edx, %eax

	sbbl	%ebp, %ebp

L(odd_entry):
	imull	VAR_INVERSE, %eax

	movl	%eax, (%edi,%ebx,4)

	incl	%ebx
	jnz	L(odd_top)

	popl	%edi
	popl	%esi

	popl	%ebp
	popl	%ebx

	ret

L(even_top):
	C eax	scratch
	C ebx	counter, limbs, negative
	C ecx	twos
	C edx
	C esi	src end
	C edi	dst end
	C ebp	carry bit, 0 or -1

	mull	PARAM_DIVISOR

	subl	%ebp, %edx		C carry bit
	movl	-4(%esi,%ebx,4), %eax	C src limb

	movl	(%esi,%ebx,4), %ebp	C and one above it

	shrdl(	%cl, %ebp, %eax)

	subl	%edx, %eax		C carry limb

	sbbl	%ebp, %ebp

L(even_entry):
	imull	VAR_INVERSE, %eax

	movl	%eax, -4(%edi,%ebx,4)
	incl	%ebx

	jnz	L(even_top)

	mull	PARAM_DIVISOR

	movl	-4(%esi), %eax		C src high limb
	subl	%ebp, %edx

L(one):
	shrl	%cl, %eax

	subl	%edx, %eax		C no carry if division is exact

	imull	VAR_INVERSE, %eax

	movl	%eax, -4(%edi)		C dst high limb
	nop				C protect against cache bank clash

	popl	%edi
	popl	%esi

	popl	%ebp
	popl	%ebx

	ret

EPILOGUE()
ASM_END()
