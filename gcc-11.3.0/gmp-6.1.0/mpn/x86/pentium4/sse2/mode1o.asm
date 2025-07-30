dnl  Intel Pentium-4 mpn_modexact_1_odd -- mpn by limb exact remainder.

dnl  Copyright 2001, 2002, 2007 Free Software Foundation, Inc.

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


C P4: 19.0 cycles/limb


C mp_limb_t mpn_modexact_1_odd (mp_srcptr src, mp_size_t size,
C                               mp_limb_t divisor);
C mp_limb_t mpn_modexact_1c_odd (mp_srcptr src, mp_size_t size,
C                                mp_limb_t divisor, mp_limb_t carry);
C

defframe(PARAM_CARRY,  16)
defframe(PARAM_DIVISOR,12)
defframe(PARAM_SIZE,   8)
defframe(PARAM_SRC,    4)

	TEXT

	ALIGN(16)
PROLOGUE(mpn_modexact_1c_odd)
deflit(`FRAME',0)

	movd	PARAM_CARRY, %mm1
	jmp	L(start_1c)

EPILOGUE()


	ALIGN(16)
PROLOGUE(mpn_modexact_1_odd)
deflit(`FRAME',0)

	pxor	%mm1, %mm1		C carry limb
L(start_1c):
	movl	PARAM_DIVISOR, %eax

	movd	PARAM_DIVISOR, %mm7

	shrl	%eax

	andl	$127, %eax		C d/2, 7 bits

ifdef(`PIC',`
	LEA(	binvert_limb_table, %edx)
	movzbl	(%eax,%edx), %eax		C inv 8 bits
',`
	movzbl	binvert_limb_table(%eax), %eax	C inv 8 bits
')

	C

	movd	%eax, %mm6		C inv

	movd	%eax, %mm0		C inv

	pmuludq	%mm6, %mm6		C inv*inv

	C

	pmuludq	%mm7, %mm6		C inv*inv*d
	paddd	%mm0, %mm0		C 2*inv

	C

	psubd	%mm6, %mm0		C inv = 2*inv - inv*inv*d
	pxor	%mm6, %mm6

	paddd	%mm0, %mm6
	pmuludq	%mm0, %mm0		C inv*inv

	C

	pmuludq	%mm7, %mm0		C inv*inv*d
	paddd	%mm6, %mm6		C 2*inv


	movl	PARAM_SRC, %eax
	movl	PARAM_SIZE, %ecx

	C

	psubd	%mm0, %mm6		C inv = 2*inv - inv*inv*d

	ASSERT(e,`	C expect d*inv == 1 mod 2^GMP_LIMB_BITS
	pushl	%eax	FRAME_pushl()
	movd	%mm6, %eax
	imul	PARAM_DIVISOR, %eax
	cmpl	$1, %eax
	popl	%eax	FRAME_popl()')

	pxor	%mm0, %mm0		C carry bit


C The dependent chain here is as follows.
C
C					latency
C	psubq	 s = (src-cbit) - climb	   2
C	pmuludq	 q = s*inverse		   8
C	pmuludq	 prod = q*divisor	   8
C	psrlq	 climb = high(prod)	   2
C					  --
C					  20
C
C Yet the loop measures 19.0 c/l, so obviously there's something gained
C there over a straight reading of the chip documentation.

L(top):
	C eax	src, incrementing
	C ebx
	C ecx	counter, limbs
	C edx
	C
	C mm0	carry bit
	C mm1	carry limb
	C mm6	inverse
	C mm7	divisor

	movd	(%eax), %mm2
	addl	$4, %eax

	psubq	%mm0, %mm2		C src - cbit

	psubq	%mm1, %mm2		C src - cbit - climb
	movq	%mm2, %mm0
	psrlq	$63, %mm0		C new cbit

	pmuludq	%mm6, %mm2		C s*inverse

	movq	%mm7, %mm1
	pmuludq	%mm2, %mm1		C q*divisor
	psrlq	$32, %mm1		C new climb

	subl	$1, %ecx
	jnz	L(top)


L(done):
	paddq	%mm1, %mm0
	movd	%mm0, %eax
	emms
	ret

EPILOGUE()
ASM_END()
