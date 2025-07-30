dnl  Intel Pentium-4 mpn_divexact_1 -- mpn by limb exact division.

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


C void mpn_divexact_1 (mp_ptr dst, mp_srcptr src, mp_size_t size,
C                      mp_limb_t divisor);
C
C Pairs of movd's are used to avoid unaligned loads.  Despite the loads not
C being on the dependent chain and there being plenty of cycles available,
C using an unaligned movq on every second iteration measured about 23 c/l.
C
C Using divl for size==1 seems a touch quicker than mul-by-inverse.  The mul
C will be about 9+2*4+2*2+10*4+19+12 = 92 cycles latency, though some of
C that might be hidden by out-of-order execution, whereas divl is around 60.
C At size==2 an extra 19 for the mul versus 60 for the divl will see the mul
C faster.

defframe(PARAM_DIVISOR,16)
defframe(PARAM_SIZE,   12)
defframe(PARAM_SRC,    8)
defframe(PARAM_DST,    4)

	TEXT

	ALIGN(16)
PROLOGUE(mpn_divexact_1)
deflit(`FRAME',0)

	movl	PARAM_SIZE, %edx

	movl	PARAM_SRC, %eax

	movl	PARAM_DIVISOR, %ecx
	subl	$1, %edx
	jnz	L(two_or_more)

	movl	(%eax), %eax
	xorl	%edx, %edx

	divl	%ecx
	movl	PARAM_DST, %ecx

	movl	%eax, (%ecx)
	ret


L(two_or_more):
	C eax	src
	C ebx
	C ecx	divisor
	C edx	size-1

	movl	%ecx, %eax
	bsfl	%ecx, %ecx		C trailing twos

	shrl	%cl, %eax		C d = divisor without twos
	movd	%eax, %mm6
	movd	%ecx, %mm7		C shift

	shrl	%eax			C d/2

	andl	$127, %eax		C d/2, 7 bits

ifdef(`PIC',`
	LEA(	binvert_limb_table, %ecx)
	movzbl	(%eax,%ecx), %eax		C inv 8 bits
',`
	movzbl	binvert_limb_table(%eax), %eax	C inv 8 bits
')

	C

	movd	%eax, %mm5		C inv

	movd	%eax, %mm0		C inv

	pmuludq	%mm5, %mm5		C inv*inv

	C

	pmuludq	%mm6, %mm5		C inv*inv*d
	paddd	%mm0, %mm0		C 2*inv

	C

	psubd	%mm5, %mm0		C inv = 2*inv - inv*inv*d
	pxor	%mm5, %mm5

	paddd	%mm0, %mm5
	pmuludq	%mm0, %mm0		C inv*inv

	pcmpeqd	%mm4, %mm4
	psrlq	$32, %mm4		C 0x00000000FFFFFFFF

	C

	pmuludq	%mm6, %mm0		C inv*inv*d
	paddd	%mm5, %mm5		C 2*inv

	movl	PARAM_SRC, %eax
	movl	PARAM_DST, %ecx
	pxor	%mm1, %mm1		C initial carry limb

	C

	psubd	%mm0, %mm5		C inv = 2*inv - inv*inv*d

	ASSERT(e,`	C expect d*inv == 1 mod 2^GMP_LIMB_BITS
	pushl	%eax	FRAME_pushl()
	movq	%mm6, %mm0
	pmuludq	%mm5, %mm0
	movd	%mm0, %eax
	cmpl	$1, %eax
	popl	%eax	FRAME_popl()')

	pxor	%mm0, %mm0		C initial carry bit


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
	C ecx	dst, incrementing
	C edx	counter, size-1 iterations
	C
	C mm0	carry bit
	C mm1	carry limb
	C mm4	0x00000000FFFFFFFF
	C mm5	inverse
	C mm6	divisor
	C mm7	shift

	movd	(%eax), %mm2
	movd	4(%eax), %mm3
	addl	$4, %eax
	punpckldq %mm3, %mm2

	psrlq	%mm7, %mm2
	pand	%mm4, %mm2		C src
	psubq	%mm0, %mm2		C src - cbit

	psubq	%mm1, %mm2		C src - cbit - climb
	movq	%mm2, %mm0
	psrlq	$63, %mm0		C new cbit

	pmuludq	%mm5, %mm2		C s*inverse
	movd	%mm2, (%ecx)		C q
	addl	$4, %ecx

	movq	%mm6, %mm1
	pmuludq	%mm2, %mm1		C q*divisor
	psrlq	$32, %mm1		C new climb

	subl	$1, %edx
	jnz	L(top)


L(done):
	movd	(%eax), %mm2
	psrlq	%mm7, %mm2		C src
	psubq	%mm0, %mm2		C src - cbit

	psubq	%mm1, %mm2		C src - cbit - climb

	pmuludq	%mm5, %mm2		C s*inverse
	movd	%mm2, (%ecx)		C q

	emms
	ret

EPILOGUE()
ASM_END()
