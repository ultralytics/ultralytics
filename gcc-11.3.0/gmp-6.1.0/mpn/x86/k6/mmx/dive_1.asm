dnl  AMD K6 mpn_divexact_1 -- mpn by limb exact division.

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


C         divisor
C       odd   even
C K6:   10.0  12.0  cycles/limb
C K6-2: 10.0  11.5


C void mpn_divexact_1 (mp_ptr dst, mp_srcptr src, mp_size_t size,
C                      mp_limb_t divisor);
C
C A simple divl is used for size==1.  This is about 10 cycles faster for an
C odd divisor or 20 cycles for an even divisor.
C
C The loops are quite sensitive to code alignment, speeds should be
C rechecked (odd and even divisor, pic and non-pic) if contemplating
C changing anything.

defframe(PARAM_DIVISOR,16)
defframe(PARAM_SIZE,   12)
defframe(PARAM_SRC,    8)
defframe(PARAM_DST,    4)

dnl  re-use parameter space
define(VAR_INVERSE,`PARAM_DST')

	TEXT

	ALIGN(32)
PROLOGUE(mpn_divexact_1)
deflit(`FRAME',0)

	movl	PARAM_SIZE, %ecx

	movl	PARAM_SRC, %eax
	xorl	%edx, %edx

	cmpl	$1, %ecx
	jnz	L(two_or_more)

	movl	(%eax), %eax

	divl	PARAM_DIVISOR

	movl	PARAM_DST, %ecx
	movl	%eax, (%ecx)

	ret


L(two_or_more):
	movl	PARAM_DIVISOR, %eax
	pushl	%ebx		FRAME_pushl()

	movl	PARAM_SRC, %ebx
	pushl	%ebp		FRAME_pushl()

L(strip_twos):
	shrl	%eax
	incl	%edx			C will get shift+1

	jnc	L(strip_twos)
	pushl	%esi		FRAME_pushl()

	leal	1(%eax,%eax), %esi	C d without twos
	andl	$127, %eax		C d/2, 7 bits

ifdef(`PIC',`
	LEA(	binvert_limb_table, %ebp)
Zdisp(	movzbl,	0,(%eax,%ebp), %eax)
',`
	movzbl	binvert_limb_table(%eax), %eax	C inv 8 bits
')
	pushl	%edi		FRAME_pushl()

	leal	(%eax,%eax), %ebp	C 2*inv

	imull	%eax, %eax		C inv*inv

	movl	PARAM_DST, %edi

	imull	%esi, %eax		C inv*inv*d

	subl	%eax, %ebp		C inv = 2*inv - inv*inv*d
	leal	(%ebp,%ebp), %eax	C 2*inv

	imull	%ebp, %ebp		C inv*inv

	movl	%esi, PARAM_DIVISOR	C d without twos
	leal	(%ebx,%ecx,4), %ebx	C src end

	imull	%esi, %ebp		C inv*inv*d

	leal	(%edi,%ecx,4), %edi	C dst end
	negl	%ecx			C -size

	subl	%ebp, %eax		C inv = 2*inv - inv*inv*d
	subl	$1, %edx		C shift amount, and clear carry

	ASSERT(e,`	C expect d*inv == 1 mod 2^GMP_LIMB_BITS
	pushl	%eax	FRAME_pushl()
	imull	PARAM_DIVISOR, %eax
	cmpl	$1, %eax
	popl	%eax	FRAME_popl()')

	movl	%eax, VAR_INVERSE
	jnz	L(even)

	movl	(%ebx,%ecx,4), %esi	C src low limb
	jmp	L(odd_entry)


	ALIGN(16)
	nop	C code alignment
L(odd_top):
	C eax	scratch
	C ebx	src end
	C ecx	counter, limbs, negative
	C edx	inverse
	C esi	next limb, adjusted for carry
	C edi	dst end
	C ebp	carry bit, 0 or -1

	imull	%edx, %esi

	movl	PARAM_DIVISOR, %eax
	movl	%esi, -4(%edi,%ecx,4)

	mull	%esi			C carry limb in edx

	subl	%ebp, %edx		C apply carry bit
	movl	(%ebx,%ecx,4), %esi

L(odd_entry):
	subl	%edx, %esi		C apply carry limb
	movl	VAR_INVERSE, %edx

	sbbl	%ebp, %ebp		C 0 or -1

	incl	%ecx
	jnz	L(odd_top)


	imull	%edx, %esi

	movl	%esi, -4(%edi,%ecx,4)

	popl	%edi
	popl	%esi

	popl	%ebp
	popl	%ebx

	ret


L(even):
	C eax
	C ebx	src end
	C ecx	-size
	C edx	twos
	C esi
	C edi	dst end
	C ebp

	xorl	%ebp, %ebp
Zdisp(	movq,	0,(%ebx,%ecx,4), %mm0)	C src[0,1]

	movd	%edx, %mm7
	movl	VAR_INVERSE, %edx

	addl	$2, %ecx
	psrlq	%mm7, %mm0

	movd	%mm0, %esi
	jz	L(even_two)		C if only two limbs


C Out-of-order execution is good enough to hide the load/rshift/movd
C latency.  Having imul at the top of the loop gives 11.5 c/l instead of 12,
C on K6-2.  In fact there's only 11 of decode, but nothing running at 11 has
C been found.  Maybe the fact every second movq is unaligned costs the extra
C 0.5.

L(even_top):
	C eax	scratch
	C ebx	src end
	C ecx	counter, limbs, negative
	C edx	inverse
	C esi	next limb, adjusted for carry
	C edi	dst end
	C ebp	carry bit, 0 or -1
	C
	C mm0	scratch, source limbs
	C mm7	twos

	imull	%edx, %esi

	movl	%esi, -8(%edi,%ecx,4)
	movl	PARAM_DIVISOR, %eax

	mull	%esi			C carry limb in edx

	movq	-4(%ebx,%ecx,4), %mm0
	psrlq	%mm7, %mm0

	movd	%mm0, %esi
	subl	%ebp, %edx		C apply carry bit

	subl	%edx, %esi		C apply carry limb
	movl	VAR_INVERSE, %edx

	sbbl	%ebp, %ebp		C 0 or -1

	incl	%ecx
	jnz	L(even_top)


L(even_two):
	movd	-4(%ebx), %mm0		C src high limb
	psrlq	%mm7, %mm0

	imull	%edx, %esi

	movl	%esi, -8(%edi)
	movl	PARAM_DIVISOR, %eax

	mull	%esi			C carry limb in edx

	movd	%mm0, %esi
	subl	%ebp, %edx		C apply carry bit

	movl	VAR_INVERSE, %eax
	subl	%edx, %esi		C apply carry limb

	imull	%eax, %esi

	movl	%esi, -4(%edi)

	popl	%edi
	popl	%esi

	popl	%ebp
	popl	%ebx

	emms_or_femms

	ret

EPILOGUE()
ASM_END()
