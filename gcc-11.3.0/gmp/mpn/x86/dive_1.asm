dnl  x86 mpn_divexact_1 -- mpn by limb exact division.

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


C     cycles/limb
C P54    30.0
C P55    29.0
C P6     13.0 odd divisor, 12.0 even (strangely)
C K6     14.0
C K7     12.0
C P4     42.0


C mp_limb_t mpn_divexact_1 (mp_ptr dst, mp_srcptr src, mp_size_t size,
C                           mp_limb_t divisor);
C

defframe(PARAM_DIVISOR,16)
defframe(PARAM_SIZE,   12)
defframe(PARAM_SRC,    8)
defframe(PARAM_DST,    4)

dnl  re-use parameter space
define(VAR_INVERSE,`PARAM_SRC')

	TEXT

	ALIGN(16)
PROLOGUE(mpn_divexact_1)
deflit(`FRAME',0)

	movl	PARAM_DIVISOR, %eax
	pushl	%ebp	FRAME_pushl()

	movl	PARAM_SIZE, %ebp
	pushl	%edi	FRAME_pushl()

	pushl	%ebx	FRAME_pushl()
	movl	$-1, %ecx		C shift count

	pushl	%esi	FRAME_pushl()

L(strip_twos):
	incl	%ecx

	shrl	%eax
	jnc	L(strip_twos)

	leal	1(%eax,%eax), %ebx	C d without twos
	andl	$127, %eax		C d/2, 7 bits

ifdef(`PIC',`
	LEA(	binvert_limb_table, %edx)
	movzbl	(%eax,%edx), %eax		C inv 8 bits
',`
	movzbl	binvert_limb_table(%eax), %eax	C inv 8 bits
')

	leal	(%eax,%eax), %edx	C 2*inv
	movl	%ebx, PARAM_DIVISOR	C d without twos

	imull	%eax, %eax		C inv*inv

	movl	PARAM_SRC, %esi
	movl	PARAM_DST, %edi

	imull	%ebx, %eax		C inv*inv*d

	subl	%eax, %edx		C inv = 2*inv - inv*inv*d
	leal	(%edx,%edx), %eax	C 2*inv

	imull	%edx, %edx		C inv*inv

	leal	(%esi,%ebp,4), %esi	C src end
	leal	(%edi,%ebp,4), %edi	C dst end
	negl	%ebp			C -size

	imull	%ebx, %edx		C inv*inv*d

	subl	%edx, %eax		C inv = 2*inv - inv*inv*d

	ASSERT(e,`	C expect d*inv == 1 mod 2^GMP_LIMB_BITS
	pushl	%eax	FRAME_pushl()
	imull	PARAM_DIVISOR, %eax
	cmpl	$1, %eax
	popl	%eax	FRAME_popl()')

	movl	%eax, VAR_INVERSE
	movl	(%esi,%ebp,4), %eax	C src[0]

	xorl	%ebx, %ebx
	xorl	%edx, %edx

	incl	%ebp
	jz	L(one)

	movl	(%esi,%ebp,4), %edx	C src[1]

	shrdl(	%cl, %edx, %eax)

	movl	VAR_INVERSE, %edx
	jmp	L(entry)


	ALIGN(8)
	nop	C k6 code alignment
	nop
L(top):
	C eax	q
	C ebx	carry bit, 0 or -1
	C ecx	shift
	C edx	carry limb
	C esi	src end
	C edi	dst end
	C ebp	counter, limbs, negative

	movl	-4(%esi,%ebp,4), %eax
	subl	%ebx, %edx		C accumulate carry bit

	movl	(%esi,%ebp,4), %ebx

	shrdl(	%cl, %ebx, %eax)

	subl	%edx, %eax		C apply carry limb
	movl	VAR_INVERSE, %edx

	sbbl	%ebx, %ebx

L(entry):
	imull	%edx, %eax

	movl	%eax, -4(%edi,%ebp,4)
	movl	PARAM_DIVISOR, %edx

	mull	%edx

	incl	%ebp
	jnz	L(top)


	movl	-4(%esi), %eax		C src high limb
L(one):
	shrl	%cl, %eax
	popl	%esi	FRAME_popl()

	addl	%ebx, %eax		C apply carry bit
	popl	%ebx	FRAME_popl()

	subl	%edx, %eax		C apply carry limb

	imull	VAR_INVERSE, %eax

	movl	%eax, -4(%edi)

	popl	%edi
	popl	%ebp

	ret

EPILOGUE()
ASM_END()
