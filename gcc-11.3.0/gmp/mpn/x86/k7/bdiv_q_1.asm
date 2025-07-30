dnl  AMD K7 mpn_bdiv_q_1 -- mpn by limb exact division.

dnl  Rearranged from mpn/x86/k7/dive_1.asm by Marco Bodrato.

dnl  Copyright 2001, 2002, 2004, 2007, 2011 Free Software Foundation, Inc.

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
C Hammer:      9.0


C void mpn_divexact_1 (mp_ptr dst, mp_srcptr src, mp_size_t size,
C                      mp_limb_t divisor);
C
C The dependent chain is mul+imul+sub for 11 cycles and that speed is
C achieved with no special effort.  The load and shrld latencies are hidden
C by out of order execution.
C
C It's a touch faster on size==1 to use the mul-by-inverse than divl.

defframe(PARAM_SHIFT,  24)
defframe(PARAM_INVERSE,20)
defframe(PARAM_DIVISOR,16)
defframe(PARAM_SIZE,   12)
defframe(PARAM_SRC,    8)
defframe(PARAM_DST,    4)

defframe(SAVE_EBX,     -4)
defframe(SAVE_ESI,     -8)
defframe(SAVE_EDI,    -12)
defframe(SAVE_EBP,    -16)
defframe(VAR_INVERSE, -20)
defframe(VAR_DST_END, -24)

deflit(STACK_SPACE, 24)

	TEXT

C mp_limb_t
C mpn_pi1_bdiv_q_1 (mp_ptr dst, mp_srcptr src, mp_size_t size, mp_limb_t divisor,
C		    mp_limb_t inverse, int shift)
	ALIGN(16)
PROLOGUE(mpn_pi1_bdiv_q_1)
deflit(`FRAME',0)

	subl	$STACK_SPACE, %esp	deflit(`FRAME',STACK_SPACE)
	movl	PARAM_SHIFT, %ecx	C shift count

	movl	%ebp, SAVE_EBP
	movl	PARAM_SIZE, %ebp

	movl	%esi, SAVE_ESI
	movl	PARAM_SRC, %esi

	movl	%edi, SAVE_EDI
	movl	PARAM_DST, %edi

	movl	%ebx, SAVE_EBX

	leal	(%esi,%ebp,4), %esi	C src end
	leal	(%edi,%ebp,4), %edi	C dst end
	negl	%ebp			C -size

	movl	PARAM_INVERSE, %eax	C inv

L(common):
	movl	%eax, VAR_INVERSE
	movl	(%esi,%ebp,4), %eax	C src[0]

	incl	%ebp
	jz	L(one)

	movl	(%esi,%ebp,4), %edx	C src[1]

	shrdl(	%cl, %edx, %eax)

	movl	%edi, VAR_DST_END
	xorl	%ebx, %ebx
	jmp	L(entry)

	ALIGN(8)
L(top):
	C eax	q
	C ebx	carry bit, 0 or 1
	C ecx	shift
	C edx
	C esi	src end
	C edi	dst end
	C ebp	counter, limbs, negative

	mull	PARAM_DIVISOR		C carry limb in edx

	movl	-4(%esi,%ebp,4), %eax
	movl	(%esi,%ebp,4), %edi

	shrdl(	%cl, %edi, %eax)

	subl	%ebx, %eax		C apply carry bit
	setc	%bl
	movl	VAR_DST_END, %edi

	subl	%edx, %eax		C apply carry limb
	adcl	$0, %ebx

L(entry):
	imull	VAR_INVERSE, %eax

	movl	%eax, -4(%edi,%ebp,4)
	incl	%ebp
	jnz	L(top)


	mull	PARAM_DIVISOR		C carry limb in edx

	movl	-4(%esi), %eax		C src high limb
	shrl	%cl, %eax
	movl	SAVE_ESI, %esi

	subl	%ebx, %eax		C apply carry bit
	movl	SAVE_EBX, %ebx
	movl	SAVE_EBP, %ebp

	subl	%edx, %eax		C apply carry limb

	imull	VAR_INVERSE, %eax

	movl	%eax, -4(%edi)
	movl	SAVE_EDI, %edi
	addl	$STACK_SPACE, %esp

	ret

L(one):
	shrl	%cl, %eax
	movl	SAVE_ESI, %esi
	movl	SAVE_EBX, %ebx

	imull	VAR_INVERSE, %eax

	movl	SAVE_EBP, %ebp

	movl	%eax, -4(%edi)
	movl	SAVE_EDI, %edi
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
	subl	$STACK_SPACE, %esp	deflit(`FRAME',STACK_SPACE)
	movl	$-1, %ecx		C shift count

	movl	%ebp, SAVE_EBP
	movl	PARAM_SIZE, %ebp

	movl	%esi, SAVE_ESI
	movl	%edi, SAVE_EDI

	C If there's usually only one or two trailing zero bits then this
	C should be faster than bsfl.
L(strip_twos):
	incl	%ecx
	shrl	%eax
	jnc	L(strip_twos)

	movl	%ebx, SAVE_EBX
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

	jmp	L(common)
EPILOGUE()
ASM_END()
