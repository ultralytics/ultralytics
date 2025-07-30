dnl  Intel Pentium mpn_modexact_1_odd -- exact division style remainder.

dnl  Copyright 2000-2002, 2014 Free Software Foundation, Inc.

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


C P5: 23.0 cycles/limb


C mp_limb_t mpn_modexact_1_odd (mp_srcptr src, mp_size_t size,
C                               mp_limb_t divisor);
C mp_limb_t mpn_modexact_1c_odd (mp_srcptr src, mp_size_t size,
C                                mp_limb_t divisor, mp_limb_t carry);
C
C There seems no way to pair up the two lone instructions in the main loop.
C
C The special case for size==1 saves about 20 cycles (non-PIC), making it
C the same as mpn_mod_1, and in fact making modexact faster than mod_1 at
C all sizes.
C
C Alternatives:
C
C Using mmx for the multiplies might be possible, with pmullw and pmulhw
C having just 3 cycle latencies, but carry bit handling would probably be
C complicated.

defframe(PARAM_CARRY,  16)
defframe(PARAM_DIVISOR,12)
defframe(PARAM_SIZE,   8)
defframe(PARAM_SRC,    4)

dnl  re-using parameter space
define(VAR_INVERSE,`PARAM_SIZE')

	TEXT

	ALIGN(16)
PROLOGUE(mpn_modexact_1c_odd)
deflit(`FRAME',0)

	movl	PARAM_DIVISOR, %eax
	movl	PARAM_CARRY, %edx

	jmp	L(start_1c)

EPILOGUE()

	ALIGN(16)
PROLOGUE(mpn_modexact_1_odd)
deflit(`FRAME',0)

	movl	PARAM_DIVISOR, %eax
	xorl	%edx, %edx		C carry

L(start_1c):

ifdef(`PIC',`
ifdef(`DARWIN',`
	shrl	%eax			C d/2
	LEA(	binvert_limb_table, %ecx)
	pushl	%ebx		FRAME_pushl()
	movl	PARAM_SIZE, %ebx

	andl	$127, %eax
	subl	$2, %ebx

	movb	(%eax,%ecx), %cl
	jc	L(one_limb)
',`
	call	L(here)		FRAME_pushl()
L(here):

	shrl	%eax			C d/2
	movl	(%esp), %ecx		C eip

	addl	$_GLOBAL_OFFSET_TABLE_+[.-L(here)], %ecx
	movl	%ebx, (%esp)		C push ebx

	andl	$127, %eax
	movl	PARAM_SIZE, %ebx

	movl	binvert_limb_table@GOT(%ecx), %ecx
	subl	$2, %ebx

	movb	(%eax,%ecx), %cl			C inv 8 bits
	jc	L(one_limb)
')
',`
dnl non-PIC
	shrl	%eax			C d/2
	pushl	%ebx		FRAME_pushl()

	movl	PARAM_SIZE, %ebx
	andl	$127, %eax

	subl	$2, %ebx
	jc	L(one_limb)

	movb	binvert_limb_table(%eax), %cl		C inv 8 bits
')

	movl	%ecx, %eax
	addl	%ecx, %ecx		C 2*inv

	imull	%eax, %eax		C inv*inv

	imull	PARAM_DIVISOR, %eax	C inv*inv*d

	subl	%eax, %ecx		C inv = 2*inv - inv*inv*d

	movl	%ecx, %eax
	addl	%ecx, %ecx		C 2*inv

	imull	%eax, %eax		C inv*inv

	imull	PARAM_DIVISOR, %eax	C inv*inv*d

	subl	%eax, %ecx		C inv = 2*inv - inv*inv*d
	pushl	%esi		FRAME_pushl()

	ASSERT(e,`	C d*inv == 1 mod 2^GMP_LIMB_BITS
	movl	%ecx, %eax
	imull	PARAM_DIVISOR, %eax
	cmpl	$1, %eax')

	movl	PARAM_SRC, %esi
	movl	%ecx, VAR_INVERSE

	movl	(%esi), %eax		C src[0]
	leal	4(%esi,%ebx,4), %esi	C &src[size-1]

	xorl	$-1, %ebx		C -(size-1)
	ASSERT(nz)
	jmp	L(entry)


C The use of VAR_INVERSE means only a store is needed for that value, rather
C than a push and pop of say %edi.

	ALIGN(16)
L(top):
	C eax	scratch, low product
	C ebx	counter, limbs, negative
	C ecx	carry bit
	C edx	scratch, high product
	C esi	&src[size-1]
	C edi
	C ebp

	mull	PARAM_DIVISOR		C h:dummy = q*d

	movl	(%esi,%ebx,4), %eax	C src[i]
	subl	%ecx, %edx		C h -= -c

L(entry):
	subl	%edx, %eax		C s = src[i] - h

	sbbl	%ecx, %ecx		C new -c (0 or -1)

	imull	VAR_INVERSE, %eax	C q = s*i

	incl	%ebx
	jnz	L(top)


	mull	PARAM_DIVISOR

	movl	(%esi), %eax		C src high
	subl	%ecx, %edx		C h -= -c

	cmpl	PARAM_DIVISOR, %eax

	jbe	L(skip_last)
deflit(FRAME_LAST,FRAME)


	subl	%edx, %eax		C s = src[i] - h
	popl	%esi		FRAME_popl()

	sbbl	%ecx, %ecx		C c (0 or -1)
	popl	%ebx		FRAME_popl()

	imull	VAR_INVERSE, %eax	C q = s*i

	mull	PARAM_DIVISOR		C h:dummy = q*d

	movl	%edx, %eax

	subl	%ecx, %eax

	ret


C When high<divisor can skip last step.

L(skip_last):
deflit(`FRAME',FRAME_LAST)
	C eax	src high
	C ebx
	C ecx
	C edx	r
	C esi

	subl	%eax, %edx	C r-s
	popl	%esi		FRAME_popl()

	sbbl	%eax, %eax	C -1 if underflow
	movl	PARAM_DIVISOR, %ebx

	andl	%ebx, %eax	C divisor if underflow
	popl	%ebx		FRAME_popl()

	addl	%edx, %eax	C addback if underflow

	ret


C Special case for size==1 using a division for r = c-a mod d.
C Could look for a-c<d and save a division sometimes, but that doesn't seem
C worth bothering about.

L(one_limb):
deflit(`FRAME',4)
	C eax
	C ebx	size-2 (==-1)
	C ecx
	C edx	carry
	C esi	src end
	C edi
	C ebp

	movl	%edx, %eax
	movl	PARAM_SRC, %edx

	movl	PARAM_DIVISOR, %ecx
	popl	%ebx		FRAME_popl()

	subl	(%edx), %eax		C c-a

	sbbl	%edx, %edx
	decl	%ecx			C d-1

	andl	%ecx, %edx		C b*d+c-a if c<a, or c-a if c>=a

	divl	PARAM_DIVISOR

	movl	%edx, %eax

	ret

EPILOGUE()
ASM_END()
