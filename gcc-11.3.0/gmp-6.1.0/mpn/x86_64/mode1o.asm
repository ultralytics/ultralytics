dnl  AMD64 mpn_modexact_1_odd -- Hensel norm remainder.

dnl  Copyright 2000-2006, 2011, 2012 Free Software Foundation, Inc.

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


C	     cycles/limb
C AMD K8,K9	10
C AMD K10	10
C Intel P4	33
C Intel core2	13
C Intel corei	14.5
C Intel atom	35
C VIA nano	 ?


C The dependent chain in the main loop is
C
C                            cycles
C	sub	%rdx, %rax	1
C	imul	%r9, %rax	4
C	mul	%r8		5
C			      ----
C       total		       10
C
C The mov load from src seems to need to be scheduled back before the jz to
C achieve this speed, out-of-order execution apparently can't completely hide
C the latency otherwise.
C
C The l=src[i]-cbit step is rotated back too, since that allows us to avoid it
C for the first iteration (where there's no cbit).
C
C The code alignment used (32-byte) for the loop also seems necessary.  Without
C that the non-PIC case has adc crossing the 0x60 offset, apparently making it
C run at 11 cycles instead of 10.


ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(32)
PROLOGUE(mpn_modexact_1_odd)
	FUNC_ENTRY(3)
	mov	$0, R32(%rcx)
IFDOS(`	jmp	L(ent)		')

PROLOGUE(mpn_modexact_1c_odd)
	FUNC_ENTRY(4)
L(ent):
	C rdi	src
	C rsi	size
	C rdx	divisor
	C rcx	carry

	mov	%rdx, %r8		C d
	shr	R32(%rdx)		C d/2

	LEA(	binvert_limb_table, %r9)

	and	$127, R32(%rdx)
	mov	%rcx, %r10		C initial carry

	movzbl	(%r9,%rdx), R32(%rdx)	C inv 8 bits

	mov	(%rdi), %rax		C src[0]
	lea	(%rdi,%rsi,8), %r11	C src end
	mov	%r8, %rdi		C d, made available to imull

	lea	(%rdx,%rdx), R32(%rcx)	C 2*inv
	imul	R32(%rdx), R32(%rdx)	C inv*inv

	neg	%rsi			C -size

	imul	R32(%rdi), R32(%rdx)	C inv*inv*d

	sub	R32(%rdx), R32(%rcx)	C inv = 2*inv - inv*inv*d, 16 bits

	lea	(%rcx,%rcx), R32(%rdx)	C 2*inv
	imul	R32(%rcx), R32(%rcx)	C inv*inv

	imul	R32(%rdi), R32(%rcx)	C inv*inv*d

	sub	R32(%rcx), R32(%rdx)	C inv = 2*inv - inv*inv*d, 32 bits
	xor	R32(%rcx), R32(%rcx)	C initial cbit

	lea	(%rdx,%rdx), %r9	C 2*inv
	imul	%rdx, %rdx		C inv*inv

	imul	%r8, %rdx		C inv*inv*d

	sub	%rdx, %r9		C inv = 2*inv - inv*inv*d, 64 bits
	mov	%r10, %rdx		C initial climb

	ASSERT(e,`	C d*inv == 1 mod 2^64
	mov	%r8, %r10
	imul	%r9, %r10
	cmp	$1, %r10')

	inc	%rsi
	jz	L(one)


	ALIGN(16)
L(top):
	C rax	l = src[i]-cbit
	C rcx	new cbit, 0 or 1
	C rdx	climb, high of last product
	C rsi	counter, limbs, negative
	C rdi
	C r8	divisor
	C r9	inverse
	C r11	src end ptr

	sub	%rdx, %rax		C l = src[i]-cbit - climb

	adc	$0, %rcx		C more cbit
	imul	%r9, %rax		C q = l * inverse

	mul	%r8			C climb = high (q * d)

	mov	(%r11,%rsi,8), %rax	C src[i+1]
	sub	%rcx, %rax		C next l = src[i+1] - cbit
	setc	R8(%rcx)		C new cbit

	inc	%rsi
	jnz	L(top)


L(one):
	sub	%rdx, %rax		C l = src[i]-cbit - climb

	adc	$0, %rcx		C more cbit
	imul	%r9, %rax		C q = l * inverse

	mul	%r8			C climb = high (q * d)

	lea	(%rcx,%rdx), %rax	C climb+cbit
	FUNC_EXIT()
	ret

EPILOGUE(mpn_modexact_1c_odd)
EPILOGUE(mpn_modexact_1_odd)
