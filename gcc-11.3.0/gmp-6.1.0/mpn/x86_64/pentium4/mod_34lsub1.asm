dnl  AMD64 mpn_mod_34lsub1 -- remainder modulo 2^48-1.

dnl  Copyright 2000-2002, 2004, 2005, 2007, 2010-2012 Free Software Foundation,
dnl  Inc.

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
C AMD K8,K9	 1.0
C AMD K10	 1.12
C Intel P4	 3.25
C Intel core2	 1.5
C Intel corei	 1.5
C Intel atom	 2.5
C VIA nano	 1.75


C INPUT PARAMETERS
define(`ap',	%rdi)
define(`n',	%rsi)

C mp_limb_t mpn_mod_34lsub1 (mp_srcptr up, mp_size_t n)

C TODO
C  * Review feed-in and wind-down code.  In particular, try to avoid adc and
C    sbb to placate Pentium4.
C  * It seems possible to reach 2.67 c/l by using a cleaner 6-way unrolling,
C    without the dual loop exits.

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(32)
PROLOGUE(mpn_mod_34lsub1)
	FUNC_ENTRY(2)

	mov	$0x0000FFFFFFFFFFFF, %r11

	sub	$2, %rsi
	ja	L(gt2)

	mov	(ap), %rax
	nop
	jb	L(1)

	mov	8(ap), %rsi
	mov	%rax, %rdx
	shr	$48, %rax		C src[0] low

	and	%r11, %rdx		C src[0] high
	add	%rdx, %rax
	mov	R32(%rsi), R32(%rdx)

	shr	$32, %rsi		C src[1] high
	add	%rsi, %rax

	shl	$16, %rdx		C src[1] low
	add	%rdx, %rax

L(1):	FUNC_EXIT()
	ret


	ALIGN(16)
L(gt2):	xor	R32(%rax), R32(%rax)
	xor	R32(%rcx), R32(%rcx)
	xor	R32(%rdx), R32(%rdx)
	xor	%r8, %r8
	xor	%r9, %r9
	xor	%r10, %r10

L(top):	add	(ap), %rax
	adc	$0, %r10
	add	8(ap), %rcx
	adc	$0, %r8
	add	16(ap), %rdx
	adc	$0, %r9

	sub	$3, %rsi
	jng	L(end)

	add	24(ap), %rax
	adc	$0, %r10
	add	32(ap), %rcx
	adc	$0, %r8
	add	40(ap), %rdx
	lea	48(ap), ap
	adc	$0, %r9

	sub	$3, %rsi
	jg	L(top)


	add	$-24, ap
L(end):	add	%r9, %rax
	adc	%r10, %rcx
	adc	%r8, %rdx

	inc	%rsi
	mov	$0x1, R32(%r10)
	js	L(combine)

	mov	$0x10000, R32(%r10)
	adc	24(ap), %rax
	dec	%rsi
	js	L(combine)

	adc	32(ap), %rcx
	mov	$0x100000000, %r10

L(combine):
	sbb	%rsi, %rsi		C carry
	mov	%rax, %rdi		C 0mod3
	shr	$48, %rax		C 0mod3 high

	and	%r10, %rsi		C carry masked
	and	%r11, %rdi		C 0mod3 low
	mov	R32(%rcx), R32(%r10)	C 1mod3

	add	%rsi, %rax		C apply carry
	shr	$32, %rcx		C 1mod3 high

	add	%rdi, %rax		C apply 0mod3 low
	movzwl	%dx, R32(%rdi)		C 2mod3
	shl	$16, %r10		C 1mod3 low

	add	%rcx, %rax		C apply 1mod3 high
	shr	$16, %rdx		C 2mod3 high

	add	%r10, %rax		C apply 1mod3 low
	shl	$32, %rdi		C 2mod3 low

	add	%rdx, %rax		C apply 2mod3 high
	add	%rdi, %rax		C apply 2mod3 low

	FUNC_EXIT()
	ret
EPILOGUE()
