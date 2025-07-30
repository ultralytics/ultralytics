dnl  AMD64 mpn_mod_34lsub1 -- remainder modulo 2^48-1.

dnl  Copyright 2000-2002, 2004, 2005, 2007, 2009-2012 Free Software Foundation,
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


C	    cycles/limb
C AMD K8,K9	 0.67	   0.583 is possible with zero-reg instead of $0, 4-way
C AMD K10	 0.67	   this seems hard to beat
C AMD bd1	 1
C AMD bobcat	 1.07
C Intel P4	 7.35	   terrible, use old code
C Intel core2	 1.25	   1+epsilon with huge unrolling
C Intel NHM	 1.15	   this seems hard to beat
C Intel SBR	 0.93
C Intel atom	 2.5
C VIA nano	 1.25	   this seems hard to beat

C INPUT PARAMETERS
define(`ap',	%rdi)
define(`n',	%rsi)

C mp_limb_t mpn_mod_34lsub1 (mp_srcptr up, mp_size_t n)

C TODO
C  * Review feed-in and wind-down code.

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(32)
PROLOGUE(mpn_mod_34lsub1)
	FUNC_ENTRY(2)

	mov	$0x0000FFFFFFFFFFFF, %r11

	mov	(ap), %rax

	cmp	$2, %rsi
	ja	L(gt2)

	jb	L(one)

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
L(one):	FUNC_EXIT()
	ret


C Don't change this, the wind-down code is not able to handle greater values
define(UNROLL,3)

L(gt2):	mov	8(ap), %rcx
	mov	16(ap), %rdx
	xor	%r9, %r9
	add	$24, ap
	sub	$eval(UNROLL*3+3), %rsi
	jc	L(end)
	ALIGN(16)
L(top):
	add	(ap), %rax
	adc	8(ap), %rcx
	adc	16(ap), %rdx
	adc	$0, %r9
forloop(i,1,UNROLL-1,`dnl
	add	eval(i*24)(ap), %rax
	adc	eval(i*24+8)(ap), %rcx
	adc	eval(i*24+16)(ap), %rdx
	adc	$0, %r9
')dnl
	add	$eval(UNROLL*24), ap
	sub	$eval(UNROLL*3), %rsi
	jnc	L(top)

L(end):
	lea	L(tab)(%rip), %r8
ifdef(`PIC',
`	movslq	36(%r8,%rsi,4), %r10
	add	%r10, %r8
	jmp	*%r8
',`
	jmp	*72(%r8,%rsi,8)
')
	JUMPTABSECT
	ALIGN(8)
L(tab):	JMPENT(	L(0), L(tab))
	JMPENT(	L(1), L(tab))
	JMPENT(	L(2), L(tab))
	JMPENT(	L(3), L(tab))
	JMPENT(	L(4), L(tab))
	JMPENT(	L(5), L(tab))
	JMPENT(	L(6), L(tab))
	JMPENT(	L(7), L(tab))
	JMPENT(	L(8), L(tab))
	TEXT

L(6):	add	(ap), %rax
	adc	8(ap), %rcx
	adc	16(ap), %rdx
	adc	$0, %r9
	add	$24, ap
L(3):	add	(ap), %rax
	adc	8(ap), %rcx
	adc	16(ap), %rdx
	jmp	L(cj1)

L(7):	add	(ap), %rax
	adc	8(ap), %rcx
	adc	16(ap), %rdx
	adc	$0, %r9
	add	$24, ap
L(4):	add	(ap), %rax
	adc	8(ap), %rcx
	adc	16(ap), %rdx
	adc	$0, %r9
	add	$24, ap
L(1):	add	(ap), %rax
	adc	$0, %rcx
	jmp	L(cj2)

L(8):	add	(ap), %rax
	adc	8(ap), %rcx
	adc	16(ap), %rdx
	adc	$0, %r9
	add	$24, ap
L(5):	add	(ap), %rax
	adc	8(ap), %rcx
	adc	16(ap), %rdx
	adc	$0, %r9
	add	$24, ap
L(2):	add	(ap), %rax
	adc	8(ap), %rcx

L(cj2):	adc	$0, %rdx
L(cj1):	adc	$0, %r9
L(0):	add	%r9, %rax
	adc	$0, %rcx
	adc	$0, %rdx
	adc	$0, %rax

	mov	%rax, %rdi		C 0mod3
	shr	$48, %rax		C 0mod3 high

	and	%r11, %rdi		C 0mod3 low
	mov	R32(%rcx), R32(%r10)	C 1mod3

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
