dnl  AMD64 mpn_mul_1 optimised for Intel Broadwell.

dnl  Copyright 2015 Free Software Foundation, Inc.

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
C AMD K8,K9	n/a
C AMD K10	n/a
C AMD bull	n/a
C AMD pile	n/a
C AMD steam	n/a
C AMD bobcat	n/a
C AMD jaguar	n/a
C Intel P4	n/a
C Intel core2	n/a
C Intel NHM	n/a
C Intel SBR	n/a
C Intel IBR	n/a
C Intel HWL	 1.68
C Intel BWL	 1.69
C Intel atom	n/a
C Intel SLM	n/a
C VIA nano	n/a

C The loop of this code is the result of running a code generation and
C optimisation tool suite written by David Harvey and Torbjorn Granlund.

C TODO
C  * Put an initial mulx before switching, targeting some free registers.
C  * Tune feed-in code.
C  * Trim nop execution after L(f2).
C  * Port to DOS64, not forgetting nop execution.

define(`rp',      `%rdi')   C rcx
define(`up',      `%rsi')   C rdx
define(`n_param', `%rdx')   C r8
define(`v0_param',`%rcx')   C r9

define(`n',       `%rcx')

dnl ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

dnl IFDOS(`	define(`up', ``%rsi'')	') dnl
dnl IFDOS(`	define(`rp', ``%rcx'')	') dnl
dnl IFDOS(`	define(`vl', ``%r9'')	') dnl
dnl IFDOS(`	define(`r9', ``rdi'')	') dnl
dnl IFDOS(`	define(`n',  ``%r8'')	') dnl
dnl IFDOS(`	define(`r8', ``r11'')	') dnl

ASM_START()
	TEXT
	ALIGN(32)
PROLOGUE(mpn_mul_1)

	mov	v0_param, %r10
	mov	n_param, n
	mov	R32(n_param), R32(%r8)
	shr	$3, n
	and	$7, R32(%r8)		C clear OF, CF as side-effect
	mov	%r10, %rdx
	lea	L(tab)(%rip), %r10
ifdef(`PIC',
`	movslq	(%r10,%r8,4), %r8
	lea	(%r8, %r10), %r10
	jmp	*%r10
',`
	jmp	*(%r10,%r8,8)
')
	JUMPTABSECT
	ALIGN(8)
L(tab):	JMPENT(	L(f0), L(tab))
	JMPENT(	L(f1), L(tab))
	JMPENT(	L(f2), L(tab))
	JMPENT(	L(f3), L(tab))
	JMPENT(	L(f4), L(tab))
	JMPENT(	L(f5), L(tab))
	JMPENT(	L(f6), L(tab))
	JMPENT(	L(f7), L(tab))
	TEXT

L(f0):	mulx(	(up), %r10, %r8)
	lea	56(up), up
	lea	-8(rp), rp
	jmp	L(b0)

L(f3):	mulx(	(up), %r9, %rax)
	lea	16(up), up
	lea	16(rp), rp
	inc	n
	jmp	L(b3)

L(f4):	mulx(	(up), %r10, %r8)
	lea	24(up), up
	lea	24(rp), rp
	inc	n
	jmp	L(b4)

L(f5):	mulx(	(up), %r9, %rax)
	lea	32(up), up
	lea	32(rp), rp
	inc	n
	jmp	L(b5)

L(f6):	mulx(	(up), %r10, %r8)
	lea	40(up), up
	lea	40(rp), rp
	inc	n
	jmp	L(b6)

L(f7):	mulx(	(up), %r9, %rax)
	lea	48(up), up
	lea	48(rp), rp
	inc	n
	jmp	L(b7)

L(f1):	mulx(	(up), %r9, %rax)
	test	n, n
	jnz	L(b1)
L(1):	mov	%r9, (rp)
	ret

L(f2):	mulx(	(up), %r10, %r8)
	lea	8(up), up
	lea	8(rp), rp
	mulx(	(up), %r9, %rax)
	test	n, n
	jz	L(end)

	ALIGN(32)
L(top):	mov	%r10, -8(rp)
	adc	%r8, %r9
L(b1):	mulx(	8,(up), %r10, %r8)
	adc	%rax, %r10
	lea	64(up), up
	mov	%r9, (rp)
L(b0):	mov	%r10, 8(rp)
	mulx(	-48,(up), %r9, %rax)
	lea	64(rp), rp
	adc	%r8, %r9
L(b7):	mulx(	-40,(up), %r10, %r8)
	mov	%r9, -48(rp)
	adc	%rax, %r10
L(b6):	mov	%r10, -40(rp)
	mulx(	-32,(up), %r9, %rax)
	adc	%r8, %r9
L(b5):	mulx(	-24,(up), %r10, %r8)
	mov	%r9, -32(rp)
	adc	%rax, %r10
L(b4):	mulx(	-16,(up), %r9, %rax)
	mov	%r10, -24(rp)
	adc	%r8, %r9
L(b3):	mulx(	-8,(up), %r10, %r8)
	adc	%rax, %r10
	mov	%r9, -16(rp)
	dec	n
	mulx(	(up), %r9, %rax)
	jnz	L(top)

L(end):	mov	%r10, -8(rp)
	adc	%r8, %r9
	mov	%r9, (rp)
	adc	%rcx, %rax
	ret
EPILOGUE()
ASM_END()
