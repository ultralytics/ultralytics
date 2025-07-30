dnl  AMD64 mpn_sec_tabselect.

dnl  Contributed to the GNU project by Torbjörn Granlund.

dnl  Copyright 2011-2013 Free Software Foundation, Inc.

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


C	     cycles/limb          good for cpu
C AMD K8,K9	 1.5			Y
C AMD K10	 1.4
C AMD bd1	 2.64
C AMD bobcat	 2.15			Y
C Intel P4	 4
C Intel core2	 1.38
C Intel NHM	 1.75
C Intel SBR	 1.25
C Intel atom	 2.5			Y
C VIA nano	 1.75			Y

C NOTES
C  * This has not been tuned for any specific processor.  Its speed should not
C    be too bad, though.
C  * Using SSE2/AVX2 could result in many-fold speedup.
C  * WORKS FOR n mod 4 = 0 ONLY!

C mpn_sec_tabselect (mp_limb_t *rp, mp_limb_t *tp, mp_size_t n, mp_size_t nents, mp_size_t which)
define(`rp',     `%rdi')
define(`tp',     `%rsi')
define(`n',      `%rdx')
define(`nents',  `%rcx')
define(`which',  `%r8')

define(`i',      `%rbp')
define(`j',      `%r9')

C rax  rbx  rcx  rdx  rdi  rsi  rbp   r8   r9  r10  r11  r12  r13  r14  r15
C          nents  n   rp   tab   i   which j    *    *    *    *    *    *

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_sec_tabselect)
	FUNC_ENTRY(4)
IFDOS(`	mov	56(%rsp), %r8d	')

	push	%rbx
	push	%rbp
	push	%r12
	push	%r13
	push	%r14
	push	%r15

	mov	n, j
	add	$-4, j
	js	L(outer_end)

L(outer_top):
	mov	nents, i
	push	tp
	xor	R32(%r12), R32(%r12)
	xor	R32(%r13), R32(%r13)
	xor	R32(%r14), R32(%r14)
	xor	R32(%r15), R32(%r15)
	mov	which, %rbx

	ALIGN(16)
L(top):	sub	$1, %rbx
	sbb	%rax, %rax
	mov	0(tp), %r10
	mov	8(tp), %r11
	and	%rax, %r10
	and	%rax, %r11
	or	%r10, %r12
	or	%r11, %r13
	mov	16(tp), %r10
	mov	24(tp), %r11
	and	%rax, %r10
	and	%rax, %r11
	or	%r10, %r14
	or	%r11, %r15
	lea	(tp,n,8), tp
	add	$-1, i
	jne	L(top)

	mov	%r12, 0(rp)
	mov	%r13, 8(rp)
	mov	%r14, 16(rp)
	mov	%r15, 24(rp)
	pop	tp
	lea	32(tp), tp
	lea	32(rp), rp
	add	$-4, j
	jns	L(outer_top)
L(outer_end):

	test	$2, R8(n)
	jz	L(b0x)
L(b1x):	mov	nents, i
	push	tp
	xor	R32(%r12), R32(%r12)
	xor	R32(%r13), R32(%r13)
	mov	which, %rbx
	ALIGN(16)
L(tp2):	sub	$1, %rbx
	sbb	%rax, %rax
	mov	0(tp), %r10
	mov	8(tp), %r11
	and	%rax, %r10
	and	%rax, %r11
	or	%r10, %r12
	or	%r11, %r13
	lea	(tp,n,8), tp
	add	$-1, i
	jne	L(tp2)
	mov	%r12, 0(rp)
	mov	%r13, 8(rp)
	pop	tp
	lea	16(tp), tp
	lea	16(rp), rp

L(b0x):	test	$1, R8(n)
	jz	L(b00)
L(b01):	mov	nents, i
	xor	R32(%r12), R32(%r12)
	mov	which, %rbx
	ALIGN(16)
L(tp1):	sub	$1, %rbx
	sbb	%rax, %rax
	mov	0(tp), %r10
	and	%rax, %r10
	or	%r10, %r12
	lea	(tp,n,8), tp
	add	$-1, i
	jne	L(tp1)
	mov	%r12, 0(rp)

L(b00):	pop	%r15
	pop	%r14
	pop	%r13
	pop	%r12
	pop	%rbp
	pop	%rbx
	FUNC_EXIT()
	ret
EPILOGUE()
