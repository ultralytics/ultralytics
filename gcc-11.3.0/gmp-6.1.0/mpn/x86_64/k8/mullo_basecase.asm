dnl  AMD64 mpn_mullo_basecase.

dnl  Contributed to the GNU project by Torbjorn Granlund.

dnl  Copyright 2008, 2009, 2011, 2012 Free Software Foundation, Inc.

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

C The inner loops of this code are the result of running a code generation and
C optimisation tool suite written by David Harvey and Torbjorn Granlund.

C NOTES
C   * There is a major stupidity in that we call mpn_mul_1 initially, for a
C     large trip count.  Instead, we should start with mul_2 for any operand
C     size congruence class.
C   * Stop iterating addmul_2 earlier, falling into straight-line triangle code
C     for the last 2-3 iterations.
C   * Perhaps implement n=4 special code.
C   * The reload of the outer loop jump address hurts branch prediction.
C   * The addmul_2 loop ends with an MUL whose high part is not used upon loop
C     exit.

C INPUT PARAMETERS
define(`rp',	   `%rdi')
define(`up',	   `%rsi')
define(`vp_param', `%rdx')
define(`n',	   `%rcx')

define(`vp',	`%r11')
define(`outer_addr', `%r8')
define(`j',	`%r9')
define(`v0',	`%r13')
define(`v1',	`%r14')
define(`w0',	`%rbx')
define(`w1',	`%r15')
define(`w2',	`%rbp')
define(`w3',	`%r10')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_mullo_basecase)
	FUNC_ENTRY(4)
	cmp	$4, n
	jge	L(gen)
	mov	(up), %rax		C u0
	mov	(vp_param), %r8		C v0

	lea	L(tab)(%rip), %r9
ifdef(`PIC',
`	movslq	(%r9,%rcx,4), %r10
	add	%r10, %r9
	jmp	*%r9
',`
	jmp	*(%r9,n,8)
')
	JUMPTABSECT
	ALIGN(8)
L(tab):	JMPENT(	L(tab), L(tab))			C not allowed
	JMPENT(	L(1), L(tab))			C 1
	JMPENT(	L(2), L(tab))			C 2
	JMPENT(	L(3), L(tab))			C 3
dnl	JMPENT(	L(0m4), L(tab))			C 4
dnl	JMPENT(	L(1m4), L(tab))			C 5
dnl	JMPENT(	L(2m4), L(tab))			C 6
dnl	JMPENT(	L(3m4), L(tab))			C 7
dnl	JMPENT(	L(0m4), L(tab))			C 8
dnl	JMPENT(	L(1m4), L(tab))			C 9
dnl	JMPENT(	L(2m4), L(tab))			C 10
dnl	JMPENT(	L(3m4), L(tab))			C 11
	TEXT

L(1):	imul	%r8, %rax
	mov	%rax, (rp)
	FUNC_EXIT()
	ret

L(2):	mov	8(vp_param), %r11
	imul	%rax, %r11		C u0 x v1
	mul	%r8			C u0 x v0
	mov	%rax, (rp)
	imul	8(up), %r8		C u1 x v0
	lea	(%r11, %rdx), %rax
	add	%r8, %rax
	mov	%rax, 8(rp)
	FUNC_EXIT()
	ret

L(3):	mov	8(vp_param), %r9	C v1
	mov	16(vp_param), %r11
	mul	%r8			C u0 x v0 -> <r1,r0>
	mov	%rax, (rp)		C r0
	mov	(up), %rax		C u0
	mov	%rdx, %rcx		C r1
	mul	%r9			C u0 x v1 -> <r2,r1>
	imul	8(up), %r9		C u1 x v1 -> r2
	mov	16(up), %r10
	imul	%r8, %r10		C u2 x v0 -> r2
	add	%rax, %rcx
	adc	%rdx, %r9
	add	%r10, %r9
	mov	8(up), %rax		C u1
	mul	%r8			C u1 x v0 -> <r2,r1>
	add	%rax, %rcx
	adc	%rdx, %r9
	mov	%r11, %rax
	imul	(up), %rax		C u0 x v2 -> r2
	add	%rax, %r9
	mov	%rcx, 8(rp)
	mov	%r9, 16(rp)
	FUNC_EXIT()
	ret

L(0m4):
L(1m4):
L(2m4):
L(3m4):
L(gen):	push	%rbx
	push	%rbp
	push	%r13
	push	%r14
	push	%r15

	mov	(up), %rax
	mov	(vp_param), v0
	mov	vp_param, vp

	lea	(rp,n,8), rp
	lea	(up,n,8), up
	neg	n

	mul	v0

	test	$1, R8(n)
	jz	L(mul_2)

L(mul_1):
	lea	-8(rp), rp
	lea	-8(up), up
	test	$2, R8(n)
	jnz	L(mul_1_prologue_3)

L(mul_1_prologue_2):		C n = 7, 11, 15, ...
	lea	-1(n), j
	lea	L(addmul_outer_1)(%rip), outer_addr
	mov	%rax, w0
	mov	%rdx, w1
	xor	R32(w2), R32(w2)
	xor	R32(w3), R32(w3)
	mov	16(up,n,8), %rax
	jmp	L(mul_1_entry_2)

L(mul_1_prologue_3):		C n = 5, 9, 13, ...
	lea	1(n), j
	lea	L(addmul_outer_3)(%rip), outer_addr
	mov	%rax, w2
	mov	%rdx, w3
	xor	R32(w0), R32(w0)
	jmp	L(mul_1_entry_0)

	ALIGN(16)
L(mul_1_top):
	mov	w0, -16(rp,j,8)
	add	%rax, w1
	mov	(up,j,8), %rax
	adc	%rdx, w2
	xor	R32(w0), R32(w0)
	mul	v0
	mov	w1, -8(rp,j,8)
	add	%rax, w2
	adc	%rdx, w3
L(mul_1_entry_0):
	mov	8(up,j,8), %rax
	mul	v0
	mov	w2, (rp,j,8)
	add	%rax, w3
	adc	%rdx, w0
	mov	16(up,j,8), %rax
	mul	v0
	mov	w3, 8(rp,j,8)
	xor	R32(w2), R32(w2)	C zero
	mov	w2, w3			C zero
	add	%rax, w0
	mov	24(up,j,8), %rax
	mov	w2, w1			C zero
	adc	%rdx, w1
L(mul_1_entry_2):
	mul	v0
	add	$4, j
	js	L(mul_1_top)

	mov	w0, -16(rp)
	add	%rax, w1
	mov	w1, -8(rp)
	adc	%rdx, w2

	imul	(up), v0
	add	v0, w2
	mov	w2, (rp)

	add	$1, n
	jz	L(ret)

	mov	8(vp), v0
	mov	16(vp), v1

	lea	16(up), up
	lea	8(vp), vp
	lea	24(rp), rp

	jmp	*outer_addr


L(mul_2):
	mov	8(vp), v1
	test	$2, R8(n)
	jz	L(mul_2_prologue_3)

	ALIGN(16)
L(mul_2_prologue_1):
	lea	0(n), j
	mov	%rax, w3
	mov	%rdx, w0
	xor	R32(w1), R32(w1)
	mov	(up,n,8), %rax
	lea	L(addmul_outer_3)(%rip), outer_addr
	jmp	L(mul_2_entry_1)

	ALIGN(16)
L(mul_2_prologue_3):
	lea	2(n), j
	mov	$0, R32(w3)
	mov	%rax, w1
	mov	(up,n,8), %rax
	mov	%rdx, w2
	lea	L(addmul_outer_1)(%rip), outer_addr
	jmp	L(mul_2_entry_3)

	ALIGN(16)
L(mul_2_top):
	mov	-32(up,j,8), %rax
	mul	v1
	add	%rax, w0
	adc	%rdx, w1
	mov	-24(up,j,8), %rax
	xor	R32(w2), R32(w2)
	mul	v0
	add	%rax, w0
	mov	-24(up,j,8), %rax
	adc	%rdx, w1
	adc	$0, R32(w2)
	mul	v1
	add	%rax, w1
	mov	w0, -24(rp,j,8)
	adc	%rdx, w2
	mov	-16(up,j,8), %rax
	mul	v0
	mov	$0, R32(w3)
	add	%rax, w1
	adc	%rdx, w2
	mov	-16(up,j,8), %rax
	adc	$0, R32(w3)
L(mul_2_entry_3):
	mov	$0, R32(w0)
	mov	w1, -16(rp,j,8)
	mul	v1
	add	%rax, w2
	mov	-8(up,j,8), %rax
	adc	%rdx, w3
	mov	$0, R32(w1)
	mul	v0
	add	%rax, w2
	mov	-8(up,j,8), %rax
	adc	%rdx, w3
	adc	R32(w1), R32(w0)
	mul	v1
	add	%rax, w3
	mov	w2, -8(rp,j,8)
	adc	%rdx, w0
	mov	(up,j,8), %rax
	mul	v0
	add	%rax, w3
	adc	%rdx, w0
	adc	$0, R32(w1)
L(mul_2_entry_1):
	add	$4, j
	mov	w3, -32(rp,j,8)
	js	L(mul_2_top)

	imul	-16(up), v1
	add	v1, w0
	imul	-8(up), v0
	add	v0, w0
	mov	w0, -8(rp)

	add	$2, n
	jz	L(ret)

	mov	16(vp), v0
	mov	24(vp), v1

	lea	16(vp), vp
	lea	16(rp), rp

	jmp	*outer_addr


L(addmul_outer_1):
	lea	-2(n), j
	mov	-16(up,n,8), %rax
	mul	v0
	mov	%rax, w3
	mov	-16(up,n,8), %rax
	mov	%rdx, w0
	xor	R32(w1), R32(w1)
	lea	L(addmul_outer_3)(%rip), outer_addr
	jmp	L(addmul_entry_1)

L(addmul_outer_3):
	lea	0(n), j
	mov	-16(up,n,8), %rax
	xor	R32(w3), R32(w3)
	mul	v0
	mov	%rax, w1
	mov	-16(up,n,8), %rax
	mov	%rdx, w2
	lea	L(addmul_outer_1)(%rip), outer_addr
	jmp	L(addmul_entry_3)

	ALIGN(16)
L(addmul_top):
	add	w3, -32(rp,j,8)
	adc	%rax, w0
	mov	-24(up,j,8), %rax
	adc	%rdx, w1
	xor	R32(w2), R32(w2)
	mul	v0
	add	%rax, w0
	mov	-24(up,j,8), %rax
	adc	%rdx, w1
	adc	R32(w2), R32(w2)
	mul	v1
	xor	R32(w3), R32(w3)
	add	w0, -24(rp,j,8)
	adc	%rax, w1
	mov	-16(up,j,8), %rax
	adc	%rdx, w2
	mul	v0
	add	%rax, w1
	mov	-16(up,j,8), %rax
	adc	%rdx, w2
	adc	$0, R32(w3)
L(addmul_entry_3):
	mul	v1
	add	w1, -16(rp,j,8)
	adc	%rax, w2
	mov	-8(up,j,8), %rax
	adc	%rdx, w3
	mul	v0
	xor	R32(w0), R32(w0)
	add	%rax, w2
	adc	%rdx, w3
	mov	$0, R32(w1)
	mov	-8(up,j,8), %rax
	adc	R32(w1), R32(w0)
	mul	v1
	add	w2, -8(rp,j,8)
	adc	%rax, w3
	adc	%rdx, w0
	mov	(up,j,8), %rax
	mul	v0
	add	%rax, w3
	mov	(up,j,8), %rax
	adc	%rdx, w0
	adc	$0, R32(w1)
L(addmul_entry_1):
	mul	v1
	add	$4, j
	js	L(addmul_top)

	add	w3, -32(rp)
	adc	%rax, w0

	imul	-24(up), v0
	add	v0, w0
	add	w0, -24(rp)

	add	$2, n
	jns	L(ret)

	lea	16(vp), vp

	mov	(vp), v0
	mov	8(vp), v1

	lea	-16(up), up

	jmp	*outer_addr

L(ret):	pop	%r15
	pop	%r14
	pop	%r13
	pop	%rbp
	pop	%rbx
	FUNC_EXIT()
	ret
EPILOGUE()
