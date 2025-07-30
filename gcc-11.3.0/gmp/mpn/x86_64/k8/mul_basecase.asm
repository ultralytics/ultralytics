dnl  AMD64 mpn_mul_basecase.

dnl  Contributed to the GNU project by Torbjorn Granlund and David Harvey.

dnl  Copyright 2008, 2012 Free Software Foundation, Inc.

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
C AMD K8,K9	 2.375
C AMD K10	 2.375
C Intel P4	15-16
C Intel core2	 4.45
C Intel corei	 4.35
C Intel atom	 ?
C VIA nano	 4.5

C The inner loops of this code are the result of running a code generation and
C optimization tool suite written by David Harvey and Torbjorn Granlund.

C TODO
C  * Use fewer registers.  (how??? I can't see it -- david)
C  * Avoid some "mov $0,r" and instead use "xor r,r".
C  * Can the top of each L(addmul_outer_n) prologue be folded into the
C    mul_1/mul_2 prologues, saving a LEA (%rip)? It would slow down the
C    case where vn = 1 or 2; is it worth it?

C INPUT PARAMETERS
define(`rp',      `%rdi')
define(`up',      `%rsi')
define(`un_param',`%rdx')
define(`vp',      `%rcx')
define(`vn',      `%r8')

define(`v0', `%r12')
define(`v1', `%r9')

define(`w0', `%rbx')
define(`w1', `%r15')
define(`w2', `%rbp')
define(`w3', `%r10')

define(`n',  `%r11')
define(`outer_addr', `%r14')
define(`un',  `%r13')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_mul_basecase)
	FUNC_ENTRY(4)
IFDOS(`	mov	56(%rsp), %r8d	')
	push	%rbx
	push	%rbp
	push	%r12
	push	%r13
	push	%r14
	push	%r15

	xor	R32(un), R32(un)
	mov	(up), %rax
	mov	(vp), v0

	sub	un_param, un		C rdx used by mul
	mov	un, n
	mov	R32(un_param), R32(w0)

	lea	(rp,un_param,8), rp
	lea	(up,un_param,8), up

	mul	v0

	test	$1, R8(vn)
	jz	L(mul_2)

C ===========================================================
C     mul_1 for vp[0] if vn is odd

L(mul_1):
	and	$3, R32(w0)
	jz	L(mul_1_prologue_0)
	cmp	$2, R32(w0)
	jc	L(mul_1_prologue_1)
	jz	L(mul_1_prologue_2)

L(mul_1_prologue_3):
	add	$-1, n
	lea	L(addmul_outer_3)(%rip), outer_addr
	mov	%rax, w3
	mov	%rdx, w0
	jmp	L(mul_1_entry_3)

L(mul_1_prologue_0):
	mov	%rax, w2
	mov	%rdx, w3		C note: already w0 == 0
	lea	L(addmul_outer_0)(%rip), outer_addr
	jmp	L(mul_1_entry_0)

L(mul_1_prologue_1):
	cmp	$-1, un
	jne	2f
	mov	%rax, -8(rp)
	mov	%rdx, (rp)
	jmp	L(ret)
2:	add	$1, n
	lea	L(addmul_outer_1)(%rip), outer_addr
	mov	%rax, w1
	mov	%rdx, w2
	xor	R32(w3), R32(w3)
	mov	(up,n,8), %rax
	jmp	L(mul_1_entry_1)

L(mul_1_prologue_2):
	add	$-2, n
	lea	L(addmul_outer_2)(%rip), outer_addr
	mov	%rax, w0
	mov	%rdx, w1
	mov	24(up,n,8), %rax
	xor	R32(w2), R32(w2)
	xor	R32(w3), R32(w3)
	jmp	L(mul_1_entry_2)


	C this loop is 10 c/loop = 2.5 c/l on K8, for all up/rp alignments

	ALIGN(16)
L(mul_1_top):
	mov	w0, -16(rp,n,8)
	add	%rax, w1
	mov	(up,n,8), %rax
	adc	%rdx, w2
L(mul_1_entry_1):
	xor	R32(w0), R32(w0)
	mul	v0
	mov	w1, -8(rp,n,8)
	add	%rax, w2
	adc	%rdx, w3
L(mul_1_entry_0):
	mov	8(up,n,8), %rax
	mul	v0
	mov	w2, (rp,n,8)
	add	%rax, w3
	adc	%rdx, w0
L(mul_1_entry_3):
	mov	16(up,n,8), %rax
	mul	v0
	mov	w3, 8(rp,n,8)
	xor	R32(w2), R32(w2)	C zero
	mov	w2, w3			C zero
	add	%rax, w0
	mov	24(up,n,8), %rax
	mov	w2, w1			C zero
	adc	%rdx, w1
L(mul_1_entry_2):
	mul	v0
	add	$4, n
	js	L(mul_1_top)

	mov	w0, -16(rp)
	add	%rax, w1
	mov	w1, -8(rp)
	adc	%rdx, w2
	mov	w2, (rp)

	add	$-1, vn			C vn -= 1
	jz	L(ret)

	mov	8(vp), v0
	mov	16(vp), v1

	lea	8(vp), vp		C vp += 1
	lea	8(rp), rp		C rp += 1

	jmp	*outer_addr

C ===========================================================
C     mul_2 for vp[0], vp[1] if vn is even

	ALIGN(16)
L(mul_2):
	mov	8(vp), v1

	and	$3, R32(w0)
	jz	L(mul_2_prologue_0)
	cmp	$2, R32(w0)
	jz	L(mul_2_prologue_2)
	jc	L(mul_2_prologue_1)

L(mul_2_prologue_3):
	lea	L(addmul_outer_3)(%rip), outer_addr
	add	$2, n
	mov	%rax, -16(rp,n,8)
	mov	%rdx, w2
	xor	R32(w3), R32(w3)
	xor	R32(w0), R32(w0)
	mov	-16(up,n,8), %rax
	jmp	L(mul_2_entry_3)

	ALIGN(16)
L(mul_2_prologue_0):
	add	$3, n
	mov	%rax, w0
	mov	%rdx, w1
	xor	R32(w2), R32(w2)
	mov	-24(up,n,8), %rax
	lea	L(addmul_outer_0)(%rip), outer_addr
	jmp	L(mul_2_entry_0)

	ALIGN(16)
L(mul_2_prologue_1):
	mov	%rax, w3
	mov	%rdx, w0
	xor	R32(w1), R32(w1)
	lea	L(addmul_outer_1)(%rip), outer_addr
	jmp	L(mul_2_entry_1)

	ALIGN(16)
L(mul_2_prologue_2):
	add	$1, n
	lea	L(addmul_outer_2)(%rip), outer_addr
	mov	$0, R32(w0)
	mov	$0, R32(w1)
	mov	%rax, w2
	mov	-8(up,n,8), %rax
	mov	%rdx, w3
	jmp	L(mul_2_entry_2)

	C this loop is 18 c/loop = 2.25 c/l on K8, for all up/rp alignments

	ALIGN(16)
L(mul_2_top):
	mov	-32(up,n,8), %rax
	mul	v1
	add	%rax, w0
	adc	%rdx, w1
	mov	-24(up,n,8), %rax
	xor	R32(w2), R32(w2)
	mul	v0
	add	%rax, w0
	mov	-24(up,n,8), %rax
	adc	%rdx, w1
	adc	$0, R32(w2)
L(mul_2_entry_0):
	mul	v1
	add	%rax, w1
	mov	w0, -24(rp,n,8)
	adc	%rdx, w2
	mov	-16(up,n,8), %rax
	mul	v0
	mov	$0, R32(w3)
	add	%rax, w1
	adc	%rdx, w2
	mov	-16(up,n,8), %rax
	adc	$0, R32(w3)
	mov	$0, R32(w0)
	mov	w1, -16(rp,n,8)
L(mul_2_entry_3):
	mul	v1
	add	%rax, w2
	mov	-8(up,n,8), %rax
	adc	%rdx, w3
	mov	$0, R32(w1)
	mul	v0
	add	%rax, w2
	mov	-8(up,n,8), %rax
	adc	%rdx, w3
	adc	R32(w1), R32(w0)	C adc $0, w0
L(mul_2_entry_2):
	mul	v1
	add	%rax, w3
	mov	w2, -8(rp,n,8)
	adc	%rdx, w0
	mov	(up,n,8), %rax
	mul	v0
	add	%rax, w3
	adc	%rdx, w0
	adc	$0, R32(w1)
L(mul_2_entry_1):
	add	$4, n
	mov	w3, -32(rp,n,8)
	js	L(mul_2_top)

	mov	-32(up,n,8), %rax	C FIXME: n is constant
	mul	v1
	add	%rax, w0
	mov	w0, (rp)
	adc	%rdx, w1
	mov	w1, 8(rp)

	add	$-2, vn			C vn -= 2
	jz	L(ret)

	mov	16(vp), v0
	mov	24(vp), v1

	lea	16(vp), vp		C vp += 2
	lea	16(rp), rp		C rp += 2

	jmp	*outer_addr


C ===========================================================
C     addmul_2 for remaining vp's

	C in the following prologues, we reuse un to store the
	C adjusted value of n that is reloaded on each iteration

L(addmul_outer_0):
	add	$3, un
	lea	0(%rip), outer_addr

	mov	un, n
	mov	-24(up,un,8), %rax
	mul	v0
	mov	%rax, w0
	mov	-24(up,un,8), %rax
	mov	%rdx, w1
	xor	R32(w2), R32(w2)
	jmp	L(addmul_entry_0)

L(addmul_outer_1):
	mov	un, n
	mov	(up,un,8), %rax
	mul	v0
	mov	%rax, w3
	mov	(up,un,8), %rax
	mov	%rdx, w0
	xor	R32(w1), R32(w1)
	jmp	L(addmul_entry_1)

L(addmul_outer_2):
	add	$1, un
	lea	0(%rip), outer_addr

	mov	un, n
	mov	-8(up,un,8), %rax
	mul	v0
	xor	R32(w0), R32(w0)
	mov	%rax, w2
	xor	R32(w1), R32(w1)
	mov	%rdx, w3
	mov	-8(up,un,8), %rax
	jmp	L(addmul_entry_2)

L(addmul_outer_3):
	add	$2, un
	lea	0(%rip), outer_addr

	mov	un, n
	mov	-16(up,un,8), %rax
	xor	R32(w3), R32(w3)
	mul	v0
	mov	%rax, w1
	mov	-16(up,un,8), %rax
	mov	%rdx, w2
	jmp	L(addmul_entry_3)

	C this loop is 19 c/loop = 2.375 c/l on K8, for all up/rp alignments

	ALIGN(16)
L(addmul_top):
	add	w3, -32(rp,n,8)
	adc	%rax, w0
	mov	-24(up,n,8), %rax
	adc	%rdx, w1
	xor	R32(w2), R32(w2)
	mul	v0
	add	%rax, w0
	mov	-24(up,n,8), %rax
	adc	%rdx, w1
	adc	R32(w2), R32(w2)	C adc $0, w2
L(addmul_entry_0):
	mul	v1
	xor	R32(w3), R32(w3)
	add	w0, -24(rp,n,8)
	adc	%rax, w1
	mov	-16(up,n,8), %rax
	adc	%rdx, w2
	mul	v0
	add	%rax, w1
	mov	-16(up,n,8), %rax
	adc	%rdx, w2
	adc	$0, R32(w3)
L(addmul_entry_3):
	mul	v1
	add	w1, -16(rp,n,8)
	adc	%rax, w2
	mov	-8(up,n,8), %rax
	adc	%rdx, w3
	mul	v0
	xor	R32(w0), R32(w0)
	add	%rax, w2
	adc	%rdx, w3
	mov	$0, R32(w1)
	mov	-8(up,n,8), %rax
	adc	R32(w1), R32(w0)	C adc $0, w0
L(addmul_entry_2):
	mul	v1
	add	w2, -8(rp,n,8)
	adc	%rax, w3
	adc	%rdx, w0
	mov	(up,n,8), %rax
	mul	v0
	add	%rax, w3
	mov	(up,n,8), %rax
	adc	%rdx, w0
	adc	$0, R32(w1)
L(addmul_entry_1):
	mul	v1
	add	$4, n
	js	L(addmul_top)

	add	w3, -8(rp)
	adc	%rax, w0
	mov	w0, (rp)
	adc	%rdx, w1
	mov	w1, 8(rp)

	add	$-2, vn			C vn -= 2
	jz	L(ret)

	lea	16(rp), rp		C rp += 2
	lea	16(vp), vp		C vp += 2

	mov	(vp), v0
	mov	8(vp), v1

	jmp	*outer_addr

	ALIGN(16)
L(ret):	pop	%r15
	pop	%r14
	pop	%r13
	pop	%r12
	pop	%rbp
	pop	%rbx
	FUNC_EXIT()
	ret

EPILOGUE()
