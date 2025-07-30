dnl  AMD64 mpn_mulmid_basecase

dnl  Contributed by David Harvey.

dnl  Copyright 2011, 2012 Free Software Foundation, Inc.

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
C K8,K9:	 2.375  (2.5 when un - vn is "small")
C K10:		 ?
C P4:		 ?
C P6-15:	 ?

C INPUT PARAMETERS
define(`rp',      `%rdi')
define(`up',      `%rsi')
define(`un_param',`%rdx')
define(`vp_param',`%rcx')
define(`vn',      `%r8')

define(`v0', `%r12')
define(`v1', `%r9')

define(`w0', `%rbx')
define(`w1', `%rcx')
define(`w2', `%rbp')
define(`w3', `%r10')

define(`n',  `%r11')
define(`outer_addr', `%r14')
define(`un',  `%r13')
define(`vp',  `%r15')

define(`vp_inner', `%r10')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_mulmid_basecase)
	FUNC_ENTRY(4)
IFDOS(`	mov	56(%rsp), %r8d	')
	push	%rbx
	push	%rbp
	push	%r12
	push	%r13
	push	%r14
	push	%r15

	mov	vp_param, vp

	C use un for row length (= un_param - vn + 1)
	lea	1(un_param), un
	sub	vn, un

	lea	(rp,un,8), rp

	cmp	$4, un		C TODO: needs tuning
	jc	L(diagonal)

	lea	(up,un_param,8), up

	test	$1, vn
	jz	L(mul_2)

C ===========================================================
C     mul_1 for vp[0] if vn is odd

L(mul_1):
	mov	R32(un), R32(w0)

	neg	un
	mov	(up,un,8), %rax
	mov	(vp), v0
	mul	v0

	and	$-4, un		C round down to multiple of 4
	mov	un, n

	and	$3, R32(w0)
	jz	L(mul_1_prologue_0)
	cmp	$2, R32(w0)
	jc	L(mul_1_prologue_1)
	jz	L(mul_1_prologue_2)

L(mul_1_prologue_3):
	mov	%rax, w3
	mov	%rdx, w0
	lea	L(addmul_prologue_3)(%rip), outer_addr
	jmp	L(mul_1_entry_3)

	ALIGN(16)
L(mul_1_prologue_0):
	mov	%rax, w2
	mov	%rdx, w3		C note already w0 == 0
	lea	L(addmul_prologue_0)(%rip), outer_addr
	jmp	L(mul_1_entry_0)

	ALIGN(16)
L(mul_1_prologue_1):
	add	$4, n
	mov	%rax, w1
	mov	%rdx, w2
	mov	$0, R32(w3)
	mov	(up,n,8), %rax
	lea	L(addmul_prologue_1)(%rip), outer_addr
	jmp	L(mul_1_entry_1)

	ALIGN(16)
L(mul_1_prologue_2):
	mov	%rax, w0
	mov	%rdx, w1
	mov	24(up,n,8), %rax
	mov	$0, R32(w2)
	mov	$0, R32(w3)
	lea	L(addmul_prologue_2)(%rip), outer_addr
	jmp	L(mul_1_entry_2)


	C this loop is 10 c/loop = 2.5 c/l on K8

	ALIGN(16)
L(mul_1_top):
	mov	w0, -16(rp,n,8)
	add	%rax, w1
	mov	(up,n,8), %rax
	adc	%rdx, w2
L(mul_1_entry_1):
	mov	$0, R32(w0)
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
	mov	$0, R32(w2)		C zero
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
	mov	w2, 8(rp)		C zero last limb of output
	adc	%rdx, w2
	mov	w2, (rp)

	dec	vn
	jz	L(ret)

	lea	-8(up), up
	lea	8(vp), vp

	mov	un, n
	mov	(vp), v0
	mov	8(vp), v1

	jmp	*outer_addr

C ===========================================================
C     mul_2 for vp[0], vp[1] if vn is even

	ALIGN(16)
L(mul_2):
	mov	R32(un), R32(w0)

	neg	un
	mov	-8(up,un,8), %rax
	mov	(vp), v0
	mov	8(vp), v1
	mul	v1

	and	$-4, un		C round down to multiple of 4
	mov	un, n

	and	$3, R32(w0)
	jz	L(mul_2_prologue_0)
	cmp	$2, R32(w0)
	jc	L(mul_2_prologue_1)
	jz	L(mul_2_prologue_2)

L(mul_2_prologue_3):
	mov	%rax, w1
	mov	%rdx, w2
	lea	L(addmul_prologue_3)(%rip), outer_addr
	jmp	L(mul_2_entry_3)

	ALIGN(16)
L(mul_2_prologue_0):
	mov	%rax, w0
	mov	%rdx, w1
	lea	L(addmul_prologue_0)(%rip), outer_addr
	jmp	L(mul_2_entry_0)

	ALIGN(16)
L(mul_2_prologue_1):
	mov	%rax, w3
	mov	%rdx, w0
	mov	$0, R32(w1)
	lea	L(addmul_prologue_1)(%rip), outer_addr
	jmp	L(mul_2_entry_1)

	ALIGN(16)
L(mul_2_prologue_2):
	mov	%rax, w2
	mov	%rdx, w3
	mov	$0, R32(w0)
	mov	16(up,n,8), %rax
	lea	L(addmul_prologue_2)(%rip), outer_addr
	jmp	L(mul_2_entry_2)


	C this loop is 18 c/loop = 2.25 c/l on K8

	ALIGN(16)
L(mul_2_top):
	mov     -8(up,n,8), %rax
	mul     v1
	add     %rax, w0
	adc     %rdx, w1
L(mul_2_entry_0):
	mov     $0, R32(w2)
	mov     (up,n,8), %rax
	mul     v0
	add     %rax, w0
	mov     (up,n,8), %rax
	adc     %rdx, w1
	adc     $0, R32(w2)
	mul     v1
	add     %rax, w1
	mov     w0, (rp,n,8)
	adc     %rdx, w2
L(mul_2_entry_3):
	mov     8(up,n,8), %rax
	mul     v0
	mov     $0, R32(w3)
	add     %rax, w1
	adc     %rdx, w2
	mov     $0, R32(w0)
	adc     $0, R32(w3)
	mov     8(up,n,8), %rax
	mov     w1, 8(rp,n,8)
	mul     v1
	add     %rax, w2
	mov     16(up,n,8), %rax
	adc     %rdx, w3
L(mul_2_entry_2):
	mov     $0, R32(w1)
	mul     v0
	add     %rax, w2
	mov     16(up,n,8), %rax
	adc     %rdx, w3
	adc     $0, R32(w0)
	mul     v1
	add     %rax, w3
	mov     w2, 16(rp,n,8)
	adc     %rdx, w0
L(mul_2_entry_1):
	mov     24(up,n,8), %rax
	mul     v0
	add     %rax, w3
	adc     %rdx, w0
	adc     $0, R32(w1)
	add     $4, n
	mov     w3, -8(rp,n,8)
	jnz     L(mul_2_top)

	mov	w0, (rp)
	mov	w1, 8(rp)

	sub	$2, vn
	jz	L(ret)

	lea	16(vp), vp
	lea	-16(up), up

	mov	un, n
	mov	(vp), v0
	mov	8(vp), v1

	jmp	*outer_addr

C ===========================================================
C     addmul_2 for remaining vp's

	ALIGN(16)
L(addmul_prologue_0):
	mov	-8(up,n,8), %rax
	mul	v1
	mov	%rax, w1
	mov	%rdx, w2
	mov	$0, R32(w3)
	jmp	L(addmul_entry_0)

	ALIGN(16)
L(addmul_prologue_1):
	mov	16(up,n,8), %rax
	mul	v1
	mov	%rax, w0
	mov	%rdx, w1
	mov	$0, R32(w2)
	mov	24(up,n,8), %rax
	jmp	L(addmul_entry_1)

	ALIGN(16)
L(addmul_prologue_2):
	mov	8(up,n,8), %rax
	mul	v1
	mov	%rax, w3
	mov	%rdx, w0
	mov	$0, R32(w1)
	jmp	L(addmul_entry_2)

	ALIGN(16)
L(addmul_prologue_3):
	mov	(up,n,8), %rax
	mul	v1
	mov	%rax, w2
	mov	%rdx, w3
	mov	$0, R32(w0)
	mov	$0, R32(w1)
	jmp	L(addmul_entry_3)

	C this loop is 19 c/loop = 2.375 c/l on K8

	ALIGN(16)
L(addmul_top):
	mov	$0, R32(w3)
	add	%rax, w0
	mov	-8(up,n,8), %rax
	adc	%rdx, w1
	adc	$0, R32(w2)
	mul	v1
	add	w0, -8(rp,n,8)
	adc	%rax, w1
	adc	%rdx, w2
L(addmul_entry_0):
	mov	(up,n,8), %rax
	mul	v0
	add	%rax, w1
	mov	(up,n,8), %rax
	adc	%rdx, w2
	adc	$0, R32(w3)
	mul	v1
	add	w1, (rp,n,8)
	mov	$0, R32(w1)
	adc	%rax, w2
	mov	$0, R32(w0)
	adc	%rdx, w3
L(addmul_entry_3):
	mov	8(up,n,8), %rax
	mul	v0
	add	%rax, w2
	mov	8(up,n,8), %rax
	adc	%rdx, w3
	adc	$0, R32(w0)
	mul	v1
	add	w2, 8(rp,n,8)
	adc	%rax, w3
	adc	%rdx, w0
L(addmul_entry_2):
	mov	16(up,n,8), %rax
	mul	v0
	add	%rax, w3
	mov	16(up,n,8), %rax
	adc	%rdx, w0
	adc	$0, R32(w1)
	mul	v1
	add	w3, 16(rp,n,8)
	nop			C don't ask...
	adc	%rax, w0
	mov	$0, R32(w2)
	mov	24(up,n,8), %rax
	adc	%rdx, w1
L(addmul_entry_1):
	mul	v0
	add	$4, n
	jnz	L(addmul_top)

	add	%rax, w0
	adc	%rdx, w1
	adc	$0, R32(w2)

	add	w0, -8(rp)
	adc	w1, (rp)
	adc	w2, 8(rp)

	sub	$2, vn
	jz	L(ret)

	lea	16(vp), vp
	lea	-16(up), up

	mov	un, n
	mov	(vp), v0
	mov	8(vp), v1

	jmp	*outer_addr

C ===========================================================
C     accumulate along diagonals if un - vn is small

	ALIGN(16)
L(diagonal):
	xor	R32(w0), R32(w0)
	xor	R32(w1), R32(w1)
	xor	R32(w2), R32(w2)

	neg	un

	mov	R32(vn), %eax
	and	$3, %eax
	jz	L(diag_prologue_0)
	cmp	$2, %eax
	jc	L(diag_prologue_1)
	jz	L(diag_prologue_2)

L(diag_prologue_3):
	lea	-8(vp), vp
	mov	vp, vp_inner
	add	$1, vn
	mov	vn, n
	lea	L(diag_entry_3)(%rip), outer_addr
	jmp	L(diag_entry_3)

L(diag_prologue_0):
	mov	vp, vp_inner
	mov	vn, n
	lea	0(%rip), outer_addr
	mov     -8(up,n,8), %rax
	jmp	L(diag_entry_0)

L(diag_prologue_1):
	lea	8(vp), vp
	mov	vp, vp_inner
	add	$3, vn
	mov	vn, n
	lea	0(%rip), outer_addr
	mov     -8(vp_inner), %rax
	jmp	L(diag_entry_1)

L(diag_prologue_2):
	lea	-16(vp), vp
	mov	vp, vp_inner
	add	$2, vn
	mov	vn, n
	lea	0(%rip), outer_addr
	mov	16(vp_inner), %rax
	jmp	L(diag_entry_2)


	C this loop is 10 c/loop = 2.5 c/l on K8

	ALIGN(16)
L(diag_top):
	add     %rax, w0
	adc     %rdx, w1
	mov     -8(up,n,8), %rax
	adc     $0, w2
L(diag_entry_0):
	mulq    (vp_inner)
	add     %rax, w0
	adc     %rdx, w1
	adc     $0, w2
L(diag_entry_3):
	mov     -16(up,n,8), %rax
	mulq    8(vp_inner)
	add     %rax, w0
	mov     16(vp_inner), %rax
	adc     %rdx, w1
	adc     $0, w2
L(diag_entry_2):
	mulq    -24(up,n,8)
	add     %rax, w0
	mov     24(vp_inner), %rax
	adc     %rdx, w1
	lea     32(vp_inner), vp_inner
	adc     $0, w2
L(diag_entry_1):
	mulq    -32(up,n,8)
	sub     $4, n
	jnz	L(diag_top)

	add	%rax, w0
	adc	%rdx, w1
	adc	$0, w2

	mov	w0, (rp,un,8)

	inc	un
	jz	L(diag_end)

	mov	vn, n
	mov	vp, vp_inner

	lea	8(up), up
	mov	w1, w0
	mov	w2, w1
	xor	R32(w2), R32(w2)

	jmp	*outer_addr

L(diag_end):
	mov	w1, (rp)
	mov	w2, 8(rp)

L(ret):	pop	%r15
	pop	%r14
	pop	%r13
	pop	%r12
	pop	%rbp
	pop	%rbx
	FUNC_EXIT()
	ret
EPILOGUE()
