dnl  X86-64 mpn_mul_basecase optimised for Intel Nehalem/Westmere.
dnl  It also seems good for Conroe/Wolfdale.

dnl  Contributed to the GNU project by Torbjörn Granlund.

dnl  Copyright 2008, 2011-2013 Free Software Foundation, Inc.

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

C cycles/limb	mul_1		mul_2		mul_3		addmul_2
C AMD K8,K9
C AMD K10
C AMD bull
C AMD pile
C AMD steam
C AMD bobcat
C AMD jaguar
C Intel P4
C Intel core	 4.0		 4.0		 -		4.18-4.25
C Intel NHM	 3.75		 3.8		 -		4.06-4.2
C Intel SBR
C Intel IBR
C Intel HWL
C Intel BWL
C Intel atom
C VIA nano

C The inner loops of this code are the result of running a code generation and
C optimisation tool suite written by David Harvey and Torbjörn Granlund.

C Code structure:
C
C
C               m_1(0m4)        m_1(1m4)        m_1(2m4)        m_1(3m4)
C                  |               |               |               |
C        m_2(0m4)  |     m_2(1m4)  |     m_2(2m4)  |     m_2(3m4)  |
C           |      /        |      /        |      /        |      /
C           |     /         |     /         |     /         |     /
C           |    /          |    /          |    /          |    /
C          \|/ |/_         \|/ |/_         \|/ |/_         \|/ |/_
C             _____           _____           _____           _____
C            /     \         /     \         /     \         /     \
C          \|/      |      \|/      |      \|/      |      \|/      |
C        am_2(0m4)  |    am_2(1m4)  |    am_2(2m4)  |    am_2(3m4)  |
C           \      /|\      \      /|\      \      /|\      \      /|\
C            \_____/         \_____/         \_____/         \_____/

C TODO
C  * Tune.  None done so far.
C  * Currently 2687 bytes, making it smaller would be nice.
C  * Implement some basecases, say for un < 4.
C  * Try zeroing with xor in m2 loops.
C  * Try re-rolling the m2 loops to avoid the current 9 insn code duplication
C    between loop header and wind-down code.
C  * Consider adc reg,reg instead of adc $0,reg in m2 loops.  This save a byte.

C When playing with pointers, set this to $2 to fall back to conservative
C indexing in wind-down code.
define(`I',`$1')

C Define this to $1 to use late loop index variable as zero, $2 to use an
C explicit $0.
define(`Z',`$1')

define(`rp',       `%rdi')
define(`up',       `%rsi')
define(`un_param', `%rdx')
define(`vp_param', `%rcx')	C FIXME reallocate vp to rcx but watch performance!
define(`vn_param', `%r8')

define(`un',       `%r9')
define(`vn',       `(%rsp)')

define(`v0',       `%r10')
define(`v1',       `%r11')
define(`w0',       `%rbx')
define(`w1',       `%rcx')
define(`w2',       `%rbp')
define(`w3',       `%r12')
define(`i',        `%r13')
define(`vp',       `%r14')

define(`X0',       `%r8')
define(`X1',       `%r15')

C rax rbx rcx rdx rdi rsi rbp r8 r9 r10 r11 r12 r13 r14 r15

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

define(`ALIGNx', `ALIGN(16)')

define(`N', 85)
ifdef(`N',,`define(`N',0)')
define(`MOV', `ifelse(eval(N & $3),0,`mov	$1, $2',`lea	($1), $2')')

ASM_START()
	TEXT
	ALIGN(32)
PROLOGUE(mpn_mul_basecase)
	FUNC_ENTRY(4)
IFDOS(`	mov	56(%rsp), %r8d	')
	mov	(up), %rax		C shared for mul_1 and mul_2
	push	%rbx
	push	%rbp
	push	%r12
	push	%r13
	push	%r14

	mov	(vp_param), v0		C shared for mul_1 and mul_2

	xor	un, un
	sub	un_param, un		C un = -un_param

	lea	(up,un_param,8), up
	lea	(rp,un_param,8), rp

	mul	v0			C shared for mul_1 and mul_2

	test	$1, R8(vn_param)
	jz	L(m2)

	lea	8(vp_param), vp		C FIXME: delay until known needed

	test	$1, R8(un)
	jnz	L(m1x1)

L(m1x0):test	$2, R8(un)
	jnz	L(m1s2)

L(m1s0):
	lea	(un), i
	mov	%rax, (rp,un,8)
	mov	8(up,un,8), %rax
	mov	%rdx, w0		C FIXME: Use lea?
	lea	L(do_am0)(%rip), %rbp
	jmp	L(m1e0)

L(m1s2):
	lea	2(un), i
	mov	%rax, (rp,un,8)
	mov	8(up,un,8), %rax
	mov	%rdx, w0		C FIXME: Use lea?
	mul	v0
	lea	L(do_am2)(%rip), %rbp
	test	i, i
	jnz	L(m1e2)
	add	%rax, w0
	adc	$0, %rdx
	mov	w0, I(-8(rp),8(rp,un,8))
	mov	%rdx, I((rp),16(rp,un,8))
	jmp	L(ret2)

L(m1x1):test	$2, R8(un)
	jz	L(m1s3)

L(m1s1):
	lea	1(un), i
	mov	%rax, (rp,un,8)
	test	i, i
	jz	L(1)
	mov	8(up,un,8), %rax
	mov	%rdx, w1		C FIXME: Use lea?
	lea	L(do_am1)(%rip), %rbp
	jmp	L(m1e1)
L(1):	mov	%rdx, I((rp),8(rp,un,8))
	jmp	L(ret2)

L(m1s3):
	lea	-1(un), i
	mov	%rax, (rp,un,8)
	mov	8(up,un,8), %rax
	mov	%rdx, w1		C FIXME: Use lea?
	lea	L(do_am3)(%rip), %rbp
	jmp	L(m1e3)

	ALIGNx
L(m1top):
	mul	v0
	mov	w1, -16(rp,i,8)
L(m1e2):xor	R32(w1), R32(w1)
	add	%rax, w0
	mov	(up,i,8), %rax
	adc	%rdx, w1
	mov	w0, -8(rp,i,8)
L(m1e1):xor	R32(w0), R32(w0)
	mul	v0
	add	%rax, w1
	mov	8(up,i,8), %rax
	adc	%rdx, w0
	mov	w1, (rp,i,8)
L(m1e0):xor	R32(w1), R32(w1)
	mul	v0
	add	%rax, w0
	mov	16(up,i,8), %rax
	adc	%rdx, w1
	mov	w0, 8(rp,i,8)
L(m1e3):xor	R32(w0), R32(w0)
	mul	v0
	add	%rax, w1
	mov	24(up,i,8), %rax
	adc	%rdx, w0
	add	$4, i
	js	L(m1top)

	mul	v0
	mov	w1, I(-16(rp),-16(rp,i,8))
	add	%rax, w0
	adc	$0, %rdx
	mov	w0, I(-8(rp),-8(rp,i,8))
	mov	%rdx, I((rp),(rp,i,8))

	dec	vn_param
	jz	L(ret2)
	lea	-8(rp), rp
	jmp	*%rbp

L(m2):
	mov	8(vp_param), v1
	lea	16(vp_param), vp	C FIXME: delay until known needed

	test	$1, R8(un)
	jnz	L(bx1)

L(bx0):	test	$2, R8(un)
	jnz	L(b10)

L(b00):	lea	(un), i
	mov	%rax, (rp,un,8)
	mov	%rdx, w1		C FIXME: Use lea?
	mov	(up,un,8), %rax
	mov	$0, R32(w2)
	jmp	L(m2e0)

L(b10):	lea	-2(un), i
	mov	%rax, w2		C FIXME: Use lea?
	mov	(up,un,8), %rax
	mov	%rdx, w3		C FIXME: Use lea?
	mov	$0, R32(w0)
	jmp	L(m2e2)

L(bx1):	test	$2, R8(un)
	jz	L(b11)

L(b01):	lea	1(un), i
	mov	%rax, (rp,un,8)
	mov	(up,un,8), %rax
	mov	%rdx, w0		C FIXME: Use lea?
	mov	$0, R32(w1)
	jmp	L(m2e1)

L(b11):	lea	-1(un), i
	mov	%rax, w1		C FIXME: Use lea?
	mov	(up,un,8), %rax
	mov	%rdx, w2		C FIXME: Use lea?
	mov	$0, R32(w3)
	jmp	L(m2e3)

	ALIGNx
L(m2top0):
	mul	v0
	add	%rax, w3
	mov	-8(up,i,8), %rax
	mov	w3, -8(rp,i,8)
	adc	%rdx, w0
	adc	$0, R32(w1)
	mul	v1
	add	%rax, w0
	adc	%rdx, w1
	mov	$0, R32(w2)
	mov	(up,i,8), %rax
	mul	v0
	add	%rax, w0
	mov	w0, (rp,i,8)
	adc	%rdx, w1
	mov	(up,i,8), %rax
	adc	$0, R32(w2)
L(m2e0):mul	v1
	add	%rax, w1
	adc	%rdx, w2
	mov	8(up,i,8), %rax
	mul	v0
	mov	$0, R32(w3)
	add	%rax, w1
	adc	%rdx, w2
	adc	$0, R32(w3)
	mov	8(up,i,8), %rax
	mul	v1
	add	%rax, w2
	mov	w1, 8(rp,i,8)
	adc	%rdx, w3
	mov	$0, R32(w0)
	mov	16(up,i,8), %rax
	mul	v0
	add	%rax, w2
	mov	16(up,i,8), %rax
	adc	%rdx, w3
	adc	$0, R32(w0)
	mul	v1
	mov	$0, R32(w1)
	add	%rax, w3
	mov	24(up,i,8), %rax
	mov	w2, 16(rp,i,8)
	adc	%rdx, w0
	add	$4, i
	js	L(m2top0)

	mul	v0
	add	%rax, w3
	mov	I(-8(up),-8(up,i,8)), %rax
	mov	w3, I(-8(rp),-8(rp,i,8))
	adc	%rdx, w0
	adc	R32(w1), R32(w1)
	mul	v1
	add	%rax, w0
	adc	%rdx, w1
	mov	w0, I((rp),(rp,i,8))
	mov	w1, I(8(rp),8(rp,i,8))

	add	$-2, vn_param
	jz	L(ret2)

L(do_am0):
	push	%r15
	push	vn_param

L(olo0):
	mov	(vp), v0
	mov	8(vp), v1
	lea	16(vp), vp
	lea	16(rp), rp
	mov	(up,un,8), %rax
C	lea	0(un), i
	mov	un, i
	mul	v0
	mov	%rax, X0
	mov	(up,un,8), %rax
	MOV(	%rdx, X1, 2)
	mul	v1
	MOV(	%rdx, w0, 4)
	mov	(rp,un,8), w2
	mov	%rax, w3
	jmp	L(lo0)

	ALIGNx
L(am2top0):
	mul	v1
	add	w0, w1
	adc	%rax, w2
	mov	(up,i,8), %rax
	MOV(	%rdx, w3, 1)
	adc	$0, w3
	mul	v0
	add	w1, X1
	mov	X1, -8(rp,i,8)
	adc	%rax, X0
	MOV(	%rdx, X1, 2)
	adc	$0, X1
	mov	(up,i,8), %rax
	mul	v1
	MOV(	%rdx, w0, 4)
	mov	(rp,i,8), w1
	add	w1, w2
	adc	%rax, w3
	adc	$0, w0
L(lo0):	mov	8(up,i,8), %rax
	mul	v0
	add	w2, X0
	adc	%rax, X1
	mov	X0, (rp,i,8)
	MOV(	%rdx, X0, 8)
	adc	$0, X0
	mov	8(up,i,8), %rax
	mov	8(rp,i,8), w2
	mul	v1
	add	w2, w3
	adc	%rax, w0
	MOV(	%rdx, w1, 16)
	adc	$0, w1
	mov	16(up,i,8), %rax
	mul	v0
	add	w3, X1
	mov	X1, 8(rp,i,8)
	adc	%rax, X0
	MOV(	%rdx, X1, 32)
	mov	16(rp,i,8), w3
	adc	$0, X1
	mov	16(up,i,8), %rax
	mul	v1
	add	w3, w0
	MOV(	%rdx, w2, 64)
	adc	%rax, w1
	mov	24(up,i,8), %rax
	adc	$0, w2
	mul	v0
	add	w0, X0
	mov	X0, 16(rp,i,8)
	MOV(	%rdx, X0, 128)
	adc	%rax, X1
	mov	24(up,i,8), %rax
	mov	24(rp,i,8), w0
	adc	$0, X0
	add	$4, i
	jnc	L(am2top0)

	mul	v1
	add	w0, w1
	adc	%rax, w2
	adc	Z(i,$0), %rdx
	add	w1, X1
	adc	Z(i,$0), X0
	mov	X1, I(-8(rp),-8(rp,i,8))
	add	w2, X0
	mov	X0, I((rp),(rp,i,8))
	adc	Z(i,$0), %rdx
	mov	%rdx, I(8(rp),8(rp,i,8))

	addl	$-2, vn
	jnz	L(olo0)

L(ret):	pop	%rax
	pop	%r15
L(ret2):pop	%r14
	pop	%r13
	pop	%r12
	pop	%rbp
	pop	%rbx
	FUNC_EXIT()
	ret


	ALIGNx
L(m2top1):
	mul	v0
	add	%rax, w3
	mov	-8(up,i,8), %rax
	mov	w3, -8(rp,i,8)
	adc	%rdx, w0
	adc	$0, R32(w1)
L(m2e1):mul	v1
	add	%rax, w0
	adc	%rdx, w1
	mov	$0, R32(w2)
	mov	(up,i,8), %rax
	mul	v0
	add	%rax, w0
	mov	w0, (rp,i,8)
	adc	%rdx, w1
	mov	(up,i,8), %rax
	adc	$0, R32(w2)
	mul	v1
	add	%rax, w1
	adc	%rdx, w2
	mov	8(up,i,8), %rax
	mul	v0
	mov	$0, R32(w3)
	add	%rax, w1
	adc	%rdx, w2
	adc	$0, R32(w3)
	mov	8(up,i,8), %rax
	mul	v1
	add	%rax, w2
	mov	w1, 8(rp,i,8)
	adc	%rdx, w3
	mov	$0, R32(w0)
	mov	16(up,i,8), %rax
	mul	v0
	add	%rax, w2
	mov	16(up,i,8), %rax
	adc	%rdx, w3
	adc	$0, R32(w0)
	mul	v1
	mov	$0, R32(w1)
	add	%rax, w3
	mov	24(up,i,8), %rax
	mov	w2, 16(rp,i,8)
	adc	%rdx, w0
	add	$4, i
	js	L(m2top1)

	mul	v0
	add	%rax, w3
	mov	I(-8(up),-8(up,i,8)), %rax
	mov	w3, I(-8(rp),-8(rp,i,8))
	adc	%rdx, w0
	adc	R32(w1), R32(w1)
	mul	v1
	add	%rax, w0
	adc	%rdx, w1
	mov	w0, I((rp),(rp,i,8))
	mov	w1, I(8(rp),8(rp,i,8))

	add	$-2, vn_param
	jz	L(ret2)

L(do_am1):
	push	%r15
	push	vn_param

L(olo1):
	mov	(vp), v0
	mov	8(vp), v1
	lea	16(vp), vp
	lea	16(rp), rp
	mov	(up,un,8), %rax
	lea	1(un), i
	mul	v0
	mov	%rax, X1
	MOV(	%rdx, X0, 128)
	mov	(up,un,8), %rax
	mov	(rp,un,8), w1
	mul	v1
	mov	%rax, w2
	mov	8(up,un,8), %rax
	MOV(	%rdx, w3, 1)
	jmp	L(lo1)

	ALIGNx
L(am2top1):
	mul	v1
	add	w0, w1
	adc	%rax, w2
	mov	(up,i,8), %rax
	MOV(	%rdx, w3, 1)
	adc	$0, w3
L(lo1):	mul	v0
	add	w1, X1
	mov	X1, -8(rp,i,8)
	adc	%rax, X0
	MOV(	%rdx, X1, 2)
	adc	$0, X1
	mov	(up,i,8), %rax
	mul	v1
	MOV(	%rdx, w0, 4)
	mov	(rp,i,8), w1
	add	w1, w2
	adc	%rax, w3
	adc	$0, w0
	mov	8(up,i,8), %rax
	mul	v0
	add	w2, X0
	adc	%rax, X1
	mov	X0, (rp,i,8)
	MOV(	%rdx, X0, 8)
	adc	$0, X0
	mov	8(up,i,8), %rax
	mov	8(rp,i,8), w2
	mul	v1
	add	w2, w3
	adc	%rax, w0
	MOV(	%rdx, w1, 16)
	adc	$0, w1
	mov	16(up,i,8), %rax
	mul	v0
	add	w3, X1
	mov	X1, 8(rp,i,8)
	adc	%rax, X0
	MOV(	%rdx, X1, 32)
	mov	16(rp,i,8), w3
	adc	$0, X1
	mov	16(up,i,8), %rax
	mul	v1
	add	w3, w0
	MOV(	%rdx, w2, 64)
	adc	%rax, w1
	mov	24(up,i,8), %rax
	adc	$0, w2
	mul	v0
	add	w0, X0
	mov	X0, 16(rp,i,8)
	MOV(	%rdx, X0, 128)
	adc	%rax, X1
	mov	24(up,i,8), %rax
	mov	24(rp,i,8), w0
	adc	$0, X0
	add	$4, i
	jnc	L(am2top1)

	mul	v1
	add	w0, w1
	adc	%rax, w2
	adc	Z(i,$0), %rdx
	add	w1, X1
	adc	Z(i,$0), X0
	mov	X1, I(-8(rp),-8(rp,i,8))
	add	w2, X0
	mov	X0, I((rp),(rp,i,8))
	adc	Z(i,$0), %rdx
	mov	%rdx, I(8(rp),8(rp,i,8))

	addl	$-2, vn
	jnz	L(olo1)

	pop	%rax
	pop	%r15
	pop	%r14
	pop	%r13
	pop	%r12
	pop	%rbp
	pop	%rbx
	FUNC_EXIT()
	ret


	ALIGNx
L(m2top2):
	mul	v0
	add	%rax, w3
	mov	-8(up,i,8), %rax
	mov	w3, -8(rp,i,8)
	adc	%rdx, w0
	adc	$0, R32(w1)
	mul	v1
	add	%rax, w0
	adc	%rdx, w1
	mov	$0, R32(w2)
	mov	(up,i,8), %rax
	mul	v0
	add	%rax, w0
	mov	w0, (rp,i,8)
	adc	%rdx, w1
	mov	(up,i,8), %rax
	adc	$0, R32(w2)
	mul	v1
	add	%rax, w1
	adc	%rdx, w2
	mov	8(up,i,8), %rax
	mul	v0
	mov	$0, R32(w3)
	add	%rax, w1
	adc	%rdx, w2
	adc	$0, R32(w3)
	mov	8(up,i,8), %rax
	mul	v1
	add	%rax, w2
	mov	w1, 8(rp,i,8)
	adc	%rdx, w3
	mov	$0, R32(w0)
	mov	16(up,i,8), %rax
	mul	v0
	add	%rax, w2
	mov	16(up,i,8), %rax
	adc	%rdx, w3
	adc	$0, R32(w0)
L(m2e2):mul	v1
	mov	$0, R32(w1)
	add	%rax, w3
	mov	24(up,i,8), %rax
	mov	w2, 16(rp,i,8)
	adc	%rdx, w0
	add	$4, i
	js	L(m2top2)

	mul	v0
	add	%rax, w3
	mov	I(-8(up),-8(up,i,8)), %rax
	mov	w3, I(-8(rp),-8(rp,i,8))
	adc	%rdx, w0
	adc	R32(w1), R32(w1)
	mul	v1
	add	%rax, w0
	adc	%rdx, w1
	mov	w0, I((rp),(rp,i,8))
	mov	w1, I(8(rp),8(rp,i,8))

	add	$-2, vn_param
	jz	L(ret2)

L(do_am2):
	push	%r15
	push	vn_param

L(olo2):
	mov	(vp), v0
	mov	8(vp), v1
	lea	16(vp), vp
	lea	16(rp), rp
	mov	(up,un,8), %rax
	lea	-2(un), i
	mul	v0
	mov	%rax, X0
	MOV(	%rdx, X1, 32)
	mov	(up,un,8), %rax
	mov	(rp,un,8), w0
	mul	v1
	mov	%rax, w1
	lea	(%rdx), w2
	mov	8(up,un,8), %rax
	jmp	L(lo2)

	ALIGNx
L(am2top2):
	mul	v1
	add	w0, w1
	adc	%rax, w2
	mov	(up,i,8), %rax
	MOV(	%rdx, w3, 1)
	adc	$0, w3
	mul	v0
	add	w1, X1
	mov	X1, -8(rp,i,8)
	adc	%rax, X0
	MOV(	%rdx, X1, 2)
	adc	$0, X1
	mov	(up,i,8), %rax
	mul	v1
	MOV(	%rdx, w0, 4)
	mov	(rp,i,8), w1
	add	w1, w2
	adc	%rax, w3
	adc	$0, w0
	mov	8(up,i,8), %rax
	mul	v0
	add	w2, X0
	adc	%rax, X1
	mov	X0, (rp,i,8)
	MOV(	%rdx, X0, 8)
	adc	$0, X0
	mov	8(up,i,8), %rax
	mov	8(rp,i,8), w2
	mul	v1
	add	w2, w3
	adc	%rax, w0
	MOV(	%rdx, w1, 16)
	adc	$0, w1
	mov	16(up,i,8), %rax
	mul	v0
	add	w3, X1
	mov	X1, 8(rp,i,8)
	adc	%rax, X0
	MOV(	%rdx, X1, 32)
	mov	16(rp,i,8), w3
	adc	$0, X1
	mov	16(up,i,8), %rax
	mul	v1
	add	w3, w0
	MOV(	%rdx, w2, 64)
	adc	%rax, w1
	mov	24(up,i,8), %rax
	adc	$0, w2
L(lo2):	mul	v0
	add	w0, X0
	mov	X0, 16(rp,i,8)
	MOV(	%rdx, X0, 128)
	adc	%rax, X1
	mov	24(up,i,8), %rax
	mov	24(rp,i,8), w0
	adc	$0, X0
	add	$4, i
	jnc	L(am2top2)

	mul	v1
	add	w0, w1
	adc	%rax, w2
	adc	Z(i,$0), %rdx
	add	w1, X1
	adc	Z(i,$0), X0
	mov	X1, I(-8(rp),-8(rp,i,8))
	add	w2, X0
	mov	X0, I((rp),(rp,i,8))
	adc	Z(i,$0), %rdx
	mov	%rdx, I(8(rp),8(rp,i,8))

	addl	$-2, vn
	jnz	L(olo2)

	pop	%rax
	pop	%r15
	pop	%r14
	pop	%r13
	pop	%r12
	pop	%rbp
	pop	%rbx
	FUNC_EXIT()
	ret


	ALIGNx
L(m2top3):
	mul	v0
	add	%rax, w3
	mov	-8(up,i,8), %rax
	mov	w3, -8(rp,i,8)
	adc	%rdx, w0
	adc	$0, R32(w1)
	mul	v1
	add	%rax, w0
	adc	%rdx, w1
	mov	$0, R32(w2)
	mov	(up,i,8), %rax
	mul	v0
	add	%rax, w0
	mov	w0, (rp,i,8)
	adc	%rdx, w1
	mov	(up,i,8), %rax
	adc	$0, R32(w2)
	mul	v1
	add	%rax, w1
	adc	%rdx, w2
	mov	8(up,i,8), %rax
	mul	v0
	mov	$0, R32(w3)
	add	%rax, w1
	adc	%rdx, w2
	adc	$0, R32(w3)
	mov	8(up,i,8), %rax
L(m2e3):mul	v1
	add	%rax, w2
	mov	w1, 8(rp,i,8)
	adc	%rdx, w3
	mov	$0, R32(w0)
	mov	16(up,i,8), %rax
	mul	v0
	add	%rax, w2
	mov	16(up,i,8), %rax
	adc	%rdx, w3
	adc	$0, R32(w0)
	mul	v1
	mov	$0, R32(w1)
	add	%rax, w3
	mov	24(up,i,8), %rax
	mov	w2, 16(rp,i,8)
	adc	%rdx, w0
	add	$4, i
	js	L(m2top3)

	mul	v0
	add	%rax, w3
	mov	I(-8(up),-8(up,i,8)), %rax
	mov	w3, I(-8(rp),-8(rp,i,8))
	adc	%rdx, w0
	adc	$0, R32(w1)
	mul	v1
	add	%rax, w0
	adc	%rdx, w1
	mov	w0, I((rp),(rp,i,8))
	mov	w1, I(8(rp),8(rp,i,8))

	add	$-2, vn_param
	jz	L(ret2)

L(do_am3):
	push	%r15
	push	vn_param

L(olo3):
	mov	(vp), v0
	mov	8(vp), v1
	lea	16(vp), vp
	lea	16(rp), rp
	mov	(up,un,8), %rax
	lea	-1(un), i
	mul	v0
	mov	%rax, X1
	MOV(	%rdx, X0, 8)
	mov	(up,un,8), %rax
	mov	(rp,un,8), w3
	mul	v1
	mov	%rax, w0
	MOV(	%rdx, w1, 16)
	mov	8(up,un,8), %rax
	jmp	L(lo3)

	ALIGNx
L(am2top3):
	mul	v1
	add	w0, w1
	adc	%rax, w2
	mov	(up,i,8), %rax
	MOV(	%rdx, w3, 1)
	adc	$0, w3
	mul	v0
	add	w1, X1
	mov	X1, -8(rp,i,8)
	adc	%rax, X0
	MOV(	%rdx, X1, 2)
	adc	$0, X1
	mov	(up,i,8), %rax
	mul	v1
	MOV(	%rdx, w0, 4)
	mov	(rp,i,8), w1
	add	w1, w2
	adc	%rax, w3
	adc	$0, w0
	mov	8(up,i,8), %rax
	mul	v0
	add	w2, X0
	adc	%rax, X1
	mov	X0, (rp,i,8)
	MOV(	%rdx, X0, 8)
	adc	$0, X0
	mov	8(up,i,8), %rax
	mov	8(rp,i,8), w2
	mul	v1
	add	w2, w3
	adc	%rax, w0
	MOV(	%rdx, w1, 16)
	adc	$0, w1
	mov	16(up,i,8), %rax
L(lo3):	mul	v0
	add	w3, X1
	mov	X1, 8(rp,i,8)
	adc	%rax, X0
	MOV(	%rdx, X1, 32)
	mov	16(rp,i,8), w3
	adc	$0, X1
	mov	16(up,i,8), %rax
	mul	v1
	add	w3, w0
	MOV(	%rdx, w2, 64)
	adc	%rax, w1
	mov	24(up,i,8), %rax
	adc	$0, w2
	mul	v0
	add	w0, X0
	mov	X0, 16(rp,i,8)
	MOV(	%rdx, X0, 128)
	adc	%rax, X1
	mov	24(up,i,8), %rax
	mov	24(rp,i,8), w0
	adc	$0, X0
	add	$4, i
	jnc	L(am2top3)

	mul	v1
	add	w0, w1
	adc	%rax, w2
	adc	Z(i,$0), %rdx
	add	w1, X1
	adc	Z(i,$0), X0
	mov	X1, I(-8(rp),-8(rp,i,8))
	add	w2, X0
	mov	X0, I((rp),(rp,i,8))
	adc	Z(i,$0), %rdx
	mov	%rdx, I(8(rp),8(rp,i,8))

	addl	$-2, vn
	jnz	L(olo3)

	pop	%rax
	pop	%r15
	pop	%r14
	pop	%r13
	pop	%r12
	pop	%rbp
	pop	%rbx
	FUNC_EXIT()
	ret
EPILOGUE()
