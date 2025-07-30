dnl  X86-64 mpn_sqr_basecase optimised for Intel Nehalem/Westmere.
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

C cycles/limb	mul_2		addmul_2	sqr_diag_addlsh1
C AMD K8,K9
C AMD K10
C AMD bull
C AMD pile
C AMD steam
C AMD bobcat
C AMD jaguar
C Intel P4
C Intel core	 4.9		4.18-4.25		 3.87
C Intel NHM	 3.8		4.06-4.2		 3.5
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
C        m_2(0m4)        m_2(2m4)        m_2(1m4)        m_2(3m4)
C           |               |               |               |
C           |               |               |               |
C           |               |               |               |
C          \|/             \|/             \|/             \|/
C              ____________                   ____________
C             /            \                 /            \
C            \|/            \               \|/            \
C         am_2(3m4)       am_2(1m4)       am_2(0m4)       am_2(2m4)
C            \            /|\                \            /|\
C             \____________/                  \____________/
C                       \                        /
C                        \                      /
C                         \                    /
C                       tail(0m2)          tail(1m2)
C                            \              /
C                             \            /
C                            sqr_diag_addlsh1

C TODO
C  * Tune.  None done so far.
C  * Currently 2761 bytes, making it smaller would be nice.
C  * Consider using a jumptab-based entry sequence.  One might even use a mask-
C    less sequence, if the table is large enough to support tuneup's needs.
C    The code would be, using non-PIC code,
C        lea tab(%rip),%rax; jmp *(n,%rax)
C    or,
C        lea tab(%rip),%rax; lea (%rip),%rbx; add (n,%rax),%rbx; jmp *%rbx
C    using PIC code.  The table entries would be Ln1,Ln2,Ln3,Lm0,Lm1,Lm2,Lm3,..
C    with the last four entries repeated a safe number of times.
C  * Consider expanding feed-in code in order to avoid zeroing registers.
C  * Zero consistently with xor.
C  * Check if using "lea (reg),reg" should be done in more places; we have some
C    explicit "mov %rax,reg" now.
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
define(`n_param',  `%rdx')

define(`n',        `%r8')

define(`v0',       `%r10')
define(`v1',       `%r11')
define(`w0',       `%rbx')
define(`w1',       `%rcx')
define(`w2',       `%rbp')
define(`w3',       `%r9')
define(`i',        `%r13')

define(`X0',       `%r12')
define(`X1',       `%r14')

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
PROLOGUE(mpn_sqr_basecase)
	FUNC_ENTRY(3)

	cmp	$4, n_param
	jl	L(small)

	push	%rbx
	push	%rbp
	push	%r12
	push	%r13
	push	%r14

	mov	(up), v0
	mov	8(up), %rax
	mov	%rax, v1

	mov	$1, R32(n)
	sub	n_param, n		C n = -n_param+1
	push	n

	lea	(up,n_param,8), up
	lea	(rp,n_param,8), rp

	mul	v0

	test	$1, R8(n)
	jnz	L(bx1)

L(bx0):	test	$2, R8(n)
	mov	%rax, (rp,n,8)
	jnz	L(b10)

L(b00):	lea	(n), i			C n = 5, 9, ...
	mov	%rdx, w1		C FIXME: Use lea?
	xor	R32(w2), R32(w2)
	jmp	L(m2e0)

L(b10):	lea	2(n), i			C n = 7, 11, ...
	mov	8(up,n,8), %rax
	mov	%rdx, w3		C FIXME: Use lea?
	xor	R32(w0), R32(w0)
	xor	R32(w1), R32(w1)
	jmp	L(m2e2)

L(bx1):	test	$2, R8(n)
	mov	%rax, (rp,n,8)
	jz	L(b11)

L(b01):	lea	1(n), i			C n = 6, 10, ...
	mov	%rdx, w0		C FIXME: Use lea?
	xor	R32(w1), R32(w1)
	jmp	L(m2e1)

L(b11):	lea	-1(n), i		C n = 4, 8, 12, ...
	mov	%rdx, w2		C FIXME: Use lea?
	xor	R32(w3), R32(w3)
	jmp	L(m2e3)


	ALIGNx
L(m2top1):
	mul	v0
	add	%rax, w3
	mov	-8(up,i,8), %rax
	mov	w3, -8(rp,i,8)
	adc	%rdx, w0
	adc	$0, R32(w1)
	mul	v1
	add	%rax, w0
	adc	%rdx, w1
L(m2e1):mov	$0, R32(w2)
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
	add	w0, %rax
	adc	w1, %rdx
	mov	%rax, I((rp),(rp,i,8))
	mov	%rdx, I(8(rp),8(rp,i,8))

	lea	16(rp), rp
	add	$2, n			C decrease |n|
	jmp	L(am2o3)

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
	mul	v1
	add	%rax, w2
	mov	w1, 8(rp,i,8)
	adc	%rdx, w3
L(m2e3):mov	$0, R32(w0)
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
	adc	R32(w1), R32(w1)
	mul	v1
	add	w0, %rax
	adc	w1, %rdx
	mov	%rax, I((rp),(rp,i,8))
	mov	%rdx, I(8(rp),8(rp,i,8))

	lea	16(rp), rp
	add	$2, n			C decrease |n|
	cmp	$-1, n
	jz	L(cor1)			C jumps iff entry n = 4

L(am2o1):
	mov	-8(up,n,8), v0
	mov	(up,n,8), %rax
	mov	%rax, v1
	lea	1(n), i
	mul	v0
	mov	%rax, X1
	MOV(	%rdx, X0, 128)
	mov	(rp,n,8), w1
	xor	R32(w2), R32(w2)
	mov	8(up,n,8), %rax
	xor	R32(w3), R32(w3)
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
	adc	w2, %rax
	adc	Z(i,$0), %rdx
	add	w1, X1
	adc	Z(i,$0), X0
	mov	X1, I(-8(rp),-8(rp,i,8))
	add	X0, %rax
	mov	%rax, I((rp),(rp,i,8))
	adc	Z(i,$0), %rdx
	mov	%rdx, I(8(rp),8(rp,i,8))

	lea	16(rp), rp
	add	$2, n

L(am2o3):
	mov	-8(up,n,8), v0
	mov	(up,n,8), %rax
	mov	%rax, v1
	lea	-1(n), i
	mul	v0
	mov	%rax, X1
	MOV(	%rdx, X0, 8)
	mov	(rp,n,8), w3
	xor	R32(w0), R32(w0)
	xor	R32(w1), R32(w1)
	mov	8(up,n,8), %rax
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
	adc	w2, %rax
	adc	Z(i,$0), %rdx
	add	w1, X1
	adc	Z(i,$0), X0
	mov	X1, I(-8(rp),-8(rp,i,8))
	add	X0, %rax
	mov	%rax, I((rp),(rp,i,8))
	adc	Z(i,$0), %rdx
	mov	%rdx, I(8(rp),8(rp,i,8))

	lea	16(rp), rp
	add	$2, n
	cmp	$-1, n
	jnz	L(am2o1)

L(cor1):pop	n
	mov	%rdx, w3
	mov	-16(up), v0
	mov	-8(up), %rax
	mul	v0
	add	w3, %rax
	adc	$0, %rdx
	mov	%rax, -8(rp)
	mov	%rdx, (rp)
	jmp	L(sqr_diag_addlsh1)

	ALIGNx
L(m2top2):
L(m2e2):mul	v0
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
	mul	v1
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
	add	w0, %rax
	adc	w1, %rdx
	mov	%rax, I((rp),(rp,i,8))
	mov	%rdx, I(8(rp),8(rp,i,8))

	lea	16(rp), rp
	add	$2, n			C decrease |n|
	jmp	L(am2o0)

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
	mul	v1
	add	%rax, w1
	adc	%rdx, w2
L(m2e0):mov	8(up,i,8), %rax
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
	add	w0, %rax
	adc	w1, %rdx
	mov	%rax, I((rp),(rp,i,8))
	mov	%rdx, I(8(rp),8(rp,i,8))

	lea	16(rp), rp
	add	$2, n			C decrease |n|
	cmp	$-2, n
	jz	L(cor2)			C jumps iff entry n = 5

L(am2o2):
	mov	-8(up,n,8), v0
	mov	(up,n,8), %rax
	mov	%rax, v1
	lea	-2(n), i
	mul	v0
	mov	%rax, X0
	MOV(	%rdx, X1, 32)
	mov	(rp,n,8), w0
	xor	R32(w1), R32(w1)
	xor	R32(w2), R32(w2)
	mov	8(up,n,8), %rax
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
	adc	w2, %rax
	adc	Z(i,$0), %rdx
	add	w1, X1
	adc	Z(i,$0), X0
	mov	X1, I(-8(rp),-8(rp,i,8))
	add	X0, %rax
	mov	%rax, I((rp),(rp,i,8))
	adc	Z(i,$0), %rdx
	mov	%rdx, I(8(rp),8(rp,i,8))

	lea	16(rp), rp
	add	$2, n

L(am2o0):
	mov	-8(up,n,8), v0
	mov	(up,n,8), %rax
	mov	%rax, v1
	lea	0(n), i
	mul	v0
	mov	%rax, X0
	MOV(	%rdx, X1, 2)
	xor	R32(w0), R32(w0)
	mov	(rp,n,8), w2
	xor	R32(w3), R32(w3)
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
	adc	w2, %rax
	adc	Z(i,$0), %rdx
	add	w1, X1
	adc	Z(i,$0), X0
	mov	X1, I(-8(rp),-8(rp,i,8))
	add	X0, %rax
	mov	%rax, I((rp),(rp,i,8))
	adc	Z(i,$0), %rdx
	mov	%rdx, I(8(rp),8(rp,i,8))

	lea	16(rp), rp
	add	$2, n
	cmp	$-2, n
	jnz	L(am2o2)

L(cor2):pop	n
	mov	-24(up), v0
	mov	%rax, w2
	mov	%rdx, w0
	mov	-16(up), %rax
	mov	%rax, v1
	mul	v0
	mov	%rax, X0
	MOV(	%rdx, X1, 32)
	mov	-8(up), %rax
	mul	v0
	add	w2, X0
	mov	X0, -16(rp)
	MOV(	%rdx, X0, 128)
	adc	%rax, X1
	mov	-8(up), %rax
	adc	$0, X0
	mul	v1
	add	w0, X1
	adc	$0, X0
	mov	X1, -8(rp)
	add	X0, %rax
	mov	%rax, (rp)
	adc	$0, %rdx
	mov	%rdx, 8(rp)
	lea	8(rp), rp

L(sqr_diag_addlsh1):
	mov	-8(up,n,8), %rax
	shl	n
	xor	R32(%rbx), R32(%rbx)
	mul	%rax
	mov	8(rp,n,8), %r11
	lea	(%rdx), %r10
	mov	16(rp,n,8), %r9
	add	%r11, %r11
	jmp	L(dm)

	ALIGNx
L(dtop):mul	%rax
	add	%r11, %r10
	mov	8(rp,n,8), %r11
	mov	%r10, -8(rp,n,8)
	adc	%r9, %rax
	lea	(%rdx,%rbx), %r10
	mov	16(rp,n,8), %r9
	adc	%r11, %r11
L(dm):	mov	%rax, (rp,n,8)
	mov	(up,n,4), %rax
	adc	%r9, %r9
	setc	R8(%rbx)
	add	$2, n
	js	L(dtop)

	mul	%rax
	add	%r11, %r10
	mov	%r10, -8(rp)
	adc	%r9, %rax
	lea	(%rdx,%rbx), %r10
	mov	%rax, (rp)
	adc	$0, %r10
	mov	%r10, 8(rp)

	pop	%r14
	pop	%r13
	pop	%r12
	pop	%rbp
	pop	%rbx
	FUNC_EXIT()
	ret

	ALIGN(16)
L(small):
	mov	(up), %rax
	cmp	$2, n_param
	jae	L(gt1)
L(n1):
	mul	%rax
	mov	%rax, (rp)
	mov	%rdx, 8(rp)
	FUNC_EXIT()
	ret

L(gt1):	jne	L(gt2)
L(n2):	mov	%rax, %r8
	mul	%rax
	mov	8(up), %r11
	mov	%rax, (rp)
	mov	%r11, %rax
	mov	%rdx, %r9
	mul	%rax
	mov	%rax, %r10
	mov	%r11, %rax
	mov	%rdx, %r11
	mul	%r8
	xor	%r8, %r8
	add	%rax, %r9
	adc	%rdx, %r10
	adc	%r8, %r11
	add	%rax, %r9
	mov	%r9, 8(rp)
	adc	%rdx, %r10
	mov	%r10, 16(rp)
	adc	%r8, %r11
	mov	%r11, 24(rp)
	FUNC_EXIT()
	ret

L(gt2):
L(n3):	mov	%rax, %r10
	mul	%rax
	mov	8(up), %r11
	mov	%rax, (rp)
	mov	%r11, %rax
	mov	%rdx, 8(rp)
	mul	%rax
	mov	16(up), %rcx
	mov	%rax, 16(rp)
	mov	%rcx, %rax
	mov	%rdx, 24(rp)
	mul	%rax
	mov	%rax, 32(rp)
	mov	%rdx, 40(rp)

	mov	%r11, %rax
	mul	%r10
	mov	%rax, %r8
	mov	%rcx, %rax
	mov	%rdx, %r9
	mul	%r10
	xor	%r10, %r10
	add	%rax, %r9
	mov	%r11, %rax
	mov	%r10, %r11
	adc	%rdx, %r10

	mul	%rcx
	add	%rax, %r10
	adc	%r11, %rdx
	add	%r8, %r8
	adc	%r9, %r9
	adc	%r10, %r10
	adc	%rdx, %rdx
	adc	%r11, %r11
	add	%r8, 8(rp)
	adc	%r9, 16(rp)
	adc	%r10, 24(rp)
	adc	%rdx, 32(rp)
	adc	%r11, 40(rp)
	FUNC_EXIT()
	ret
EPILOGUE()
