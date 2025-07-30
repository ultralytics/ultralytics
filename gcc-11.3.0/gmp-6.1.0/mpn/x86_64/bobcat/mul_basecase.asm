dnl  AMD64 mpn_mul_basecase optimised for AMD bobcat.

dnl  Copyright 2003-2005, 2007, 2008, 2011, 2012 Free Software Foundation, Inc.

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
C AMD K8,K9	 4.5
C AMD K10	 4.5
C AMD bd1	 4.75
C AMD bobcat	 5
C Intel P4	17.7
C Intel core2	 5.5
C Intel NHM	 5.43
C Intel SBR	 3.92
C Intel atom	23
C VIA nano	 5.63

C This mul_basecase is based on mul_1 and addmul_1, since these both run at the
C multiply insn bandwidth, without any apparent loop branch exit pipeline
C replays experienced on K8.  The structure is unusual: it falls into mul_1 in
C the same way for all n, then it splits into 4 different wind-down blocks and
C 4 separate addmul_1 loops.
C
C We have not tried using the same addmul_1 loops with a switch into feed-in
C code, as we do in other basecase implementations.  Doing that could save
C substantial code volume, but would also probably add some overhead.

C TODO
C  * Tune un < 3 code.
C  * Fix slowdown for un=vn=3 (67->71) compared to default code.
C  * This is 1263 bytes, compared to 1099 bytes for default code.  Consider
C    combining addmul loops like that code.  Tolerable slowdown?
C  * Lots of space could be saved by replacing the "switch" code by gradual
C    jumps out from mul_1 winddown code, perhaps with no added overhead.
C  * Are the ALIGN(16) really necessary?  They add about 25 bytes of padding.

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

C Standard parameters
define(`rp',              `%rdi')
define(`up',              `%rsi')
define(`un_param',        `%rdx')
define(`vp',              `%rcx')
define(`vn',              `%r8')
C Standard allocations
define(`un',              `%rbx')
define(`w0',              `%r10')
define(`w1',              `%r11')
define(`w2',              `%r12')
define(`w3',              `%r13')
define(`n',               `%rbp')
define(`v0',              `%r9')

C Temp macro for allowing control over indexing.
C Define to return $1 for more conservative ptr handling.
define(`X',`$2')


ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_mul_basecase)
	FUNC_ENTRY(4)
IFDOS(`	mov	56(%rsp), %r8d	')

	mov	(up), %rax
	mov	(vp), v0

	cmp	$2, un_param
	ja	L(ge3)
	jz	L(u2)

	mul	v0			C u0 x v0
	mov	%rax, (rp)
	mov	%rdx, 8(rp)
	FUNC_EXIT()
	ret

L(u2):	mul	v0			C u0 x v0
	mov	%rax, (rp)
	mov	8(up), %rax
	mov	%rdx, w0
	mul	v0
	add	%rax, w0
	mov	%rdx, w1
	adc	$0, w1
	cmp	$1, R32(vn)
	jnz	L(u2v2)
	mov	w0, 8(rp)
	mov	w1, 16(rp)
	FUNC_EXIT()
	ret

L(u2v2):mov	8(vp), v0
	mov	(up), %rax
	mul	v0
	add	%rax, w0
	mov	w0, 8(rp)
	mov	%rdx, %r8		C CAUTION: r8 realloc
	adc	$0, %r8
	mov	8(up), %rax
	mul	v0
	add	w1, %r8
	adc	$0, %rdx
	add	%r8, %rax
	adc	$0, %rdx
	mov	%rax, 16(rp)
	mov	%rdx, 24(rp)
	FUNC_EXIT()
	ret


L(ge3):	push	%rbx
	push	%rbp
	push	%r12
	push	%r13

	lea	8(vp), vp

	lea	-24(rp,un_param,8), rp
	lea	-24(up,un_param,8), up
	xor	R32(un), R32(un)
	mov	$2, R32(n)
	sub	un_param, un
	sub	un_param, n

	mul	v0
	mov	%rax, w2
	mov	%rdx, w3
	jmp	L(L3)

	ALIGN(16)
L(top):	mov	w0, -16(rp,n,8)
	add	w1, w2
	adc	$0, w3
	mov	(up,n,8), %rax
	mul	v0
	mov	%rax, w0
	mov	%rdx, w1
	mov	w2, -8(rp,n,8)
	add	w3, w0
	adc	$0, w1
	mov	8(up,n,8), %rax
	mul	v0
	mov	%rax, w2
	mov	%rdx, w3
	mov	w0, (rp,n,8)
	add	w1, w2
	adc	$0, w3
L(L3):	mov	16(up,n,8), %rax
	mul	v0
	mov	%rax, w0
	mov	%rdx, w1
	mov	w2, 8(rp,n,8)
	add	w3, w0
	adc	$0, w1
	mov	24(up,n,8), %rax
	mul	v0
	mov	%rax, w2
	mov	%rdx, w3
	add	$4, n
	js	L(top)

	mov	w0, -16(rp,n,8)
	add	w1, w2
	adc	$0, w3

C Switch on n into right addmul_l loop
	test	n, n
	jz	L(r2)
	cmp	$2, R32(n)
	ja	L(r3)
	jz	L(r0)
	jmp	L(r1)


L(r3):	mov	w2, X(-8(rp,n,8),16(rp))
	mov	w3, X((rp,n,8),24(rp))
	add	$2, un

C outer loop(3)
L(to3):	dec	vn
	jz	L(ret)
	mov	(vp), v0
	mov	8(up,un,8), %rax
	lea	8(vp), vp
	lea	8(rp), rp
	mov	un, n
	mul	v0
	mov	%rax, w2
	mov	%rdx, w3
	jmp	L(al3)

	ALIGN(16)
L(ta3):	add	w0, -16(rp,n,8)
	adc	w1, w2
	adc	$0, w3
	mov	(up,n,8), %rax
	mul	v0
	mov	%rax, w0
	mov	%rdx, w1
	add	w2, -8(rp,n,8)
	adc	w3, w0
	adc	$0, w1
	mov	8(up,n,8), %rax
	mul	v0
	mov	%rax, w2
	mov	%rdx, w3
	add	w0, (rp,n,8)
	adc	w1, w2
	adc	$0, w3
L(al3):	mov	16(up,n,8), %rax
	mul	v0
	mov	%rax, w0
	mov	%rdx, w1
	add	w2, 8(rp,n,8)
	adc	w3, w0
	adc	$0, w1
	mov	24(up,n,8), %rax
	mul	v0
	mov	%rax, w2
	mov	%rdx, w3
	add	$4, n
	js	L(ta3)

	add	w0, X(-16(rp,n,8),8(rp))
	adc	w1, w2
	adc	$0, w3
	add	w2, X(-8(rp,n,8),16(rp))
	adc	$0, w3
	mov	w3, X((rp,n,8),24(rp))
	jmp	L(to3)


L(r2):	mov	X(0(up,n,8),(up)), %rax
	mul	v0
	mov	%rax, w0
	mov	%rdx, w1
	mov	w2, X(-8(rp,n,8),-8(rp))
	add	w3, w0
	adc	$0, w1
	mov	X(8(up,n,8),8(up)), %rax
	mul	v0
	mov	%rax, w2
	mov	%rdx, w3
	mov	w0, X((rp,n,8),(rp))
	add	w1, w2
	adc	$0, w3
	mov	X(16(up,n,8),16(up)), %rax
	mul	v0
	mov	%rax, w0
	mov	%rdx, w1
	mov	w2, X(8(rp,n,8),8(rp))
	add	w3, w0
	adc	$0, w1
	mov	w0, X(16(rp,n,8),16(rp))
	adc	$0, w3
	mov	w1, X(24(rp,n,8),24(rp))
	inc	un

C outer loop(2)
L(to2):	dec	vn
	jz	L(ret)
	mov	(vp), v0
	mov	16(up,un,8), %rax
	lea	8(vp), vp
	lea	8(rp), rp
	mov	un, n
	mul	v0
	mov	%rax, w0
	mov	%rdx, w1
	jmp	L(al2)

	ALIGN(16)
L(ta2):	add	w0, -16(rp,n,8)
	adc	w1, w2
	adc	$0, w3
	mov	(up,n,8), %rax
	mul	v0
	mov	%rax, w0
	mov	%rdx, w1
	add	w2, -8(rp,n,8)
	adc	w3, w0
	adc	$0, w1
	mov	8(up,n,8), %rax
	mul	v0
	mov	%rax, w2
	mov	%rdx, w3
	add	w0, (rp,n,8)
	adc	w1, w2
	adc	$0, w3
	mov	16(up,n,8), %rax
	mul	v0
	mov	%rax, w0
	mov	%rdx, w1
	add	w2, 8(rp,n,8)
	adc	w3, w0
	adc	$0, w1
L(al2):	mov	24(up,n,8), %rax
	mul	v0
	mov	%rax, w2
	mov	%rdx, w3
	add	$4, n
	js	L(ta2)

	add	w0, X(-16(rp,n,8),8(rp))
	adc	w1, w2
	adc	$0, w3
	add	w2, X(-8(rp,n,8),16(rp))
	adc	$0, w3
	mov	w3, X((rp,n,8),24(rp))
	jmp	L(to2)


L(r1):	mov	X(0(up,n,8),8(up)), %rax
	mul	v0
	mov	%rax, w0
	mov	%rdx, w1
	mov	w2, X(-8(rp,n,8),(rp))
	add	w3, w0
	adc	$0, w1
	mov	X(8(up,n,8),16(up)), %rax
	mul	v0
	mov	%rax, w2
	mov	%rdx, w3
	mov	w0, X((rp,n,8),8(rp))
	add	w1, w2
	adc	$0, w3
	mov	w2, X(8(rp,n,8),16(rp))
	mov	w3, X(16(rp,n,8),24(rp))
	add	$4, un

C outer loop(1)
L(to1):	dec	vn
	jz	L(ret)
	mov	(vp), v0
	mov	-8(up,un,8), %rax
	lea	8(vp), vp
	lea	8(rp), rp
	mov	un, n
	mul	v0
	mov	%rax, w2
	mov	%rdx, w3
	jmp	L(al1)

	ALIGN(16)
L(ta1):	add	w0, -16(rp,n,8)
	adc	w1, w2
	adc	$0, w3
L(al1):	mov	(up,n,8), %rax
	mul	v0
	mov	%rax, w0
	mov	%rdx, w1
	add	w2, -8(rp,n,8)
	adc	w3, w0
	adc	$0, w1
	mov	8(up,n,8), %rax
	mul	v0
	mov	%rax, w2
	mov	%rdx, w3
	add	w0, (rp,n,8)
	adc	w1, w2
	adc	$0, w3
	mov	16(up,n,8), %rax
	mul	v0
	mov	%rax, w0
	mov	%rdx, w1
	add	w2, 8(rp,n,8)
	adc	w3, w0
	adc	$0, w1
	mov	24(up,n,8), %rax
	mul	v0
	mov	%rax, w2
	mov	%rdx, w3
	add	$4, n
	js	L(ta1)

	add	w0, X(-16(rp,n,8),8(rp))
	adc	w1, w2
	adc	$0, w3
	add	w2, X(-8(rp,n,8),16(rp))
	adc	$0, w3
	mov	w3, X((rp,n,8),24(rp))
	jmp	L(to1)


L(r0):	mov	X((up,n,8),16(up)), %rax
	mul	v0
	mov	%rax, w0
	mov	%rdx, w1
	mov	w2, X(-8(rp,n,8),8(rp))
	add	w3, w0
	adc	$0, w1
	mov	w0, X((rp,n,8),16(rp))
	mov	w1, X(8(rp,n,8),24(rp))
	add	$3, un

C outer loop(0)
L(to0):	dec	vn
	jz	L(ret)
	mov	(vp), v0
	mov	(up,un,8), %rax
	lea	8(vp), vp
	lea	8(rp), rp
	mov	un, n
	mul	v0
	mov	%rax, w0
	mov	%rdx, w1
	jmp	L(al0)

	ALIGN(16)
L(ta0):	add	w0, -16(rp,n,8)
	adc	w1, w2
	adc	$0, w3
	mov	(up,n,8), %rax
	mul	v0
	mov	%rax, w0
	mov	%rdx, w1
	add	w2, -8(rp,n,8)
	adc	w3, w0
	adc	$0, w1
L(al0):	mov	8(up,n,8), %rax
	mul	v0
	mov	%rax, w2
	mov	%rdx, w3
	add	w0, (rp,n,8)
	adc	w1, w2
	adc	$0, w3
	mov	16(up,n,8), %rax
	mul	v0
	mov	%rax, w0
	mov	%rdx, w1
	add	w2, 8(rp,n,8)
	adc	w3, w0
	adc	$0, w1
	mov	24(up,n,8), %rax
	mul	v0
	mov	%rax, w2
	mov	%rdx, w3
	add	$4, n
	js	L(ta0)

	add	w0, X(-16(rp,n,8),8(rp))
	adc	w1, w2
	adc	$0, w3
	add	w2, X(-8(rp,n,8),16(rp))
	adc	$0, w3
	mov	w3, X((rp,n,8),24(rp))
	jmp	L(to0)


L(ret):	pop	%r13
	pop	%r12
	pop	%rbp
	pop	%rbx
	FUNC_EXIT()
	ret
EPILOGUE()
