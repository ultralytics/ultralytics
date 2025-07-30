dnl  x86-64 mpn_divrem_1 -- mpn by limb division.

dnl  Copyright 2004, 2005, 2007-2012, 2014 Free Software Foundation, Inc.

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


C		norm	unorm	frac
C AMD K8,K9	13	13	12
C AMD K10	13	13	12
C Intel P4	43	44	43
C Intel core2	24.5	24.5	19.5
C Intel corei	20.5	19.5	18
C Intel atom	43	46	36
C VIA nano	25.5	25.5	24

C mp_limb_t
C mpn_divrem_1 (mp_ptr qp, mp_size_t fn,
C               mp_srcptr np, mp_size_t nn, mp_limb_t d)

C mp_limb_t
C mpn_preinv_divrem_1 (mp_ptr qp, mp_size_t fn,
C                      mp_srcptr np, mp_size_t nn, mp_limb_t d,
C                      mp_limb_t dinv, int cnt)

C INPUT PARAMETERS
define(`qp',		`%rdi')
define(`fn_param',	`%rsi')
define(`up_param',	`%rdx')
define(`un_param',	`%rcx')
define(`d',		`%r8')
define(`dinv',		`%r9')		C only for mpn_preinv_divrem_1
C       shift passed on stack		C only for mpn_preinv_divrem_1

define(`cnt',		`%rcx')
define(`up',		`%rsi')
define(`fn',		`%r12')
define(`un',		`%rbx')


C rax rbx rcx rdx rsi rdi rbp r8  r9  r10 r11 r12 r13 r14 r15
C         cnt         qp      d  dinv

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

IFSTD(`define(`CNTOFF',		`40($1)')')
IFDOS(`define(`CNTOFF',		`104($1)')')

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_preinv_divrem_1)
	FUNC_ENTRY(4)
IFDOS(`	mov	56(%rsp), %r8	')
IFDOS(`	mov	64(%rsp), %r9	')
	xor	R32(%rax), R32(%rax)
	push	%r13
	push	%r12
	push	%rbp
	push	%rbx

	mov	fn_param, fn
	mov	un_param, un
	add	fn_param, un_param
	mov	up_param, up

	lea	-8(qp,un_param,8), qp

	test	d, d
	js	L(nent)

	mov	CNTOFF(%rsp), R8(cnt)
	shl	R8(cnt), d
	jmp	L(uent)
EPILOGUE()

	ALIGN(16)
PROLOGUE(mpn_divrem_1)
	FUNC_ENTRY(4)
IFDOS(`	mov	56(%rsp), %r8	')
	xor	R32(%rax), R32(%rax)
	push	%r13
	push	%r12
	push	%rbp
	push	%rbx

	mov	fn_param, fn
	mov	un_param, un
	add	fn_param, un_param
	mov	up_param, up
	je	L(ret)

	lea	-8(qp,un_param,8), qp
	xor	R32(%rbp), R32(%rbp)

	test	d, d
	jns	L(unnormalized)

L(normalized):
	test	un, un
	je	L(8)			C un == 0
	mov	-8(up,un,8), %rbp
	dec	un
	mov	%rbp, %rax
	sub	d, %rbp
	cmovc	%rax, %rbp
	sbb	R32(%rax), R32(%rax)
	inc	R32(%rax)
	mov	%rax, (qp)
	lea	-8(qp), qp
L(8):
IFSTD(`	push	%rdi		')
IFSTD(`	push	%rsi		')
	push	%r8
IFSTD(`	mov	d, %rdi		')
IFDOS(`	sub	$32, %rsp	')
IFDOS(`	mov	d, %rcx		')
	ASSERT(nz, `test $15, %rsp')
	CALL(	mpn_invert_limb)
IFDOS(`	add	$32, %rsp	')
	pop	%r8
IFSTD(`	pop	%rsi		')
IFSTD(`	pop	%rdi		')

	mov	%rax, dinv
	mov	%rbp, %rax
	jmp	L(nent)

	ALIGN(16)
L(ntop):mov	(up,un,8), %r10		C	    K8-K10  P6-CNR P6-NHM  P4
	mul	dinv			C	      0,13   0,20   0,18   0,45
	add	%r10, %rax		C	      4      8      3     12
	adc	%rbp, %rdx		C	      5      9     10     13
	mov	%rax, %rbp		C	      5      9      4     13
	mov	%rdx, %r13		C	      6     11     12     23
	imul	d, %rdx			C	      6     11     11     23
	sub	%rdx, %r10		C	     10     16     14     33
	mov	d, %rax			C
	add	%r10, %rax		C	     11     17     15     34
	cmp	%rbp, %r10		C	     11     17     15     34
	cmovc	%r10, %rax		C	     12     18     16     35
	adc	$-1, %r13		C
	cmp	d, %rax			C
	jae	L(nfx)			C
L(nok):	mov	%r13, (qp)		C
	sub	$8, qp			C
L(nent):lea	1(%rax), %rbp		C
	dec	un			C
	jns	L(ntop)			C

	xor	R32(%rcx), R32(%rcx)
	jmp	L(frac)

L(nfx):	sub	d, %rax
	inc	%r13
	jmp	L(nok)

L(unnormalized):
	test	un, un
	je	L(44)
	mov	-8(up,un,8), %rax
	cmp	d, %rax
	jae	L(44)
	mov	%rbp, (qp)
	mov	%rax, %rbp
	lea	-8(qp), qp
	je	L(ret)
	dec	un
L(44):
	bsr	d, %rcx
	not	R32(%rcx)
	shl	R8(%rcx), d
	shl	R8(%rcx), %rbp

	push	%rcx
IFSTD(`	push	%rdi		')
IFSTD(`	push	%rsi		')
	push	%r8
IFSTD(`	sub	$8, %rsp	')
IFSTD(`	mov	d, %rdi		')
IFDOS(`	sub	$40, %rsp	')
IFDOS(`	mov	d, %rcx		')
	ASSERT(nz, `test $15, %rsp')
	CALL(	mpn_invert_limb)
IFSTD(`	add	$8, %rsp	')
IFDOS(`	add	$40, %rsp	')
	pop	%r8
IFSTD(`	pop	%rsi		')
IFSTD(`	pop	%rdi		')
	pop	%rcx

	mov	%rax, dinv
	mov	%rbp, %rax
	test	un, un
	je	L(frac)

L(uent):dec	un
	mov	(up,un,8), %rbp
	neg	R32(%rcx)
	shr	R8(%rcx), %rbp
	neg	R32(%rcx)
	or	%rbp, %rax
	jmp	L(ent)

	ALIGN(16)
L(utop):mov	(up,un,8), %r10
	shl	R8(%rcx), %rbp
	neg	R32(%rcx)
	shr	R8(%rcx), %r10
	neg	R32(%rcx)
	or	%r10, %rbp
	mul	dinv
	add	%rbp, %rax
	adc	%r11, %rdx
	mov	%rax, %r11
	mov	%rdx, %r13
	imul	d, %rdx
	sub	%rdx, %rbp
	mov	d, %rax
	add	%rbp, %rax
	cmp	%r11, %rbp
	cmovc	%rbp, %rax
	adc	$-1, %r13
	cmp	d, %rax
	jae	L(ufx)
L(uok):	mov	%r13, (qp)
	sub	$8, qp
L(ent):	mov	(up,un,8), %rbp
	dec	un
	lea	1(%rax), %r11
	jns	L(utop)

L(uend):shl	R8(%rcx), %rbp
	mul	dinv
	add	%rbp, %rax
	adc	%r11, %rdx
	mov	%rax, %r11
	mov	%rdx, %r13
	imul	d, %rdx
	sub	%rdx, %rbp
	mov	d, %rax
	add	%rbp, %rax
	cmp	%r11, %rbp
	cmovc	%rbp, %rax
	adc	$-1, %r13
	cmp	d, %rax
	jae	L(efx)
L(eok):	mov	%r13, (qp)
	sub	$8, qp
	jmp	L(frac)

L(ufx):	sub	d, %rax
	inc	%r13
	jmp	L(uok)
L(efx):	sub	d, %rax
	inc	%r13
	jmp	L(eok)

L(frac):mov	d, %rbp
	neg	%rbp
	jmp	L(fent)

	ALIGN(16)			C	    K8-K10  P6-CNR P6-NHM  P4
L(ftop):mul	dinv			C	      0,12   0,17   0,17
	add	%r11, %rdx		C	      5      8     10
	mov	%rax, %r11		C	      4      8      3
	mov	%rdx, %r13		C	      6      9     11
	imul	%rbp, %rdx		C	      6      9     11
	mov	d, %rax			C
	add	%rdx, %rax		C	     10     14     14
	cmp	%r11, %rdx		C	     10     14     14
	cmovc	%rdx, %rax		C	     11     15     15
	adc	$-1, %r13		C
	mov	%r13, (qp)		C
	sub	$8, qp			C
L(fent):lea	1(%rax), %r11		C
	dec	fn			C
	jns	L(ftop)			C

	shr	R8(%rcx), %rax
L(ret):	pop	%rbx
	pop	%rbp
	pop	%r12
	pop	%r13
	FUNC_EXIT()
	ret
EPILOGUE()
