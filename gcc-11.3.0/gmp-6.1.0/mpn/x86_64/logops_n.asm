dnl  AMD64 logops.

dnl  Copyright 2004-2006, 2011, 2012 Free Software Foundation, Inc.

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
C AMD K8,K9	 1.5	with fluctuations for variant 2 and 3
C AMD K10	 1.5	with fluctuations for all variants
C Intel P4	 2.8/3.35/3.60 (variant1/variant2/variant3)
C Intel core2	 2
C Intel NHM	 2
C Intel SBR	 1.5/1.75/1.75
C Intel atom	 3.75
C VIA nano	 3.25

ifdef(`OPERATION_and_n',`
  define(`func',`mpn_and_n')
  define(`VARIANT_1')
  define(`LOGOP',`andq')')
ifdef(`OPERATION_andn_n',`
  define(`func',`mpn_andn_n')
  define(`VARIANT_2')
  define(`LOGOP',`andq')')
ifdef(`OPERATION_nand_n',`
  define(`func',`mpn_nand_n')
  define(`VARIANT_3')
  define(`LOGOP',`andq')')
ifdef(`OPERATION_ior_n',`
  define(`func',`mpn_ior_n')
  define(`VARIANT_1')
  define(`LOGOP',`orq')')
ifdef(`OPERATION_iorn_n',`
  define(`func',`mpn_iorn_n')
  define(`VARIANT_2')
  define(`LOGOP',`orq')')
ifdef(`OPERATION_nior_n',`
  define(`func',`mpn_nior_n')
  define(`VARIANT_3')
  define(`LOGOP',`orq')')
ifdef(`OPERATION_xor_n',`
  define(`func',`mpn_xor_n')
  define(`VARIANT_1')
  define(`LOGOP',`xorq')')
ifdef(`OPERATION_xnor_n',`
  define(`func',`mpn_xnor_n')
  define(`VARIANT_2')
  define(`LOGOP',`xorq')')


MULFUNC_PROLOGUE(mpn_and_n mpn_andn_n mpn_nand_n mpn_ior_n mpn_iorn_n mpn_nior_n mpn_xor_n mpn_xnor_n)

C INPUT PARAMETERS
define(`rp',`%rdi')
define(`up',`%rsi')
define(`vp',`%rdx')
define(`n',`%rcx')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()

ifdef(`VARIANT_1',`
	TEXT
	ALIGN(32)
PROLOGUE(func)
	FUNC_ENTRY(4)
	movq	(vp), %r8
	movl	R32(%rcx), R32(%rax)
	leaq	(vp,n,8), vp
	leaq	(up,n,8), up
	leaq	(rp,n,8), rp
	negq	n
	andl	$3, R32(%rax)
	je	L(b00)
	cmpl	$2, R32(%rax)
	jc	L(b01)
	je	L(b10)

L(b11):	LOGOP	(up,n,8), %r8
	movq	%r8, (rp,n,8)
	decq	n
	jmp	L(e11)
L(b10):	addq	$-2, n
	jmp	L(e10)
L(b01):	LOGOP	(up,n,8), %r8
	movq	%r8, (rp,n,8)
	incq	n
	jz	L(ret)

L(oop):	movq	(vp,n,8), %r8
L(b00):	movq	8(vp,n,8), %r9
	LOGOP	(up,n,8), %r8
	LOGOP	8(up,n,8), %r9
	nop
	movq	%r8, (rp,n,8)
	movq	%r9, 8(rp,n,8)
L(e11):	movq	16(vp,n,8), %r8
L(e10):	movq	24(vp,n,8), %r9
	LOGOP	16(up,n,8), %r8
	LOGOP	24(up,n,8), %r9
	movq	%r8, 16(rp,n,8)
	movq	%r9, 24(rp,n,8)
	addq	$4, n
	jnc	L(oop)
L(ret):	FUNC_EXIT()
	ret
EPILOGUE()
')

ifdef(`VARIANT_2',`
	TEXT
	ALIGN(32)
PROLOGUE(func)
	FUNC_ENTRY(4)
	movq	(vp), %r8
	notq	%r8
	movl	R32(%rcx), R32(%rax)
	leaq	(vp,n,8), vp
	leaq	(up,n,8), up
	leaq	(rp,n,8), rp
	negq	n
	andl	$3, R32(%rax)
	je	L(b00)
	cmpl	$2, R32(%rax)
	jc	L(b01)
	je	L(b10)

L(b11):	LOGOP	(up,n,8), %r8
	movq	%r8, (rp,n,8)
	decq	n
	jmp	L(e11)
L(b10):	addq	$-2, n
	jmp	L(e10)
	.byte	0x90,0x90,0x90,0x90,0x90,0x90,0x90,0x90,0x90,0x90,0x90,0x90,0x90
L(b01):	LOGOP	(up,n,8), %r8
	movq	%r8, (rp,n,8)
	incq	n
	jz	L(ret)

L(oop):	movq	(vp,n,8), %r8
	notq	%r8
L(b00):	movq	8(vp,n,8), %r9
	notq	%r9
	LOGOP	(up,n,8), %r8
	LOGOP	8(up,n,8), %r9
	movq	%r8, (rp,n,8)
	movq	%r9, 8(rp,n,8)
L(e11):	movq	16(vp,n,8), %r8
	notq	%r8
L(e10):	movq	24(vp,n,8), %r9
	notq	%r9
	LOGOP	16(up,n,8), %r8
	LOGOP	24(up,n,8), %r9
	movq	%r8, 16(rp,n,8)
	movq	%r9, 24(rp,n,8)
	addq	$4, n
	jnc	L(oop)
L(ret):	FUNC_EXIT()
	ret
EPILOGUE()
')

ifdef(`VARIANT_3',`
	TEXT
	ALIGN(32)
PROLOGUE(func)
	FUNC_ENTRY(4)
	movq	(vp), %r8
	movl	R32(%rcx), R32(%rax)
	leaq	(vp,n,8), vp
	leaq	(up,n,8), up
	leaq	(rp,n,8), rp
	negq	n
	andl	$3, R32(%rax)
	je	L(b00)
	cmpl	$2, R32(%rax)
	jc	L(b01)
	je	L(b10)

L(b11):	LOGOP	(up,n,8), %r8
	notq	%r8
	movq	%r8, (rp,n,8)
	decq	n
	jmp	L(e11)
L(b10):	addq	$-2, n
	jmp	L(e10)
	.byte	0x90,0x90,0x90,0x90,0x90,0x90,0x90,0x90,0x90,0x90
L(b01):	LOGOP	(up,n,8), %r8
	notq	%r8
	movq	%r8, (rp,n,8)
	incq	n
	jz	L(ret)

L(oop):	movq	(vp,n,8), %r8
L(b00):	movq	8(vp,n,8), %r9
	LOGOP	(up,n,8), %r8
	notq	%r8
	LOGOP	8(up,n,8), %r9
	notq	%r9
	movq	%r8, (rp,n,8)
	movq	%r9, 8(rp,n,8)
L(e11):	movq	16(vp,n,8), %r8
L(e10):	movq	24(vp,n,8), %r9
	LOGOP	16(up,n,8), %r8
	notq	%r8
	LOGOP	24(up,n,8), %r9
	notq	%r9
	movq	%r8, 16(rp,n,8)
	movq	%r9, 24(rp,n,8)
	addq	$4, n
	jnc	L(oop)
L(ret):	FUNC_EXIT()
	ret
EPILOGUE()
')
