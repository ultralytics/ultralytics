dnl  S/390-32 logops.

dnl  Copyright 2011 Free Software Foundation, Inc.

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

C cycles/limb     variant 1           variant 2       variant 3
C	        rp!=up  rp=up
C z900		 ?	 ?		 ?		 ?
C z990		 2.5	 1		 2.75		 2.75
C z9		 ?			 ?		 ?
C z10		 ?			 ?		 ?
C z196		 ?			 ?		 ?

C INPUT PARAMETERS
define(`rp',	`%r2')
define(`up',	`%r3')
define(`vp',	`%r4')
define(`nn',	`%r5')

ifdef(`OPERATION_and_n',`
  define(`func',`mpn_and_n')
  define(`VARIANT_1')
  define(`LOGOPC',`nc')
  define(`LOGOP',`n')')
ifdef(`OPERATION_andn_n',`
  define(`func',`mpn_andn_n')
  define(`VARIANT_2')
  define(`LOGOP',`n')')
ifdef(`OPERATION_nand_n',`
  define(`func',`mpn_nand_n')
  define(`VARIANT_3')
  define(`LOGOP',`n')')
ifdef(`OPERATION_ior_n',`
  define(`func',`mpn_ior_n')
  define(`VARIANT_1')
  define(`LOGOPC',`oc')
  define(`LOGOP',`o')')
ifdef(`OPERATION_iorn_n',`
  define(`func',`mpn_iorn_n')
  define(`VARIANT_2')
  define(`LOGOP',`o')')
ifdef(`OPERATION_nior_n',`
  define(`func',`mpn_nior_n')
  define(`VARIANT_3')
  define(`LOGOP',`o')')
ifdef(`OPERATION_xor_n',`
  define(`func',`mpn_xor_n')
  define(`VARIANT_1')
  define(`LOGOPC',`xc')
  define(`LOGOP',`x')')
ifdef(`OPERATION_xnor_n',`
  define(`func',`mpn_xnor_n')
  define(`VARIANT_2')
  define(`LOGOP',`x')')

MULFUNC_PROLOGUE(mpn_and_n mpn_andn_n mpn_nand_n mpn_ior_n mpn_iorn_n mpn_nior_n mpn_xor_n mpn_xnor_n)

ASM_START()
PROLOGUE(func)
ifdef(`VARIANT_1',`
	cr	rp, up
	jne	L(normal)

	sll	nn, 2
	ahi	nn, -1
	lr	%r1, nn
	srl	%r1, 8
	ltr	%r1, %r1		C < 256 bytes to copy?
	je	L(1)

L(tp):	LOGOPC	0(256, rp), 0(vp)
	la	rp, 256(rp)
	la	vp, 256(vp)
	brct	%r1, L(tp)

L(1):	bras	%r1, L(2)		C make r1 point to mvc insn
	LOGOPC	0(1, rp), 0(vp)
L(2):	ex	nn, 0(%r1)		C execute mvc with length ((nn-1) mod 256)+1
L(rtn):	br	%r14


L(normal):
	stm	%r6, %r8, 12(%r15)
	ahi	nn, 3
	lhi	%r7, 3
	lr	%r0, nn
	srl	%r0, 2
	nr	%r7, nn			C nn mod 4
	je	L(b1)
	chi	%r7, 2
	jl	L(b2)
	jne	L(top)

L(b3):	lm	%r5, %r7, 0(up)
	la	up, 12(up)
	LOGOP	%r5, 0(vp)
	LOGOP	%r6, 4(vp)
	LOGOP	%r7, 8(vp)
	stm	%r5, %r7, 0(rp)
	la	rp, 12(rp)
	la	vp, 12(vp)
	j	L(mid)

L(b1):	l	%r5, 0(up)
	la	up, 4(up)
	LOGOP	%r5, 0(vp)
	st	%r5, 0(rp)
	la	rp, 4(rp)
	la	vp, 4(vp)
	j	L(mid)

L(b2):	lm	%r5, %r6, 0(up)
	la	up, 8(up)
	LOGOP	%r5, 0(vp)
	LOGOP	%r6, 4(vp)
	stm	%r5, %r6, 0(rp)
	la	rp, 8(rp)
	la	vp, 8(vp)
	j	L(mid)

L(top):	lm	%r5, %r8, 0(up)
	la	up, 16(up)
	LOGOP	%r5, 0(vp)
	LOGOP	%r6, 4(vp)
	LOGOP	%r7, 8(vp)
	LOGOP	%r8, 12(vp)
	stm	%r5, %r8, 0(rp)
	la	rp, 16(rp)
	la	vp, 16(vp)
L(mid):	brct	%r0, L(top)

	lm	%r6, %r8, 12(%r15)
	br	%r14
')

ifdef(`VARIANT_2',`
	stm	%r6, %r8, 12(%r15)
	lhi	%r1, -1

	ahi	nn, 3
	lhi	%r7, 3
	lr	%r0, nn
	srl	%r0, 2
	nr	%r7, nn			C nn mod 4
	je	L(b1)
	chi	%r7, 2
	jl	L(b2)
	jne	L(top)

L(b3):	lm	%r5, %r7, 0(vp)
	la	vp, 12(vp)
	xr	%r5, %r1
	xr	%r6, %r1
	xr	%r7, %r1
	LOGOP	%r5, 0(up)
	LOGOP	%r6, 4(up)
	LOGOP	%r7, 8(up)
	stm	%r5, %r7, 0(rp)
	la	rp, 12(rp)
	la	up, 12(up)
	j	L(mid)

L(b1):	l	%r5, 0(vp)
	la	vp, 4(vp)
	xr	%r5, %r1
	LOGOP	%r5, 0(up)
	st	%r5, 0(rp)
	la	rp, 4(rp)
	la	up, 4(up)
	j	L(mid)

L(b2):	lm	%r5, %r6, 0(vp)
	la	vp, 8(vp)
	xr	%r5, %r1
	xr	%r6, %r1
	LOGOP	%r5, 0(up)
	LOGOP	%r6, 4(up)
	stm	%r5, %r6, 0(rp)
	la	rp, 8(rp)
	la	up, 8(up)
	j	L(mid)

L(top):	lm	%r5, %r8, 0(vp)
	la	vp, 16(vp)
	xr	%r5, %r1
	xr	%r6, %r1
	xr	%r7, %r1
	xr	%r8, %r1
	LOGOP	%r5, 0(up)
	LOGOP	%r6, 4(up)
	LOGOP	%r7, 8(up)
	LOGOP	%r8, 12(up)
	la	up, 16(up)
	stm	%r5, %r8, 0(rp)
	la	rp, 16(rp)
L(mid):	brct	%r0, L(top)

	lm	%r6, %r8, 12(%r15)
	br	%r14
')

ifdef(`VARIANT_3',`
	stm	%r6, %r8, 12(%r15)
	lhi	%r1, -1

	ahi	nn, 3
	lhi	%r7, 3
	lr	%r0, nn
	srl	%r0, 2
	nr	%r7, nn			C nn mod 4
	je	L(b1)
	chi	%r7, 2
	jl	L(b2)
	jne	L(top)

L(b3):	lm	%r5, %r7, 0(vp)
	la	vp, 12(vp)
	LOGOP	%r5, 0(up)
	LOGOP	%r6, 4(up)
	xr	%r5, %r1
	xr	%r6, %r1
	LOGOP	%r7, 8(up)
	xr	%r7, %r1
	stm	%r5, %r7, 0(rp)
	la	rp, 12(rp)
	la	up, 12(up)
	j	L(mid)

L(b1):	l	%r5, 0(vp)
	la	vp, 4(vp)
	LOGOP	%r5, 0(up)
	xr	%r5, %r1
	st	%r5, 0(rp)
	la	rp, 4(rp)
	la	up, 4(up)
	j	L(mid)

L(b2):	lm	%r5, %r6, 0(vp)
	la	vp, 8(vp)
	LOGOP	%r5, 0(up)
	LOGOP	%r6, 4(up)
	xr	%r5, %r1
	xr	%r6, %r1
	stm	%r5, %r6, 0(rp)
	la	rp, 8(rp)
	la	up, 8(up)
	j	L(mid)

L(top):	lm	%r5, %r8, 0(vp)
	la	vp, 16(vp)
	LOGOP	%r5, 0(up)
	LOGOP	%r6, 4(up)
	xr	%r5, %r1
	xr	%r6, %r1
	LOGOP	%r7, 8(up)
	LOGOP	%r8, 12(up)
	xr	%r7, %r1
	xr	%r8, %r1
	stm	%r5, %r8, 0(rp)
	la	up, 16(up)
	la	rp, 16(rp)
L(mid):	brct	%r0, L(top)

	lm	%r6, %r8, 12(%r15)
	br	%r14
')

EPILOGUE()
