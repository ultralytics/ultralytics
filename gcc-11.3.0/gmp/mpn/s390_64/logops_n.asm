dnl  S/390-64 logops.

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
C z900		 4.5	 2.25		 5.5		 5.5
C z990		 2.75	 2		 3.25		 3.25
C z9		 ?			 ?		 ?
C z10		 3.25			 3.75		 3.75
C z196		 ?			 ?		 ?

C INPUT PARAMETERS
define(`rp',	`%r2')
define(`up',	`%r3')
define(`vp',	`%r4')
define(`n',	`%r5')

ifdef(`OPERATION_and_n',`
  define(`func',`mpn_and_n')
  define(`VARIANT_1')
  define(`LOGOPC',`nc')
  define(`LOGOP',`ng')')
ifdef(`OPERATION_andn_n',`
  define(`func',`mpn_andn_n')
  define(`VARIANT_2')
  define(`LOGOP',`ng')')
ifdef(`OPERATION_nand_n',`
  define(`func',`mpn_nand_n')
  define(`VARIANT_3')
  define(`LOGOP',`ng')')
ifdef(`OPERATION_ior_n',`
  define(`func',`mpn_ior_n')
  define(`VARIANT_1')
  define(`LOGOPC',`oc')
  define(`LOGOP',`og')')
ifdef(`OPERATION_iorn_n',`
  define(`func',`mpn_iorn_n')
  define(`VARIANT_2')
  define(`LOGOP',`og')')
ifdef(`OPERATION_nior_n',`
  define(`func',`mpn_nior_n')
  define(`VARIANT_3')
  define(`LOGOP',`og')')
ifdef(`OPERATION_xor_n',`
  define(`func',`mpn_xor_n')
  define(`VARIANT_1')
  define(`LOGOPC',`xc')
  define(`LOGOP',`xg')')
ifdef(`OPERATION_xnor_n',`
  define(`func',`mpn_xnor_n')
  define(`VARIANT_2')
  define(`LOGOP',`xg')')

MULFUNC_PROLOGUE(mpn_and_n mpn_andn_n mpn_nand_n mpn_ior_n mpn_iorn_n mpn_nior_n mpn_xor_n mpn_xnor_n)

ASM_START()
PROLOGUE(func)
ifdef(`VARIANT_1',`
	cgr	rp, up
	jne	L(normal)

	sllg	n, n, 3
	aghi	n, -1
	srlg	%r1, n, 8
	ltgr	%r1, %r1		C < 256 bytes to copy?
	je	L(1)

L(tp):	LOGOPC	0(256, rp), 0(vp)
	la	rp, 256(rp)
	la	vp, 256(vp)
	brctg	%r1, L(tp)

L(1):	bras	%r1, L(2)		C make r1 point to mvc insn
	LOGOPC	0(1, rp), 0(vp)
L(2):	ex	n, 0(%r1)		C execute mvc with length ((n-1) mod 256)+1
L(rtn):	br	%r14


L(normal):
	stmg	%r6, %r8, 48(%r15)
	aghi	n, 3
	lghi	%r7, 3
	srlg	%r0, n, 2
	ngr	%r7, n			C n mod 4
	je	L(b1)
	cghi	%r7, 2
	jl	L(b2)
	jne	L(top)

L(b3):	lmg	%r5, %r7, 0(up)
	la	up, 24(up)
	LOGOP	%r5, 0(vp)
	LOGOP	%r6, 8(vp)
	LOGOP	%r7, 16(vp)
	stmg	%r5, %r7, 0(rp)
	la	rp, 24(rp)
	la	vp, 24(vp)
	j	L(mid)

L(b1):	lg	%r5, 0(up)
	la	up, 8(up)
	LOGOP	%r5, 0(vp)
	stg	%r5, 0(rp)
	la	rp, 8(rp)
	la	vp, 8(vp)
	j	L(mid)

L(b2):	lmg	%r5, %r6, 0(up)
	la	up, 16(up)
	LOGOP	%r5, 0(vp)
	LOGOP	%r6, 8(vp)
	stmg	%r5, %r6, 0(rp)
	la	rp, 16(rp)
	la	vp, 16(vp)
	j	L(mid)

L(top):	lmg	%r5, %r8, 0(up)
	la	up, 32(up)
	LOGOP	%r5, 0(vp)
	LOGOP	%r6, 8(vp)
	LOGOP	%r7, 16(vp)
	LOGOP	%r8, 24(vp)
	stmg	%r5, %r8, 0(rp)
	la	rp, 32(rp)
	la	vp, 32(vp)
L(mid):	brctg	%r0, L(top)

	lmg	%r6, %r8, 48(%r15)
	br	%r14
')

ifdef(`VARIANT_2',`
	stmg	%r6, %r8, 48(%r15)
	lghi	%r1, -1

	aghi	n, 3
	lghi	%r7, 3
	srlg	%r0, n, 2
	ngr	%r7, n			C n mod 4
	je	L(b1)
	cghi	%r7, 2
	jl	L(b2)
	jne	L(top)

L(b3):	lmg	%r5, %r7, 0(vp)
	la	vp, 24(vp)
	xgr	%r5, %r1
	xgr	%r6, %r1
	xgr	%r7, %r1
	LOGOP	%r5, 0(up)
	LOGOP	%r6, 8(up)
	LOGOP	%r7, 16(up)
	stmg	%r5, %r7, 0(rp)
	la	rp, 24(rp)
	la	up, 24(up)
	j	L(mid)

L(b1):	lg	%r5, 0(vp)
	la	vp, 8(vp)
	xgr	%r5, %r1
	LOGOP	%r5, 0(up)
	stg	%r5, 0(rp)
	la	rp, 8(rp)
	la	up, 8(up)
	j	L(mid)

L(b2):	lmg	%r5, %r6, 0(vp)
	la	vp, 16(vp)
	xgr	%r5, %r1
	xgr	%r6, %r1
	LOGOP	%r5, 0(up)
	LOGOP	%r6, 8(up)
	stmg	%r5, %r6, 0(rp)
	la	rp, 16(rp)
	la	up, 16(up)
	j	L(mid)

L(top):	lmg	%r5, %r8, 0(vp)
	la	vp, 32(vp)
	xgr	%r5, %r1
	xgr	%r6, %r1
	xgr	%r7, %r1
	xgr	%r8, %r1
	LOGOP	%r5, 0(up)
	LOGOP	%r6, 8(up)
	LOGOP	%r7, 16(up)
	LOGOP	%r8, 24(up)
	la	up, 32(up)
	stmg	%r5, %r8, 0(rp)
	la	rp, 32(rp)
L(mid):	brctg	%r0, L(top)

	lmg	%r6, %r8, 48(%r15)
	br	%r14
')

ifdef(`VARIANT_3',`
	stmg	%r6, %r8, 48(%r15)
	lghi	%r1, -1

	aghi	n, 3
	lghi	%r7, 3
	srlg	%r0, n, 2
	ngr	%r7, n			C n mod 4
	je	L(b1)
	cghi	%r7, 2
	jl	L(b2)
	jne	L(top)

L(b3):	lmg	%r5, %r7, 0(vp)
	la	vp, 24(vp)
	LOGOP	%r5, 0(up)
	LOGOP	%r6, 8(up)
	xgr	%r5, %r1
	xgr	%r6, %r1
	LOGOP	%r7, 16(up)
	xgr	%r7, %r1
	stmg	%r5, %r7, 0(rp)
	la	rp, 24(rp)
	la	up, 24(up)
	j	L(mid)

L(b1):	lg	%r5, 0(vp)
	la	vp, 8(vp)
	LOGOP	%r5, 0(up)
	xgr	%r5, %r1
	stg	%r5, 0(rp)
	la	rp, 8(rp)
	la	up, 8(up)
	j	L(mid)

L(b2):	lmg	%r5, %r6, 0(vp)
	la	vp, 16(vp)
	LOGOP	%r5, 0(up)
	LOGOP	%r6, 8(up)
	xgr	%r5, %r1
	xgr	%r6, %r1
	stmg	%r5, %r6, 0(rp)
	la	rp, 16(rp)
	la	up, 16(up)
	j	L(mid)

L(top):	lmg	%r5, %r8, 0(vp)
	la	vp, 32(vp)
	LOGOP	%r5, 0(up)
	LOGOP	%r6, 8(up)
	xgr	%r5, %r1
	xgr	%r6, %r1
	LOGOP	%r7, 16(up)
	LOGOP	%r8, 24(up)
	xgr	%r7, %r1
	xgr	%r8, %r1
	stmg	%r5, %r8, 0(rp)
	la	up, 32(up)
	la	rp, 32(rp)
L(mid):	brctg	%r0, L(top)

	lmg	%r6, %r8, 48(%r15)
	br	%r14
')

EPILOGUE()
