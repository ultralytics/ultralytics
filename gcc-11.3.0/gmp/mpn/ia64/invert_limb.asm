dnl  IA-64 mpn_invert_limb -- Invert a normalized limb.

dnl  Contributed to the GNU project by Torbjorn Granlund and Kevin Ryde.

dnl  Copyright 2000, 2002, 2004 Free Software Foundation, Inc.

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

C INPUT PARAMETERS
C d = r32

C           cycles
C Itanium:    74
C Itanium 2:  50+6

C It should be possible to avoid the xmpy.hu and the following tests by
C explicitly chopping in the last fma.  That would save about 10 cycles.

ASM_START()
	.sdata
	.align 16
ifdef(`HAVE_DOUBLE_IEEE_LITTLE_ENDIAN',`
.LC0:	data4 0x00000000, 0x80000000, 0x0000403f, 0x00000000	C 2^64
.LC1:	data4 0x00000000, 0x80000000, 0x0000407f, 0x00000000	C 2^128

',`ifdef(`HAVE_DOUBLE_IEEE_BIG_ENDIAN',`
.LC0:	data4 0x403f8000, 0x00000000, 0x00000000, 0x00000000	C 2^64
.LC1:	data4 0x407f8000, 0x00000000, 0x00000000, 0x00000000	C 2^128

',`m4_error(`Oops, need to know float endianness
')')')


PROLOGUE(mpn_invert_limb)
		C 00
	addl		r14 = @gprel(.LC0), gp
	addl		r15 = @gprel(.LC1), gp
	setf.sig	f7 = r32
	add		r9 = r32, r32		C check for d = 2^63
	;;	C 01
	ldfe		f10 = [r14]		C 2^64
	ldfe		f8 = [r15]		C 2^128
	cmp.eq		p6, p0 = 0, r9		C check for d = 2^63
	mov		r8 = -1			C retval for 2^63
   (p6)	br.ret.spnt.many b0
	;;	C 07
	fmpy.s1		f11 = f7, f10		C f11 = d * 2^64
	fnma.s1		f6 = f7, f10, f8	C f6 = 2^128 - d * 2^64
	;;	C 11
	frcpa.s1	f8, p6 = f6, f7
	;;	C 15
   (p6)	fnma.s1		f9 = f7, f8, f1
   (p6)	fmpy.s1		f10 = f6, f8
	;;	C 19
   (p6)	fmpy.s1		f11 = f9, f9
   (p6)	fma.s1		f10 = f9, f10, f10
	;;	C 23
   (p6)	fma.s1		f8 = f9, f8, f8
   (p6)	fma.s1		f9 = f11, f10, f10
	;;	C 27
   (p6)	fma.s1		f8 = f11, f8, f8
   (p6)	fnma.s1		f10 = f7, f9, f6
	;;	C 31
   (p6)	fma.s1		f8 = f10, f8, f9
	;;	C 35
	fcvt.fxu.trunc.s1 f8 = f8
	;;	C 39
	getf.sig	r8 = f8
	xmpy.hu		f10 = f8, f7		C di * d
	;;	C 43
	getf.sig	r14 = f10
	andcm		r9 = -1, r32		C one's complement
	;;	C 48
	cmp.ltu		p6, p0 = r9, r14	C got overflow?
	;;	C 49
   (p6)	add		r8 = -1, r8		C adjust di down
	br.ret.sptk.many b0
EPILOGUE()
ASM_END()
