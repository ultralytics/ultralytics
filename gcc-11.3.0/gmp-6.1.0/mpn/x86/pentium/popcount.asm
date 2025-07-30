dnl  Intel P5 mpn_popcount -- mpn bit population count.

dnl  Copyright 2001, 2002, 2014, 2015 Free Software Foundation, Inc.

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


C P5: 8.0 cycles/limb


C unsigned long mpn_popcount (mp_srcptr src, mp_size_t size);
C
C An arithmetic approach has been found to be slower than the table lookup,
C due to needing too many instructions.

C The slightly strange quoting here helps the renaming done by tune/many.pl.
deflit(TABLE_NAME,
m4_assert_defined(`GSYM_PREFIX')
GSYM_PREFIX`'mpn_popcount``'_table')

C FIXME: exporting the table to hamdist is incorrect as it hurt incremental
C linking.

	RODATA
	ALIGN(8)
	GLOBL	TABLE_NAME
TABLE_NAME:
forloop(i,0,255,
`	.byte	m4_popcount(i)
')

defframe(PARAM_SIZE,8)
defframe(PARAM_SRC, 4)

	TEXT
	ALIGN(8)

PROLOGUE(mpn_popcount)
deflit(`FRAME',0)

	movl	PARAM_SIZE, %ecx
	pushl	%esi	FRAME_pushl()

ifdef(`PIC',`
	pushl	%ebx	FRAME_pushl()
	pushl	%ebp	FRAME_pushl()
ifdef(`DARWIN',`
	shll	%ecx		C size in byte pairs
	LEA(	TABLE_NAME, %ebp)
	movl	PARAM_SRC, %esi
	xorl	%eax, %eax	C total
	xorl	%ebx, %ebx	C byte
	xorl	%edx, %edx	C byte
',`
	call	L(here)
L(here):
	popl	%ebp
	shll	%ecx		C size in byte pairs

	addl	$_GLOBAL_OFFSET_TABLE_+[.-L(here)], %ebp
	movl	PARAM_SRC, %esi

	xorl	%eax, %eax	C total
	xorl	%ebx, %ebx	C byte

	movl	TABLE_NAME@GOT(%ebp), %ebp
	xorl	%edx, %edx	C byte
')
define(TABLE,`(%ebp,$1)')
',`
dnl non-PIC
	shll	%ecx		C size in byte pairs
	movl	PARAM_SRC, %esi

	pushl	%ebx	FRAME_pushl()
	xorl	%eax, %eax	C total

	xorl	%ebx, %ebx	C byte
	xorl	%edx, %edx	C byte

define(TABLE,`TABLE_NAME`'($1)')
')


	ALIGN(8)	C necessary on P55 for claimed speed
L(top):
	C eax	total
	C ebx	byte
	C ecx	counter, 2*size to 2
	C edx	byte
	C esi	src
	C edi
	C ebp	[PIC] table

	addl	%ebx, %eax
	movb	-1(%esi,%ecx,2), %bl

	addl	%edx, %eax
	movb	-2(%esi,%ecx,2), %dl

	movb	TABLE(%ebx), %bl
	decl	%ecx

	movb	TABLE(%edx), %dl
	jnz	L(top)


ifdef(`PIC',`
	popl	%ebp
')
	addl	%ebx, %eax
	popl	%ebx

	addl	%edx, %eax
	popl	%esi

	ret

EPILOGUE()
ASM_END()
