dnl  Intel P5 mpn_hamdist -- mpn hamming distance.

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


C P5: 14.0 cycles/limb


C unsigned long mpn_hamdist (mp_srcptr src1, mp_srcptr src2, mp_size_t size);
C
C It might be possible to shave 1 cycle from the loop, and hence 2
C cycles/limb.  The xorb is taking 2 cycles, but a separate load and xor
C would be 1, if the right schedule could be found (not found so far).
C Wanting to avoid potential cache bank clashes makes it tricky.

C The slightly strange quoting here helps the renaming done by tune/many.pl.
deflit(TABLE_NAME,
m4_assert_defined(`GSYM_PREFIX')
GSYM_PREFIX`'mpn_popcount``'_table')

C FIXME: referencing popcount.asm's table is incorrect as it hurt incremental
C linking.

defframe(PARAM_SIZE,12)
defframe(PARAM_SRC2, 8)
defframe(PARAM_SRC1, 4)

	TEXT
	ALIGN(8)

PROLOGUE(mpn_hamdist)
deflit(`FRAME',0)

	movl	PARAM_SIZE, %ecx
	pushl	%esi	FRAME_pushl()

	shll	%ecx		C size in byte pairs
	pushl	%edi	FRAME_pushl()

ifdef(`PIC',`
	pushl	%ebx	FRAME_pushl()
	pushl	%ebp	FRAME_pushl()
ifdef(`DARWIN',`
	movl	PARAM_SRC1, %esi
	movl	PARAM_SRC2, %edi
	LEA(	TABLE_NAME, %ebp)
	xorl	%ebx, %ebx	C byte
	xorl	%edx, %edx	C byte
	xorl	%eax, %eax	C total
',`
	call	L(here)	FRAME_pushl()
L(here):
	movl	PARAM_SRC1, %esi
	popl	%ebp	FRAME_popl()

	movl	PARAM_SRC2, %edi
	addl	$_GLOBAL_OFFSET_TABLE_+[.-L(here)], %ebp

	xorl	%ebx, %ebx	C byte
	xorl	%edx, %edx	C byte

	movl	TABLE_NAME@GOT(%ebp), %ebp
	xorl	%eax, %eax	C total
')
define(TABLE,`(%ebp,$1)')
',`
dnl non-PIC
	movl	PARAM_SRC1, %esi
	movl	PARAM_SRC2, %edi

	xorl	%eax, %eax	C total
	pushl	%ebx	FRAME_pushl()

	xorl	%edx, %edx	C byte
	xorl	%ebx, %ebx	C byte

define(TABLE,`TABLE_NAME($1)')
')


	C The nop after the xorb seems necessary.  Although a movb might be
	C expected to go down the V pipe in the second cycle of the xorb, it
	C doesn't and costs an extra 2 cycles.
L(top):
	C eax	total
	C ebx	byte
	C ecx	counter, 2*size to 2
	C edx	byte
	C esi	src1
	C edi	src2
	C ebp	[PIC] table

	addl	%ebx, %eax
	movb	-1(%esi,%ecx,2), %bl

	addl	%edx, %eax
	movb	-1(%edi,%ecx,2), %dl

	xorb	%dl, %bl
	movb	-2(%esi,%ecx,2), %dl

	xorb	-2(%edi,%ecx,2), %dl
	nop

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
	popl	%edi

	popl	%esi

	ret

EPILOGUE()
ASM_END()
