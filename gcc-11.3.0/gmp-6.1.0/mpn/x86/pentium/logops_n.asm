dnl  Intel Pentium mpn_and_n,...,mpn_xnor_n -- bitwise logical operations.

dnl  Copyright 2001, 2002 Free Software Foundation, Inc.

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


C P5: 3.0 c/l  and, ior, xor
C     3.5 c/l  andn, iorn, nand, nior, xnor


define(M4_choose_op,
`ifdef(`OPERATION_$1',`
define(`M4_function', `mpn_$1')
define(`M4_want_pre', `$4')
define(`M4op',        `$3')
define(`M4_want_post',`$2')
')')
define(M4pre, `ifelse(M4_want_pre, yes,`$1')')
define(M4post,`ifelse(M4_want_post,yes,`$1')')

M4_choose_op( and_n,     , andl,    )
M4_choose_op( andn_n,    , andl, yes)
M4_choose_op( nand_n, yes, andl,    )
M4_choose_op( ior_n,     ,  orl,    )
M4_choose_op( iorn_n,    ,  orl, yes)
M4_choose_op( nior_n, yes,  orl,    )
M4_choose_op( xor_n,     , xorl,    )
M4_choose_op( xnor_n, yes, xorl,    )

ifdef(`M4_function',,
`m4_error(`Unrecognised or undefined OPERATION symbol
')')

MULFUNC_PROLOGUE(mpn_and_n mpn_andn_n mpn_nand_n mpn_ior_n mpn_iorn_n mpn_nior_n mpn_xor_n mpn_xnor_n)

NAILS_SUPPORT(0-31)


C void M4_function (mp_ptr wp, mp_srcptr xp, mp_srcptr yp, mp_size_t size);
C
C Nothing complicated here, just some care to avoid data cache bank clashes
C and AGIs.
C
C We're one register short of being able to do a simple 4 loads, 2 ops, 2
C stores.  Instead %ebp is juggled a bit and nops are introduced to keep the
C pairings as intended.  An in-place operation would free up a register, for
C an 0.5 c/l speedup, if that's worth bothering with.
C
C This code seems best for P55 too.  Data alignment is a big problem for MMX
C and the pairing restrictions on movq and integer instructions make life
C difficult.

defframe(PARAM_SIZE,16)
defframe(PARAM_YP,  12)
defframe(PARAM_XP,   8)
defframe(PARAM_WP,   4)

	TEXT
	ALIGN(8)

PROLOGUE(M4_function)
deflit(`FRAME',0)

	pushl	%ebx	FRAME_pushl()
	pushl	%esi	FRAME_pushl()

	pushl	%edi	FRAME_pushl()
	pushl	%ebp	FRAME_pushl()

	movl	PARAM_SIZE, %ecx
	movl	PARAM_XP, %ebx

	movl	PARAM_YP, %esi
	movl	PARAM_WP, %edi

	shrl	%ecx
	jnc	L(entry)

	movl	(%ebx,%ecx,8), %eax	C risk of data cache bank clash here
	movl	(%esi,%ecx,8), %edx

M4pre(`	notl_or_xorl_GMP_NUMB_MASK(%edx)')

	M4op	%edx, %eax

M4post(`xorl	$GMP_NUMB_MASK, %eax')
	orl	%ecx, %ecx

	movl	%eax, (%edi,%ecx,8)
	jz	L(done)

	jmp	L(entry)


L(top):
	C eax
	C ebx	xp
	C ecx	counter, limb pairs, decrementing
	C edx
	C esi	yp
	C edi	wp
	C ebp

	M4op	%ebp, %edx
	nop

M4post(`xorl	$GMP_NUMB_MASK, %eax')
M4post(`xorl	$GMP_NUMB_MASK, %edx')

	movl	%eax, 4(%edi,%ecx,8)
	movl	%edx, (%edi,%ecx,8)

L(entry):
	movl	-4(%ebx,%ecx,8), %ebp
	nop

	movl	-4(%esi,%ecx,8), %eax
	movl	-8(%esi,%ecx,8), %edx

M4pre(`	xorl	$GMP_NUMB_MASK, %eax')
M4pre(`	xorl	$GMP_NUMB_MASK, %edx')

	M4op	%ebp, %eax
	movl	-8(%ebx,%ecx,8), %ebp

	decl	%ecx
	jnz	L(top)


	M4op	%ebp, %edx
	nop

M4post(`xorl	$GMP_NUMB_MASK, %eax')
M4post(`xorl	$GMP_NUMB_MASK, %edx')

	movl	%eax, 4(%edi,%ecx,8)
	movl	%edx, (%edi,%ecx,8)


L(done):
	popl	%ebp
	popl	%edi

	popl	%esi
	popl	%ebx

	ret

EPILOGUE()
