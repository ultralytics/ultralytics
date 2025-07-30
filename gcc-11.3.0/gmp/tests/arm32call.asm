dnl  ARM32 calling conventions checking.

dnl  Copyright 2000, 2003, 2004, 2006, 2007, 2010, 2013 Free Software
dnl  Foundation, Inc.

dnl  This file is part of the GNU MP Library test suite.

dnl  The GNU MP Library test suite is free software; you can redistribute it
dnl  and/or modify it under the terms of the GNU General Public License as
dnl  published by the Free Software Foundation; either version 3 of the
dnl  License, or (at your option) any later version.

dnl  The GNU MP Library test suite is distributed in the hope that it will be
dnl  useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
dnl  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
dnl  Public License for more details.

dnl  You should have received a copy of the GNU General Public License along
dnl  with the GNU MP Library test suite.  If not, see
dnl  https://www.gnu.org/licenses/.


dnl  The current version of the code attempts to keep the call/return
dnl  prediction stack valid, but matching calls and returns.

include(`../config.m4')


C int calling_conventions (...);
C
C The global variable "calling_conventions_function" is the function to
C call, with the arguments as passed here.

define(`WANT_CALLEE_SAVES',	eval(4*0))
define(`SAVE_CALLEE_SAVES',	eval(4*8))
define(`RETADDR',		eval(4*16))
define(`GOT_CALLEE_SAVES',	eval(4*17))
define(`JUNK_PARAMS',		eval(4*25))

	TEXT
	ALIGN(32)
PROLOGUE(calling_conventions)
	LEA(	r12, calling_conventions_values)

	C Preserve callee-saves registers, including the link register r14
	add	r12, r12, #SAVE_CALLEE_SAVES
	stm	r12, {r4-r11,r14}
	sub	r12, r12, #SAVE_CALLEE_SAVES

	C Put chosen junk into callee-saves registers
	add	r12, r12, #WANT_CALLEE_SAVES
	ldm	r12, {r4-r11}
	sub	r12, r12, #WANT_CALLEE_SAVES

	C No callee-saves registers on arm except r12 and parameter registers
	C

	C Make the actual call
	LEA(	r12, calling_conventions_function)
	ldr	r12, [r12]
	mov	r14, pc
	bx	r12

	LEA(	r12, calling_conventions_values)

	C Save callee-saves registers after call
	add	r12, r12, #GOT_CALLEE_SAVES
	stm	r12, {r4-r11}
	sub	r12, r12, #GOT_CALLEE_SAVES

	C Restore callee-saves registers, including the link register r14
	add	r12, r12, #SAVE_CALLEE_SAVES
	ldm	r12, {r4-r11,r14}
	sub	r12, r12, #SAVE_CALLEE_SAVES

	C Overwrite parameter registers.  Note that we overwrite r1, which
	C could hold one half of a 64-bit return value, since we don't use that
	C in GMP.
	add	r12, r12, #JUNK_PARAMS
	ldm	r12, {r1-r3}

	bx	r14
EPILOGUE()
