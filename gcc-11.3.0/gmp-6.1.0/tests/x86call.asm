dnl  x86 calling conventions checking.

dnl  Copyright 2000, 2003, 2010, 2013 Free Software Foundation, Inc.

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


C void x86_fldcw (unsigned short cw);
C
C Execute an fldcw, setting the x87 control word to cw.

PROLOGUE(x86_fldcw)
	fldcw	4(%esp)
	ret
EPILOGUE()


C unsigned short x86_fstcw (void);
C
C Execute an fstcw, returning the current x87 control word.

PROLOGUE(x86_fstcw)
	xor	%eax, %eax
	push	%eax
	fstcw	(%esp)
	pop	%eax
	ret
EPILOGUE()


dnl  Instrumented profiling doesn't come out quite right below, since we don't
dnl  do an actual "ret".  There's only a few instructions here, so there's no
dnl  great need to get them separately accounted, just let them get attributed
dnl  to the caller.  FIXME this comment might no longer be true.

ifelse(WANT_PROFILING,instrument,
`define(`WANT_PROFILING',no)')


C int calling_conventions (...);
C
C The global variable "calling_conventions_function" is the function to
C call, with the arguments as passed here.
C
C Perhaps the finit should be done only if the tags word isn't clear, but
C nothing uses the rounding mode or anything at the moment.

define(`WANT_EBX', eval(4*0)($1))
define(`WANT_EBP', eval(4*1)($1))
define(`WANT_ESI', eval(4*2)($1))
define(`WANT_EDI', eval(4*3)($1))

define(`JUNK_EAX', eval(4*4)($1))
define(`JUNK_ECX', eval(4*5)($1))
define(`JUNK_EDX', eval(4*6)($1))

define(`SAVE_EBX', eval(4*7)($1))
define(`SAVE_EBP', eval(4*8)($1))
define(`SAVE_ESI', eval(4*9)($1))
define(`SAVE_EDI', eval(4*10)($1))

define(`RETADDR',  eval(4*11)($1))

define(`EBX',	   eval(4*12)($1))
define(`EBP',	   eval(4*13)($1))
define(`ESI',	   eval(4*14)($1))
define(`EDI',	   eval(4*15)($1))
define(`EFLAGS',   eval(4*16)($1))


define(G,
m4_assert_numargs(1)
`GSYM_PREFIX`'$1')

	TEXT
	ALIGN(8)
PROLOGUE(calling_conventions)
	LEA(	G(calling_conventions_values), %ecx)
	pop	RETADDR(%ecx)

	mov	%ebx, SAVE_EBX(%ecx)
	mov	%ebp, SAVE_EBP(%ecx)
	mov	%esi, SAVE_ESI(%ecx)
	mov	%edi, SAVE_EDI(%ecx)

	C Values we expect to see unchanged, as per amd64check.c
	mov	WANT_EBX(%ecx), %ebx
	mov	WANT_EBP(%ecx), %ebp
	mov	WANT_ESI(%ecx), %esi
	mov	WANT_EDI(%ecx), %edi

	C Try to provoke a problem by starting with junk in the caller-saves
	C registers, especially in %eax and %edx which will be return values
	mov	JUNK_EAX(%ecx), %eax
	mov	JUNK_EDX(%ecx), %edx
C	mov	JUNK_ECX(%ecx), %ecx

ifdef(`PIC',`
	LEA(	G(calling_conventions_function), %ecx)
	call	*(%ecx)
',`
	call	*G(calling_conventions_function)
')

	LEA(	G(calling_conventions_values), %ecx)

	mov	%ebx, EBX(%ecx)
	mov	%ebp, EBP(%ecx)
	mov	%esi, ESI(%ecx)
	mov	%edi, EDI(%ecx)

	pushf
	pop	%ebx
	mov	%ebx, EFLAGS(%ecx)

	mov	SAVE_EBX(%ecx), %ebx
	mov	SAVE_ESI(%ecx), %esi
	mov	SAVE_EDI(%ecx), %edi
	mov	SAVE_EBP(%ecx), %ebp

	push	RETADDR(%ecx)

ifdef(`PIC',`
	LEA(	G(calling_conventions_fenv), %ecx)
	fstenv	(%ecx)
',`
	fstenv	G(calling_conventions_fenv)
')
	finit

	ret

EPILOGUE()
ASM_END()
