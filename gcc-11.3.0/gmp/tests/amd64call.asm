dnl  AMD64 calling conventions checking.

dnl  Copyright 2000, 2003, 2004, 2006, 2007, 2010 Free Software Foundation, Inc.

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
	mov	%rdi, -8(%rsp)
	fldcw	-8(%rsp)
	ret
EPILOGUE()


C unsigned short x86_fstcw (void);
C
C Execute an fstcw, returning the current x87 control word.

PROLOGUE(x86_fstcw)
	movq	$0, -8(%rsp)
	fstcw	-8(%rsp)
	mov	-8(%rsp), %rax
	ret
EPILOGUE()


dnl  Instrumented profiling won't come out quite right below, since we don't do
dnl  an actual "ret".  There's only a few instructions here, so there's no
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

define(`WANT_RBX', eval(8*0)($1))
define(`WANT_RBP', eval(8*1)($1))
define(`WANT_R12', eval(8*2)($1))
define(`WANT_R13', eval(8*3)($1))
define(`WANT_R14', eval(8*4)($1))
define(`WANT_R15', eval(8*5)($1))

define(`JUNK_RAX', eval(8*6)($1))
define(`JUNK_R10', eval(8*7)($1))
define(`JUNK_R11', eval(8*8)($1))

define(`SAVE_RBX', eval(8*9)($1))
define(`SAVE_RBP', eval(8*10)($1))
define(`SAVE_R12', eval(8*11)($1))
define(`SAVE_R13', eval(8*12)($1))
define(`SAVE_R14', eval(8*13)($1))
define(`SAVE_R15', eval(8*14)($1))

define(`RETADDR',  eval(8*15)($1))

define(`RBX',	   eval(8*16)($1))
define(`RBP',	   eval(8*17)($1))
define(`R12',	   eval(8*18)($1))
define(`R13',	   eval(8*19)($1))
define(`R14',	   eval(8*20)($1))
define(`R15',	   eval(8*21)($1))
define(`RFLAGS',   eval(8*22)($1))


define(G,
m4_assert_numargs(1)
`GSYM_PREFIX`'$1')

	TEXT
	ALIGN(32)
PROLOGUE(calling_conventions)
	mov	G(calling_conventions_values)@GOTPCREL(%rip), %rax
	pop	RETADDR(%rax)

	mov	%rbx, SAVE_RBX(%rax)
	mov	%rbp, SAVE_RBP(%rax)
	mov	%r12, SAVE_R12(%rax)
	mov	%r13, SAVE_R13(%rax)
	mov	%r14, SAVE_R14(%rax)
	mov	%r15, SAVE_R15(%rax)

	C Values we expect to see unchanged, as per amd64check.c
	mov	WANT_RBX(%rax), %rbx
	mov	WANT_RBP(%rax), %rbp
	mov	WANT_R12(%rax), %r12
	mov	WANT_R13(%rax), %r13
	mov	WANT_R14(%rax), %r14
	mov	WANT_R15(%rax), %r15

	C Try to provoke a problem by starting with junk in the caller-saves
	C registers, especially %rax which will be the return value.
C	mov	JUNK_RAX(%rax), %rax		C overwritten below anyway
	mov	JUNK_R10(%rax), %r10
	mov	JUNK_R11(%rax), %r11

	mov	G(calling_conventions_function)@GOTPCREL(%rip), %rax
	call	*(%rax)

	mov	G(calling_conventions_values)@GOTPCREL(%rip), %rcx

	mov	%rbx, RBX(%rcx)
	mov	%rbp, RBP(%rcx)
	mov	%r12, R12(%rcx)
	mov	%r13, R13(%rcx)
	mov	%r14, R14(%rcx)
	mov	%r15, R15(%rcx)

	pushf
	pop	%rbx
	mov	%rbx, RFLAGS(%rcx)

	mov	SAVE_RBX(%rcx), %rbx
	mov	SAVE_RBP(%rcx), %rbp
	mov	SAVE_R12(%rcx), %r12
	mov	SAVE_R13(%rcx), %r13
	mov	SAVE_R14(%rcx), %r14
	mov	SAVE_R15(%rcx), %r15

	C Overwrite parameter registers
C	mov	JUNK_R9(%rcx), %r9
C	mov	JUNK_R8(%rcx), %r8
C	mov	JUNK_RCX(%rcx), %rcx
C	mov	JUNK_RDX(%rcx), %rdx
C	mov	JUNK_RSI(%rcx), %rsi
C	mov	JUNK_RDI(%rcx), %rdi

	push	RETADDR(%rcx)

	mov	G(calling_conventions_fenv)@GOTPCREL(%rip), %rcx
	fstenv	(%rcx)
	finit

	ret

EPILOGUE()
