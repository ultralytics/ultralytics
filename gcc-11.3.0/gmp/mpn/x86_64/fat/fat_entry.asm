dnl  x86 fat binary entrypoints.

dnl  Contributed to the GNU project by Kevin Ryde (original x86_32 code) and
dnl  Torbjorn Granlund (port to x86_64)

dnl  Copyright 2003, 2009, 2011-2014 Free Software Foundation, Inc.

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


dnl  Forcibly disable profiling.
dnl
dnl  The entrypoints and inits are small enough not to worry about, the real
dnl  routines arrived at will have any profiling.  Also, the way the code
dnl  here ends with a jump means we won't work properly with the
dnl  "instrument" profiling scheme anyway.

define(`WANT_PROFILING',no)


dnl  We define PRETEND_PIC as a helper symbol, the use it for suppressing
dnl  normal, fast call code, since that triggers problems on Darwin and
dnl  OpenBSD.

ifdef(`DARWIN',
`define(`PRETEND_PIC')')
ifdef(`OPENBSD',
`define(`PRETEND_PIC')')
ifdef(`PIC',
`define(`PRETEND_PIC')')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

	TEXT

dnl  Usage: FAT_ENTRY(name, offset)
dnl
dnl  Emit a fat binary entrypoint function of the given name.  This is the
dnl  normal entry for applications, eg. __gmpn_add_n.
dnl
dnl  The code simply jumps through the function pointer in __gmpn_cpuvec at
dnl  the given "offset" (in bytes).
dnl
dnl  For non-PIC, the jumps are 5 bytes each, aligning them to 8 should be
dnl  fine for all x86s.
dnl
dnl  For ELF/DARWIN PIC, the jumps are 20 bytes each, and are best aligned to
dnl  16 to ensure at least the first two instructions don't cross a cache line
dnl  boundary.
dnl
dnl  For DOS64, the jumps are 6 bytes.  The same form works also for GNU/Linux
dnl  (at least with certain assembler/linkers) but FreeBSD 8.2 crashes.  Not
dnl  tested on Darwin, Slowaris, NetBSD, etc.
dnl
dnl  Note the extra `' ahead of PROLOGUE obscures it from the HAVE_NATIVE
dnl  grepping in configure, stopping that code trying to eval something with
dnl  $1 in it.

define(FAT_ENTRY,
m4_assert_numargs(2)
`ifdef(`HOST_DOS64',
`	ALIGN(8)
`'PROLOGUE($1)
	jmp	*$2+GSYM_PREFIX`'__gmpn_cpuvec(%rip)
EPILOGUE()
',
`	ALIGN(ifdef(`PIC',16,8))
`'PROLOGUE($1)
ifdef(`PRETEND_PIC',
`	LEA(	GSYM_PREFIX`'__gmpn_cpuvec, %rax)
	jmp	*$2(%rax)
',`dnl non-PIC
	jmp	*GSYM_PREFIX`'__gmpn_cpuvec+$2
')
EPILOGUE()
')')


dnl  FAT_ENTRY for each CPUVEC_FUNCS_LIST
dnl

define(`CPUVEC_offset',0)
foreach(i,
`FAT_ENTRY(MPN(i),CPUVEC_offset)
define(`CPUVEC_offset',eval(CPUVEC_offset + 8))',
CPUVEC_FUNCS_LIST)


dnl  Usage: FAT_INIT(name, offset)
dnl
dnl  Emit a fat binary initializer function of the given name.  These
dnl  functions are the initial values for the pointers in __gmpn_cpuvec.
dnl
dnl  The code simply calls __gmpn_cpuvec_init, and then jumps back through
dnl  the __gmpn_cpuvec pointer, at the given "offset" (in bytes).
dnl  __gmpn_cpuvec_init will have stored the address of the selected
dnl  implementation there.
dnl
dnl  Only one of these routines will be executed, and only once, since after
dnl  that all the __gmpn_cpuvec pointers go to real routines.  So there's no
dnl  need for anything special here, just something small and simple.  To
dnl  keep code size down, "fat_init" is a shared bit of code, arrived at
dnl  with the offset in %al.  %al is used since the movb instruction is 2
dnl  bytes where %eax would be 4.
dnl
dnl  Note having `PROLOGUE in FAT_INIT obscures that PROLOGUE from the
dnl  HAVE_NATIVE grepping in configure, preventing that code trying to eval
dnl  something with $1 in it.
dnl
dnl  We need to preserve parameter registers over the __gmpn_cpuvec_init call

define(FAT_INIT,
m4_assert_numargs(2)
`PROLOGUE($1)
	mov	$`'$2, %al
	jmp	L(fat_init)
EPILOGUE()
')

dnl  FAT_INIT for each CPUVEC_FUNCS_LIST
dnl

define(`CPUVEC_offset',0)
foreach(i,
`FAT_INIT(MPN(i`'_init),CPUVEC_offset)
define(`CPUVEC_offset',eval(CPUVEC_offset + 1))',
CPUVEC_FUNCS_LIST)

L(fat_init):
	C al	__gmpn_cpuvec byte offset

	movzbl	%al, %eax
IFSTD(`	push	%rdi	')
IFSTD(`	push	%rsi	')
	push	%rdx
	push	%rcx
	push	%r8
	push	%r9
	push	%rax
	CALL(	__gmpn_cpuvec_init)
	pop	%rax
	pop	%r9
	pop	%r8
	pop	%rcx
	pop	%rdx
IFSTD(`	pop	%rsi	')
IFSTD(`	pop	%rdi	')
ifdef(`PRETEND_PIC',`
	LEA(	GSYM_PREFIX`'__gmpn_cpuvec, %r10)
	jmp	*(%r10,%rax,8)
',`dnl non-PIC
	jmp	*GSYM_PREFIX`'__gmpn_cpuvec(,%rax,8)
')


C long __gmpn_cpuid (char dst[12], int id);
C
C This is called only 3 times, so just something simple and compact is fine.
C
C The rcx/ecx zeroing here is needed for the BMI2 check.

define(`rp',  `%rdi')
define(`idx', `%rsi')

PROLOGUE(__gmpn_cpuid)
	FUNC_ENTRY(2)
	mov	%rbx, %r8
	mov	R32(idx), R32(%rax)
	xor	%ecx, %ecx
	cpuid
	mov	%ebx, (rp)
	mov	%edx, 4(rp)
	mov	%ecx, 8(rp)
	mov	%r8, %rbx
	FUNC_EXIT()
	ret
EPILOGUE()
