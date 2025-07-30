divert(-1)

dnl  m4 macros for amd64 assembler.

dnl  Copyright 1999-2005, 2008, 2009, 2011-2013 Free Software Foundation, Inc.

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


dnl  Usage: CPUVEC_FUNCS_LIST
dnl
dnl  A list of the functions from gmp-impl.h x86 struct cpuvec_t, in the
dnl  order they appear in that structure.

define(CPUVEC_FUNCS_LIST,
``add_n',
`addlsh1_n',
`addlsh2_n',
`addmul_1',
`addmul_2',
`bdiv_dbm1c',
`cnd_add_n',
`cnd_sub_n',
`com',
`copyd',
`copyi',
`divexact_1',
`divrem_1',
`gcd_1',
`lshift',
`lshiftc',
`mod_1',
`mod_1_1p',
`mod_1_1p_cps',
`mod_1s_2p',
`mod_1s_2p_cps',
`mod_1s_4p',
`mod_1s_4p_cps',
`mod_34lsub1',
`modexact_1c_odd',
`mul_1',
`mul_basecase',
`mullo_basecase',
`preinv_divrem_1',
`preinv_mod_1',
`redc_1',
`redc_2',
`rshift',
`sqr_basecase',
`sub_n',
`sublsh1_n',
`submul_1'')


dnl  Called: PROLOGUE_cpu(GSYM_PREFIX`'foo)
dnl
dnl  In the amd64 code we use explicit TEXT and ALIGN() calls in the code,
dnl  since different alignments are wanted in various circumstances.  So for
dnl  instance,
dnl
dnl                  TEXT
dnl                  ALIGN(16)
dnl          PROLOGUE(mpn_add_n)
dnl                  ...
dnl          EPILOGUE()

define(`PROLOGUE_cpu',
m4_assert_numargs(1)
`	GLOBL	$1
	TYPE($1,`function')
$1:
')


dnl  Usage: ASSERT([cond][,instructions])
dnl
dnl  If WANT_ASSERT is 1, output the given instructions and expect the given
dnl  flags condition to then be satisfied.  For example,
dnl
dnl         ASSERT(ne, `cmpq %rax, %rbx')
dnl
dnl  The instructions can be omitted to just assert a flags condition with
dnl  no extra calculation.  For example,
dnl
dnl         ASSERT(nc)
dnl
dnl  When `instructions' is not empty, a pushfq/popfq is added for
dnl  convenience to preserve the flags, but the instructions themselves must
dnl  preserve any registers that matter.
dnl
dnl  The condition can be omitted to just output the given instructions when
dnl  assertion checking is wanted.  In this case the pushf/popf is omitted.
dnl  For example,
dnl
dnl         ASSERT(, `movq %rax, VAR_KEEPVAL')

define(ASSERT,
m4_assert_numargs_range(1,2)
m4_assert_defined(`WANT_ASSERT')
`ifelse(WANT_ASSERT,1,
`ifelse(`$1',,
`	$2',
`ifelse(`$2',,,
`	pushfq')
	$2
	`j$1'	L(ASSERT_ok`'ASSERT_counter)
	ud2	C assertion failed
L(ASSERT_ok`'ASSERT_counter):
ifelse(`$2',,,`	popfq')
define(`ASSERT_counter',incr(ASSERT_counter))')')')

define(ASSERT_counter,1)

define(`LEA',`dnl
ifdef(`PIC',
	`mov	$1@GOTPCREL(%rip), $2'
,
	`movabs	`$'$1, $2')
')


define(`DEF_OBJECT',
m4_assert_numargs_range(1,2)
`	RODATA
	ALIGN(ifelse($#,1,2,$2))
$1:
')

define(`END_OBJECT',
m4_assert_numargs(1)
`	SIZE(`$1',.-`$1')')


define(`R32',
	`ifelse($1,`%rax',`%eax',
		$1,`%rbx',`%ebx',
		$1,`%rcx',`%ecx',
		$1,`%rdx',`%edx',
		$1,`%rsi',`%esi',
		$1,`%rdi',`%edi',
		$1,`%rbp',`%ebp',
		$1,`%r8',`%r8d',
		$1,`%r9',`%r9d',
		$1,`%r10',`%r10d',
		$1,`%r11',`%r11d',
		$1,`%r12',`%r12d',
		$1,`%r13',`%r13d',
		$1,`%r14',`%r14d',
		$1,`%r15',`%r15d')')
define(`R8',
	`ifelse($1,`%rax',`%al',
		$1,`%rbx',`%bl',
		$1,`%rcx',`%cl',
		$1,`%rdx',`%dl',
		$1,`%rsi',`%sil',
		$1,`%rdi',`%dil',
		$1,`%rbp',`%bpl',
		$1,`%r8',`%r8b',
		$1,`%r9',`%r9b',
		$1,`%r10',`%r10b',
		$1,`%r11',`%r11b',
		$1,`%r12',`%r12b',
		$1,`%r13',`%r13b',
		$1,`%r14',`%r14b',
		$1,`%r15',`%r15b')')


dnl  Usage: CALL(funcname)
dnl

define(`CALL',`dnl
ifdef(`PIC',
	`call	GSYM_PREFIX`'$1@PLT'
,
	`call	GSYM_PREFIX`'$1'
)')


define(`JUMPTABSECT', `.section	.data.rel.ro.local,"aw",@progbits')


dnl  Usage: JMPENT(targlabel,tablabel)

define(`JMPENT',`dnl
ifdef(`PIC',
	`.long	$1-$2'dnl
,
	`.quad	$1'dnl
)')


dnl  These macros are defined just for DOS64, where they provide calling
dnl  sequence glue code.

define(`FUNC_ENTRY',`')
define(`FUNC_EXIT',`')


dnl  Target ABI macros.

define(`IFDOS',   `')
define(`IFSTD',   `$1')
define(`IFELF',   `$1')


dnl  Usage: PROTECT(symbol)
dnl
dnl  Used for private GMP symbols that should never be overridden by users.
dnl  This can save reloc entries and improve shlib sharing as well as
dnl  application startup times

define(`PROTECT',  `.hidden $1')


dnl  Usage: x86_lookup(target, key,value, key,value, ...)
dnl
dnl  Look for `target' among the `key' parameters.
dnl
dnl  x86_lookup expands to the corresponding `value', or generates an error
dnl  if `target' isn't found.

define(x86_lookup,
m4_assert_numargs_range(1,999)
`ifelse(eval($#<3),1,
`m4_error(`unrecognised part of x86 instruction: $1
')',
`ifelse(`$1',`$2', `$3',
`x86_lookup(`$1',shift(shift(shift($@))))')')')


dnl  Usage: x86_opcode_regxmm(reg)
dnl
dnl  Validate the given xmm register, and return its number, 0 to 7.

define(x86_opcode_regxmm,
m4_assert_numargs(1)
`x86_lookup(`$1',x86_opcode_regxmm_list)')

define(x86_opcode_regxmm_list,
``%xmm0',0,
`%xmm1',1,
`%xmm2',2,
`%xmm3',3,
`%xmm4',4,
`%xmm5',5,
`%xmm6',6,
`%xmm7',7,
`%xmm8',8,
`%xmm9',9,
`%xmm10',10,
`%xmm11',11,
`%xmm12',12,
`%xmm13',13,
`%xmm14',14,
`%xmm15',15')

dnl  Usage: palignr($imm,%srcreg,%dstreg)
dnl
dnl  Emit a palignr instruction, using a .byte sequence, since obsolete but
dnl  still distributed versions of gas don't know SSSE3 instructions.

define(`palignr',
m4_assert_numargs(3)
`.byte	0x66,dnl
ifelse(eval(x86_opcode_regxmm($3) >= 8 || x86_opcode_regxmm($2) >= 8),1,
       `eval(0x40+x86_opcode_regxmm($3)/8*4+x86_opcode_regxmm($2)/8),')dnl
0x0f,0x3a,0x0f,dnl
eval(0xc0+x86_opcode_regxmm($3)%8*8+x86_opcode_regxmm($2)%8),dnl
substr($1,1)')


dnl  Usage
dnl
dnl    regnum(op)   raw operand index (so slightly misnamed)
dnl    regnumh(op)  high bit of register operand nimber
dnl    ix(op)       0 for reg operand, 1 for plain pointer operand.
dnl

define(`regnum',`x86_lookup(`$1',oplist)')
define(`regnumh',`eval(regnum($1)/8 & 1)')
define(`ix',`eval(regnum($1)/16)')
define(`oplist',
``%rax',   0, `%rcx',   1, `%rdx',   2,  `%rbx',   3,
 `%rsp',   4, `%rbp',   5, `%rsi',   6,  `%rdi',   7,
 `%r8',    8, `%r9',    9, `%r10',  10,  `%r11',  11,
 `%r12',  12, `%r13',  13, `%r14',  14,  `%r15',  15,
 `(%rax)',16, `(%rcx)',17, `(%rdx)',18,  `(%rbx)',19,
 `(%rsp)',20, `(%rbp)',21, `(%rsi)',22,  `(%rdi)',23,
 `(%r8)', 24, `(%r9)', 25, `(%r10)',26,  `(%r11)',27,
 `(%r12)',28, `(%r13)',29, `(%r14)',30,  `(%r15)',31')


dnl  Usage
dnl
dnl     mulx(reg1,reg2,reg3)
dnl
dnl  or
dnl
dnl     mulx((reg1),reg2,reg3)
dnl
dnl  where reg1 is any register but rsp,rbp,r12,r13, or
dnl
dnl     mulx(off,(reg1),reg2,reg3)
dnl
dnl  where reg1 is any register but rsp,r12.
dnl
dnl  The exceptions are due to special coding needed for some registers; rsp
dnl  and r12 need an extra byte 0x24 at the end while rbp and r13 lack the
dnl  offset-less form.
dnl
dnl  Other addressing forms are not handled.  Invalid forms are not properly
dnl  detected.  Offsets that don't fit one byte are not handled correctly.

define(`mulx',`dnl
.byte	0xc4`'dnl
ifelse(`$#',3,`dnl
,eval(0xe2^32*regnumh($1)^128*regnumh($3))`'dnl
,eval(0xfb-8*regnum($2))`'dnl
,0xf6`'dnl
,eval(0xc0+(7 & regnum($1))+8*(7 & regnum($3))-0xc0*ix($1))`'dnl
',`$#',4,`dnl
,eval(0xe2^32*regnumh($2)^128*regnumh($4))`'dnl
,eval(0xfb-8*regnum($3))`'dnl
,0xf6`'dnl
,eval(0x40+(7 & regnum($2))+8*(7 & regnum($4)))`'dnl
,eval(($1 + 256) % 256)`'dnl
')')

dnl  Usage
dnl
dnl     adcx(reg1,reg2)
dnl     adox(reg1,reg2)
dnl
dnl  or
dnl
dnl     adcx((reg1),reg2)
dnl     adox((reg1),reg2)
dnl
dnl  where reg1 is any register but rsp,rbp,r12,r13, or
dnl
dnl     adcx(off,(reg1),reg2)
dnl     adox(off,(reg1),reg2)
dnl
dnl  where reg1 is any register but rsp,r12.
dnl
dnl  The exceptions are due to special coding needed for some registers; rsp
dnl  and r12 need an extra byte 0x24 at the end while rbp and r13 lack the
dnl  offset-less form.
dnl
dnl  Other addressing forms are not handled.  Invalid forms are not properly
dnl  detected.  Offsets that don't fit one byte are not handled correctly.

define(`adx_helper',`dnl
,eval(0x48+regnumh($1)+4*regnumh($2))`'dnl
,0x0f`'dnl
,0x38`'dnl
,0xf6`'dnl
')

define(`adx',`dnl
ifelse(`$#',2,`dnl
adx_helper($1,$2)dnl
,eval(0xc0+(7 & regnum($1))+8*(7 & regnum($2))-0xc0*ix($1))`'dnl
',`$#',3,`dnl
adx_helper($2,$3)dnl
,eval(0x40+(7 & regnum($2))+8*(7 & regnum($3)))`'dnl
,eval(($1 + 256) % 256)`'dnl
')')

define(`adcx',`dnl
.byte	0x66`'dnl
adx($@)')

define(`adox',`dnl
.byte	0xf3`'dnl
adx($@)')

divert`'dnl
