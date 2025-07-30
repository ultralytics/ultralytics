#!/usr/bin/perl -w
#
# Copyright 2001, 2002 Free Software Foundation, Inc.
#
#  This file is part of the GNU MP Library.
#
#  The GNU MP Library is free software; you can redistribute it and/or modify
#  it under the terms of either:
#
#    * the GNU Lesser General Public License as published by the Free
#      Software Foundation; either version 3 of the License, or (at your
#      option) any later version.
#
#  or
#
#    * the GNU General Public License as published by the Free Software
#      Foundation; either version 2 of the License, or (at your option) any
#      later version.
#
#  or both in parallel, as here.
#
#  The GNU MP Library is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
#  for more details.
#
#  You should have received copies of the GNU General Public License and the
#  GNU Lesser General Public License along with the GNU MP Library.  If not,
#  see https://www.gnu.org/licenses/.


# Usage: cd $(builddir)/mpn
#        $(srcdir)/x86/t-zdisp2.pl
#
# Grep for any "0(reg...)" addressing modes coming out of the x86 .asm
# files.  Additive expressions like "12+4-16" are recognised too.
#
# Old gas doesn't preserve the "0" displacement, so if it's wanted then
# Zdisp ought to be used to give explicit .byte sequences.  See
# mpn/x86/README.
#
# No output means everything is ok.  All the asm files are put through m4 in
# PIC and non-PIC modes, and in each multi-function form, all of which can
# take a while to run.
#
# This program is only meant for use during development.

use strict;
use File::Find;
use File::Basename;
use Getopt::Std;

my %opt;
getopts('t', \%opt);


my $srcdir;
open IN, '<Makefile' or die;
while (<IN>) {
  if (/^srcdir[ \t]*=[ \t]*(.*)/) {
    $srcdir = $1;
    last;
  }
}
close IN or die;
defined $srcdir or die "Cannot find \$srcdir in Makefile\n";

my $filecount = 0;

my $tempfile = 't-zdisp2.tmp';
open KARA, ">$tempfile" or die;
close KARA or die;

find({ wanted => \&process, preprocess => \&process_mparam, no_chdir => 1 },
     "$srcdir/x86");

sub process {
  if (/gmp-mparam.h$/) {
    process_mparam($_);
  } elsif (/\.asm$/) {
    process_asm($_);
  }
}

# Ensure we're using the right SQR_TOOM2_THRESHOLD for the part of the
# tree being processed.
sub process_mparam {
  my $file = "$File::Find::dir/gmp-mparam.h";
  if (-f $file) {
    print "$file\n" if $opt{'t'};
    open MPARAM, "<$file" or die;
    while (<MPARAM>) {
      if (/^#define SQR_TOOM2_THRESHOLD[ \t]*([0-9][0-9]*)/) {
        open KARA, ">$tempfile" or die;
        print KARA "define(\`SQR_TOOM2_THRESHOLD',$1)\n\n";
        print "define(\`SQR_TOOM2_THRESHOLD',$1)\n" if $opt{'t'};
        close KARA or die;
        last;
      }
    }
    close MPARAM or die;
  }
  return @_;
}

sub process_asm {
  my ($file) = @_;
  my $base = basename ($file, '.asm');

  my @funs;
  if    ($base eq 'aors_n')    { @funs = qw(add_n sub_n); }
  elsif ($base eq 'aorsmul_1') { @funs = qw(addmul_1 submul_1); }
  elsif ($base eq 'popham')    { @funs = qw(popcount hamdist); }
  elsif ($base eq 'logops_n')  { @funs = qw(and_n andn_n nand_n ior_n iorn_n nior_n xor_n xnor_n); }
  elsif ($base eq 'lorrshift') { @funs = qw(lshift rshift); }
  else                         { @funs = ($base); }

  foreach my $fun (@funs) {
    foreach my $pic ('', ' -DPIC') {
      my $header = "$file: 0: $pic\n";
      $filecount++;

      my $m4 = "m4 -DHAVE_HOST_CPU_athlon -DOPERATION_$fun $pic ../config.m4 $tempfile $file";
      print "$m4\n" if $opt{'t'};

      open IN, "$m4 |" or die;
      while (<IN>) {
        next unless /([0-9+-][0-9 \t+-]*)\(%/;
        my $pat=$1;
        $pat = eval($pat);
        next if ($pat != 0);
        print "$header$_";
        $header='';
      }
      close IN or die;
    }
  }
}

unlink($tempfile);
print "total $filecount processed\n";
exit 0;


# Local variables:
# perl-indent-level: 2
# End:
