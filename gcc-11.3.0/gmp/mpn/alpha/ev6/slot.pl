#!/usr/bin/perl -w

# Copyright 2000, 2001, 2003-2005, 2011 Free Software Foundation, Inc.
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


# Usage: slot.pl [filename.o]...
#
# Run "objdump" to produce a disassembly of the given object file(s) and
# annotate the output with "U" or "L" slotting which Alpha EV6 will use.
#
# When an instruction is E (ie. either U or L), an "eU" or "eL" is shown, as
# a reminder that it wasn't a fixed requirement that gave the U or L, but
# the octaword slotting rules.
#
# If an instruction is not recognised, that octaword does not get any U/L
# shown, only lower-case "u", "l" or "e" for the instructions which are
# known.  Add any unknown instructions to %optable below.


use strict;

# The U or L which various instructions demand, or E if either.
#
my %optable =
  (
   'addq'   => 'E',
   'and'    => 'E',
   'andnot' => 'E',
   'beq'    => 'U',
   'bge'    => 'U',
   'bgt'    => 'U',
   'bic'    => 'E',
   'bis'    => 'E',
   'blt'    => 'U',
   'bne'    => 'U',
   'br'     => 'L',
   'clr'    => 'E',
   'cmpule' => 'E',
   'cmpult' => 'E',
   'cmpeq'  => 'E',
   'cmoveq' => 'E',
   'cmovne' => 'E',
   'ctpop'  => 'U',
   'ctlz'   => 'U',
   'cttz'   => 'U',
   'extbl'  => 'U',
   'extlh'  => 'U',
   'extll'  => 'U',
   'extqh'  => 'U',
   'extql'  => 'U',
   'extwh'  => 'U',
   'extwl'  => 'U',
   'jsr'    => 'L',
   'lda'    => 'E',
   'ldah'   => 'E',
   'ldbu'   => 'L',
   'ldl'    => 'L',
   'ldq'    => 'L',
   'ldt'    => 'L',
   'ret'    => 'L',
   'mov'    => 'E',
   'mull'   => 'U',
   'mulq'   => 'U',
   'negq'   => 'E',
   'nop'    => 'E',
   'not'    => 'E',
   's8addq' => 'E',
   's8subq' => 'E',
   # 'sextb'  => ?
   # 'sextl'  => ?
   'sll'    => 'U',
   'srl'    => 'U',
   'stq'    => 'L',
   'subq'   => 'E',
   'umulh'  => 'U',
   'unop'   => 'E',
   'xor'    => 'E',
  );

# Slottings used for a given pattern of U/L/E in an octaword.  This is as
# per the "Ebox Slotting" section of the EV6 hardware reference manual.
#
my %slottable =
  (
   'EEEE' => 'ULUL',
   'EEEL' => 'ULUL',
   'EEEU' => 'ULLU',
   'EELE' => 'ULLU',
   'EELL' => 'UULL',
   'EELU' => 'ULLU',
   'EEUE' => 'ULUL',
   'EEUL' => 'ULUL',
   'EEUU' => 'LLUU',
   'ELEE' => 'ULUL',
   'ELEL' => 'ULUL',
   'ELEU' => 'ULLU',
   'ELLE' => 'ULLU',
   'ELLL' => 'ULLL',
   'ELLU' => 'ULLU',
   'ELUE' => 'ULUL',
   'ELUL' => 'ULUL',

   'LLLL' => 'LLLL',
   'LLLU' => 'LLLU',
   'LLUE' => 'LLUU',
   'LLUL' => 'LLUL',
   'LLUU' => 'LLUU',
   'LUEE' => 'LULU',
   'LUEL' => 'LUUL',
   'LUEU' => 'LULU',
   'LULE' => 'LULU',
   'LULL' => 'LULL',
   'LULU' => 'LULU',
   'LUUE' => 'LUUL',
   'LUUL' => 'LUUL',
   'LUUU' => 'LUUU',
   'UEEE' => 'ULUL',
   'UEEL' => 'ULUL',
   'UEEU' => 'ULLU',

   'ELUU' => 'LLUU',
   'EUEE' => 'LULU',
   'EUEL' => 'LUUL',
   'EUEU' => 'LULU',
   'EULE' => 'LULU',
   'EULL' => 'UULL',
   'EULU' => 'LULU',
   'EUUE' => 'LUUL',
   'EUUL' => 'LUUL',
   'EUUU' => 'LUUU',
   'LEEE' => 'LULU',
   'LEEL' => 'LUUL',
   'LEEU' => 'LULU',
   'LELE' => 'LULU',
   'LELL' => 'LULL',
   'LELU' => 'LULU',
   'LEUE' => 'LUUL',
   'LEUL' => 'LUUL',
   'LEUU' => 'LLUU',
   'LLEE' => 'LLUU',
   'LLEL' => 'LLUL',
   'LLEU' => 'LLUU',
   'LLLE' => 'LLLU',

   'UELE' => 'ULLU',
   'UELL' => 'UULL',
   'UELU' => 'ULLU',
   'UEUE' => 'ULUL',
   'UEUL' => 'ULUL',
   'UEUU' => 'ULUU',
   'ULEE' => 'ULUL',
   'ULEL' => 'ULUL',
   'ULEU' => 'ULLU',
   'ULLE' => 'ULLU',
   'ULLL' => 'ULLL',
   'ULLU' => 'ULLU',
   'ULUE' => 'ULUL',
   'ULUL' => 'ULUL',
   'ULUU' => 'ULUU',
   'UUEE' => 'UULL',
   'UUEL' => 'UULL',
   'UUEU' => 'UULU',
   'UULE' => 'UULL',
   'UULL' => 'UULL',
   'UULU' => 'UULU',
   'UUUE' => 'UUUL',
   'UUUL' => 'UUUL',
   'UUUU' => 'UUUU',
  );

# Check all combinations of U/L/E are present in %slottable.
sub coverage {
  foreach my $a ('U', 'L', 'E') {
    foreach my $b ('U', 'L', 'E') {
      foreach my $c ('U', 'L', 'E') {
        foreach my $d ('U', 'L', 'E') {
          my $x = $a . $b . $c . $d;
          if (! defined $slottable{$x}) {
            print "slottable missing: $x\n"
          }
        }
      }
    }
  }
}

# Certain consistency checks for %slottable.
sub check {
  foreach my $x (keys %slottable) {
    my $a = substr($x,0,1);
    my $b = substr($x,1,1);
    my $c = substr($x,2,1);
    my $d = substr($x,3,1);
    my $es = ($a eq 'E') + ($b eq 'E') + ($c eq 'E') + ($d eq 'E');
    my $ls = ($a eq 'L') + ($b eq 'L') + ($c eq 'L') + ($d eq 'L');
    my $us = ($a eq 'U') + ($b eq 'U') + ($c eq 'U') + ($d eq 'U');

    my $got = $slottable{$x};
    my $want = $x;

    if ($es == 0) {

    } elsif ($es == 1) {
      # when only one E, it's mapped to whichever of U or L is otherwise
      # used the least
      if ($ls > $us) {
        $want =~ s/E/U/;
      } else {
        $want =~ s/E/L/;
      }
    } elsif ($es == 2) {
      # when two E's and two U, then the E's map to L; vice versa for two E
      # and two L
      if ($ls == 2) {
        $want =~ s/E/U/g;
      } elsif ($us == 2) {
        $want =~ s/E/L/g;
      } else {
        next;
      }
    } elsif ($es == 3) {
      next;

    } else { # $es == 4
      next;
    }

    if ($want ne $got) {
      print "slottable $x want $want got $got\n";
    }
  }
}

sub disassemble {
  my ($file) = @_;

  open (IN, "objdump -Srfh $file |") || die "Cannot open pipe from objdump\n";

  my (%pre, %post, %type);
  while (<IN>) {
    my $line = $_ . "";

    if ($line =~ /(^[ \t]*[0-9a-f]*([0-9a-f]):[ \t]*[0-9a-f][0-9a-f] [0-9a-f][0-9a-f] [0-9a-f][0-9a-f] [0-9a-f][0-9a-f] )\t(([a-z0-9]+).*)/) {
      my ($this_pre, $addr, $this_post, $opcode) = ($1, $2, $3, $4);

      my $this_type = $optable{$opcode};
      if (! defined ($this_type)) { $this_type = ' '; }

      $pre{$addr} = $this_pre;
      $post{$addr} = $this_post;
      $type{$addr} = $this_type;

      if ($addr eq 'c') {
        my %slot = ('0'=>' ', '4'=>' ', '8'=>' ', 'c'=>' ');

        my $str = $type{'c'} . $type{'8'} . $type{'4'} . $type{'0'};
        $str = $slottable{$str};
        if (defined $str) {
          $slot{'c'} = substr($str,0,1);
          $slot{'8'} = substr($str,1,1);
          $slot{'4'} = substr($str,2,1);
          $slot{'0'} = substr($str,3,1);
        }

        foreach my $i ('0', '4', '8', 'c') {
          if ($slot{$i} eq $type{$i}) { $type{$i} = ' '; }
          print $pre{$i}, ' ', lc($type{$i}),$slot{$i}, '  ', $post{$i}, "\n";
        }

        %pre = ();
        %type = ();
        %post = ();
      }
    }
  }

  close IN || die "Error from objdump (or objdump not available)\n";
}

coverage();
check();

my @files;
if ($#ARGV >= 0) {
  @files = @ARGV;
} else {
  die
}

foreach (@files)  {
    disassemble($_);
}
