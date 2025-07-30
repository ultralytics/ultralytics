for (int c0 = 1; c0 <= 18; c0 += 1)
  for (int c1 = 1; c1 <= 9; c1 += 1) {
    if (c0 % 2 == 0)
      s0(c1, c0 / 2);
    if (c0 <= 9)
      s1(c1, c0);
  }
