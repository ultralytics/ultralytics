for (int c0 = 1; c0 <= 8; c0 += 1)
  for (int c1 = 0; c1 <= 7; c1 += 1) {
    if (c0 >= 2 && c0 <= 6 && c1 <= 4)
      s1(c0, c1);
    if (c1 + 1 >= c0)
      s0(c0, c1);
  }
