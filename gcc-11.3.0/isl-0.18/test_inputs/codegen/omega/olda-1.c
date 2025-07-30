for (int c0 = 1; c0 <= morb; c0 += 1)
  for (int c1 = 1; c1 <= np; c1 += 1) {
    for (int c2 = 1; c2 < c1; c2 += 1)
      s1(c1, c2, c0);
    s0(c1, c1, c0);
    s1(c1, c1, c0);
    for (int c2 = c1 + 1; c2 <= np; c2 += 1)
      s0(c2, c1, c0);
  }
