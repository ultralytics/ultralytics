if (n >= 2)
  for (int c0 = 1; c0 <= n; c0 += 1) {
    for (int c1 = 2; c1 <= c0; c1 += 1)
      for (int c3 = 1; c3 < c1; c3 += 1)
        s1(c3, c1, c0);
    for (int c1 = c0 + 1; c1 <= n; c1 += 1) {
      for (int c3 = 1; c3 < c0; c3 += 1)
        s1(c3, c1, c0);
      s0(c0, c1);
    }
  }
