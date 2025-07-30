for (int c0 = 2; c0 <= n; c0 += 1)
  for (int c1 = 1; c1 <= n; c1 += 1) {
    for (int c3 = 1; c3 < min(c0, c1); c3 += 1)
      s1(c3, c0, c1);
    if (c0 >= c1 + 1)
      s0(c1, c0);
  }
