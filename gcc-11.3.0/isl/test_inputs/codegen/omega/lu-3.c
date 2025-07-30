for (int c0 = 1; c0 < n; c0 += 64)
  for (int c1 = c0 - 1; c1 <= n; c1 += 64) {
    for (int c2 = c0; c2 <= min(n, c0 + 63); c2 += 1) {
      for (int c3 = c0; c3 <= min(c1 + 62, c2 - 1); c3 += 1)
        for (int c4 = max(c1, c3 + 1); c4 <= min(n, c1 + 63); c4 += 1)
          s1(c3, c4, c2);
      for (int c4 = max(c1, c2 + 1); c4 <= min(n, c1 + 63); c4 += 1)
        s0(c2, c4);
    }
    for (int c2 = c0 + 64; c2 <= n; c2 += 1)
      for (int c3 = c0; c3 <= min(c0 + 63, c1 + 62); c3 += 1)
        for (int c4 = max(c1, c3 + 1); c4 <= min(n, c1 + 63); c4 += 1)
          s1(c3, c4, c2);
  }
