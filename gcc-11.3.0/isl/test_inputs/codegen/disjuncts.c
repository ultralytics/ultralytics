for (int c0 = 0; c0 <= n; c0 += 1)
  for (int c1 = 0; c1 <= n; c1 += 1)
    if (c1 == n || c0 == n || c1 == 0 || c0 == 0) {
      for (int c3 = 0; c3 <= n; c3 += 1)
        for (int c4 = 0; c4 <= n; c4 += 1)
          a(c0, c1, c3, c4);
      for (int c3 = 0; c3 <= n; c3 += 1)
        for (int c4 = 0; c4 <= n; c4 += 1)
          b(c0, c1, c3, c4);
    }
