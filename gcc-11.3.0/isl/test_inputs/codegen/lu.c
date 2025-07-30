for (int c0 = 0; c0 < n - 1; c0 += 32)
  for (int c1 = c0; c1 < n; c1 += 32)
    for (int c2 = c0; c2 < n; c2 += 32) {
      if (c1 >= c0 + 32) {
        for (int c3 = c0; c3 <= min(c0 + 31, c2 + 30); c3 += 1)
          for (int c4 = c1; c4 <= min(n - 1, c1 + 31); c4 += 1)
            for (int c5 = max(c2, c3 + 1); c5 <= min(n - 1, c2 + 31); c5 += 1)
              S_6(c3, c4, c5);
      } else {
        for (int c3 = c0; c3 <= min(min(n - 2, c0 + 31), c2 + 30); c3 += 1) {
          for (int c5 = max(c2, c3 + 1); c5 <= min(n - 1, c2 + 31); c5 += 1)
            S_2(c3, c5);
          for (int c4 = c3 + 1; c4 <= min(n - 1, c0 + 31); c4 += 1)
            for (int c5 = max(c2, c3 + 1); c5 <= min(n - 1, c2 + 31); c5 += 1)
              S_6(c3, c4, c5);
        }
      }
    }
