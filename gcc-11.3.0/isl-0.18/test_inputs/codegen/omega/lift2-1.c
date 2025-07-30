for (int c0 = 1; c0 <= 100; c0 += 1)
  for (int c1 = 1; c1 <= 100; c1 += 1)
    for (int c2 = 1; c2 <= 100; c2 += 1)
      for (int c3 = 1; c3 <= 100; c3 += 1) {
        if (c0 >= 61) {
          for (int c4 = 1; c4 <= 100; c4 += 1)
            s1(c0, c1, c2, c3, c4);
        } else if (c0 <= 4) {
          for (int c4 = 1; c4 <= 100; c4 += 1)
            s1(c0, c1, c2, c3, c4);
        } else {
          for (int c4 = 1; c4 <= 100; c4 += 1) {
            s1(c0, c1, c2, c3, c4);
            s0(c0, c1, c2, c3, c4);
          }
        }
      }
