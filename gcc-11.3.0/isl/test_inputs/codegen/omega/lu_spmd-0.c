if (ub >= lb)
  for (int c0 = 1; c0 <= ub; c0 += 1)
    for (int c1 = c0; c1 <= n; c1 += 1) {
      if (c0 >= lb && c1 >= c0 + 1) {
        s0(c0, c1);
        if (n >= ub + 1)
          s2(c0, c1);
      } else if (lb >= c0 + 1) {
        s3(c0, c1, lb, c0, c1);
      }
      for (int c3 = max(lb, c0); c3 <= ub; c3 += 1)
        s1(c0, c1, c3);
    }
