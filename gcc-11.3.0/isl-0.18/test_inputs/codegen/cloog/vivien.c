{
  for (int c0 = -27 * n + 2; c0 <= 1; c0 += 1)
    S1(c0 - 1);
  for (int c0 = 2; c0 <= min(2 * n, n + 29); c0 += 1) {
    if (c0 >= 3) {
      if (2 * n >= c0 + 1) {
        S4(c0 - c0 / 2 - 1, c0 / 2 + 1);
        if (c0 + 2 >= 2 * n) {
          for (int c2 = 1; c2 < -n + c0; c2 += 1)
            S5(-n + c0, n, c2);
        } else if (c0 >= 5) {
          S4(c0 - c0 / 2 - 2, c0 / 2 + 2);
          for (int c2 = 1; c2 < c0 - c0 / 2 - 1; c2 += 1)
            S5(c0 - c0 / 2 - 1, c0 / 2 + 1, c2);
        }
      }
      for (int c1 = -c0 + c0 / 2 + 3; c1 <= min(-1, n - c0); c1 += 1) {
        S4(-c1, c0 + c1);
        S6(-c1 + 2, c0 + c1 - 2);
        for (int c2 = 1; c2 <= -c1; c2 += 1)
          S5(-c1 + 1, c0 + c1 - 1, c2);
      }
      if (2 * n >= c0 + 3 && c0 >= n + 2) {
        S6(-n + c0 + 1, n - 1);
        for (int c2 = 1; c2 < -n + c0; c2 += 1)
          S5(-n + c0, n, c2);
      }
      if (n >= 3 && c0 == n + 2) {
        S6(2, n);
        S1(n + 1);
      } else {
        if (c0 >= n + 3 && 2 * n >= c0 + 1)
          S6(-n + c0, n);
        if (c0 >= n + 3) {
          S1(c0 - 1);
        } else {
          if (n + 1 >= c0 && c0 <= 4) {
            S1(c0 - 1);
          } else if (c0 >= 5 && n + 1 >= c0) {
            S6(2, c0 - 2);
            S1(c0 - 1);
          }
          if (n + 1 >= c0)
            S6(1, c0 - 1);
        }
      }
      if (n == 2 && c0 == 4)
        S1(3);
    } else {
      S1(1);
    }
    if (c0 % 2 == 0)
      S3(c0 / 2);
    for (int c1 = max(1, -n + c0); c1 < (c0 + 1) / 2; c1 += 1)
      S2(c0 - c1, c1);
  }
  for (int c0 = max(2 * n + 1, -27 * n + 2); c0 <= n + 29; c0 += 1)
    S1(c0 - 1);
  for (int c0 = n + 30; c0 <= 2 * n; c0 += 1) {
    if (2 * n >= c0 + 1) {
      S4(c0 - c0 / 2 - 1, c0 / 2 + 1);
      if (c0 + 2 >= 2 * n) {
        for (int c2 = 1; c2 < -n + c0; c2 += 1)
          S5(-n + c0, n, c2);
      } else {
        S4(c0 - c0 / 2 - 2, c0 / 2 + 2);
        for (int c2 = 1; c2 < c0 - c0 / 2 - 1; c2 += 1)
          S5(c0 - c0 / 2 - 1, c0 / 2 + 1, c2);
      }
      for (int c1 = -c0 + c0 / 2 + 3; c1 <= n - c0; c1 += 1) {
        S4(-c1, c0 + c1);
        S6(-c1 + 2, c0 + c1 - 2);
        for (int c2 = 1; c2 <= -c1; c2 += 1)
          S5(-c1 + 1, c0 + c1 - 1, c2);
      }
      if (2 * n >= c0 + 3) {
        S6(-n + c0 + 1, n - 1);
        for (int c2 = 1; c2 < -n + c0; c2 += 1)
          S5(-n + c0, n, c2);
      }
      S6(-n + c0, n);
    }
    if (c0 % 2 == 0)
      S3(c0 / 2);
    for (int c1 = -n + c0; c1 < (c0 + 1) / 2; c1 += 1)
      S2(c0 - c1, c1);
  }
}
