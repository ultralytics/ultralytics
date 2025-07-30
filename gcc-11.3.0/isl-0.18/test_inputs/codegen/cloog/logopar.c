{
  for (int c1 = 0; c1 <= m; c1 += 1)
    S1(1, c1);
  for (int c0 = 2; c0 <= n; c0 += 1) {
    for (int c1 = 0; c1 < c0 - 1; c1 += 1)
      S2(c0, c1);
    for (int c1 = c0 - 1; c1 <= n; c1 += 1) {
      S1(c0, c1);
      S2(c0, c1);
    }
    for (int c1 = n + 1; c1 <= m; c1 += 1)
      S1(c0, c1);
  }
  for (int c0 = n + 1; c0 <= m + 1; c0 += 1)
    for (int c1 = c0 - 1; c1 <= m; c1 += 1)
      S1(c0, c1);
}
