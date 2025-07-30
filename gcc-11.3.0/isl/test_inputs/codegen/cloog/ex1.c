{
  for (int c0 = 0; c0 <= 14; c0 += 1)
    for (int c1 = 0; c1 < n - 14; c1 += 1)
      S1(c0, c1);
  for (int c0 = 15; c0 <= n; c0 += 1) {
    for (int c1 = 0; c1 <= 9; c1 += 1)
      S1(c0, c1);
    for (int c1 = 10; c1 < n - 14; c1 += 1) {
      S1(c0, c1);
      S2(c0, c1);
    }
    for (int c1 = n - 14; c1 <= n; c1 += 1)
      S2(c0, c1);
  }
}
