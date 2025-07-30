{
  for (int c0 = 1; c0 < n; c0 += 1) {
    for (int c1 = 1; c1 < c0; c1 += 1)
      for (int c2 = c1 + 1; c2 <= n; c2 += 1)
        S2(c1, c2, c0);
    for (int c2 = c0 + 1; c2 <= n; c2 += 1)
      S1(c0, c2);
  }
  for (int c1 = 1; c1 < n; c1 += 1)
    for (int c2 = c1 + 1; c2 <= n; c2 += 1)
      S2(c1, c2, n);
}
