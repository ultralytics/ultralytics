for (int c0 = 1; c0 <= n; c0 += 1) {
  for (int c1 = 2; c1 <= n; c1 += 1)
    for (int c2 = 1; c2 < min(c0, c1); c2 += 1)
      S2(c0, c1, c2, c1, c0);
  for (int c3 = c0 + 1; c3 <= n; c3 += 1)
    S1(c0, n, c0, c3);
}
