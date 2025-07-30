for (int c0 = 0; c0 < n - 1; c0 += 1) {
  for (int c3 = 0; c3 < n - c0 - 1; c3 += 1)
    s0(c0 + 1, n - c3);
  for (int c3 = 0; c3 < n - c0 - 1; c3 += 1)
    for (int c6 = c0 + 2; c6 <= n; c6 += 1)
      s1(c0 + 1, n - c3, c6);
}
