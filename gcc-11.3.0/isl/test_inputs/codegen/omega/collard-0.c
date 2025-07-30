{
  for (int c4 = 1; c4 <= n; c4 += 1)
    s2(c4);
  for (int c1 = 1; c1 < n; c1 += 1) {
    for (int c4 = 0; c4 < n - c1; c4 += 1)
      s0(c1, n - c4);
    for (int c3 = 0; c3 < n - c1; c3 += 1)
      for (int c4 = c1 + 1; c4 <= n; c4 += 1)
        s1(c1, n - c3, c4);
  }
  for (int c1 = 1; c1 <= n; c1 += 1) {
    s4(c1);
    for (int c3 = c1 + 1; c3 <= n; c3 += 1)
      s3(c3, c1);
  }
}
