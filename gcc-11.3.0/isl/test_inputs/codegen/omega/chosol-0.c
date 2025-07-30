{
  for (int c1 = 2; c1 <= n; c1 += 1)
    s0(c1);
  for (int c1 = 1; c1 < n; c1 += 1) {
    for (int c3 = c1 + 1; c3 <= n; c3 += 1)
      s1(c3, c1);
    s2(c1 + 1);
  }
}
